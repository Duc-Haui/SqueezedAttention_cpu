"""
Online Value-Aware Retrieval cho Squeezed Attention.

Thay thế bước "centroid lookup" gốc khi inference:
- Tính S_i như cũ
- Boost theo value variance đã precompute
- Chọn các cluster có S_tilde_i > threshold
- Compute exact attention chỉ trên các cluster được chọn
"""

from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F


def compute_base_scores(
    query: torch.Tensor,           # (H, D)
    key_centroids: torch.Tensor,   # (H, K, D)
    cluster_sizes: torch.Tensor,   # (H, K)
) -> torch.Tensor:
    """
    Tính S_i theo công thức (1) trong paper Squeezed Attention:
        S_i = exp(q · C_i^T) / sum_j ( N_j * exp(q · C_j^T) )

    Returns:
        S: (H, K)
    """
    if query.dim() == 3:
        query = query.squeeze(0)

    # Dot product q với centroids: (H, K)
    logits = torch.einsum("hd,hkd->hk", query, key_centroids)

    # Stable softmax
    logits_max = logits.max(dim=-1, keepdim=True).values
    logits_shifted = logits - logits_max
    exp_logits = torch.exp(logits_shifted)
    weighted = cluster_sizes * exp_logits
    denom = weighted.sum(dim=-1, keepdim=True)

    S = exp_logits / (denom + 1e-12)
    return S


def value_aware_retrieve(
    query: torch.Tensor,                # (H, D)
    key_centroids: torch.Tensor,        # (H, K, D)
    cluster_sizes: torch.Tensor,        # (H, K)
    normalized_variance: torch.Tensor,  # (H, K)
    threshold: float,
    gamma: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trả về cluster mask và adjusted scores.
    
    Returns:
        cluster_mask: (H, K) bool
        adjusted_scores: (H, K) - dùng cho debug
    """
    S = compute_base_scores(query, key_centroids, cluster_sizes)
    S_adjusted = S * (1.0 + gamma * normalized_variance)
    cluster_mask = S_adjusted > threshold
    return cluster_mask, S_adjusted


def keys_mask_from_clusters(
    cluster_mask: torch.Tensor,  # (H, K)
    labels: torch.Tensor,        # (H, N)
) -> torch.Tensor:
    """Convert cluster mask -> per-token mask. Returns (H, N) bool."""
    return torch.gather(cluster_mask, 1, labels)


def squeezed_attention_forward(
    query: torch.Tensor,                # (H, D)
    full_keys: torch.Tensor,            # (H, N, D)
    full_values: torch.Tensor,          # (H, N, D_v)
    key_centroids: torch.Tensor,        # (H, K, D)
    cluster_sizes: torch.Tensor,        # (H, K)
    normalized_variance: torch.Tensor,  # (H, K)
    labels: torch.Tensor,               # (H, N)
    threshold: float,
    gamma: float = 0.3,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Reference Value-Aware Squeezed Attention.
    Output: (H, D_v), info dict
    """
    H, N, D = full_keys.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    cluster_mask, _ = value_aware_retrieve(
        query, key_centroids, cluster_sizes, normalized_variance, threshold, gamma
    )
    key_mask = keys_mask_from_clusters(cluster_mask, labels)  # (H, N)

    attn_logits = torch.einsum("hd,hnd->hn", query, full_keys) * scale
    attn_logits = attn_logits.masked_fill(~key_mask, float("-inf"))

    # Edge case: nếu một head bị mask hết -> output 0
    valid = key_mask.any(dim=-1, keepdim=True)
    attn_weights = F.softmax(attn_logits, dim=-1)
    attn_weights = torch.where(valid, attn_weights, torch.zeros_like(attn_weights))

    output = torch.einsum("hn,hnd->hd", attn_weights, full_values)

    info = {
        "num_clusters_kept": cluster_mask.float().sum(dim=-1).mean().item(),
        "num_keys_kept": key_mask.float().sum(dim=-1).mean().item(),
        "total_keys": N,
        "kv_budget": key_mask.float().mean().item(),
    }
    return output, info


def key_only_attention_forward(
    query: torch.Tensor,           # (H, D)
    full_keys: torch.Tensor,       # (H, N, D)
    full_values: torch.Tensor,     # (H, N, D_v)
    key_centroids: torch.Tensor,   # (H, K, D)
    cluster_sizes: torch.Tensor,   # (H, K)
    labels: torch.Tensor,          # (H, N)
    threshold: float,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, dict]:
    """Squeezed Attention gốc (key-only). Tương đương với gamma=0."""
    H, K = cluster_sizes.shape
    zero_var = torch.zeros(H, K, device=full_keys.device, dtype=full_keys.dtype)
    return squeezed_attention_forward(
        query, full_keys, full_values,
        key_centroids, cluster_sizes, zero_var, labels,
        threshold=threshold, gamma=0.0, scale=scale,
    )


def baseline_full_attention(
    query: torch.Tensor,        # (H, D)
    full_keys: torch.Tensor,    # (H, N, D)
    full_values: torch.Tensor,  # (H, N, D_v)
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Full attention - ground truth."""
    H, N, D = full_keys.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    attn_logits = torch.einsum("hd,hnd->hn", query, full_keys) * scale
    attn_weights = F.softmax(attn_logits, dim=-1)
    output = torch.einsum("hn,hnd->hd", attn_weights, full_values)
    return output


# ---------------------------------------------------------------------------
# QUEST-style baseline (cluster theo physical proximity)
# ---------------------------------------------------------------------------
def quest_style_clustering(
    keys: torch.Tensor,    # (H, N, D)
    values: torch.Tensor,  # (H, N, D_v)
    chunk_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    QUEST-style: chia keys thành chunks liên tiếp (theo vị trí), 
    mỗi chunk có 1 representative.
    
    Đại diện theo paper QUEST: dùng MIN, MAX của chunk để bound attention scores.
    Ở đây để công bằng với 1-centroid Squeezed, ta dùng MEAN.
    
    Returns:
        rep_keys: (H, num_chunks, D) - representative keys
        labels:   (H, N) - chunk assignment
        sizes:    (H, num_chunks) - chunk sizes
    """
    H, N, D = keys.shape
    num_chunks = (N + chunk_size - 1) // chunk_size

    # Tạo labels: token i -> chunk i // chunk_size
    labels = (
        torch.arange(N, device=keys.device) // chunk_size
    ).unsqueeze(0).expand(H, -1).clone()

    # Tính rep_keys = mean của chunk + sizes
    rep_keys = torch.zeros(H, num_chunks, D, device=keys.device, dtype=keys.dtype)
    sizes = torch.zeros(H, num_chunks, device=keys.device, dtype=keys.dtype)
    labels_expanded = labels.unsqueeze(-1).expand(-1, -1, D)
    rep_keys.scatter_add_(1, labels_expanded, keys)
    ones = torch.ones(H, N, device=keys.device, dtype=keys.dtype)
    sizes.scatter_add_(1, labels, ones)
    rep_keys = rep_keys / sizes.clamp(min=1).unsqueeze(-1)
    rep_keys = F.normalize(rep_keys, dim=-1)

    return rep_keys, labels, sizes


def quest_attention_forward(
    query: torch.Tensor,
    full_keys: torch.Tensor,
    full_values: torch.Tensor,
    rep_keys: torch.Tensor,
    cluster_sizes: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, dict]:
    """QUEST-style retrieval (giống Squeezed nhưng cluster theo position)."""
    return key_only_attention_forward(
        query, full_keys, full_values,
        rep_keys, cluster_sizes, labels,
        threshold=threshold, scale=scale,
    )
