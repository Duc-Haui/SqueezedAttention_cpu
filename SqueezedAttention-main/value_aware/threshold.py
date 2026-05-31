"""
Threshold calibration cho Value-Aware Squeezed Attention.

Tương thích với `run_global_threshold` của repo gốc (output dict gồm các quantile).
Khác biệt chính: tính importance scores có boost theo value variance.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .retrieval import compute_base_scores


def run_value_aware_global_threshold(
    keys_layers: List[torch.Tensor],         # list[L] of (1, H, N, D)
    queries_layers: List[torch.Tensor],      # list[L] of (1, H, N, D)
    key_centroids_dict: Dict[int, torch.Tensor],     # {l: (1, H, K, D)}
    labels_dict: Dict[int, torch.Tensor],            # {l: (1, H, N - obs)}
    normalized_variance_dict: Dict[int, torch.Tensor],  # {l: (1, H, K)}
    num_clusters: int,
    observation_window: int = 100,
    gamma: float = 0.3,
    print_log: bool = False,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Drop-in thay thế run_global_threshold của repo gốc.
    
    Tính scores có value-aware boost, sau đó tính các quantile threshold.
    
    Returns:
        tdict: {0.5: T1, 0.7: T2, 0.8: T3, 0.9: T4,
                'shared_prefix_length': ..., 'observation_window': ...,
                'gamma': gamma}
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_layers = len(queries_layers)
    K = num_clusters

    # Sẽ collect tất cả scaled scores để tính quantile cuối
    attn_score_centroid_list = []

    for layer_idx in range(num_layers):
        if print_log:
            print(f"  [VA-thresh] layer {layer_idx}")

        centroids = key_centroids_dict[layer_idx].squeeze(0).to(device)  # (H, K, D)
        labels = labels_dict[layer_idx].squeeze(0).to(device)            # (H, N-obs)
        nvar = normalized_variance_dict[layer_idx].squeeze(0).to(device) # (H, K)

        keys = keys_layers[layer_idx].squeeze(0).to(device)
        queries = queries_layers[layer_idx].squeeze(0).to(device)

        # Lấy queries của observation window cuối
        queries_obs = queries[:, -observation_window:, :].float()  # (H, OBS, D)

        # Tính q · C^T cho cả batch obs queries
        attn_scores_centroids = torch.matmul(
            queries_obs, centroids.transpose(1, 2).float()
        ) / math.sqrt(keys.shape[-1])  # (H, OBS, K)

        # Tính số keys per cluster
        num_keys_per_cluster = torch.zeros(
            (keys.shape[0], K), device=device, dtype=torch.float
        )
        for k in range(K):
            label_mask = (labels == k).float()
            num_keys_per_cluster[:, k] = label_mask.sum(dim=-1)

        # Mảng denominator: (H, OBS) = sum_k N_k * exp(...)
        attn_exp = torch.exp(attn_scores_centroids)
        num_keys_unsq = num_keys_per_cluster.unsqueeze(-2)  # (H, 1, K)
        denom = (num_keys_unsq * attn_exp).sum(dim=-1)  # (H, OBS)

        # S_i per (H, OBS, K)
        S = attn_exp / denom.unsqueeze(-1)  # (H, OBS, K)

        # Apply value-aware boost
        # nvar shape (H, K) -> broadcast over OBS
        S_adjusted = S * (1.0 + gamma * nvar.unsqueeze(-2))  # (H, OBS, K)

        # Bước cuối: cần "expand" S_adjusted về per-token để lấy quantile
        # giống code gốc: scores[h, n, q] = S_adjusted[h, q, label[h,n]]
        # Tức là với mỗi key n, score của nó cho query q = score của cluster chứa nó.
        # Để có cùng format với code gốc (lấy quantile trên scores per-token), ta tính:
        labels_expanded = labels.unsqueeze(-1).expand(-1, -1, observation_window)  # (H, N-obs, OBS)
        # Gather: shape (H, N-obs, OBS), với mỗi (h, n, q) lấy S_adj[h, q, labels[h,n]]
        # Cần broadcast: S_adj có shape (H, OBS, K), muốn lấy theo cluster idx -> permute
        S_adj_perm = S_adjusted.permute(0, 2, 1)  # (H, K, OBS)
        # gather theo dim=1 (K dim)
        scores_per_token = torch.gather(
            S_adj_perm, 1, labels_expanded
        )  # (H, N-obs, OBS)

        # Average over OBS queries -> (H, N-obs)
        avg_scores = scores_per_token.mean(dim=-1, dtype=torch.float32)
        attn_score_centroid_list.append(avg_scores)

        del keys, queries

    # Stack: (L, H, N-obs)
    full_scores = torch.stack(attn_score_centroid_list, dim=0)

    qlist = [0.5, 0.7, 0.8, 0.9, 0.95]
    full_scores_cpu = full_scores.cpu().numpy()
    quantile_result = np.quantile(full_scores_cpu, qlist)

    tdict = {}
    for i, q in enumerate(qlist):
        tdict[q] = float(quantile_result[i])

    tdict["shared_prefix_length"] = queries_layers[0].shape[-2]
    tdict["observation_window"] = observation_window
    tdict["gamma"] = gamma

    return tdict


# ---------------------------------------------------------------------------
# Calibrate threshold theo target sparsity (alternative API, dễ dùng hơn)
# ---------------------------------------------------------------------------
def calibrate_threshold(
    queries_calib: torch.Tensor,         # (Q, H, D)
    key_centroids: torch.Tensor,         # (H, K, D)
    cluster_sizes: torch.Tensor,         # (H, K)
    normalized_variance: torch.Tensor,   # (H, K)
    labels: torch.Tensor,                # (H, N)
    target_sparsity: float = 0.9,
    gamma: float = 0.3,
    num_threshold_search: int = 100,
) -> float:
    """
    Tìm threshold sao cho phần keys giữ ≈ (1 - target_sparsity).
    
    Args:
        queries_calib: (Q, H, D) calibration queries
        target_sparsity: 0.9 = giữ 10%
    """
    Q = queries_calib.shape[0]

    accumulated = torch.zeros_like(cluster_sizes)
    for q_idx in range(Q):
        S = compute_base_scores(queries_calib[q_idx], key_centroids, cluster_sizes)
        S_adj = S * (1.0 + gamma * normalized_variance)
        accumulated += S_adj
    avg_scores = accumulated / Q

    target_keep = 1.0 - target_sparsity
    candidates = torch.linspace(
        avg_scores.min().item(),
        avg_scores.max().item(),
        num_threshold_search,
        device=avg_scores.device,
    )

    best_T = candidates[0].item()
    best_diff = float("inf")
    for T in candidates:
        cluster_mask = avg_scores > T
        kept = (cluster_mask.float() * cluster_sizes).sum(dim=-1)
        total = cluster_sizes.sum(dim=-1).clamp(min=1)
        keep_ratio = (kept / total).mean().item()
        diff = abs(keep_ratio - target_keep)
        if diff < best_diff:
            best_diff = diff
            best_T = T.item()

    return best_T
