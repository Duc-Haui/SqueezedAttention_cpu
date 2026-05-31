"""
Value-Aware Clustering for Squeezed Attention.

Module này thay thế bước offline K-means của repo gốc, dùng joint K-V clustering 
và precompute value variance per cluster.

Drop-in compatible: nếu beta=0 và gamma=0 ở downstream, kết quả gần với repo gốc.
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core K-means
# ---------------------------------------------------------------------------
def _kmeans_cosine(
    x: torch.Tensor,
    num_clusters: int,
    num_iters: int = 10,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cosine K-means batched theo head.
    
    Args:
        x: (H, N, D) - đã normalize L2
        num_clusters: số cluster
        num_iters: số vòng lặp Lloyd
        seed: random seed
    
    Returns:
        centroids: (H, K, D) đã normalize
        labels:    (H, N) int64
    """
    H, N, D = x.shape
    device = x.device
    dtype = x.dtype

    if N <= num_clusters:
        # Quá ít điểm: trả về chính các điểm làm centroid
        centroids = F.pad(x, (0, 0, 0, num_clusters - N), value=0)
        labels = torch.arange(N, device=device).unsqueeze(0).expand(H, -1).clone()
        # Pad labels tới N, các vị trí pad không có ý nghĩa (không xảy ra trong thực tế)
        return centroids, labels

    # K-means++ style init: chọn ngẫu nhiên một điểm rồi chọn các điểm xa nhất
    g = torch.Generator(device=device).manual_seed(seed)
    init_idx = torch.randint(0, N, (H, num_clusters), generator=g, device=device)
    centroids = torch.gather(
        x, 1, init_idx.unsqueeze(-1).expand(-1, -1, D)
    ).clone()

    for it in range(num_iters):
        # Cosine sim = dot product khi cả hai đều normalized
        sims = torch.bmm(x, centroids.transpose(1, 2))  # (H, N, K)
        labels = sims.argmax(dim=-1)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(H, num_clusters, device=device, dtype=dtype)
        labels_expanded = labels.unsqueeze(-1).expand(-1, -1, D)
        new_centroids.scatter_add_(1, labels_expanded, x)
        ones = torch.ones(H, N, device=device, dtype=dtype)
        counts.scatter_add_(1, labels, ones)

        empty_mask = counts == 0
        counts_safe = counts.clamp(min=1).unsqueeze(-1)
        new_centroids = new_centroids / counts_safe
        new_centroids = torch.where(
            empty_mask.unsqueeze(-1), centroids, new_centroids
        )
        new_centroids = F.normalize(new_centroids, dim=-1)
        centroids = new_centroids

    sims = torch.bmm(x, centroids.transpose(1, 2))
    labels = sims.argmax(dim=-1)
    return centroids, labels


# ---------------------------------------------------------------------------
# Joint K-V K-means
# ---------------------------------------------------------------------------
def value_aware_kmeans(
    keys: torch.Tensor,
    values: torch.Tensor,
    num_clusters: int,
    alpha: float = 1.0,
    beta: float = 0.5,
    num_iters: int = 10,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Joint clustering trên concat(alpha*K_norm, beta*V_norm).
    
    Args:
        keys:   (H, N, D_k) raw keys
        values: (H, N, D_v) raw values
        num_clusters: số cluster
        alpha: trọng số phần K (>= 0). Mặc định 1.0
        beta:  trọng số phần V (>= 0). Mặc định 0.5
        num_iters: số iter K-means
        seed: random seed
    
    Returns:
        key_centroids:   (H, K, D_k) đã chuẩn hóa
        value_centroids: (H, K, D_v) raw mean
        labels:          (H, N) int64
        value_variance:  (H, K) trace variance trong từng cluster
    """
    H, N, D_k = keys.shape
    _, _, D_v = values.shape
    device = keys.device

    keys_n = F.normalize(keys, dim=-1)
    values_n = F.normalize(values, dim=-1)

    if beta > 0:
        joint = torch.cat([alpha * keys_n, beta * values_n], dim=-1)
        joint = F.normalize(joint, dim=-1)
    else:
        joint = keys_n  # baseline: K-only

    _, labels = _kmeans_cosine(joint, num_clusters, num_iters=num_iters, seed=seed)

    # Tính centroids (key + value) và variance
    key_centroids = torch.zeros(H, num_clusters, D_k, device=device, dtype=keys.dtype)
    value_centroids = torch.zeros(H, num_clusters, D_v, device=device, dtype=values.dtype)
    counts = torch.zeros(H, num_clusters, device=device, dtype=keys.dtype)

    labels_k = labels.unsqueeze(-1).expand(-1, -1, D_k)
    labels_v = labels.unsqueeze(-1).expand(-1, -1, D_v)

    key_centroids.scatter_add_(1, labels_k, keys)
    value_centroids.scatter_add_(1, labels_v, values)
    ones = torch.ones(H, N, device=device, dtype=keys.dtype)
    counts.scatter_add_(1, labels, ones)

    counts_safe = counts.clamp(min=1).unsqueeze(-1)
    key_centroids = key_centroids / counts_safe
    value_centroids = value_centroids / counts_safe

    # Chuẩn hóa key centroid để tương thích với pipeline gốc (q · C^T)
    key_centroids = F.normalize(key_centroids, dim=-1)

    # Tính variance: E[||v - mu||^2] = E[||v||^2] - ||mu||^2
    expanded_v_centroids = torch.gather(value_centroids, 1, labels_v)  # (H, N, D_v)
    sq_diff = (values - expanded_v_centroids).pow(2).sum(dim=-1)  # (H, N)

    var_sum = torch.zeros(H, num_clusters, device=device, dtype=keys.dtype)
    var_sum.scatter_add_(1, labels, sq_diff)
    value_variance = var_sum / counts.clamp(min=1)

    return key_centroids, value_centroids, labels, value_variance


# ---------------------------------------------------------------------------
# Top-level pipeline - tương thích với offline_clustering.py của repo gốc
# ---------------------------------------------------------------------------
def run_value_aware_clustering(
    keys_layers: List[torch.Tensor],
    values_layers: List[torch.Tensor],
    num_clusters: int,
    observation_window: int = 100,
    alpha: float = 1.0,
    beta: float = 0.5,
    num_iters: int = 10,
    print_log: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """
    Drop-in thay thế cho `run_clustering` của repo gốc, nhưng thêm value-aware.
    
    Tương thích shape với code gốc:
    - Input: list of (1, H, N, D) tensors (1 cho batch dim)
    - Output: dicts with format same as run_clustering, plus value_centroids and variance
    
    Args:
        keys_layers:   list[L] of (1, H, N, D_k)
        values_layers: list[L] of (1, H, N, D_v)
        num_clusters: số cluster mỗi head
        observation_window: số tokens cuối không cluster
        alpha, beta: trọng số K, V trong joint clustering
        num_iters: K-means iters
        print_log: in tiến trình
        device: device để chạy. None -> cuda:0 nếu có
    
    Returns:
        key_centroids_dict:   {layer: (1, H, K, D_k)} - tương thích shape gốc
        value_centroids_dict: {layer: (1, H, K, D_v)} - mới
        labels_dict:          {layer: (1, H, N - obs_window)} int64
        value_variance_dict:  {layer: (1, H, K)} - mới
        normalized_variance_dict: {layer: (1, H, K)} - đã normalize [0,1]
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_layers = len(keys_layers)
    assert len(values_layers) == num_layers

    key_centroids_dict = {}
    value_centroids_dict = {}
    labels_dict = {}
    value_variance_dict = {}

    t0 = time.time()
    for layer_idx in range(num_layers):
        if print_log:
            print(f"  [VA] layer {layer_idx}/{num_layers}")

        keys_l = keys_layers[layer_idx].squeeze(0).to(device)    # (H, N, D_k)
        values_l = values_layers[layer_idx].squeeze(0).to(device)  # (H, N, D_v)

        if observation_window > 0:
            keys_to_cluster = keys_l[:, :-observation_window, :]
            values_to_cluster = values_l[:, :-observation_window, :]
        else:
            keys_to_cluster = keys_l
            values_to_cluster = values_l

        N_cluster = keys_to_cluster.shape[1]
        K = min(num_clusters, max(1, N_cluster))

        kc, vc, lbl, vvar = value_aware_kmeans(
            keys_to_cluster.float(),
            values_to_cluster.float(),
            K,
            alpha=alpha,
            beta=beta,
            num_iters=num_iters,
            seed=layer_idx,
        )

        # Cast về dtype gốc và thêm batch dim 1 để tương thích với code gốc
        target_dtype = keys_layers[layer_idx].dtype
        key_centroids_dict[layer_idx] = kc.to(target_dtype).unsqueeze(0)
        value_centroids_dict[layer_idx] = vc.to(target_dtype).unsqueeze(0)
        labels_dict[layer_idx] = lbl.unsqueeze(0).to(torch.int64)
        value_variance_dict[layer_idx] = vvar.to(target_dtype).unsqueeze(0)

    if print_log:
        print(f"  [VA] Total clustering time: {time.time() - t0:.2f}s")

    # Normalize variance per (layer, head)
    normalized_variance_dict = normalize_value_variance(value_variance_dict)

    return (
        key_centroids_dict,
        value_centroids_dict,
        labels_dict,
        value_variance_dict,
        normalized_variance_dict,
    )


def normalize_value_variance(
    value_variance_dict: Dict[int, torch.Tensor],
    method: str = "minmax",
) -> Dict[int, torch.Tensor]:
    """
    Chuẩn hóa value variance về [0, 1] per (layer, head).
    
    Args:
        method: 'minmax' (default), 'mean' (chia trung bình), 'rank' (rank/K)
    """
    normalized = {}
    for layer_idx, vvar in value_variance_dict.items():
        # vvar shape: (1, H, K) hoặc (H, K)
        squeezed = vvar.squeeze(0) if vvar.dim() == 3 else vvar  # (H, K)

        if method == "minmax":
            v_min = squeezed.min(dim=-1, keepdim=True).values
            v_max = squeezed.max(dim=-1, keepdim=True).values
            v_norm = (squeezed - v_min) / (v_max - v_min + 1e-8)
        elif method == "mean":
            v_mean = squeezed.mean(dim=-1, keepdim=True).clamp(min=1e-8)
            v_norm = (squeezed / v_mean).clamp(0, 2) / 2  # cap ở 1
        elif method == "rank":
            ranks = squeezed.argsort(dim=-1).argsort(dim=-1).float()
            K = squeezed.shape[-1]
            v_norm = ranks / max(K - 1, 1)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Restore original dim
        if vvar.dim() == 3:
            v_norm = v_norm.unsqueeze(0)
        normalized[layer_idx] = v_norm
    return normalized


# ---------------------------------------------------------------------------
# Compatibility wrapper: same signature as repo's run_clustering
# ---------------------------------------------------------------------------
def run_clustering_compat(
    tdict: Dict[int, torch.Tensor],
    vdict: Optional[Dict[int, torch.Tensor]],
    num_clusters: int,
    observation_window: int = 100,
    alpha: float = 1.0,
    beta: float = 0.5,
    print_log: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[Dict, Dict]:
    """
    Wrapper trả về CHỈ (centroids_tensor_dict, centroids_labels_dict) giống repo gốc.
    Side effect: lưu value_variance vào file riêng nếu cần.
    
    Khi vdict=None hoặc beta=0 -> hành xử giống run_clustering gốc.
    """
    keys_list = [tdict[i] for i in sorted(tdict.keys())]
    if vdict is not None and beta > 0:
        values_list = [vdict[i] for i in sorted(vdict.keys())]
    else:
        # Fallback: dùng keys làm placeholder cho values khi beta=0
        # (clustering vẫn dựa hoàn toàn trên keys)
        values_list = keys_list
        beta = 0.0

    kc_dict, _, lbl_dict, _, _ = run_value_aware_clustering(
        keys_list, values_list, num_clusters,
        observation_window=observation_window,
        alpha=alpha, beta=beta,
        print_log=print_log, device=device,
    )
    return kc_dict, lbl_dict
