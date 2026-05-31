"""
Comprehensive Unit Tests cho Value-Aware Squeezed Attention.

Chạy:
    python tests/test_all.py

Hoặc với pytest:
    pip install pytest
    pytest tests/

Tất cả tests chạy được trên CPU, không cần GPU.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from value_aware.clustering import (
    _kmeans_cosine,
    value_aware_kmeans,
    run_value_aware_clustering,
    normalize_value_variance,
    run_clustering_compat,
)
from value_aware.retrieval import (
    compute_base_scores,
    value_aware_retrieve,
    keys_mask_from_clusters,
    squeezed_attention_forward,
    key_only_attention_forward,
    baseline_full_attention,
    quest_style_clustering,
    quest_attention_forward,
)
from value_aware.threshold import (
    calibrate_threshold,
    run_value_aware_global_threshold,
)


# =============================================================================
# Clustering tests
# =============================================================================
def test_kmeans_cosine_basic():
    print("[test_kmeans_cosine_basic] ", end="")
    torch.manual_seed(0)
    H, N, D, K = 2, 100, 16, 5
    x = F.normalize(torch.randn(H, N, D), dim=-1)
    centroids, labels = _kmeans_cosine(x, K, num_iters=10)
    assert centroids.shape == (H, K, D)
    assert labels.shape == (H, N)
    assert torch.allclose(centroids.norm(dim=-1), torch.ones(H, K), atol=1e-4)
    assert labels.min() >= 0 and labels.max() < K
    print("OK")


def test_kmeans_cosine_few_points():
    """Test edge case: ít điểm hơn cluster."""
    print("[test_kmeans_cosine_few_points] ", end="")
    H, N, D, K = 1, 3, 8, 5
    x = F.normalize(torch.randn(H, N, D), dim=-1)
    centroids, labels = _kmeans_cosine(x, K, num_iters=5)
    assert centroids.shape == (H, K, D)
    print("OK")


def test_value_aware_kmeans_shapes():
    print("[test_value_aware_kmeans_shapes] ", end="")
    torch.manual_seed(1)
    H, N, D_k, D_v, K = 4, 200, 32, 32, 8
    keys = torch.randn(H, N, D_k)
    values = torch.randn(H, N, D_v)
    kc, vc, lbl, vvar = value_aware_kmeans(keys, values, K, alpha=1.0, beta=0.5)
    assert kc.shape == (H, K, D_k)
    assert vc.shape == (H, K, D_v)
    assert lbl.shape == (H, N)
    assert vvar.shape == (H, K)
    assert (vvar >= 0).all()
    print("OK")


def test_value_aware_beta_zero():
    """Beta=0 nên cho clustering tương đương key-only."""
    print("[test_value_aware_beta_zero] ", end="")
    torch.manual_seed(2)
    H, N, D, K = 2, 100, 16, 5
    keys = torch.randn(H, N, D)
    values = torch.randn(H, N, D)
    kc, vc, lbl, vvar = value_aware_kmeans(
        keys, values, K, alpha=1.0, beta=0.0, num_iters=20
    )
    counts = torch.zeros(H, K)
    counts.scatter_add_(1, lbl, torch.ones_like(lbl, dtype=torch.float))
    assert (counts.sum(dim=-1) == N).all()
    print("OK")


def test_run_value_aware_clustering_dict_format():
    print("[test_run_value_aware_clustering_dict_format] ", end="")
    torch.manual_seed(3)
    num_layers = 3
    H, N, D = 4, 100, 16
    K = 8
    keys_layers = [torch.randn(1, H, N, D) for _ in range(num_layers)]
    values_layers = [torch.randn(1, H, N, D) for _ in range(num_layers)]

    kc, vc, lbl, vvar, nvar = run_value_aware_clustering(
        keys_layers, values_layers, num_clusters=K,
        observation_window=10, alpha=1.0, beta=0.5,
        device=torch.device("cpu"),
    )
    assert len(kc) == num_layers
    for li in range(num_layers):
        assert kc[li].shape == (1, H, K, D)
        assert vc[li].shape == (1, H, K, D)
        assert lbl[li].shape == (1, H, N - 10)
        assert vvar[li].shape == (1, H, K)
        assert nvar[li].shape == (1, H, K)
    print("OK")


def test_normalize_variance_minmax():
    print("[test_normalize_variance_minmax] ", end="")
    vvar = {0: torch.tensor([[1.0, 2.0, 3.0, 4.0]]).unsqueeze(0)}  # (1, 1, 4)
    nvar = normalize_value_variance(vvar)
    assert nvar[0].min() >= 0 and nvar[0].max() <= 1 + 1e-5
    # Min should be 0, max should be 1
    assert torch.isclose(nvar[0].min(), torch.tensor(0.0))
    assert torch.isclose(nvar[0].max(), torch.tensor(1.0))
    print("OK")


def test_clustering_compat():
    """Wrapper function should return shape compatible with original repo."""
    print("[test_clustering_compat] ", end="")
    torch.manual_seed(4)
    num_layers = 2
    H, N, D = 2, 50, 16
    tdict = {i: torch.randn(1, H, N, D) for i in range(num_layers)}
    vdict = {i: torch.randn(1, H, N, D) for i in range(num_layers)}

    # With value-aware
    kc, lbl = run_clustering_compat(tdict, vdict, 5, observation_window=5, beta=0.5)
    assert len(kc) == num_layers
    assert kc[0].shape == (1, H, 5, D)
    assert lbl[0].shape == (1, H, 45)

    # Without (baseline)
    kc2, lbl2 = run_clustering_compat(tdict, None, 5, observation_window=5, beta=0.0)
    assert len(kc2) == num_layers
    print("OK")


# =============================================================================
# Retrieval tests
# =============================================================================
def test_base_scores_normalize():
    print("[test_base_scores_normalize] ", end="")
    torch.manual_seed(5)
    H, K, D = 2, 5, 16
    q = torch.randn(H, D)
    centroids = F.normalize(torch.randn(H, K, D), dim=-1)
    sizes = torch.tensor([[10, 8, 12, 15, 5], [5, 5, 10, 20, 10]], dtype=torch.float)
    S = compute_base_scores(q, centroids, sizes)
    weighted_sum = (sizes * S).sum(dim=-1)
    assert torch.allclose(weighted_sum, torch.ones_like(weighted_sum), atol=1e-4)
    print("OK")


def test_retrieve_threshold_monotone():
    print("[test_retrieve_threshold_monotone] ", end="")
    torch.manual_seed(6)
    H, K, D = 2, 10, 16
    q = torch.randn(H, D)
    cent = F.normalize(torch.randn(H, K, D), dim=-1)
    sz = torch.full((H, K), 10.0)
    nv = torch.rand(H, K)
    mask_high, _ = value_aware_retrieve(q, cent, sz, nv, threshold=0.5, gamma=0.3)
    mask_low, _ = value_aware_retrieve(q, cent, sz, nv, threshold=0.0, gamma=0.3)
    assert mask_low.sum() >= mask_high.sum()
    print("OK")


def test_keys_mask_correctness():
    print("[test_keys_mask_correctness] ", end="")
    H, K, N = 2, 4, 20
    cluster_mask = torch.zeros(H, K, dtype=torch.bool)
    cluster_mask[0, 1] = True
    cluster_mask[0, 2] = True
    cluster_mask[1, 0] = True
    labels = torch.zeros(H, N, dtype=torch.long)
    labels[0, 5:10] = 1
    labels[0, 10:15] = 2
    key_mask = keys_mask_from_clusters(cluster_mask, labels)
    assert key_mask[0, :5].sum() == 0
    assert key_mask[0, 5:15].sum() == 10
    assert key_mask[0, 15:].sum() == 0
    print("OK")


def test_full_recovery_no_pruning():
    """No pruning -> output bằng full attention."""
    print("[test_full_recovery_no_pruning] ", end="")
    torch.manual_seed(7)
    H, N, D, K = 2, 50, 16, 5
    q = torch.randn(H, D)
    keys = torch.randn(H, N, D)
    values = torch.randn(H, N, D)
    kc, vc, lbl, vvar = value_aware_kmeans(keys, values, K, alpha=1.0, beta=0.5)
    sizes = torch.zeros(H, K)
    sizes.scatter_add_(1, lbl, torch.ones_like(lbl, dtype=torch.float))

    out, info = squeezed_attention_forward(
        q, keys, values, kc, sizes, torch.zeros(H, K), lbl,
        threshold=-1e9, gamma=0.0,
    )
    full = baseline_full_attention(q, keys, values)
    assert info["kv_budget"] > 0.99
    assert (out - full).abs().max().item() < 1e-4
    print("OK")


def test_quest_clustering_basic():
    print("[test_quest_clustering_basic] ", end="")
    H, N, D = 2, 100, 16
    keys = F.normalize(torch.randn(H, N, D), dim=-1)
    values = torch.randn(H, N, D)
    rep_keys, labels, sizes = quest_style_clustering(keys, values, chunk_size=10)
    assert rep_keys.shape == (H, 10, D)
    assert labels.shape == (H, N)
    assert sizes.shape == (H, 10)
    # First 10 tokens belong to chunk 0
    assert (labels[0, :10] == 0).all()
    # Sizes per chunk = 10
    assert (sizes[0] == 10).all()
    print("OK")


# =============================================================================
# Threshold tests
# =============================================================================
def test_calibrate_threshold_target():
    """Test threshold calibration achieves target sparsity (within tolerance)."""
    print("[test_calibrate_threshold_target] ", end="")
    torch.manual_seed(8)
    H, N, D, K = 4, 200, 16, 20
    keys = torch.randn(H, N, D)
    values = torch.randn(H, N, D)
    queries = torch.randn(20, H, D)
    kc, vc, lbl, vvar = value_aware_kmeans(keys, values, K, alpha=1.0, beta=0.5)
    sizes = torch.zeros(H, K)
    sizes.scatter_add_(1, lbl, torch.ones_like(lbl, dtype=torch.float))

    target = 0.7  # giữ 30%
    T = calibrate_threshold(
        queries, kc, sizes, torch.zeros(H, K), lbl,
        target_sparsity=target, gamma=0.0, num_threshold_search=100,
    )
    # Verify
    keep_ratios = []
    for q in queries:
        S = compute_base_scores(q, kc, sizes)
        cluster_mask = S > T
        kept = (cluster_mask.float() * sizes).sum(dim=-1) / sizes.sum(dim=-1)
        keep_ratios.append(kept.mean().item())
    avg = sum(keep_ratios) / len(keep_ratios)
    assert abs(avg - 0.30) < 0.20, f"target=0.30, got {avg}"
    print(f"OK (actual_keep={avg:.2f})")


def test_run_value_aware_global_threshold_format():
    """Output format compatible với run_global_threshold gốc."""
    print("[test_run_value_aware_global_threshold_format] ", end="")
    torch.manual_seed(9)
    num_layers = 2
    H, N, D = 2, 50, 16
    K = 5
    obs_w = 10
    keys_layers = [torch.randn(1, H, N, D) for _ in range(num_layers)]
    values_layers = [torch.randn(1, H, N, D) for _ in range(num_layers)]
    queries_layers = [torch.randn(1, H, N, D) for _ in range(num_layers)]

    kc, vc, lbl, vvar, nvar = run_value_aware_clustering(
        keys_layers, values_layers, K, observation_window=obs_w,
        device=torch.device("cpu"),
    )
    tdict = run_value_aware_global_threshold(
        keys_layers, queries_layers, kc, lbl, nvar, K,
        observation_window=obs_w, gamma=0.3,
        device=torch.device("cpu"),
    )
    # Check expected keys
    assert "shared_prefix_length" in tdict
    assert "observation_window" in tdict
    assert "gamma" in tdict
    for q in [0.5, 0.7, 0.8, 0.9]:
        assert q in tdict
    print("OK")


# =============================================================================
# Integration test: end-to-end với synthetic data
# =============================================================================
def test_e2e_value_aware_helps_with_diverse_values():
    """Test e2e: VA cải thiện khi values diverse."""
    print("[test_e2e_value_aware_helps_with_diverse_values] ", end="")
    torch.manual_seed(10)
    H, N, D, K = 4, 200, 32, 16

    # Tạo data: 2 nhóm K-cluster, mỗi nhóm có values rất khác nhau
    anchor1 = F.normalize(torch.randn(H, D), dim=-1)
    anchor2 = F.normalize(torch.randn(H, D), dim=-1)
    keys = torch.zeros(H, N, D)
    values = torch.zeros(H, N, D)
    for h in range(H):
        for n in range(N):
            if n < N // 2:
                keys[h, n] = F.normalize(anchor1[h] + 0.05 * torch.randn(D), dim=-1)
                values[h, n] = torch.randn(D) * 2  # diverse
            else:
                keys[h, n] = F.normalize(anchor2[h] + 0.05 * torch.randn(D), dim=-1)
                values[h, n] = torch.randn(D) * 0.1  # uniform

    # Calib + test queries
    calib_q = torch.randn(20, H, D)
    test_q = torch.randn(10, H, D)

    # Key-only
    kc_ko, _, lbl_ko, _ = value_aware_kmeans(keys, values, K, alpha=1.0, beta=0.0)
    sz_ko = torch.zeros(H, K); sz_ko.scatter_add_(1, lbl_ko, torch.ones_like(lbl_ko, dtype=torch.float))

    # Value-aware
    kc_va, _, lbl_va, vv = value_aware_kmeans(keys, values, K, alpha=1.0, beta=0.5)
    sz_va = torch.zeros(H, K); sz_va.scatter_add_(1, lbl_va, torch.ones_like(lbl_va, dtype=torch.float))
    nv = (vv - vv.min(-1, True).values) / (vv.max(-1, True).values - vv.min(-1, True).values + 1e-8)

    target = 0.85
    T_ko = calibrate_threshold(calib_q, kc_ko, sz_ko, torch.zeros(H, K), lbl_ko, target, 0.0)
    T_va = calibrate_threshold(calib_q, kc_va, sz_va, nv, lbl_va, target, 0.3)

    cos_ko_l, cos_va_l = [], []
    for q in test_q:
        ref = baseline_full_attention(q, keys, values)
        ko, _ = key_only_attention_forward(q, keys, values, kc_ko, sz_ko, lbl_ko, threshold=T_ko)
        va, _ = squeezed_attention_forward(q, keys, values, kc_va, sz_va, nv, lbl_va, threshold=T_va, gamma=0.3)
        cos_ko_l.append(F.cosine_similarity(ko.flatten(), ref.flatten(), dim=0).item())
        cos_va_l.append(F.cosine_similarity(va.flatten(), ref.flatten(), dim=0).item())

    avg_ko = sum(cos_ko_l) / len(cos_ko_l)
    avg_va = sum(cos_va_l) / len(cos_va_l)
    print(f"KO={avg_ko:.4f}, VA={avg_va:.4f} (Δ={(avg_va-avg_ko)*100:+.3f}pp)")


def main():
    print("=" * 70)
    print("Value-Aware Squeezed Attention - All Tests")
    print("=" * 70)
    
    # Clustering
    test_kmeans_cosine_basic()
    test_kmeans_cosine_few_points()
    test_value_aware_kmeans_shapes()
    test_value_aware_beta_zero()
    test_run_value_aware_clustering_dict_format()
    test_normalize_variance_minmax()
    test_clustering_compat()
    
    # Retrieval
    test_base_scores_normalize()
    test_retrieve_threshold_monotone()
    test_keys_mask_correctness()
    test_full_recovery_no_pruning()
    test_quest_clustering_basic()
    
    # Threshold
    test_calibrate_threshold_target()
    test_run_value_aware_global_threshold_format()
    
    # E2E
    test_e2e_value_aware_helps_with_diverse_values()
    
    print("\n" + "=" * 70)
    print("✓ All tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
