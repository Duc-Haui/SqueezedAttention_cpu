"""
Latency Benchmark cho Value-Aware Squeezed Attention.

Đo latency của các bước:
1. Clustering offline (one-time cost)
2. Centroid lookup online (per-query)
3. Sparse attention forward (per-query)

So sánh với:
- Full attention (baseline)
- Key-only Squeezed Attention
- Value-Aware Squeezed Attention
- QUEST-style

Tương đương với Figure 4 trong paper Squeezed Attention nhưng đơn giản hóa.

Cách chạy:
    python benchmarks/benchmark_latency.py \\
        --context_length 16384 \\
        --num_heads 32 --head_dim 128 \\
        --sparsity 0.9 --num_runs 50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from value_aware.clustering import value_aware_kmeans, _kmeans_cosine
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
from value_aware.threshold import calibrate_threshold


def synth_kv(H, N, D, device, dtype, seed=0):
    """Tạo random K, V."""
    g = torch.Generator(device=device).manual_seed(seed)
    keys = F.normalize(torch.randn(H, N, D, device=device, dtype=dtype, generator=g), dim=-1)
    values = torch.randn(H, N, D, device=device, dtype=dtype, generator=g)
    return keys, values


def time_op(fn, num_runs, warmup=5):
    """Time a callable averaged over num_runs (after warmup)."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / num_runs * 1000  # ms


def benchmark_clustering(args, device, dtype):
    """Đo offline clustering time."""
    print("\n=== OFFLINE CLUSTERING TIME ===")
    H, N, D = args.num_heads, args.context_length, args.head_dim
    K = max(1, int(args.percent_clusters / 100.0 * N))

    keys, values = synth_kv(H, N, D, device, dtype)

    # Key-only
    k_norm = F.normalize(keys.float(), dim=-1)
    t0 = time.perf_counter()
    _kmeans_cosine(k_norm, K, num_iters=args.kmeans_iters)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_ko = time.perf_counter() - t0

    # Value-aware
    t0 = time.perf_counter()
    value_aware_kmeans(keys.float(), values.float(), K,
                      alpha=1.0, beta=0.5, num_iters=args.kmeans_iters)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_va = time.perf_counter() - t0

    # QUEST (mean over chunks)
    t0 = time.perf_counter()
    quest_style_clustering(keys.float(), values.float(),
                           chunk_size=max(1, N // K))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_q = time.perf_counter() - t0

    print(f"Context length: {N}, Heads: {H}, Clusters: {K}")
    print(f"  Key-only K-means:    {t_ko*1000:>8.2f} ms")
    print(f"  Value-aware K-means: {t_va*1000:>8.2f} ms ({t_va/t_ko:.2f}x)")
    print(f"  QUEST chunking:      {t_q*1000:>8.2f} ms")

    return {"key_only": t_ko, "value_aware": t_va, "quest": t_q}


def benchmark_online(args, device, dtype):
    """Đo online lookup + attention time per query."""
    print("\n=== ONLINE PER-QUERY TIME ===")
    H, N, D = args.num_heads, args.context_length, args.head_dim
    K = max(1, int(args.percent_clusters / 100.0 * N))

    keys, values = synth_kv(H, N, D, device, dtype)

    # Pre-cluster
    k_norm = F.normalize(keys.float(), dim=-1)
    kc_ko, lbl_ko = _kmeans_cosine(k_norm, K, num_iters=args.kmeans_iters)
    sizes_ko = torch.zeros(H, K, device=device, dtype=torch.float)
    sizes_ko.scatter_add_(1, lbl_ko, torch.ones_like(lbl_ko, dtype=torch.float))

    kc_va, _, lbl_va, vvar = value_aware_kmeans(
        keys.float(), values.float(), K, alpha=1.0, beta=0.5, num_iters=args.kmeans_iters
    )
    sizes_va = torch.zeros(H, K, device=device, dtype=torch.float)
    sizes_va.scatter_add_(1, lbl_va, torch.ones_like(lbl_va, dtype=torch.float))
    vmin = vvar.min(-1, keepdim=True).values
    vmax = vvar.max(-1, keepdim=True).values
    nvar_va = (vvar - vmin) / (vmax - vmin + 1e-8)

    # QUEST
    chunk_size = max(1, N // K)
    rep_q, lbl_q, sizes_q = quest_style_clustering(keys.float(), values.float(), chunk_size=chunk_size)

    # Generate query
    q = F.normalize(torch.randn(H, D, device=device, dtype=torch.float), dim=-1)

    # Calibrate
    calib_q = q.unsqueeze(0).repeat(20, 1, 1) + 0.1 * torch.randn(20, H, D, device=device)
    T_ko = calibrate_threshold(calib_q, kc_ko, sizes_ko, torch.zeros_like(sizes_ko), lbl_ko,
                              args.sparsity, gamma=0.0, num_threshold_search=50)
    T_va = calibrate_threshold(calib_q, kc_va, sizes_va, nvar_va, lbl_va,
                              args.sparsity, gamma=0.3, num_threshold_search=50)
    T_q = calibrate_threshold(calib_q, rep_q, sizes_q, torch.zeros_like(sizes_q), lbl_q,
                             args.sparsity, gamma=0.0, num_threshold_search=50)

    keys_f = keys.float()
    values_f = values.float()
    scale = 1.0 / (D ** 0.5)

    # Define benchmark fns
    def fn_full():
        return baseline_full_attention(q, keys_f, values_f, scale)

    def fn_ko():
        return key_only_attention_forward(
            q, keys_f, values_f, kc_ko, sizes_ko, lbl_ko,
            threshold=T_ko, scale=scale,
        )

    def fn_va():
        return squeezed_attention_forward(
            q, keys_f, values_f, kc_va, sizes_va, nvar_va, lbl_va,
            threshold=T_va, gamma=0.3, scale=scale,
        )

    def fn_quest():
        return quest_attention_forward(
            q, keys_f, values_f, rep_q, sizes_q, lbl_q,
            threshold=T_q, scale=scale,
        )

    # Time each
    t_full = time_op(fn_full, args.num_runs)
    t_ko = time_op(fn_ko, args.num_runs)
    t_va = time_op(fn_va, args.num_runs)
    t_q = time_op(fn_quest, args.num_runs)

    print(f"Per-query latency (averaged over {args.num_runs} runs after 5 warmup):")
    print(f"  Full attention:  {t_full:>7.3f} ms (baseline)")
    print(f"  Key-only:        {t_ko:>7.3f} ms ({t_full/t_ko:.2f}x speedup)")
    print(f"  Value-aware:     {t_va:>7.3f} ms ({t_full/t_va:.2f}x speedup)")
    print(f"  QUEST:           {t_q:>7.3f} ms ({t_full/t_q:.2f}x speedup)")

    return {
        "full": t_full, "key_only": t_ko, "value_aware": t_va, "quest": t_q,
    }


def benchmark_centroid_lookup_only(args, device, dtype):
    """Đo riêng centroid lookup (không kèm attention)."""
    print("\n=== CENTROID LOOKUP ONLY ===")
    H, N, D = args.num_heads, args.context_length, args.head_dim
    K = max(1, int(args.percent_clusters / 100.0 * N))

    keys, values = synth_kv(H, N, D, device, dtype)
    kc, _, lbl, vvar = value_aware_kmeans(
        keys.float(), values.float(), K, alpha=1.0, beta=0.5, num_iters=args.kmeans_iters
    )
    sizes = torch.zeros(H, K, device=device, dtype=torch.float)
    sizes.scatter_add_(1, lbl, torch.ones_like(lbl, dtype=torch.float))
    vmin = vvar.min(-1, keepdim=True).values
    vmax = vvar.max(-1, keepdim=True).values
    nvar = (vvar - vmin) / (vmax - vmin + 1e-8)

    q = F.normalize(torch.randn(H, D, device=device, dtype=torch.float), dim=-1)

    def fn_lookup_ko():
        return compute_base_scores(q, kc, sizes)

    def fn_lookup_va():
        return value_aware_retrieve(q, kc, sizes, nvar, threshold=0.001, gamma=0.3)

    t_ko = time_op(fn_lookup_ko, args.num_runs)
    t_va = time_op(fn_lookup_va, args.num_runs)

    print(f"  Lookup base score (key-only style): {t_ko:>7.3f} ms")
    print(f"  Lookup with VA boost:              {t_va:>7.3f} ms (+{(t_va-t_ko)/t_ko*100:.1f}%)")
    print(f"  -> overhead value-aware boost: {t_va - t_ko:.4f} ms")

    return {"lookup_key_only": t_ko, "lookup_value_aware": t_va}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--context_length", type=int, default=8192)
    p.add_argument("--num_heads", type=int, default=32)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--percent_clusters", type=float, default=5.0)
    p.add_argument("--sparsity", type=float, default=0.9)
    p.add_argument("--kmeans_iters", type=int, default=10)
    p.add_argument("--num_runs", type=int, default=50)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--output", default="results/latency_benchmark.json")
    args = p.parse_args()

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        dtype = torch.float32
        print(f"=== Running on CPU (slow but works for verify) ===")
    else:
        device = torch.device(f"cuda:{args.device}")
        dtype = torch.float16
        print(f"=== Running on {device} (fp16) ===")

    print(f"Config: ctx={args.context_length}, H={args.num_heads}, D={args.head_dim}")
    print(f"        clusters={args.percent_clusters}%, sparsity={args.sparsity}")

    results = {}
    results["clustering"] = benchmark_clustering(args, device, dtype)
    results["centroid_lookup"] = benchmark_centroid_lookup_only(args, device, dtype)
    results["online"] = benchmark_online(args, device, dtype)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        results["config"] = vars(args)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
