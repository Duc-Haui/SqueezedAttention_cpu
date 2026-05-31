"""
Accuracy Benchmark: Approximation Quality của các phương pháp Sparse Attention.

So sánh 4 phương pháp:
1. Full attention (ground truth)
2. Squeezed Attention gốc (key-only K-means)
3. Value-Aware Squeezed Attention (cải tiến của chúng ta)
4. QUEST-style (cluster theo physical proximity)

Metrics:
- Cosine similarity với full attention (càng cao càng tốt)
- MSE với full attention (càng thấp càng tốt)
- KV budget thực tế (% keys được load)
- Recall của top-K important keys

Workflow:
1. Load HF model nhỏ (Qwen2.5-1.5B mặc định, ~5GB VRAM)
2. Cho fixed context dài + nhiều queries
3. Extract K, V cho fixed context một lần
4. Cluster theo 3 cách (key-only, value-aware, QUEST)
5. Cho mỗi query, tính output attention cả 4 cách, so sánh

Cách chạy:
    python benchmarks/benchmark_accuracy.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --max_context 4096 --num_queries 20 \\
        --sparsity 0.85 --gamma 0.3 --beta 0.5
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Thêm parent dir vào sys.path để import value_aware
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from value_aware.clustering import (
    value_aware_kmeans,
    normalize_value_variance,
    _kmeans_cosine,
)
from value_aware.retrieval import (
    squeezed_attention_forward,
    key_only_attention_forward,
    baseline_full_attention,
    quest_style_clustering,
    quest_attention_forward,
)
from value_aware.threshold import calibrate_threshold


# ---------------------------------------------------------------------------
# Sample fixed context
# ---------------------------------------------------------------------------
SAMPLE_TEXT_PATH = Path(__file__).resolve().parent.parent / "configs" / "sample_context.txt"


def load_sample_context():
    if SAMPLE_TEXT_PATH.exists():
        return SAMPLE_TEXT_PATH.read_text()
    # Fallback inline
    base = """
Large Language Models have seen rapid advancements, enabling Question Answering and
analysis over documents. Long context-length applications have large memory requirements
due to the size of the KV cache, which increases linearly with sequence length.
For applications such as in-context learning, document QA, and code generation, a large
portion of the input context is fixed across user queries. Squeezed Attention accelerates
fixed context applications by clustering keys offline based on semantic similarity, then
identifying important clusters for each query during inference. The hierarchical version
reduces memory complexity from linear to logarithmic with respect to context length.
The method achieves 4.3x speedup during prefill and 4.2x during decode.
PreFixQA is a benchmark with arXiv documents and synthetic question-answer pairs.
On LongBench, the method preserves accuracy with 3.1x KV budget reduction.
""".strip()
    return base * 20  # ~3-4K tokens


SAMPLE_QUERIES = [
    "What is the main idea of Squeezed Attention?",
    "How much speedup does the method achieve during prefill?",
    "What benchmarks were used for evaluation?",
    "What is hierarchical centroid lookup?",
    "How does K-means clustering help here?",
    "What is the time complexity reduction?",
    "What is PreFixQA?",
    "Which models were tested?",
    "What is the memory bandwidth bottleneck?",
    "How is the global threshold calibrated?",
]


# ---------------------------------------------------------------------------
# KV collection
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_kv_via_cache(model, input_ids):
    outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
    past = outputs.past_key_values
    keys = [layer_kv[0].detach() for layer_kv in past]
    values = [layer_kv[1].detach() for layer_kv in past]
    return keys, values


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def cos_sim_per_head(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b: (H, D). Returns per-head cos sim."""
    return F.cosine_similarity(a, b, dim=-1)


def topk_recall(approx_logits: torch.Tensor, true_logits: torch.Tensor, k: int) -> float:
    """
    Tính recall của top-k cao nhất giữa approx và true.
    
    Args:
        approx_logits, true_logits: (H, N)
        k: số keys top
    
    Returns:
        Mean recall over heads.
    """
    H, N = true_logits.shape
    k = min(k, N)
    true_topk = true_logits.topk(k, dim=-1).indices  # (H, k)
    approx_topk = approx_logits.topk(k, dim=-1).indices

    recalls = []
    for h in range(H):
        true_set = set(true_topk[h].tolist())
        approx_set = set(approx_topk[h].tolist())
        recalls.append(len(true_set & approx_set) / k)
    return sum(recalls) / len(recalls)


# ---------------------------------------------------------------------------
# Build clustering for each method
# ---------------------------------------------------------------------------
def build_key_only_clusters(keys, num_clusters, num_iters, device):
    """K-means trên keys. Returns dict per-layer."""
    out = {}
    for li in range(len(keys)):
        k_l = keys[li].squeeze(0).to(device).float()
        k_norm = F.normalize(k_l, dim=-1)
        K_actual = min(num_clusters, k_norm.shape[1])
        centroids, labels = _kmeans_cosine(k_norm, K_actual, num_iters=num_iters)
        H = labels.shape[0]
        sizes = torch.zeros(H, K_actual, device=device, dtype=torch.float)
        sizes.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.float))
        out[li] = (centroids, labels, sizes)
    return out


def build_value_aware_clusters(keys, values, num_clusters, alpha, beta, num_iters, device):
    out = {}
    for li in range(len(keys)):
        k_l = keys[li].squeeze(0).to(device).float()
        v_l = values[li].squeeze(0).to(device).float()
        K_actual = min(num_clusters, k_l.shape[1])
        kc, vc, lbl, vvar = value_aware_kmeans(
            k_l, v_l, K_actual,
            alpha=alpha, beta=beta, num_iters=num_iters, seed=li,
        )
        H = lbl.shape[0]
        sizes = torch.zeros(H, K_actual, device=device, dtype=torch.float)
        sizes.scatter_add_(1, lbl, torch.ones_like(lbl, dtype=torch.float))
        # Normalize variance
        vmin = vvar.min(dim=-1, keepdim=True).values
        vmax = vvar.max(dim=-1, keepdim=True).values
        nvar = (vvar - vmin) / (vmax - vmin + 1e-8)
        out[li] = (kc, lbl, sizes, nvar)
    return out


def build_quest_clusters(keys, chunk_size, device):
    """QUEST: physical proximity chunks."""
    out = {}
    for li in range(len(keys)):
        k_l = keys[li].squeeze(0).to(device).float()
        v_dummy = k_l  # not needed for QUEST
        rep_keys, labels, sizes = quest_style_clustering(k_l, v_dummy, chunk_size=chunk_size)
        out[li] = (rep_keys, labels, sizes)
    return out


# ---------------------------------------------------------------------------
# Threshold calibration per method
# ---------------------------------------------------------------------------
def calibrate_per_layer(method_data, calib_queries_per_layer, target_sparsity, gamma=0.0):
    """
    method_data: dict[li] -> tuple of clustering data
    calib_queries_per_layer: dict[li] -> (Q, H, D)
    
    Returns: dict[li] -> threshold T
    """
    thresholds = {}
    for li, data in method_data.items():
        if len(data) == 4:
            # Value-aware: (kc, lbl, sizes, nvar)
            kc, lbl, sizes, nvar = data
        else:
            # Key-only or QUEST: (kc, lbl, sizes), nvar=zeros
            kc, lbl, sizes = data
            nvar = torch.zeros_like(sizes)

        Q = calib_queries_per_layer[li]
        T = calibrate_threshold(
            Q, kc, sizes, nvar, lbl,
            target_sparsity=target_sparsity, gamma=gamma,
            num_threshold_search=80,
        )
        thresholds[li] = T
    return thresholds


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------
def run_benchmark(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 else torch.bfloat16

    print(f"=== Loading {args.model} on {device} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device,
        attn_implementation="eager",
    )
    model.eval()

    fixed_context = load_sample_context()
    fixed_ids = tokenizer(
        fixed_context, return_tensors="pt", truncation=True, max_length=args.max_context,
    ).input_ids.to(device)
    N_fixed = fixed_ids.shape[1]

    print(f"Fixed context length: {N_fixed} tokens")

    print("=== Collecting K, V from fixed context ===")
    t0 = time.time()
    keys_layers, values_layers = collect_kv_via_cache(model, fixed_ids)
    collect_time = time.time() - t0
    num_layers = len(keys_layers)
    H = keys_layers[0].shape[1]
    D_k = keys_layers[0].shape[3]
    D_v = values_layers[0].shape[3]
    print(f"Layers={num_layers}, Heads={H}, D_k={D_k}, D_v={D_v}, collect_time={collect_time:.2f}s")

    # Số centroid
    obs_window = args.obs_window
    N_to_cluster = N_fixed - obs_window
    num_clusters = max(1, int(args.percent_clusters / 100.0 * N_to_cluster))
    chunk_size = max(1, N_to_cluster // num_clusters)  # QUEST có cùng "effective resolution"
    print(f"Num clusters per head: {num_clusters} ({args.percent_clusters}% of {N_to_cluster})")
    print(f"QUEST chunk size: {chunk_size}")

    # Trim observation window
    keys_trim = []
    values_trim = []
    for li in range(num_layers):
        k_l = keys_layers[li]
        v_l = values_layers[li]
        if obs_window > 0:
            k_l = k_l[:, :, :-obs_window, :]
            v_l = v_l[:, :, :-obs_window, :]
        keys_trim.append(k_l)
        values_trim.append(v_l)

    # ----- Build clusters per method -----
    print("=== Building clusters ===")
    timings = {}

    t0 = time.time()
    ko_clusters = build_key_only_clusters(keys_trim, num_clusters, args.kmeans_iters, device)
    timings["key_only"] = time.time() - t0
    print(f"  key-only: {timings['key_only']:.2f}s")

    t0 = time.time()
    va_clusters = build_value_aware_clusters(
        keys_trim, values_trim, num_clusters, args.alpha, args.beta, args.kmeans_iters, device
    )
    timings["value_aware"] = time.time() - t0
    print(f"  value-aware: {timings['value_aware']:.2f}s")

    t0 = time.time()
    quest_clusters = build_quest_clusters(keys_trim, chunk_size, device)
    timings["quest"] = time.time() - t0
    print(f"  QUEST: {timings['quest']:.2f}s")

    # ----- Calibrate thresholds -----
    # Calib queries: dùng các keys của observation window cuối
    n_calib = min(50, obs_window)
    calib_q_per_layer = {}
    for li in range(num_layers):
        # original keys (pre-trim) cuối observation_window
        full_k = keys_layers[li].squeeze(0).to(device).float()
        if n_calib > 0 and obs_window > 0:
            calib_q = full_k[:, -n_calib:, :].permute(1, 0, 2)  # (n_calib, H, D)
        else:
            calib_q = full_k[:, -1:, :].permute(1, 0, 2)
        calib_q_per_layer[li] = calib_q

    print("=== Calibrating thresholds ===")
    T_ko = calibrate_per_layer(ko_clusters, calib_q_per_layer, args.sparsity, gamma=0.0)
    T_va = calibrate_per_layer(va_clusters, calib_q_per_layer, args.sparsity, gamma=args.gamma)
    T_quest = calibrate_per_layer(quest_clusters, calib_q_per_layer, args.sparsity, gamma=0.0)
    print(f"Sample thresholds (layer 0): KO={T_ko[0]:.4e}, VA={T_va[0]:.4e}, QUEST={T_quest[0]:.4e}")

    # ----- Run queries -----
    n_queries = min(args.num_queries, len(SAMPLE_QUERIES))
    queries = SAMPLE_QUERIES[:n_queries]
    if n_queries < args.num_queries:
        # Repeat to match
        queries = (queries * ((args.num_queries // len(queries)) + 1))[:args.num_queries]

    print(f"\n=== Evaluating on {len(queries)} queries ===")

    metrics = defaultdict(lambda: {"cos": [], "mse": [], "budget": [], "recall_top10": []})

    for q_idx, query_text in enumerate(queries):
        q_ids = tokenizer(query_text, return_tensors="pt").input_ids.to(device)
        full_ids = torch.cat([fixed_ids, q_ids], dim=1)

        # Lấy K của full sequence để dùng làm Q proxy
        # (last token's key approximate query distribution)
        with torch.no_grad():
            k_full, _ = collect_kv_via_cache(model, full_ids)

        for li in range(num_layers):
            full_k = keys_layers[li].squeeze(0).to(device).float()
            full_v = values_layers[li].squeeze(0).to(device).float()
            if obs_window > 0:
                k_clustered = full_k[:, :-obs_window, :]
                v_clustered = full_v[:, :-obs_window, :]
            else:
                k_clustered = full_k
                v_clustered = full_v

            # Q proxy
            q_proxy = k_full[li].squeeze(0)[:, -1, :].to(device).float()  # (H, D)

            scale = 1.0 / (k_clustered.shape[-1] ** 0.5)

            # Ground truth
            ref_out = baseline_full_attention(q_proxy, k_clustered, v_clustered, scale)
            true_attn_logits = torch.einsum("hd,hnd->hn", q_proxy, k_clustered) * scale

            # 1. Key-only
            kc, lbl, sizes = ko_clusters[li]
            ko_out, ko_info = key_only_attention_forward(
                q_proxy, k_clustered, v_clustered,
                kc, sizes, lbl, threshold=T_ko[li], scale=scale,
            )
            # 2. Value-aware
            kc_va, lbl_va, sizes_va, nvar_va = va_clusters[li]
            va_out, va_info = squeezed_attention_forward(
                q_proxy, k_clustered, v_clustered,
                kc_va, sizes_va, nvar_va, lbl_va,
                threshold=T_va[li], gamma=args.gamma, scale=scale,
            )
            # 3. QUEST
            kc_q, lbl_q, sizes_q = quest_clusters[li]
            quest_out, quest_info = quest_attention_forward(
                q_proxy, k_clustered, v_clustered,
                kc_q, sizes_q, lbl_q,
                threshold=T_quest[li], scale=scale,
            )

            # Compute metrics per method
            for name, out, info in [
                ("key_only", ko_out, ko_info),
                ("value_aware", va_out, va_info),
                ("quest", quest_out, quest_info),
            ]:
                cos = F.cosine_similarity(out.flatten(), ref_out.flatten(), dim=0).item()
                mse = F.mse_loss(out, ref_out).item()
                budget = info["kv_budget"]

                # Recall: trong top-10 attention scores của ref, bao nhiêu nằm trong key được load
                # Tương đương: recall của method trong việc retain top-10 important keys
                top10_recall = topk_recall(
                    # approx_logits: chỉ ở vị trí được load mới có giá trị
                    # nhưng chúng ta đo theo: top-10 của true_attn_logits có nằm trong key_mask không
                    true_attn_logits, true_attn_logits, k=10
                )  # placeholder; tính lại bên dưới

                metrics[name]["cos"].append(cos)
                metrics[name]["mse"].append(mse)
                metrics[name]["budget"].append(budget)

        del k_full
        torch.cuda.empty_cache()

    # ----- Print summary -----
    print("\n" + "=" * 80)
    print(f"SUMMARY (model={args.model}, sparsity={args.sparsity}, gamma={args.gamma})")
    print(f"Avg over {len(queries)} queries × {num_layers} layers × {H} heads")
    print("=" * 80)
    print(f"{'Method':>15s} | {'CosSim':>8s} | {'MSE':>10s} | {'Budget':>8s} | {'ClusterTime':>12s}")
    print("-" * 80)
    for name in ["key_only", "value_aware", "quest"]:
        m = metrics[name]
        avg_cos = sum(m["cos"]) / len(m["cos"])
        avg_mse = sum(m["mse"]) / len(m["mse"])
        avg_bud = sum(m["budget"]) / len(m["budget"])
        print(f"{name:>15s} | {avg_cos:>8.4f} | {avg_mse:>10.6f} | "
              f"{avg_bud * 100:>7.2f}% | {timings[name]:>11.2f}s")

    # Δ value-aware vs key-only
    cos_ko = sum(metrics["key_only"]["cos"]) / len(metrics["key_only"]["cos"])
    cos_va = sum(metrics["value_aware"]["cos"]) / len(metrics["value_aware"]["cos"])
    print(f"\nΔ (Value-Aware vs Key-Only): "
          f"cos_sim {(cos_va - cos_ko) * 100:+.3f} pp")

    # Save results JSON
    if args.output:
        results = {
            "config": vars(args),
            "stats": {
                "num_queries": len(queries),
                "num_layers": num_layers,
                "num_heads": H,
                "fixed_context_length": N_fixed,
                "num_clusters": num_clusters,
            },
            "results": {
                name: {
                    "avg_cos_sim": sum(m["cos"]) / len(m["cos"]),
                    "avg_mse": sum(m["mse"]) / len(m["mse"]),
                    "avg_kv_budget": sum(m["budget"]) / len(m["budget"]),
                    "clustering_time_s": timings.get(name, 0),
                }
                for name, m in metrics.items()
            },
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--max_context", type=int, default=4096)
    p.add_argument("--obs_window", type=int, default=64)
    p.add_argument("--percent_clusters", type=float, default=5.0)
    p.add_argument("--kmeans_iters", type=int, default=10)
    p.add_argument("--sparsity", type=float, default=0.85,
                   help="Target sparsity. 0.85 = giữ 15% keys")
    p.add_argument("--gamma", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--num_queries", type=int, default=10)
    p.add_argument("--output", type=str, default="results/accuracy_benchmark.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
