"""
Ablation Study cho Value-Aware Squeezed Attention.

Đo đóng góp của từng thành phần:
1. (β=0, γ=0): Squeezed Attention gốc baseline
2. (β=0.5, γ=0): Joint K-V clustering (chỉ thay đổi clustering)
3. (β=0, γ=0.3): Variance boost cho key-only clustering (chỉ thay đổi retrieve)
4. (β=0.5, γ=0.3): Full method
5. (β=0.5, γ=0.5): Aggressive variance boost

Tương tự Bảng 6 trong paper Squeezed Attention - nhưng cho variants của chúng ta.

Cách chạy:
    python benchmarks/benchmark_ablation.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import synthetic_benchmark logic
from benchmarks.synthetic_benchmark import synthetic_kv, synthetic_queries, run_one_setting


# Ablation configurations
ABLATIONS = [
    {"name": "Baseline (β=0, γ=0)",     "beta": 0.0, "gamma": 0.0, "label": "baseline"},
    {"name": "Joint K-V only (β=0.5, γ=0)",  "beta": 0.5, "gamma": 0.0, "label": "joint_kv"},
    {"name": "Variance boost only (β=0, γ=0.3)", "beta": 0.0, "gamma": 0.3, "label": "var_boost"},
    {"name": "Full method (β=0.5, γ=0.3)",   "beta": 0.5, "gamma": 0.3, "label": "full"},
    {"name": "Aggressive (β=0.5, γ=0.5)",    "beta": 0.5, "gamma": 0.5, "label": "aggressive"},
    {"name": "V-dominant (β=1.0, γ=0.3)",    "beta": 1.0, "gamma": 0.3, "label": "v_dominant"},
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--N", type=int, default=512)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--num_queries", type=int, default=20)
    p.add_argument("--diversity", type=float, default=1.0)
    p.add_argument("--sparsity", type=float, default=0.85)
    p.add_argument("--num_seeds", type=int, default=3, help="Multiple seeds để giảm variance")
    p.add_argument("--output", default="results/ablation.json")
    args = p.parse_args()

    print("=" * 80)
    print(f"ABLATION STUDY (similar to Table 6 in paper)")
    print(f"Setup: K={args.K} clusters, H={args.H} heads, N={args.N} keys")
    print(f"       diversity={args.diversity}, target_sparsity={args.sparsity}")
    print(f"       averaged over {args.num_seeds} seeds")
    print("=" * 80)

    print(f"\n{'Configuration':>30s} | {'CosSim':>8s} | {'MSE':>10s} | {'Budget':>8s} | {'Δ vs base':>10s}")
    print("-" * 90)

    results = {}
    baseline_cos = None

    for ab in ABLATIONS:
        # Average over seeds
        cos_list = []
        mse_list = []
        bud_list = []
        for seed in range(args.num_seeds):
            torch.manual_seed(seed)
            r = run_one_setting(
                diversity=args.diversity,
                gamma=ab["gamma"],
                beta=ab["beta"],
                K=args.K,
                target_sparsity=args.sparsity,
                num_queries=args.num_queries,
                H=args.H, N=args.N, D=args.D,
            )
            # value_aware là phương pháp chúng ta đang đo
            method_key = "value_aware" if (ab["beta"] > 0 or ab["gamma"] > 0) else "key_only"
            cos_list.append(r[method_key]["cos"])
            mse_list.append(r[method_key]["mse"])
            bud_list.append(r[method_key]["budget"])

        avg_cos = sum(cos_list) / len(cos_list)
        avg_mse = sum(mse_list) / len(mse_list)
        avg_bud = sum(bud_list) / len(bud_list)

        if baseline_cos is None:
            baseline_cos = avg_cos
            delta_str = "—"
        else:
            delta = (avg_cos - baseline_cos) * 100
            delta_str = f"{delta:+.3f}pp"

        print(f"{ab['name']:>30s} | {avg_cos:>8.4f} | {avg_mse:>10.6f} | {avg_bud*100:>7.2f}% | {delta_str:>10s}")
        results[ab["label"]] = {
            "config": {"beta": ab["beta"], "gamma": ab["gamma"]},
            "cos": avg_cos,
            "mse": avg_mse,
            "budget": avg_bud,
            "name": ab["name"],
        }

    print("\nDiễn giải:")
    print(" - Joint K-V only: chỉ thay đổi clustering, không boost retrieval")
    print(" - Variance boost only: chỉ boost retrieve, dùng key-only clustering")
    print(" - Full method: kết hợp cả hai")
    print(" - Δ pp dương = cải thiện so với baseline")

    # ----- Diversity sweep ablation -----
    print("\n" + "=" * 80)
    print("DIVERSITY SWEEP (Full method vs Baseline)")
    print("=" * 80)
    print(f"\n{'Diversity':>10s} | {'Method':>14s} | {'CosSim':>8s} | {'MSE':>10s} | {'Δ pp':>8s}")
    print("-" * 75)

    div_results = {}
    for div in [0.1, 0.5, 1.0, 2.0]:
        baseline_cos_d = None
        for ab in [ABLATIONS[0], ABLATIONS[3]]:  # baseline & full
            cos_list = []
            for seed in range(args.num_seeds):
                torch.manual_seed(seed)
                r = run_one_setting(
                    diversity=div, gamma=ab["gamma"], beta=ab["beta"],
                    K=args.K, target_sparsity=args.sparsity,
                    num_queries=args.num_queries,
                    H=args.H, N=args.N, D=args.D,
                )
                method_key = "value_aware" if (ab["beta"] > 0 or ab["gamma"] > 0) else "key_only"
                cos_list.append(r[method_key]["cos"])
            avg = sum(cos_list) / len(cos_list)
            if baseline_cos_d is None:
                baseline_cos_d = avg
                delta = "—"
            else:
                delta = f"{(avg - baseline_cos_d)*100:+.3f}"
            print(f"{div:>10.2f} | {ab['name'][:14]:>14s} | {avg:>8.4f} | "
                  f"{'-':>10s} | {delta:>8s}")
        div_results[div] = avg
        print()

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "config": vars(args),
                "ablation_results": results,
            }, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
