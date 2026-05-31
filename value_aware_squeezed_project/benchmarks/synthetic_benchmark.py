"""
Synthetic Benchmark cho Value-Aware Retrieval.

Mục đích: dùng dữ liệu nhân tạo có kiểm soát để verify rằng cải tiến thực sự có lợi
trong các điều kiện cụ thể (đặc biệt khi value diversity cao).

Chạy CPU được, không cần GPU/model thật. Dùng để:
- Verify code đúng đắn
- Hiểu khi nào value-aware có lợi nhất
- Nhanh chóng tune hyperparameters

Cách chạy:
    python benchmarks/synthetic_benchmark.py
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from value_aware.clustering import value_aware_kmeans, normalize_value_variance
from value_aware.retrieval import (
    squeezed_attention_forward,
    key_only_attention_forward,
    baseline_full_attention,
    quest_style_clustering,
    quest_attention_forward,
)
from value_aware.threshold import calibrate_threshold


def synthetic_kv(H=8, N=512, D=64, num_groups=20, value_diversity=1.0, seed=0):
    """
    Tạo synthetic K, V có cấu trúc cluster.
    
    Args:
        num_groups: số "topic" (anchor keys)
        value_diversity: 0 = values đồng nhất trong topic, 1.0+ = đa dạng
    """
    torch.manual_seed(seed)
    anchors_k = F.normalize(torch.randn(H, num_groups, D), dim=-1)
    anchors_v = torch.randn(H, num_groups, D)

    group_assignments = torch.randint(0, num_groups, (H, N))

    keys = torch.zeros(H, N, D)
    values = torch.zeros(H, N, D)
    for h in range(H):
        for n in range(N):
            g = group_assignments[h, n].item()
            keys[h, n] = F.normalize(
                anchors_k[h, g] + 0.05 * torch.randn(D), dim=-1
            )
            values[h, n] = anchors_v[h, g] + value_diversity * torch.randn(D)
    return keys, values, group_assignments


def synthetic_queries(keys, num_queries, noise=0.1, seed=99):
    """Tạo queries gần với một số keys ngẫu nhiên trong context."""
    torch.manual_seed(seed)
    H, N, D = keys.shape
    queries = []
    for _ in range(num_queries):
        target_n = torch.randint(0, N, (1,)).item()
        q = F.normalize(keys[:, target_n, :] + noise * torch.randn(H, D), dim=-1)
        queries.append(q)
    return torch.stack(queries)


def run_one_setting(diversity, gamma, beta, K, target_sparsity,
                    num_queries=20, H=8, N=512, D=64, num_groups=15, verbose=False):
    """Chạy 1 setting và trả về metrics cho 3 phương pháp."""
    keys, values, _ = synthetic_kv(
        H=H, N=N, D=D, num_groups=num_groups, value_diversity=diversity, seed=42
    )

    # Calib queries
    calib_q = synthetic_queries(keys, num_queries=20, seed=99)
    test_q = synthetic_queries(keys, num_queries=num_queries, seed=2024)

    # === Key-only baseline ===
    kc_ko, _, lbl_ko, _ = value_aware_kmeans(
        keys, values, K, alpha=1.0, beta=0.0, num_iters=15
    )
    sizes_ko = torch.zeros(H, K)
    sizes_ko.scatter_add_(1, lbl_ko, torch.ones_like(lbl_ko, dtype=torch.float))
    nvar_zero = torch.zeros(H, K)

    # === Value-aware ===
    kc_va, _, lbl_va, vvar = value_aware_kmeans(
        keys, values, K, alpha=1.0, beta=beta, num_iters=15
    )
    sizes_va = torch.zeros(H, K)
    sizes_va.scatter_add_(1, lbl_va, torch.ones_like(lbl_va, dtype=torch.float))
    nvar_va = normalize_value_variance({0: vvar.unsqueeze(0)})[0].squeeze(0)

    # === QUEST ===
    chunk_size = max(1, N // K)
    rep_k_q, lbl_q, sizes_q = quest_style_clustering(keys, values, chunk_size=chunk_size)
    K_quest = rep_k_q.shape[1]
    nvar_q = torch.zeros(H, K_quest)

    # Calibrate thresholds
    T_ko = calibrate_threshold(
        calib_q, kc_ko, sizes_ko, nvar_zero, lbl_ko,
        target_sparsity=target_sparsity, gamma=0.0, num_threshold_search=80,
    )
    T_va = calibrate_threshold(
        calib_q, kc_va, sizes_va, nvar_va, lbl_va,
        target_sparsity=target_sparsity, gamma=gamma, num_threshold_search=80,
    )
    T_q = calibrate_threshold(
        calib_q, rep_k_q, sizes_q, nvar_q, lbl_q,
        target_sparsity=target_sparsity, gamma=0.0, num_threshold_search=80,
    )

    # Run on test queries
    metrics = {m: {"cos": [], "mse": [], "budget": []} for m in ["key_only", "value_aware", "quest"]}

    for q in test_q:
        ref = baseline_full_attention(q, keys, values)

        ko_out, ko_info = key_only_attention_forward(
            q, keys, values, kc_ko, sizes_ko, lbl_ko, threshold=T_ko,
        )
        va_out, va_info = squeezed_attention_forward(
            q, keys, values, kc_va, sizes_va, nvar_va, lbl_va,
            threshold=T_va, gamma=gamma,
        )
        quest_out, quest_info = quest_attention_forward(
            q, keys, values, rep_k_q, sizes_q, lbl_q, threshold=T_q,
        )

        for name, out, info in [
            ("key_only", ko_out, ko_info),
            ("value_aware", va_out, va_info),
            ("quest", quest_out, quest_info),
        ]:
            metrics[name]["cos"].append(
                F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
            )
            metrics[name]["mse"].append(F.mse_loss(out, ref).item())
            metrics[name]["budget"].append(info["kv_budget"])

    return {
        m: {
            "cos": sum(v["cos"]) / len(v["cos"]),
            "mse": sum(v["mse"]) / len(v["mse"]),
            "budget": sum(v["budget"]) / len(v["budget"]),
        }
        for m, v in metrics.items()
    }


def print_table(results, label):
    print(f"\n{label}")
    print(f"{'Method':>14s} | {'CosSim':>8s} | {'MSE':>10s} | {'Budget':>8s}")
    print("-" * 55)
    for name, m in results.items():
        print(f"{name:>14s} | {m['cos']:>8.4f} | {m['mse']:>10.6f} | {m['budget']*100:>7.2f}%")


def experiment_diversity_sweep(args):
    """Sweep over value_diversity để xem khi nào VA có lợi nhất."""
    print("=" * 70)
    print("EXPERIMENT 1: Sweep value diversity")
    print("=" * 70)
    print(f"Config: K={args.K}, gamma={args.gamma}, beta={args.beta}, sparsity={args.sparsity}")

    print(f"\n{'Diversity':>10s} | {'Method':>14s} | {'CosSim':>8s} | {'MSE':>10s} | {'Budget':>8s}")
    print("-" * 70)
    for diversity in [0.1, 0.3, 0.5, 1.0, 2.0]:
        r = run_one_setting(
            diversity=diversity, gamma=args.gamma, beta=args.beta,
            K=args.K, target_sparsity=args.sparsity,
            num_queries=args.num_queries, H=args.H, N=args.N, D=args.D,
        )
        for name, m in r.items():
            print(f"{diversity:>10.2f} | {name:>14s} | "
                  f"{m['cos']:>8.4f} | {m['mse']:>10.6f} | {m['budget']*100:>7.2f}%")
        # Delta (cos sim in pp = percentage points = *100)
        d_cos_pp = (r["value_aware"]["cos"] - r["key_only"]["cos"]) * 100
        d_mse = (r["key_only"]["mse"] - r["value_aware"]["mse"]) / max(r["key_only"]["mse"], 1e-9) * 100
        print(f"{'':>10s} | {'Δ VA-vs-KO':>14s} | {d_cos_pp:>+7.3f}pp | {d_mse:>+9.2f}% MSE↓")
        print()


def experiment_gamma_sweep(args):
    """Sweep gamma để tìm optimal."""
    print("=" * 70)
    print("EXPERIMENT 2: Sweep gamma (with fixed beta=0.5, diversity=1.0)")
    print("=" * 70)

    print(f"\n{'Gamma':>8s} | {'Method':>14s} | {'CosSim':>8s} | {'MSE':>10s} | {'Budget':>8s}")
    print("-" * 70)
    for gamma in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        r = run_one_setting(
            diversity=1.0, gamma=gamma, beta=0.5,
            K=args.K, target_sparsity=args.sparsity,
            num_queries=args.num_queries,
        )
        # Print only va & ko
        for name in ["key_only", "value_aware"]:
            m = r[name]
            print(f"{gamma:>8.2f} | {name:>14s} | "
                  f"{m['cos']:>8.4f} | {m['mse']:>10.6f} | {m['budget']*100:>7.2f}%")
        print()


def experiment_beta_sweep(args):
    """Sweep beta để tìm trade-off."""
    print("=" * 70)
    print("EXPERIMENT 3: Sweep beta (gamma=0.3, diversity=1.0)")
    print("=" * 70)

    print(f"\n{'Beta':>8s} | {'Method':>14s} | {'CosSim':>8s} | {'MSE':>10s} | {'Budget':>8s}")
    print("-" * 70)
    for beta in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
        r = run_one_setting(
            diversity=1.0, gamma=0.3, beta=beta,
            K=args.K, target_sparsity=args.sparsity,
            num_queries=args.num_queries,
        )
        m = r["value_aware"]
        print(f"{beta:>8.2f} | {'value_aware':>14s} | "
              f"{m['cos']:>8.4f} | {m['mse']:>10.6f} | {m['budget']*100:>7.2f}%")


def experiment_sparsity_sweep(args):
    """Sweep target sparsity (giống Table 2 của paper - 70%, 80%, 90%)."""
    print("=" * 70)
    print("EXPERIMENT 4: Sparsity sweep (như Table 2 trong paper)")
    print("=" * 70)
    print(f"diversity=1.0, gamma={args.gamma}, beta={args.beta}, K={args.K}")

    print(f"\n{'Sparsity':>9s} | {'Method':>14s} | {'CosSim':>8s} | {'MSE':>10s} | {'Budget':>8s}")
    print("-" * 70)
    for s in [0.7, 0.8, 0.9, 0.95]:
        r = run_one_setting(
            diversity=1.0, gamma=args.gamma, beta=args.beta,
            K=args.K, target_sparsity=s,
            num_queries=args.num_queries,
        )
        for name, m in r.items():
            print(f"{s:>9.2f} | {name:>14s} | "
                  f"{m['cos']:>8.4f} | {m['mse']:>10.6f} | {m['budget']*100:>7.2f}%")
        print()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=32, help="Số clusters")
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--N", type=int, default=512)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--num_queries", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--sparsity", type=float, default=0.85)
    p.add_argument("--experiment", type=str, default="all",
                   choices=["all", "diversity", "gamma", "beta", "sparsity"])
    return p.parse_args()


def main():
    args = parse_args()
    if args.experiment in ("all", "diversity"):
        experiment_diversity_sweep(args)
    if args.experiment in ("all", "gamma"):
        experiment_gamma_sweep(args)
    if args.experiment in ("all", "beta"):
        experiment_beta_sweep(args)
    if args.experiment in ("all", "sparsity"):
        experiment_sparsity_sweep(args)
    print("\n" + "=" * 70)
    print("✓ Synthetic benchmarks complete.")
    print("Diễn giải: Δ CosSim dương = value-aware tốt hơn key-only.")
    print("Δ Budget gần 0 = so sánh công bằng cùng KV cost.")
    print("=" * 70)


if __name__ == "__main__":
    main()
