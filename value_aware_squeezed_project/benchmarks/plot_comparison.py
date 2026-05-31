"""
Plot Comparison - Vẽ biểu đồ so sánh các phương pháp.

Đọc JSON kết quả từ các benchmark và vẽ:
1. CosSim vs KV Budget (Pareto front)
2. Latency comparison bar chart
3. Ablation contributions

Cách chạy:
    python benchmarks/plot_comparison.py \\
        --accuracy_json results/accuracy_benchmark.json \\
        --latency_json results/latency_benchmark.json \\
        --ablation_json results/ablation.json \\
        --output_dir figs/
"""

import argparse
import json
import os
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib not installed. Install with: pip install matplotlib")
    raise


def plot_accuracy_comparison(accuracy_data, save_path):
    """Bar chart: cos sim của mỗi method."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    methods = list(accuracy_data["results"].keys())
    cos_sims = [accuracy_data["results"][m]["avg_cos_sim"] for m in methods]
    budgets = [accuracy_data["results"][m]["avg_kv_budget"] * 100 for m in methods]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    method_names = {"key_only": "Key-only\n(Squeezed)", "value_aware": "Value-Aware\n(Ours)", "quest": "QUEST-style"}

    # Plot 1: cos sim
    bars = axes[0].bar(
        [method_names.get(m, m) for m in methods], cos_sims,
        color=colors[:len(methods)],
    )
    axes[0].set_ylabel("Cosine Similarity to Full Attention")
    axes[0].set_title("Approximation Quality")
    axes[0].set_ylim(min(cos_sims) - 0.02, 1.0)
    for bar, val in zip(bars, cos_sims):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                     f"{val:.4f}", ha='center', fontsize=9)

    # Plot 2: KV budget
    bars = axes[1].bar(
        [method_names.get(m, m) for m in methods], budgets,
        color=colors[:len(methods)],
    )
    axes[1].set_ylabel("KV Budget (% keys loaded)")
    axes[1].set_title("Memory Cost")
    for bar, val in zip(bars, budgets):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                     f"{val:.1f}%", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_latency_comparison(latency_data, save_path):
    """Bar chart latency for online operations."""
    fig, ax = plt.subplots(figsize=(8, 4))

    online = latency_data.get("online", {})
    methods = ["full", "key_only", "value_aware", "quest"]
    latencies = [online.get(m, 0) for m in methods]

    colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c']
    labels = ["Full\nAttention", "Key-only\n(Squeezed)", "Value-Aware\n(Ours)", "QUEST"]

    bars = ax.bar(labels, latencies, color=colors)
    ax.set_ylabel("Latency per query (ms)")
    ax.set_title("Online Latency Comparison")

    # Add speedup annotations
    if latencies[0] > 0:
        for bar, val, m in zip(bars, latencies, methods):
            if val > 0 and m != "full":
                speedup = latencies[0] / val
                ax.text(bar.get_x() + bar.get_width() / 2, val + max(latencies) * 0.02,
                        f"{val:.2f}ms\n({speedup:.2f}x)", ha='center', fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, val + max(latencies) * 0.02,
                        f"{val:.2f}ms", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_ablation(ablation_data, save_path):
    """Bar chart of cos_sim contributions of each ablation."""
    fig, ax = plt.subplots(figsize=(11, 5))

    results = ablation_data["ablation_results"]
    labels = [results[k]["name"] for k in results]
    cos_sims = [results[k]["cos"] for k in results]

    # Compute delta vs first (baseline)
    base = cos_sims[0]
    deltas = [(c - base) * 100 for c in cos_sims]

    colors = ['#7f7f7f'] + ['#1f77b4' if d > 0 else '#d62728' for d in deltas[1:]]

    bars = ax.bar(range(len(labels)), cos_sims, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Ablation: contribution of each component")
    ax.axhline(y=base, color='gray', linestyle='--', alpha=0.5, label=f"Baseline = {base:.4f}")

    for i, (bar, val, d) in enumerate(zip(bars, cos_sims, deltas)):
        annotation = f"{val:.4f}"
        if i > 0:
            annotation += f"\n({d:+.2f}pp)"
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                annotation, ha='center', fontsize=8)

    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_pareto(accuracy_jsons, save_path):
    """Pareto plot: cos_sim vs KV budget across multiple sparsity settings."""
    fig, ax = plt.subplots(figsize=(8, 6))

    method_data = {"key_only": [], "value_aware": [], "quest": []}
    for j in accuracy_jsons:
        for method in method_data:
            if method in j["results"]:
                r = j["results"][method]
                method_data[method].append((r["avg_kv_budget"] * 100, r["avg_cos_sim"]))

    markers = {"key_only": "o", "value_aware": "s", "quest": "^"}
    colors = {"key_only": "#1f77b4", "value_aware": "#ff7f0e", "quest": "#2ca02c"}
    labels = {"key_only": "Key-only Squeezed", "value_aware": "Value-Aware (Ours)", "quest": "QUEST-style"}

    for method, points in method_data.items():
        if not points:
            continue
        points.sort()  # by budget
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker=markers[method], color=colors[method],
                label=labels[method], linewidth=2, markersize=8)

    ax.set_xlabel("KV Budget (% keys loaded)")
    ax.set_ylabel("Cosine Similarity to Full Attention")
    ax.set_title("Pareto Front: Quality vs Memory Cost")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--accuracy_json", default="results/accuracy_benchmark.json")
    p.add_argument("--latency_json", default="results/latency_benchmark.json")
    p.add_argument("--ablation_json", default="results/ablation.json")
    p.add_argument("--pareto_jsons", nargs="+", default=[],
                   help="Multiple accuracy JSONs với sparsity khác nhau cho Pareto plot")
    p.add_argument("--output_dir", default="figs/")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(args.accuracy_json):
        with open(args.accuracy_json) as f:
            data = json.load(f)
        plot_accuracy_comparison(data, os.path.join(args.output_dir, "accuracy.png"))

    if os.path.exists(args.latency_json):
        with open(args.latency_json) as f:
            data = json.load(f)
        plot_latency_comparison(data, os.path.join(args.output_dir, "latency.png"))

    if os.path.exists(args.ablation_json):
        with open(args.ablation_json) as f:
            data = json.load(f)
        plot_ablation(data, os.path.join(args.output_dir, "ablation.png"))

    if args.pareto_jsons:
        all_data = []
        for path in args.pareto_jsons:
            if os.path.exists(path):
                with open(path) as f:
                    all_data.append(json.load(f))
        if all_data:
            plot_pareto(all_data, os.path.join(args.output_dir, "pareto.png"))


if __name__ == "__main__":
    main()
