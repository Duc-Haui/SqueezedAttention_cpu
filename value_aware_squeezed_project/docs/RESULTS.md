# Results

Template để fill in kết quả sau khi chạy benchmarks. Format giống Tables trong paper Squeezed Attention.

## Setup

- Model: <fill in, e.g. Qwen2.5-1.5B-Instruct>
- Context length: <fill in>
- GPU: <fill in>
- Date: <fill in>

---

## Table 1: Approximation Quality (như Table 2 paper)

`benchmarks/benchmark_accuracy.py` output. Compare cosine similarity với full attention output.

| Method | KV Budget | CosSim ↑ | MSE ↓ |
|--------|-----------|----------|-------|
| Full attention | 100% | 1.0000 | 0.0000 |
| Key-only Squeezed | <fill> | <fill> | <fill> |
| **Value-Aware (Ours)** | <fill> | <fill> | <fill> |
| QUEST-style | <fill> | <fill> | <fill> |

**Δ Value-Aware vs Key-only**: <fill> pp CosSim improvement at similar budget.

---

## Table 2: Sparsity Sweep (Pareto front)

| Sparsity | Method | CosSim ↑ |
|----------|--------|----------|
| 70% | Key-only | <fill> |
| 70% | Value-Aware | <fill> |
| 80% | Key-only | <fill> |
| 80% | Value-Aware | <fill> |
| 85% | Key-only | <fill> |
| 85% | Value-Aware | <fill> |
| 90% | Key-only | <fill> |
| 90% | Value-Aware | <fill> |

Xem `figs/pareto.png` để vẽ Pareto front.

---

## Table 3: Ablation Study (như Table 6 paper)

`benchmarks/benchmark_ablation.py` output.

| Configuration (β, γ) | CosSim ↑ | Δ vs baseline |
|---------------------|----------|---------------|
| Baseline (0, 0) | <fill> | — |
| Joint K-V only (0.5, 0) | <fill> | <fill> pp |
| Variance boost only (0, 0.3) | <fill> | <fill> pp |
| **Full method (0.5, 0.3)** | <fill> | <fill> pp |
| Aggressive (0.5, 0.5) | <fill> | <fill> pp |
| V-dominant (1.0, 0.3) | <fill> | <fill> pp |

---

## Table 4: Latency Comparison (như Figure 4 paper)

`benchmarks/benchmark_latency.py` output.

Context length = <fill>, num_heads = <fill>, head_dim = <fill>:

| Component | Full | Key-only | Value-Aware | QUEST |
|-----------|------|----------|-------------|-------|
| Centroid lookup (ms) | — | <fill> | <fill> | <fill> |
| Sparse attention (ms) | <fill> | <fill> | <fill> | <fill> |
| Total per-query (ms) | <fill> | <fill> | <fill> | <fill> |
| Speedup vs full | 1.00× | <fill>× | <fill>× | <fill>× |
| Offline clustering (s) | — | <fill> | <fill> | <fill> |

---

## Table 5: LongBench Tasks (nếu chạy được)

`benchmarks/benchmark_longbench.py` output.

| Task | Metric | Full | Key-only | **Value-Aware** | QUEST |
|------|--------|------|----------|-----------------|-------|
| NarrativeQA | F1 | <fill> | <fill> | <fill> | <fill> |
| Qasper | F1 | <fill> | <fill> | <fill> | <fill> |
| HotpotQA | F1 | <fill> | <fill> | <fill> | <fill> |
| 2WikiMQA | F1 | <fill> | <fill> | <fill> | <fill> |
| TREC | Accuracy | <fill> | <fill> | <fill> | <fill> |
| TriviaQA | F1 | <fill> | <fill> | <fill> | <fill> |
| **Average** | | <fill> | <fill> | <fill> | <fill> |

---

## Diversity Sweep (synthetic, Table 6 supplementary)

| Value Diversity | Key-only CosSim | Value-Aware CosSim | Δ pp |
|-----------------|-----------------|--------------------|------|
| 0.1 | <fill> | <fill> | <fill> |
| 0.5 | <fill> | <fill> | <fill> |
| 1.0 | <fill> | <fill> | <fill> |
| 2.0 | <fill> | <fill> | <fill> |

---

## Hyperparameter Recommendations

Dựa trên các sweep:

- **β** (joint clustering weight): <fill> sweet spot
- **γ** (variance boost): <fill> sweet spot  
- **K** (số clusters): <fill>% of context length
- **observation_window**: <fill> tokens

---

## Files Generated

- `results/synthetic.log` - Synthetic benchmark output
- `results/ablation.json` - Ablation results (JSON)
- `results/accuracy_benchmark.json` - Accuracy benchmark (JSON)
- `results/latency_benchmark.json` - Latency benchmark (JSON)
- `results/longbench_eval.json` - LongBench results (JSON, optional)
- `figs/accuracy.png` - Accuracy bar chart
- `figs/latency.png` - Latency bar chart
- `figs/ablation.png` - Ablation bar chart
- `figs/pareto.png` - Pareto front
