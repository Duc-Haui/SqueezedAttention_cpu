!/bin/bash
# ============================================================================
# Run all benchmarks sequentially
# ============================================================================
# Cách dùng:
#   bash scripts/run_all_benchmarks.sh
#
# Yêu cầu: GPU 24GB hoặc CPU (chạy lâu hơn).
# Kết quả lưu vào results/, plots vào figs/.
# ============================================================================

set -e
cd "$(dirname "$0")/.."

mkdir -p results figs

MODEL=${MODEL:-"Qwen/Qwen2.5-1.5B-Instruct"}
GAMMA=${GAMMA:-0.3}
BETA=${BETA:-0.5}
SPARSITY=${SPARSITY:-0.85}

echo "=========================================="
echo "Value-Aware Squeezed Attention Benchmark Suite"
echo "Model:    $MODEL"
echo "Gamma:    $GAMMA"
echo "Beta:     $BETA"
echo "Sparsity: $SPARSITY"
echo "=========================================="

echo ""
echo "[1/5] Running unit tests..."
python tests/test_all.py

echo ""
echo "[2/5] Synthetic benchmark (no GPU needed)..."
python benchmarks/synthetic_benchmark.py --experiment all \
    --gamma $GAMMA --beta $BETA --sparsity $SPARSITY \
    | tee results/synthetic.log

echo ""
echo "[3/5] Ablation study..."
python benchmarks/benchmark_ablation.py \
    --num_seeds 3 --sparsity $SPARSITY \
    --output results/ablation.json | tee results/ablation.log

echo ""
echo "[4/5] Latency benchmark..."
python benchmarks/benchmark_latency.py \
    --context_length 4096 --num_heads 16 --head_dim 64 \
    --num_runs 30 --sparsity $SPARSITY \
    --output results/latency_benchmark.json | tee results/latency.log

echo ""
echo "[5/5] Accuracy benchmark on real model (cần GPU)..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    python benchmarks/benchmark_accuracy.py \
        --model $MODEL \
        --max_context 4096 --num_queries 10 \
        --gamma $GAMMA --beta $BETA --sparsity $SPARSITY \
        --output results/accuracy_benchmark.json | tee results/accuracy.log
else
    echo "  Không có GPU - bỏ qua bước này. Chạy thủ công sau khi có GPU:"
    echo "    python benchmarks/benchmark_accuracy.py --model $MODEL"
fi

echo ""
echo "[6/6] Vẽ biểu đồ..."
python benchmarks/plot_comparison.py \
    --accuracy_json results/accuracy_benchmark.json \
    --latency_json results/latency_benchmark.json \
    --ablation_json results/ablation.json \
    --output_dir figs/ || echo "  (Cần matplotlib: pip install matplotlib)"

echo ""
echo "=========================================="
echo "Done. Kết quả ở results/, biểu đồ ở figs/"
echo "=========================================="
