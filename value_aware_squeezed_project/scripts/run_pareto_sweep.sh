#!/bin/bash

# ==========================================
# Cấu hình siêu nhẹ cho CPU (SmolLM2-135M)
# ==========================================
MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
MAX_CONTEXT=512
OBS_WINDOW=32
NUM_QUERIES=2
GAMMA=0.3
BETA=0.5

# Danh sách các mức sparsity cần quét để vẽ Pareto
SPARSITIES=(0.7 0.8 0.85 0.9 0.95)

# Thư mục lưu kết quả
OUT_DIR="results/pareto_sweep"
mkdir -p $OUT_DIR

echo "Bắt đầu chạy Pareto Sweep với mô hình: $MODEL"
echo "--------------------------------------------------------"

for S in "${SPARSITIES[@]}"; do
    echo ">>> Đang chạy với Sparsity = $S <<<"
    
    python benchmarks/benchmark_accuracy.py \
        --model "$MODEL" \
        --max_context $MAX_CONTEXT \
        --obs_window $OBS_WINDOW \
        --num_queries $NUM_QUERIES \
        --sparsity $S \
        --gamma $GAMMA \
        --beta $BETA \
        --output "$OUT_DIR/accuracy_s${S}.json"
        
    echo "Hoàn thành Sparsity $S!"
    echo "--------------------------------------------------------"
done

echo "Đã chạy xong toàn bộ các mức nén!"

# (Tùy chọn) Chạy script vẽ biểu đồ nếu file plot_comparison.py của bạn hỗ trợ đọc từ thư mục
# python benchmarks/plot_comparison.py --input_dir $OUT_DIR --output figs/pareto_front.png
