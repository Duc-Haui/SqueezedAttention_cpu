#!/bin/bash

# # Đồng bộ môi trường
# export PYTHONPATH="/home/ubuntu/Desktop/SqueezedAttention-main/SqueezedAttention-main:$PYTHONPATH"
# export PYTHONPATH="/home/ubuntu/Desktop/SqueezedAttention-main/value_aware_squeezed_project:$PYTHONPATH"

# cd /home/ubuntu/Desktop/SqueezedAttention-main/SqueezedAttention-main/LongBench

# PATH_TO_CLUSTERS="/home/ubuntu/Desktop/SqueezedAttention-main/SqueezedAttention-main/fixed-prompt-clusters_v2/"
# PERC_CLUSTERS="5"
# PERCENTILES=("0.70" "0.80" "0.90")
# DATASETS=("narrativeqa" "qasper" "multifieldqa_en")

# for DATASET in "${DATASETS[@]}"; do
#     for PERCENTILE in "${PERCENTILES[@]}"; do
#         echo "========================================================="
#         echo "🚀 ĐANG CHẠY SUY LUẬN: Task=${DATASET} | Cắt tỉa=${PERCENTILE}"
#         echo "========================================================="

#         /home/ubuntu/Desktop/venv_39/bin/python pred.py \
#             --model SmolLM2-135M \
#             --use_centroids \
#             --percentile $PERCENTILE \
#             --percent_clusters $PERC_CLUSTERS \
#             --path_to_clusters $PATH_TO_CLUSTERS \
#             --task $DATASET

#         echo " Chấm điểm cho: Task=${DATASET} | Cắt tỉa=${PERCENTILE}"

#         /home/ubuntu/Desktop/venv_39/bin/python eval.py \
#             --model SmolLM2-135M \
#             --use_centroids \
#             --percentile $PERCENTILE \
#             --percent_clusters $PERC_CLUSTERS
#     done
# done
# echo "✅ HOÀN THÀNH!"

# Đồng bộ môi trường
export PYTHONPATH="/home/ubuntu/Desktop/SqueezedAttention-main/SqueezedAttention-main:$PYTHONPATH"
export PYTHONPATH="/home/ubuntu/Desktop/SqueezedAttention-main/value_aware_squeezed_project:$PYTHONPATH"

REPO="/home/ubuntu/Desktop/SqueezedAttention-main/SqueezedAttention-main"
PYTHON="/home/ubuntu/Desktop/venv_39/bin/python"
PATH_TO_CLUSTERS="${REPO}/fixed-prompt-clusters_v2"
PERC_CLUSTERS="5"
PERCENTILES=("0.70" "0.80" "0.90")
DATASETS=("narrativeqa" "qasper" "multifieldqa_en")

# ─── BƯỚC 1: CLUSTERING ───────────────────────────────────────
echo "========================================================="
echo ">>> BƯỚC 1: Chạy Offline Clustering cho tất cả datasets"
echo "========================================================="

cd $REPO

for DATASET in "${DATASETS[@]}"; do
    mkdir -p "${PATH_TO_CLUSTERS}/${DATASET}"
    echo ">>> Clustering: ${DATASET}"
    $PYTHON offline_clustering.py SmolLM2-135M \
        --dataset $DATASET \
        --percent_clusters $PERC_CLUSTERS \
        --output_path "${PATH_TO_CLUSTERS}/${DATASET}"
    
    if [ $? -ne 0 ]; then
        echo "❌ Clustering thất bại cho ${DATASET}, dừng lại."
        exit 1
    fi
    echo "✅ Clustering xong: ${DATASET}"
done

# ─── BƯỚC 2: INFERENCE + EVAL ─────────────────────────────────
echo "========================================================="
echo ">>> BƯỚC 2: Chạy Inference và Đánh giá"
echo "========================================================="

cd ${REPO}/LongBench

for DATASET in "${DATASETS[@]}"; do
    for PERCENTILE in "${PERCENTILES[@]}"; do
        echo "========================================================="
        echo "🚀 SUY LUẬN: Task=${DATASET} | Cắt tỉa=${PERCENTILE}"
        echo "========================================================="

        $PYTHON pred.py \
            --model SmolLM2-135M \
            --use_centroids \
            --percentile $PERCENTILE \
            --percent_clusters $PERC_CLUSTERS \
            --path_to_clusters $PATH_TO_CLUSTERS \
            --task $DATASET

        echo "📊 Chấm điểm: Task=${DATASET} | Cắt tỉa=${PERCENTILE}"

        $PYTHON eval.py \
            --model SmolLM2-135M \
            --use_centroids \
            --percentile $PERCENTILE \
            --percent_clusters $PERC_CLUSTERS
    done
done

echo "✅ HOÀN THÀNH TOÀN BỘ!"
SCRIPT
