##!/bin/bash

# # export CUDA_VISIBLE_DEVICES=2

# # arguments:
# PATH_TO_CLUSTERS="/home/ubuntu/Desktop/SqueezedAttention-main/SqueezedAttention-main/fixed-prompt-clusters/"
# PERCENTILE="0.70" # percentile for pruning
# DATASET="narrativeqa"
# PERC_CLUSTERS="5"

# # run evaluation
# python pred.py --model SmolLM2-135M --use_centroids --percentile $PERCENTILE --percent_clusters $PERC_CLUSTERS \
#                --path_to_clusters $PATH_TO_CLUSTERS --task $DATASET

# # check accuracy
# python eval.py --model SmolLM2-135M --use_centroids --percentile $PERCENTILE --percent_clusters $PERC_CLUSTERS
#!/bin/bash

# 1. Ép buộc Python phải ưu tiên đọc code trong thư mục dự án của bạn trước
export PYTHONPATH="/home/ubuntu/Desktop/SqueezedAttention-main/SqueezedAttention-main:$PYTHONPATH"

# arguments:
PATH_TO_CLUSTERS="/home/ubuntu/Desktop/SqueezedAttention-main/SqueezedAttention-main/fixed-prompt-clusters/"
PERCENTILE="0.70" 
DATASET="narrativeqa"
PERC_CLUSTERS="5"

# 2. Di chuyển vào thư mục LongBench trước khi gọi lệnh để pred.py tìm đúng config dữ liệu
cd /home/ubuntu/Desktop/SqueezedAttention-main/SqueezedAttention-main/LongBench

# 3. Chạy dự đoán suy luận cắt tỉa (Pruning) 70%
/home/ubuntu/Desktop/venv_39/bin/python pred.py --model SmolLM2-135M --use_centroids --percentile $PERCENTILE --percent_clusters $PERC_CLUSTERS \
               --path_to_clusters $PATH_TO_CLUSTERS --task $DATASET

# 4. Chạy chấm điểm kết quả thu được
/home/ubuntu/Desktop/venv_39/bin/python eval.py --model SmolLM2-135M --use_centroids --percentile $PERCENTILE --percent_clusters $PERC_CLUSTERS