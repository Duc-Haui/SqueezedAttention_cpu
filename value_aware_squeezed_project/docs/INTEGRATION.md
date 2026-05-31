# Tích hợp Value-Aware Retrieval vào repo SqueezedAttention gốc

Project này được thiết kế **drop-in compatible** với repo gốc
[SqueezeAILab/SqueezedAttention](https://github.com/SqueezeAILab/SqueezedAttention).

Có 3 cách tích hợp tùy mức độ sâu mong muốn:

| Mức | Mô tả | File cần sửa | Khó |
|-----|-------|-------------|-----|
| 1 | Chỉ thay clustering offline (giữ nguyên online) | `offline_clustering.py` | ★ |
| 2 | Thay cả threshold calibration | `offline_clustering.py` + load thêm variance khi serve | ★★ |
| 3 | Patch attention layer cho variance-aware retrieve online | custom transformers `LlamaAttention.forward` | ★★★ |

Mức 1+2 đủ để **paper-style evaluation** (offline accuracy proxy). Mức 3 cần để **measure real speedup**.

---

## Mức 1+2: Offline integration (recommended)

### Bước 1. Setup paths

Giả sử bạn có:
```
parent_dir/
├── SqueezedAttention/         <- repo gốc
└── value_aware_squeezed_project/   <- project này
```

Thêm `value_aware_squeezed_project` vào `PYTHONPATH`:
```bash
cd parent_dir/SqueezedAttention
export PYTHONPATH="../value_aware_squeezed_project:$PYTHONPATH"
```

### Bước 2. Sửa `offline_clustering.py`

Mở file `SqueezedAttention/offline_clustering.py`, **thay đổi nhỏ ở 3 chỗ**:

#### (a) Thêm import value-aware (dòng ~20)

```python
# Thêm sau dòng `from squeezedattention.clustering import run_clustering, run_global_threshold`
from value_aware.clustering import run_value_aware_clustering, normalize_value_variance
from value_aware.threshold import run_value_aware_global_threshold
```

#### (b) Thêm CLI args (dòng ~30)

```python
parser.add_argument('--value_aware', action='store_true',
                    help='Enable value-aware retrieval')
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--gamma', type=float, default=0.3)
```

#### (c) Thay logic clustering trong loop (dòng ~155, không hierarchical case)

Thay đoạn:
```python
centroids_tensor_dict, centroids_labels_dict = run_clustering(
    all_keys_layers, num_centroids,
    observation_window=args.observation_window, device=DEV,
)
global_threshold_dict = run_global_threshold(
    all_keys_layers, all_queries_layers, ...
)
```

Bằng:
```python
if args.value_aware:
    kc_dict, vc_dict, lbl_dict, vvar_dict, nvar_dict = run_value_aware_clustering(
        all_keys_layers, all_values_layers,
        num_clusters=num_centroids,
        observation_window=args.observation_window,
        alpha=args.alpha, beta=args.beta,
        device=DEV,
    )
    centroids_tensor_dict, centroids_labels_dict = kc_dict, lbl_dict
    
    # Save thêm variance để dùng online
    torch.save(nvar_dict, f'{args.output_path}/normalized_variance_{dataidx}_{num_centroids}.pt')
    torch.save(vc_dict, f'{args.output_path}/value_centroids_{dataidx}_{num_centroids}.pt')
    
    # Threshold value-aware
    global_threshold_dict = run_value_aware_global_threshold(
        keys_layers=all_keys_layers,
        queries_layers=all_queries_layers,
        key_centroids_dict=kc_dict,
        labels_dict=lbl_dict,
        normalized_variance_dict=nvar_dict,
        num_clusters=num_centroids,
        observation_window=args.observation_window,
        gamma=args.gamma,
        device=DEV,
    )
else:
    # Code gốc, không thay
    centroids_tensor_dict, centroids_labels_dict = run_clustering(
        all_keys_layers, num_centroids,
        observation_window=args.observation_window, device=DEV,
    )
    global_threshold_dict = run_global_threshold(
        all_keys_layers, all_queries_layers,
        centroids_tensor_dict, centroids_labels_dict, num_centroids,
        observation_window=args.observation_window, device=DEV,
    )
```

### Bước 3. Chạy

```bash
# Baseline gốc (như cũ)
bash run_offline_clustering.sh

# Value-aware version
python offline_clustering.py LLaMA-2-7B-32K \
    --dataset 2wikimqa \
    --output_path /tmp/clusters_va/2wikimqa/ \
    --percent_clusters 5 \
    --observation_window 100 \
    --value_aware --alpha 1.0 --beta 0.5 --gamma 0.3 \
    --device 0
```

**Hoặc** dùng patched script đã có sẵn:
```bash
python ../value_aware_squeezed_project/patches/offline_clustering_value_aware.py \
    LLaMA-2-7B-32K \
    --dataset 2wikimqa \
    --output_path /tmp/clusters_va/2wikimqa/ \
    --alpha 1.0 --beta 0.5 --gamma 0.3
```

---

## Mức 3: Online attention layer patch (advanced)

Để có speedup thực tế, cần sửa attention forward trong custom transformers
(file `transformers/src/transformers/models/llama/modeling_llama.py` hoặc tương đương).

### Tìm chỗ tính cluster scores

Trong `LlamaAttention.forward` (hoặc layer attention tương đương), tìm chỗ tính:
```python
# Original: S_i = exp(q · C_i^T) / sum_j (N_j · exp(q · C_j^T))
attn_scores_centroids = torch.matmul(q, centroids.transpose(-2, -1))
# ... softmax + threshold ...
```

### Thay bằng value-aware

```python
# Load normalized_variance đã precompute (từ offline_clustering.py value_aware mode)
# nvar shape: (B, H, K)
attn_scores_centroids = torch.matmul(q, centroids.transpose(-2, -1))

# Stable softmax-style với cluster sizes
exp_logits = torch.exp(attn_scores_centroids - attn_scores_centroids.max(-1, keepdim=True).values)
weighted = num_keys_per_cluster.unsqueeze(-2) * exp_logits
S = exp_logits / weighted.sum(-1, keepdim=True)

# === Value-aware boost ===
gamma = self.config.value_aware_gamma  # 0.3 mặc định
S_adjusted = S * (1.0 + gamma * normalized_variance.unsqueeze(-2))

# Threshold filter
cluster_mask = S_adjusted > self.threshold  # tự load từ tdict[args.percentile]
```

### Files cần sửa trong custom transformers

Repo gốc fork transformers từ HF. Các điểm patch chính:
- `transformers/src/transformers/models/llama/modeling_llama.py`:
  - Class `LlamaAttention`: sửa `forward()` để load `normalized_variance`, thêm boost
  - Class `LlamaConfig`: thêm các attribute `value_aware_gamma`, `use_value_aware`

Ví dụ flow đề xuất (pseudocode):

```python
class LlamaAttention(nn.Module):
    def __init__(self, config, ...):
        super().__init__()
        self.use_value_aware = getattr(config, "use_value_aware", False)
        self.gamma = getattr(config, "value_aware_gamma", 0.3)
        # ... rest of init ...

    def forward(self, hidden_states, ..., past_key_value=None, ...):
        # ... compute q, k, v, do RoPE ...
        
        if self.use_value_aware and past_key_value is not None:
            # Load precomputed cluster data
            cluster_data = past_key_value.get_cluster_data(self.layer_idx)
            centroids = cluster_data['centroids']
            cluster_sizes = cluster_data['sizes']
            nvar = cluster_data['normalized_variance']
            labels = cluster_data['labels']
            threshold = cluster_data['threshold']
            
            # Centroid lookup with value-aware boost
            S = self._compute_base_scores(q, centroids, cluster_sizes)
            S_adj = S * (1.0 + self.gamma * nvar.unsqueeze(-2))
            cluster_mask = S_adj > threshold
            
            # Map back to per-token mask
            key_mask = torch.gather(cluster_mask, -1, labels)
            
            # Sparse attention with masked keys
            attn_logits = ... # standard q@k.T
            attn_logits = attn_logits.masked_fill(~key_mask, float('-inf'))
            # ... softmax + matmul with v ...
```

Để giữ tương thích Triton kernel, cần truyền thêm 2 tensor:
- `normalized_variance`: shape `(B, H, K)` - per-cluster boost factor
- `gamma`: scalar - hệ số boost

Xem `squeezedattention/kernels.py` của repo gốc để hiểu cách kernel nhận inputs.
Việc sửa kernel phức tạp hơn — nếu chỉ muốn measure accuracy, **bỏ qua mức 3** và dùng PyTorch implementation reference của project này.

---

## Verification sau khi tích hợp

Sau khi tích hợp, chạy lại offline clustering với cả 2 mode để verify:

```bash
# Baseline
python offline_clustering.py LLaMA-2-7B-32K --dataset 2wikimqa \
    --output_path /tmp/clusters_baseline/

# Value-aware với β=0, γ=0 -> nên cho kết quả tương đương baseline
python offline_clustering.py LLaMA-2-7B-32K --dataset 2wikimqa \
    --value_aware --alpha 1.0 --beta 0.0 --gamma 0.0 \
    --output_path /tmp/clusters_va_zero/

# Value-aware full
python offline_clustering.py LLaMA-2-7B-32K --dataset 2wikimqa \
    --value_aware --alpha 1.0 --beta 0.5 --gamma 0.3 \
    --output_path /tmp/clusters_va_full/
```

Centroid của 2 setting đầu nên gần như giống nhau (chỉ khác do init random của K-means).

---

## Troubleshooting

| Lỗi | Nguyên nhân | Fix |
|-----|------------|-----|
| `ModuleNotFoundError: value_aware` | PYTHONPATH chưa có project | `export PYTHONPATH=path/to/value_aware_squeezed_project:$PYTHONPATH` |
| `ImportError: cuml` ở repo gốc | Repo gốc dùng cuML cho K-means | Project này dùng PyTorch K-means - không cần cuML |
| Output shape mismatch | Code gốc expect `(1, H, K, D)` | `run_value_aware_clustering` trả về đúng format này (có batch dim) |
| KV budget không đạt target | gamma làm thay đổi distribution | Calibrate threshold lại với gamma đúng (`run_value_aware_global_threshold` đã làm sẵn) |

---

## Đối với hierarchical lookup

Repo gốc có hierarchical lookup (cluster L1 trên L2). Để tích hợp value-aware:
1. Chạy `run_value_aware_clustering` cho L2 (level fine)
2. Chạy lại `run_value_aware_clustering` trên `value_centroids_L2` cho L1 (level coarse)
3. Khi retrieve L1 -> L2, dùng cùng công thức boost với `nvar_L1`

Code support sẵn — chỉ cần gọi `run_value_aware_clustering` 2 lần.
