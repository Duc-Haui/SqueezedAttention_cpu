# Methodology: Value-Aware Retrieval

## Vấn đề với Squeezed Attention gốc

Squeezed Attention (Hooper et al., 2024) cluster keys $K \in \mathbb{R}^{N \times d}$ thành $K$ clusters bằng K-means trên không gian key, sau đó với mỗi query $q$:

$$S_i = \frac{\exp(q \cdot C_i^\top)}{\sum_j N_j \cdot \exp(q \cdot C_j^\top)}$$

trong đó $C_i$ là centroid của cluster $i$, $N_j$ là số keys trong cluster $j$. Cluster có $S_i > T$ được giữ.

**Vấn đề**: phương pháp giả định **values trong cùng key-cluster có vai trò tương tự**. Điều này không luôn đúng:

- Hai tokens có keys gần nhau (về mặt cosine) có thể có values rất khác nhau, ví dụ:
  - Token A: key gần "geography" topic, value chứa thông tin về "Paris"
  - Token B: key cũng gần "geography", nhưng value chứa thông tin về "Tokyo"
- Nếu cluster gộp cả A và B vào nhau và cluster này được prune, chúng ta mất cả 2 thông tin **dù chúng vốn nên được retrieve khác nhau cho query khác nhau**.

## Cải tiến: Value-Aware Retrieval

Cải tiến gồm 3 thành phần:

### 1. Joint K-V Clustering

Thay vì K-means trên $K_\text{norm}$, ta cluster trên không gian kết hợp:

$$z_n = \text{normalize}\left(\begin{bmatrix} \alpha \cdot k_n^\text{norm} \\ \beta \cdot v_n^\text{norm} \end{bmatrix}\right)$$

với $\alpha, \beta \geq 0$. Việc normalize riêng $k$ và $v$ trước rồi mới scale đảm bảo $\alpha, \beta$ đóng vai trò trọng số đúng nghĩa, không bị ảnh hưởng bởi norm trung bình của $K$ và $V$ (vốn có thể rất khác nhau theo head/layer).

**Trade-off**:
- $\beta = 0$: tương đương Squeezed Attention gốc
- $\beta$ lớn: clustering hoàn toàn theo $V$ → có thể phá retrieval quality vì matching $q \cdot C^\top$ không còn ý nghĩa
- Sweet spot: $\beta \in [0.3, 0.7]$. Mặc định $0.5$.

### 2. Per-Cluster Value Variance

Sau khi cluster, ta tính:

$$\sigma^2_{v,i} = \frac{1}{|C_i|} \sum_{n \in C_i} \|v_n - \mu_{v,i}\|_2^2$$

với $\mu_{v,i}$ là mean của values trong cluster $i$. Đây là **trace của covariance** values trong cluster, đo mức độ "đa dạng" của values.

Nếu $\sigma^2_{v,i}$ lớn → values trong cluster $i$ rất khác nhau → đại diện bằng 1 centroid value sẽ làm mất nhiều thông tin → cần **giữ lại cluster này một cách thận trọng hơn**.

Variance được normalize per (layer, head) bằng min-max scaling:

$$\tilde{\sigma}_{v,i} = \frac{\sigma^2_{v,i} - \min_j \sigma^2_{v,j}}{\max_j \sigma^2_{v,j} - \min_j \sigma^2_{v,j}}$$

→ $\tilde{\sigma}_{v,i} \in [0, 1]$.

### 3. Variance-Adjusted Importance Score

Khi retrieve, thay $S_i$ bằng:

$$\tilde{S}_i = S_i \cdot (1 + \gamma \cdot \tilde{\sigma}_{v,i})$$

với $\gamma$ là hyperparameter điều chỉnh cường độ boost.

Cluster mask: $\tilde{S}_i > T$.

**Trade-off**:
- $\gamma = 0$: bỏ qua variance, hành vi gốc
- $\gamma$ lớn: ưu tiên các cluster có values đa dạng → KV budget có thể tăng
- Mặc định: $\gamma = 0.3$

---

## Tại sao công thức nhân chứ không cộng?

Có 2 lựa chọn boost:
- **Cộng**: $\tilde{S}_i = S_i + \gamma \cdot \tilde{\sigma}_{v,i}$
- **Nhân**: $\tilde{S}_i = S_i \cdot (1 + \gamma \cdot \tilde{\sigma}_{v,i})$

Lý do chọn **nhân**:

1. **Bảo toàn ranking khi $\sigma^2$ giống nhau**. Nếu mọi cluster có cùng variance, ranking $\tilde{S}_i$ giống ranking $S_i$.

2. **Scale-invariance**. $S_i$ có thể có range rất khác nhau giữa heads/queries. Cộng làm "nhiễu" variance át $S_i$ ở những head có $S$ nhỏ. Nhân giữ ranking ngữ nghĩa.

3. **Bottom-up monotonic**. Cluster có $S$ rất nhỏ vẫn không bị "dối" nhiều bởi variance — boost tối đa là $1+\gamma$ lần.

---

## Tại sao normalize variance per (layer, head)?

Mỗi head trong attention học các pattern khác nhau với scale rất khác nhau:
- Head A có thể có $\|v\| \approx 1$, variance $\sim 0.5$
- Head B có $\|v\| \approx 10$, variance $\sim 50$

Nếu boost với raw variance, head B sẽ bị boost mạnh hơn 100 lần dù về mặt **relative** values trong cluster của A có thể đa dạng hơn. Min-max normalize trong từng (layer, head) sửa vấn đề này.

---

## Tại sao có thể đảm bảo backward compatibility?

Khi $\beta = 0$ và $\gamma = 0$:
- Joint feature $z = k_\text{norm}$ (sau normalize) → K-means giống hệt baseline
- $\tilde{S}_i = S_i$ → retrieve giống baseline

Như vậy, "value-aware" với hyperparameter zero **chính xác** là Squeezed Attention gốc (modulo random seed của K-means init).

---

## Memory overhead

Per (layer, head, cluster):
- Original: $(D_k)$ floats cho centroid
- Value-aware: $(D_k + D_v + 1)$ floats — thêm value centroid và variance scalar

Với $K$ centroids per head, $H$ heads, $L$ layers:
- Original: $L \cdot H \cdot K \cdot D_k$
- Value-aware: $L \cdot H \cdot K \cdot (D_k + D_v + 1) \approx 2\times$

Vì $K \ll N$ (typically $K = 5\% \cdot N$), tổng memory overhead **không đáng kể** so với KV cache đầy đủ.

Ví dụ, model 7B với context 32K:
- KV cache: $L \cdot H \cdot N \cdot D_k \cdot 2 = 32 \cdot 32 \cdot 32000 \cdot 128 \cdot 2 = 8.4$ GB
- Centroid cache (5%): $32 \cdot 32 \cdot 1600 \cdot 128 = 0.21$ GB
- VA centroid cache: $\approx 0.42$ GB

→ Overhead $0.21$ GB là rất nhỏ.

---

## Compute overhead

Online (per query token):

| Bước | Original | Value-aware | Overhead |
|------|---------|-------------|----------|
| $q \cdot C^\top$ | $O(K \cdot D)$ | $O(K \cdot D)$ | 0 |
| Compute $S_i$ | $O(K)$ | $O(K)$ | 0 |
| Apply boost $S \cdot (1+\gamma\tilde{\sigma})$ | — | $O(K)$ | $O(K)$ |
| Threshold filter | $O(K)$ | $O(K)$ | 0 |

Boost step thêm $O(K)$ multiplications — **negligible** so với $O(N \cdot D)$ của attention chính.

Offline (one-time):
- K-means cùng phức tạp $O(\text{iters} \cdot N \cdot K \cdot D)$
- Variance computation thêm $O(N \cdot D_v)$ — 1 pass duy nhất

→ Cải tiến **gần như free** về compute cost.

---

## Khi nào value-aware có lợi nhất?

Theo hệ thống synthetic experiments (`benchmarks/synthetic_benchmark.py`):

| Điều kiện | Value-aware có lợi? |
|-----------|---------------------|
| Values đồng nhất trong key-cluster (low diversity) | Không (gần như giống baseline) |
| Values đa dạng trong key-cluster (high diversity) | Có |
| Sparsity rất cao (>95%) | Có (mỗi cluster đại diện nhiều token) |
| Sparsity thấp (<70%) | Ít rõ rệt |

Trên LongBench tasks (đặc biệt QA tasks như NarrativeQA, HotpotQA), value diversity có xu hướng cao do model phải retrieve facts cụ thể (entities, dates) — điều này gợi ý value-aware có lợi.

---

## Hạn chế

1. **Phụ thuộc vào diversity của data**. Nếu values vốn đã đồng nhất, cải tiến không có nhiều giá trị.
2. **Yêu cầu offline computation thêm**. Cần lưu thêm `normalized_variance` (~1 float / cluster).
3. **Cần calibrate gamma**. Mặc định 0.3 thường ổn nhưng có thể cần tune theo task.
4. **Không tương thích trực tiếp với Triton kernels của repo gốc**. Cần chỉnh kernel để truyền `nvar` (xem `docs/INTEGRATION.md`).

---

## Tài liệu tham khảo

```bibtex
@article{hooper2024squeezed,
  title={Squeezed Attention: Accelerating Long Context Length LLM Inference},
  author={Hooper, Coleman and Kim, Sehoon and Mohammadzadeh, Hiva and 
          Maheswaran, Monishwaran and Paik, June and Mahoney, Michael W and 
          Keutzer, Kurt and Gholami, Amir},
  journal={arXiv preprint arXiv:2411.09688},
  year={2024}
}
```
