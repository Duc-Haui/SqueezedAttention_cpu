"""
LongBench-style Evaluation - tương tự Table 2 trong paper Squeezed Attention.

Script này:
1. Load LongBench dataset (1 trong 14 tasks: NQA, Qasper, MFQA, HotpotQA, ...)
2. Cho mỗi sample, dùng full prompt làm "fixed context"
3. Chạy 4 phương pháp generate output:
   - Full attention (baseline accuracy)
   - Key-only Squeezed Attention
   - Value-Aware Squeezed Attention (cải tiến)
   - QUEST-style
4. Compute metric (F1/ROUGE/Code-sim tùy task)

QUAN TRỌNG: Đây là evaluation **mô phỏng**, vì attention sparse phải được patch vào
attention layer của HF model thực tế. Implementation chính thức cần monkey-patch
LlamaAttention.forward (như repo gốc làm trong custom transformers).

Để giữ đơn giản và chạy được mà không cần custom transformers, script này:
- Dùng sparse attention chỉ ở vài layers cuối (proxy approximation)
- Hoặc: dùng "intervention" approach: thay K, V bằng version đã được mask cho sparse heads

Nếu bạn muốn fully sát paper, hãy patch LlamaAttention - xem hướng dẫn trong docs/INTEGRATION.md

Cách chạy:
    # Cài deps trước:
    pip install datasets rouge fuzzywuzzy jieba

    # Run trên một task (subset):
    python benchmarks/benchmark_longbench.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --dataset 2wikimqa \\
        --max_samples 5 \\
        --sparsity 0.85
"""

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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


# ---------------------------------------------------------------------------
# Metrics (từ LongBench/metrics.py - copy gọn lại)
# ---------------------------------------------------------------------------
def normalize_answer(s):
    """Lowercase, remove punctuation, articles, extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    s = " ".join(s.split())
    return s


def f1_token(pred, gold):
    """Token-level F1."""
    pred = normalize_answer(pred).split()
    gold = normalize_answer(gold).split()
    common = Counter(pred) & Counter(gold)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred) if pred else 0
    recall = num_same / len(gold) if gold else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction, ground_truths, **kwargs):
    """Max F1 qua các ground truths."""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    return max(f1_token(prediction, gt) for gt in ground_truths)


def classification_score(prediction, ground_truths, all_classes=None):
    """Trec, classification."""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    em = []
    classes = all_classes or []
    for c in classes:
        if c in prediction:
            em.append(c)
    score = 0.0
    for gt in ground_truths:
        if gt in em and len(em) > 0:
            score = max(score, 1.0 / len(em))
    return score


# Map dataset -> metric
DATASET_METRICS = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    # Các task khác cần rouge - skip cho version đơn giản này
}

# Prompt format
DATASET_PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
}

DATASET_MAXLEN = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64,
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32,
    "trec": 64, "triviaqa": 32,
}


# ---------------------------------------------------------------------------
# Sparse-attention generation (simulation approach)
# ---------------------------------------------------------------------------
class SparseAttentionPatcher:
    """
    Patch attention layer của model để dùng sparse attention với fixed context.
    
    Chiến lược đơn giản: hook vào forward của LlamaAttention (hoặc tương đương),
    intercept K, V cho phần fixed context, áp dụng masking, rồi gọi attention thường.
    
    Lưu ý: đây là approximation. Implementation production yêu cầu monkey-patch
    chi tiết hơn (xem repo gốc).
    """

    def __init__(self, model, fixed_context_len, method="full",
                 cluster_data=None, gamma=0.0, sparse_layers=None):
        """
        Args:
            method: 'full', 'key_only', 'value_aware', 'quest'
            cluster_data: dict[layer_idx] -> tuple(centroids, labels, sizes, [nvar])
            sparse_layers: list of layer indices to apply sparse. None = all.
        """
        self.model = model
        self.fixed_len = fixed_context_len
        self.method = method
        self.cluster_data = cluster_data or {}
        self.gamma = gamma
        self.sparse_layers = sparse_layers
        self.thresholds = {}
        self.handles = []

    def set_thresholds(self, thresholds):
        self.thresholds = thresholds

    def attach(self):
        # Để giữ demo đơn giản, không patch thực sự attention layer mà
        # chỉ chạy generation thường. Sparse attention chỉ áp dụng ở verify step.
        # (Để có patching đầy đủ, xem docs/INTEGRATION.md)
        pass

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ---------------------------------------------------------------------------
# Approximation evaluation: thay vì generation, ta đo "how close" attention output
# với phương pháp sparse với attention output full.
# Đây là proxy cho accuracy: nếu attention output sát thì generation cũng sát.
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_kv_via_cache(model, input_ids):
    outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
    past = outputs.past_key_values
    keys = [layer_kv[0].detach() for layer_kv in past]
    values = [layer_kv[1].detach() for layer_kv in past]
    return keys, values


@torch.no_grad()
def generate_with_full_attention(model, tokenizer, prompt, max_new_tokens=64, device="cuda"):
    """Standard greedy decoding để có 'reference' output."""
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=4096).input_ids.to(device)
    out = model.generate(
        input_ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip()


@torch.no_grad()
def evaluate_attention_quality(
    model, tokenizer, fixed_context, query_text,
    method, num_clusters, alpha=1.0, beta=0.5, gamma=0.3,
    target_sparsity=0.85, obs_window=64, device="cuda",
):
    """
    Thay vì generate full text, đo similarity của attention output trên một
    sample query. Trả về metrics dict.
    """
    fixed_ids = tokenizer(fixed_context, return_tensors="pt", truncation=True,
                          max_length=4096).input_ids.to(device)
    q_ids = tokenizer(query_text, return_tensors="pt").input_ids.to(device)
    full_ids = torch.cat([fixed_ids, q_ids], dim=1)

    keys_l, values_l = collect_kv_via_cache(model, fixed_ids)
    k_full_l, _ = collect_kv_via_cache(model, full_ids)
    num_layers = len(keys_l)

    metrics_per_layer = []
    for li in range(num_layers):
        k = keys_l[li].squeeze(0).to(device).float()  # (H, N, D)
        v = values_l[li].squeeze(0).to(device).float()
        if obs_window > 0:
            k_c = k[:, :-obs_window, :]
            v_c = v[:, :-obs_window, :]
        else:
            k_c, v_c = k, v
        H, N, D = k_c.shape

        K_actual = min(num_clusters, N)
        scale = 1.0 / (D ** 0.5)

        # Q proxy = key cuối của full
        q = k_full_l[li].squeeze(0)[:, -1, :].to(device).float()

        # Reference
        ref = baseline_full_attention(q, k_c, v_c, scale)

        if method == "full":
            return  # don't compute, ref is already there

        # Build clustering
        if method == "key_only":
            kc, lbl, sizes, nvar = _build_keyonly(k_c, K_actual)
            T = calibrate_threshold(
                q.unsqueeze(0), kc, sizes, torch.zeros_like(sizes), lbl,
                target_sparsity=target_sparsity, gamma=0.0, num_threshold_search=60,
            )
            out, info = key_only_attention_forward(
                q, k_c, v_c, kc, sizes, lbl, threshold=T, scale=scale,
            )
        elif method == "value_aware":
            kc, lbl, sizes, nvar = _build_va(k_c, v_c, K_actual, alpha, beta)
            T = calibrate_threshold(
                q.unsqueeze(0), kc, sizes, nvar, lbl,
                target_sparsity=target_sparsity, gamma=gamma, num_threshold_search=60,
            )
            out, info = squeezed_attention_forward(
                q, k_c, v_c, kc, sizes, nvar, lbl, threshold=T, gamma=gamma, scale=scale,
            )
        elif method == "quest":
            chunk_size = max(1, N // K_actual)
            kc, lbl, sizes = quest_style_clustering(k_c, v_c, chunk_size=chunk_size)
            nvar_q = torch.zeros_like(sizes)
            T = calibrate_threshold(
                q.unsqueeze(0), kc, sizes, nvar_q, lbl,
                target_sparsity=target_sparsity, gamma=0.0, num_threshold_search=60,
            )
            out, info = quest_attention_forward(
                q, k_c, v_c, kc, sizes, lbl, threshold=T, scale=scale,
            )
        else:
            raise ValueError(method)

        cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
        mse = F.mse_loss(out, ref).item()
        metrics_per_layer.append({
            "cos": cos, "mse": mse, "budget": info["kv_budget"],
        })

    return metrics_per_layer


def _build_keyonly(keys, K):
    H, N, D = keys.shape
    k_norm = F.normalize(keys, dim=-1)
    from value_aware.clustering import _kmeans_cosine
    centroids, labels = _kmeans_cosine(k_norm, K, num_iters=10)
    sizes = torch.zeros(H, K, device=keys.device, dtype=torch.float)
    sizes.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.float))
    nvar = torch.zeros_like(sizes)
    return centroids, labels, sizes, nvar


def _build_va(keys, values, K, alpha, beta):
    H, N, D = keys.shape
    kc, vc, lbl, vvar = value_aware_kmeans(
        keys, values, K, alpha=alpha, beta=beta, num_iters=10,
    )
    sizes = torch.zeros(H, K, device=keys.device, dtype=torch.float)
    sizes.scatter_add_(1, lbl, torch.ones_like(lbl, dtype=torch.float))
    vmin = vvar.min(dim=-1, keepdim=True).values
    vmax = vvar.max(dim=-1, keepdim=True).values
    nvar = (vvar - vmin) / (vmax - vmin + 1e-8)
    return kc, lbl, sizes, nvar


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dataset", default="2wikimqa", choices=list(DATASET_METRICS.keys()))
    p.add_argument("--max_samples", type=int, default=5)
    p.add_argument("--max_context", type=int, default=4096)
    p.add_argument("--obs_window", type=int, default=64)
    p.add_argument("--percent_clusters", type=float, default=5.0)
    p.add_argument("--sparsity", type=float, default=0.85)
    p.add_argument("--gamma", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--output", default="results/longbench_eval.json")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 else torch.bfloat16

    print(f"=== Loading {args.model} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device,
        attn_implementation="eager",
    ).eval()

    # Load dataset
    print(f"=== Loading LongBench dataset: {args.dataset} ===")
    try:
        from datasets import load_dataset
        data = load_dataset("THUDM/LongBench", args.dataset, split="test", trust_remote_code=True)
    except Exception as e:
        print(f"Cannot load dataset: {e}")
        print("Hint: pip install datasets, hoặc dùng --offline_data <path>")
        return

    samples = list(data)[:args.max_samples]
    print(f"Evaluating {len(samples)} samples")

    metric_fn = DATASET_METRICS[args.dataset]
    prompt_format = DATASET_PROMPT.get(args.dataset)
    max_gen = DATASET_MAXLEN.get(args.dataset, 32)

    if not prompt_format:
        print(f"No prompt format for {args.dataset}, skipping")
        return

    results_per_method = {
        "full": {"per_sample_metric": [], "attention_quality": []},
        "key_only": {"per_sample_metric": [], "attention_quality": []},
        "value_aware": {"per_sample_metric": [], "attention_quality": []},
        "quest": {"per_sample_metric": [], "attention_quality": []},
    }

    for sample in tqdm(samples):
        try:
            full_prompt = prompt_format.format(**sample)
        except KeyError as e:
            print(f"Format error: {e}, skipping sample")
            continue

        # Truncate
        ids = tokenizer(full_prompt, return_tensors="pt", truncation=True,
                        max_length=args.max_context).input_ids
        if ids.shape[1] < 512:  # quá ngắn, bỏ qua
            continue
        truncated_prompt = tokenizer.decode(ids[0], skip_special_tokens=True)

        # Reference generation
        full_pred = generate_with_full_attention(
            model, tokenizer, truncated_prompt,
            max_new_tokens=max_gen, device=device,
        )

        answers = sample.get("answers", [])
        all_classes = sample.get("all_classes", [])
        kwargs = {"all_classes": all_classes} if all_classes else {}

        try:
            score_full = metric_fn(full_pred, answers, **kwargs)
        except Exception as e:
            print(f"Metric error: {e}")
            score_full = 0.0
        results_per_method["full"]["per_sample_metric"].append(score_full)

        # Đối với KO/VA/QUEST, ta đánh giá ATTENTION OUTPUT QUALITY (proxy cho accuracy)
        # vì để có generation thật cần patch model
        # Tách context và question
        # Heuristic: split tại "Question:" hoặc dùng cả prompt làm context
        context_part = truncated_prompt
        question_part = sample.get("input", "")

        for method in ["key_only", "value_aware", "quest"]:
            try:
                num_clusters = max(1, int(args.percent_clusters / 100.0 *
                                          (len(tokenizer(context_part).input_ids) - args.obs_window)))
                m_layers = evaluate_attention_quality(
                    model, tokenizer, context_part, question_part,
                    method=method, num_clusters=num_clusters,
                    alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                    target_sparsity=args.sparsity, obs_window=args.obs_window,
                    device=device,
                )
                if m_layers:
                    avg_cos = sum(m["cos"] for m in m_layers) / len(m_layers)
                    avg_mse = sum(m["mse"] for m in m_layers) / len(m_layers)
                    avg_bud = sum(m["budget"] for m in m_layers) / len(m_layers)
                    results_per_method[method]["attention_quality"].append({
                        "cos": avg_cos, "mse": avg_mse, "budget": avg_bud,
                    })
            except Exception as e:
                print(f"  Error in {method}: {e}")
                continue

        torch.cuda.empty_cache()

    # ----- Print summary -----
    print("\n" + "=" * 80)
    print(f"LongBench Evaluation: {args.dataset} | model={args.model}")
    print(f"sparsity={args.sparsity}, gamma={args.gamma}, beta={args.beta}")
    print("=" * 80)

    if results_per_method["full"]["per_sample_metric"]:
        avg_full_metric = sum(results_per_method["full"]["per_sample_metric"]) / \
                          len(results_per_method["full"]["per_sample_metric"])
        print(f"\nFull Attention {args.dataset} score: {avg_full_metric*100:.2f}")

    print(f"\nAttention Output Quality (proxy for accuracy):")
    print(f"{'Method':>14s} | {'CosSim':>8s} | {'MSE':>10s} | {'Budget':>8s}")
    print("-" * 60)
    for method in ["key_only", "value_aware", "quest"]:
        aq = results_per_method[method]["attention_quality"]
        if aq:
            avg_cos = sum(a["cos"] for a in aq) / len(aq)
            avg_mse = sum(a["mse"] for a in aq) / len(aq)
            avg_bud = sum(a["budget"] for a in aq) / len(aq)
            print(f"{method:>14s} | {avg_cos:>8.4f} | {avg_mse:>10.6f} | {avg_bud*100:>7.2f}%")

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out = {"args": vars(args), "results": {}}
        for k, v in results_per_method.items():
            out["results"][k] = {
                "metric_avg": (
                    sum(v["per_sample_metric"]) / len(v["per_sample_metric"])
                    if v["per_sample_metric"] else None
                ),
                "attention_quality_avg": (
                    {
                        "cos": sum(a["cos"] for a in v["attention_quality"]) / len(v["attention_quality"]),
                        "mse": sum(a["mse"] for a in v["attention_quality"]) / len(v["attention_quality"]),
                        "budget": sum(a["budget"] for a in v["attention_quality"]) / len(v["attention_quality"]),
                    } if v["attention_quality"] else None
                ),
                "n_samples": len(v["per_sample_metric"]) or len(v["attention_quality"]),
            }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
