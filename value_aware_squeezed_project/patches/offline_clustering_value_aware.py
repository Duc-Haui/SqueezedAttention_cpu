"""
Patched offline clustering script tương thích với repo gốc SqueezedAttention.

So với offline_clustering.py gốc:
1. Dùng `run_value_aware_clustering` từ value_aware package thay vì run_clustering gốc
2. Lưu thêm value_centroids và normalized_variance
3. Thêm CLI args: --alpha, --beta, --gamma

Cách dùng:
    Copy file này vào root của repo SqueezedAttention (cùng cấp với offline_clustering.py gốc)
    rồi chạy:

    python offline_clustering_value_aware.py LLaMA-2-7B-32K \\
        --dataset 2wikimqa \\
        --output_path /tmp/clusters/2wikimqa/ \\
        --percent_clusters 5 \\
        --observation_window 100 \\
        --alpha 1.0 --beta 0.5 --gamma 0.3 \\
        --device 0
"""

import argparse
import json
import os
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

# Local imports - giả định chạy từ root của repo SqueezedAttention
from utils.modelutils import *  # noqa
from utils.datautils import *   # noqa
from utils.model_parse import parse_model, get_layers
from squeezedattention.utils import build_chat, truncate_fn

# Value-aware extension - đảm bảo `value_aware` package nằm trên PYTHONPATH
# hoặc cùng thư mục
from value_aware.clustering import run_value_aware_clustering
from value_aware.threshold import run_value_aware_global_threshold


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=str, help="Model name (key trong model2path.json)")
    p.add_argument("--output_path", type=str, default="output/")
    p.add_argument(
        "--dataset", type=str, default="trec",
        choices=[
            "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
            "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
            "samsum", "lcc", "repobench-p",
        ],
    )
    p.add_argument("--percent_clusters", type=int, default=5,
                   help="% centroids so với fixed context length")
    p.add_argument("--observation_window", type=int, default=100)
    p.add_argument("--device", type=int, default=0)
    # Value-aware specific
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Trọng số K trong joint K-V clustering")
    p.add_argument("--beta", type=float, default=0.5,
                   help="Trọng số V. 0 = tắt value-aware (về baseline gốc)")
    p.add_argument("--gamma", type=float, default=0.3,
                   help="Hệ số boost variance khi tính threshold/retrieve")
    p.add_argument("--kmeans_iters", type=int, default=10)
    p.add_argument("--max_samples", type=int, default=-1,
                   help="Giới hạn số sample. -1 = tất cả")
    return p.parse_args()


def main():
    args = parse_args()
    DEV = torch.device(f"cuda:{args.device}")

    # Load config
    model2path = json.load(open("LongBench/config/model2path.json", "r"))
    model2maxlen = json.load(open("LongBench/config/model2maxlen.json", "r"))
    model_path = model2path[args.model]
    max_length = model2maxlen[args.model]

    print(f"=== Value-Aware Squeezed Attention Offline Clustering ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Config: alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}")
    print(f"Centroids: {args.percent_clusters}% of context length")
    print(f"Output: {args.output_path}")
    print()

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = LlamaConfig.from_pretrained(model_path)
    config.return_qkv_states = True
    config._flash_attn_2_enabled = True
    config._attn_implementation = "flash_attention_2"
    model = LlamaForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.bfloat16
    ).to(DEV).eval()

    model_type = parse_model(model)
    layers = get_layers(model, model_type)

    # Load LongBench dataset
    from datasets import load_dataset
    dataset = args.dataset
    dataset_name_prompt = dataset + "_prompt"
    data = load_dataset("THUDM/LongBench", dataset, split="test")

    dataset2prompt = json.load(open("LongBench/config/dataset2prompt.json", "r"))
    prompt_format = dataset2prompt[dataset]
    prompt_only_format = dataset2prompt[dataset_name_prompt]

    # Compute shared prefix lengths
    data_all = list(data)
    if args.max_samples > 0:
        data_all = data_all[: args.max_samples]
    shared_prefix_length = {}
    for i, sample in enumerate(data_all):
        prompt = prompt_format.format(**sample)
        prompt_only = prompt_only_format.format(**sample)
        prompt, sp_len = truncate_fn(
            prompt, prompt_only, tokenizer, max_length, dataset, DEV
        )
        shared_prefix_length[i] = sp_len
        assert sp_len > 0

    # Hooks để collect K, V, Q
    all_queries_layers = []
    all_keys_layers = []
    all_values_layers = []

    def hook(module, inp, out):
        _, qkv, _ = out
        queries, keys, values = qkv
        sp_len = shared_prefix_length[dataidx]
        queries = queries[:, :, :sp_len]
        keys = keys[:, :, :sp_len]
        values = values[:, :, :sp_len]
        all_queries_layers.append(queries)
        all_keys_layers.append(keys)
        all_values_layers.append(values)

    for layer in layers:
        layer.self_attn.register_forward_hook(hook)

    os.makedirs(args.output_path, exist_ok=True)

    # Loop qua samples
    for dataidx, d in enumerate(tqdm(data_all)):
        all_queries_layers.clear()
        all_keys_layers.clear()
        all_values_layers.clear()

        prompt = prompt_format.format(**d)
        prompt_only = prompt_only_format.format(**d)
        prompt, _ = truncate_fn(
            prompt, prompt_only, tokenizer, max_length, dataset, DEV
        )
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to(DEV)

        # Forward pass
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=1,
                use_cache=False,
                output_attentions=True,
            )

        # Số centroid
        sp_len = shared_prefix_length[dataidx]
        N_to_cluster = sp_len - args.observation_window
        num_clusters = max(1, int(args.percent_clusters / 100.0 * N_to_cluster))

        t0 = time.time()
        # === Value-Aware Clustering ===
        kc_dict, vc_dict, lbl_dict, vvar_dict, nvar_dict = run_value_aware_clustering(
            all_keys_layers,
            all_values_layers,
            num_clusters=num_clusters,
            observation_window=args.observation_window,
            alpha=args.alpha,
            beta=args.beta,
            num_iters=args.kmeans_iters,
            print_log=False,
            device=DEV,
        )

        # === Value-Aware Global Threshold ===
        global_threshold_dict = run_value_aware_global_threshold(
            keys_layers=all_keys_layers,
            queries_layers=all_queries_layers,
            key_centroids_dict=kc_dict,
            labels_dict=lbl_dict,
            normalized_variance_dict=nvar_dict,
            num_clusters=num_clusters,
            observation_window=args.observation_window,
            gamma=args.gamma,
            device=DEV,
        )
        clustering_time = time.time() - t0

        # Save (CPU)
        for k in kc_dict:
            kc_dict[k] = kc_dict[k].cpu()
            vc_dict[k] = vc_dict[k].cpu()
            lbl_dict[k] = lbl_dict[k].cpu()
            vvar_dict[k] = vvar_dict[k].cpu()
            nvar_dict[k] = nvar_dict[k].cpu()

        prefix = f"{args.output_path}/sample_{dataidx}_clusters_{num_clusters}"
        torch.save(kc_dict, f"{prefix}_key_centroids.pt")
        torch.save(vc_dict, f"{prefix}_value_centroids.pt")
        torch.save(lbl_dict, f"{prefix}_labels.pt")
        torch.save(vvar_dict, f"{prefix}_value_variance.pt")
        torch.save(nvar_dict, f"{prefix}_normalized_variance.pt")
        torch.save(global_threshold_dict, f"{prefix}_thresholds.pt")

        # Cleanup
        n_layers = len(all_keys_layers)
        for _ in range(n_layers):
            del all_queries_layers[0]
            del all_keys_layers[0]
            del all_values_layers[0]
        torch.cuda.empty_cache()

        if dataidx == 0:
            print(f"  [first sample] clustering took {clustering_time:.2f}s, "
                  f"thresholds: {global_threshold_dict}")

    print(f"\nDone. Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
