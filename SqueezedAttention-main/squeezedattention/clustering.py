import torch
import torch.nn.functional as F
import numpy as np
import math
import time
from sklearn.cluster import KMeans  # Dùng thư viện chuẩn cho CPU thay vì cuml


def run_clustering(
    tdict, num_clusters, observation_window=100, print_log=False, device=None
):
    if device is None:
        device = "cpu"  # Ép mặc định về CPU

    # initialize dicts to return
    centroids_tensor_dict = {}
    centroids_labels_dict = {}

    # compute num heads
    num_heads = tdict[0].shape[-3]
    num_lyrs = len(tdict)

    # compute shared prefix length
    shared_prefix_length = tdict[0].shape[-2]
    promptlen = shared_prefix_length - observation_window

    # loop over layers
    t1 = time.time()
    for layer_num in range(num_lyrs):
        if print_log:
            print("layer: ", layer_num)

        keys = tdict[layer_num].squeeze(0).float().to(device)
        K = num_clusters

        assert len(keys.shape) == 3

        if observation_window > 0:
            keys = keys[:, :-observation_window, :]
        num_heads = keys.shape[0]
        kdim = keys.shape[2]

        cluster_labels_list = []
        cluster_centers_list = []

        # iterate over heads
        for H in range(num_heads):
            head_data = keys[H]
            data_normalized = F.normalize(head_data, p=2, dim=-1)

            # CHUYỂN ĐỔI SANG NUMPY ĐỂ CHẠY TRÊN CPU
            data_np = data_normalized.cpu().numpy()

            kmeans = KMeans(
                n_clusters=K, max_iter=300, init="k-means++", n_init=10, random_state=0
            )
            kmeans.fit(data_np)
            cluster_labels = kmeans.labels_

            # convert labels to pytorch tensor
            labels = torch.tensor(cluster_labels, dtype=torch.long, device=device)

            # Compute cluster centers (centroids)
            cluster_centers = []
            for i in range(K):
                mask = labels == i
                cluster_keys = head_data[mask]
                if len(cluster_keys) > 0:
                    centroid = torch.mean(cluster_keys, dim=0)
                else:
                    centroid = torch.zeros(
                        head_data.shape[1], dtype=head_data.dtype, device=device
                    )
                cluster_centers.append(centroid)
            cluster_centers = torch.stack(cluster_centers, dim=0)

            cluster_labels_list.append(labels)
            cluster_centers_list.append(cluster_centers)

        a = torch.stack(cluster_centers_list, dim=0).unsqueeze(0)
        b = torch.stack(cluster_labels_list, dim=0).unsqueeze(0).to(torch.int64)

        centroids_tensor_dict[layer_num] = a
        centroids_labels_dict[layer_num] = b

    return centroids_tensor_dict, centroids_labels_dict


def run_global_threshold(
    key_dict,
    query_dict,
    centroids_tensor_dict,
    centroids_labels_dict,
    num_clusters,
    observation_window=100,
    print_log=False,
    device=None,
):

    if device is None:
        device = "cpu"

    # get shared prefix length here
    shared_prefix_length = query_dict[0].shape[-2]
    num_heads = query_dict[0].shape[-3]
    num_lyrs = len(query_dict)

    # global dict for centroids
    K = num_clusters

    # loop over layers
    attn_score_centroid_list = []
    for layer_num in range(num_lyrs):
        if print_log:
            print("layer: ", layer_num)

        # load centroids
        centroids_tensor = centroids_tensor_dict[layer_num].squeeze(0).to(device)
        centroids_labels = centroids_labels_dict[layer_num].squeeze(0).to(device)

        keys = key_dict[layer_num].squeeze(0).to(device)
        queries = query_dict[layer_num].squeeze(0).to(device)
        keys_shared_prefix = keys[:, :-observation_window, :]

        # compute attention to centroids
        queries_obs_window = queries[:, -observation_window:, :].float()
        attn_scores_centroids = torch.matmul(
            queries_obs_window, centroids_tensor.transpose(1, 2)
        ) / math.sqrt(keys.shape[-1])

        # initialize score
        shape = (
            keys_shared_prefix.shape[0],
            keys_shared_prefix.shape[1],
            observation_window,
        )
        scores = torch.zeros(shape, device=device)

        # loop over centroid and copy centroid scores onto the centroids
        for k in range(K):
            label_mask = centroids_labels == k
            current_attn_scores_centroids = attn_scores_centroids[:, :, k].unsqueeze(-2)
            scores = scores + label_mask.unsqueeze(-1) * current_attn_scores_centroids

        # compute number of keys per cluster
        num_keys_per_cluster = torch.zeros(
            (keys_shared_prefix.shape[0], K), device=device
        )
        for k in range(K):
            label_mask = centroids_labels == k
            num_keys_per_cluster[:, k] = torch.sum(label_mask, dim=-1)

        # estimate denominator here
        attn_scores_centroids_est_exp = torch.exp(attn_scores_centroids)
        num_keys_per_cluster = num_keys_per_cluster.unsqueeze(-2)
        denom_est_tmp = num_keys_per_cluster * attn_scores_centroids_est_exp
        denom_est = torch.sum(denom_est_tmp, dim=-1)  # per-head estimate

        # divide centroid scores (copied per-token) by the denominator estimate
        scores_scaled_sm = torch.exp(scores) / denom_est.unsqueeze(-2)

        # compute average across tokens
        scored_scaled_sm_sum = torch.mean(scores_scaled_sm, dim=-1, dtype=torch.float32)
        attn_score_centroid_list.append(scored_scaled_sm_sum)

    # stack all scores
    full_centroid_scores = torch.stack(attn_score_centroid_list, dim=0)

    # compute global thresholds here
    qlist = [0.5, 0.7, 0.8, 0.9]
    q = torch.tensor(qlist, device="cpu")

    # for long sequence lengths, we need to move to CPU
    full_centroid_scores_cpu = full_centroid_scores.cpu().numpy()
    quantile_result = np.quantile(full_centroid_scores_cpu, q)
    thresholds = torch.tensor(quantile_result)

    tdict = {}
    i = 0
    for q_idx in qlist:
        tdict[q_idx] = thresholds[i].item()
        i += 1

    # save shared prefix length here
    tdict["shared_prefix_length"] = shared_prefix_length
    tdict["observation_window"] = observation_window

    return tdict
