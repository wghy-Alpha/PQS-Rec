import argparse
import math
from typing import Dict, List

import torch

from data import build_data_splits, compute_global_popularity, normalize_dataset_name
from model import ModelConfig, RLMRecFramework


def auc_from_scores(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().cpu()
    labels = labels.detach().cpu()

    pos_mask = labels.eq(1)
    neg_mask = labels.eq(0)
    n_pos = int(pos_mask.sum().item())
    n_neg = int(neg_mask.sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float32)
    pos_rank_sum = ranks[pos_mask].sum().item()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def recall_at_k(rank_index: int, k: int) -> float:
    return 1.0 if rank_index < k else 0.0


def ndcg_at_k(rank_index: int, k: int) -> float:
    if rank_index >= k:
        return 0.0
    return 1.0 / math.log2(rank_index + 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLMRec 评估脚本")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--item-emb-path", type=str, default="")
    parser.add_argument("--user-emb-path", type=str, default="")
    parser.add_argument("--max-seq-len", type=int, default=20)
    parser.add_argument("--negative-ratio-eval", type=int, default=100)
    parser.add_argument("--topk", type=str, default="10,20")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset = normalize_dataset_name(args.dataset)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg_dict = ckpt.get("model_config")
    if cfg_dict is None:
        raise ValueError("checkpoint 缺少 model_config")

    item_emb_path = args.item_emb_path or ckpt.get("item_emb_path", "")
    user_emb_path = args.user_emb_path or ckpt.get("user_emb_path", "")
    if not item_emb_path:
        raise ValueError("请提供 --item-emb-path 或使用包含该字段的checkpoint")

    splits = build_data_splits(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
        padding_item_id=0,
        item_emb_path=item_emb_path,
        user_emb_path=user_emb_path if user_emb_path else None,
        negative_ratio_train=1,
        negative_ratio_eval=args.negative_ratio_eval,
        seed=2026,
    )

    model = RLMRecFramework(ModelConfig(**cfg_dict))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    popularity = compute_global_popularity(
        num_items=splits.meta.num_items,
        item_ids=splits.popularity_item_ids,
        padding_item_id=0,
    )

    topks = [int(x.strip()) for x in args.topk.split(",") if x.strip()]
    recall_sum = {k: 0.0 for k in topks}
    ndcg_sum = {k: 0.0 for k in topks}
    auc_list: List[float] = []

    ds = splits.test
    group_size = 1 + ds.negative_ratio
    n_groups = len(ds.positive_samples)

    with torch.no_grad():
        for gidx in range(n_groups):
            samples = [ds[gidx * group_size + j] for j in range(group_size)]

            target_item_ids = torch.cat([s.target_item_id for s in samples], dim=0)
            target_item_text = torch.stack([s.target_item_text for s in samples], dim=0)
            hist_item_ids = torch.stack([s.hist_item_ids for s in samples], dim=0)
            hist_item_text = torch.stack([s.hist_item_text for s in samples], dim=0)
            user_profile_text = torch.stack([s.user_profile_text for s in samples], dim=0)
            labels = torch.cat([s.label for s in samples], dim=0)

            hist_popularity = popularity[hist_item_ids]
            target_popularity = popularity[target_item_ids]

            out = model(
                target_item_ids=target_item_ids,
                target_item_text=target_item_text,
                hist_item_ids=hist_item_ids,
                hist_item_text=hist_item_text,
                hist_popularity=hist_popularity,
                target_item_popularity=target_popularity,
                user_profile_text=user_profile_text,
            )
            scores = out["logits"]

            sorted_idx = torch.argsort(scores, descending=True)
            pos_idx = int(torch.where(labels.eq(1))[0][0].item())
            rank_index = int(torch.where(sorted_idx.eq(pos_idx))[0][0].item())

            for k in topks:
                recall_sum[k] += recall_at_k(rank_index, k)
                ndcg_sum[k] += ndcg_at_k(rank_index, k)

            auc_list.append(auc_from_scores(scores, labels))

    n = max(1, n_groups)
    print(f"dataset={args.dataset} groups={n_groups} negatives={ds.negative_ratio}")
    for k in topks:
        print(f"Recall@{k}: {recall_sum[k] / n:.6f}")
        print(f"NDCG@{k}: {ndcg_sum[k] / n:.6f}")
    print(f"AUC: {sum(auc_list) / max(1, len(auc_list)):.6f}")


if __name__ == "__main__":
    main()
