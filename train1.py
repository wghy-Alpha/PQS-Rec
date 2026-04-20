import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import (
    SequenceRecDataset,
    build_data_splits,
    collate_fn,
    compute_global_popularity,
    normalize_dataset_name,
)
from model import ModelConfig, RLMRecFramework, info_nce_loss


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


def run_epoch(
    model: RLMRecFramework,
    loader: DataLoader,
    popularity: torch.Tensor,
    bce: nn.Module,
    contrastive_weight: float,
    optimizer: optim.Optimizer | None,
    label_smoothing: float = 0.0,
    grad_clip_norm: float = 0.0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_rec = 0.0
    total_cl = 0.0
    all_scores = []
    all_labels = []
    n_batches = 0

    for batch in loader:
        # 【关键修复 1】：将整个 batch 的数据推送到 GPU
        batch = {k: v.cuda() for k, v in batch.items()}
        
        target_item_ids = batch["target_item_ids"]
        target_item_text = batch["target_item_text"]
        hist_item_ids = batch["hist_item_ids"]
        hist_item_text = batch["hist_item_text"]
        user_profile_text = batch["user_profile_text"]
        labels = batch["labels"]

        hist_popularity = popularity[hist_item_ids]
        target_popularity = popularity[target_item_ids]

        if is_train:
            if label_smoothing > 0:
                labels_for_loss = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
            else:
                labels_for_loss = labels

            out = model(
                target_item_ids=target_item_ids,
                target_item_text=target_item_text,
                hist_item_ids=hist_item_ids,
                hist_item_text=hist_item_text,
                hist_popularity=hist_popularity,
                target_item_popularity=target_popularity,
                user_profile_text=user_profile_text,
            )
            rec_loss = bce(out["logits"], labels_for_loss)
            user_cl = info_nce_loss(out["user_sem"], out["user_cf"], temperature=0.2)
            item_cl = info_nce_loss(out["item_sem"], out["item_cf"], temperature=0.2)
            cl_loss = 0.5 * (user_cl + item_cl)
            loss = rec_loss + contrastive_weight * cl_loss

            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        else:
            with torch.no_grad():
                out = model(
                    target_item_ids=target_item_ids,
                    target_item_text=target_item_text,
                    hist_item_ids=hist_item_ids,
                    hist_item_text=hist_item_text,
                    hist_popularity=hist_popularity,
                    target_item_popularity=target_popularity,
                    user_profile_text=user_profile_text,
                )
                rec_loss = bce(out["logits"], labels)
                user_cl = info_nce_loss(out["user_sem"], out["user_cf"], temperature=0.2)
                item_cl = info_nce_loss(out["item_sem"], out["item_cf"], temperature=0.2)
                cl_loss = 0.5 * (user_cl + item_cl)
                loss = rec_loss + contrastive_weight * cl_loss

        total_loss += float(loss.item())
        total_rec += float(rec_loss.item())
        total_cl += float(cl_loss.item())
        all_scores.append(out["logits"].detach())
        all_labels.append(labels.detach())
        n_batches += 1

    if n_batches == 0:
        return {"loss": 0.0, "rec_loss": 0.0, "cl_loss": 0.0, "auc": 0.5}

    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)
    auc = auc_from_scores(scores, labels)

    return {
        "loss": total_loss / n_batches,
        "rec_loss": total_rec / n_batches,
        "cl_loss": total_cl / n_batches,
        "auc": auc,
    }


def evaluate_ranking(
    model: RLMRecFramework,
    dataset: SequenceRecDataset,
    popularity: torch.Tensor,
    topks: List[int],
    num_negatives: int = 100,
) -> Dict[str, float]:
    model.eval()
    recall_sum = {k: 0.0 for k in topks}
    ndcg_sum = {k: 0.0 for k in topks}
    auc_sum = 0.0
    valid_groups = 0

    num_items = dataset.num_items
    pad_id = dataset.padding_item_id

    with torch.no_grad():
        for user_idx, hist_ids_raw, target_item in dataset.positive_samples:
            hist_set = set(hist_ids_raw)

            # 动态随机抽样 100 个用户未交互过的 item
            negatives = set()
            while len(negatives) < num_negatives:
                neg_item = random.randint(1, num_items - 1)
                if neg_item not in hist_set and neg_item != target_item and neg_item != pad_id:
                    negatives.add(neg_item)

            # 构造 101 个评估序列：第 0 个为正样本
            candidates = [target_item] + list(negatives)

            # 构建 CPU ID 并对齐 shape
            hist_item_ids_cpu = torch.full((1, dataset.max_seq_len), pad_id, dtype=torch.long)
            hist = list(hist_ids_raw)[-dataset.max_seq_len :]
            if hist:
                hist_item_ids_cpu[0, : len(hist)] = torch.tensor(hist, dtype=torch.long)
            
            # 放入 GPU 并提取特征
            hist_item_ids = hist_item_ids_cpu.cuda()
            hist_item_text = dataset.item_text_table[hist_item_ids_cpu].cuda()
            hist_popularity = popularity[hist_item_ids]
            user_profile_text = dataset.user_text_table[user_idx].unsqueeze(0).cuda()

            # 候选 items 放入 GPU 提取特征
            target_item_ids_cpu = torch.tensor(candidates, dtype=torch.long)
            target_item_ids = target_item_ids_cpu.cuda()
            target_item_text = dataset.item_text_table[target_item_ids_cpu].cuda()
            target_popularity = popularity[target_item_ids]

            # 利用 expand 并行打分 101 个 item
            bsz = target_item_ids.size(0)
            out = model(
                target_item_ids=target_item_ids,
                target_item_text=target_item_text,
                hist_item_ids=hist_item_ids.expand(bsz, -1),
                hist_item_text=hist_item_text.expand(bsz, -1, -1),
                hist_popularity=hist_popularity.expand(bsz, -1),
                target_item_popularity=target_popularity,
                user_profile_text=user_profile_text.expand(bsz, -1),
            )
            
            # 计算指标
            scores = out["logits"].view(-1)
            target_score = scores[0] 
            
            rank_index = int((scores > target_score).sum().item())
            auc = float((scores[1:] < target_score).sum().item()) / num_negatives

            for k in topks:
                recall_sum[k] += recall_at_k(rank_index, k)
                ndcg_sum[k] += ndcg_at_k(rank_index, k)
                
            auc_sum += auc
            valid_groups += 1

    denom = max(1, valid_groups)
    metrics: Dict[str, float] = {
        "rank_groups": float(valid_groups),
        "rank_auc": auc_sum / denom,
    }
    for k in topks:
        metrics[f"recall@{k}"] = recall_sum[k] / denom
        metrics[f"ndcg@{k}"] = ndcg_sum[k] / denom
    return metrics


def evaluate_ranking_full(
    model: RLMRecFramework,
    dataset: SequenceRecDataset,
    popularity: torch.Tensor,
    topks: List[int],
    candidate_batch_size: int,
) -> Dict[str, float]:
    """
    全量评估（all-rank）：
    对每个测试样本，在全物品候选集合上打分并计算 rank 指标。
    为控制显存，候选物品按 candidate_batch_size 分块。
    """
    model.eval()
    recall_sum = {k: 0.0 for k in topks}
    ndcg_sum = {k: 0.0 for k in topks}
    auc_sum = 0.0
    valid_groups = 0

    num_items = dataset.num_items
    pad_id = dataset.padding_item_id

    with torch.no_grad():
        for user_idx, hist_ids_raw, target_item in dataset.positive_samples:
            hist_set = set(hist_ids_raw)
            # all-rank: 候选为所有物品（去除 padding 与当前历史中的已见物品）
            candidates = [iid for iid in range(1, num_items) if iid not in hist_set]
            if target_item not in candidates:
                candidates.append(target_item)

            if not candidates:
                continue

            # 1. 先在 CPU 上准备好 ID
            hist_item_ids_cpu = torch.full((1, dataset.max_seq_len), pad_id, dtype=torch.long)
            hist = list(hist_ids_raw)[-dataset.max_seq_len :]
            if hist:
                hist_item_ids_cpu[0, : len(hist)] = torch.tensor(hist, dtype=torch.long)
            
            # 2. 用 CPU 的 ID 去切片 CPU 的表格，然后送进 GPU
            hist_item_text = dataset.item_text_table[hist_item_ids_cpu].cuda()
            
            # 3. 将 ID 本身也送进 GPU，供后续 popularity 查表使用
            hist_item_ids = hist_item_ids_cpu.cuda()
            hist_popularity = popularity[hist_item_ids]
            user_profile_text = dataset.user_text_table[user_idx].unsqueeze(0).cuda()

            score_parts: List[torch.Tensor] = []
            for st in range(0, len(candidates), candidate_batch_size):
                chunk = candidates[st : st + candidate_batch_size]
                # 1. 在 CPU 创建 ID
                target_item_ids_cpu = torch.tensor(chunk, dtype=torch.long)
                
                # 2. 在 CPU 查表并放入 GPU
                target_item_text = dataset.item_text_table[target_item_ids_cpu].cuda()
                
                # 3. 把 ID 放入 GPU
                target_item_ids = target_item_ids_cpu.cuda()
                target_popularity = popularity[target_item_ids]

                bsz = target_item_ids.size(0)
                out = model(
                    target_item_ids=target_item_ids,
                    target_item_text=target_item_text,
                    hist_item_ids=hist_item_ids.expand(bsz, -1),
                    hist_item_text=hist_item_text.expand(bsz, -1, -1),
                    hist_popularity=hist_popularity.expand(bsz, -1),
                    target_item_popularity=target_popularity,
                    user_profile_text=user_profile_text.expand(bsz, -1),
                )
                score_parts.append(out["logits"].detach().cpu())

            scores = torch.cat(score_parts, dim=0)
            target_idx = candidates.index(target_item)
            target_score = scores[target_idx]

            rank_index = int((scores > target_score).sum().item())
            num_neg = max(1, len(candidates) - 1)
            auc = float((scores < target_score).sum().item()) / num_neg

            for k in topks:
                recall_sum[k] += recall_at_k(rank_index, k)
                ndcg_sum[k] += ndcg_at_k(rank_index, k)
            auc_sum += auc
            valid_groups += 1

    denom = max(1, valid_groups)
    metrics: Dict[str, float] = {
        "rank_groups": float(valid_groups),
        "rank_auc": auc_sum / denom,
    }
    for k in topks:
        metrics[f"recall@{k}"] = recall_sum[k] / denom
        metrics[f"ndcg@{k}"] = ndcg_sum[k] / denom
    return metrics


def append_csv_row(path: str, row: Dict[str, float | str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLMRec 训练脚本（支持切分/早停/排序评估）")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)

    parser.add_argument("--item-emb-path", type=str, required=True, help="item 画像向量文件(.pt/.jsonl)")
    parser.add_argument("--user-emb-path", type=str, default="", help="user 画像向量文件(.pt/.jsonl)，可选")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)

    parser.add_argument("--fusion-strategy", type=str, default="gate", choices=["gate", "concat_mlp", "mean"])
    parser.add_argument("--seq-hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--id-dim", type=int, default=128)
    parser.add_argument("--rq-num-layers", type=int, default=3)
    parser.add_argument("--rq-num-centroids", type=int, default=256)

    parser.add_argument("--negative-ratio-train", type=int, default=1)
    parser.add_argument("--negative-ratio-eval", type=int, default=100)
    parser.add_argument("--user-sem-eps", type=float, default=1e-8)
    parser.add_argument("--user-sem-w-max", type=float, default=10.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--early-stop-metric", type=str, default="auc", choices=["auc", "loss"])
    parser.add_argument("--rank-topk", type=str, default="10,20")
    parser.add_argument("--ranking-eval-mode", type=str, default="sampled", choices=["sampled", "full"])
    parser.add_argument("--full-eval-candidate-batch-size", type=int, default=512)

    parser.add_argument("--save-path", type=str, default="checkpoints/best.pt")
    parser.add_argument("--report-path", type=str, default="reports/latest_report.json")
    parser.add_argument("--report-csv", type=str, default="reports/experiments.csv")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def train_and_evaluate(args: argparse.Namespace) -> Dict[str, float]:
    torch.manual_seed(args.seed)
    args.dataset = normalize_dataset_name(args.dataset)
    ranking_eval_mode = getattr(args, "ranking_eval_mode", "sampled")
    full_eval_candidate_batch_size = getattr(args, "full_eval_candidate_batch_size", 512)
    dropout = getattr(args, "dropout", 0.3)
    weight_decay = getattr(args, "weight_decay", 1e-4)
    label_smoothing = getattr(args, "label_smoothing", 0.05)
    grad_clip_norm = getattr(args, "grad_clip_norm", 1.0)

    splits = build_data_splits(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
        padding_item_id=0,
        item_emb_path=args.item_emb_path,
        user_emb_path=args.user_emb_path if args.user_emb_path else None,
        negative_ratio_train=args.negative_ratio_train,
        negative_ratio_eval=args.negative_ratio_eval,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
        max_test_samples=args.max_test_samples,
        user_sem_eps=args.user_sem_eps,
        user_sem_w_max=args.user_sem_w_max,
    )

    cfg = ModelConfig(
        num_users=splits.meta.num_users,
        num_items=splits.meta.num_items,
        padding_item_id=0,
        id_dim=args.id_dim,
        text_dim=splits.meta.text_dim,
        num_subspaces=48,
        subspace_dim=16,
        num_centroids=args.rq_num_centroids,
        seq_hidden_dim=args.seq_hidden_dim,
        fusion_dim=args.seq_hidden_dim,
        fusion_strategy=args.fusion_strategy,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=dropout,
    )

    train_loader = DataLoader(
        splits.train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        splits.valid,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        splits.test,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    # 【关键修复 3】：把模型和基础参数都放到 GPU 上！
    model = RLMRecFramework(cfg).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss().cuda()

    popularity = compute_global_popularity(
        num_items=cfg.num_items,
        item_ids=splits.popularity_item_ids,
        padding_item_id=cfg.padding_item_id,
    ).cuda()

    best_metric = -1e18 if args.early_stop_metric == "auc" else 1e18
    best_epoch = -1
    no_improve = 0
    best_state = None

    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            popularity=popularity,
            bce=bce,
            contrastive_weight=args.contrastive_weight,
            optimizer=optimizer,
            label_smoothing=label_smoothing,
            grad_clip_norm=grad_clip_norm,
        )
        valid_metrics = run_epoch(
            model=model,
            loader=valid_loader,
            popularity=popularity,
            bce=bce,
            contrastive_weight=args.contrastive_weight,
            optimizer=None,
            label_smoothing=0.0,
            grad_clip_norm=0.0,
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_auc={train_metrics['auc']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} valid_auc={valid_metrics['auc']:.4f}"
        )

        current_metric = valid_metrics["auc"] if args.early_stop_metric == "auc" else valid_metrics["loss"]
        improved = (current_metric > best_metric) if args.early_stop_metric == "auc" else (current_metric < best_metric)

        if improved:
            best_metric = current_metric
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"early stop at epoch={epoch}, best_epoch={best_epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        popularity=popularity,
        bce=bce,
        contrastive_weight=args.contrastive_weight,
        optimizer=None,
    )

    topks = [int(x.strip()) for x in args.rank_topk.split(",") if x.strip()]
    if ranking_eval_mode == "full":
        rank_metrics = evaluate_ranking_full(
            model=model,
            dataset=splits.test,
            popularity=popularity,
            topks=topks,
            candidate_batch_size=full_eval_candidate_batch_size,
        )
    else:
        rank_metrics = evaluate_ranking(
            model=model,
            dataset=splits.test,
            popularity=popularity,
            topks=topks,
            num_negatives=args.negative_ratio_eval, 
        )

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(cfg),
            "dataset": args.dataset,
            "data_dir": args.data_dir,
            "item_emb_path": args.item_emb_path,
            "user_emb_path": args.user_emb_path,
            "max_seq_len": args.max_seq_len,
            "best_epoch": best_epoch,
            "test_metrics": test_metrics,
            "rank_metrics": rank_metrics,
        },
        args.save_path,
    )

    result = {
        "best_epoch": float(best_epoch),
        "best_valid_metric": float(best_metric),
        "test_loss": float(test_metrics["loss"]),
        "test_auc": float(test_metrics["auc"]),
        **{k: float(v) for k, v in rank_metrics.items()},
    }

    report = {
        "config": {
            "dataset": args.dataset,
            "data_dir": args.data_dir,
            "item_emb_path": args.item_emb_path,
            "user_emb_path": args.user_emb_path,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "label_smoothing": label_smoothing,
            "grad_clip_norm": grad_clip_norm,
            "contrastive_weight": args.contrastive_weight,
            "fusion_strategy": args.fusion_strategy,
            "seq_hidden_dim": args.seq_hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "id_dim": args.id_dim,
            "rq_num_layers": args.rq_num_layers,
            "rq_num_centroids": args.rq_num_centroids,
            "patience": args.patience,
            "early_stop_metric": args.early_stop_metric,
            "rank_topk": args.rank_topk,
            "ranking_eval_mode": ranking_eval_mode,
            "full_eval_candidate_batch_size": full_eval_candidate_batch_size,
        },
        "result": result,
        "checkpoint": args.save_path,
    }

    report_dir = os.path.dirname(args.report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    csv_row: Dict[str, float | str] = {
        "dataset": args.dataset,
        "lr": args.lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "label_smoothing": label_smoothing,
        "grad_clip_norm": grad_clip_norm,
        "contrastive_weight": args.contrastive_weight,
        "batch_size": args.batch_size,
        "fusion_strategy": args.fusion_strategy,
        "seq_hidden_dim": args.seq_hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "id_dim": args.id_dim,
        "rq_num_layers": args.rq_num_layers,
        "rq_num_centroids": args.rq_num_centroids,
        "best_epoch": result["best_epoch"],
        "best_valid_metric": result["best_valid_metric"],
        "test_loss": result["test_loss"],
        "test_auc": result["test_auc"],
        "rank_auc": result["rank_auc"],
        "checkpoint": args.save_path,
        "report": args.report_path,
    }
    for k in topks:
        csv_row[f"recall@{k}"] = result[f"recall@{k}"]
        csv_row[f"ndcg@{k}"] = result[f"ndcg@{k}"]
    append_csv_row(args.report_csv, csv_row)

    print("result_json=" + json.dumps(result, ensure_ascii=False))
    print(f"checkpoint saved: {args.save_path}")
    print(f"report saved: {args.report_path}")
    print(f"csv appended: {args.report_csv}")
    return result


def main() -> None:
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()