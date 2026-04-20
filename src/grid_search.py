import argparse
import csv
import itertools
import json
import os
from types import SimpleNamespace
from typing import Dict, List

from data import normalize_dataset_name
from train1 import train_and_evaluate


def parse_list_float(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_int(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_str(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def append_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLMRec 超参网格搜索")
    parser.add_argument("--dataset", type=str, default="fashion")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--item-emb-path", type=str, required=True)
    parser.add_argument("--user-emb-path", type=str, default="")

    parser.add_argument("--embedding-size-grid", type=str, default="64,256")
    parser.add_argument("--rq-centroids-grid", type=str, default="8,48,256,512")
    parser.add_argument("--cl-lossweight-grid", type=str, default="1e-4,1e-3,1e-2,0.1,1")
    parser.add_argument("--rq-num-layers", type=int, default=3)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fusion-strategy", type=str, default="gate")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--early-stop-metric", type=str, default="auc", choices=["auc", "loss"])
    parser.add_argument("--rank-topk", type=str, default="10,20")
    parser.add_argument("--ranking-eval-mode", type=str, default="sampled", choices=["sampled", "full"])
    parser.add_argument("--full-eval-candidate-batch-size", type=int, default=512)

    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    parser.add_argument("--max-seq-len", type=int, default=20)
    parser.add_argument("--negative-ratio-train", type=int, default=1)
    parser.add_argument("--negative-ratio-eval", type=int, default=100)
    parser.add_argument("--user-sem-eps", type=float, default=1e-8)
    parser.add_argument("--user-sem-w-max", type=float, default=10.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)

    parser.add_argument("--out-dir", type=str, default="grid_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset = normalize_dataset_name("fashion")
    emb_sizes = parse_list_int(args.embedding_size_grid)
    rq_centroids = parse_list_int(args.rq_centroids_grid)
    cl_weights = parse_list_float(args.cl_lossweight_grid)

    os.makedirs(args.out_dir, exist_ok=True)
    all_results: List[Dict] = []

    combos = list(itertools.product(emb_sizes, rq_centroids, cl_weights))
    print(f"[grid] total_combinations={len(combos)}")

    for run_id, (emb, rq_c, cl_w) in enumerate(combos, start=1):
        save_path = os.path.join(
            args.out_dir,
            f"run_{run_id}_ds{args.dataset}_emb{emb}_rqL{args.rq_num_layers}_rqC{rq_c}_cl{cl_w}.pt",
        )
        report_path = os.path.join(args.out_dir, f"run_{run_id}_report.json")
        train_args = SimpleNamespace(
            dataset=args.dataset,
            data_dir=args.data_dir,
            item_emb_path=args.item_emb_path,
            user_emb_path=args.user_emb_path,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            label_smoothing=args.label_smoothing,
            grad_clip_norm=args.grad_clip_norm,
            contrastive_weight=cl_w,
            fusion_strategy=args.fusion_strategy,
            seq_hidden_dim=emb,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            id_dim=emb,
            rq_num_layers=args.rq_num_layers,
            rq_num_centroids=rq_c,
            negative_ratio_train=args.negative_ratio_train,
            negative_ratio_eval=args.negative_ratio_eval,
            user_sem_eps=args.user_sem_eps,
            user_sem_w_max=args.user_sem_w_max,
            num_workers=args.num_workers,
            max_train_samples=args.max_train_samples,
            max_valid_samples=args.max_valid_samples,
            max_test_samples=args.max_test_samples,
            patience=args.patience,
            early_stop_metric=args.early_stop_metric,
            rank_topk=args.rank_topk,
            ranking_eval_mode=args.ranking_eval_mode,
            full_eval_candidate_batch_size=args.full_eval_candidate_batch_size,
            save_path=save_path,
            report_path=report_path,
            report_csv=os.path.join(args.out_dir, "train_reports.csv"),
            seed=args.seed,
        )

        print(
            f"[grid] run={run_id} dataset={args.dataset} emb={emb} "
            f"rq_layers={args.rq_num_layers} rq_centroids={rq_c} cl_weight={cl_w}"
        )

        metrics = train_and_evaluate(train_args)
        row = {
            "run": run_id,
            "dataset": args.dataset,
            "embedding_size": emb,
            "rq_num_layers": args.rq_num_layers,
            "rq_num_centroids": rq_c,
            "contrastive_weight": cl_w,
            **metrics,
            "checkpoint": save_path,
            "report": report_path,
        }
        all_results.append(row)

    jsonl_path = os.path.join(args.out_dir, "results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in all_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = os.path.join(args.out_dir, "results.csv")
    append_csv(csv_path, all_results)

    if args.early_stop_metric == "auc":
        best = max(all_results, key=lambda x: x["best_valid_metric"])
    else:
        best = min(all_results, key=lambda x: x["best_valid_metric"])

    summary_path = os.path.join(args.out_dir, "best_result.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print("best=" + json.dumps(best, ensure_ascii=False))
    print(f"jsonl saved: {jsonl_path}")
    print(f"csv saved: {csv_path}")
    print(f"summary saved: {summary_path}")


if __name__ == "__main__":
    main()
