import argparse
import json
import os
from typing import Dict, List

import torch
from transformers import AutoModel, AutoTokenizer


def load_profile_texts(path: str, id_key: str) -> Dict[str, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到输入文件: {path}")

    result: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
                rid = str(obj[id_key])
            text = f"{obj.get('summarization', '')}\n{obj.get('reasoning', '')}".strip()
            result[rid] = text if text else "None"
    return result


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


def batch_encode_texts(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int,
    pooling: str,
    device: torch.device,
) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        out = model(**encoded)
        if pooling == "cls":
            emb = out.last_hidden_state[:, 0, :]
        else:
            emb = mean_pool(out.last_hidden_state, encoded["attention_mask"])
    return emb.cpu()


def save_embeddings_pt(path: str, emb_map: Dict[str, List[float]], dim: int, meta: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {
        "dim": dim,
        "embeddings": emb_map,
        "meta": meta,
    }
    torch.save(obj, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用BERT将画像文本编码为向量")
    parser.add_argument("--input", type=str, required=True, help="item_profiles.jsonl 或 user_profiles.jsonl")
    parser.add_argument("--id-key", type=str, required=True, choices=["item_id", "user_id"])
    parser.add_argument("--output", type=str, required=True, help="输出 .pt")

    parser.add_argument("--model-name", type=str, default="bert-base-chinese")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128, choices=[128, 256])
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    id_to_text = load_profile_texts(args.input, args.id_key)
    ids = sorted(id_to_text.keys())

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.eval()
    model.to(device)

    hidden_size = int(model.config.hidden_size)
    if hidden_size != 768:
        print(f"[warn] 当前模型 hidden_size={hidden_size}，若需 48x16 量化，建议使用 768 维模型。")

    emb_map: Dict[str, List[float]] = {}
    total = len(ids)

    for i in range(0, total, args.batch_size):
        batch_ids = ids[i : i + args.batch_size]
        batch_texts = [id_to_text[x] for x in batch_ids]

        emb = batch_encode_texts(
            model=model,
            tokenizer=tokenizer,
            texts=batch_texts,
            max_length=args.max_length,
            pooling=args.pooling,
            device=device,
        )

        for rid, vec in zip(batch_ids, emb):
            emb_map[rid] = vec.tolist()

        if ((i // args.batch_size) + 1) % 20 == 0:
            print(f"progress {min(i + args.batch_size, total)}/{total}")

    save_embeddings_pt(
        path=args.output,
        emb_map=emb_map,
        dim=hidden_size,
        meta={
            "model_name": args.model_name,
            "max_length": args.max_length,
            "pooling": args.pooling,
        },
    )
    print(f"saved embeddings: {args.output}")
    print(f"dim: {hidden_size}")


if __name__ == "__main__":
    main()
