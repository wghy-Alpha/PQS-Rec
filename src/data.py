import csv
import gzip
import json
import os
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


SUPPORTED_DATASETS = ["ml100k", "ml25m", "amazon_all_beauty", "amazon_fashion", "yelp"]

DATASET_ALIASES = {
    "ml100k": "ml100k",
    "ml-100k": "ml100k",
    "ml25m": "ml25m",
    "ml-25m": "ml25m",
    "amazon_all_beauty": "amazon_all_beauty",
    "amazonallbeauty": "amazon_all_beauty",
    "all_beauty": "amazon_all_beauty",
    "amazon_beauty": "amazon_all_beauty",
    "amazon_fashion": "amazon_fashion",
    "amazonfashion": "amazon_fashion",
    "fashion": "amazon_fashion",
    "amazon_fasion": "amazon_fashion",
    "amazonfasion": "amazon_fashion",
    "yelp": "yelp",
}


def normalize_dataset_name(dataset_name: str) -> str:
    key = dataset_name.strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"不支持的数据集: {dataset_name}，可选: {', '.join(SUPPORTED_DATASETS)}")
    return DATASET_ALIASES[key]


@dataclass
class RecSample:
    user_id: torch.Tensor
    user_profile_text: torch.Tensor
    target_item_id: torch.Tensor
    target_item_text: torch.Tensor
    hist_item_ids: torch.Tensor
    hist_item_text: torch.Tensor
    label: torch.Tensor


@dataclass
class DatasetMeta:
    num_users: int
    num_items: int
    text_dim: int
    name: str


@dataclass
class DataSplits:
    train: Dataset
    valid: Dataset
    test: Dataset
    meta: DatasetMeta
    popularity_item_ids: torch.Tensor


class SequenceRecDataset(Dataset):
    """序列推荐数据集，支持在线负采样。"""

    def __init__(
        self,
        positive_samples: List[Tuple[int, Tuple[int, ...], int]],
        item_text_table: torch.Tensor,
        user_text_table: torch.Tensor,
        user_seen_items: List[set[int]],
        max_seq_len: int,
        padding_item_id: int,
        negative_ratio: int,
        seed: int = 2026,
    ) -> None:
        super().__init__()
        self.positive_samples = positive_samples
        self.item_text_table = item_text_table
        self.user_text_table = user_text_table
        self.user_seen_items = user_seen_items
        self.max_seq_len = max_seq_len
        self.padding_item_id = padding_item_id
        self.negative_ratio = max(0, negative_ratio)
        self.num_items = item_text_table.size(0)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.positive_samples) * (1 + self.negative_ratio)

    def _sample_negative_item(self, user_idx: int, positive_item: int) -> int:
        seen = self.user_seen_items[user_idx]
        for _ in range(50):
            neg = self.rng.randint(1, self.num_items - 1)
            if neg not in seen:
                return neg
        neg = self.rng.randint(1, self.num_items - 1)
        if neg == positive_item and self.num_items > 2:
            neg = 1 if positive_item != 1 else 2
        return neg

    def __getitem__(self, idx: int) -> RecSample:
        pos_index = idx // (1 + self.negative_ratio)
        offset = idx % (1 + self.negative_ratio)

        user_idx, hist_ids_raw, pos_target = self.positive_samples[pos_index]
        is_positive = offset == 0
        target_item = pos_target if is_positive else self._sample_negative_item(user_idx, pos_target)
        label = 1.0 if is_positive else 0.0

        hist_item_ids = torch.full((self.max_seq_len,), self.padding_item_id, dtype=torch.long)
        hist = list(hist_ids_raw)[-self.max_seq_len :]
        if hist:
            hist_item_ids[: len(hist)] = torch.tensor(hist, dtype=torch.long)

        target_item_id = torch.tensor([target_item], dtype=torch.long)
        target_item_text = self.item_text_table[target_item]
        hist_item_text = self.item_text_table[hist_item_ids]
        user_profile_text = self.user_text_table[user_idx]

        return RecSample(
            user_id=torch.tensor([user_idx], dtype=torch.long),
            user_profile_text=user_profile_text,
            target_item_id=target_item_id,
            target_item_text=target_item_text,
            hist_item_ids=hist_item_ids,
            hist_item_text=hist_item_text,
            label=torch.tensor([label], dtype=torch.float32),
        )


def compute_global_popularity(num_items: int, item_ids: torch.Tensor, padding_item_id: int = 0) -> torch.Tensor:
    valid = item_ids[item_ids.ne(padding_item_id)]
    counts = torch.zeros(num_items, dtype=torch.float32)
    if valid.numel() > 0:
        unique, freq = torch.unique(valid, return_counts=True)
        counts[unique] = freq.float()
    total = counts.sum().clamp_min(1.0)
    return counts / total


def _open_text_maybe_gzip(path: str):
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _find_existing_file(data_dir: str, candidates: List[str]) -> str:
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"未找到文件，候选: {candidates}")


def _iter_records(path: str) -> Iterable[Dict]:
    lower = path.lower()
    if lower.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
        return

    with _open_text_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _parse_ts(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        ts = int(value)
        return ts // 1000 if ts > 10**12 else ts

    text = str(value).strip()
    if not text:
        return 0
    if text.isdigit():
        ts = int(text)
        return ts // 1000 if ts > 10**12 else ts

    # Yelp 常见日期格式：YYYY-MM-DD HH:MM:SS
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
        try:
            dt = datetime.strptime(text, fmt)
            return int(dt.timestamp())
        except ValueError:
            continue
    return 0


def _collect_item_ids_from_events(user_events: Dict[str, List[Tuple[int, str]]]) -> List[str]:
    item_ids = set()
    for events in user_events.values():
        for _, iid in events:
            item_ids.add(iid)
    return sorted(item_ids)


def _extract_item_id_from_record(rec: Dict) -> Optional[str]:
    for key in ["movieId", "asin", "parent_asin", "item_id", "business_id"]:
        val = rec.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def _extract_user_item_ts_rating(rec: Dict) -> Tuple[Optional[str], Optional[str], int, float]:
    uid = None
    for k in ["userId", "reviewerID", "user_id"]:
        if rec.get(k) is not None and str(rec.get(k)).strip():
            uid = str(rec.get(k)).strip()
            break

    iid = None
    for k in ["movieId", "asin", "parent_asin", "item_id", "business_id"]:
        if rec.get(k) is not None and str(rec.get(k)).strip():
            iid = str(rec.get(k)).strip()
            break

    ts_val = None
    for k in ["timestamp", "unixReviewTime", "time", "date"]:
        if rec.get(k) is not None:
            ts_val = rec.get(k)
            break
    ts = _parse_ts(ts_val)

    rating_val = None
    for k in ["rating", "overall", "stars"]:
        if rec.get(k) is not None:
            rating_val = rec.get(k)
            break
    try:
        rating = float(rating_val) if rating_val is not None else 1.0
    except (TypeError, ValueError):
        rating = 1.0

    return uid, iid, ts, rating


def _load_movie_ids(dataset_name: str, data_dir: str) -> List[str]:
    if dataset_name == "ml100k":
        movie_file = os.path.join(data_dir, "u.item")
        if not os.path.exists(movie_file):
            raise FileNotFoundError(f"未找到文件: {movie_file}")
        item_ids: List[str] = []
        with open(movie_file, "r", encoding="latin-1") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) >= 2:
                    item_ids.append(str(parts[0]).strip())
        return sorted(set(item_ids))

    if dataset_name == "ml25m":
        movie_file = os.path.join(data_dir, "movies.csv")
        if not os.path.exists(movie_file):
            raise FileNotFoundError(f"未找到文件: {movie_file}")
        item_ids = []
        with open(movie_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item_ids.append(str(row["movieId"]).strip())
        return sorted(set(item_ids))

    if dataset_name == "amazon_all_beauty":
        meta_file = _find_existing_file(
            data_dir,
            [
                "meta_All_Beauty.jsonl",
                "meta_All_Beauty.jsonl.gz",
                "meta_All_Beauty.json",
                "meta_All_Beauty.json.gz",
            ],
        )
    elif dataset_name == "amazon_fashion":
        meta_file = _find_existing_file(
            data_dir,
            [
                "meta_Amazon_Fashion.jsonl",
                "meta_Amazon_Fashion.jsonl.gz",
                "meta_AMAZON_FASHION.json",
                "meta_AMAZON_FASHION.json.gz",
            ],
        )
    else:
        meta_file = _find_existing_file(
            data_dir,
            [
                "yelp_business.json",
                "yelp_academic_dataset_business.json",
                "business.json",
            ],
        )

    item_ids = []
    for rec in _iter_records(meta_file):
        iid = _extract_item_id_from_record(rec)
        if iid is not None:
            item_ids.append(iid)
    return sorted(set(item_ids))


def _load_user_events(dataset_name: str, data_dir: str) -> Dict[str, List[Tuple[int, str]]]:
    user_events: Dict[str, List[Tuple[int, str]]] = {}
    if dataset_name == "ml100k":
        rating_file = os.path.join(data_dir, "u.data")
        if not os.path.exists(rating_file):
            raise FileNotFoundError(f"未找到文件: {rating_file}")
        with open(rating_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 4:
                    continue
                uid = str(parts[0]).strip()
                iid = str(parts[1]).strip()
                ts = int(parts[3])
                user_events.setdefault(uid, []).append((ts, iid))
    elif dataset_name == "ml25m":
        rating_file = os.path.join(data_dir, "ratings.csv")
        if not os.path.exists(rating_file):
            raise FileNotFoundError(f"未找到文件: {rating_file}")
        with open(rating_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = str(row["userId"]).strip()
                iid = str(row["movieId"]).strip()
                ts = int(row["timestamp"])
                user_events.setdefault(uid, []).append((ts, iid))
    else:
        if dataset_name == "amazon_all_beauty":
            review_file = _find_existing_file(
                data_dir,
                [
                    "All_Beauty.jsonl",
                    "All_Beauty.jsonl.gz",
                    "All_Beauty_5.json",
                    "All_Beauty_5.json.gz",
                    "reviews_All_Beauty.jsonl",
                    "reviews_All_Beauty.jsonl.gz",
                ],
            )
        elif dataset_name == "amazon_fashion":
            review_file = _find_existing_file(
                data_dir,
                [
                    "Amazon_Fashion.jsonl",
                    "AMAZON_FASHION.json",
                    "Amazon_Fashion.jsonl.gz",
                    "AMAZON_FASHION_5.json",
                    "AMAZON_FASHION_5.json.gz",
                    "reviews_Amazon_Fashion.jsonl",
                    "reviews_Amazon_Fashion.jsonl.gz",
                ],
            )
        else:
            review_file = _find_existing_file(
                data_dir,
                [
                    "yelp_review.json",
                    "yelp_academic_dataset_review.json",
                    "review.json",
                ],
            )

        for rec in _iter_records(review_file):
            uid, iid, ts, _ = _extract_user_item_ts_rating(rec)
            if uid is None or iid is None:
                continue
            user_events.setdefault(uid, []).append((ts, iid))

    for uid in user_events:
        user_events[uid].sort(key=lambda x: x[0])
    return user_events


def _load_embedding_map(path: str, key_name: str) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    支持两种格式：
    1) torch.save({"dim": 768, "embeddings": {raw_id: [..]}})
    2) JSONL 每行包含 key_name 和 embedding 字段
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 embedding 文件: {path}")

    if path.lower().endswith(".pt") or path.lower().endswith(".pth"):
        obj = torch.load(path, map_location="cpu")
        mp_raw = obj.get("embeddings", {})
        dim = int(obj.get("dim", 0))
        mp: Dict[str, torch.Tensor] = {}
        for k, v in mp_raw.items():
            key = str(k)
            tensor = torch.tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v.float()
            mp[key] = tensor
        if dim <= 0 and mp:
            dim = int(next(iter(mp.values())).numel())
        if dim <= 0:
            raise ValueError("embedding 维度无法识别")
        return mp, dim

    mp = {}
    dim = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw_id = str(obj[key_name])
            emb = torch.tensor(obj["embedding"], dtype=torch.float32)
            mp[raw_id] = emb
            if dim == 0:
                dim = int(emb.numel())
    if dim == 0:
        raise ValueError("embedding 文件为空或格式不正确")
    return mp, dim


def _build_tables_from_embeddings(
    all_user_ids: List[str],
    all_item_ids: List[str],
    item_emb_path: str,
    user_emb_path: Optional[str],
    padding_item_id: int,
) -> Tuple[Dict[str, int], Dict[str, int], torch.Tensor, torch.Tensor, int]:
    user2idx = {uid: idx for idx, uid in enumerate(all_user_ids)}
    item2idx = {iid: idx + 1 for idx, iid in enumerate(all_item_ids)}

    item_mp, text_dim = _load_embedding_map(item_emb_path, key_name="item_id")
    item_table = torch.zeros(len(item2idx) + 1, text_dim, dtype=torch.float32)
    item_table[padding_item_id] = 0.0

    missing_items = 0
    for raw_iid, internal_iid in item2idx.items():
        emb = item_mp.get(raw_iid)
        if emb is None:
            missing_items += 1
            continue
        if emb.numel() != text_dim:
            raise ValueError(f"item_id={raw_iid} 的 embedding 维度不一致")
        item_table[internal_iid] = emb

    if missing_items > 0:
        print(f"[warn] {missing_items} 个 item 缺少画像向量，已置零")

    user_table = torch.zeros(len(user2idx), text_dim, dtype=torch.float32)
    if user_emb_path:
        user_mp, user_dim = _load_embedding_map(user_emb_path, key_name="user_id")
        if user_dim != text_dim:
            raise ValueError("user embedding 维度与 item embedding 维度不一致")
        missing_users = 0
        for raw_uid, internal_uid in user2idx.items():
            emb = user_mp.get(raw_uid)
            if emb is None:
                missing_users += 1
                continue
            user_table[internal_uid] = emb
        if missing_users > 0:
            print(f"[warn] {missing_users} 个 user 缺少画像向量，已置零")

    return user2idx, item2idx, item_table, user_table, text_dim


def _build_user_table_from_history_mean(
    user_events: Dict[str, List[Tuple[int, str]]],
    user2idx: Dict[str, int],
    item2idx: Dict[str, int],
    item_table: torch.Tensor,
    text_dim: int,
    item_popularity: torch.Tensor,
    eps: float,
    w_max: float,
) -> torch.Tensor:
    """
    根据用户交互历史，将交互过的 item embedding 按流行度平滑加权后聚合。
    权重公式与 PPT 一致：w_i = min(log(1 / (p_i + eps)), w_max)。
    """
    user_table = torch.zeros(len(user2idx), text_dim, dtype=torch.float32)
    cap = torch.tensor(w_max, dtype=torch.float32)

    for raw_uid, events in user_events.items():
        if raw_uid not in user2idx:
            continue
        u = user2idx[raw_uid]
        internal_items: List[int] = []
        for _, raw_iid in events:
            iid = item2idx.get(raw_iid)
            if iid is not None:
                internal_items.append(iid)

        if not internal_items:
            continue

        item_ids = torch.tensor(internal_items, dtype=torch.long)
        item_embs = item_table[item_ids]

        # 按物品全局流行度计算平滑权重，强化长尾物品贡献
        p_i = item_popularity[item_ids]
        weights = torch.minimum(torch.log(1.0 / (p_i + eps)), cap)
        weights = weights / weights.sum().clamp_min(eps)

        user_table[u] = (item_embs * weights.unsqueeze(-1)).sum(dim=0)
    return user_table


def _compute_item_popularity_from_events(
    user_events: Dict[str, List[Tuple[int, str]]],
    item2idx: Dict[str, int],
    num_items: int,
    padding_item_id: int,
) -> torch.Tensor:
    interacted: List[int] = []
    for events in user_events.values():
        for _, raw_iid in events:
            iid = item2idx.get(raw_iid)
            if iid is not None:
                interacted.append(iid)

    if not interacted:
        return torch.zeros(num_items, dtype=torch.float32)

    item_ids = torch.tensor(interacted, dtype=torch.long)
    return compute_global_popularity(num_items=num_items, item_ids=item_ids, padding_item_id=padding_item_id)


def _build_train_visible_user_events(
    user_events: Dict[str, List[Tuple[int, str]]],
    item2idx: Dict[str, int],
) -> Dict[str, List[Tuple[int, str]]]:
    """仅保留每个用户在训练阶段可见的交互事件（按时间切分规则截断）。"""
    train_visible_events: Dict[str, List[Tuple[int, str]]] = {}

    for raw_uid, events in user_events.items():
        filtered_events = [(ts, raw_iid) for ts, raw_iid in events if raw_iid in item2idx]
        if len(filtered_events) < 2:
            continue

        if len(filtered_events) >= 4:
            # valid/test 分别占倒数第二和倒数第一
            train_events = filtered_events[:-2]
        elif len(filtered_events) == 3:
            # test 占倒数第一
            train_events = filtered_events[:-1]
        else:
            train_events = filtered_events

        if train_events:
            train_visible_events[raw_uid] = train_events

    return train_visible_events


def _build_time_splits(
    user_events: Dict[str, List[Tuple[int, str]]],
    user2idx: Dict[str, int],
    item2idx: Dict[str, int],
    max_seq_len: int,
) -> Tuple[
    List[Tuple[int, Tuple[int, ...], int]],
    List[Tuple[int, Tuple[int, ...], int]],
    List[Tuple[int, Tuple[int, ...], int]],
    List[set[int]],
    List[set[int]],
    torch.Tensor,
]:
    train_samples: List[Tuple[int, Tuple[int, ...], int]] = []
    valid_samples: List[Tuple[int, Tuple[int, ...], int]] = []
    test_samples: List[Tuple[int, Tuple[int, ...], int]] = []
    user_seen_items_train: List[set[int]] = [set() for _ in range(len(user2idx))]
    user_seen_items_all: List[set[int]] = [set() for _ in range(len(user2idx))]

    popularity_items: List[int] = []

    for raw_uid, events in user_events.items():
        if raw_uid not in user2idx:
            continue
        u = user2idx[raw_uid]
        seq = [item2idx[iid] for _, iid in events if iid in item2idx]
        if len(seq) < 2:
            continue

        user_seen_items_all[u].update(seq)
        if len(seq) >= 4:
            user_seen_items_train[u].update(seq[:-2])
        elif len(seq) == 3:
            user_seen_items_train[u].update(seq[:-1])
        else:
            user_seen_items_train[u].update(seq)

        # train/valid/test 时间切分
        # len>=4: train(前面), valid(倒数第二), test(倒数第一)
        # len=3: train(第2个), test(第3个), valid为空
        # len=2: 仅 train
        if len(seq) >= 4:
            for t in range(1, len(seq) - 2):
                hist = tuple(seq[max(0, t - max_seq_len) : t])
                target = seq[t]
                train_samples.append((u, hist, target))
                popularity_items.append(target)

            t_valid = len(seq) - 2
            hist_valid = tuple(seq[max(0, t_valid - max_seq_len) : t_valid])
            valid_samples.append((u, hist_valid, seq[t_valid]))

            t_test = len(seq) - 1
            hist_test = tuple(seq[max(0, t_test - max_seq_len) : t_test])
            test_samples.append((u, hist_test, seq[t_test]))

        elif len(seq) == 3:
            train_samples.append((u, tuple(seq[:1]), seq[1]))
            popularity_items.append(seq[1])
            test_samples.append((u, tuple(seq[:2]), seq[2]))
        else:
            train_samples.append((u, tuple(seq[:1]), seq[1]))
            popularity_items.append(seq[1])

    if len(train_samples) == 0:
        raise ValueError("未构建出训练样本，请检查数据或切分规则")

    pop_tensor = torch.tensor(popularity_items if popularity_items else [0], dtype=torch.long)
    return train_samples, valid_samples, test_samples, user_seen_items_train, user_seen_items_all, pop_tensor


def build_data_splits(
    dataset_name: str,
    data_dir: Optional[str],
    max_seq_len: int,
    padding_item_id: int,
    item_emb_path: Optional[str],
    user_emb_path: Optional[str],
    negative_ratio_train: int = 1,
    negative_ratio_eval: int = 1,
    seed: int = 2026,
    max_train_samples: int = 0,
    max_valid_samples: int = 0,
    max_test_samples: int = 0,
    user_sem_eps: float = 1e-8,
    user_sem_w_max: float = 10.0,
) -> DataSplits:
    dataset_key = normalize_dataset_name(dataset_name)
    if dataset_key not in set(SUPPORTED_DATASETS):
        raise ValueError(f"当前切分版数据管线仅支持: {', '.join(SUPPORTED_DATASETS)}")
    if not data_dir:
        raise ValueError("必须提供 data_dir")
    if not item_emb_path:
        raise ValueError("必须提供 item_emb_path（LLM画像向量文件）")

    user_events = _load_user_events(dataset_key, data_dir)
    all_user_ids = sorted(user_events.keys())
    all_item_ids = sorted(set(_load_movie_ids(dataset_key, data_dir)) | set(_collect_item_ids_from_events(user_events)))

    user2idx, item2idx, item_table, user_table, text_dim = _build_tables_from_embeddings(
        all_user_ids=all_user_ids,
        all_item_ids=all_item_ids,
        item_emb_path=item_emb_path,
        user_emb_path=user_emb_path,
        padding_item_id=padding_item_id,
    )

    # 新默认策略：用户语义向量 = 历史交互物品 embedding 均值。
    # 仅当显式提供 user_emb_path 时，才使用外部用户向量。
    # 否则仅基于训练阶段可见历史做平滑加权聚合，避免 test 泄露。
    if not user_emb_path:
        train_visible_events = _build_train_visible_user_events(user_events=user_events, item2idx=item2idx)
        item_popularity = _compute_item_popularity_from_events(
            user_events=train_visible_events,
            item2idx=item2idx,
            num_items=len(item2idx) + 1,
            padding_item_id=padding_item_id,
        )
        user_table = _build_user_table_from_history_mean(
            user_events=train_visible_events,
            user2idx=user2idx,
            item2idx=item2idx,
            item_table=item_table,
            text_dim=text_dim,
            item_popularity=item_popularity,
            eps=user_sem_eps,
            w_max=user_sem_w_max,
        )

    train_samples, valid_samples, test_samples, user_seen_items_train, user_seen_items_all, pop_tensor = _build_time_splits(
        user_events=user_events,
        user2idx=user2idx,
        item2idx=item2idx,
        max_seq_len=max_seq_len,
    )

    rng = random.Random(seed)
    if max_train_samples > 0 and len(train_samples) > max_train_samples:
        train_samples = rng.sample(train_samples, max_train_samples)
    if max_valid_samples > 0 and len(valid_samples) > max_valid_samples:
        valid_samples = rng.sample(valid_samples, max_valid_samples)
    if max_test_samples > 0 and len(test_samples) > max_test_samples:
        test_samples = rng.sample(test_samples, max_test_samples)

    train_ds = SequenceRecDataset(
        positive_samples=train_samples,
        item_text_table=item_table,
        user_text_table=user_table,
        user_seen_items=user_seen_items_train,
        max_seq_len=max_seq_len,
        padding_item_id=padding_item_id,
        negative_ratio=negative_ratio_train,
        seed=seed,
    )
    valid_ds = SequenceRecDataset(
        positive_samples=valid_samples,
        item_text_table=item_table,
        user_text_table=user_table,
        user_seen_items=user_seen_items_all,
        max_seq_len=max_seq_len,
        padding_item_id=padding_item_id,
        negative_ratio=negative_ratio_eval,
        seed=seed + 7,
    )
    test_ds = SequenceRecDataset(
        positive_samples=test_samples,
        item_text_table=item_table,
        user_text_table=user_table,
        user_seen_items=user_seen_items_all,
        max_seq_len=max_seq_len,
        padding_item_id=padding_item_id,
        negative_ratio=negative_ratio_eval,
        seed=seed + 13,
    )

    meta = DatasetMeta(
        num_users=len(user2idx),
        num_items=len(item2idx) + 1,
        text_dim=text_dim,
        name=dataset_key,
    )
    return DataSplits(train=train_ds, valid=valid_ds, test=test_ds, meta=meta, popularity_item_ids=pop_tensor)


# 兼容旧接口（返回 train split）
def build_dataset(
    dataset_name: str,
    text_dim: int,
    max_seq_len: int,
    padding_item_id: int,
    data_dir: str | None = None,
    max_positive_samples: int = 200000,
    negative_ratio: int = 1,
):
    _ = dataset_name
    _ = text_dim
    _ = max_seq_len
    _ = padding_item_id
    _ = data_dir
    _ = max_positive_samples
    _ = negative_ratio
    raise ValueError(
        "build_dataset 旧接口已弃用。请改用 build_data_splits，并传入 item_emb_path/user_emb_path。"
    )


def collate_fn(batch: List[RecSample]) -> Dict[str, torch.Tensor]:
    return {
        "user_ids": torch.cat([b.user_id for b in batch], dim=0),
        "user_profile_text": torch.stack([b.user_profile_text for b in batch], dim=0),
        "target_item_ids": torch.cat([b.target_item_id for b in batch], dim=0),
        "target_item_text": torch.stack([b.target_item_text for b in batch], dim=0),
        "hist_item_ids": torch.stack([b.hist_item_ids for b in batch], dim=0),
        "hist_item_text": torch.stack([b.hist_item_text for b in batch], dim=0),
        "labels": torch.cat([b.label for b in batch], dim=0),
    }
