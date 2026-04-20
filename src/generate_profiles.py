import argparse
import csv
import gzip
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional, Tuple
from urllib import error, request

from data import normalize_dataset_name


@dataclass
class ItemInfo:
    raw_item_id: str
    title: str
    genres: str
    description: str = ""


def _open_text_maybe_gzip(path: str):
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _find_existing_file(data_dir: str, candidates: List[str]) -> str:
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"缺少文件，候选: {candidates}")


def _iter_json_records(path: str):
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

    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
        try:
            return int(datetime.strptime(text, fmt).timestamp())
        except ValueError:
            continue
    return 0


class OpenAICompatibleClient:
    """调用本地 OpenAI 兼容接口（如 vLLM/sglang/ollama-openai）。"""

    def __init__(self, api_base: str, model: str, api_key: str = "EMPTY", timeout: int = 120) -> None:
        self.url = api_base.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 400,
        retry: int = 3,
    ) -> Dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        for i in range(retry):
            try:
                req = request.Request(self.url, data=body, headers=headers, method="POST")
                with request.urlopen(req, timeout=self.timeout) as resp:
                    text = resp.read().decode("utf-8")
                data = json.loads(text)
                content = data["choices"][0]["message"]["content"]
                parsed = self._extract_json(content)
                if parsed is None:
                    raise ValueError("模型输出不是有效 JSON")
                return parsed
            except (error.HTTPError, error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
                if i == retry - 1:
                    raise
                time.sleep(1.2 * (i + 1))

        raise RuntimeError("chat_json unexpected fallthrough")

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """尽量从模型输出中提取 JSON 对象。"""
        text = text.strip()
        # 直接整体是 JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # 尝试抓取首个 {...}
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None


def load_ml100k_items(data_dir: str) -> Dict[str, ItemInfo]:
    movie_file = os.path.join(data_dir, "u.item")
    if not os.path.exists(movie_file):
        raise FileNotFoundError(f"缺少文件: {movie_file}")

    genre_names = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    items: Dict[str, ItemInfo] = {}
    with open(movie_file, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 24:
                continue
            iid = str(parts[0]).strip()
            title = parts[1]
            flags = parts[-19:]
            genres = [genre_names[i] for i, v in enumerate(flags) if v == "1"]
            genre_text = "|".join(genres) if genres else "unknown"
            items[iid] = ItemInfo(raw_item_id=iid, title=title, genres=genre_text, description="")
    return items


def load_ml25m_items(data_dir: str) -> Dict[str, ItemInfo]:
    movie_file = os.path.join(data_dir, "movies.csv")
    if not os.path.exists(movie_file):
        raise FileNotFoundError(f"缺少文件: {movie_file}")

    items: Dict[str, ItemInfo] = {}
    with open(movie_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = str(row["movieId"]).strip()
            title = row.get("title", "")
            genres = row.get("genres", "")
            items[iid] = ItemInfo(raw_item_id=iid, title=title, genres=genres, description="")
    return items


def load_ml100k_user_interactions(data_dir: str) -> Dict[str, List[Tuple[int, str, float]]]:
    rating_file = os.path.join(data_dir, "u.data")
    if not os.path.exists(rating_file):
        raise FileNotFoundError(f"缺少文件: {rating_file}")

    user_events: Dict[str, List[Tuple[int, str, float]]] = {}
    with open(rating_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            uid = str(parts[0]).strip()
            iid = str(parts[1]).strip()
            ts = int(parts[3])
            rating = float(parts[2])
            user_events.setdefault(uid, []).append((ts, iid, rating))

    for uid in user_events:
        user_events[uid].sort(key=lambda x: x[0])
    return user_events


def load_ml25m_user_interactions(data_dir: str) -> Dict[str, List[Tuple[int, str, float]]]:
    rating_file = os.path.join(data_dir, "ratings.csv")
    if not os.path.exists(rating_file):
        raise FileNotFoundError(f"缺少文件: {rating_file}")

    user_events: Dict[str, List[Tuple[int, str, float]]] = {}
    with open(rating_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = str(row["userId"]).strip()
            iid = str(row["movieId"]).strip()
            ts = int(row["timestamp"])
            rating = float(row.get("rating", 0.0))
            user_events.setdefault(uid, []).append((ts, iid, rating))

    for uid in user_events:
        user_events[uid].sort(key=lambda x: x[0])
    return user_events


def _extract_amazon_item_meta(rec: Dict) -> Tuple[str, str, str, str]:
    iid = str(rec.get("parent_asin") or rec.get("asin") or rec.get("item_id") or "").strip()
    
    # 同样对 title 做截断，防止某些极其离谱的标题
    title = str(rec.get("title") or rec.get("name") or "").strip()[:200]

    categories = rec.get("categories")
    if isinstance(categories, list):
        flat = []
        for c in categories:
            if isinstance(c, list):
                flat.extend([str(x).strip() for x in c if str(x).strip()])
            elif str(c).strip():
                flat.append(str(c).strip())
        genres = "|".join(flat[:8]) if flat else "unknown"
    else:
        genres = str(rec.get("category") or "unknown").strip() or "unknown"

    desc = rec.get("description")
    if isinstance(desc, list):
        description = " ".join(str(x).strip() for x in desc if str(x).strip())
    else:
        description = str(desc or "").strip()
        
    if not description:
        features = rec.get("features")
        if isinstance(features, list):
            description = " ".join(str(x).strip() for x in features if str(x).strip())

    # 强制截断 description 并清理可能导致 JSON 解析失败的换行
    description = str(description or "").replace("\n", " ")[:500]
    
    return iid, title, genres, description


def load_amazon_items(data_dir: str, dataset_key: str) -> Dict[str, ItemInfo]:
    if dataset_key == "amazon_all_beauty":
        meta_file = _find_existing_file(
            data_dir,
            [
                "meta_All_Beauty.jsonl",
                "meta_All_Beauty.jsonl.gz",
                "meta_All_Beauty.json",
                "meta_All_Beauty.json.gz",
            ],
        )
    else:
        meta_file = _find_existing_file(
            data_dir,
            [
                "meta_Amazon_Fashion.jsonl",
                "meta_Amazon_Fashion.jsonl.gz",
                "meta_AMAZON_FASHION.json",
                "meta_AMAZON_FASHION.json.gz",
            ],
        )

    items: Dict[str, ItemInfo] = {}
    for rec in _iter_json_records(meta_file):
        iid, title, genres, description = _extract_amazon_item_meta(rec)
        if not iid:
            continue
        items[iid] = ItemInfo(raw_item_id=iid, title=title or iid, genres=genres, description=description)
    return items


def load_amazon_user_interactions(data_dir: str, dataset_key: str) -> Dict[str, List[Tuple[int, str, float]]]:
    if dataset_key == "amazon_all_beauty":
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
    else:
        review_file = _find_existing_file(
            data_dir,
            [
                "Amazon_Fashion.jsonl",
                "Amazon_Fashion.jsonl.gz",
                "AMAZON_FASHION_5.json",
                "AMAZON_FASHION_5.json.gz",
                "reviews_Amazon_Fashion.jsonl",
                "reviews_Amazon_Fashion.jsonl.gz",
            ],
        )

    user_events: Dict[str, List[Tuple[int, str, float]]] = {}
    for rec in _iter_json_records(review_file):
        uid = str(rec.get("user_id") or rec.get("reviewerID") or "").strip()
        iid = str(rec.get("parent_asin") or rec.get("asin") or rec.get("item_id") or "").strip()
        if not uid or not iid:
            continue
        ts = _parse_ts(rec.get("timestamp") or rec.get("unixReviewTime") or rec.get("time"))
        try:
            rating = float(rec.get("rating") or rec.get("overall") or 0.0)
        except (TypeError, ValueError):
            rating = 0.0
        user_events.setdefault(uid, []).append((ts, iid, rating))

    for uid in user_events:
        user_events[uid].sort(key=lambda x: x[0])
    return user_events


def load_yelp_items(data_dir: str) -> Dict[str, ItemInfo]:
    business_file = _find_existing_file(
        data_dir,
        [
            "yelp_business.json",
            "yelp_academic_dataset_business.json",
            "business.json",
        ],
    )

    items: Dict[str, ItemInfo] = {}
    for rec in _iter_json_records(business_file):
        iid = str(rec.get("business_id") or "").strip()
        if not iid:
            continue
        title = str(rec.get("name") or iid).strip()
        genres = str(rec.get("categories") or "unknown").strip() or "unknown"
        city = str(rec.get("city") or "").strip()
        state = str(rec.get("state") or "").strip()
        stars = rec.get("stars")
        review_count = rec.get("review_count")
        description = f"city={city}; state={state}; stars={stars}; review_count={review_count}"
        items[iid] = ItemInfo(raw_item_id=iid, title=title, genres=genres, description=description)
    return items


def load_yelp_user_interactions(data_dir: str) -> Dict[str, List[Tuple[int, str, float]]]:
    review_file = _find_existing_file(
        data_dir,
        [
            "yelp_review.json",
            "yelp_academic_dataset_review.json",
            "review.json",
        ],
    )

    user_events: Dict[str, List[Tuple[int, str, float]]] = {}
    for rec in _iter_json_records(review_file):
        uid = str(rec.get("user_id") or "").strip()
        iid = str(rec.get("business_id") or "").strip()
        if not uid or not iid:
            continue
        ts = _parse_ts(rec.get("date"))
        try:
            rating = float(rec.get("stars") or 0.0)
        except (TypeError, ValueError):
            rating = 0.0
        user_events.setdefault(uid, []).append((ts, iid, rating))

    for uid in user_events:
        user_events[uid].sort(key=lambda x: x[0])
    return user_events


def read_done_ids(path: str, key: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if key in obj:
                    done.add(str(obj[key]))
            except Exception:
                continue
    return done


def append_jsonl(path: str, obj: Dict, lock: Lock) -> None:
    text = json.dumps(obj, ensure_ascii=False)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def _normalize_json_profile(res: Dict) -> Dict[str, str]:
    """对模型输出做字段标准化，避免键名漂移导致后续失败。"""
    candidates_sum = ["summarization", "summary", "profile", "user_profile", "item_profile"]
    candidates_reason = ["reasoning", "rationale", "analysis", "evidence"]

    summary = ""
    for k in candidates_sum:
        if k in res and str(res[k]).strip():
            summary = str(res[k]).strip()
            break

    reasoning = ""
    for k in candidates_reason:
        if k in res and str(res[k]).strip():
            reasoning = str(res[k]).strip()
            break

    if not summary:
        summary = "None"
    if not reasoning:
        reasoning = "None"
    return {"summarization": summary, "reasoning": reasoning}


def build_item_prompts(item: ItemInfo) -> Tuple[str, str]:
    system_prompt = (
        "你是推荐系统画像助手。"
        "任务是根据提供的物品（如电影、服饰、商家等）信息生成用户偏好画像。"
        "请严格遵守："
        "1) 仅输出一个 JSON 对象，不得包含 Markdown、解释性前后缀或代码块；"
        "2) JSON 必须包含键 summarization 和 reasoning；"
        "3) summarization 聚焦该物品吸引的用户类型、偏好或使用场景；"
        "4) reasoning 需说明依据（标题/类别/描述/属性）；"
        "5) 若信息不足，填 None，禁止编造。"
    )
    user_prompt = json.dumps(
        {
            "instruction": "请根据输入电影信息，输出该电影吸引的用户画像。",
            "constraints": {
                "output_format": {
                    "summarization": "string",
                    "reasoning": "string",
                },
                "summarization_max_words": 100,
                "reasoning_max_words": 200,
                "language": "zh",
            },
            "item": {
                "item_id": item.raw_item_id,
                "title": item.title,
                "genres": item.genres,
                "description": item.description or "None",
            },
        },
        ensure_ascii=False,
    )
    return system_prompt, user_prompt


def build_user_prompts(
    user_id: str,
    history_items: List[Tuple[ItemInfo, int, float]],
    item_profiles: Dict[str, str],
) -> Tuple[str, str]:
    system_prompt = (
        "你是推荐系统用户画像助手。"
        "任务是根据用户历史交互和物品画像，生成用户偏好画像。"
        "请严格遵守："
        "1) 仅输出一个 JSON 对象，不得包含 Markdown、解释性前后缀或代码块；"
        "2) JSON 必须包含键 summarization 和 reasoning；"
        "3) summarization 描述该用户偏好的电影类型/主题/风格；"
        "4) reasoning 必须引用输入中的证据（历史物品画像和交互信号）；"
        "5) 若信息不足，填 None，禁止编造。"
    )

    serialized_hist = []
    for it, ts, rating in history_items:
        serialized_hist.append(
            {
                "item_id": it.raw_item_id,
                "title": it.title,
                "genres": it.genres,
                "item_profile": item_profiles.get(it.raw_item_id, "None"),
                "user_feedback": {
                    "rating": rating,
                    "timestamp": ts,
                },
            }
        )

    user_prompt = json.dumps(
        {
            "instruction": "请总结该用户可能偏好的电影类型及兴趣方向。",
            "constraints": {
                "output_format": {
                    "summarization": "string",
                    "reasoning": "string",
                },
                "summarization_max_words": 120,
                "reasoning_max_words": 220,
                "language": "zh",
            },
            "user": {
                "user_id": user_id,
                "history_items": serialized_hist,
            },
        },
        ensure_ascii=False,
    )
    return system_prompt, user_prompt


def load_item_profiles_map(item_profile_path: str) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    if not os.path.exists(item_profile_path):
        return mp
    with open(item_profile_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                iid = str(obj["item_id"])
                profile = str(obj.get("summarization", ""))
                mp[iid] = profile
            except Exception:
                continue
    return mp


def generate_item_profiles(
    client: OpenAICompatibleClient,
    items: Dict[str, ItemInfo],
    out_path: str,
    max_items: int,
    workers: int,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = read_done_ids(out_path, key="item_id")

    all_items = sorted(items.values(), key=lambda x: x.raw_item_id)
    if max_items > 0:
        all_items = all_items[:max_items]
    targets = [it for it in all_items if it.raw_item_id not in done]

    print(f"[item] total={len(all_items)} done={len(done)} todo={len(targets)}")
    lock = Lock()

    def _work(it: ItemInfo) -> Tuple[str, Dict]:
        system_prompt, user_prompt = build_item_prompts(it)
        res = client.chat_json(system_prompt, user_prompt)
        normalized = _normalize_json_profile(res)
        obj = {
            "item_id": it.raw_item_id,
            "title": it.title,
            "genres": it.genres,
            "summarization": normalized["summarization"],
            "reasoning": normalized["reasoning"],
        }
        return it.raw_item_id, obj

    if workers <= 1:
        for idx, it in enumerate(targets, start=1):
            try:
                _, obj = _work(it)
                append_jsonl(out_path, obj, lock)
                if idx % 50 == 0:
                    print(f"[item] progress {idx}/{len(targets)}")
            except Exception as e:
                print(f"[item] failed item_id={it.raw_item_id} error={e}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_id = {ex.submit(_work, it): it.raw_item_id for it in targets}
            done_cnt = 0
            for fut in as_completed(fut_to_id):
                iid = fut_to_id[fut]
                try:
                    _, obj = fut.result()
                    append_jsonl(out_path, obj, lock)
                except Exception as e:
                    print(f"[item] failed item_id={iid} error={e}")
                done_cnt += 1
                if done_cnt % 50 == 0:
                    print(f"[item] progress {done_cnt}/{len(targets)}")


def generate_user_profiles(
    client: OpenAICompatibleClient,
    items: Dict[str, ItemInfo],
    user_events: Dict[str, List[Tuple[int, str, float]]],
    item_profile_path: str,
    out_path: str,
    max_users: int,
    max_history_items: int,
    workers: int,
    seed: int,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = read_done_ids(out_path, key="user_id")
    item_profiles = load_item_profiles_map(item_profile_path)

    all_users = sorted(user_events.keys())
    if max_users > 0:
        all_users = all_users[:max_users]
    targets = [u for u in all_users if u not in done]

    print(f"[user] total={len(all_users)} done={len(done)} todo={len(targets)}")
    lock = Lock()
    rng = random.Random(seed)

    def _sample_history(uid: str) -> List[Tuple[ItemInfo, int, float]]:
        seq = [(iid, ts, rating) for ts, iid, rating in user_events[uid] if iid in items]
        if len(seq) == 0:
            return []
        # 尽量保留最近行为，同时控制输入长度
        seq = seq[-max_history_items * 2 :]
        if len(seq) > max_history_items:
            seq = rng.sample(seq, k=max_history_items)
        seq.sort(key=lambda x: x[1])
        return [(items[iid], ts, rating) for iid, ts, rating in seq]

    def _work(uid: str) -> Tuple[str, Dict]:
        hist = _sample_history(uid)
        if len(hist) == 0:
            return uid, {
                "user_id": uid,
                "summarization": "",
                "reasoning": "no_history",
                "history_size": 0,
            }

        system_prompt, user_prompt = build_user_prompts(uid, hist, item_profiles)
        res = client.chat_json(system_prompt, user_prompt)
        normalized = _normalize_json_profile(res)
        obj = {
            "user_id": uid,
            "summarization": normalized["summarization"],
            "reasoning": normalized["reasoning"],
            "history_size": len(hist),
        }
        return uid, obj

    if workers <= 1:
        for idx, uid in enumerate(targets, start=1):
            try:
                _, obj = _work(uid)
                append_jsonl(out_path, obj, lock)
                if idx % 50 == 0:
                    print(f"[user] progress {idx}/{len(targets)}")
            except Exception as e:
                print(f"[user] failed user_id={uid} error={e}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_uid = {ex.submit(_work, uid): uid for uid in targets}
            done_cnt = 0
            for fut in as_completed(fut_to_uid):
                uid = fut_to_uid[fut]
                try:
                    _, obj = fut.result()
                    append_jsonl(out_path, obj, lock)
                except Exception as e:
                    print(f"[user] failed user_id={uid} error={e}")
                done_cnt += 1
                if done_cnt % 50 == 0:
                    print(f"[user] progress {done_cnt}/{len(targets)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用本地Qwen生成推荐画像（默认仅物品）")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--stage", type=str, default="item", choices=["item", "user", "all"])
    parser.add_argument("--output-dir", type=str, default="profiles")

    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--model", type=str, default="Qwen2.5-32B-Instruct")

    parser.add_argument("--max-items", type=int, default=0, help="0表示全部")
    parser.add_argument("--max-users", type=int, default=0, help="0表示全部")
    parser.add_argument("--max-history-items", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--enable-user-profile-generation",
        action="store_true",
        help="显式开启用户画像生成（默认关闭，避免误触发慢速流程）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset = normalize_dataset_name(args.dataset)

    dataset_dir = args.data_dir
    if args.dataset == "ml100k":
        items = load_ml100k_items(dataset_dir)
    elif args.dataset == "ml25m":
        items = load_ml25m_items(dataset_dir)
    elif args.dataset in {"amazon_all_beauty", "amazon_fashion"}:
        items = load_amazon_items(dataset_dir, args.dataset)
    else:
        items = load_yelp_items(dataset_dir)

    need_user_stage = args.stage in {"user", "all"} and args.enable_user_profile_generation
    user_events: Dict[str, List[Tuple[int, str, float]]] = {}
    if need_user_stage:
        if args.dataset == "ml100k":
            user_events = load_ml100k_user_interactions(dataset_dir)
        elif args.dataset == "ml25m":
            user_events = load_ml25m_user_interactions(dataset_dir)
        elif args.dataset in {"amazon_all_beauty", "amazon_fashion"}:
            user_events = load_amazon_user_interactions(dataset_dir, args.dataset)
        else:
            user_events = load_yelp_user_interactions(dataset_dir)

    client = OpenAICompatibleClient(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        timeout=180,
    )

    out_root = os.path.join(args.output_dir, args.dataset)
    item_out = os.path.join(out_root, "item_profiles.jsonl")
    user_out = os.path.join(out_root, "user_profiles.jsonl")

    if args.stage in {"item", "all"}:
        generate_item_profiles(
            client=client,
            items=items,
            out_path=item_out,
            max_items=args.max_items,
            workers=args.workers,
        )

    if args.stage in {"user", "all"}:
        if not args.enable_user_profile_generation:
            if args.stage == "user":
                raise ValueError(
                    "当前默认关闭用户画像生成。若确需生成，请添加 --enable-user-profile-generation"
                )
            print("[user] skip: 用户画像生成默认关闭；如需开启请添加 --enable-user-profile-generation")
        else:
            if not os.path.exists(item_out):
                raise FileNotFoundError(
                    f"未找到 item profiles: {item_out}。请先运行 --stage item 或 --stage all"
                )
            generate_user_profiles(
                client=client,
                items=items,
                user_events=user_events,
                item_profile_path=item_out,
                out_path=user_out,
                max_users=args.max_users,
                max_history_items=args.max_history_items,
                workers=args.workers,
                seed=args.seed,
            )

    print("done")


if __name__ == "__main__":
    main()
