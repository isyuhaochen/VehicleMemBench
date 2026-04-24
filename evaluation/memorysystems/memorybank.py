import json
import math
import os
import random
import shutil
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from .common import (
    collect_history_files,
    load_hourly_history,
    require_value,
    resolve_memory_key,
    resolve_memory_url,
    run_add_jobs,
)


TAG = "MEMORYBANK"
USER_ID_PREFIX = "memorybank"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

STORE_ROOT = os.environ.get(
    "MEMORYBANK_STORE_ROOT",
    os.path.join(_ROOT_DIR, "log", "memorybank"),
)


def _resolve_embedding_api_key(args) -> Optional[str]:
    direct = getattr(args, "embedding_api_key", None)
    if direct:
        return direct
    return resolve_memory_key(args, "EMBEDDING_API_KEY")


def _resolve_embedding_api_base(args) -> Optional[str]:
    direct = getattr(args, "embedding_api_base", None)
    if direct:
        return direct
    return resolve_memory_url(args, "EMBEDDING_API_BASE")


def _resolve_embedding_model(args) -> str:
    direct = getattr(args, "embedding_model", None)
    if direct:
        return direct
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def _resolve_reference_date() -> Optional[str]:
    return os.getenv("MEMORYBANK_REFERENCE_DATE")


def _resolve_enable_summary() -> bool:
    return os.getenv("MEMORYBANK_ENABLE_SUMMARY", "").lower() in ("1", "true", "yes")


def _resolve_disable_forgetting() -> bool:
    return os.getenv("MEMORYBANK_DISABLE_FORGETTING", "").lower() in ("1", "true", "yes")


def _resolve_seed() -> Optional[int]:
    raw = os.getenv("MEMORYBANK_SEED")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def _resolve_store_root(args) -> str:
    return getattr(args, "store_root", None) or STORE_ROOT


def _user_store_dir(user_id: str, store_root: str = STORE_ROOT) -> str:
    return os.path.join(store_root, f"user_{user_id}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, mode=0o700, exist_ok=True)



def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


class MemoryBankClient:

    def __init__(
        self,
        *,
        embedding_api_base: str,
        embedding_api_key: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        enable_forgetting: bool = True,
        enable_summary: bool = False,
        seed: Optional[int] = None,
        reference_date: Optional[str] = None,
        llm_api_base: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        store_root: str = STORE_ROOT,
    ):
        import openai as _openai

        self._store_root = store_root

        self.embedding_api_base = embedding_api_base
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model
        self.enable_forgetting = enable_forgetting
        self.enable_summary = enable_summary
        self.seed = seed
        self.reference_date = reference_date

        self._embedding_dim: Optional[int] = None
        self._indices: Dict[str, faiss.IndexIDMap] = {}
        self._metadata: Dict[str, List[dict]] = {}
        self._next_id: Dict[str, int] = {}
        self._rng = random.Random(seed)

        self._embed_client = _openai.OpenAI(
            base_url=embedding_api_base,
            api_key=embedding_api_key,
        )

        self._llm_client = None
        if enable_summary and llm_api_base and llm_api_key:
            self._llm_client = _openai.OpenAI(
                base_url=llm_api_base,
                api_key=llm_api_key,
            )
            self._llm_model = llm_model or "gpt-4o-mini"

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        max_retries = 5
        resp = None
        for attempt in range(max_retries):
            try:
                resp = self._embed_client.embeddings.create(
                    input=texts,
                    model=self.embedding_model,
                )
                break
            except Exception:
                if attempt < max_retries - 1:
                    jitter = random.random()
                    time.sleep(2 ** attempt + jitter)
                else:
                    raise

        vectors = []
        for item in resp.data:
            vec = np.array(item.embedding, dtype=np.float32)
            vec = _l2_normalize(vec)
            vectors.append(vec.tolist())

        if self._embedding_dim is None and vectors:
            self._embedding_dim = len(vectors[0])

        return vectors

    def _get_or_create_index(self, user_id: str) -> Tuple[faiss.IndexIDMap, List[dict]]:
        if user_id in self._indices:
            return self._indices[user_id], self._metadata[user_id]

        store_dir = _user_store_dir(user_id, self._store_root)
        index_path = os.path.join(store_dir, "index.faiss")
        meta_path = os.path.join(store_dir, "metadata.json")

        if os.path.isfile(index_path) and os.path.isfile(meta_path):
            index = faiss.read_index(index_path)
            if not isinstance(index, faiss.IndexIDMap):
                dim = index.d
                new_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
                n = index.ntotal
                if n > 0:
                    all_vecs = np.zeros((n, dim), dtype=np.float32)
                    for i in range(n):
                        all_vecs[i] = index.reconstruct(i)
                    ids = np.arange(n, dtype=np.int64)
                    new_index.add_with_ids(all_vecs, ids)
                index = new_index
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            for i, meta in enumerate(metadata):
                if "faiss_id" not in meta:
                    meta["faiss_id"] = i
            self._next_id[user_id] = max(
                (m["faiss_id"] for m in metadata), default=-1
            ) + 1
        else:
            dim = self._embedding_dim or 1536
            index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
            metadata = []
            self._next_id[user_id] = 0

        self._indices[user_id] = index
        self._metadata[user_id] = metadata
        return index, metadata

    def _alloc_id(self, user_id: str) -> int:
        fid = self._next_id.get(user_id, 0)
        self._next_id[user_id] = fid + 1
        return fid

    def _add_vector(self, user_id: str, text: str, embedding: List[float], timestamp: str, extra_meta: Optional[dict] = None) -> None:
        index, metadata = self._get_or_create_index(user_id)
        fid = self._alloc_id(user_id)
        vec = np.array([embedding], dtype=np.float32)
        index.add_with_ids(vec, np.array([fid], dtype=np.int64))
        meta_entry = {
            "text": text,
            "timestamp": timestamp,
            "memory_strength": 1,
            "last_recall_date": timestamp[:10] if len(timestamp) >= 10 else timestamp,
            "faiss_id": fid,
        }
        if extra_meta:
            meta_entry.update(extra_meta)
        metadata.append(meta_entry)

    def save_index(self, user_id: str) -> None:
        if user_id not in self._indices:
            return
        store_dir = _user_store_dir(user_id, self._store_root)
        _ensure_dir(store_dir)
        index_path = os.path.join(store_dir, "index.faiss")
        meta_path = os.path.join(store_dir, "metadata.json")
        faiss.write_index(self._indices[user_id], index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata[user_id], f, ensure_ascii=False, indent=2)

    def add(self, messages: List[dict], user_id: str, timestamp: str) -> None:
        lines: List[str] = []
        for msg in messages:
            content = msg.get("content", "")
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)

        if not lines:
            return

        embeddings = self._get_embeddings(lines)

        for line, emb in zip(lines, embeddings, strict=True):
            self._add_vector(user_id, line, emb, timestamp)

    def _summarize(self, text: str) -> str:
        if not self._llm_client:
            return ""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Summarize the following conversation concisely, "
                                "extracting user preferences, conditions, and specific values "
                                "(e.g., temperature settings, seat positions, color choices). "
                                "Organize by topic."
                            ),
                        },
                        {"role": "user", "content": text},
                    ],
                    max_tokens=400,
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return ""

    def _generate_daily_summaries(self, user_id: str) -> None:
        if not self._llm_client:
            return

        metadata = self._metadata.get(user_id, [])
        daily_texts: Dict[str, List[str]] = {}
        for meta in metadata:
            if meta.get("type") in ("daily_summary", "overall_summary"):
                continue
            date_key = meta.get("timestamp", "")[:10]
            if not date_key:
                continue
            daily_texts.setdefault(date_key, []).append(meta["text"])

        for date_key, texts in sorted(daily_texts.items()):
            combined = "\n".join(texts)
            summary = self._summarize(combined)
            if summary:
                ts = f"{date_key}T00:00:00"
                summary_emb = self._get_embeddings([summary])[0]
                self._add_vector(user_id, summary, summary_emb, ts, {"type": "daily_summary"})

    def _generate_overall_summary(self, user_id: str) -> None:
        if not self._llm_client:
            return

        metadata = self._metadata.get(user_id, [])
        daily_summaries = [
            m["text"] for m in metadata if m.get("type") == "daily_summary"
        ]
        if not daily_summaries:
            return

        combined = "\n\n".join(daily_summaries)
        summary = self._summarize(combined)
        if summary:
            ts = metadata[-1]["timestamp"] if metadata else datetime.now().isoformat()
            summary_emb = self._get_embeddings([summary])[0]
            self._add_vector(user_id, summary, summary_emb, ts, {"type": "overall_summary"})

    def _forgetting_retention(self, days_elapsed: float, memory_strength: int) -> float:
        return math.exp(-days_elapsed / (5 * memory_strength))

    def _forget_at_ingestion(self, user_id: str) -> None:
        if not self.enable_forgetting or not self.reference_date:
            return

        index, metadata = self._get_or_create_index(user_id)

        ref_dt = datetime.strptime(self.reference_date[:10], "%Y-%m-%d")
        ids_to_remove: List[int] = []
        indices_to_keep: List[int] = []

        for i, meta in enumerate(metadata):
            ts_str = meta.get("last_recall_date", meta.get("timestamp", ""))[:10]
            try:
                mem_dt = datetime.strptime(ts_str, "%Y-%m-%d")
            except ValueError:
                indices_to_keep.append(i)
                continue
            days_elapsed = (ref_dt - mem_dt).days
            strength = meta.get("memory_strength", 1)
            retention = self._forgetting_retention(days_elapsed, strength)
            if self._rng.random() > retention:
                ids_to_remove.append(meta["faiss_id"])
            else:
                indices_to_keep.append(i)

        if ids_to_remove:
            index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
            self._metadata[user_id] = [metadata[i] for i in indices_to_keep]
            self._indices[user_id] = index

    def search(self, query: str, user_id: str, top_k: int = 5) -> List[dict]:
        index, metadata = self._get_or_create_index(user_id)

        if index.ntotal == 0:
            return []

        query_emb = self._get_embeddings([query])[0]
        query_vec = np.array([query_emb], dtype=np.float32)

        k = min(top_k, index.ntotal)
        scores, indices = index.search(query_vec, k)

        id_to_meta = {m["faiss_id"]: i for i, m in enumerate(metadata)}

        results: List[dict] = []
        for score, faiss_id in zip(scores[0], indices[0]):
            meta_idx = id_to_meta.get(int(faiss_id))
            if meta_idx is None:
                continue
            meta = dict(metadata[meta_idx])
            meta["score"] = float(score)
            meta["_meta_idx"] = meta_idx
            results.append(meta)

        for r in results:
            meta_idx = r.pop("_meta_idx", None)
            if meta_idx is not None and 0 <= meta_idx < len(metadata):
                metadata[meta_idx]["memory_strength"] = metadata[meta_idx].get("memory_strength", 1) + 1
                if self.reference_date:
                    metadata[meta_idx]["last_recall_date"] = self.reference_date[:10]

        return results


def validate_add_args(args) -> None:
    require_value(
        _resolve_embedding_api_key(args),
        "Embedding API key is required: pass --memory_key or set MEMORY_KEY/EMBEDDING_API_KEY",
    )
    require_value(
        _resolve_embedding_api_base(args),
        "Embedding API base URL is required: pass --memory_url or set MEMORY_URL/EMBEDDING_API_BASE",
    )


def validate_test_args(args) -> None:
    validate_add_args(args)


def _build_client(args, user_id: str = "") -> MemoryBankClient:
    api_key = require_value(
        _resolve_embedding_api_key(args),
        "Embedding API key is required: pass --memory_key or set MEMORY_KEY/EMBEDDING_API_KEY",
    )
    api_base = require_value(
        _resolve_embedding_api_base(args),
        "Embedding API base URL is required: pass --memory_url or set MEMORY_URL/EMBEDDING_API_BASE",
    )

    disable_forgetting = _resolve_disable_forgetting()
    enable_summary = _resolve_enable_summary()
    seed = _resolve_seed()
    reference_date = _resolve_reference_date()

    llm_api_base = resolve_memory_url(args, "LLM_API_BASE")
    llm_api_key = resolve_memory_key(args, "LLM_API_KEY")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    return MemoryBankClient(
        embedding_api_base=api_base,
        embedding_api_key=api_key,
        embedding_model=_resolve_embedding_model(args),
        enable_forgetting=not disable_forgetting,
        enable_summary=enable_summary,
        seed=seed,
        reference_date=reference_date,
        llm_api_base=llm_api_base,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        store_root=_resolve_store_root(args),
    )


def _compute_reference_date(history_dir: str, file_range: Optional[str]) -> str:
    history_files = collect_history_files(history_dir, file_range)
    max_ts: Optional[datetime] = None
    for _, path in history_files:
        for bucket in load_hourly_history(path):
            if bucket.dt is not None:
                if max_ts is None or bucket.dt > max_ts:
                    max_ts = bucket.dt
    if max_ts is None:
        max_ts = datetime.now()
    ref_date = max_ts + timedelta(days=1)
    return ref_date.strftime("%Y-%m-%d")


def run_add(args) -> None:
    validate_add_args(args)
    history_dir = os.path.abspath(args.history_dir)
    if not os.path.isdir(history_dir):
        raise FileNotFoundError(f"history directory not found: {history_dir}")

    history_files = collect_history_files(history_dir, args.file_range)
    print(
        f"[{TAG} ADD] history_dir={history_dir} files={len(history_files)} max_workers={args.max_workers} store_root={_resolve_store_root(args)}"
    )

    reference_date = _resolve_reference_date()
    if not reference_date:
        reference_date = _compute_reference_date(history_dir, args.file_range)

    def processor(idx: int, history_path: str) -> Tuple[int, int, Optional[str]]:
        client = _build_client(args)
        client.reference_date = reference_date
        user_id = f"{USER_ID_PREFIX}_{idx}"
        store_dir = _user_store_dir(user_id, client._store_root)
        if os.path.isdir(store_dir):
            shutil.rmtree(store_dir)
        try:
            message_count = 0
            for bucket in load_hourly_history(history_path):
                ts = bucket.dt.isoformat() if bucket.dt else datetime.now().isoformat()
                messages = [{"role": "user", "content": "\n".join(bucket.lines)}]
                client.add(messages=messages, user_id=user_id, timestamp=ts)
                message_count += len(bucket.lines)

            if _resolve_enable_summary():
                client._generate_daily_summaries(user_id)
                client._generate_overall_summary(user_id)

            client._forget_at_ingestion(user_id)

            client.save_index(user_id)
            return idx, message_count, None
        except Exception as exc:
            return idx, 0, str(exc)

    run_add_jobs(
        history_files=history_files,
        tag=TAG,
        max_workers=args.max_workers,
        processor=processor,
    )


def init_test_state(args, file_numbers, user_id_prefix):
    del file_numbers, user_id_prefix
    validate_test_args(args)
    return None


def build_test_client(args, file_num: int, user_id_prefix: str, shared_state: Any):
    del shared_state
    client = _build_client(args)
    if not client.reference_date:
        history_dir = os.path.abspath(args.history_dir)
        client.reference_date = _compute_reference_date(history_dir, str(file_num))
    uid = f"{user_id_prefix}_{file_num}"
    client._get_or_create_index(uid)
    return _MemoryBankTestWrapper(client, uid)


class _MemoryBankTestWrapper:
    def __init__(self, client: MemoryBankClient, user_id: str):
        self._client = client
        self._user_id = user_id

    def search(self, query: str, user_id: Optional[str] = None, top_k: int = 5) -> List[dict]:
        uid = user_id if user_id is not None else self._user_id
        return self._client.search(query=query, user_id=uid, top_k=top_k)


def close_test_state(shared_state: Any) -> None:
    del shared_state


def is_test_sequential() -> bool:
    return False


def format_search_results(search_result: Any) -> Tuple[str, int]:
    if not isinstance(search_result, list):
        return "", 0
    if not search_result:
        return "", 0

    lines = []
    for idx, item in enumerate(search_result, 1):
        text = item.get("text", "")
        strength = item.get("memory_strength", 1)
        lines.append(f"{idx}. [memory_strength={strength}] {text}")

    return "\n\n".join(lines), len(search_result)
