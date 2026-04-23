import hashlib
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
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

STORE_ROOT = os.environ.get(
    "MEMORYBANK_STORE_ROOT",
    os.path.join(_ROOT_DIR, "log", "memorybank"),
)

_SPLIT_SEPARATORS = ["\n\n", "\n", ". ", " "]


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


def _stable_hash(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, mode=0o700, exist_ok=True)


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    for sep in _SPLIT_SEPARATORS:
        if sep in text:
            parts = text.split(sep)
            chunks: List[str] = []
            current = ""
            for part in parts:
                candidate = f"{current}{sep}{part}" if current else part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current.strip():
                        chunks.append(current)
                    if len(part) > chunk_size:
                        start = 0
                        while start < len(part):
                            end = start + chunk_size
                            chunk = part[start:end]
                            if chunk.strip():
                                chunks.append(chunk)
                            start += chunk_size - chunk_overlap
                    else:
                        current = part
            if current.strip():
                chunks.append(current)
            return chunks

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


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
        self._indices: Dict[str, faiss.IndexFlatIP] = {}
        self._metadata: Dict[str, List[dict]] = {}

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

    def _get_or_create_index(self, user_id: str) -> Tuple[faiss.IndexFlatIP, List[dict]]:
        if user_id in self._indices:
            return self._indices[user_id], self._metadata[user_id]

        store_dir = _user_store_dir(user_id, self._store_root)
        index_path = os.path.join(store_dir, "index.faiss")
        meta_path = os.path.join(store_dir, "metadata.json")

        if os.path.isfile(index_path) and os.path.isfile(meta_path):
            index = faiss.read_index(index_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            dim = self._embedding_dim or 1536
            index = faiss.IndexFlatIP(dim)
            metadata = []

        self._indices[user_id] = index
        self._metadata[user_id] = metadata
        return index, metadata

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
        all_chunks: List[str] = []
        for msg in messages:
            content = msg.get("content", "")
            if not content.strip():
                continue
            chunks = _split_text(content)
            all_chunks.extend(chunks)

        if not all_chunks:
            return

        embeddings = self._get_embeddings(all_chunks)

        index, metadata = self._get_or_create_index(user_id)

        for i, (chunk, emb) in enumerate(zip(all_chunks, embeddings)):
            vec = np.array([emb], dtype=np.float32)
            index.add(vec)
            meta_entry = {
                "text": chunk,
                "timestamp": timestamp,
                "memory_strength": 1,
                "last_recall_date": timestamp[:10] if len(timestamp) >= 10 else timestamp,
            }
            metadata.append(meta_entry)

        if self.enable_summary and self._llm_client and all_chunks:
            full_text = " ".join(all_chunks)
            summary = self._summarize(full_text)
            if summary:
                summary_emb = self._get_embeddings([summary])[0]
                vec = np.array([summary_emb], dtype=np.float32)
                index.add(vec)
                metadata.append({
                    "text": summary,
                    "timestamp": timestamp,
                    "memory_strength": 1,
                    "last_recall_date": timestamp[:10] if len(timestamp) >= 10 else timestamp,
                })

        self._indices[user_id] = index
        self._metadata[user_id] = metadata

    def _summarize(self, text: str) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    messages=[
                        {"role": "system", "content": "Summarize the following text concisely."},
                        {"role": "user", "content": text},
                    ],
                    max_tokens=200,
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return ""

    def _forgetting_retention(self, days_elapsed: float, memory_strength: int) -> float:
        return math.exp(-days_elapsed / (5 * memory_strength))

    def _apply_forgetting(
        self, candidates: List[dict], reference_date: str, user_id: str
    ) -> List[dict]:
        seed = self.seed if self.seed is not None else _stable_hash(user_id)
        rng = random.Random(seed)

        ref_dt = datetime.strptime(reference_date[:10], "%Y-%m-%d")
        kept = []
        for candidate in candidates:
            ts_str = candidate.get("last_recall_date", candidate.get("timestamp", ""))[:10]
            try:
                mem_dt = datetime.strptime(ts_str, "%Y-%m-%d")
            except ValueError:
                kept.append(candidate)
                continue
            days_elapsed = (ref_dt - mem_dt).days
            strength = candidate.get("memory_strength", 1)
            retention = self._forgetting_retention(days_elapsed, strength)
            if rng.random() <= retention:
                kept.append(candidate)
        return kept

    def search(self, query: str, user_id: str, top_k: int = 5) -> List[dict]:
        index, metadata = self._get_or_create_index(user_id)

        if index.ntotal == 0:
            return []

        query_emb = self._get_embeddings([query])[0]
        query_vec = np.array([query_emb], dtype=np.float32)

        oversample = min(top_k * 2, index.ntotal)
        scores, indices = index.search(query_vec, oversample)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            meta = dict(metadata[idx])
            meta["score"] = float(score)
            meta["_orig_idx"] = int(idx)
            candidates.append(meta)

        if self.enable_forgetting and self.reference_date:
            candidates = self._apply_forgetting(candidates, self.reference_date, user_id)

        results = candidates[:top_k]

        for r in results:
            orig_idx = r.pop("_orig_idx", None)
            if orig_idx is not None and 0 <= orig_idx < len(metadata):
                metadata[orig_idx]["memory_strength"] = metadata[orig_idx].get("memory_strength", 1) + 1
                if self.reference_date:
                    metadata[orig_idx]["last_recall_date"] = self.reference_date[:10]

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
    reference_date = _resolve_reference_date()
    if not reference_date:
        reference_date = _compute_reference_date(
            os.path.abspath(args.history_dir), args.file_range
        )
    return {"reference_date": reference_date}


def build_test_client(args, file_num: int, user_id_prefix: str, shared_state: Any):
    client = _build_client(args)
    if not client.reference_date:
        client.reference_date = shared_state["reference_date"]
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
    pass


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
