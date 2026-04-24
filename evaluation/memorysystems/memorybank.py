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
CHUNK_SIZE = 200
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


def _separate_list(ls: List[int]) -> List[List[int]]:
    if not ls:
        return []
    lists: List[List[int]] = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists


def _split_by_source(indices: List[int], metadata: List[dict]) -> List[List[int]]:
    if not indices:
        return []
    groups: List[List[int]] = [[indices[0]]]
    for idx in indices[1:]:
        cur_source = metadata[idx].get("source", "")
        prev_source = metadata[groups[-1][-1]].get("source", "")
        if cur_source == prev_source:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


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

    def _merge_neighbors(self, results: List[dict], user_id: str) -> List[dict]:
        if not results:
            return results

        metadata = self._metadata.get(user_id, [])
        if not metadata:
            return results

        indexed = [(r, r["_meta_idx"]) for r in results if r.get("_meta_idx") is not None]
        if not indexed:
            return results
        non_indexed = [r for r in results if r.get("_meta_idx") is None]

        id_set: set = set()
        idx_to_score: Dict[int, float] = {}
        for r, meta_idx in indexed:
            idx_to_score[meta_idx] = float(r.get("score", 0.0))
            source = r.get("source", "")
            docs_len = len(metadata[meta_idx].get("text", ""))
            id_set.add(meta_idx)

            forward_ok = True
            backward_ok = True
            max_offset = max(len(metadata) - meta_idx, meta_idx + 1)

            for offset in range(1, max_offset):
                if not forward_ok and not backward_ok:
                    break

                if forward_ok:
                    neighbor_pos = meta_idx + offset
                    if neighbor_pos >= len(metadata):
                        forward_ok = False
                    elif metadata[neighbor_pos].get("source") != source:
                        forward_ok = False
                    else:
                        neighbor_text = metadata[neighbor_pos].get("text", "")
                        if docs_len + len(neighbor_text) > CHUNK_SIZE:
                            forward_ok = False
                        else:
                            docs_len += len(neighbor_text)
                            id_set.add(neighbor_pos)

                if backward_ok:
                    neighbor_pos = meta_idx - offset
                    if neighbor_pos < 0:
                        backward_ok = False
                    elif metadata[neighbor_pos].get("source") != source:
                        backward_ok = False
                    else:
                        neighbor_text = metadata[neighbor_pos].get("text", "")
                        if docs_len + len(neighbor_text) > CHUNK_SIZE:
                            backward_ok = False
                        else:
                            docs_len += len(neighbor_text)
                            id_set.add(neighbor_pos)

        sorted_ids = sorted(id_set)
        contiguous_groups = _separate_list(sorted_ids)

        merged_results: List[dict] = []
        for contiguous in contiguous_groups:
            same_source_groups = _split_by_source(contiguous, metadata)
            for group in same_source_groups:
                combined_text = "".join(metadata[i].get("text", "") for i in group)
                group_scores = [idx_to_score[i] for i in group if i in idx_to_score]
                best_score = max(group_scores) if group_scores else 0.0

                base_meta = dict(metadata[group[0]])
                base_meta["text"] = combined_text
                base_meta["score"] = float(best_score)
                base_meta["memory_strength"] = max(
                    metadata[i].get("memory_strength", 1) for i in group
                )
                base_meta["_merged_indices"] = group
                merged_results.append(base_meta)

        merged_results.extend(non_indexed)
        merged_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return merged_results

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

    @staticmethod
    def _parse_speaker(line: str) -> Tuple[str, str]:
        colon_pos = line.find(": ")
        if colon_pos > 0:
            return line[:colon_pos].strip(), line[colon_pos + 2:].strip()
        return "Speaker", line.strip()

    def add(self, messages: List[dict], user_id: str, timestamp: str) -> None:
        date_key = timestamp[:10] if len(timestamp) >= 10 else timestamp
        all_entries: List[Tuple[str, str]] = []
        for msg in messages:
            content = msg.get("content", "")
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped:
                    speaker, text = self._parse_speaker(stripped)
                    all_entries.append((speaker, text))

        if not all_entries:
            return

        formatted_pairs: List[str] = []
        for i in range(0, len(all_entries), 2):
            speaker_a, text_a = all_entries[i]
            if i + 1 < len(all_entries):
                speaker_b, text_b = all_entries[i + 1]
                formatted = (
                    f"Conversation content on {date_key}:"
                    f"[|{speaker_a}|]: {text_a}; "
                    f"[|{speaker_b}|]: {text_b}"
                )
            else:
                formatted = (
                    f"Conversation content on {date_key}:"
                    f"[|{speaker_a}|]: {text_a}"
                )
            formatted_pairs.append(formatted)

        embeddings = self._get_embeddings(formatted_pairs)

        for text, emb in zip(formatted_pairs, embeddings, strict=True):
            self._add_vector(
                user_id, text, emb, timestamp,
                extra_meta={"source": date_key},
            )

    def _call_llm(self, last_user_content: str) -> str:
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
                                "Below is a transcript of a conversation between a human "
                                "and an AI assistant that is intelligent and knowledgeable "
                                "in psychology."
                            ),
                        },
                        {
                            "role": "user",
                            "content": "Hello! Please help me summarize the content of the conversation.",
                        },
                        {
                            "role": "system",
                            "content": "Sure, I will do my best to assist you.",
                        },
                        {"role": "user", "content": last_user_content},
                    ],
                    max_tokens=400,
                    temperature=0.7,
                    top_p=1.0,
                    frequency_penalty=0.4,
                    presence_penalty=0.2,
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return ""

    def _summarize(self, text: str) -> str:
        return self._call_llm(
            "Please summarize the following dialogue as concisely as "
            "possible, extracting the main themes and key information. "
            "If there are multiple key events, you may summarize them "
            f"separately. Dialogue content:\n{text}\n"
            "Summarization："
        )

    def _generate_daily_summaries(self, user_id: str) -> None:
        if not self._llm_client:
            return

        metadata = self._metadata.get(user_id, [])
        daily_texts: Dict[str, List[str]] = {}
        for meta in metadata:
            if meta.get("type") in ("daily_summary", "overall_summary"):
                continue
            date_key = meta.get("source", meta.get("timestamp", "")[:10])
            if not date_key:
                continue
            daily_texts.setdefault(date_key, []).append(meta["text"])

        for date_key, texts in sorted(daily_texts.items()):
            combined = "\n".join(texts)
            summary = self._summarize(combined)
            if summary:
                ts = f"{date_key}T00:00:00"
                summary_emb = self._get_embeddings([summary])[0]
                self._add_vector(user_id, summary, summary_emb, ts,
                                 {"type": "daily_summary", "source": date_key})

    def _generate_overall_summary(self, user_id: str) -> None:
        if not self._llm_client:
            return

        metadata = self._metadata.get(user_id, [])
        daily_summaries = [
            m for m in metadata if m.get("type") == "daily_summary"
        ]
        if not daily_summaries:
            return

        summary_parts = []
        for m in daily_summaries:
            date = m.get("source", m.get("timestamp", "")[:10])
            summary_parts.append((date, m["text"]))

        prompt = (
            "Please provide a highly concise summary of the following event, "
            "capturing the essential key information as succinctly as possible. "
            "Summarize the event:\n"
        )
        for date, text in summary_parts:
            prompt += f"\nAt {date}, the events are {text.strip()}"
        prompt += "\nSummarization："

        ts = metadata[-1]["timestamp"] if metadata else datetime.now().isoformat()
        summary = self._call_llm(prompt)
        if summary:
            summary_emb = self._get_embeddings([summary])[0]
            self._add_vector(user_id, summary, summary_emb, ts,
                             {"type": "overall_summary", "source": "overall"})

    def _analyze_personality(self, text: str) -> str:
        return self._call_llm(
            "Based on the following dialogue, please summarize the user's "
            "personality traits and emotions, and devise response strategies "
            f"based on your speculation. Dialogue content:\n{text}\n"
            "The user's personality traits, emotions, and the AI's response "
            "strategy are:"
        )

    def _generate_daily_personalities(self, user_id: str) -> None:
        if not self._llm_client:
            return

        metadata = self._metadata.get(user_id, [])
        daily_texts: Dict[str, List[str]] = {}
        for meta in metadata:
            if meta.get("type") in ("daily_summary", "overall_summary",
                                     "daily_personality", "overall_personality"):
                continue
            date_key = meta.get("source", meta.get("timestamp", "")[:10])
            if not date_key:
                continue
            daily_texts.setdefault(date_key, []).append(meta["text"])

        for date_key, texts in sorted(daily_texts.items()):
            combined = "\n".join(texts)
            personality = self._analyze_personality(combined)
            if personality:
                ts = f"{date_key}T00:00:00"
                personality_emb = self._get_embeddings([personality])[0]
                self._add_vector(user_id, personality, personality_emb, ts,
                                 {"type": "daily_personality", "source": date_key})

    def _generate_overall_personality(self, user_id: str) -> None:
        if not self._llm_client:
            return

        metadata = self._metadata.get(user_id, [])
        daily_personalities = [
            m for m in metadata if m.get("type") == "daily_personality"
        ]
        if not daily_personalities:
            return

        prompt = (
            "The following are the user's exhibited personality traits and emotions "
            "throughout multiple dialogues, along with appropriate response strategies "
            "for the current situation:\n"
        )
        for m in daily_personalities:
            date = m.get("source", m.get("timestamp", "")[:10])
            prompt += f"\nAt {date}, the analysis shows {m['text'].strip()}"
        prompt += (
            "\nPlease provide a highly concise and general summary of the user's "
            "personality and the most appropriate response strategy for the AI, "
            "summarized as:"
        )

        ts = metadata[-1]["timestamp"] if metadata else datetime.now().isoformat()
        personality = self._call_llm(prompt)
        if personality:
            personality_emb = self._get_embeddings([personality])[0]
            self._add_vector(user_id, personality, personality_emb, ts,
                             {"type": "overall_personality", "source": "overall"})

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

        merged = self._merge_neighbors(results, user_id)

        for r in merged:
            merged_indices: List[int] = r.pop("_merged_indices", [])
            meta_idx = r.pop("_meta_idx", None)
            all_indices: List[int] = merged_indices if merged_indices else ([meta_idx] if meta_idx is not None else [])
            for mi in all_indices:
                if 0 <= mi < len(metadata):
                    metadata[mi]["memory_strength"] = metadata[mi].get("memory_strength", 1) + 1
                    if self.reference_date:
                        metadata[mi]["last_recall_date"] = self.reference_date[:10]

        return merged


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
                client._generate_daily_personalities(user_id)
                client._generate_overall_personality(user_id)

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

    sorted_results = sorted(search_result, key=lambda r: r.get("source", ""))

    groups: List[Tuple[str, str, List[dict]]] = []
    for item in sorted_results:
        text = item.get("text", "")
        source = item.get("source", "")
        strength = item.get("memory_strength", 1)

        prefix = f"Conversation content on {source}:"
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

        if not groups or groups[-1][0] != source:
            groups.append((source, text, [item]))
        else:
            groups[-1] = (groups[-1][0], groups[-1][1] + "\n" + text, groups[-1][2] + [item])

    lines: List[str] = []
    for idx, (source, combined_text, items) in enumerate(groups, 1):
        max_strength = max(it.get("memory_strength", 1) for it in items)
        date_info = f" [date={source}]" if source else ""
        lines.append(f"{idx}. [memory_strength={max_strength}]{date_info} {combined_text}")

    return "\n\n".join(lines), len(search_result)
