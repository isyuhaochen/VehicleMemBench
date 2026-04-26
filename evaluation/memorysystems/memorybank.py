# ruff: noqa: RUF002, RUF003
"""
MemoryBank: 基于 FAISS 向量检索的本地记忆系统，复刻自原项目并适配 VehicleMemBench 测评场景。

原项目: https://github.com/zhongwanjun/MemoryBank-SiliconFriend
论文: https://arxiv.org/abs/2305.10250

相较于原项目的主要变更（搜索 `[DIFF]` 可定位所有差异点）:
- 向量索引: LangChain FAISS (IndexFlatL2) → 原生 FAISS IndexFlatIP + L2 归一化
- 嵌入模型: 本地 HuggingFace → OpenAI Embedding API (或兼容接口)
- 说话人格式: 固定 `[|User|]`/`[|AI|]` → 动态解析历史行中的说话人名称
- 遗忘公式: 修正原项目运算符优先级 bug (`-t/5*S` → `-t/(5*S)`)
- CHUNK_SIZE: 200 → 500（英文长对话适配）
- 移除 ChatGLM/BELLE 专用的 stop 序列
- 搜索后持久化 memory_strength（原项目缺失时的行为差异）
- 合并逻辑: 修正原项目 `break` 只跳出内层循环导致另一方向被跳过的 bug，
  用独立 `forward_ok`/`backward_ok` 标志替代
- 合并文本: 原项目直接拼接且英文模式下前缀未被剥离（混乱格式），
  改为先剥离前缀再用 "; " 分隔
- 合并 source 字段: 原项目 `forget_memory.py` 用 memory_id（唯一 ID）导致
  相邻合并完全失效，改为 date_key（同日期共享）使合并逻辑可用
- 原项目 `similarity_search_with_score_by_vector` 中 `len(docs)` 误用了
  结果累积器变量的大小而非索引总条目数，限制了邻居搜索范围
"""

import json
import logging
import math
import os
import random
import shutil
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

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

logger = logging.getLogger(__name__)


TAG = "MEMORYBANK"
USER_ID_PREFIX = "memorybank"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
# [DIFF] 原项目 CHUNK_SIZE=200，适配中文短对话（~80字符/对）。
# 本测试集为英文长对话（平均 ~272 字符/对），200 会导致合并逻辑完全失效，
# 因此改为 500 以保留与原项目等价的合并效果（合并 2-3 个邻居）。
DEFAULT_CHUNK_SIZE = 500
MEMORY_SKIP_TYPES = frozenset({"daily_summary"})


def _resolve_chunk_size() -> int:
    """从环境变量 MEMORYBANK_CHUNK_SIZE 解析分块大小。"""
    raw = os.getenv("MEMORYBANK_CHUNK_SIZE")
    if raw is not None:
        try:
            parsed = int(raw)
        except ValueError:
            logger.warning(
                "MemoryBank: MEMORYBANK_CHUNK_SIZE=%r is not a valid int, "
                "falling back to %d", raw, DEFAULT_CHUNK_SIZE,
            )
            return DEFAULT_CHUNK_SIZE
        if parsed <= 0:
            logger.warning(
                "MemoryBank: MEMORYBANK_CHUNK_SIZE=%d is not positive, "
                "falling back to %d", parsed, DEFAULT_CHUNK_SIZE,
            )
            return DEFAULT_CHUNK_SIZE
        return parsed
    return DEFAULT_CHUNK_SIZE


CHUNK_SIZE = _resolve_chunk_size()
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

STORE_ROOT = os.environ.get(
    "MEMORYBANK_STORE_ROOT",
    os.path.join(_ROOT_DIR, "log", "memorybank"),
)


def _resolve_embedding_api_key(args) -> Optional[str]:
    """获取 Embedding API 密钥，优先使用命令行参数，回退到环境变量。"""
    return getattr(args, "embedding_api_key", None) or resolve_memory_key(args, "EMBEDDING_API_KEY")


def _resolve_embedding_api_base(args) -> Optional[str]:
    """获取 Embedding API 基础 URL，优先使用命令行参数，回退到环境变量。"""
    return getattr(args, "embedding_api_base", None) or resolve_memory_url(args, "EMBEDDING_API_BASE")


def _resolve_reference_date() -> Optional[str]:
    """从环境变量 MEMORYBANK_REFERENCE_DATE 读取参考日期。"""
    return os.getenv("MEMORYBANK_REFERENCE_DATE")


_TRUTHY_TOKENS = frozenset({"1", "true", "yes", "on", "y"})
_FALSY_TOKENS = frozenset({"0", "false", "no", "off", "n"})


def _parse_bool_token(raw: str) -> Optional[bool]:
    """将非空字符串解析为布尔值，无法识别时返回 None。"""
    normalized = raw.strip().lower()
    if normalized in _TRUTHY_TOKENS:
        return True
    if normalized in _FALSY_TOKENS:
        return False
    return None


def _resolve_bool_env(name: str, default: bool) -> bool:
    """从环境变量解析布尔值，支持常见 truthy/falsy 词元。"""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    parsed = _parse_bool_token(value)
    if parsed is not None:
        return parsed
    logger.warning(
        "MemoryBank: env %s=%r not recognized as boolean "
        "(truthy: %s, falsy: %s); treating as False",
        name, value, sorted(_TRUTHY_TOKENS), sorted(_FALSY_TOKENS),
    )
    return False


def _resolve_enable_summary() -> bool:
    """从环境变量 MEMORYBANK_ENABLE_SUMMARY 读取是否启用摘要生成。"""
    return _resolve_bool_env("MEMORYBANK_ENABLE_SUMMARY", True)


def _resolve_enable_forgetting() -> bool:
    """从环境变量 MEMORYBANK_ENABLE_FORGETTING 读取是否启用遗忘机制。"""
    # [DIFF] 原项目遗忘机制始终启用。本测评场景默认禁用，以保证结果可复现性。
    # 需要启用时设置 MEMORYBANK_ENABLE_FORGETTING=1。
    new_val = os.getenv("MEMORYBANK_ENABLE_FORGETTING")
    if new_val is not None:
        if new_val.strip():
            parsed = _parse_bool_token(new_val)
            if parsed is not None:
                return parsed
            logger.warning(
                "MemoryBank: MEMORYBANK_ENABLE_FORGETTING=%r not recognized as boolean",
                new_val,
            )
        return False
    old_val = os.getenv("MEMORYBANK_DISABLE_FORGETTING")
    if old_val is not None and old_val.strip():
        logger.warning(
            "MemoryBank: MEMORYBANK_DISABLE_FORGETTING is deprecated; "
            "use MEMORYBANK_ENABLE_FORGETTING instead "
            "(MEMORYBANK_DISABLE_FORGETTING=1 means enable_forgetting=False)"
        )
        parsed = _parse_bool_token(old_val)
        if parsed is not None:
            return not parsed
        return False
    return False


def _resolve_seed() -> Optional[int]:
    """从环境变量 MEMORYBANK_SEED 读取随机种子。"""
    raw = os.getenv("MEMORYBANK_SEED")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            logger.warning(
                "MemoryBank: MEMORYBANK_SEED=%r is not a valid int, "
                "falling back to None", raw,
            )
            return None
    return None


def _resolve_store_root(args) -> str:
    """获取存储根目录，优先使用命令行参数，回退到环境变量或默认值。"""
    return getattr(args, "store_root", None) or STORE_ROOT


def _user_store_dir(user_id: str, store_root: str = STORE_ROOT) -> str:
    """返回指定用户的存储目录路径。"""
    return os.path.join(store_root, f"user_{user_id}")


def _strip_source_prefix(text: str, date_part: str) -> str:
    """去除对话内容或摘要的前缀标记。"""
    # [DIFF] 原项目 search_memory 仅去除中文前缀 `时间{date}的对话内容：`，
    # 英文模式下前缀不会被去除（bug）。此处正确处理英文前缀。
    for pfx in (
        f"Conversation content on {date_part}:",
        f"The summary of the conversation on {date_part} is:",
    ):
        if text.startswith(pfx):
            return text[len(pfx):]
    return text


def _group_consecutive(indices: List[int]) -> List[List[int]]:
    """将整数列表按连续性拆分为若干子列表。"""
    if not indices:
        return []
    result: List[List[int]] = []
    current = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i - 1] + 1 == indices[i]:
            current.append(indices[i])
        else:
            result.append(current)
            current = [indices[i]]
    result.append(current)
    return result


class MemoryBankClient:

    def __init__(
        self,
        *,
        embedding_api_base: str,
        embedding_api_key: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        enable_forgetting: bool = False,
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

        self._extra_metadata: Dict[str, dict] = {}

        self._embedding_client = _openai.OpenAI(
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
        """调用 Embedding API 将文本列表转为向量，带指数退避重试。"""
        max_retries = 5
        resp = None
        for attempt in range(max_retries):
            try:
                resp = self._embedding_client.embeddings.create(
                    input=texts,
                    model=self.embedding_model,
                )
                break
            except Exception:
                if attempt < max_retries - 1:
                    jitter = self._rng.random()
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
        """获取或创建用户的 FAISS 索引和元数据列表，支持从磁盘加载已有索引。"""
        if user_id in self._indices:
            return self._indices[user_id], self._metadata[user_id]

        store_dir = _user_store_dir(user_id, self._store_root)
        index_path = os.path.join(store_dir, "index.faiss")
        meta_path = os.path.join(store_dir, "metadata.json")
        extra_path = os.path.join(store_dir, "extra_metadata.json")

        if os.path.isfile(index_path) and os.path.isfile(meta_path):
            index = faiss.read_index(index_path)
            if not isinstance(index, faiss.IndexIDMap):
                dim = index.d
                # [DIFF] 原项目使用 LangChain FAISS 封装（默认 IndexFlatL2）。
                # 本实现改用原生 FAISS + IndexFlatIP（内积），配合 L2 归一化
                # 等价于余弦相似度，更适合 OpenAI Embedding API 的场景。
                new_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
                n = index.ntotal
                if n > 0:
                    all_vecs = index.reconstruct_n(0, n)
                    faiss.normalize_L2(all_vecs)
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
            if os.path.isfile(extra_path):
                with open(extra_path, "r", encoding="utf-8") as f:
                    self._extra_metadata[user_id] = json.load(f)
        else:
            dim = self._embedding_dim or 1536
            index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
            metadata = []
            self._next_id[user_id] = 0

        self._indices[user_id] = index
        self._metadata[user_id] = metadata
        return index, metadata

    def _allocate_id(self, user_id: str) -> int:
        """为指定用户分配一个递增的 FAISS 向量 ID。"""
        vector_id = self._next_id.get(user_id, 0)
        self._next_id[user_id] = vector_id + 1
        return vector_id

    def _add_vector(self, user_id: str, text: str, embedding: List[float], timestamp: str, extra_meta: Optional[dict] = None) -> None:
        """向用户索引中添加一条向量记录及对应元数据。"""
        index, metadata = self._get_or_create_index(user_id)
        vector_id = self._allocate_id(user_id)
        vec = np.array([embedding], dtype=np.float32)
        # [DIFF] 原项目使用 L2 距离无需归一化。改用 IndexFlatIP 后必须 L2 归一化，
        # 否则内积不等价于余弦相似度，未归一化的向量模长会偏置检索结果。
        faiss.normalize_L2(vec)
        index.add_with_ids(vec, np.array([vector_id], dtype=np.int64))
        meta_entry = {
            "text": text,
            "timestamp": timestamp,
            "memory_strength": 1,
            "last_recall_date": timestamp[:10] if len(timestamp) >= 10 else timestamp,
            "faiss_id": vector_id,
        }
        if extra_meta:
            meta_entry.update(extra_meta)
        metadata.append(meta_entry)

    def _merge_neighbors(self, results: List[dict], user_id: str) -> List[dict]:
        """合并检索结果中来自同一来源的相邻条目，减少碎片化。"""
        # [DIFF] 原项目有两处关键 bug：
        # 1. source=memory_id（唯一 ID）→ 相邻条目永远不会共享 source，
        #    合并逻辑完全失效。本实现 source=date_key（同日期共享）。
        # 2. 邻居遍历中 `break` 只跳出内层 `for l` 循环，导致另一方向
        #    被跳过。本实现用独立 `forward_ok`/`backward_ok` 标志修复。
        # 3. 合并后的文本直接拼接（无分隔符），英文模式下前缀未被剥离，
        #    格式混乱。本实现先剥离前缀再用 "; " 连接。
        if not results:
            return results

        metadata = self._metadata.get(user_id, [])
        if not metadata:
            return results

        indexed = [(r, r["_meta_idx"]) for r in results if r.get("_meta_idx") is not None]
        if not indexed:
            return results
        non_indexed = [r for r in results if r.get("_meta_idx") is None]

        merged_results: List[dict] = []
        seen_indices: Set[int] = set()

        for r, meta_idx in indexed:
            score = float(r.get("score", 0.0))
            source = r.get("source", "")
            total_length = len(metadata[meta_idx].get("text", ""))

            neighbor_indices: List[int] = [meta_idx]

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
                        if total_length + len(neighbor_text) > CHUNK_SIZE:
                            forward_ok = False
                        else:
                            total_length += len(neighbor_text)
                            neighbor_indices.append(neighbor_pos)

                if backward_ok:
                    neighbor_pos = meta_idx - offset
                    if neighbor_pos < 0:
                        backward_ok = False
                    elif metadata[neighbor_pos].get("source") != source:
                        backward_ok = False
                    else:
                        neighbor_text = metadata[neighbor_pos].get("text", "")
                        if total_length + len(neighbor_text) > CHUNK_SIZE:
                            backward_ok = False
                        else:
                            total_length += len(neighbor_text)
                            neighbor_indices.append(neighbor_pos)

            sorted_local = sorted(neighbor_indices)
            contiguous_groups = _group_consecutive(sorted_local)

            for contiguous in contiguous_groups:
                new_indices = [i for i in contiguous if i not in seen_indices]
                if not new_indices:
                    continue

                for run in _group_consecutive(new_indices):
                    for i in run:
                        seen_indices.add(i)

                    parts: List[str] = []
                    for i in run:
                        t = metadata[i].get("text", "")
                        src = metadata[i].get("source", "")
                        date_part = src.removeprefix("summary_")
                        t = _strip_source_prefix(t, date_part)
                        parts.append(t.strip())
                    # [DIFF] 原项目将合并文本直接拼接（无分隔符），英文模式下前缀未被剥离，
                    # 导致合并后出现重复前缀和混乱格式。此处用 "; " 连接并剥离前缀。
                    combined_text = "; ".join(parts)
                    base_meta = dict(metadata[run[0]])
                    base_meta["text"] = combined_text
                    base_meta["score"] = float(score)
                    base_meta["memory_strength"] = max(
                        metadata[i].get("memory_strength", 1) for i in run
                    )
                    base_meta["_merged_indices"] = run
                    merged_results.append(base_meta)

        merged_results.extend(non_indexed)
        merged_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return merged_results

    def save_index(self, user_id: str) -> None:
        """将用户的 FAISS 索引和元数据持久化到磁盘。"""
        if user_id not in self._indices:
            return
        store_dir = _user_store_dir(user_id, self._store_root)
        os.makedirs(store_dir, mode=0o700, exist_ok=True)
        index_path = os.path.join(store_dir, "index.faiss")
        meta_path = os.path.join(store_dir, "metadata.json")
        extra_path = os.path.join(store_dir, "extra_metadata.json")
        faiss.write_index(self._indices[user_id], index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata[user_id], f, ensure_ascii=False, indent=2)
        extra = self._extra_metadata.get(user_id, {})
        if extra:
            with open(extra_path, "w", encoding="utf-8") as f:
                json.dump(extra, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _parse_speaker(line: str) -> Tuple[str, str]:
        """从对话行中解析说话人和内容，格式为 "Speaker: content"。

        [DIFF] 原项目使用固定标签 `[|User|]` / `[|AI|]`（用户↔AI 双人对话）。
        本测试集为多用户车载场景（如 Gary、Justin、Patricia 等），
        需要动态解析说话人名称以保留身份信息。
        """
        colon_pos = line.find(": ")
        if colon_pos > 0:
            return line[:colon_pos].strip(), line[colon_pos + 2:].strip()
        return "Speaker", line.strip()

    def add(self, messages: List[dict], user_id: str, timestamp: str) -> None:
        """将对话消息分对编码为向量并存入用户索引。"""
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

        pair_texts: List[str] = []
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
            pair_texts.append(formatted)

        embeddings = self._get_embeddings(pair_texts)

        for text, emb in zip(pair_texts, embeddings, strict=True):
            self._add_vector(
                user_id, text, emb, timestamp,
                # [DIFF] 原项目 source=memory_id（每个对话独立，如 f'{user}_{date}_{i}'），
                # 导致合并逻辑实际无效。本实现 source=date_key（同日期共享），使同一日期的
                # 连续条目可在 _merge_neighbors 中合并，检索结果更连贯。
                extra_meta={"source": date_key},
            )

    def _call_llm(self, last_user_content: str) -> str:
        """调用 LLM 生成回复，带重试逻辑处理可恢复的 API 错误和上下文长度回退。"""
        if not self._llm_client:
            return ""
        max_retries = 3
        content = last_user_content
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
                        {"role": "user", "content": content},
                    ],
                    max_tokens=400,
                    temperature=0.7,
                    top_p=1.0,
                    frequency_penalty=0.4,
                    presence_penalty=0.2,
                    # [DIFF] 原项目含 stop=["<|im_end|>", "¬人类¬"]，为 ChatGLM/
                    # BELLE 模型和中文场景专用。英文 OpenAI 兼容 API 无需设置。
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                import openai as _openai_exc

                context_exceeded = any(
                    pattern in str(exc).lower()
                    for pattern in (
                        "maximum context",
                        "context length",
                        "too long",
                        "reduce the length",
                    )
                )
                if context_exceeded and attempt < max_retries - 1:
                    cut_length = max(1800 - 200 * attempt, 500)
                    logger.warning(
                        "MemoryBank _call_llm context length exceeded, "
                        "trimming to last %d chars (attempt %d/%d)",
                        cut_length, attempt + 1, max_retries,
                    )
                    content = content[-cut_length:]
                    continue

                retryable = isinstance(exc, (
                    _openai_exc.APIConnectionError,
                    _openai_exc.APITimeoutError,
                    _openai_exc.RateLimitError,
                ))
                if not retryable and isinstance(exc, _openai_exc.APIStatusError):
                    retryable = exc.status_code >= 500

                if retryable and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue

                if not retryable:
                    raise

                logger.warning(
                    "MemoryBank _call_llm failed after %d retries: %s",
                    max_retries, exc,
                )
                return ""

    def _summarize(self, text: str) -> str:
        """调用 LLM 对对话文本生成摘要。"""
        return self._call_llm(
            "Please summarize the following dialogue as concisely as "
            "possible, extracting the main themes and key information. "
            "If there are multiple key events, you may summarize them "
            f"separately. Dialogue content:\n{text}\n"
            "Summarization："  # noqa: RUF001
        )

    def _generate_daily_summaries(self, user_id: str) -> None:
        """按日期聚合对话内容并生成每日摘要向量。"""
        if not self._llm_client:
            return

        metadata = self._metadata.get(user_id, [])
        existing_summary_dates = {
            (m.get("source") or "").removeprefix("summary_")
            for m in metadata
            if m.get("type") == "daily_summary"
        }
        daily_texts: Dict[str, List[str]] = {}
        for meta in metadata:
            if meta.get("type") == "daily_summary":
                continue
            date_key = meta.get("source", meta.get("timestamp", "")[:10])
            if not date_key or date_key in existing_summary_dates:
                continue
            daily_texts.setdefault(date_key, []).append(meta["text"])

        for date_key, texts in sorted(daily_texts.items()):
            cleaned = [_strip_source_prefix(t, date_key).strip() for t in texts]
            combined = "\n".join(cleaned)
            summary = self._summarize(combined)
            if summary:
                summary_text = (
                    f"The summary of the conversation on {date_key} is: {summary}"
                )
                ts = f"{date_key}T00:00:00"
                summary_emb = self._get_embeddings([summary_text])[0]
                self._add_vector(user_id, summary_text, summary_emb, ts,
                                 {"type": "daily_summary", "source": f"summary_{date_key}"})

    def _generate_overall_summary(self, user_id: str) -> None:
        """基于所有每日摘要生成整体摘要，存入额外元数据。"""
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
            raw_source = m.get("source")
            date = (raw_source if raw_source else m.get("timestamp", "")[:10]).removeprefix("summary_")
            text = m["text"]
            prefix = f"The summary of the conversation on {date} is: "
            if text.startswith(prefix):
                text = text[len(prefix):]
            summary_parts.append((date, text))

        prompt_parts = [
            "Please provide a highly concise summary of the following event, "
            "capturing the essential key information as succinctly as possible. "
            "Summarize the event:\n",
        ]
        for date, text in summary_parts:
            prompt_parts.append(f"\nAt {date}, the events are {text.strip()}")
        prompt_parts.append("\nSummarization：")  # noqa: RUF001
        prompt = "".join(prompt_parts)

        summary = self._call_llm(prompt)
        if summary:
            extra = self._extra_metadata.setdefault(user_id, {})
            extra["overall_summary"] = summary

    def _analyze_personality(self, text: str) -> str:
        """调用 LLM 分析对话中体现的用户性格特征和情绪。"""
        # [DIFF] 原项目使用具体用户名 `{user_name}'s personality traits...`
        # 和 AI 名称 `{boot_name}'s response strategy`。
        # 本测试集为多用户场景，无单一用户/AI 对应关系，改为通用表述。
        return self._call_llm(
            "Based on the following dialogue, please summarize the user's "
            "personality traits and emotions, and devise response strategies "
            f"based on your speculation. Dialogue content:\n{text}\n"
            "The user's personality traits, emotions, and the AI's response "
            "strategy are:"
        )

    def _generate_daily_personalities(self, user_id: str) -> None:
        """按日期聚合对话并分析每日用户性格，存入额外元数据。"""
        if not self._llm_client:
            return

        metadata = self._metadata.get(user_id, [])
        extra = self._extra_metadata.setdefault(user_id, {})
        existing_personalities = extra.setdefault("daily_personalities", {})
        daily_texts: Dict[str, List[str]] = {}
        for meta in metadata:
            if meta.get("type") == "daily_summary":
                continue
            date_key = meta.get("source", meta.get("timestamp", "")[:10])
            if not date_key or date_key in existing_personalities:
                continue
            daily_texts.setdefault(date_key, []).append(meta["text"])

        for date_key, texts in sorted(daily_texts.items()):
            cleaned = [_strip_source_prefix(t, date_key).strip() for t in texts]
            combined = "\n".join(cleaned)
            personality = self._analyze_personality(combined)
            if personality:
                existing_personalities[date_key] = personality

    def _generate_overall_personality(self, user_id: str) -> None:
        """基于每日性格分析生成整体性格画像，存入额外元数据。"""
        if not self._llm_client:
            return

        extra = self._extra_metadata.get(user_id, {})
        daily_personalities = extra.get("daily_personalities", {})
        if not daily_personalities:
            return

        prompt_parts = [
            "The following are the user's exhibited personality traits and emotions "
            "throughout multiple dialogues, along with appropriate response strategies "
            "for the current situation:\n",
        ]
        for date, text in sorted(daily_personalities.items()):
            prompt_parts.append(f"\nAt {date}, the analysis shows {text.strip()}")
        prompt_parts.append(
            # [DIFF] 原项目为 "AI lover"（AI 伴侣场景），改为 "AI"（车载助手场景）。
            "\nPlease provide a highly concise and general summary of the user's "
            "personality and the most appropriate response strategy for the AI, "
            "summarized as:"
        )
        prompt = "".join(prompt_parts)

        personality = self._call_llm(prompt)
        if personality:
            extra = self._extra_metadata.setdefault(user_id, {})
            extra["overall_personality"] = personality

    def _forgetting_retention(self, days_elapsed: float, memory_strength: int) -> float:
        """基于艾宾浩斯遗忘曲线计算记忆保留概率。

        [DIFF] 原项目公式为 `math.exp(-t / 5*S)`，因 Python 运算符优先级
        实际计算为 `math.exp((-t/5) * S)`，导致 memory_strength 越大遗忘越多，
        与艾宾浩斯曲线和代码注释的描述矛盾。此处修正为正确公式
        `math.exp(-t / (5*S))`，使 strength 越大保留率越高。
        """
        return min(1.0, math.exp(-days_elapsed / (5 * memory_strength)))

    def _forget_at_ingestion(self, user_id: str) -> None:
        """在数据摄入阶段根据遗忘曲线概率性地丢弃部分记忆。"""
        if not self.enable_forgetting or not self.reference_date:
            return

        index, metadata = self._get_or_create_index(user_id)

        ref_dt = datetime.strptime(self.reference_date[:10], "%Y-%m-%d")
        ids_to_remove: List[int] = []
        indices_to_keep: List[int] = []

        for i, meta in enumerate(metadata):
            if meta.get("type") in MEMORY_SKIP_TYPES:
                indices_to_keep.append(i)
                continue

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
        """基于向量相似度检索与查询最相关的记忆，并合并相邻条目。"""
        index, metadata = self._get_or_create_index(user_id)

        if index.ntotal == 0:
            return []

        query_emb = self._get_embeddings([query])[0]
        query_vec = np.array([query_emb], dtype=np.float32)
        # [DIFF] 同 _add_vector，查询向量也需 L2 归一化以保证 IP ≈ 余弦相似度。
        faiss.normalize_L2(query_vec)

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
            mi = r.get("_meta_idx")
            if mi is not None and 0 <= mi < len(metadata):
                metadata[mi]["memory_strength"] = metadata[mi].get("memory_strength", 1) + 1
                if self.reference_date:
                    metadata[mi]["last_recall_date"] = self.reference_date[:10]
        # [DIFF] 原项目 search 后仅更新 history 中对话条目的 memory_strength，
        # 不更新 daily_summary 条目的强度。本实现对所有搜索结果均更新强度以保持
        # 语义一致性；当前 summary 属于 MEMORY_SKIP_TYPES，故不影响遗忘行为。

        merged = self._merge_neighbors(results, user_id)

        for r in merged:
            r.pop("_merged_indices", None)
            r.pop("_meta_idx", None)

        # [DIFF] 原项目在 search 后通过 update_memory_when_searched → write_memories
        # 持久化 memory_strength 和 last_recall_date。缺少此步会导致遗忘机制跨会话失效。
        self.save_index(user_id)
        return merged

    def get_extra_metadata(self, user_id: str) -> dict:
        """获取用户的额外元数据（整体摘要、性格画像等）。"""
        return self._extra_metadata.get(user_id, {})


def validate_add_args(args) -> None:
    """验证 add 操作所需的 Embedding API 凭据是否已提供。"""
    require_value(
        _resolve_embedding_api_key(args),
        "Embedding API key is required: pass --memory_key or set MEMORY_KEY/EMBEDDING_API_KEY",
    )
    require_value(
        _resolve_embedding_api_base(args),
        "Embedding API base URL is required: pass --memory_url or set MEMORY_URL/EMBEDDING_API_BASE",
    )


def validate_test_args(args) -> None:
    """验证测试操作所需参数，委托给 validate_add_args。"""
    validate_add_args(args)


_warned_llm_fallback = False


def _resolve_llm_credentials(
    args: Any,
    api_base: str,
    api_key: str,
) -> Tuple[str, str]:
    """解析 LLM API 凭据，缺失时回退到 Embedding API 凭据并记录警告。"""
    global _warned_llm_fallback

    explicit_base = resolve_memory_url(args, "LLM_API_BASE")
    explicit_key = resolve_memory_key(args, "LLM_API_KEY")

    if explicit_base and explicit_key:
        return explicit_base, explicit_key

    if not _warned_llm_fallback:
        _warned_llm_fallback = True
        if explicit_base and not explicit_key:
            logger.warning(
                "MemoryBank: LLM_API_KEY not set; using embedding API key "
                "for LLM calls (LLM_API_BASE is explicit)"
            )
        elif explicit_key and not explicit_base:
            logger.warning(
                "MemoryBank: LLM_API_BASE not set; using embedding API base "
                "for LLM calls (LLM_API_KEY is explicit)"
            )
        else:
            logger.warning(
                "MemoryBank: LLM_API_BASE and LLM_API_KEY not set; "
                "falling back to embedding API credentials for LLM calls"
            )

    return explicit_base or api_base, explicit_key or api_key


def _build_client(args, user_id: str = "") -> MemoryBankClient:
    """根据命令行参数和环境变量构建 MemoryBankClient 实例。"""
    api_key = require_value(
        _resolve_embedding_api_key(args),
        "Embedding API key is required: pass --memory_key or set MEMORY_KEY/EMBEDDING_API_KEY",
    )
    api_base = require_value(
        _resolve_embedding_api_base(args),
        "Embedding API base URL is required: pass --memory_url or set MEMORY_URL/EMBEDDING_API_BASE",
    )

    enable_summary = _resolve_enable_summary()
    enable_forgetting = _resolve_enable_forgetting()
    seed = _resolve_seed()
    reference_date = _resolve_reference_date()

    llm_api_base, llm_api_key = _resolve_llm_credentials(args, api_base, api_key)
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    return MemoryBankClient(
        embedding_api_base=api_base,
        embedding_api_key=api_key,
        embedding_model=getattr(args, "embedding_model", None) or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        enable_forgetting=enable_forgetting,
        enable_summary=enable_summary,
        seed=seed,
        reference_date=reference_date,
        llm_api_base=llm_api_base,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        store_root=_resolve_store_root(args),
    )


def _compute_reference_date(history_dir: str, file_range: Optional[str]) -> str:
    """扫描历史文件中的时间戳，计算最新日期的下一天作为参考日期。"""
    # [DIFF] 原项目使用 `datetime.date.today()` 作为参考日期，可能距最后对话
    # 数周/数月（遗忘更激进）。本实现使用历史文件最新日期的下一天，使遗忘量
    # 保持合理且结果可复现，适合测评场景。
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
    """将对话历史摄入到 MemoryBank，构建向量索引并可选生成摘要和遗忘。"""
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
            daily_lines: Dict[str, List[str]] = {}
            for bucket in load_hourly_history(history_path):
                if bucket.dt:
                    day_key = bucket.dt.strftime("%Y-%m-%d")
                else:
                    day_key = datetime.now().strftime("%Y-%m-%d")
                daily_lines.setdefault(day_key, []).extend(bucket.lines)

            for day_key, lines in sorted(daily_lines.items()):
                ts = f"{day_key}T00:00:00"
                messages = [{"role": "user", "content": "\n".join(lines)}]
                client.add(messages=messages, user_id=user_id, timestamp=ts)
                message_count += len(lines)

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
    """初始化测试状态（MemoryBank 不需要共享状态）。"""
    del file_numbers, user_id_prefix
    validate_test_args(args)
    return None


def build_test_client(args, file_num: int, user_id_prefix: str, shared_state: Any):
    """构建用于测试的 MemoryBank 客户端包装器。"""
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
        """检索记忆并附带整体摘要和性格画像。"""
        uid = user_id if user_id is not None else self._user_id
        results = list(self._client.search(query=query, user_id=uid, top_k=top_k))

        extra = self._client.get_extra_metadata(uid)
        overall_summary = extra.get("overall_summary", "")
        overall_personality = extra.get("overall_personality", "")

        if overall_summary or overall_personality:
            parts = []
            if overall_summary:
                parts.append(
                    f"Overall summary of past memories: {overall_summary}"
                )
            if overall_personality:
                parts.append(
                    f"User personality and response strategy: {overall_personality}"
                )
            results.append({
                "_type": "overall_context",
                "text": "\n".join(parts),
                "source": "overall",
                "memory_strength": 1,
                "score": 0.0,
            })

        return results


def close_test_state(shared_state: Any) -> None:
    """清理测试状态（MemoryBank 无需清理）。"""
    del shared_state


def is_test_sequential() -> bool:
    """MemoryBank 测试支持并行执行。"""
    return False


def format_search_results(search_result: Any) -> Tuple[str, int]:
    """将检索结果格式化为带编号的文本，按日期分组并标注记忆强度。"""
    if not isinstance(search_result, list):
        return "", 0
    if not search_result:
        return "", 0

    overall_items = [r for r in search_result if r.get("_type") == "overall_context"]
    non_overall = [r for r in search_result if r.get("_type") != "overall_context"]
    sorted_results = sorted(
        non_overall,
        key=lambda r: (r.get("source") or "").removeprefix("summary_"),
    )

    groups: List[Tuple[str, str, List[dict]]] = []
    for item in sorted_results:
        text = item.get("text", "")
        date_part = (item.get("source") or "").removeprefix("summary_")
        text = _strip_source_prefix(text, date_part).strip()

        if not groups or groups[-1][0] != date_part:
            groups.append((date_part, text, [item]))
        else:
            groups[-1] = (groups[-1][0], groups[-1][1] + "\n" + text, groups[-1][2] + [item])

    lines: List[str] = []
    for idx, (date_part, combined_text, items) in enumerate(groups, 1):
        max_strength = max(it.get("memory_strength", 1) for it in items)
        date_info = f" [date={date_part}]" if date_part else ""
        lines.append(f"{idx}. [memory_strength={max_strength}]{date_info} {combined_text}")

    for idx, item in enumerate(overall_items, start=len(groups) + 1):
        lines.append(f"{idx}. [memory_strength={item.get('memory_strength', 1)}] {item.get('text', '')}")

    return "\n\n".join(lines), len(non_overall)
