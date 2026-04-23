#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_DIR="${ROOT_DIR}/evaluation"
HISTORY_DIR="${ROOT_DIR}/benchmark/history"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --file_range)
      if [[ $# -lt 2 ]]; then
        echo "Error: --file_range requires a value" >&2; exit 1
      fi
      FILE_RANGE="$2"; shift 2
      ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done
FILE_RANGE="${FILE_RANGE:-1-50}"

echo "Available models:"
echo "  1) glm-4.5-air"
echo "  2) glm-5.1"
echo "  3) qwen3.5-9b"
echo ""
read -rp "Select model [1-3]: " MODEL_CHOICE

case "$MODEL_CHOICE" in
  1)
    API_BASE="https://open.bigmodel.cn/api/coding/paas/v4"
    API_KEY="${ZHIPU_API_KEY:?Set ZHIPU_API_KEY}"
    MODEL="glm-4.5-air"
    PREFIX="glm45"
    ;;
  2)
    API_BASE="https://open.bigmodel.cn/api/coding/paas/v4"
    API_KEY="${ZHIPU_API_KEY:?Set ZHIPU_API_KEY}"
    MODEL="glm-5.1"
    PREFIX="glm51"
    ;;
  3)
    API_BASE="https://openrouter.ai/api/v1"
    API_KEY="${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY}"
    MODEL="qwen/qwen3.5-9b"
    PREFIX="qwen35"
    ;;
  *)
    echo "Invalid choice: $MODEL_CHOICE"
    exit 1
    ;;
esac

echo ""
echo "Available memory types:"
echo "  1) none"
echo "  2) gold"
echo "  3) key_value"
echo "  4) summary"
echo "  5) memorybank"
echo ""
read -rp "Select memory types [1-4,5 / all] (default 1-4): " MEM_INPUT
MEM_INPUT="${MEM_INPUT:-1,2,3,4}"

SELECTED_TYPES=()
USE_MEMORYBANK=false
if [[ "${MEM_INPUT,,}" == "all" ]]; then
  SELECTED_TYPES=("none" "gold" "key_value" "summary")
  USE_MEMORYBANK=true
else
  IFS=',' read -ra _CHOICES <<< "$MEM_INPUT"
  for _c in "${_CHOICES[@]}"; do
    _c="${_c// /}"
    case "$_c" in
      1) SELECTED_TYPES+=("none") ;;
      2) SELECTED_TYPES+=("gold") ;;
      3) SELECTED_TYPES+=("key_value") ;;
      4) SELECTED_TYPES+=("summary") ;;
      5) USE_MEMORYBANK=true ;;
      *) echo "Unknown memory type choice: $_c"; exit 1 ;;
    esac
  done
fi

if [[ ${#SELECTED_TYPES[@]} -eq 0 && "$USE_MEMORYBANK" == false ]]; then
  echo "No memory types selected."
  exit 1
fi

if [[ "$USE_MEMORYBANK" == true ]]; then
  echo ""
  echo "Available embedding models:"
  echo "  1) bge-m3 (via OpenRouter)"
  echo ""
  read -rp "Select embedding model [1]: " EMB_CHOICE
  EMB_CHOICE="${EMB_CHOICE:-1}"

  case "$EMB_CHOICE" in
    1)
      EMBEDDING_API_BASE="https://openrouter.ai/api/v1"
      EMBEDDING_API_KEY="${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY}"
      EMBEDDING_MODEL="baai/bge-m3"
      ;;
    *)
      echo "Invalid choice: $EMB_CHOICE"
      exit 1
      ;;
  esac
fi

get_max_workers() {
  local memory_type="$1"
  if [[ "$MODEL_CHOICE" == "3" ]]; then
    case "$memory_type" in
      key_value|summary) echo 8 ;;
      *) echo 10 ;;
    esac
  else
    echo 3
  fi
}

BENCHMARK_DIR="${ROOT_DIR}/benchmark/qa_data"

cd "$EVAL_DIR"

for MEMORY_TYPE in "${SELECTED_TYPES[@]}"; do
  WIT="${PREFIX}_think_${MEMORY_TYPE}"
  WORKERS=$(get_max_workers "$MEMORY_TYPE")
  if [[ "$MEMORY_TYPE" == "key_value" ]]; then
    REFLECT_NUM=20
  else
    REFLECT_NUM=10
  fi

  echo "=== Running ${MEMORY_TYPE} mode (model=${MODEL}, workers=${WORKERS}) ==="

  uv run model_evaluation.py \
    --memory_type "$MEMORY_TYPE" --enable_thinking true \
    --benchmark_dir "$BENCHMARK_DIR" \
    --file_range "$FILE_RANGE" \
    --api_base "$API_BASE" --api_key "$API_KEY" --model "$MODEL" \
    --prefix "$WIT" \
    --max_workers "$WORKERS" --reflect_num "$REFLECT_NUM"
done

if [[ "$USE_MEMORYBANK" == true ]]; then
  MB_PREFIX="${PREFIX}_think_memorybank"
  MODEL_SAFE="${MODEL//\//_}"
  SESSION_TS=$(date +%Y%m%d_%H%M%S)
  MB_STORE_ROOT="${ROOT_DIR}/log/${MB_PREFIX}_${MODEL_SAFE}_${SESSION_TS}"

  echo "=== Running memorybank add stage (store_root=${MB_STORE_ROOT}) ==="
  uv run memorysystem_evaluation.py add \
    --memory_system memorybank \
    --history_dir "$HISTORY_DIR" \
    --file_range "$FILE_RANGE" \
    --max_workers 3 \
    --store_root "$MB_STORE_ROOT" \
    --embedding_api_base "$EMBEDDING_API_BASE" \
    --embedding_api_key "$EMBEDDING_API_KEY" \
    --embedding_model "$EMBEDDING_MODEL"

  echo "=== Running memorybank test stage (model=${MODEL}) ==="
  uv run memorysystem_evaluation.py test \
    --memory_system memorybank \
    --benchmark_dir "$BENCHMARK_DIR" \
    --file_range "$FILE_RANGE" \
    --api_base "$API_BASE" --api_key "$API_KEY" --model "$MODEL" \
    --prefix "$MB_PREFIX" \
    --enable_thinking true \
    --reflect_num 10 \
    --max_workers 3 \
    --output_dir "${ROOT_DIR}/log" \
    --store_root "$MB_STORE_ROOT" \
    --embedding_api_base "$EMBEDDING_API_BASE" \
    --embedding_api_key "$EMBEDDING_API_KEY" \
    --embedding_model "$EMBEDDING_MODEL"
fi
