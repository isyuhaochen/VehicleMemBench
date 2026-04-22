#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_DIR="${ROOT_DIR}/evaluation"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --file_range) FILE_RANGE="$2"; shift 2 ;;
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

for MEMORY_TYPE in none gold key_value summary; do
  WIT="${PREFIX}_think_${MEMORY_TYPE}"
  WORKERS=$(get_max_workers "$MEMORY_TYPE")
  if [[ "$MEMORY_TYPE" == "key_value" ]]; then
    REFLECT_NUM=20
  else
    REFLECT_NUM=10
  fi

  echo "=== Running ${MEMORY_TYPE} mode (model=${MODEL}, workers=${WORKERS}) ==="

  python model_evaluation.py \
    --memory_type "$MEMORY_TYPE" --enable_thinking true \
    --benchmark_dir "$BENCHMARK_DIR" \
    --file_range "$FILE_RANGE" \
    --api_base "$API_BASE" --api_key "$API_KEY" --model "$MODEL" \
    --prefix "$WIT" \
    --max_workers "$WORKERS" --reflect_num "$REFLECT_NUM"
done
