"""
Unified memory-system ingestion and evaluation for VehicleMemBench.

This file contains two stages:
1. `add`: write benchmark history into the selected memory system.
2. `test`: evaluate a model that can query the selected memory system.
"""

import argparse
import json
import logging
import os
import re
import statistics
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

    tqdm.write = print

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from environment.utils import modules_dict, save_json_file
from environment.vehicleworld import VehicleWorld
from evaluation.eval_utils import calculate_turn_result, score_tool_calls
from evaluation.memorysystems import SUPPORTED_MEMORY_SYSTEMS, get_system_module
from evaluation.memorysystems.common import parse_file_range


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def get_search_memory_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search user memory from the selected memory system. "
                "Use this when the query depends on user preferences or history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Memory retrieval query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of retrieved memories",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    }


def _collect_event_chain_numbers(benchmark_dir: str) -> List[int]:
    file_numbers = []
    for name in os.listdir(benchmark_dir):
        match = re.match(r"qa_(\d+)\.json$", name)
        if match:
            file_numbers.append(int(match.group(1)))
    return sorted(file_numbers)


@lru_cache(maxsize=1)
def _get_runtime_helpers():
    from evaluation.agent_client import AgentClient
    from evaluation.model_evaluation import (
        build_tool_env,
        create_chat_completion_with_retry,
        get_functions_schema_for_module,
        get_list_module_tools_schema,
        parse_answer_to_tools,
    )

    return {
        "AgentClient": AgentClient,
        "build_tool_env": build_tool_env,
        "create_chat_completion_with_retry": create_chat_completion_with_retry,
        "get_functions_schema_for_module": get_functions_schema_for_module,
        "get_list_module_tools_schema": get_list_module_tools_schema,
        "parse_answer_to_tools": parse_answer_to_tools,
    }


def process_task_with_memorysystem(
    *,
    task: Dict[str, Any],
    task_id: str,
    agent_client: Any,
    reflect_num: int,
    mem_client: Any,
    user_id: str,
    memory_module: Any,
) -> Optional[Dict[str, Any]]:
    try:
        runtime = _get_runtime_helpers()
        build_tool_env = runtime["build_tool_env"]
        create_chat_completion_with_retry = runtime["create_chat_completion_with_retry"]
        get_functions_schema_for_module = runtime["get_functions_schema_for_module"]
        get_list_module_tools_schema = runtime["get_list_module_tools_schema"]

        query = task["query"]
        reasoning_type = task.get("reasoning_type", "unknown")

        vw_pred = VehicleWorld()
        local_vars = build_tool_env(vw_pred)
        vw_ref = VehicleWorld()

        available_tools = [
            {"type": "function", "function": get_list_module_tools_schema()},
            get_search_memory_schema(),
        ]
        loaded_modules = set()

        modules_info = "\n".join(f"- {name}: {desc}" for name, desc in modules_dict.items())
        system_instruction = f"""
You are an intelligent in-car AI assistant responsible for fulfilling user requests by calling vehicle APIs.

You can query external memory if needed:
- search_memory(query, top_k): retrieve user memories from the selected memory system

Available modules:
{modules_info}

Rules:
1. Call search_memory first when the request depends on user preferences or history.
2. Use list_module_tools(module_name=...) to discover vehicle tools.
3. Call vehicle tools to satisfy the query.
4. Avoid unnecessary parameter changes if exact values are unavailable.
5. Do not repeatedly query the same memory information or invoke the same vehicle tool in consecutive steps unless new evidence requires it.
"""

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": query},
        ]

        input_token_list: List[int] = []
        output_token_list: List[int] = []
        pred_calls: List[Dict[str, Any]] = []
        pred_call_outputs: List[str] = []
        memory_tool_calls: List[Dict[str, Any]] = []

        for _ in range(reflect_num):
            try:
                response = create_chat_completion_with_retry(
                    agent_client,
                    model=agent_client.model,
                    messages=messages,
                    temperature=agent_client.temperature,
                    max_tokens=agent_client.max_tokens,
                    tools=available_tools,
                    tool_choice="auto",
                    context=f"memorysystem_eval task={task_id}",
                )
            except Exception as exc:
                tqdm.write(f"API Error: {exc}")
                break

            message = response.choices[0].message
            messages.append(message)

            usage = response.usage
            if usage:
                input_token_list.append(usage.prompt_tokens)
                output_token_list.append(usage.completion_tokens)

            tool_calls = message.tool_calls
            if not tool_calls:
                break

            for tool_call in tool_calls:
                func_name = tool_call.function.name
                args_str = tool_call.function.arguments
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {}

                if func_name == "search_memory":
                    memory_tool_calls.append({"name": func_name, "args": args})
                    search_query = args.get("query", query)
                    top_k = args.get("top_k", 5)
                    try:
                        top_k = int(top_k)
                    except Exception:
                        top_k = 5
                    top_k = max(1, min(top_k, 20))

                    try:
                        raw = mem_client.search(query=search_query, user_id=user_id, top_k=top_k)
                        formatted, count = memory_module.format_search_results(raw)
                        result = {
                            "success": True,
                            "memory_text": formatted,
                            "count": count,
                        }
                    except Exception as exc:
                        result = {"success": False, "error": str(exc)}

                elif func_name == "list_module_tools":
                    module_name = args.get("module_name", "")
                    if module_name not in loaded_modules:
                        module_functions = get_functions_schema_for_module(module_name)
                        if module_functions:
                            for func_schema in module_functions:
                                available_tools.append(
                                    {"type": "function", "function": func_schema}
                                )
                            loaded_modules.add(module_name)
                            result = {
                                "success": True,
                                "message": f"Loaded {len(module_functions)} tools from {module_name}",
                                "tools": [func["name"] for func in module_functions],
                            }
                        else:
                            result = {
                                "success": False,
                                "error": f"Module {module_name} not found",
                            }
                    else:
                        result = {
                            "success": True,
                            "message": f"Module {module_name} already loaded",
                        }

                else:
                    pred_calls.append({"name": func_name, "args": args})
                    func = local_vars.get(func_name)
                    if func and callable(func):
                        try:
                            result = func(**args)
                        except Exception as exc:
                            result = {"success": False, "error": str(exc)}
                    else:
                        result = {
                            "success": False,
                            "error": f"Function {func_name} not found",
                        }

                pred_call_outputs.append(str(result))
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        ref_calls = task.get("tools", [])
        ref_env = build_tool_env(vw_ref)
        for ref_call in ref_calls:
            name = ref_call.get("name")
            args = ref_call.get("args", {})
            func = ref_env.get(name)
            if callable(func):
                try:
                    func(**args)
                except Exception:
                    pass

        initial_world = VehicleWorld().to_dict()
        ref_world = vw_ref.to_dict()
        pred_world = vw_pred.to_dict()

        state_score = calculate_turn_result(
            initial_world,
            ref_world,
            initial_world,
            pred_world,
        )
        tool_score = score_tool_calls(pred_calls, ref_calls)

        exact_match = (
            len(state_score.get("differences", [])) == 0
            and state_score.get("FP", 0) == 0
            and state_score.get("negative_FP", 0) == 0
        )
        skipped = state_score.get("skipped", False)

        last_msg = messages[-1] if messages else None
        if hasattr(last_msg, "content"):
            final_response = last_msg.content if last_msg.content else ""
        elif isinstance(last_msg, dict):
            final_response = last_msg.get("content", "")
        else:
            final_response = ""

        return {
            "task_id": str(task_id),
            "query": query,
            "reasoning_type": reasoning_type,
            "pred_calls": pred_calls,
            "ref_calls": ref_calls,
            "memory_tool_calls": memory_tool_calls,
            "state_score": state_score,
            "tool_score": tool_score,
            "exact_match": exact_match,
            "skipped": skipped,
            "num_pred_calls": len(pred_calls),
            "num_ref_calls": len(ref_calls),
            "num_memory_calls": len(memory_tool_calls),
            "output_token": sum(output_token_list),
            "input_token": sum(input_token_list),
            "model_output": final_response,
            "pred_call_outputs": pred_call_outputs,
        }
    except Exception:
        stack_trace = "".join(traceback.format_exc())
        print(f"Task {task_id} error:\n{stack_trace}")
        return None


def _safe_mean(values) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def _get_pred_call_count(result: Dict[str, Any]) -> int:
    pred_call_outputs = result.get("pred_call_outputs", [])
    return len(pred_call_outputs) if isinstance(pred_call_outputs, list) else 0


def _build_metric(
    *,
    all_results: List[Dict[str, Any]],
    model: str,
    memory_system: str,
) -> Dict[str, Any]:
    valid_results = [result for result in all_results if not result.get("skipped", False)]
    skipped_results = [result for result in all_results if result.get("skipped", False)]

    metric = {
        "model": model,
        "memory_system": memory_system,
        "completed_tasks": len(all_results),
        "valid_tasks": len(valid_results),
        "skipped_tasks": len(skipped_results),
        "exact_match_rate": _safe_mean(
            1 if result.get("exact_match") else 0 for result in valid_results
        ),
        "change_accuracy": _safe_mean(
            result.get("state_score", {}).get("change_accuracy", 0)
            for result in valid_results
        ),
        "state_f1_positive": _safe_mean(
            result.get("state_score", {}).get("f1_positive", 0)
            for result in valid_results
        ),
        "state_f1_negative": _safe_mean(
            result.get("state_score", {}).get("f1_negative", 0)
            for result in valid_results
        ),
        "state_acc_positive": _safe_mean(
            result.get("state_score", {}).get("acc_positive", 0)
            for result in valid_results
        ),
        "state_precision_positive": _safe_mean(
            result.get("state_score", {}).get("precision_positive", 0)
            for result in valid_results
        ),
        "state_f1_change": _safe_mean(
            result.get("state_score", {}).get("f1_change", 0)
            for result in valid_results
        ),
        "state_acc_negative": _safe_mean(
            result.get("state_score", {}).get("acc_negative", 0)
            for result in valid_results
        ),
        "state_precision_change": _safe_mean(
            result.get("state_score", {}).get("precision_change", 0)
            for result in valid_results
        ),
        "avg_pred_calls": _safe_mean(
            _get_pred_call_count(result) for result in valid_results
        ),
        "avg_output_token": _safe_mean(
            result.get("output_token", 0) for result in valid_results
        ),
        "skipped_queries": [result.get("query", "")[:100] for result in skipped_results],
    }

    reasoning_types = sorted({result.get("reasoning_type", "unknown") for result in valid_results})
    by_reasoning_type = {}
    for reasoning_type in reasoning_types:
        type_results = [
            result for result in valid_results if result.get("reasoning_type") == reasoning_type
        ]
        if not type_results:
            continue
        by_reasoning_type[reasoning_type] = {
            "count": len(type_results),
            "exact_match_rate": _safe_mean(
                1 if result.get("exact_match") else 0 for result in type_results
            ),
            "change_accuracy": _safe_mean(
                result.get("state_score", {}).get("change_accuracy", 0)
                for result in type_results
            ),
            "state_f1_positive": _safe_mean(
                result.get("state_score", {}).get("f1_positive", 0)
                for result in type_results
            ),
            "state_f1_negative": _safe_mean(
                result.get("state_score", {}).get("f1_negative", 0)
                for result in type_results
            ),
            "state_acc_positive": _safe_mean(
                result.get("state_score", {}).get("acc_positive", 0)
                for result in type_results
            ),
            "state_precision_positive": _safe_mean(
                result.get("state_score", {}).get("precision_positive", 0)
                for result in type_results
            ),
            "state_f1_change": _safe_mean(
                result.get("state_score", {}).get("f1_change", 0)
                for result in type_results
            ),
            "state_acc_negative": _safe_mean(
                result.get("state_score", {}).get("acc_negative", 0)
                for result in type_results
            ),
            "state_precision_change": _safe_mean(
                result.get("state_score", {}).get("precision_change", 0)
                for result in type_results
            ),
            "avg_pred_calls": _safe_mean(
                _get_pred_call_count(result) for result in type_results
            ),
            "avg_output_token": _safe_mean(
                result.get("output_token", 0) for result in type_results
            ),
        }
    metric["by_reasoning_type"] = by_reasoning_type
    return metric


def _fmt_pct(value) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_num(value) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def generate_report_txt(metric: Dict[str, Any], output_path: str) -> None:
    lines = []
    model = metric.get("model", "unknown")
    memory_system = metric.get("memory_system", "unknown")
    lines.append("=" * 60)
    lines.append(f"Memory Evaluation Results - {model} ({memory_system})")
    lines.append("=" * 60)
    lines.append(
        f"Tasks: {metric.get('valid_tasks', 0)}/{metric.get('completed_tasks', 0)} "
        f"valid ({metric.get('skipped_tasks', 0)} skipped)"
    )
    lines.append("")
    lines.append("--- Overall Metrics ---")
    lines.append(f"Exact Match Rate:     {_fmt_pct(metric.get('exact_match_rate', 0))}")
    lines.append("")
    lines.append("  [Field-Level]")
    lines.append(f"  Acc Positive:       {_fmt_pct(metric.get('state_acc_positive', 0))}")
    lines.append(f"  Prec Positive:      {_fmt_pct(metric.get('state_precision_positive', 0))}")
    lines.append(f"  F1 Positive:        {_fmt_pct(metric.get('state_f1_positive', 0))}")
    lines.append("  ---")
    lines.append(f"  Acc Negative:       {_fmt_pct(metric.get('state_acc_negative', 0))}")
    lines.append(f"  F1 Negative:        {_fmt_pct(metric.get('state_f1_negative', 0))}")
    lines.append("")
    lines.append("  [Value-Level]")
    lines.append(f"  Change Accuracy:    {_fmt_pct(metric.get('change_accuracy', 0))}")
    lines.append(f"  Prec Change:        {_fmt_pct(metric.get('state_precision_change', 0))}")
    lines.append(f"  F1 Change:          {_fmt_pct(metric.get('state_f1_change', 0))}")
    lines.append("")
    lines.append("  [Efficiency]")
    lines.append(f"  Avg Pred Calls:     {_fmt_num(metric.get('avg_pred_calls', 0))}")
    lines.append(f"  Avg Output Token:   {_fmt_num(metric.get('avg_output_token', 0))}")

    by_type = metric.get("by_reasoning_type", {})
    if by_type:
        lines.append("")
        lines.append("--- Metrics by Reasoning Type ---")
        for reasoning_type in sorted(by_type.keys()):
            item = by_type[reasoning_type]
            lines.append("")
            lines.append(f"[{reasoning_type}] (n={item.get('count', 0)})")
            lines.append(f"  Exact Match:        {_fmt_pct(item.get('exact_match_rate', 0))}")
            lines.append(
                "  [Field] Acc/Prec/F1:  "
                f"{_fmt_pct(item.get('state_acc_positive', 0))} / "
                f"{_fmt_pct(item.get('state_precision_positive', 0))} / "
                f"{_fmt_pct(item.get('state_f1_positive', 0))}"
            )
            lines.append(
                "  [Value] Acc/Prec/F1:  "
                f"{_fmt_pct(item.get('change_accuracy', 0))} / "
                f"{_fmt_pct(item.get('state_precision_change', 0))} / "
                f"{_fmt_pct(item.get('state_f1_change', 0))}"
            )
            lines.append(
                "  [Negative] Acc/F1:    "
                f"{_fmt_pct(item.get('state_acc_negative', 0))} / "
                f"{_fmt_pct(item.get('state_f1_negative', 0))}"
            )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _print_metric_summary(metric: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(f"Memory Evaluation Results - {metric['model']} ({metric['memory_system']})")
    print("=" * 60)
    print(
        f"Tasks: {metric['valid_tasks']}/{metric['completed_tasks']} "
        f"valid ({metric['skipped_tasks']} skipped)"
    )

    print("\n--- Overall Metrics ---")
    print(f"Exact Match Rate:     {metric['exact_match_rate']:.2%}")

    print("\n  [Field-Level]")
    print(f"  Acc Positive:       {metric['state_acc_positive']:.2%}")
    print(f"  Prec Positive:      {metric['state_precision_positive']:.2%}")
    print(f"  F1 Positive:        {metric['state_f1_positive']:.2%}")
    print("  ---")
    print(f"  Acc Negative:       {metric['state_acc_negative']:.2%}")
    print(f"  F1 Negative:        {metric['state_f1_negative']:.2%}")

    print("\n  [Value-Level]")
    print(f"  Change Accuracy:    {metric['change_accuracy']:.2%}")
    print(f"  Prec Change:        {metric['state_precision_change']:.2%}")
    print(f"  F1 Change:          {metric['state_f1_change']:.2%}")

    print("\n  [Efficiency]")
    print(f"  Avg Pred Calls:     {metric['avg_pred_calls']:.2f}")
    print(f"  Avg Output Token:   {metric['avg_output_token']:.2f}")

    print("\n--- Metrics by Reasoning Type ---")
    for reasoning_type, item in sorted(metric["by_reasoning_type"].items()):
        print(f"\n[{reasoning_type}] (n={item['count']})")
        print(f"  Exact Match:        {item['exact_match_rate']:.2%}")
        print(
            "  [Field] Acc/Prec/F1:  "
            f"{item['state_acc_positive']:.2%} / "
            f"{item['state_precision_positive']:.2%} / "
            f"{item['state_f1_positive']:.2%}"
        )
        print(
            "  [Value] Acc/Prec/F1:  "
            f"{item['change_accuracy']:.2%} / "
            f"{item['state_precision_change']:.2%} / "
            f"{item['state_f1_change']:.2%}"
        )
        print(
            "  [Negative] Acc/F1:    "
            f"{item['state_acc_negative']:.2%} / "
            f"{item['state_f1_negative']:.2%}"
        )
    print("=" * 60 + "\n")


def memorysystem_add(
    *,
    memory_system: str,
    history_dir: str,
    file_range: Optional[str] = None,
    max_workers: int = 1,
    memory_url: Optional[str] = None,
    memory_key: Optional[str] = None,
    enable_graph: bool = False,
    model: str = "gpt-4o-mini",
    device: str = "cpu",
    embedding_api_base: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
    store_root: Optional[str] = None,
) -> None:
    memory_module = get_system_module(memory_system)
    args = argparse.Namespace(
        memory_system=memory_system,
        history_dir=history_dir,
        file_range=file_range,
        max_workers=max_workers,
        memory_url=memory_url,
        memory_key=memory_key,
        enable_graph=enable_graph,
        model=model,
        device=device,
        embedding_api_base=embedding_api_base,
        embedding_api_key=embedding_api_key,
        embedding_model=embedding_model,
        store_root=store_root,
    )
    memory_module.validate_add_args(args)
    memory_module.run_add(args)


def memorysystem_evaluation(
    *,
    benchmark_dir: str,
    api_base: str,
    api_key: str,
    model: str,
    memory_system: str,
    reflect_num: int = 10,
    prefix: str = "memorysystem_eval",
    file_range: Optional[str] = None,
    output_dir: Optional[str] = None,
    sample_size: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
    enable_graph: bool = False,
    user_id_prefix: Optional[str] = None,
    memory_url: Optional[str] = None,
    memory_key: Optional[str] = None,
    max_workers: int = 6,
    lightmem_model: str = "gpt-4o-mini",
    lightmem_device: str = "cpu",
    embedding_api_base: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
    history_dir: Optional[str] = None,
    store_root: Optional[str] = None,
) -> None:
    runtime = _get_runtime_helpers()
    AgentClient = runtime["AgentClient"]
    parse_answer_to_tools = runtime["parse_answer_to_tools"]

    memory_module = get_system_module(memory_system)
    args = argparse.Namespace(
        benchmark_dir=benchmark_dir,
        api_base=api_base,
        api_key=api_key,
        model=model,
        memory_system=memory_system,
        reflect_num=reflect_num,
        prefix=prefix,
        file_range=file_range,
        output_dir=output_dir,
        sample_size=sample_size,
        enable_thinking=enable_thinking,
        enable_graph=enable_graph,
        user_id_prefix=user_id_prefix,
        memory_url=memory_url,
        memory_key=memory_key,
        max_workers=max_workers,
        lightmem_model=lightmem_model,
        lightmem_device=lightmem_device,
        embedding_api_base=embedding_api_base,
        embedding_api_key=embedding_api_key,
        embedding_model=embedding_model,
        history_dir=history_dir or os.path.join(ROOT_DIR, "benchmark", "history"),
        store_root=store_root,
    )
    memory_module.validate_test_args(args)

    benchmark_dir = os.path.abspath(benchmark_dir)
    if not os.path.isdir(benchmark_dir):
        raise FileNotFoundError(f"benchmark directory not found: {benchmark_dir}")

    file_numbers = parse_file_range(file_range) or _collect_event_chain_numbers(benchmark_dir)
    if not file_numbers:
        raise FileNotFoundError(f"No qa_*.json files found in {benchmark_dir}")

    user_id_prefix = user_id_prefix or getattr(memory_module, "USER_ID_PREFIX", memory_system)
    output_dir = output_dir or os.path.join(ROOT_DIR, "memory_system_log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = os.path.join(
        output_dir,
        f"{prefix}_{memory_system}_{model.replace('/', '_')}_{timestamp}",
    )
    os.makedirs(output_subdir, exist_ok=True)

    shared_state = memory_module.init_test_state(args, file_numbers, user_id_prefix)
    all_results: List[Dict[str, Any]] = []

    def init_clients_with_retry(file_num: int):
        max_retries = 5
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                agent_client = AgentClient(api_base=api_base, api_key=api_key, model=model)
                agent_client.enable_thinking = enable_thinking
                mem_client = memory_module.build_test_client(
                    args=args,
                    file_num=file_num,
                    user_id_prefix=user_id_prefix,
                    shared_state=shared_state,
                )
                return agent_client, mem_client
            except Exception as exc:
                last_error = exc
                if attempt == max_retries:
                    break
                logger.warning(
                    f"[File {file_num}] client init failed "
                    f"(attempt {attempt}/{max_retries}): {exc}. Retrying..."
                )
                time.sleep(2 ** (attempt - 1))
        raise RuntimeError(
            f"[File {file_num}] client init failed after {max_retries} retries: {last_error}"
        )

    def process_qa_file(file_num: int) -> Tuple[int, List[Dict[str, Any]], str]:
        qa_file_path = os.path.join(benchmark_dir, f"qa_{file_num}.json")
        if not os.path.exists(qa_file_path):
            return file_num, [], "missing"

        with open(qa_file_path, "r", encoding="utf-8") as handle:
            qa_payload = json.load(handle)

        related_events = qa_payload.get("related_to_vehicle_preference", [])
        queries = []
        for event_index, event_item in enumerate(related_events):
            query = event_item.get("query", "")
            new_answer = event_item.get("new_answer", [])
            reasoning_type = event_item.get("reasoning_type", "unknown")
            if not query or not new_answer:
                continue

            queries.append(
                {
                    "query": query,
                    "tools": parse_answer_to_tools(new_answer),
                    "reasoning_type": reasoning_type,
                    "event_index": event_index,
                    "source_file": f"qa_{file_num}.json",
                }
            )

        if sample_size is not None and sample_size < len(queries):
            queries = queries[:sample_size]

        try:
            agent_client, mem_client = init_clients_with_retry(file_num)
        except Exception as exc:
            logger.error(str(exc))
            return file_num, [], "failed"

        file_results: List[Dict[str, Any]] = []
        user_id = f"{user_id_prefix}_{file_num}"
        for idx, query_data in enumerate(tqdm(queries, desc=f"Evaluating file {file_num}")):
            result = process_task_with_memorysystem(
                task=query_data,
                task_id=f"{file_num}_{idx}",
                agent_client=agent_client,
                reflect_num=reflect_num,
                mem_client=mem_client,
                user_id=user_id,
                memory_module=memory_module,
            )
            if result:
                result["source_file"] = query_data["source_file"]
                result["event_index"] = query_data["event_index"]
                file_results.append(result)

        return file_num, file_results, "completed"

    actual_workers = (
        1
        if memory_module.is_test_sequential()
        else max(1, min(max_workers, len(file_numbers)))
    )
    logger.info(
        f"Starting memory-system evaluation with {actual_workers} worker(s) "
        f"for {len(file_numbers)} files."
    )

    try:
        if actual_workers == 1:
            for file_num in file_numbers:
                current_file_num, file_results, status = process_qa_file(file_num)
                if status == "missing":
                    logger.info(f"[File {current_file_num}] skipped (file not found)")
                    continue
                if status == "failed":
                    logger.error(f"[File {current_file_num}] failed during initialization or evaluation")
                    continue
                save_json_file(
                    file_results,
                    os.path.join(output_subdir, f"results_{current_file_num}.json"),
                )
                all_results.extend(file_results)
                logger.info(
                    f"[File {current_file_num}] completed with {len(file_results)} results"
                )
        else:
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                future_to_num = {
                    executor.submit(process_qa_file, file_num): file_num
                    for file_num in file_numbers
                }
                for future in as_completed(future_to_num):
                    submitted_num = future_to_num[future]
                    try:
                        current_file_num, file_results, status = future.result()
                        if status == "missing":
                            logger.info(f"[File {current_file_num}] skipped (file not found)")
                            continue
                        if status == "failed":
                            logger.error(f"[File {current_file_num}] failed during initialization or evaluation")
                            continue
                        save_json_file(
                            file_results,
                            os.path.join(output_subdir, f"results_{current_file_num}.json"),
                        )
                        all_results.extend(file_results)
                        logger.info(
                            f"[File {current_file_num}] completed with {len(file_results)} results"
                        )
                    except Exception as exc:
                        logger.error(f"[File {submitted_num}] failed: {exc}")
    finally:
        memory_module.close_test_state(shared_state)

    if not all_results:
        logger.warning("No results to report")
        return

    metric = _build_metric(
        all_results=all_results,
        model=model,
        memory_system=memory_system,
    )
    generate_report_txt(metric, os.path.join(output_subdir, "report.txt"))
    save_json_file(metric, os.path.join(output_subdir, "metric.json"))
    save_json_file(all_results, os.path.join(output_subdir, "all_results.json"))

    config = {
        "timestamp": timestamp,
        "model": model,
        "memory_system": memory_system,
        "api_base": api_base,
        "benchmark_dir": benchmark_dir,
        "file_range": file_range,
        "reflect_num": reflect_num,
        "prefix": prefix,
        "sample_size": sample_size,
        "enable_thinking": enable_thinking,
        "enable_graph": enable_graph,
        "user_id_prefix": user_id_prefix,
        "max_workers": max_workers,
        "store_root": getattr(args, "store_root", None) or store_root,
    }
    save_json_file(config, os.path.join(output_subdir, "config.json"))

    _print_metric_summary(metric)
    logger.info(f"Results saved to {output_subdir}")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Memory-system ingestion and evaluation for VehicleMemBench"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Write benchmark history into a memory system")
    add_parser.add_argument(
        "--memory_system",
        type=str,
        required=True,
        choices=SUPPORTED_MEMORY_SYSTEMS,
    )
    add_parser.add_argument(
        "--history_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "benchmark", "history"),
    )
    add_parser.add_argument("--file_range", type=str, default=None)
    add_parser.add_argument("--max_workers", type=int, default=1)
    add_parser.add_argument("--enable_graph", action="store_true")
    add_parser.add_argument("--memory_url", type=str, default=None)
    add_parser.add_argument("--memory_key", type=str, default=None)
    add_parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LightMem LLM model")
    add_parser.add_argument("--device", type=str, default="cpu", help="LightMem device")
    add_parser.add_argument("--embedding_api_base", type=str, default=None, help="MemoryBank embedding API base URL")
    add_parser.add_argument("--embedding_api_key", type=str, default=None, help="MemoryBank embedding API key")
    add_parser.add_argument("--embedding_model", type=str, default=None, help="MemoryBank embedding model name")
    add_parser.add_argument("--store_root", type=str, default=None, help="MemoryBank FAISS store root directory")

    test_parser = subparsers.add_parser(
        "test",
        help="Evaluate a model that can query the selected memory system",
    )
    test_parser.add_argument(
        "--benchmark_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "benchmark", "qa_data"),
    )
    test_parser.add_argument("--api_base", type=str, default="")
    test_parser.add_argument("--api_key", type=str, default="")
    test_parser.add_argument("--model", type=str, default="gpt-4")
    test_parser.add_argument(
        "--memory_system",
        type=str,
        required=True,
        choices=SUPPORTED_MEMORY_SYSTEMS,
    )
    test_parser.add_argument("--reflect_num", type=int, default=10)
    test_parser.add_argument("--prefix", type=str, default="memorysystem_eval")
    test_parser.add_argument("--file_range", type=str, default=None)
    test_parser.add_argument("--output_dir", type=str, default=None)
    test_parser.add_argument("--sample_size", type=int, default=None)
    test_parser.add_argument("--max_workers", type=int, default=6)
    test_parser.add_argument("--enable_graph", action="store_true")
    test_parser.add_argument("--user_id_prefix", type=str, default=None)
    test_parser.add_argument("--memory_url", type=str, default=None)
    test_parser.add_argument("--memory_key", type=str, default=None)
    test_parser.add_argument("--lightmem_model", type=str, default="gpt-4o-mini")
    test_parser.add_argument("--lightmem_device", type=str, default="cpu")
    test_parser.add_argument("--embedding_api_base", type=str, default=None, help="MemoryBank embedding API base URL")
    test_parser.add_argument("--embedding_api_key", type=str, default=None, help="MemoryBank embedding API key")
    test_parser.add_argument("--embedding_model", type=str, default=None, help="MemoryBank embedding model name")
    test_parser.add_argument("--history_dir", type=str, default=None, help="History data directory (for MemoryBank reference date)")
    test_parser.add_argument("--store_root", type=str, default=None, help="MemoryBank FAISS store root directory")
    test_parser.add_argument(
        "--enable_thinking",
        type=str2bool,
        default=None,
        help="Optional thinking mode flag (true/false). If omitted, the field is not sent.",
    )

    return parser


if __name__ == "__main__":
    parser = _build_cli_parser()
    cli_args = parser.parse_args()

    if cli_args.command == "add":
        memorysystem_add(
            memory_system=cli_args.memory_system,
            history_dir=cli_args.history_dir,
            file_range=cli_args.file_range,
            max_workers=cli_args.max_workers,
            memory_url=cli_args.memory_url,
            memory_key=cli_args.memory_key,
            enable_graph=cli_args.enable_graph,
            model=cli_args.model,
            device=cli_args.device,
            embedding_api_base=cli_args.embedding_api_base,
            embedding_api_key=cli_args.embedding_api_key,
            embedding_model=cli_args.embedding_model,
            store_root=cli_args.store_root,
        )
    else:
        memorysystem_evaluation(
            benchmark_dir=cli_args.benchmark_dir,
            api_base=cli_args.api_base,
            api_key=cli_args.api_key,
            model=cli_args.model,
            memory_system=cli_args.memory_system,
            reflect_num=cli_args.reflect_num,
            prefix=cli_args.prefix,
            file_range=cli_args.file_range,
            output_dir=cli_args.output_dir,
            sample_size=cli_args.sample_size,
            enable_thinking=cli_args.enable_thinking,
            enable_graph=cli_args.enable_graph,
            user_id_prefix=cli_args.user_id_prefix,
            memory_url=cli_args.memory_url,
            memory_key=cli_args.memory_key,
            max_workers=cli_args.max_workers,
            lightmem_model=cli_args.lightmem_model,
            lightmem_device=cli_args.lightmem_device,
            embedding_api_base=cli_args.embedding_api_base,
            embedding_api_key=cli_args.embedding_api_key,
            embedding_model=cli_args.embedding_model,
            history_dir=cli_args.history_dir,
            store_root=cli_args.store_root,
        )
