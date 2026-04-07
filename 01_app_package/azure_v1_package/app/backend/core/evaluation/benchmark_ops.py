from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.config.operational import LLM_CONFIG
from core.evaluation.framework import SCORING_RUBRIC
from core.schemas.router import module_families_for_value


BASE_DIR = Path(os.getenv("IWERP_BASE_DIR", "/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod"))
ROOT_DIR = BASE_DIR.parent
BENCHMARK_ROOT = BASE_DIR / "production_benchmark"
TRACE_ROOT = BASE_DIR / "ops_traces"
DATASET_ROOT = BENCHMARK_ROOT / "datasets"
CANONICAL_1000_PATH = ROOT_DIR / "test_cases" / "oracle_fusion_llm_test_cases_1000.json"
EXPANDED_5000_PATH = DATASET_ROOT / "oracle_fusion_llm_test_cases_5000_expanded.json"
SQL_PILOT_CASES_PATH = BASE_DIR / "specialization_tracks" / "pilot_benchmarks" / "sql_pilot_cases.jsonl"
FAST_FORMULA_PILOT_CASES_PATH = BASE_DIR / "specialization_tracks" / "pilot_benchmarks" / "fast_formula_pilot_cases.jsonl"
RUN_META_FILENAME = "run_meta.json"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _slugify(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-") or "run"


def _variant_questions(question: str, module: str, count: int = 5) -> List[str]:
    base = question.strip().rstrip("?")
    variants = [
        f"{base}?",
        f"Oracle Fusion {module}: {base}?",
        f"{base}? Provide the grounded Oracle Fusion steps only.",
        f"{base}? Include the exact Oracle Fusion procedure or resolution path.",
        f"Please answer this Oracle Fusion {module} request: {base}?",
    ]
    return variants[:count]


def _specialized_variant_questions(question: str, module: str, lane: str, expected_behavior: str, count: int = 5) -> List[str]:
    base = question.strip().rstrip("?")
    label = "SQL" if lane == "sql" else "Fast Formula"
    strict_suffix = (
        " Return a safe refusal if grounded support is missing."
        if expected_behavior == "refusal"
        else " Adapt from grounded patterns only."
    )
    variants = [
        f"{base}?",
        f"Oracle Fusion {module} {label} request: {base}?{strict_suffix}",
        f"{base}? Keep module alignment strict and emit only grounded {label}.",
        f"{base}? Use the specialized {label} lane and preserve safe refusal behavior.",
        f"Please handle this Oracle Fusion {module} {label} case: {base}?{strict_suffix}",
    ]
    return variants[:count]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_case_payload(path: Path) -> Dict[str, Any]:
    payload = _read_json(path, default={})
    if not isinstance(payload, dict) or "test_cases" not in payload:
        raise ValueError(f"Unsupported benchmark dataset format: {path}")
    return payload


def build_expanded_5000_dataset(force: bool = False) -> Path:
    if EXPANDED_5000_PATH.exists() and not force:
        return EXPANDED_5000_PATH

    payload = _load_case_payload(CANONICAL_1000_PATH)
    expanded_cases: List[Dict[str, Any]] = []
    for case_index, case in enumerate(payload["test_cases"], start=1):
        case_id = str(case.get("id") or "")
        module = str(case.get("module") or "UNKNOWN")
        variant_count = 5 if case_index <= 500 else 4
        for index, question in enumerate(_variant_questions(str(case.get("question") or ""), module, count=variant_count), start=1):
            variant = dict(case)
            variant["id"] = f"{case_id}-V{index}"
            variant["question"] = question
            variant["variant_of"] = case_id
            variant["variant_index"] = index
            expanded_cases.append(variant)

    specialized_rows = _load_jsonl(SQL_PILOT_CASES_PATH) + _load_jsonl(FAST_FORMULA_PILOT_CASES_PATH)
    for row in specialized_rows:
        lane = str(row.get("lane") or "specialized")
        case_id = str(row.get("id") or "")
        module = str(row.get("module") or "UNKNOWN")
        expected_behavior = str(row.get("expected_behavior") or "answer")
        if lane == "sql":
            answer = "Grounded SQL should align to verified schema patterns."
            if row.get("expected_tables"):
                answer = f"Grounded SQL should use tables: {', '.join(row.get('expected_tables') or [])}."
        else:
            answer = "Grounded Fast Formula should align to verified formula examples."
            if row.get("expected_formula_type"):
                answer = f"Grounded Fast Formula should align to formula type: {row.get('expected_formula_type')}."

        for index, question in enumerate(
            _specialized_variant_questions(str(row.get("question") or ""), module, lane, expected_behavior, count=5),
            start=1,
        ):
            expanded_cases.append(
                {
                    "id": f"{case_id}-V{index}",
                    "category": "Specialized",
                    "module": module,
                    "difficulty": "specialized",
                    "question": question,
                    "answer": answer,
                    "tags": [lane, expected_behavior],
                    "variant_of": case_id,
                    "variant_index": index,
                    "lane": lane,
                    "task_type": row.get("task_type"),
                    "expected_behavior": expected_behavior,
                    "benchmark_segment": "sql_generation" if lane == "sql" else "fast_formula_generation",
                    "expected_tables": row.get("expected_tables") or [],
                    "expected_columns": row.get("expected_columns") or [],
                    "expected_formula_type": row.get("expected_formula_type"),
                    "expected_database_items": row.get("expected_database_items") or [],
                    "source_title": row.get("source_title"),
                    "source_uri": row.get("source_uri"),
                }
            )

    output = {
        "dataset_name": "oracle_fusion_llm_test_cases_5000_expanded",
        "description": "Deterministic certification mix built from the canonical 1000-case benchmark plus SQL and Fast Formula pilot cases.",
        "total_cases": len(expanded_cases),
        "schema": payload.get("schema", {}),
        "source_dataset": str(CANONICAL_1000_PATH),
        "specialized_sources": [str(SQL_PILOT_CASES_PATH), str(FAST_FORMULA_PILOT_CASES_PATH)],
        "test_cases": expanded_cases,
    }
    _write_json(EXPANDED_5000_PATH, output)
    return EXPANDED_5000_PATH


def _load_cases_for_filtering(path: Path) -> List[Dict[str, Any]]:
    return list(_load_case_payload(path).get("test_cases", []))


def _infer_task_keyword(question: str) -> str:
    query = question.lower()
    if any(token in query for token in ["sql", "select ", " join ", "query "]):
        return "sql_generation"
    if "fast formula" in query or "formula" in query:
        return "fast_formula_generation"
    if any(token in query for token in ["troubleshoot", "error", "failure", "failed", "fix "]):
        return "troubleshooting"
    if query.startswith("how do you") or query.startswith("what steps"):
        return "procedure"
    if query.startswith("what is") or query.startswith("what are") or query.startswith("summarize"):
        return "summary"
    return "general"


def build_filtered_dataset(
    *,
    source_path: Path,
    output_path: Path,
    limit: Optional[int] = None,
    module_filters: Optional[Iterable[str]] = None,
    task_filters: Optional[Iterable[str]] = None,
    case_ids: Optional[Iterable[str]] = None,
) -> Path:
    module_filters = {item.strip() for item in (module_filters or []) if item and item.strip()}
    task_filters = {item.strip() for item in (task_filters or []) if item and item.strip()}
    case_ids = {str(item).strip() for item in (case_ids or []) if str(item).strip()}

    selected: List[Dict[str, Any]] = []
    for case in _load_cases_for_filtering(source_path):
        module = str(case.get("module") or "UNKNOWN")
        task = _infer_task_keyword(str(case.get("question") or ""))
        if case_ids and str(case.get("id") or "") not in case_ids:
            continue
        if module_filters:
            module_family = next(iter(module_families_for_value(module)), "UNKNOWN")
            if module not in module_filters and module_family not in module_filters:
                continue
        if task_filters and task not in task_filters:
            continue
        selected.append(case)
        if limit and len(selected) >= limit:
            break

    payload = {
        "dataset_name": f"{source_path.stem}_filtered",
        "description": f"Filtered dataset derived from {source_path.name}",
        "total_cases": len(selected),
        "schema": _load_case_payload(source_path).get("schema", {}),
        "source_dataset": str(source_path),
        "test_cases": selected,
    }
    _write_json(output_path, payload)
    return output_path


def resolve_dataset_path(
    dataset: str,
    *,
    label: str,
    module_filters: Optional[Iterable[str]] = None,
    task_filters: Optional[Iterable[str]] = None,
    case_ids: Optional[Iterable[str]] = None,
    custom_input_path: Optional[str] = None,
) -> Path:
    _ensure_dir(DATASET_ROOT)
    if dataset == "custom":
        if not custom_input_path:
            raise ValueError("custom_input_path is required when dataset='custom'")
        source = Path(custom_input_path)
    elif dataset == "5000":
        source = build_expanded_5000_dataset()
    else:
        source = CANONICAL_1000_PATH

    if dataset == "200":
        return build_filtered_dataset(
            source_path=source,
            output_path=DATASET_ROOT / f"{_slugify(label)}_200.json",
            limit=200,
            module_filters=module_filters,
            task_filters=task_filters,
            case_ids=case_ids,
        )

    if module_filters or task_filters or case_ids:
        return build_filtered_dataset(
            source_path=source,
            output_path=DATASET_ROOT / f"{_slugify(label)}_filtered.json",
            module_filters=module_filters,
            task_filters=task_filters,
            case_ids=case_ids,
        )
    return source


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _python_binary() -> str:
    return sys.executable


def launch_benchmark_run(
    *,
    label: str,
    dataset_path: Path,
    max_tokens: int,
    flush_every: int,
    case_timeout_sec: int,
    case_retries: int,
    recycle_worker_every: int,
) -> Dict[str, Any]:
    _ensure_dir(BENCHMARK_ROOT)
    run_dir = BENCHMARK_ROOT / label
    _ensure_dir(run_dir)
    command = [
        _python_binary(),
        str(BASE_DIR / "backend" / "scripts" / "run_production_benchmark.py"),
        "--label",
        label,
        "--input-path",
        str(dataset_path),
        "--max-tokens",
        str(max_tokens),
        "--flush-every",
        str(flush_every),
        "--case-timeout-sec",
        str(case_timeout_sec),
        "--case-retries",
        str(case_retries),
        "--recycle-worker-every",
        str(recycle_worker_every),
    ]
    stdout_path = run_dir / "launcher.stdout.log"
    stderr_path = run_dir / "launcher.stderr.log"
    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open("a", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(BASE_DIR),
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )

    metadata = {
        "run_id": label,
        "label": label,
        "pid": process.pid,
        "status": "running",
        "dataset_path": str(dataset_path),
        "command": command,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }
    _write_json(run_dir / RUN_META_FILENAME, metadata)
    return metadata


def _run_meta(run_dir: Path) -> Dict[str, Any]:
    metadata = _read_json(run_dir / RUN_META_FILENAME, default={}) or {}
    summary = _read_json(run_dir / "benchmark_summary.json", default={}) or {}
    plan_summary = _read_json(run_dir / "benchmark_plan_summary.json", default={}) or {}
    pid = metadata.get("pid")
    status = metadata.get("status") or "unknown"
    if pid and _pid_alive(pid):
        status = "running"
    elif summary.get("sample_size") and summary.get("sample_size") == plan_summary.get("total_cases"):
        status = "completed"
    elif summary.get("sample_size"):
        status = "partial"
    metadata["status"] = status
    metadata["updated_at"] = int(time.time())
    return metadata


def list_benchmark_runs(limit: int = 20) -> List[Dict[str, Any]]:
    if not BENCHMARK_ROOT.exists():
        return []
    run_dirs = [path for path in BENCHMARK_ROOT.iterdir() if path.is_dir()]
    items: List[Dict[str, Any]] = []
    for run_dir in sorted(run_dirs, key=lambda path: path.stat().st_mtime, reverse=True):
        metadata = _run_meta(run_dir)
        summary = _read_json(run_dir / "benchmark_summary.json", default={}) or {}
        items.append(
            {
                "run_id": run_dir.name,
                "label": metadata.get("label") or run_dir.name,
                "status": metadata.get("status", "unknown"),
                "sample_size": int(summary.get("sample_size") or 0),
                "created_at": metadata.get("created_at"),
                "updated_at": metadata.get("updated_at"),
                "summary_path": str(run_dir / "benchmark_summary.json"),
                "primary_metrics": (summary.get("evaluation_summary") or {}).get("primary_metrics", {}),
            }
        )
        if len(items) >= limit:
            break
    return items


def benchmark_summary(run_id: str) -> Dict[str, Any]:
    run_dir = BENCHMARK_ROOT / run_id
    if not run_dir.exists():
        raise FileNotFoundError(run_id)
    summary = _read_json(run_dir / "benchmark_summary.json", default={}) or {}
    metadata = _run_meta(run_dir)
    return {
        "run_id": run_id,
        "label": metadata.get("label") or run_id,
        "status": metadata.get("status", "unknown"),
        "sample_size": int(summary.get("sample_size") or 0),
        "summary": summary,
        "history": list_benchmark_runs(limit=10),
    }


def benchmark_cases(
    run_id: str,
    *,
    module: Optional[str] = None,
    task_type: Optional[str] = None,
    failure_category: Optional[str] = None,
    primary_verdict: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    run_dir = BENCHMARK_ROOT / run_id
    results_path = run_dir / "benchmark_results.jsonl"
    if not results_path.exists():
        return {"run_id": run_id, "total_cases": 0, "returned_cases": 0, "cases": []}

    cases: List[Dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            evaluation = row.get("evaluation") or {}
            if module and str(row.get("benchmark_module") or row.get("module") or "") != module:
                continue
            if task_type and str(evaluation.get("task_type") or row.get("benchmark_intent") or "") != task_type:
                continue
            if failure_category and str(evaluation.get("failure_category") or row.get("failure_category") or "") != failure_category:
                continue
            if primary_verdict and str(evaluation.get("primary_verdict") or "") != primary_verdict:
                continue
            cases.append(row)
            if len(cases) >= limit:
                break
    return {"run_id": run_id, "total_cases": len(cases), "returned_cases": len(cases), "cases": cases}


def latest_benchmark_summary() -> Dict[str, Any]:
    runs = list_benchmark_runs(limit=1)
    if not runs:
        return {}
    return benchmark_summary(runs[0]["run_id"])


def persist_trace(
    *,
    trace_id: str,
    tenant_id: Optional[str],
    request_payload: Dict[str, Any],
    response_payload: Dict[str, Any],
) -> Path:
    _ensure_dir(TRACE_ROOT)
    path = TRACE_ROOT / f"{trace_id}.json"
    payload = {
        "trace_id": trace_id,
        "tenant_id": tenant_id,
        "request": request_payload,
        "response": response_payload,
        "decision_trace": response_payload.get("decision_trace") or {},
        "grounding_trace": response_payload.get("grounding_trace") or {},
        "audit": response_payload.get("audit") or {},
        "created_at": int(time.time()),
    }
    _write_json(path, payload)
    return path


def load_trace(trace_id: str) -> Dict[str, Any]:
    path = TRACE_ROOT / f"{trace_id}.json"
    if not path.exists():
        raise FileNotFoundError(trace_id)
    return _read_json(path, default={}) or {}


def health_readiness_payload() -> Dict[str, Any]:
    latest = latest_benchmark_summary()
    latest_summary = latest.get("summary") or {}
    return {
        "status": "healthy",
        "version": "1.1.0",
        "model_backend": str(LLM_CONFIG.get("inference_backend") or "unknown"),
        "corpus_status": {
            "benchmark_root": str(BENCHMARK_ROOT),
            "trace_root": str(TRACE_ROOT),
            "dataset_1000": str(CANONICAL_1000_PATH),
            "dataset_5000": str(EXPANDED_5000_PATH if EXPANDED_5000_PATH.exists() else build_expanded_5000_dataset()),
        },
        "latest_benchmark": {
            "run_id": latest.get("run_id"),
            "status": latest.get("status"),
            "sample_size": latest.get("sample_size"),
            "primary_metrics": (latest_summary.get("evaluation_summary") or {}).get("primary_metrics", {}),
        },
        "scoring_rubric": SCORING_RUBRIC,
        "safety_flags": {
            "hallucination_must_remain_zero": True,
            "wrong_module_must_remain_near_zero": True,
            "citations_required_for_grounded_answers": True,
            "trace_hidden_by_default": True,
        },
    }
