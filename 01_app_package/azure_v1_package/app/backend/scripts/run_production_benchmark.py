import argparse
import asyncio
import json
import multiprocessing as mp
import queue
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "backend"))

from backend.core.grounding.task_semantics import TaskSemanticAnalyzer
from backend.core.retrieval.router import TaskRouter
from backend.core.evaluation.framework import SCORING_RUBRIC, aggregate_scorecards, build_case_scorecard
from backend.core.schemas.router import TaskType, module_families_for_value

VALIDATION_SOURCE = ROOT_DIR / "test_cases" / "oracle_fusion_llm_test_cases_1000.json"
OUTPUT_DIR = BASE_DIR / "production_benchmark"
PRIMARY_SEGMENTS = [
    "doc_grounded_procedure",
    "troubleshooting",
    "module_ambiguous",
    "safe_refusal_expected",
    "sql_generation",
    "fast_formula_generation",
    "cross-module negative cases",
]
SQL_EXPECTED_TASKS = {"sql_generation", "report_logic"}
DOCS_EXPECTED_TASKS = {"procedure", "navigation", "troubleshooting", "general", "integration", "summary"}
AMBIGUOUS_MODULES = {"Financials", "Procurement", "HCM", "Supply Chain"}
LEAF_FINANCIALS = {"Payables", "Receivables", "General Ledger", "Cash Management", "Assets", "Tax", "Expenses"}
FAIL_CLOSED_MESSAGE = "Insufficient grounded data. Cannot generate verified answer."
FAIL_OPEN_STATUSES = {"FAILED_TASK_SEMANTIC_NO_STRONG_MATCH", "FAILED_TASK_MODULE_CORRECTION", "FAILED_FINANCE_LEAF_NO_EXACT_DOCS"}
DEFAULT_CASE_TIMEOUT_SEC = 120
DEFAULT_CASE_RETRIES = 1
DEFAULT_WORKER_STARTUP_TIMEOUT_SEC = 240
DEFAULT_WORKER_RECYCLE_EVERY = 50
GENERIC_PATTERNS = [
    "relevant work area",
    "complete the required attributes",
    "submit or save it according to your approval",
    "based on the provided documentation",
]
INFERENCE_PATTERNS = [
    "not explicitly mentioned",
    "however, based on",
    "it can be inferred",
    "it can be assumed",
]


def numeric_case_id(case_id: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", case_id or "")
    if not match:
        return (0, case_id or "")
    return (int(match.group(1)), case_id or "")


def family_for(value: Any) -> str:
    families = module_families_for_value(str(value)) if value else {"UNKNOWN"}
    return next(iter(families)) if families else "UNKNOWN"


def extract_sql(output: str) -> str | None:
    if not output or output == FAIL_CLOSED_MESSAGE:
        return None

    section = re.search(r"\[SQL\]\s*(.*?)(?:\n\[[A-Za-z]+\]|\Z)", output, flags=re.IGNORECASE | re.DOTALL)
    if section:
        sql_block = section.group(1).strip()
        return sql_block or None

    fenced = re.search(r"```sql\s*(.*?)```", output, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return None


def is_safe_rejection(output: str) -> bool:
    normalized = (output or "").strip()
    return FAIL_CLOSED_MESSAGE in normalized


def correction_present(output: str, correction: Optional[str]) -> bool:
    if not output or not correction:
        return False
    return correction.lower().strip() in output.lower()


def benchmark_intent(question: str, routed_task: str) -> str:
    query = (question or "").lower().strip()
    if any(token in query for token in [" exact sql", " sql ", "query ", " select ", " join ", " where ", " report query"]):
        return "sql_generation"
    if any(token in query for token in ["troubleshoot", " error", " failure", " failed", " fix ", " debug "]):
        return "troubleshooting"
    if any(token in query for token in ["navigate", "menu", "where is", "path to", "navigator", "screen"]):
        return "navigation"
    if query.startswith("what is") or query.startswith("what are") or query.startswith("summarize"):
        return "summary"
    if query.startswith("how do you") or query.startswith("what steps are required") or query.startswith("how is "):
        return "procedure"
    if routed_task in {"sql_generation", "troubleshooting", "navigation", "summary", "procedure", "integration", "report_logic", "table_lookup"}:
        return routed_task
    return "general"


def classify_case(case: Dict[str, Any], router: TaskRouter) -> Dict[str, Any]:
    question = str(case.get("question") or "")
    module = str(case.get("module") or "UNKNOWN")
    routed = router.route(question)
    query_profile = TaskSemanticAnalyzer.extract_query_signals(question)
    top_signal = query_profile.get("top_task")
    preferred_modules = list((query_profile.get("signals") or [{}])[0].get("preferred_modules", [])) if top_signal else []
    module_family = family_for(module)
    intent = benchmark_intent(question, routed.task_type.value)
    module_ambiguous_case = module in AMBIGUOUS_MODULES
    cross_module_negative_case = False
    safe_refusal_expected = False
    expected_behavior = "answer"
    primary_segment = "doc_grounded_procedure"

    if intent in SQL_EXPECTED_TASKS:
        primary_segment = "sql_generation"
        expected_behavior = "sql"
    elif intent == "troubleshooting":
        primary_segment = "troubleshooting"
        expected_behavior = "troubleshooting"
    elif module_ambiguous_case:
        primary_segment = "module_ambiguous"
        expected_behavior = "answer"

    if top_signal and preferred_modules:
        if not module_ambiguous_case and module not in preferred_modules:
            cross_module_negative_case = True
            expected_behavior = "correction_then_refusal"
        elif module_ambiguous_case:
            preferred_families = {family_for(item) for item in preferred_modules}
            if module_family not in preferred_families:
                safe_refusal_expected = True
                expected_behavior = "refusal"

    if cross_module_negative_case:
        primary_segment = "cross-module negative cases"
    elif safe_refusal_expected:
        primary_segment = "safe_refusal_expected"

    if not top_signal and module_ambiguous_case and primary_segment == "module_ambiguous":
        expected_behavior = "answer"

    return {
        "benchmark_intent": intent,
        "benchmark_task_type": routed.task_type.value,
        "benchmark_module_family": module_family,
        "benchmark_segment": primary_segment,
        "expected_behavior": expected_behavior,
        "cross_module_negative_case": cross_module_negative_case,
        "safe_refusal_expected": safe_refusal_expected or expected_behavior in {"refusal", "correction_then_refusal"},
        "module_ambiguous_case": module_ambiguous_case,
        "query_task_signals_expected": query_profile.get("signals") or [],
        "benchmark_task_signal": top_signal,
        "benchmark_task_confidence": float(query_profile.get("top_confidence") or 0.0),
        "benchmark_preferred_modules": preferred_modules,
    }


def load_cases(
    limit: Optional[int] = None,
    input_path: Optional[Path] = None,
    case_ids: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    source_path = input_path or VALIDATION_SOURCE
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    cases = sorted(payload["test_cases"], key=lambda row: numeric_case_id(str(row.get("id") or "")))
    if case_ids:
        cases = [row for row in cases if str(row.get("id") or "") in case_ids]
    return cases[:limit] if limit else cases


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def build_runtime_error_record(
    case: Dict[str, Any],
    started_at: float,
    runtime_error: str,
    verifier_status: str = "RUNTIME_ERROR",
) -> Dict[str, Any]:
    return {
        **case,
        "module_detected": "UNKNOWN",
        "module_family_detected": "UNKNOWN",
        "intent_detected": None,
        "retrieved_docs": [],
        "retrieved_doc_count": 0,
        "docs_passed_to_prompt_count": 0,
        "schema_objects_used": [],
        "unknown_schema_usage": False,
        "hallucination_score": 0.0,
        "sql_index_used": False,
        "sql_generated": None,
        "verifier_status": verifier_status,
        "verifier_passed": False,
        "citations_present": False,
        "citation_count": 0,
        "citations": [],
        "rejected": True,
        "query_task_signals_runtime": [],
        "task_semantic_gate": "FAILED",
        "task_semantic_strong_doc_count": 0,
        "task_semantic_medium_doc_count": 0,
        "task_match_rate": 0.0,
        "task_semantic_correction": None,
        "task_gate_reason": None,
        "correction_present": False,
        "intent_confidence": 0.0,
        "module_confidence": 0.0,
        "decision_execution_mode": "REFUSE",
        "decision_reason": "runtime_error",
        "decision_refusal_reason": runtime_error,
        "grounding_availability_score": 0.0,
        "decision_grounding_signal_present": False,
        "decision_sufficient_grounding_signal": False,
        "decision_trace_summary": {
            "intent_classification": None,
            "intent_confidence": 0.0,
            "module_confidence": 0.0,
            "grounding_availability_score": 0.0,
            "decision_execution_mode": "REFUSE",
            "decision_reason": "runtime_error",
            "decision_refusal_reason": runtime_error,
        },
        "grounding_trace_summary": {
            "docs_retrieved_count": 0,
            "docs_passed_to_prompt_count": 0,
            "citation_count": 0,
            "exact_module_doc_count": 0,
            "task_semantic_gate": "FAILED",
            "task_match_rate": 0.0,
        },
        "response_time_sec": round(time.perf_counter() - started_at, 2),
        "output": FAIL_CLOSED_MESSAGE,
        "runtime_error": runtime_error,
        "exact_module_top_hit": False,
        "same_family_bleed_through": False,
    }


def build_case_record_from_response(case: Dict[str, Any], response: Any, started_at: float) -> Dict[str, Any]:
    output = response.choices[0]["message"]["content"]
    retrieved_chunks = response.retrieved_chunks or []
    detected_intent = str(response.audit.get("task_type") or "").strip().lower()
    sql_like_intent = detected_intent in SQL_EXPECTED_TASKS
    schema_objects = sorted(
        {
            obj
            for chunk in retrieved_chunks
            for obj in ((chunk.get("metadata") or {}).get("trusted_schema_objects") or [])
        }
    )
    record: Dict[str, Any] = {
        **case,
        "module_detected": response.audit.get("module"),
        "module_family_detected": response.audit.get("module_family"),
        "intent_detected": response.audit.get("task_type"),
        "retrieved_docs": [
            {
                "citation_id": chunk.get("citation_id"),
                "title": chunk.get("title"),
                "module": (chunk.get("metadata") or {}).get("module"),
                "module_family": (chunk.get("metadata") or {}).get("module_family"),
                "corpus": (chunk.get("metadata") or {}).get("corpus"),
                "source_system": (chunk.get("metadata") or {}).get("source_system"),
                "authority_tier": (chunk.get("metadata") or {}).get("authority_tier"),
                "task_signals": (chunk.get("metadata") or {}).get("task_signals") or [],
                "task_confidence": float((chunk.get("metadata") or {}).get("task_confidence") or 0.0),
                "task_match_score": float((chunk.get("metadata") or {}).get("task_match_score") or 0.0),
                "task_match_strength": (chunk.get("metadata") or {}).get("task_match_strength") or "none",
                "score": float(chunk.get("combined_score") or chunk.get("score") or 0.0),
                "source_uri": chunk.get("source_uri"),
            }
            for chunk in retrieved_chunks
        ],
        "retrieved_doc_count": len(retrieved_chunks),
        "docs_passed_to_prompt_count": int(response.audit.get("docs_passed_to_prompt_count") or 0),
        "schema_objects_used": schema_objects,
        "unknown_schema_usage": bool(
            sql_like_intent
            and any(obj == "UNKNOWN" for obj in response.audit.get("normalization_tags", {}).get("unconfirmed", []))
        ),
        "hallucination_score": float(response.audit.get("hallucination_score") or 0.0),
        "sql_index_used": bool(response.audit.get("sql_index_used")),
        "sql_generated": extract_sql(output),
        "verifier_status": response.audit.get("verification_status"),
        "verifier_passed": response.audit.get("verification_status") == "PASSED",
        "citations_present": bool(response.citations),
        "citation_count": len(response.citations),
        "citations": [citation.model_dump() for citation in response.citations],
        "rejected": is_safe_rejection(output),
        "query_task_signals_runtime": response.audit.get("query_task_signals") or [],
        "task_semantic_gate": response.audit.get("task_semantic_gate"),
        "task_semantic_strong_doc_count": int(response.audit.get("task_semantic_strong_doc_count") or 0),
        "task_semantic_medium_doc_count": int(response.audit.get("task_semantic_medium_doc_count") or 0),
        "task_match_rate": float(response.audit.get("task_match_rate") or 0.0),
        "task_semantic_correction": response.audit.get("task_semantic_correction"),
        "task_gate_reason": response.audit.get("task_gate_reason"),
        "intent_confidence": float(response.audit.get("intent_confidence") or 0.0),
        "module_confidence": float(response.audit.get("module_confidence") or 0.0),
        "decision_execution_mode": response.audit.get("decision_execution_mode"),
        "decision_reason": response.audit.get("decision_reason"),
        "decision_refusal_reason": response.audit.get("decision_refusal_reason"),
        "grounding_availability_score": float(response.audit.get("grounding_availability_score") or 0.0),
        "decision_grounding_signal_present": bool(response.audit.get("decision_grounding_signal_present")),
        "decision_sufficient_grounding_signal": bool(response.audit.get("decision_sufficient_grounding_signal")),
        "decision_trace_summary": {
            "intent_classification": response.audit.get("intent_classification"),
            "intent_confidence": float(response.audit.get("intent_confidence") or 0.0),
            "module_confidence": float(response.audit.get("module_confidence") or 0.0),
            "grounding_availability_score": float(response.audit.get("grounding_availability_score") or 0.0),
            "grounding_confidence_tier": response.audit.get("grounding_confidence_tier"),
            "decision_confidence_tier": response.audit.get("decision_confidence_tier"),
            "decision_execution_mode": response.audit.get("decision_execution_mode"),
            "decision_reason": response.audit.get("decision_reason"),
            "decision_refusal_reason": response.audit.get("decision_refusal_reason"),
        },
        "grounding_trace_summary": {
            "docs_retrieved_count": len(retrieved_chunks),
            "docs_passed_to_prompt_count": int(response.audit.get("docs_passed_to_prompt_count") or 0),
            "citation_count": len(response.citations or []),
            "exact_module_doc_count": int(response.audit.get("exact_module_doc_count") or 0),
            "task_semantic_gate": response.audit.get("task_semantic_gate"),
            "task_match_rate": float(response.audit.get("task_match_rate") or 0.0),
            "selected_corpora": sorted(
                {
                    str((chunk.get("metadata") or {}).get("corpus") or "")
                    for chunk in retrieved_chunks
                    if (chunk.get("metadata") or {}).get("corpus")
                }
            ),
        },
        "response_time_sec": round(time.perf_counter() - started_at, 2),
        "output": output,
        "runtime_error": None,
    }
    record["correction_present"] = correction_present(output, record.get("task_semantic_correction"))
    record["exact_module_top_hit"] = bool(
        record["retrieved_docs"] and record["retrieved_docs"][0].get("module") == record.get("benchmark_module")
    )
    record["same_family_bleed_through"] = bool(
        record.get("benchmark_module") in LEAF_FINANCIALS
        and any(
            family_for(doc.get("module") or doc.get("module_family")) == family_for(record.get("benchmark_module"))
            and doc.get("module") not in {record.get("benchmark_module"), None, ""}
            for doc in record["retrieved_docs"]
        )
    )
    return record


async def execute_case_with_engine(engine: Any, tenant: Any, case: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    from backend.core.schemas.api import ChatRequest, Message, Role

    started_at = time.perf_counter()
    request = ChatRequest(
        messages=[Message(role=Role.USER, content=case["question"])],
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=0.9,
        repeat_penalty=1.1,
    )
    try:
        response = await engine.chat(db=None, tenant=tenant, request=request, require_citations=True)
        return build_case_record_from_response(case, response, started_at)
    except Exception as exc:
        return build_runtime_error_record(case, started_at, str(exc))


def benchmark_worker_main(request_queue: Any, response_queue: Any, max_tokens: int) -> None:
    from backend.core.rag.engine import RAGEngine

    engine = RAGEngine()
    tenant = SimpleNamespace(id="demo")
    response_queue.put({"type": "READY"})

    while True:
        job = request_queue.get()
        if job is None:
            return
        case = job["case"]
        record = asyncio.run(execute_case_with_engine(engine, tenant, case, max_tokens))
        response_queue.put({"type": "RESULT", "case_id": case["id"], "record": record})


class BenchmarkWorkerError(RuntimeError):
    pass


class BenchmarkWorkerSession:
    def __init__(
        self,
        max_tokens: int,
        case_timeout_sec: int,
        startup_timeout_sec: int,
    ) -> None:
        self.ctx = mp.get_context("spawn")
        self.max_tokens = max_tokens
        self.case_timeout_sec = case_timeout_sec
        self.startup_timeout_sec = startup_timeout_sec
        self.request_queue: Any | None = None
        self.response_queue: Any | None = None
        self.process: Any | None = None

    def start(self) -> None:
        if self.process and self.process.is_alive():
            return

        self.request_queue = self.ctx.Queue(maxsize=1)
        self.response_queue = self.ctx.Queue(maxsize=1)
        self.process = self.ctx.Process(
            target=benchmark_worker_main,
            args=(self.request_queue, self.response_queue, self.max_tokens),
            daemon=True,
        )
        self.process.start()
        message = self._get_message(self.startup_timeout_sec)
        if message.get("type") != "READY":
            self.stop()
            raise BenchmarkWorkerError(f"worker failed to initialize: {message}")

    def stop(self) -> None:
        if self.request_queue is not None:
            try:
                self.request_queue.put_nowait(None)
            except Exception:
                pass
        if self.process is not None:
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
        self.process = None
        self.request_queue = None
        self.response_queue = None

    def restart(self) -> None:
        self.stop()
        self.start()

    def run_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        self.start()
        if self.request_queue is None:
            raise BenchmarkWorkerError("worker request queue unavailable")
        self.request_queue.put({"case": case})
        message = self._get_message(self.case_timeout_sec)
        if message.get("type") != "RESULT" or message.get("case_id") != case["id"]:
            raise BenchmarkWorkerError(f"unexpected worker response: {message}")
        return message["record"]

    def _get_message(self, timeout_sec: int) -> Dict[str, Any]:
        if self.response_queue is None:
            raise BenchmarkWorkerError("worker response queue unavailable")
        try:
            return self.response_queue.get(timeout=timeout_sec)
        except queue.Empty as exc:
            raise TimeoutError(f"worker timed out after {timeout_sec} seconds") from exc


def load_completed_records(output_path: Path, cases_dir: Path, valid_case_ids: set[str]) -> Dict[str, Dict[str, Any]]:
    completed: Dict[str, Dict[str, Any]] = {}

    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("id") or "")
            if case_id in valid_case_ids:
                completed[case_id] = row

    if cases_dir.exists():
        for case_file in sorted(cases_dir.glob("*.json")):
            row = json.loads(case_file.read_text(encoding="utf-8"))
            case_id = str(row.get("id") or case_file.stem)
            if case_id in valid_case_ids:
                completed[case_id] = row

    return completed


def quality_tag(record: Dict[str, Any]) -> str:
    output = (record.get("output") or "").lower()
    benchmark_module = record.get("benchmark_module")
    detected_module = record.get("module_detected")
    detected_family = record.get("module_family_detected")
    intent = record.get("intent_detected") or record.get("benchmark_intent") or ""

    if record.get("runtime_error"):
        return "runtime_error"
    if record.get("rejected"):
        return "over_rejection"
    if benchmark_module in LEAF_FINANCIALS and detected_module != benchmark_module:
        return "wrong_module_grounded"
    if benchmark_module not in LEAF_FINANCIALS and family_for(detected_family or detected_module) != family_for(benchmark_module):
        return "wrong_module_grounded"
    if record.get("sql_generated") and intent not in SQL_EXPECTED_TASKS and intent != "troubleshooting":
        return "sql_misuse"
    if FAIL_CLOSED_MESSAGE.lower() in output and output.strip() != FAIL_CLOSED_MESSAGE.lower():
        return "mixed_refusal"
    if intent in DOCS_EXPECTED_TASKS and not record.get("docs_passed_to_prompt_count"):
        return "ranking_issue_missing_docs"
    if any(pattern in output for pattern in INFERENCE_PATTERNS):
        return "weak_inference"
    if any(pattern in output for pattern in GENERIC_PATTERNS):
        return "weak_generic"
    if len((record.get("output") or "").split()) < 28 and intent in DOCS_EXPECTED_TASKS:
        return "weak_explanation"
    return "likely_correct"


def score_case(record: Dict[str, Any]) -> str:
    expected = record.get("expected_behavior")
    verifier_status = record.get("verifier_status")
    correction_ok = bool(record.get("correction_present"))

    if expected == "sql":
        if record.get("rejected") or not record.get("sql_generated") or verifier_status != "PASSED":
            return "sql_failure"
        if not record.get("citations_present"):
            return "missing-citation"
        return "grounded_correct"

    if expected in {"refusal", "correction_then_refusal"}:
        if record.get("rejected"):
            if expected == "correction_then_refusal":
                if correction_ok or verifier_status == "FAILED_TASK_MODULE_CORRECTION":
                    return "safe_refusal_correct"
                return "semantic_mismatch"
            return "safe_refusal_correct"
        if record.get("quality_tag") == "wrong_module_grounded" or record.get("same_family_bleed_through"):
            return "wrong-module answer"
        if not record.get("citations_present"):
            return "missing-citation"
        return "semantic_mismatch"

    if record.get("runtime_error"):
        return "verifier_failure"
    if verifier_status != "PASSED" or record.get("rejected"):
        return "verifier_failure"
    if not record.get("citations_present"):
        return "missing-citation"
    if record.get("quality_tag") == "wrong_module_grounded" or record.get("same_family_bleed_through"):
        return "wrong-module answer"
    if record.get("unknown_schema_usage") or float(record.get("hallucination_score") or 0.0) > 0.0:
        return "hallucination_error"
    if record.get("quality_tag") in {"weak_inference", "weak_generic", "weak_explanation", "ranking_issue_missing_docs", "mixed_refusal"}:
        return "semantic_mismatch"
    if expected == "troubleshooting" and not record.get("docs_passed_to_prompt_count"):
        return "semantic_mismatch"
    return "grounded_correct"


def aggregate_breakdown(records: List[Dict[str, Any]], key_name: str) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(key_name) or "UNKNOWN")].append(record)

    summary: Dict[str, Any] = {}
    for key, items in sorted(grouped.items()):
        total = len(items)
        outcomes = Counter(item["scoring_outcome"] for item in items)
        summary[key] = {
            "total": total,
            "grounded_correct": outcomes.get("grounded_correct", 0),
            "safe_refusal_correct": outcomes.get("safe_refusal_correct", 0),
            "hallucination_error": outcomes.get("hallucination_error", 0),
            "semantic_mismatch": outcomes.get("semantic_mismatch", 0),
            "wrong-module answer": outcomes.get("wrong-module answer", 0),
            "missing-citation": outcomes.get("missing-citation", 0),
            "verifier_failure": outcomes.get("verifier_failure", 0),
            "sql_failure": outcomes.get("sql_failure", 0),
            "success_rate_pct": round(
                (outcomes.get("grounded_correct", 0) + outcomes.get("safe_refusal_correct", 0)) / max(total, 1) * 100,
                2,
            ),
        }
    return summary


def top_failure_patterns(records: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    counter: Counter[Tuple[str, str, str, str]] = Counter()
    examples: Dict[Tuple[str, str, str, str], List[str]] = defaultdict(list)

    for record in records:
        if record["scoring_outcome"] in {"grounded_correct", "safe_refusal_correct"}:
            continue
        key = (
            record["scoring_outcome"],
            str(record.get("benchmark_module") or "UNKNOWN"),
            str(record.get("benchmark_intent") or "UNKNOWN"),
            str(record.get("benchmark_task_signal") or "none"),
        )
        counter[key] += 1
        if len(examples[key]) < 5:
            examples[key].append(str(record.get("id") or ""))

    results: List[Dict[str, Any]] = []
    for key, count in counter.most_common(limit):
        outcome, module, intent, task = key
        results.append(
            {
                "count": count,
                "scoring_outcome": outcome,
                "module": module,
                "intent": intent,
                "task_signal": task,
                "example_ids": examples[key],
            }
        )
    return results


def failure_category(record: Dict[str, Any]) -> str:
    outcome = str(record.get("scoring_outcome") or "UNKNOWN")
    expected = str(record.get("expected_behavior") or "")
    intent = str(record.get("benchmark_intent") or "")
    quality = str(record.get("quality_tag") or "")
    verifier_status = str(record.get("verifier_status") or "")
    task_gate = str(record.get("task_semantic_gate") or "")

    if outcome == "wrong-module answer":
        if expected == "correction_then_refusal":
            return "under-refusal on cross-module negative"
        if intent == "troubleshooting":
            return "wrong module routing in troubleshooting"
        return "wrong module routing on positive query"
    if outcome == "semantic_mismatch":
        if expected == "correction_then_refusal":
            return "task-semantic mismatch on correction/refusal case"
        if task_gate == "FAILED":
            return "task-semantic gate fired without safe refusal"
        return "task-semantic mismatch despite passing gate"
    if outcome == "verifier_failure":
        if record.get("rejected"):
            if intent == "troubleshooting":
                return "over-refusal in troubleshooting"
            if verifier_status in FAIL_OPEN_STATUSES:
                return "over-refusal from semantic or module gate"
            return "over-refusal despite retained grounding"
        if intent == "troubleshooting":
            return "troubleshooting answer failed verifier"
        return "answer failed verifier"
    if outcome == "sql_failure":
        return "sql lane failure"
    if quality == "wrong_module_grounded":
        return "wrong module grounding"
    return outcome


def top_failure_categories(records: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    examples: Dict[str, List[str]] = defaultdict(list)

    for record in records:
        if record["scoring_outcome"] in {"grounded_correct", "safe_refusal_correct"}:
            continue
        category = failure_category(record)
        counter[category] += 1
        if len(examples[category]) < 5:
            examples[category].append(str(record.get("id") or ""))

    return [
        {
            "category": category,
            "count": count,
            "example_ids": examples[category],
        }
        for category, count in counter.most_common(limit)
    ]


def benchmark_go_no_go(summary: Dict[str, Any]) -> Dict[str, Any]:
    module_results = summary["results_by_module"]
    ready_modules: List[str] = []
    not_ready_modules: List[str] = []

    for module, stats in module_results.items():
        success_rate = float(stats.get("success_rate_pct") or 0.0)
        hard_failures = (
            int(stats.get("hallucination_error", 0))
            + int(stats.get("wrong-module answer", 0))
            + int(stats.get("missing-citation", 0))
        )
        if success_rate >= 80.0 and hard_failures == 0:
            ready_modules.append(module)
        else:
            not_ready_modules.append(module)

    overall_success = float(summary.get("trusted_outcome_rate_pct") or 0.0)
    hallucination_rate = float(summary.get("hallucination_rate_pct") or 0.0)
    semantic_rate = float(summary.get("semantic_mismatch_rate_pct") or 0.0)
    sql_total = int(summary.get("requested_buckets", {}).get("sql_generation", 0))
    sql_failures = int(summary.get("scoring_breakdown", {}).get("sql_failure", 0))

    return {
        "safe_for_constrained_pilot": overall_success >= 75.0 and hallucination_rate == 0.0,
        "safe_for_broad_non_sql_rollout": overall_success >= 85.0 and hallucination_rate == 0.0 and semantic_rate <= 10.0,
        "not_ready_for_sql_rollout": sql_total == 0 or sql_failures > 0,
        "modules_ready": ready_modules,
        "modules_not_ready": not_ready_modules,
    }


def summarize_results(records: List[Dict[str, Any]], label: str, max_tokens: int, plan_summary: Dict[str, Any]) -> Dict[str, Any]:
    outcome_counts = Counter(record["scoring_outcome"] for record in records)
    answered_records = [record for record in records if not record.get("rejected")]
    refusal_expected_records = [record for record in records if record.get("safe_refusal_expected")]
    corrections_expected_records = [record for record in records if record.get("expected_behavior") == "correction_then_refusal"]

    evaluation_scorecards = [record.get("evaluation") or build_case_scorecard(record) for record in records]
    evaluation_summary = aggregate_scorecards(evaluation_scorecards, records)

    summary = {
        "label": label,
        "sample_size": len(records),
        "max_tokens": max_tokens,
        "primary_segments": plan_summary["primary_segments"],
        "requested_buckets": plan_summary["requested_buckets"],
        "expected_behavior": plan_summary["expected_behavior"],
        "scoring_breakdown": dict(outcome_counts),
        "trusted_outcome_rate_pct": round(
            (outcome_counts.get("grounded_correct", 0) + outcome_counts.get("safe_refusal_correct", 0))
            / max(len(records), 1)
            * 100,
            2,
        ),
        "refusal_correctness_pct": round(
            outcome_counts.get("safe_refusal_correct", 0) / max(len(refusal_expected_records), 1) * 100,
            2,
        ),
        "hallucination_rate_pct": round(outcome_counts.get("hallucination_error", 0) / max(len(records), 1) * 100, 2),
        "semantic_mismatch_rate_pct": round(outcome_counts.get("semantic_mismatch", 0) / max(len(records), 1) * 100, 2),
        "wrong_module_answer_rate_pct": round(outcome_counts.get("wrong-module answer", 0) / max(len(records), 1) * 100, 2),
        "citation_presence_pct": round(
            sum(1 for record in answered_records if record.get("citations_present")) / max(len(answered_records), 1) * 100,
            2,
        ),
        "verifier_pass_pct": round(
            sum(1 for record in records if record.get("verifier_status") == "PASSED") / max(len(records), 1) * 100,
            2,
        ),
        "module_detection_accuracy_pct": round(
            sum(
                1
                for record in records
                if family_for(record.get("module_detected") or record.get("module_family_detected"))
                == family_for(record.get("benchmark_module"))
            )
            / max(len(records), 1)
            * 100,
            2,
        ),
        "intent_match_pct": round(
            sum(1 for record in records if record.get("intent_detected") == record.get("benchmark_intent")) / max(len(records), 1) * 100,
            2,
        ),
        "correction_then_refusal_correct_pct": round(
            sum(
                1
                for record in corrections_expected_records
                if record.get("scoring_outcome") == "safe_refusal_correct" and record.get("correction_present")
            )
            / max(len(corrections_expected_records), 1)
            * 100,
            2,
        ),
        "results_by_module": aggregate_breakdown(records, "benchmark_module"),
        "results_by_intent": aggregate_breakdown(records, "benchmark_intent"),
        "results_by_segment": aggregate_breakdown(records, "benchmark_segment"),
        "top_failure_categories": top_failure_categories(records, limit=10),
        "top_failure_patterns": top_failure_patterns(records, limit=20),
        "top_20_failure_patterns": top_failure_patterns(records, limit=20),
        "evaluation_summary": evaluation_summary,
        "task_aware_primary_metrics": evaluation_summary.get("primary_metrics", {}),
        "per_task_type_metrics": evaluation_summary.get("per_task_type", {}),
        "per_module_metrics": evaluation_summary.get("per_module", {}),
        "per_difficulty_metrics": evaluation_summary.get("per_difficulty", {}),
        "citation_correctness_pct": float(evaluation_summary.get("primary_metrics", {}).get("citation_correctness_pct", 0.0)),
        "semantic_correctness_pct": float(evaluation_summary.get("primary_metrics", {}).get("semantic_correctness_pct", 0.0)),
        "over_refusal_pct": float(evaluation_summary.get("primary_metrics", {}).get("over_refusal_pct", 0.0)),
        "grounding_supported_answer_rate_pct": float(
            evaluation_summary.get("primary_metrics", {}).get("grounding_supported_answer_rate_pct", 0.0)
        ),
        "scoring_rubric_version": SCORING_RUBRIC["version"],
    }
    summary["go_no_go"] = benchmark_go_no_go(summary)
    return summary


def build_plan(cases: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    router = TaskRouter()
    plan_records: List[Dict[str, Any]] = []
    primary_segment_counts = Counter()
    behavior_counts = Counter()

    for case in cases:
        plan = classify_case(case, router)
        override_intent = str(case.get("task_type") or "").strip()
        override_behavior = str(case.get("expected_behavior") or "").strip()
        override_segment = str(case.get("benchmark_segment") or "").strip()
        lane = str(case.get("lane") or "").strip()

        if override_intent:
            plan["benchmark_intent"] = override_intent
        if override_behavior:
            if override_behavior == "sql":
                plan["expected_behavior"] = "sql"
            elif override_behavior == "refusal":
                plan["expected_behavior"] = "refusal"
                plan["safe_refusal_expected"] = True
            elif override_behavior == "formula":
                plan["expected_behavior"] = "answer"
            else:
                plan["expected_behavior"] = override_behavior
        if override_segment:
            plan["benchmark_segment"] = override_segment
        elif lane == "sql":
            plan["benchmark_segment"] = "sql_generation"
        elif lane == "fast_formula":
            plan["benchmark_segment"] = "fast_formula_generation"
        elif lane == "troubleshooting":
            plan["benchmark_segment"] = "troubleshooting"

        record = {
            "id": case["id"],
            "category": case.get("category"),
            "module": case.get("module"),
            "benchmark_module": case.get("module"),
            "difficulty": case.get("difficulty"),
            "benchmark_answer": case.get("answer"),
            "question": case.get("question"),
            "tags": case.get("tags") or [],
            **plan,
        }
        plan_records.append(record)
        primary_segment_counts[record["benchmark_segment"]] += 1
        behavior_counts[record["expected_behavior"]] += 1

    requested_buckets = {
        "doc_grounded_procedure": primary_segment_counts.get("doc_grounded_procedure", 0),
        "troubleshooting": primary_segment_counts.get("troubleshooting", 0),
        "module_ambiguous": primary_segment_counts.get("module_ambiguous", 0),
        "safe_refusal_expected": sum(1 for record in plan_records if record["safe_refusal_expected"]),
        "sql_generation": primary_segment_counts.get("sql_generation", 0),
        "fast_formula_generation": primary_segment_counts.get("fast_formula_generation", 0),
        "cross-module negative cases": sum(1 for record in plan_records if record["cross_module_negative_case"]),
    }

    plan_summary = {
        "total_cases": len(plan_records),
        "primary_segments": {segment: primary_segment_counts.get(segment, 0) for segment in PRIMARY_SEGMENTS},
        "requested_buckets": requested_buckets,
        "segments": requested_buckets,
        "expected_behavior": dict(sorted(behavior_counts.items())),
        "modules": dict(sorted(Counter(record["module"] for record in plan_records).items())),
        "intents": dict(sorted(Counter(record["benchmark_intent"] for record in plan_records).items())),
        "cross_module_negative_cases": requested_buckets["cross-module negative cases"],
        "safe_refusal_expected": requested_buckets["safe_refusal_expected"],
        "sql_generation_cases": requested_buckets["sql_generation"],
    }
    return plan_records, plan_summary


async def run_benchmark(
    label: str,
    max_tokens: int,
    limit: Optional[int],
    input_path: Optional[Path],
    case_ids: Optional[set[str]],
    plan_only: bool,
    resume: bool,
    flush_every: int,
    case_timeout_sec: int,
    case_retries: int,
    worker_startup_timeout_sec: int,
    recycle_worker_every: int,
) -> Dict[str, Any]:
    cases = load_cases(limit=limit, input_path=input_path, case_ids=case_ids)
    plan_records, plan_summary = build_plan(cases)

    run_dir = OUTPUT_DIR / label
    cases_dir = run_dir / "cases"
    run_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)

    plan_path = run_dir / "benchmark_plan.jsonl"
    plan_summary_path = run_dir / "benchmark_plan_summary.json"
    with plan_path.open("w", encoding="utf-8") as handle:
        for record in plan_records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    write_json(plan_summary_path, plan_summary)

    if plan_only:
        return {
            "label": label,
            "plan_only": True,
            "plan_path": str(plan_path),
            "plan_summary_path": str(plan_summary_path),
            **plan_summary,
        }

    output_path = run_dir / "benchmark_results.jsonl"
    summary_path = run_dir / "benchmark_summary.json"
    progress_log_path = run_dir / "benchmark_progress.log"

    completed: Dict[str, Dict[str, Any]] = {}
    if resume:
        completed = load_completed_records(
            output_path=output_path,
            cases_dir=cases_dir,
            valid_case_ids={row["id"] for row in plan_records},
        )
    results: List[Dict[str, Any]] = [completed[row["id"]] for row in plan_records if row["id"] in completed]
    worker = BenchmarkWorkerSession(
        max_tokens=max_tokens,
        case_timeout_sec=case_timeout_sec,
        startup_timeout_sec=worker_startup_timeout_sec,
    )

    def persist_partial() -> None:
        deduped = {record["id"]: record for record in results}
        ordered_results = [deduped[row["id"]] for row in plan_records if row["id"] in deduped]
        with output_path.open("w", encoding="utf-8") as handle:
            for record in ordered_results:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        summary = summarize_results(ordered_results, label=label, max_tokens=max_tokens, plan_summary=plan_summary)
        write_json(summary_path, summary)

    if results:
        persist_partial()

    try:
        for index, case in enumerate(plan_records, start=1):
            case_id = case["id"]
            if case_id in completed:
                continue

            if recycle_worker_every and len(results) and len(results) % recycle_worker_every == 0:
                worker.restart()

            message = f"[benchmark] case {index}/{len(plan_records)}: {case_id} | {case['module']} | {case['benchmark_segment']} | {case['question'][:100]}"
            print(message)
            with progress_log_path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

            started_at = time.perf_counter()
            record: Dict[str, Any] | None = None
            last_error = ""

            for attempt in range(case_retries + 1):
                try:
                    record = worker.run_case(case)
                    break
                except TimeoutError as exc:
                    last_error = f"CASE_TIMEOUT after {case_timeout_sec}s"
                    with progress_log_path.open("a", encoding="utf-8") as handle:
                        handle.write(f"[benchmark] timeout {case_id} attempt {attempt + 1}/{case_retries + 1}: {exc}\n")
                    worker.stop()
                except BenchmarkWorkerError as exc:
                    last_error = f"WORKER_ERROR: {exc}"
                    with progress_log_path.open("a", encoding="utf-8") as handle:
                        handle.write(f"[benchmark] worker_error {case_id} attempt {attempt + 1}/{case_retries + 1}: {exc}\n")
                    worker.stop()

            if record is None:
                record = build_runtime_error_record(case, started_at, last_error or "UNKNOWN_WORKER_FAILURE", verifier_status="RUNTIME_TIMEOUT")

            record["quality_tag"] = quality_tag(record)
            record["scoring_outcome"] = score_case(record)
            record["failure_category"] = failure_category(record)
            record["evaluation"] = build_case_scorecard(record)
            results.append(record)
            completed[case_id] = record
            write_json(cases_dir / f"{case_id}.json", record)

            if len(results) % flush_every == 0:
                persist_partial()
    finally:
        worker.stop()

    persist_partial()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["output_path"] = str(output_path)
    summary["plan_path"] = str(plan_path)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run segmented production-style Oracle Fusion benchmark.")
    parser.add_argument("--label", type=str, default="benchmark_1000")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--input-path", type=str, default="")
    parser.add_argument("--case-id-file", type=str, default="")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--case-timeout-sec", type=int, default=DEFAULT_CASE_TIMEOUT_SEC)
    parser.add_argument("--case-retries", type=int, default=DEFAULT_CASE_RETRIES)
    parser.add_argument("--worker-startup-timeout-sec", type=int, default=DEFAULT_WORKER_STARTUP_TIMEOUT_SEC)
    parser.add_argument("--recycle-worker-every", type=int, default=DEFAULT_WORKER_RECYCLE_EVERY)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_path = Path(args.input_path) if args.input_path else None
    case_ids = None
    if args.case_id_file:
        case_ids = {
            str(item).strip()
            for item in json.loads(Path(args.case_id_file).read_text(encoding="utf-8"))
            if str(item).strip()
        }
    result = asyncio.run(
        run_benchmark(
            label=args.label,
            max_tokens=args.max_tokens,
            limit=args.limit or None,
            input_path=input_path,
            case_ids=case_ids,
            plan_only=args.plan_only,
            resume=args.resume,
            flush_every=max(1, args.flush_every),
            case_timeout_sec=max(1, args.case_timeout_sec),
            case_retries=max(0, args.case_retries),
            worker_startup_timeout_sec=max(30, args.worker_startup_timeout_sec),
            recycle_worker_every=max(0, args.recycle_worker_every),
        )
    )
    if result:
        print(json.dumps(result, indent=2, ensure_ascii=True))
