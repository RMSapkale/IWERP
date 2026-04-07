import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "backend"))

from backend.scripts.run_production_benchmark import summarize_results

INPUT_RESULTS = BASE_DIR / "production_benchmark" / "production_1000_certification_v1" / "benchmark_results.jsonl"
OUTPUT_DIR = BASE_DIR / "production_benchmark" / "decision_layer_correction_pass_v1"
SLICE_SIZE = 200
CATEGORY_TARGETS = {
    "A. over-strict gating": 34,
    "B. missing intent patterns": 36,
    "C. ambiguous query misclassification": 24,
    "D. module boundary confusion": 44,
    "E. insufficient grounding confidence signals": 18,
    "F. overly aggressive fail-closed triggers": 44,
}
AMBIGUOUS_MODULES = {"Financials", "HCM", "Procurement", "Supply Chain"}


def numeric_case_id(case_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in str(case_id or "") if ch.isdigit())
    return (int(digits or "0"), str(case_id or ""))


def load_records() -> List[Dict[str, Any]]:
    return [json.loads(line) for line in INPUT_RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]


def classify_root_cause(record: Dict[str, Any]) -> str:
    outcome = str(record.get("scoring_outcome") or "")
    expected = str(record.get("expected_behavior") or "")
    verifier_status = str(record.get("verifier_status") or "")
    benchmark_module = str(record.get("benchmark_module") or "")
    benchmark_segment = str(record.get("benchmark_segment") or "")
    rejected = bool(record.get("rejected"))

    if outcome == "wrong-module answer":
        if benchmark_segment == "module_ambiguous" or benchmark_module in AMBIGUOUS_MODULES:
            return "C. ambiguous query misclassification"
        return "D. module boundary confusion"

    if verifier_status == "FAILED_FINANCE_LEAF_NO_EXACT_DOCS":
        return "F. overly aggressive fail-closed triggers"

    if verifier_status in {"FAILED_TASK_SEMANTIC_NO_STRONG_MATCH", "FAILED_TASK_MODULE_CORRECTION"}:
        return "A. over-strict gating"

    if rejected and verifier_status == "PASSED":
        return "E. insufficient grounding confidence signals"

    if outcome == "semantic_mismatch":
        if benchmark_segment == "module_ambiguous" or benchmark_module in AMBIGUOUS_MODULES:
            return "C. ambiguous query misclassification"
        if expected in {"correction_then_refusal", "refusal"}:
            return "B. missing intent patterns"
        if rejected:
            return "E. insufficient grounding confidence signals"
        return "B. missing intent patterns"

    if rejected:
        return "A. over-strict gating"

    return "B. missing intent patterns"


def refusal_reason(record: Dict[str, Any]) -> str | None:
    if not record.get("rejected"):
        return None
    return str(record.get("decision_refusal_reason") or record.get("verifier_status") or record.get("task_gate_reason") or "")


def gating_path(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_semantic_gate": record.get("task_semantic_gate"),
        "task_gate_reason": record.get("task_gate_reason"),
        "verifier_status": record.get("verifier_status"),
        "decision_execution_mode": record.get("decision_execution_mode"),
        "decision_reason": record.get("decision_reason"),
        "decision_refusal_reason": record.get("decision_refusal_reason"),
        "intent_confidence": record.get("intent_confidence"),
        "module_confidence": record.get("module_confidence"),
        "grounding_availability_score": record.get("grounding_availability_score"),
    }


def build_failure_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    decision_failures: List[Dict[str, Any]] = []
    for record in records:
        outcome = str(record.get("scoring_outcome") or "")
        if outcome not in {"semantic_mismatch", "wrong-module answer", "verifier_failure"}:
            continue
        if str(record.get("expected_behavior") or "") == "sql":
            continue
        row = {
            "case_id": record.get("id"),
            "user_query": record.get("question"),
            "expected_module": record.get("benchmark_module"),
            "selected_module": record.get("module_detected"),
            "intent": record.get("benchmark_intent"),
            "expected_behavior": record.get("expected_behavior"),
            "scoring_outcome": outcome,
            "refusal_reason": refusal_reason(record),
            "gating_decision_path": gating_path(record),
            "retrieved_docs": [
                {
                    "module": doc.get("module"),
                    "title": doc.get("title"),
                    "task_match_strength": doc.get("task_match_strength"),
                    "score": doc.get("score"),
                }
                for doc in (record.get("retrieved_docs") or [])[:5]
            ],
            "task_match": {
                "gate": record.get("task_semantic_gate"),
                "reason": record.get("task_gate_reason"),
                "rate": record.get("task_match_rate"),
            },
            "module_match": {
                "benchmark_module": record.get("benchmark_module"),
                "selected_module": record.get("module_detected"),
                "same_family_bleed_through": record.get("same_family_bleed_through"),
            },
            "refusal_or_wrong_answer": "refusal" if record.get("rejected") else "wrong_answer",
            "root_cause_category": classify_root_cause(record),
        }
        decision_failures.append(row)
    decision_failures.sort(key=lambda item: numeric_case_id(str(item["case_id"])))
    return decision_failures


def build_slice_ids(failure_rows: List[Dict[str, Any]]) -> List[str]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    seen: set[str] = set()
    for row in failure_rows:
        grouped[row["root_cause_category"]].append(str(row["case_id"]))

    selected: List[str] = []
    for category, target in CATEGORY_TARGETS.items():
        for case_id in grouped.get(category, []):
            if case_id in seen:
                continue
            selected.append(case_id)
            seen.add(case_id)
            if len([item for item in selected if item in grouped.get(category, [])]) >= target:
                break

    if len(selected) < SLICE_SIZE:
        for row in failure_rows:
            case_id = str(row["case_id"])
            if case_id in seen:
                continue
            selected.append(case_id)
            seen.add(case_id)
            if len(selected) >= SLICE_SIZE:
                break

    return sorted(selected[:SLICE_SIZE], key=numeric_case_id)


def plan_summary_for_slice(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    primary_segments = Counter(str(record.get("benchmark_segment") or "UNKNOWN") for record in records)
    requested_buckets = {
        "doc_grounded_procedure": primary_segments.get("doc_grounded_procedure", 0),
        "troubleshooting": primary_segments.get("troubleshooting", 0),
        "module_ambiguous": primary_segments.get("module_ambiguous", 0),
        "safe_refusal_expected": sum(1 for record in records if record.get("safe_refusal_expected")),
        "sql_generation": primary_segments.get("sql_generation", 0),
        "cross-module negative cases": sum(1 for record in records if record.get("cross_module_negative_case")),
    }
    return {
        "primary_segments": dict(primary_segments),
        "requested_buckets": requested_buckets,
        "expected_behavior": dict(Counter(str(record.get("expected_behavior") or "UNKNOWN") for record in records)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    failure_rows = build_failure_rows(records)
    slice_ids = build_slice_ids(failure_rows)
    slice_records = [record for record in records if str(record.get("id")) in set(slice_ids)]
    slice_records.sort(key=lambda item: numeric_case_id(str(item.get("id") or "")))

    failure_report_path = OUTPUT_DIR / "decision_failure_report.json"
    failure_rows_path = OUTPUT_DIR / "decision_failure_rows.jsonl"
    slice_ids_path = OUTPUT_DIR / "decision_slice_case_ids.json"
    slice_baseline_path = OUTPUT_DIR / "decision_slice_baseline_summary.json"

    failure_summary = {
        "source_results": str(INPUT_RESULTS),
        "decision_failure_count": len(failure_rows),
        "root_cause_counts": dict(Counter(row["root_cause_category"] for row in failure_rows)),
        "selected_slice_size": len(slice_ids),
        "selected_slice_case_ids": slice_ids,
    }

    failure_report_path.write_text(json.dumps(failure_summary, indent=2, ensure_ascii=True), encoding="utf-8")
    with failure_rows_path.open("w", encoding="utf-8") as handle:
        for row in failure_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    slice_ids_path.write_text(json.dumps(slice_ids, indent=2, ensure_ascii=True), encoding="utf-8")

    baseline_summary = summarize_results(
        slice_records,
        label="decision_layer_200_slice_baseline",
        max_tokens=120,
        plan_summary=plan_summary_for_slice(slice_records),
    )
    slice_baseline_path.write_text(json.dumps(baseline_summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(json.dumps(failure_summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
