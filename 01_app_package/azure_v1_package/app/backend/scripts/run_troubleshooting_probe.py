import argparse
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "backend"))

from backend.scripts.run_production_benchmark import (
    FAIL_CLOSED_MESSAGE,
    LEAF_FINANCIALS,
    correction_present,
    extract_sql,
    family_for,
    is_safe_rejection,
    load_cases,
    quality_tag,
    score_case,
    summarize_results,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a troubleshooting-only benchmark probe.")
    parser.add_argument("--label", default="stabilization_troubleshooting_probe_v6")
    parser.add_argument(
        "--subset-plan",
        default=str(BASE_DIR / "production_benchmark" / "stabilization_200_subset_v2" / "subset_plan.jsonl"),
    )
    parser.add_argument(
        "--baseline-results",
        default=str(BASE_DIR / "production_benchmark" / "production_1000_v1" / "benchmark_results.jsonl"),
    )
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_probe_plan(path: Path, limit: int = 0) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    probe = [row for row in records if row.get("benchmark_segment") == "troubleshooting"]
    return probe[:limit] if limit else probe


def persist_partial(
    results: list[dict],
    probe_ids: list[str],
    summary_path: Path,
    partial_path: Path,
    label: str,
    max_tokens: int,
) -> dict:
    ordered = sorted(results, key=lambda row: probe_ids.index(row["id"]))
    with partial_path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    summary = summarize_results(
        ordered,
        label=label,
        max_tokens=max_tokens,
        plan_summary={
            "primary_segments": {"troubleshooting": len(probe_ids)},
            "requested_buckets": {"troubleshooting": len(probe_ids)},
            "expected_behavior": {"troubleshooting": len(probe_ids)},
        },
    )
    write_json(summary_path, summary)
    return summary


async def run_probe(args: argparse.Namespace) -> dict:
    run_dir = BASE_DIR / "production_benchmark" / args.label
    run_dir.mkdir(parents=True, exist_ok=True)

    subset_plan_path = Path(args.subset_plan)
    baseline_results_path = Path(args.baseline_results)
    log_path = run_dir / "run.log"
    error_path = run_dir / "error.log"
    results_path = run_dir / "results.jsonl"
    partial_path = run_dir / "results.partial.jsonl"
    summary_path = run_dir / "summary.json"
    delta_path = run_dir / "delta.json"
    baseline_summary_path = run_dir / "baseline_summary.json"
    probe_case_ids_path = run_dir / "probe_case_ids.json"

    probe_plan = load_probe_plan(subset_plan_path, limit=args.limit)
    probe_ids = [row["id"] for row in probe_plan]
    plan_by_id = {row["id"]: row for row in probe_plan}
    write_json(probe_case_ids_path, probe_ids)

    baseline_results = [json.loads(line) for line in baseline_results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    baseline_probe = [row for row in baseline_results if row.get("id") in plan_by_id]
    baseline_summary = summarize_results(
        baseline_probe,
        label=f"{args.label}_baseline",
        max_tokens=args.max_tokens,
        plan_summary={
            "primary_segments": {"troubleshooting": len(probe_ids)},
            "requested_buckets": {"troubleshooting": len(probe_ids)},
            "expected_behavior": {"troubleshooting": len(probe_ids)},
        },
    )
    write_json(baseline_summary_path, baseline_summary)

    cases = {row["id"]: row for row in load_cases(limit=None)}

    try:
        from backend.core.rag.engine import RAGEngine
        from backend.core.schemas.api import ChatRequest, Message, Role
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise

    tenant = SimpleNamespace(id="demo")
    try:
        engine = RAGEngine()
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise

    results: list[dict] = []

    for idx, case_id in enumerate(probe_ids, start=1):
        case = cases[case_id]
        plan = plan_by_id[case_id]
        message = f"[troubleshooting-probe] {idx}/{len(probe_ids)} {case_id} | {case['module']} | {case['question'][:120]}"
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(message + "\n")

        request = ChatRequest(
            messages=[Message(role=Role.USER, content=case["question"])],
            max_tokens=args.max_tokens,
            temperature=0.0,
            top_p=0.9,
            repeat_penalty=1.1,
        )
        started_at = time.perf_counter()

        try:
            response = await engine.chat(db=None, tenant=tenant, request=request, require_citations=True)
            output = response.choices[0]["message"]["content"]
            retrieved_chunks = response.retrieved_chunks or []
            schema_objects = sorted(
                {
                    obj
                    for chunk in retrieved_chunks
                    for obj in ((chunk.get("metadata") or {}).get("trusted_schema_objects") or [])
                }
            )
            record = {
                **plan,
                "category": case.get("category"),
                "benchmark_answer": case.get("answer"),
                "tags": case.get("tags") or [],
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
                    any(obj == "UNKNOWN" for obj in response.audit.get("normalization_tags", {}).get("unconfirmed", []))
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
        except Exception as exc:
            with error_path.open("a", encoding="utf-8") as handle:
                handle.write(f"case_id={case_id}\n")
                handle.write(traceback.format_exc())
                handle.write("\n")
            record = {
                **plan,
                "category": case.get("category"),
                "benchmark_answer": case.get("answer"),
                "tags": case.get("tags") or [],
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
                "verifier_status": "RUNTIME_ERROR",
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
                "correction_present": False,
                "response_time_sec": round(time.perf_counter() - started_at, 2),
                "output": FAIL_CLOSED_MESSAGE,
                "runtime_error": str(exc),
                "exact_module_top_hit": False,
                "same_family_bleed_through": False,
            }

        record["quality_tag"] = quality_tag(record)
        record["scoring_outcome"] = score_case(record)
        results.append(record)
        persist_partial(results, probe_ids, summary_path, partial_path, args.label, args.max_tokens)

    summary = persist_partial(results, probe_ids, summary_path, partial_path, args.label, args.max_tokens)
    with results_path.open("w", encoding="utf-8") as handle:
        for row in sorted(results, key=lambda row: probe_ids.index(row["id"])):
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    delta = {
        "sample_size": len(results),
        "trusted_outcome_rate_pct_before": baseline_summary.get("trusted_outcome_rate_pct"),
        "trusted_outcome_rate_pct_after": summary.get("trusted_outcome_rate_pct"),
        "trusted_outcome_delta": round(float(summary.get("trusted_outcome_rate_pct") or 0.0) - float(baseline_summary.get("trusted_outcome_rate_pct") or 0.0), 2),
        "refusal_correctness_pct_before": baseline_summary.get("refusal_correctness_pct"),
        "refusal_correctness_pct_after": summary.get("refusal_correctness_pct"),
        "refusal_correctness_delta": round(float(summary.get("refusal_correctness_pct") or 0.0) - float(baseline_summary.get("refusal_correctness_pct") or 0.0), 2),
        "wrong_module_answer_rate_pct_before": baseline_summary.get("wrong_module_answer_rate_pct"),
        "wrong_module_answer_rate_pct_after": summary.get("wrong_module_answer_rate_pct"),
        "wrong_module_answer_rate_delta": round(float(summary.get("wrong_module_answer_rate_pct") or 0.0) - float(baseline_summary.get("wrong_module_answer_rate_pct") or 0.0), 2),
        "troubleshooting_success_pct_before": ((baseline_summary.get("results_by_segment") or {}).get("troubleshooting") or {}).get("success_rate_pct"),
        "troubleshooting_success_pct_after": ((summary.get("results_by_segment") or {}).get("troubleshooting") or {}).get("success_rate_pct"),
        "troubleshooting_success_delta": round(
            float((((summary.get("results_by_segment") or {}).get("troubleshooting") or {}).get("success_rate_pct")) or 0.0)
            - float((((baseline_summary.get("results_by_segment") or {}).get("troubleshooting") or {}).get("success_rate_pct")) or 0.0),
            2,
        ),
        "verifier_pass_pct_before": baseline_summary.get("verifier_pass_pct"),
        "verifier_pass_pct_after": summary.get("verifier_pass_pct"),
        "verifier_pass_delta": round(float(summary.get("verifier_pass_pct") or 0.0) - float(baseline_summary.get("verifier_pass_pct") or 0.0), 2),
    }
    write_json(delta_path, delta)
    print(json.dumps({"summary": summary, "delta": delta}, indent=2, ensure_ascii=True))
    return {"summary": summary, "delta": delta}


if __name__ == "__main__":
    asyncio.run(run_probe(parse_args()))
