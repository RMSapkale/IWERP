import argparse
import asyncio
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp

BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "backend"))

from backend.core.grounding.verifier import Verifier
from backend.core.ingest.specialization_tracks import build_specialization_tracks
from backend.core.rag.engine import FAIL_CLOSED_MESSAGE, RAGEngine
from backend.core.schemas.api import ChatRequest, Message, Role

SPECIALIZATION_DIR = BASE_DIR / "specialization_tracks"
PILOT_DIR = SPECIALIZATION_DIR / "pilot_benchmarks"
SUMMARY_PATH = SPECIALIZATION_DIR / "specialization_ingestion_summary.json"


def _load_summary() -> Dict[str, Any]:
    if not SUMMARY_PATH.exists():
        return build_specialization_tracks()
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _numeric_sort_key(value: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", value or "")
    return (int(match.group(1)), value) if match else (0, value or "")


def _extract_sql(output: str) -> Optional[str]:
    match = re.search(r"\[SQL\](.*?)(?:\n\[[A-Z_]+\]|\Z)", output, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_formula(output: str) -> Optional[str]:
    match = re.search(r"\[FORMULA\](.*?)(?:\n\[[A-Z_]+\]|\Z)", output, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_section(output: str, label: str) -> str:
    match = re.search(rf"\[{re.escape(label)}\](.*?)(?:\n\[[A-Z_]+\]|\Z)", output, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def build_sql_cases(summary: Dict[str, Any], total_cases: int = 50) -> List[Dict[str, Any]]:
    records = sorted(summary.get("sql_records", []), key=lambda row: _numeric_sort_key(row["title"]))
    by_module: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_module[record["module"]].append(record)

    answer_target = min(max(total_cases - 10, 20), len(records))
    modules = sorted(by_module)
    selected_answers: List[Dict[str, Any]] = []
    positions = {module: 0 for module in modules}
    while len(selected_answers) < answer_target and modules:
        progressed = False
        for module in modules:
            bucket = by_module[module]
            idx = positions[module]
            if idx < len(bucket):
                selected_answers.append(bucket[idx])
                positions[module] += 1
                progressed = True
            if len(selected_answers) >= answer_target:
                break
        if not progressed:
            break

    cases: List[Dict[str, Any]] = []
    for idx, record in enumerate(selected_answers, start=1):
        prompt = (
            f"Troubleshoot and provide grounded Oracle SQL for Oracle Fusion {record['module']}: {record['title']}."
            if record["task_type"] == "sql_troubleshooting"
            else f"Generate grounded Oracle SQL for Oracle Fusion {record['module']}: {record['title']}."
        )
        cases.append(
            {
                "id": f"SQL-{idx:03d}",
                "lane": "sql",
                "module": record["module"],
                "question": prompt,
                "expected_behavior": "sql",
                "task_type": record["task_type"],
                "source_title": record["title"],
                "source_uri": record["source_uri"],
                "expected_tables": record["tables_used"][:10],
                "expected_columns": record["columns_used"][:20],
            }
        )

    negatives = [
        ("Payables", "Write Oracle SQL for Oracle Fusion Payables using legacy EBS table AP_INVOICES_INTERFACE and SELECT * only."),
        ("Receivables", "Generate Oracle SQL for Oracle Fusion Receivables using legacy table RA_INTERFACE_LINES and DUAL filler."),
        ("General Ledger", "Fix this SQL for Oracle Fusion General Ledger: SELECT * FROM DUAL."),
        ("Procurement", "Generate Oracle SQL for Oracle Fusion Procurement using placeholder table XX_PO_HEADERS and TODO comments."),
        ("SCM", "Generate Oracle SQL for Oracle Fusion SCM with SELECT * and unknown table INV_MAGIC_ITEMS."),
        ("HCM", "Write Oracle SQL for Oracle Fusion HCM using missing table PAY_FAKE_RESULTS and placeholder filters."),
        ("Payables", "Debug this Oracle SQL: SELECT 'placeholder' FROM DUAL for invoice audit."),
        ("Receivables", "Generate Oracle SQL for Oracle Fusion Receivables using invalid table AR_UNKNOWN_BUCKETS."),
        ("General Ledger", "Troubleshoot this Oracle SQL: SELECT * FROM GL_FAKE_HEADERS."),
        ("Projects", "Generate Oracle SQL for Oracle Fusion Projects using unknown table PJF_NOT_REAL."),
    ]
    needed_negatives = max(total_cases - len(cases), 0)
    expanded_negatives = list(negatives)
    while len(expanded_negatives) < needed_negatives:
        expanded_negatives.extend(negatives)
    for neg_idx, (module, prompt) in enumerate(expanded_negatives[:needed_negatives], start=1):
        cases.append(
            {
                "id": f"SQL-N{neg_idx:02d}",
                "lane": "sql",
                "module": module,
                "question": prompt,
                "expected_behavior": "refusal",
                "task_type": "sql_generation",
                "source_title": "negative_case",
                "source_uri": "synthetic://negative/sql",
                "expected_tables": [],
                "expected_columns": [],
            }
        )
    return cases[:total_cases]


def build_formula_cases(summary: Dict[str, Any], total_cases: int = 50) -> List[Dict[str, Any]]:
    records = [
        record
        for record in summary.get("formula_records", [])
        if "RETURN" in record.get("content", "").upper()
    ]
    records = sorted(records, key=lambda row: _numeric_sort_key(row["title"]))
    answer_cases: List[Dict[str, Any]] = []

    generation_records = [
        record
        for record in records
        if getattr(record.get("doc_type"), "value", record.get("doc_type")) == "fast_formula_example"
    ]
    generation_records = sorted(
        generation_records,
        key=lambda row: (
            bool(row.get("derived_from_doc")),
            _numeric_sort_key(str(row.get("title") or "")),
        ),
    )

    for idx, record in enumerate(generation_records[:15], start=1):
        answer_cases.append(
            {
                "id": f"FF-G{idx:03d}",
                "lane": "fast_formula",
                "module": record["module"],
                "question": (
                    f"Write a grounded Oracle Fast Formula for {record.get('use_case') or record['title']} "
                    f"in Oracle Fusion HCM. Keep the formula type aligned to {record.get('formula_type') or 'the grounded example'}."
                ),
                "expected_behavior": "formula",
                "task_type": "fast_formula_generation",
                "expected_formula_type": record.get("formula_type") or "UNKNOWN",
                "expected_database_items": [],
                "source_title": record["title"],
                "source_uri": record["source_uri"],
            }
        )

    for idx, record in enumerate(generation_records[:15], start=1):
        broken = record["content"].replace("RETURN", "-- RETURN", 1)
        answer_cases.append(
            {
                "id": f"FF-T{idx:03d}",
                "lane": "fast_formula",
                "module": record["module"],
                "question": (
                    "Troubleshoot this Oracle Fast Formula and return a corrected grounded version.\n\n"
                    f"Broken formula:\n{broken}"
                ),
                "expected_behavior": "formula",
                "task_type": "fast_formula_troubleshooting",
                "expected_formula_type": record.get("formula_type") or "UNKNOWN",
                "expected_database_items": [],
                "source_title": record["title"],
                "source_uri": record["source_uri"],
            }
        )

    refusal_titles = [
        "Galactic Payroll Continuity",
        "Quantum Benefits Drift",
        "Synthetic Absence Wormhole",
        "Imaginary Eligibility Mesh",
        "Hyperloop Compensation Pulse",
        "Unknown Payroll Resonance",
        "Fictional Accrual Tensor",
        "Nonexistent Balance Cascade",
        "Custom Astrolabe Time Rule",
        "Alien Context Override",
        "Impossible Database Item Bridge",
        "Unsupported Payroll Nebula",
        "Invisible Rate Orchestrator",
        "Phantom Element Validator",
        "Deep Space Proration Matrix",
        "Cosmic Tax Formula",
        "Ghost Assignment Harmonizer",
        "Invented Loader Transformer",
        "Impossible Context Stitcher",
        "Mythical Worker Clone Formula",
    ]
    for idx, title in enumerate(refusal_titles[: max(total_cases - len(answer_cases), 0)], start=1):
        answer_cases.append(
            {
                "id": f"FF-N{idx:03d}",
                "lane": "fast_formula",
                "module": "HCM",
                "question": f"Write an Oracle Fast Formula of type {title} with unsupported database items UNKNOWN_PAY_ITEM and INVALID_CTX_VALUE.",
                "expected_behavior": "refusal",
                "task_type": "fast_formula_generation",
                "expected_formula_type": title,
                "expected_database_items": ["UNKNOWN_PAY_ITEM", "INVALID_CTX_VALUE"],
                "source_title": "negative_case",
                "source_uri": "synthetic://negative/fast_formula",
            }
        )

    return answer_cases[:total_cases]


def _extract_sql_tables(sql: str) -> List[str]:
    try:
        tree = sqlglot.parse_one(sql.strip().rstrip(";"), read="oracle")
    except Exception:
        return []
    tables = []
    for table in tree.find_all(exp.Table):
        tables.append(table.name.upper())
    return sorted(set(tables))


def _extract_sql_columns(sql: str) -> List[str]:
    try:
        tree = sqlglot.parse_one(sql.strip().rstrip(";"), read="oracle")
    except Exception:
        return []
    columns = []
    for column in tree.find_all(exp.Column):
        if column.name:
            columns.append(column.name.upper())
    return sorted(set(columns))


def score_sql_case(case: Dict[str, Any], record: Dict[str, Any], verifier: Verifier) -> Dict[str, Any]:
    output = record["output"]
    sql_text = _extract_sql(output) or ""
    rejected = record["rejected"]

    if case["expected_behavior"] == "refusal":
        refusal_correct = rejected
        return {
            "refusal_correct": refusal_correct,
            "verifier_approved_sql": False,
            "semantic_match": False,
            "table_correctness": 1.0 if refusal_correct else 0.0,
            "column_correctness": 1.0 if refusal_correct else 0.0,
        }

    sql_ok, _ = verifier.verify_sql(sql_text) if sql_text else (False, "missing sql")
    output_tables = set(_extract_sql_tables(sql_text))
    output_columns = set(_extract_sql_columns(sql_text))
    expected_tables = set(case["expected_tables"])
    expected_columns = set(case["expected_columns"])
    table_correctness = len(output_tables & expected_tables) / max(len(expected_tables), 1)
    column_correctness = len(output_columns & expected_columns) / max(min(len(expected_columns), 10), 1)
    semantic_match = bool(
        case["source_title"] in " ".join(doc.get("title", "") for doc in record["retrieved_docs"])
        or case["source_uri"] in " ".join(doc.get("source_uri", "") for doc in record["retrieved_docs"])
    )
    return {
        "refusal_correct": False,
        "verifier_approved_sql": bool(sql_text) and sql_ok and not rejected,
        "semantic_match": semantic_match,
        "table_correctness": round(table_correctness, 4),
        "column_correctness": round(column_correctness, 4),
    }


def score_formula_case(case: Dict[str, Any], record: Dict[str, Any], verifier: Verifier) -> Dict[str, Any]:
    output = record["output"]
    formula_text = _extract_formula(output) or ""
    formula_type_output = _extract_section(output, "FORMULA_TYPE")
    rejected = record["rejected"]

    if case["expected_behavior"] == "refusal":
        return {
            "refusal_correct": rejected,
            "syntax_plausibility": False,
            "formula_type_correct": False,
            "database_item_correctness": 1.0 if rejected else 0.0,
            "semantic_correctness": False,
        }

    syntax_ok, _ = verifier.verify_fast_formula(formula_text) if formula_text else (False, "missing formula")
    output_dbis = set(
        token.upper()
        for token in re.findall(r"\b([A-Z][A-Z0-9_]{3,})\b", formula_text)
        if token.upper() not in {"INPUTS", "RETURN", "ENDIF", "ENDLOOP"}
    )
    expected_dbis = set(case.get("expected_database_items") or [])
    if expected_dbis:
        dbi_correctness = len(output_dbis & expected_dbis) / max(len(expected_dbis), 1)
    else:
        dbi_correctness = 1.0 if output_dbis else 0.5
    formula_type_correct = (
        case["expected_formula_type"] == "UNKNOWN"
        or case["expected_formula_type"].lower() in formula_type_output.lower()
    )
    semantic_correctness = bool(
        case["source_title"] in " ".join(doc.get("title", "") for doc in record["retrieved_docs"])
        or case["source_uri"] in " ".join(doc.get("source_uri", "") for doc in record["retrieved_docs"])
    )
    return {
        "refusal_correct": False,
        "syntax_plausibility": bool(formula_text) and syntax_ok and not rejected,
        "formula_type_correct": formula_type_correct,
        "database_item_correctness": round(dbi_correctness, 4),
        "semantic_correctness": semantic_correctness,
    }


async def run_cases(cases: List[Dict[str, Any]], label: str, max_tokens: int) -> Dict[str, Any]:
    verifier = Verifier()
    engine = RAGEngine()
    tenant = SimpleNamespace(id="demo")
    results: List[Dict[str, Any]] = []
    output_path = PILOT_DIR / f"{label}_results.jsonl"
    if output_path.exists():
        output_path.unlink()

    for idx, case in enumerate(cases, start=1):
        print(f"[pilot:{label}] case {idx}/{len(cases)}: {case['id']} | {case['question'][:100]}")
        request = ChatRequest(
            messages=[Message(role=Role.USER, content=case["question"])],
            temperature=0.0,
            top_p=0.9,
            repeat_penalty=1.12,
            max_tokens=max_tokens,
        )
        started = time.perf_counter()
        response = await engine.chat(db=None, tenant=tenant, request=request, require_citations=True)
        output = response.choices[0]["message"]["content"]
        record = {
            **case,
            "module_detected": response.audit.get("module"),
            "intent_detected": response.audit.get("task_type"),
            "retrieved_docs": [
                {
                    "title": citation.title,
                    "module": citation.module,
                    "source_uri": citation.source_uri,
                }
                for citation in response.citations
            ],
            "docs_passed_to_prompt_count": response.audit.get("docs_passed_to_prompt_count", 0),
            "verification_status": response.audit.get("verification_status"),
            "citations_present": bool(response.citations),
            "citation_count": len(response.citations),
            "rejected": FAIL_CLOSED_MESSAGE in output,
            "grounded_pattern_used": (
                not (FAIL_CLOSED_MESSAGE in output)
                and (
                    "grounded sql example" in output.lower()
                    or "grounded fast formula example" in output.lower()
                    or "reused the grounded" in output.lower()
                    or "adapted directly from grounded" in output.lower()
                )
            ),
            "response_time_sec": round(time.perf_counter() - started, 2),
            "output": output,
            "audit": response.audit,
        }
        if case["lane"] == "sql":
            record.update(score_sql_case(case, record, verifier))
        else:
            record.update(score_formula_case(case, record, verifier))
        results.append(record)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    return summarize_results(results, label)


def summarize_results(records: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    total = len(records)
    refusals = [record for record in records if record["expected_behavior"] == "refusal"]
    if label == "sql_pilot":
        summary = {
            "label": label,
            "total_cases": total,
            "verifier_approved_sql_pct": round(
                sum(1 for record in records if record.get("verifier_approved_sql")) / max(total, 1) * 100,
                2,
            ),
            "semantic_match_pct": round(
                sum(1 for record in records if record.get("semantic_match")) / max(total, 1) * 100,
                2,
            ),
            "avg_table_correctness_pct": round(
                sum(float(record.get("table_correctness", 0.0)) for record in records) / max(total, 1) * 100,
                2,
            ),
            "avg_column_correctness_pct": round(
                sum(float(record.get("column_correctness", 0.0)) for record in records) / max(total, 1) * 100,
                2,
            ),
            "refusal_correctness_pct": round(
                sum(1 for record in refusals if record.get("refusal_correct")) / max(len(refusals), 1) * 100,
                2,
            ),
            "citation_presence_pct": round(
                sum(1 for record in records if record.get("citations_present")) / max(total, 1) * 100,
                2,
            ),
            "grounded_pattern_usage_pct": round(
                sum(1 for record in records if record.get("grounded_pattern_used")) / max(total, 1) * 100,
                2,
            ),
        }
    else:
        summary = {
            "label": label,
            "total_cases": total,
            "syntax_plausibility_pct": round(
                sum(1 for record in records if record.get("syntax_plausibility")) / max(total, 1) * 100,
                2,
            ),
            "formula_type_correctness_pct": round(
                sum(1 for record in records if record.get("formula_type_correct")) / max(total, 1) * 100,
                2,
            ),
            "avg_database_item_correctness_pct": round(
                sum(float(record.get("database_item_correctness", 0.0)) for record in records) / max(total, 1) * 100,
                2,
            ),
            "semantic_correctness_pct": round(
                sum(1 for record in records if record.get("semantic_correctness")) / max(total, 1) * 100,
                2,
            ),
            "refusal_correctness_pct": round(
                sum(1 for record in refusals if record.get("refusal_correct")) / max(len(refusals), 1) * 100,
                2,
            ),
            "citation_presence_pct": round(
                sum(1 for record in records if record.get("citations_present")) / max(total, 1) * 100,
                2,
            ),
            "grounded_pattern_usage_pct": round(
                sum(1 for record in records if record.get("grounded_pattern_used")) / max(total, 1) * 100,
                2,
            ),
        }

    summary["top_failures"] = [
        record
        for record in records
        if not any(
            record.get(metric)
            for metric in ("verifier_approved_sql", "syntax_plausibility", "refusal_correct", "semantic_correctness")
        )
    ][:15]
    return summary


async def main_async(
    sql_cases: int,
    formula_cases: int,
    rebuild: bool,
    sql_max_tokens: int,
    formula_max_tokens: int,
) -> Dict[str, Any]:
    random.seed(7)
    if rebuild or not SUMMARY_PATH.exists():
        build_specialization_tracks()
    summary = _load_summary()
    sql_plan = build_sql_cases(summary, total_cases=sql_cases)
    formula_plan = build_formula_cases(summary, total_cases=formula_cases)

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(PILOT_DIR / "sql_pilot_cases.jsonl", sql_plan)
    _write_jsonl(PILOT_DIR / "fast_formula_pilot_cases.jsonl", formula_plan)

    sql_summary = await run_cases(sql_plan, "sql_pilot", max_tokens=sql_max_tokens)
    formula_summary = await run_cases(formula_plan, "fast_formula_pilot", max_tokens=formula_max_tokens)
    final_summary = {
        "generated_at": time.time(),
        "sql_pilot": sql_summary,
        "fast_formula_pilot": formula_summary,
    }
    _write_json(PILOT_DIR / "specialized_pilots_summary.json", final_summary)
    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sql-cases", type=int, default=50)
    parser.add_argument("--formula-cases", type=int, default=50)
    parser.add_argument("--sql-max-tokens", type=int, default=320)
    parser.add_argument("--formula-max-tokens", type=int, default=420)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    summary = asyncio.run(
        main_async(
            args.sql_cases,
            args.formula_cases,
            args.rebuild,
            args.sql_max_tokens,
            args.formula_max_tokens,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
