#!/usr/bin/env python3
"""
Run targeted cross-module SQL validation for the hardened Oracle Fusion SQL lane.

This validator exercises the SQL request-shape parser, report-family support registry,
grounded SQL builders, and strict verifier in-process. It does not require a live
HTTP server and is intended to provide durable regression evidence for the SQL lane
itself when local network binding is unavailable.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import sqlglot
from sqlglot import exp

import sys


SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parent.parent
PROJECT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from core.rag.engine import RAGEngine
from core.schemas.router import FusionModule, ModuleFamily


@dataclass
class SqlPromptCase:
    case_id: str
    family: str
    prompt: str
    expected_generated: bool = True


PRIMARY_PROMPTS: List[SqlPromptCase] = [
    SqlPromptCase(
        "AP-001",
        "AP Invoice Details",
        "Create an Oracle Fusion SQL query to extract AP invoice details with Business Unit, Invoice Number, Invoice Date, Supplier Name, Supplier Number, Invoice Amount, Invoice Currency, Invoice Status, Accounting Date. Filter Invoice Date between :P_FROM_DATE and :P_TO_DATE and order by Invoice Date.",
    ),
    SqlPromptCase(
        "AP-002",
        "AP Invoice Distribution & GL Account",
        "Create an Oracle Fusion SQL query to extract AP invoice distribution details with Invoice Number, Supplier Name, Distribution Line Number, Distribution Amount, Natural Account Segment, Cost Center, Liability Account. Display only invoices that are Validated and Accounted.",
    ),
    SqlPromptCase(
        "AP-003",
        "AP Payments Report",
        "Create an Oracle Fusion SQL query to extract AP payments report with Supplier Name, Invoice Number, Payment Number, Payment Date, Payment Method, Paid Amount, Bank Account Name. Filter Payment Date = :P_PAYMENT_DATE.",
    ),
    SqlPromptCase(
        "AR-001",
        "AR Transaction Report",
        "Create an Oracle Fusion SQL query to extract AR transaction report with Customer Name, Transaction Number, Transaction Date, Business Unit, Amount Due Original, Remaining Amount, Due Date, Payment Status. Show only open transactions as of :P_AS_OF_DATE and order by Transaction Date.",
    ),
    SqlPromptCase(
        "AR-002",
        "AR Receipts & Applications",
        "Create an Oracle Fusion SQL query to extract AR receipts and applications with Customer Name, Receipt Number, Receipt Date, Receipt Amount, Transaction Number, Applied Amount, Receipt Status. Show applied receipts and order by Receipt Date.",
    ),
    SqlPromptCase(
        "AR-003",
        "AR Aging Report",
        "Create an Oracle Fusion SQL query to extract AR aging report with Customer Name, Transaction Number, Due Date, Remaining Amount, Aging Bucket as of :P_AS_OF_DATE and order by Customer Name.",
    ),
    SqlPromptCase(
        "GL-001",
        "GL Account Balances",
        "Create an Oracle Fusion SQL query to extract GL account balances with Ledger Name, Period Name, Account Combination, Period Net Dr, Period Net Cr, Ending Balance. Filter Ledger Name = :P_LEDGER_NAME and Period Name = :P_PERIOD_NAME and order by Account Combination.",
    ),
    SqlPromptCase(
        "GL-002",
        "GL Journal Details",
        "Create an Oracle Fusion SQL query to extract GL journal details with Ledger Name, Journal Name, Period Name, Journal Source, Journal Category, Journal Status, Journal Line Number, Account Combination, Debit Amount, Credit Amount. Show posted journals for Ledger Name = :P_LEDGER_NAME and Period Name = :P_PERIOD_NAME and order by Journal Name.",
    ),
    SqlPromptCase(
        "PO-001",
        "Purchase Order Details",
        "Create an Oracle Fusion SQL query to extract Purchase Order details with PO Number, PO Date, PO Status, Supplier Name, Supplier Number, Site Code, Line Number, Item Description, Ordered Quantity, Unit Price and order by PO Number.",
    ),
    SqlPromptCase(
        "PO-002",
        "PO Receiving & Invoicing Match",
        "Create an Oracle Fusion SQL query to extract PO receiving and invoicing match details with PO Number, Supplier Name, Line Number, Ordered Quantity, Received Quantity, Billed Quantity, Receipt Date and order by PO Number.",
    ),
]

VARIANT_PROMPTS: List[SqlPromptCase] = [
    SqlPromptCase(
        "AP-V001",
        "AP Variant",
        "Generate Oracle Fusion Payables SQL for an invoice detail report. Include: - Business Unit - Supplier Name - Supplier Number - Invoice Number - Invoice Date - Invoice Amount - Invoice Currency - Invoice Status - Accounting Date. Use Invoice Date BETWEEN :P_FROM_DATE AND :P_TO_DATE. Sort by Invoice Date.",
    ),
    SqlPromptCase(
        "AR-V001",
        "AR Variant",
        "Generate Oracle Fusion Receivables aging SQL. Include Customer Name, Transaction Number, Remaining Amount, Due Date, Aging Bucket. Use :P_AS_OF_DATE as the as-of date and order by Customer Name.",
    ),
    SqlPromptCase(
        "GL-V001",
        "GL Variant",
        "Write Oracle Fusion General Ledger SQL for posted journal detail reporting with Ledger Name, Journal Name, Period Name, Journal Source, Journal Category, Journal Status, Journal Line Number, Account Combination, Debit Amount, Credit Amount. Filter Ledger Name = :P_LEDGER_NAME and Period Name = :P_PERIOD_NAME. Sort by Journal Name.",
    ),
    SqlPromptCase(
        "PO-V001",
        "PO Variant",
        "Write Oracle Fusion Purchasing SQL for a purchase order detail report with Supplier Name, Supplier Number, Site Code, PO Number, PO Date, PO Status, Line Number, Item Description, Ordered Quantity, Unit Price. Order by PO Number.",
    ),
]

REFUSAL_PROMPTS: List[SqlPromptCase] = [
    SqlPromptCase(
        "AP-N001",
        "AP Unsupported Field",
        "Create an Oracle Fusion SQL query to extract AP payments report with Supplier Name, Invoice Number, Payment Number, Payment Date, Payment Method, Paid Amount, Bank Account Name, IBAN Number. Filter Payment Date = :P_PAYMENT_DATE.",
        expected_generated=False,
    ),
    SqlPromptCase(
        "AR-N001",
        "AR Unsupported Calculation",
        "Create an Oracle Fusion SQL query to extract AR aging report with Customer Name, Transaction Number, Due Date, Remaining Amount, Aging Bucket, and a custom 121-150 day bucket as of :P_AS_OF_DATE.",
        expected_generated=False,
    ),
    SqlPromptCase(
        "PO-N001",
        "PO Unsupported Field",
        "Create an Oracle Fusion SQL query to extract PO receiving and invoicing match details with PO Number, Supplier Name, Line Number, Ordered Quantity, Received Quantity, Billed Quantity, Receipt Date, Invoiced Amount and order by PO Number.",
        expected_generated=False,
    ),
]


def _module_family_for_case(module: FusionModule) -> ModuleFamily:
    if module in {FusionModule.PAYABLES, FusionModule.RECEIVABLES, FusionModule.GENERAL_LEDGER}:
        return ModuleFamily.FINANCIALS
    if module == FusionModule.PROCUREMENT:
        return ModuleFamily.PROCUREMENT
    return ModuleFamily.UNKNOWN


def _coverage_breakdown(engine: RAGEngine, sql_text: str, request_shape: Dict[str, Any]) -> Dict[str, bool]:
    try:
        parsed = sqlglot.parse_one((sql_text or "").strip().strip(";"), read="oracle")
    except Exception:
        return {
            "field_coverage": False,
            "join_coverage": False,
            "filter_coverage": False,
            "ordering_coverage": False,
        }

    projection_columns = {column.name.upper() for expr in list(getattr(parsed, "expressions", []) or []) for column in expr.find_all(exp.Column)}
    projection_aliases = {
        str(getattr(expr, "alias_or_name", "") or "").upper()
        for expr in list(getattr(parsed, "expressions", []) or [])
        if str(getattr(expr, "alias_or_name", "") or "").strip()
    }

    where_clause = parsed.args.get("where")
    where_columns = {column.name.upper() for column in where_clause.find_all(exp.Column)} if where_clause is not None else set()
    where_text = where_clause.sql(dialect="oracle").upper() if where_clause is not None else ""

    order_clause = parsed.args.get("order")
    order_columns = {column.name.upper() for column in order_clause.find_all(exp.Column)} if order_clause is not None else set()
    order_text = order_clause.sql(dialect="oracle").upper() if order_clause is not None else ""

    field_coverage = True
    for field in request_shape.get("required_fields") or []:
        candidate_columns = {str(item).upper() for item in (field.get("columns") or [])}
        candidate_aliases = {str(item).upper() for item in (field.get("aliases") or [])}
        if not (candidate_columns & projection_columns or candidate_aliases & projection_aliases):
            field_coverage = False
            break

    filter_coverage = True
    for filter_spec in request_shape.get("required_filters") or []:
        candidate_columns = {str(item).upper() for item in (filter_spec.get("columns") or [])}
        candidate_values = {str(item).upper() for item in (filter_spec.get("values") or [])}
        if candidate_columns & where_columns and (not candidate_values or all(value in where_text for value in candidate_values)):
            continue
        filter_coverage = False
        break

    ordering_coverage = True
    for ordering in request_shape.get("required_ordering") or []:
        candidate_columns = {str(item).upper() for item in (ordering.get("columns") or [])}
        candidate_aliases = {str(item).upper() for item in (ordering.get("aliases") or [])}
        if candidate_columns & order_columns or any(alias in order_text for alias in candidate_aliases):
            continue
        ordering_coverage = False
        break

    join_coverage = True
    required_pairs = list(request_shape.get("required_join_pairs") or [])
    if required_pairs:
        alias_map = {}
        for table in parsed.find_all(exp.Table):
            alias = str(table.alias_or_name or "").upper()
            table_name = engine.verifier.registry.resolve_object_name(str(table.name or "").upper()) or str(table.name or "").upper()
            if alias:
                alias_map[alias] = table_name.upper()

        adjacency = set()
        for join in parsed.find_all(exp.Join):
            on_clause = join.args.get("on")
            if on_clause is None:
                continue
            for eq in on_clause.find_all(exp.EQ):
                left = eq.left
                right = eq.right
                if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                    continue
                left_table = alias_map.get(str(left.table or "").upper())
                right_table = alias_map.get(str(right.table or "").upper())
                if not left_table or not right_table or left_table == right_table:
                    continue
                adjacency.add((left_table, right_table))
                adjacency.add((right_table, left_table))
        for left, right in required_pairs:
            if (str(left).upper(), str(right).upper()) not in adjacency:
                join_coverage = False
                break

    return {
        "field_coverage": field_coverage,
        "join_coverage": join_coverage,
        "filter_coverage": filter_coverage,
        "ordering_coverage": ordering_coverage,
    }


def _top_catalog_hits(engine: RAGEngine, route_info: Any, request_shape: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    catalog = engine._load_specialization_catalog()
    ranked = []
    for record in catalog.get("sql_records") or []:
        if not engine._specialized_module_compatible(route_info, str(record.get("module") or "")):
            continue
        score = engine._score_sql_pattern_for_request(str(record.get("content") or ""), record, request_shape)
        if score <= -5.0:
            continue
        ranked.append(
            {
                "title": str(record.get("title") or record.get("source_file") or ""),
                "module": str(record.get("module") or ""),
                "score": round(score, 4),
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:limit]


def _evaluate_case(engine: RAGEngine, case: SqlPromptCase) -> Dict[str, Any]:
    route_info = engine.router.route(case.prompt)
    request_shape = engine._parse_sql_request_shape(case.prompt, route_info)
    sql_text, refusal_reason = engine._build_supported_sql_report(request_shape, case.prompt)

    module_hint = engine._infer_sql_module_hint(case.prompt)
    module_alignment_target = engine._sql_alignment_target(route_info, module_hint)
    if str(request_shape.get("report_family") or "").startswith("procurement_"):
        module_alignment_target = ModuleFamily.UNKNOWN

    selected_pattern_hits = _top_catalog_hits(engine, route_info, request_shape)
    selected_pattern_example = selected_pattern_hits[0]["title"] if selected_pattern_hits else ""

    result: Dict[str, Any] = {
        "case_id": case.case_id,
        "family": case.family,
        "prompt": case.prompt,
        "expected_generated": case.expected_generated,
        "route_selected": getattr(route_info.task_type, "value", route_info.task_type),
        "module_selected": getattr(route_info.module, "value", route_info.module),
        "report_family": request_shape.get("report_family") or "",
        "schema_metadata_hits": request_shape.get("required_tables") or [],
        "selected_pattern_example": selected_pattern_example,
        "catalog_hits": selected_pattern_hits,
        "generated": bool(sql_text),
        "sql": sql_text or "",
        "refusal_reason": refusal_reason or "",
        "verifier_pass": False,
        "verifier_reason": "",
        "required_field_coverage": False,
        "join_coverage": False,
        "filter_coverage": False,
        "ordering_coverage": False,
        "hardcoding_pass": False,
        "full_shape_pass": False,
        "pass": False,
        "pass_fail_reason": "",
    }

    if not sql_text:
        result["hardcoding_pass"] = True
        if case.expected_generated:
            result["pass_fail_reason"] = refusal_reason or "unexpected_refusal"
            return result
        specific = bool(refusal_reason) and "Missing grounded support" in refusal_reason
        result["pass"] = specific
        result["pass_fail_reason"] = "correct_refusal" if specific else "nonspecific_refusal"
        return result

    verifier_ok, verifier_reason = engine._verify_sql_candidate(
        sql_text,
        module_alignment_target=module_alignment_target,
        request_shape=request_shape,
    )
    style_ok, style_reason = engine.verifier.verify_sql_style(sql_text)
    coverage = _coverage_breakdown(engine, sql_text, request_shape)

    result["verifier_pass"] = verifier_ok
    result["verifier_reason"] = verifier_reason or style_reason or ""
    result["required_field_coverage"] = coverage["field_coverage"]
    result["join_coverage"] = coverage["join_coverage"]
    result["filter_coverage"] = coverage["filter_coverage"]
    result["ordering_coverage"] = coverage["ordering_coverage"]
    result["hardcoding_pass"] = style_ok
    result["full_shape_pass"] = all(coverage.values())
    result["pass"] = case.expected_generated and verifier_ok and result["full_shape_pass"] and style_ok
    result["pass_fail_reason"] = "ok" if result["pass"] else (verifier_reason or style_reason or "verification_failed")
    return result


def run_validation(output_dir: Path) -> Dict[str, Any]:
    engine = RAGEngine()
    cases = PRIMARY_PROMPTS + VARIANT_PROMPTS + REFUSAL_PROMPTS
    results = [_evaluate_case(engine, case) for case in cases]

    total_generated = sum(1 for item in results if item["generated"])
    total_correct_refusals = sum(
        1 for item in results if (not item["generated"]) and item["pass"] and not item["expected_generated"]
    )
    total_failures = sum(1 for item in results if not item["pass"])

    generated_results = [item for item in results if item["generated"]]
    no_hardcoding_compliance = (
        round((sum(1 for item in generated_results if item["hardcoding_pass"]) / len(generated_results)) * 100.0, 2)
        if generated_results
        else 100.0
    )
    full_shape_compliance = (
        round((sum(1 for item in generated_results if item["full_shape_pass"]) / len(generated_results)) * 100.0, 2)
        if generated_results
        else 100.0
    )

    summary = {
        "mode": "engine_direct_sql_lane_validation",
        "total_prompts_tested": len(cases),
        "primary_prompts_tested": len(PRIMARY_PROMPTS),
        "variant_prompts_tested": len(VARIANT_PROMPTS),
        "refusal_prompts_tested": len(REFUSAL_PROMPTS),
        "total_generated": total_generated,
        "total_correct_refusals": total_correct_refusals,
        "total_failures": total_failures,
        "no_hardcoding_compliance_pct": no_hardcoding_compliance,
        "full_shape_compliance_pct": full_shape_compliance,
        "failure_reason_counts": dict(Counter(item["pass_fail_reason"] for item in results if not item["pass"])),
        "primary_prompt_pass_rate_pct": round((sum(1 for item in results[: len(PRIMARY_PROMPTS)] if item["pass"]) / len(PRIMARY_PROMPTS)) * 100.0, 2),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "cross_module_sql_showcase_validation_results.json"
    summary_path = output_dir / "cross_module_sql_showcase_summary.json"
    results_path.write_text(json.dumps(results, indent=2))
    summary_path.write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "results_path": str(results_path), "summary_path": str(summary_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "sql_validation"),
        help="Directory for validation outputs.",
    )
    args = parser.parse_args()
    outcome = run_validation(Path(args.output_dir))
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
