#!/usr/bin/env python3
"""
Run local V1 UI/API parity validation through the sovereign API path.

Flow:
1) Register a temporary tenant.
2) Call /v1/sovereign/chat/completions for required prompt sets.
3) Save structured results for deployment readiness review.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


FAIL_CLOSED_MARKERS = (
    "insufficient grounded data",
    "cannot generate verified answer",
)


@dataclass
class PromptCase:
    category: str
    prompt: str
    allow_clean_refusal: bool = False
    notes: str | None = None
    required_answer_substrings: tuple[str, ...] = ()


PROMPTS: List[PromptCase] = [
    PromptCase(
        "sql",
        "Create an Oracle Fusion SQL query to extract AP invoice distribution details. Include Invoice Number, Supplier Name, Distribution Line Number, Distribution Amount, Natural Account Segment, Cost Center, Liability Account. Show only validated and accounted invoices.",
        allow_clean_refusal=True,
        notes="Full-shape SQL is preferred; safe refusal is acceptable if grounded joins/fields/filters are insufficient.",
    ),
    PromptCase("sql", "Generate Oracle Fusion SQL to list AP invoices with supplier name and business unit for validated invoices."),
    PromptCase("sql", "Generate Oracle Fusion SQL for posted GL journal lines including natural account and cost center segments."),
    PromptCase("fast_formula", "Create a Fast Formula for sick leave accrual with DEFAULT handling, INPUTS, and RETURN logic."),
    PromptCase(
        "fast_formula",
        "Create a Fast Formula for payroll gratuity eligibility and payout logic with explicit RETURN.",
        allow_clean_refusal=True,
        notes="Guarded refusal is acceptable if gratuity formula grounding is insufficient.",
    ),
    PromptCase("fast_formula", "Troubleshoot Fast Formula compile error: database item not found. Provide symptom, cause, and fix."),
    PromptCase("summary", "what is EPM?", required_answer_substrings=("enterprise performance management",)),
    PromptCase("summary", "what is GL?", required_answer_substrings=("general ledger",)),
    PromptCase("summary", "what is RMCS?", required_answer_substrings=("revenue management cloud service",)),
    PromptCase(
        "summary",
        "what is payroll gratuity?",
        allow_clean_refusal=True,
        notes="Concept answer is preferred; clean refusal is acceptable if exact gratuity grounding is absent.",
        required_answer_substrings=("gratuity",),
    ),
    PromptCase(
        "summary",
        "what is three-way match in Oracle Fusion Procurement?",
        required_answer_substrings=("three-way match", "purchase order", "receipt"),
    ),
    PromptCase("procedure", "how to create custom ESS job?"),
    PromptCase("procedure", "How do you create supplier site setup in Oracle Fusion Payables?"),
    PromptCase("procedure", "How do you create a journal in Oracle Fusion General Ledger?"),
    PromptCase("procedure", "How do you create an expense report in Oracle Fusion Expenses?"),
    PromptCase("procedure", "How do you create a purchase order in Oracle Fusion Purchasing?"),
    PromptCase("troubleshooting", "ESS job submission failed in Oracle Fusion. How should I troubleshoot it?"),
    PromptCase("troubleshooting", "AP invoice validation failed due to account combination error. How to troubleshoot?"),
    PromptCase("troubleshooting", "Receivables accounting transfer failed. What checks should be done?"),
    PromptCase("troubleshooting", "Procurement catalog upload failed with map set validation error. How to resolve?"),
    PromptCase("troubleshooting", "Cash Management bank statement import failed. Provide troubleshooting steps."),
]


def _http_json(method: str, url: str, payload: Dict[str, Any] | None = None, headers: Dict[str, str] | None = None) -> tuple[int, Dict[str, Any]]:
    body = None
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method=method, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8") if exc.fp else ""
        parsed = {}
        if raw:
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {"raw": raw}
        return exc.code, parsed


def _is_refusal(response_text: str, refusal_flag: bool) -> bool:
    if refusal_flag:
        return True
    lowered = (response_text or "").lower()
    return any(marker in lowered for marker in FAIL_CLOSED_MARKERS)


def _infer_result_type(task_type: str, response_text: str, refusal: bool) -> str:
    if refusal:
        return "refusal"
    task = (task_type or "").lower()
    text = (response_text or "").lower()
    if task in {"sql_generation", "sql_troubleshooting"}:
        return "sql"
    if task in {"fast_formula_generation", "fast_formula_troubleshooting"}:
        return "fast_formula"
    if task in {"procedure", "navigation"}:
        return "procedure"
    if task == "troubleshooting":
        return "troubleshooting"
    if task in {"summary", "qa", "general"}:
        return "summary"
    if "sql" in task or "[sql]" in text or "select " in text:
        return "sql"
    if "fast_formula" in task or "[formula]" in text or "default for" in text or "inputs are" in text:
        return "fast_formula"
    if "symptom" in text or "resolution" in text:
        return "troubleshooting"
    if "steps:" in text or "ordered steps" in text:
        return "procedure"
    return "summary"


def _case_pass(case: PromptCase, response: Dict[str, Any]) -> tuple[bool, str]:
    output = str(response.get("output_text") or "")
    refusal = _is_refusal(output, bool(response.get("refusal")))
    citations = response.get("citations") or []
    task_type = str(response.get("task_type") or "")
    result_type = _infer_result_type(task_type, output, refusal)

    if not output.strip():
        return False, "empty_output"
    if refusal:
        if not case.allow_clean_refusal:
            return False, "unexpected_refusal"
        return True, "ok_clean_refusal"
    if not citations:
        return False, "missing_citations"

    lowered_output = output.lower()
    if case.required_answer_substrings:
        missing = [
            token
            for token in case.required_answer_substrings
            if token.lower() not in lowered_output
        ]
        if missing:
            return False, f"missing_expected_content:{','.join(missing)}"

    if case.category == "sql" and result_type != "sql":
        return False, f"expected_sql_result_got_{result_type}"
    if case.category == "fast_formula" and result_type != "fast_formula":
        return False, f"expected_fast_formula_result_got_{result_type}"
    if case.category == "procedure" and result_type not in {"procedure", "summary"}:
        return False, f"expected_procedure_result_got_{result_type}"
    if case.category == "troubleshooting" and result_type != "troubleshooting":
        return False, f"expected_troubleshooting_result_got_{result_type}"

    return True, "ok"


def run_validation(base_url: str, output_path: Path) -> Dict[str, Any]:
    started = time.time()
    tenant_suffix = str(int(time.time()))
    username = f"ui_v1_validation_{tenant_suffix}"
    password = "Iwerp@12345"
    tenant_name = f"UI V1 Validation {tenant_suffix}"

    register_status, register_payload = _http_json(
        "POST",
        f"{base_url}/v1/auth/register",
        {"username": username, "password": password, "tenant_name": tenant_name},
    )
    if register_status != 200:
        raise RuntimeError(f"register_failed status={register_status} payload={register_payload}")

    token = register_payload.get("access_token")
    if not token:
        raise RuntimeError("register_success_but_missing_access_token")

    headers = {"Authorization": f"Bearer {token}"}
    cases: List[Dict[str, Any]] = []
    per_category = defaultdict(lambda: {"total": 0, "pass": 0, "fail": 0})
    failure_reasons = Counter()

    for idx, case in enumerate(PROMPTS, start=1):
        payload = {
            "messages": [{"role": "user", "content": case.prompt}],
            "metadata": {"session_id": f"ui-v1-{tenant_suffix}", "surface": "ui_v1_validation"},
            "debug": False,
        }
        status_code, data = _http_json("POST", f"{base_url}/v1/sovereign/chat/completions", payload, headers=headers)
        output_text = str(data.get("output_text") or "")
        refusal = _is_refusal(output_text, bool(data.get("refusal")))
        result_type = _infer_result_type(str(data.get("task_type") or ""), output_text, refusal)
        passed, reason = (False, f"http_{status_code}")
        if status_code == 200:
            passed, reason = _case_pass(case, data)

        per_category[case.category]["total"] += 1
        per_category[case.category]["pass"] += int(passed)
        per_category[case.category]["fail"] += int(not passed)
        if not passed:
            failure_reasons[reason] += 1

        cases.append(
            {
                "index": idx,
                "prompt": case.prompt,
                "category": case.category,
                "allow_clean_refusal": case.allow_clean_refusal,
                "notes": case.notes,
                "route_selected": data.get("task_type"),
                "selected_module": data.get("selected_module"),
                "result_type": result_type,
                "refusal": refusal,
                "verifier_status": data.get("verifier_status"),
                "citation_count": len(data.get("citations") or []),
                "pass": passed,
                "failure_reason": None if passed else reason,
                "ui_rendering_issue": None,
            }
        )

    total = len(cases)
    passed = sum(1 for c in cases if c["pass"])
    failed = total - passed
    duration = round(time.time() - started, 2)
    summary = {
        "run_label": "v1_ui_local_validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "tenant_name": tenant_name,
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed,
        "pass_rate_pct": round((passed / total) * 100, 2),
        "duration_sec": duration,
        "per_category": per_category,
        "failure_reasons": dict(failure_reasons.most_common()),
        "ui_rendering_issues": [],
    }

    report = {"summary": summary, "cases": cases}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_package_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    cases = report.get("cases") or []
    summary = dict(report.get("summary") or {})
    summary["failed_case_prompts"] = [str(case.get("prompt") or "") for case in cases if not case.get("pass")]
    summary["accepted_guarded_refusals"] = [
        {
            "prompt": str(case.get("prompt") or ""),
            "category": str(case.get("category") or ""),
            "notes": case.get("notes"),
        }
        for case in cases
        if case.get("pass") and case.get("refusal") and case.get("allow_clean_refusal")
    ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local V1 UI/API validation.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--output",
        default="/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod/deploy/v1_ui_local_validation_report.json",
    )
    parser.add_argument(
        "--package-summary-output",
        default="/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod/deploy/azure_v1_package/checks/ui_deployed_validation_summary.json",
    )
    args = parser.parse_args()

    report = run_validation(args.base_url.rstrip("/"), Path(args.output))
    package_summary = build_package_summary(report)
    Path(args.package_summary_output).write_text(json.dumps(package_summary, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
