#!/usr/bin/env python3
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap")
BASE_DIR = ROOT_DIR / "iwerp-prod"
ORACLEWINGS_DIR = ROOT_DIR / "Oraclewings_ai"
DISCOVERY_DIR = BASE_DIR / "oraclewings_readonly_inventory"
OUTPUT_DIR = BASE_DIR / "procedure_troubleshooting_ingestion_schema_pass"

TRANSFORM_VERSION = "procedure_troubleshooting_schema_v1"
SOURCE_REPO = "Oraclewings_ai"
SOURCE_BRANCH = "local_desktop_snapshot"

KNOWN_MODULES = {
    "Financials",
    "Payables",
    "Receivables",
    "General Ledger",
    "Cash Management",
    "Assets",
    "Tax",
    "Expenses",
    "HCM",
    "Procurement",
    "Purchasing",
    "Self Service Procurement",
    "Sourcing",
    "Supplier Portal",
    "SCM",
    "Manufacturing",
}

GENERIC_TASK_TERMS = {
    "introduction",
    "overview",
    "key concepts",
    "types of extracts",
    "faq",
    "common concepts",
}


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    source_path: str
    asset_type: str
    lane_target: str
    source_format: str
    likely_signal_quality: str
    expected_cleanup_needs: str
    transform_strategy: str


SOURCE_SPECS: List[SourceSpec] = [
    SourceSpec(
        source_id="ow_hcm_hdl_extract_flow",
        source_path="backend/orawing_ai/data/sources/hcm/HDL_EXTRACT_FLOW_KNOWLEDGE.json",
        asset_type="hcm_extract_process_knowledge",
        lane_target="doc-grounded procedure; troubleshooting",
        source_format="json",
        likely_signal_quality="high",
        expected_cleanup_needs="low",
        transform_strategy="structured_json_task_and_issue_extraction",
    ),
    SourceSpec(
        source_id="ow_hcm_extract_knowledge_compiled",
        source_path="backend/backend_hcm/app/data/extract_knowledge.json",
        asset_type="hcm_extract_knowledge_bundle",
        lane_target="doc-grounded procedure; troubleshooting",
        source_format="json",
        likely_signal_quality="medium",
        expected_cleanup_needs="medium",
        transform_strategy="embedded_team_guide_section_extraction",
    ),
    SourceSpec(
        source_id="ow_hcm_extract_docx_csv",
        source_path="backend/backend_hcm/hcm_extracts_docx_converted.csv",
        asset_type="hcm_extract_doc",
        lane_target="doc-grounded procedure",
        source_format="csv",
        likely_signal_quality="medium",
        expected_cleanup_needs="medium",
        transform_strategy="section_and_numbered_step_extraction",
    ),
    SourceSpec(
        source_id="ow_hcm_extract_info_csv",
        source_path="backend/backend_hcm/HCM_Extract_Oracle_info.csv",
        asset_type="hcm_extract_team_guide",
        lane_target="doc-grounded procedure; troubleshooting",
        source_format="csv",
        likely_signal_quality="medium",
        expected_cleanup_needs="high",
        transform_strategy="flow_block_and_issue_pattern_extraction",
    ),
    SourceSpec(
        source_id="ow_scm_troubleshooting_kb",
        source_path="backend/scm_bot_backend/knowledge_base/troubleshooting_kb.json",
        asset_type="structured_troubleshooting_kb",
        lane_target="troubleshooting",
        source_format="json",
        likely_signal_quality="high",
        expected_cleanup_needs="low",
        transform_strategy="json_issue_cause_solution_mapping",
    ),
    SourceSpec(
        source_id="ow_scm_e2e_processes_kb",
        source_path="backend/scm_bot_backend/knowledge_base/e2e_processes_kb.json",
        asset_type="structured_process_kb",
        lane_target="doc-grounded procedure",
        source_format="json",
        likely_signal_quality="high",
        expected_cleanup_needs="low",
        transform_strategy="json_process_step_mapping",
    ),
    SourceSpec(
        source_id="ow_scm_functional_kb",
        source_path="backend/scm_bot_backend/knowledge_base/functional_kb.json",
        asset_type="functional_scenario_kb",
        lane_target="doc-grounded procedure",
        source_format="json",
        likely_signal_quality="medium",
        expected_cleanup_needs="medium",
        transform_strategy="scenario_step_mapping_with_aliases",
    ),
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _read_csv_content(path: Path) -> str:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, {})
    return str(row.get("content", "")).strip()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _stable_id(prefix: str, parts: Iterable[str]) -> str:
    joined = "||".join(_normalize_text(p) for p in parts if p)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _split_sentences(value: str) -> List[str]:
    text = _normalize_text(value)
    if not text:
        return []
    bits = re.split(r"(?<=[\.\!\?])\s+", text)
    return [b.strip(" -\u2022") for b in bits if b.strip(" -\u2022")]


def _extract_numbered_steps(text: str) -> List[str]:
    normalized = text.replace("\r", "\n")
    candidates = re.findall(
        r"(?m)(?:^|\n)\s*(?:Step\s*\d+|\d+)[\.\)]\s*([^\n]+)",
        normalized,
        flags=re.IGNORECASE,
    )
    cleaned: List[str] = []
    seen = set()
    for line in candidates:
        step = _normalize_text(line)
        if len(step) < 6:
            continue
        if step.lower() in seen:
            continue
        seen.add(step.lower())
        cleaned.append(step)
    return cleaned


def _list_to_steps(items: List[str]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        output.append(
            {
                "step_number": idx,
                "step_text": _normalize_text(item),
                "substeps": [],
                "decision_notes": [],
                "validation_checks": [],
            }
        )
    return output


def _extract_navigation_path(text: str) -> Optional[str]:
    match = re.search(r"(?i)(navigate(?:\s+to)?\s*:\s*[^\n]+)", text)
    if match:
        return _normalize_text(match.group(1))
    arrow_match = re.search(r"[A-Za-z][A-Za-z0-9 /]+(?:\s*→\s*[A-Za-z0-9 /]+){1,}", text)
    if arrow_match:
        return _normalize_text(arrow_match.group(0))
    return None


def _infer_module_from_text(text: str, fallback: str = "HCM") -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("payables", "invoice", "supplier site", "payment terms")):
        return "Payables"
    if any(token in lowered for token in ("receivables", "receipt application", "customer")):
        return "Receivables"
    if any(token in lowered for token in ("general ledger", "journal", "period close", "ledger")):
        return "General Ledger"
    if any(token in lowered for token in ("expenses", "expense report")):
        return "Expenses"
    if any(token in lowered for token in ("procurement", "purchase requisition", "purchase order", "supplier")):
        return "Procurement"
    if any(token in lowered for token in ("sourcing", "rfq", "auction")):
        return "Sourcing"
    if any(token in lowered for token in ("manufacturing", "work order")):
        return "Manufacturing"
    if any(token in lowered for token in ("hcm", "payroll", "extract", "hdl", "hsdl", "worker", "assignment")):
        return "HCM"
    return fallback


def _confidence(base: float, penalties: float = 0.0, floor: float = 0.1) -> float:
    return max(floor, round(base - penalties, 2))


def _citation(source_path: str, anchor: str, excerpt: str) -> Dict[str, Any]:
    return {
        "source_path": source_path,
        "source_anchor": anchor,
        "source_excerpt": _normalize_text(excerpt)[:320],
    }


def _provenance(source_id: str, source_path: str, anchor: str) -> Dict[str, Any]:
    return {
        "source_repo": SOURCE_REPO,
        "source_branch": SOURCE_BRANCH,
        "source_id": source_id,
        "source_path": source_path,
        "source_anchor": anchor,
        "transform_version": TRANSFORM_VERSION,
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _procedure_object(
    *,
    source_id: str,
    source_path: str,
    module: str,
    task_name: str,
    task_aliases: List[str],
    prerequisites: List[str],
    ordered_steps: List[Dict[str, Any]],
    navigation_path: Optional[str],
    roles_or_personas: List[str],
    warnings_or_constraints: List[str],
    expected_outcome: str,
    keywords: List[str],
    citation_anchor: str,
    citation_excerpt: str,
    confidence_score: float,
    submodule: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "object_type": "procedure",
        "source_id": source_id,
        "source_path": source_path,
        "module": module,
        "submodule": submodule,
        "task_name": task_name,
        "task_aliases": sorted({a for a in task_aliases if a}),
        "intent_type": "procedure",
        "prerequisites": [p for p in map(_normalize_text, prerequisites) if p],
        "ordered_steps": ordered_steps,
        "step_count": len(ordered_steps),
        "navigation_path": navigation_path,
        "roles_or_personas": [r for r in map(_normalize_text, roles_or_personas) if r],
        "warnings_or_constraints": [w for w in map(_normalize_text, warnings_or_constraints) if w],
        "expected_outcome": _normalize_text(expected_outcome),
        "keywords": sorted({k.lower() for k in keywords if k}),
        "citations": [_citation(source_path, citation_anchor, citation_excerpt)],
        "confidence_score": confidence_score,
        "provenance_metadata": _provenance(source_id, source_path, citation_anchor),
    }


def _troubleshooting_object(
    *,
    source_id: str,
    source_path: str,
    module: str,
    symptom: str,
    symptom_aliases: List[str],
    probable_causes: List[Dict[str, Any]],
    diagnostics: List[str],
    resolution_steps: List[str],
    prevention_notes: List[str],
    related_tasks: List[str],
    keywords: List[str],
    citation_anchor: str,
    citation_excerpt: str,
    confidence_score: float,
    submodule: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "object_type": "troubleshooting",
        "source_id": source_id,
        "source_path": source_path,
        "module": module,
        "submodule": submodule,
        "symptom": _normalize_text(symptom),
        "symptom_aliases": sorted({a for a in symptom_aliases if a}),
        "probable_causes": probable_causes,
        "diagnostics": [d for d in map(_normalize_text, diagnostics) if d],
        "resolution_steps": [s for s in map(_normalize_text, resolution_steps) if s],
        "prevention_notes": [p for p in map(_normalize_text, prevention_notes) if p],
        "related_tasks": [r for r in map(_normalize_text, related_tasks) if r],
        "keywords": sorted({k.lower() for k in keywords if k}),
        "citations": [_citation(source_path, citation_anchor, citation_excerpt)],
        "confidence_score": confidence_score,
        "provenance_metadata": _provenance(source_id, source_path, citation_anchor),
    }


def _cause_item(label: str, description: str, evidence: str, fix_steps: List[str]) -> Dict[str, Any]:
    return {
        "cause_label": _normalize_text(label),
        "cause_description": _normalize_text(description),
        "evidence_text": _normalize_text(evidence),
        "fix_steps": [s for s in map(_normalize_text, fix_steps) if s],
        "validation_of_fix": "",
    }


def _parse_hdl_extract_flow(source: SourceSpec) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    procedures: List[Dict[str, Any]] = []
    troubleshooting: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    source_path = ORACLEWINGS_DIR / source.source_path
    data = _read_json(source_path)
    prerequisites = [str(x) for x in data.get("prerequisites", [])]
    related_resources = data.get("related_resources", [])
    reference_links = [r.get("link", "") for r in related_resources if isinstance(r, dict)]

    for task in data.get("step_by_step_tasks", []):
        task_number = task.get("task_number")
        task_name = _normalize_text(task.get("task_name", "HCM Extract Task"))
        step_rows = task.get("steps", [])
        ordered_steps: List[Dict[str, Any]] = []
        for idx, row in enumerate(step_rows, start=1):
            step_text_bits = [row.get("step", ""), row.get("action", ""), row.get("details", ""), row.get("purpose", "")]
            step_text = ". ".join([_normalize_text(str(x)) for x in step_text_bits if str(x).strip()])
            decision_notes = []
            for key in ("type", "settings", "binding_type", "parameters", "success_criteria", "structure", "example"):
                if key in row and row.get(key):
                    decision_notes.append(f"{key}: {_normalize_text(str(row.get(key)))}")
            ordered_steps.append(
                {
                    "step_number": idx,
                    "step_text": step_text,
                    "substeps": [],
                    "decision_notes": decision_notes,
                    "validation_checks": [f"Validate completion of task step {idx} for {task_name}."],
                }
            )

        object_id = _stable_id("proc", [source.source_id, task_name, str(task_number)])
        nav_text = " ".join(str(row.get("action", "")) for row in step_rows)
        procedure = _procedure_object(
            source_id=source.source_id,
            source_path=source.source_path,
            module="HCM",
            submodule="HCM Extracts",
            task_name=task_name,
            task_aliases=[task_name.lower(), f"hcm extracts {task_name.lower()}"],
            prerequisites=prerequisites,
            ordered_steps=ordered_steps,
            navigation_path=_extract_navigation_path(nav_text),
            roles_or_personas=["HCM Extract Administrator", "Payroll Administrator"],
            warnings_or_constraints=[
                "Use production encryption settings for outbound files.",
                "Ensure Auto Load is configured correctly when flow automation is required.",
            ],
            expected_outcome="Task completes with extract configuration validated and ready for downstream processing.",
            keywords=["hcm extract", "hdl", "hsdl", "payroll flow", "data exchange", task_name],
            citation_anchor=f"step_by_step_tasks.task_number={task_number}",
            citation_excerpt=json.dumps(task, ensure_ascii=True),
            confidence_score=_confidence(0.94),
        )
        procedure["object_id"] = object_id
        procedures.append(procedure)

    for idx, issue in enumerate(data.get("common_errors_and_troubleshooting", []), start=1):
        issue_text = str(issue.get("issue", "")).strip()
        solution_text = str(issue.get("solution", "")).strip()
        if not issue_text or not solution_text:
            rejected.append(
                {
                    "source_id": source.source_id,
                    "source_path": source.source_path,
                    "reason": "missing_issue_or_solution",
                    "raw_object": issue,
                }
            )
            continue
        fixes = _split_sentences(solution_text)
        causes = [
            _cause_item(
                label=f"Likely cause for {issue_text}",
                description=f"Configuration or flow setup issue leading to: {issue_text}.",
                evidence=issue_text,
                fix_steps=fixes,
            )
        ]
        t = _troubleshooting_object(
            source_id=source.source_id,
            source_path=source.source_path,
            module="HCM",
            submodule="HCM Extracts",
            symptom=issue_text,
            symptom_aliases=[issue_text.lower(), f"hcm extract {issue_text.lower()}"],
            probable_causes=causes,
            diagnostics=[
                "Review extract execution tree and validation messages.",
                "Inspect delivery option parameter values and runtime flow bindings.",
            ],
            resolution_steps=fixes,
            prevention_notes=["Validate extract definition after each delivery option change."],
            related_tasks=["Run HCM Extract", "View Extract Results", "Run HCM Data Loader"],
            keywords=["hcm extract", "troubleshooting", "hdl", "flow", issue_text],
            citation_anchor=f"common_errors_and_troubleshooting[{idx}]",
            citation_excerpt=json.dumps(issue, ensure_ascii=True),
            confidence_score=_confidence(0.9),
        )
        t["object_id"] = _stable_id("trb", [source.source_id, issue_text, str(idx)])
        troubleshooting.append(t)
    return procedures, troubleshooting, rejected


def _parse_extract_knowledge(source: SourceSpec) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    procedures: List[Dict[str, Any]] = []
    troubleshooting: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    source_path = ORACLEWINGS_DIR / source.source_path
    data = _read_json(source_path)
    docs = data.get("documentation", [])
    for doc in docs:
        doc_source = str(doc.get("source", ""))
        content = str(doc.get("content", ""))
        if not content:
            continue

        if doc_source == "Submit_Extract_Steps.csv":
            reader = csv.DictReader(content.splitlines())
            steps = []
            for row in reader:
                detail = _normalize_text(row.get("Details", ""))
                if detail:
                    steps.append(detail)
            ordered_steps = _list_to_steps(steps)
            procedure = _procedure_object(
                source_id=source.source_id,
                source_path=source.source_path,
                module="HCM",
                submodule="HCM Extracts",
                task_name="Submit and Monitor HCM Extract Run",
                task_aliases=["submit extract", "view extract results", "run extract"],
                prerequisites=["Extract definition exists and is validated."],
                ordered_steps=ordered_steps,
                navigation_path="My Client Groups -> Data Exchange -> Submit Extracts",
                roles_or_personas=["HCM Extract Operator"],
                warnings_or_constraints=["Parameters and effective date must align with extract design."],
                expected_outcome="Extract run completes and output/log files are available for download.",
                keywords=["hcm extract", "submit extracts", "view extract results", "scheduled processes"],
                citation_anchor="documentation[source=Submit_Extract_Steps.csv]",
                citation_excerpt=content[:700],
                confidence_score=_confidence(0.95),
            )
            procedure["object_id"] = _stable_id("proc", [source.source_id, procedure["task_name"]])
            procedures.append(procedure)
            continue

        if doc_source == "hcm_extracts_docx_converted.csv":
            section_flow = re.search(r"3\.\s*Extract Process Flow:(.+?)4\.\s*Types of Extracts:", content, flags=re.IGNORECASE | re.DOTALL)
            if section_flow:
                steps = _extract_numbered_steps(section_flow.group(1))
                if steps:
                    procedure = _procedure_object(
                        source_id=source.source_id,
                        source_path=source.source_path,
                        module="HCM",
                        submodule="HCM Extracts",
                        task_name="Execute HCM Extract Process Flow",
                        task_aliases=["hcm extract process flow", "run extract flow"],
                        prerequisites=["HCM Extract definition and delivery options are configured."],
                        ordered_steps=_list_to_steps(steps),
                        navigation_path="Data Exchange -> HCM Extracts",
                        roles_or_personas=["HCM Extract Administrator"],
                        warnings_or_constraints=["Monitor extract status before downloading output."],
                        expected_outcome="Process flow completes from design through output delivery.",
                        keywords=["extract process flow", "hcm extracts", "view extract results"],
                        citation_anchor="documentation[source=hcm_extracts_docx_converted.csv].section=Extract Process Flow",
                        citation_excerpt=section_flow.group(1)[:700],
                        confidence_score=_confidence(0.92),
                    )
                    procedure["object_id"] = _stable_id("proc", [source.source_id, procedure["task_name"]])
                    procedures.append(procedure)

            section_simple = re.search(
                r"5\.\s*Creating a Simple Extract.*?Steps:(.+?)6\.\s*Key Concepts:",
                content,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if section_simple:
                lines = [ln.strip(" -\u2022") for ln in section_simple.group(1).splitlines() if ln.strip()]
                steps = [ln for ln in lines if len(ln) > 6]
                if len(steps) >= 4:
                    procedure = _procedure_object(
                        source_id=source.source_id,
                        source_path=source.source_path,
                        module="HCM",
                        submodule="HCM Extracts",
                        task_name="Create Simple Employee Details Extract",
                        task_aliases=["create hcm extract", "employee details extract"],
                        prerequisites=["Access to My Client Groups and Data Exchange."],
                        ordered_steps=_list_to_steps(steps[:12]),
                        navigation_path="My Client Groups -> Data Exchange -> HCM Extracts",
                        roles_or_personas=["HCM Functional Consultant"],
                        warnings_or_constraints=[
                            "Use appropriate data group and primary key values.",
                            "Validate extract before submit.",
                        ],
                        expected_outcome="Employee detail extract is created, executed, and output is available.",
                        keywords=["employee details extract", "hcm extracts", "data group", "delivery option"],
                        citation_anchor="documentation[source=hcm_extracts_docx_converted.csv].section=Creating a Simple Extract",
                        citation_excerpt=section_simple.group(1)[:700],
                        confidence_score=_confidence(0.9),
                    )
                    procedure["object_id"] = _stable_id("proc", [source.source_id, procedure["task_name"]])
                    procedures.append(procedure)
            continue

        if doc_source == "HCM_Extract_Oracle_info.csv":
            block = re.search(
                r"How HCM Extract Works \(Simple Flow\)(.+?)Advantages",
                content,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if block:
                flow_steps = []
                for raw in block.group(1).splitlines():
                    line = _normalize_text(raw)
                    if not line:
                        continue
                    if "–" in line:
                        flow_steps.append(line)
                    elif line.lower().startswith(("create extract", "add attributes", "create delivery option", "submit extract", "extract engine", "file delivered")):
                        flow_steps.append(line)
                if len(flow_steps) >= 4:
                    procedure = _procedure_object(
                        source_id=source.source_id,
                        source_path=source.source_path,
                        module="HCM",
                        submodule="HCM Extracts",
                        task_name="Run HCM Extract Simple Flow",
                        task_aliases=["simple flow", "hcm extract workflow"],
                        prerequisites=["Extract purpose and output format are defined."],
                        ordered_steps=_list_to_steps(flow_steps[:10]),
                        navigation_path="Data Exchange -> HCM Extracts",
                        roles_or_personas=["HCM Extract Specialist"],
                        warnings_or_constraints=["Delivery target (SFTP/UCM/Email) must be configured before runtime."],
                        expected_outcome="Extract engine generates and delivers output file.",
                        keywords=["hcm extract", "simple flow", "delivery option", "submit extract"],
                        citation_anchor="documentation[source=HCM_Extract_Oracle_info.csv].section=How HCM Extract Works",
                        citation_excerpt=block.group(1)[:700],
                        confidence_score=_confidence(0.86, penalties=0.05),
                    )
                    procedure["object_id"] = _stable_id("proc", [source.source_id, procedure["task_name"]])
                    procedures.append(procedure)

            issue_pairs = re.findall(
                r"(?is)(?:Issue|Error)\s*[:\-]\s*(.+?)\n(?:Solution|Fix)\s*[:\-]\s*(.+?)(?=\n(?:Issue|Error)\s*[:\-]|\Z)",
                content,
            )
            for idx, (issue_text, solution_text) in enumerate(issue_pairs, start=1):
                fixes = _split_sentences(solution_text)
                if not fixes:
                    continue
                obj = _troubleshooting_object(
                    source_id=source.source_id,
                    source_path=source.source_path,
                    module="HCM",
                    submodule="HCM Extracts",
                    symptom=issue_text,
                    symptom_aliases=[_normalize_text(issue_text).lower()],
                    probable_causes=[
                        _cause_item(
                            label="Likely extract configuration issue",
                            description="The extract setup or parameterization is inconsistent with expected runtime configuration.",
                            evidence=issue_text,
                            fix_steps=fixes,
                        )
                    ],
                    diagnostics=["Review Extract Results logs and parameter values used at submission time."],
                    resolution_steps=fixes,
                    prevention_notes=["Validate extract and delivery configuration prior to schedule."],
                    related_tasks=["Submit Extracts", "View Extract Results"],
                    keywords=["hcm extract", "error", "troubleshooting", issue_text],
                    citation_anchor=f"documentation[source=HCM_Extract_Oracle_info.csv].issue[{idx}]",
                    citation_excerpt=f"Issue: {issue_text} Solution: {solution_text}",
                    confidence_score=_confidence(0.72),
                )
                obj["object_id"] = _stable_id("trb", [source.source_id, issue_text, str(idx)])
                troubleshooting.append(obj)
            continue

        if doc_source == "hcm_extracts_converted.csv":
            rejected.append(
                {
                    "source_id": source.source_id,
                    "source_path": source.source_path,
                    "reason": "ocr_noise_high_for_direct_object_extraction",
                    "raw_excerpt": content[:700],
                }
            )
            continue

    return procedures, troubleshooting, rejected


def _parse_scm_troubleshooting(source: SourceSpec) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    procedures: List[Dict[str, Any]] = []
    troubleshooting: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    data = _read_json(ORACLEWINGS_DIR / source.source_path)

    def add_issue(symptom: str, causes: List[str], solutions: List[str], anchor: str, extra_keywords: Optional[List[str]] = None) -> None:
        if not symptom or not solutions:
            rejected.append(
                {
                    "source_id": source.source_id,
                    "source_path": source.source_path,
                    "reason": "missing_symptom_or_solution",
                    "raw_object": {"symptom": symptom, "causes": causes, "solutions": solutions},
                }
            )
            return
        cause_rows = causes or ["Configuration mismatch or data quality issue."]
        probable_causes = []
        for idx, cause in enumerate(cause_rows, start=1):
            probable_causes.append(
                _cause_item(
                    label=f"Cause {idx}",
                    description=cause,
                    evidence=symptom,
                    fix_steps=solutions,
                )
            )
        obj = _troubleshooting_object(
            source_id=source.source_id,
            source_path=source.source_path,
            module="Procurement",
            submodule="SCM",
            symptom=symptom,
            symptom_aliases=[symptom.lower()],
            probable_causes=probable_causes,
            diagnostics=[
                "Check transaction status and approvals in Fusion work area.",
                "Review workflow/BPM and setup configuration logs.",
            ],
            resolution_steps=solutions,
            prevention_notes=["Keep approval rules and setup configuration aligned with transaction volume and policy."],
            related_tasks=["Purchase Requisition", "Purchase Order", "Workflow Approvals"],
            keywords=["troubleshooting", "scm", "procurement", symptom] + (extra_keywords or []),
            citation_anchor=anchor,
            citation_excerpt=symptom,
            confidence_score=_confidence(0.9 if causes else 0.8),
        )
        obj["object_id"] = _stable_id("trb", [source.source_id, symptom, anchor])
        troubleshooting.append(obj)

    for idx, row in enumerate(data.get("fusion_errors", []), start=1):
        add_issue(
            symptom=str(row.get("error", "")).strip(),
            causes=[str(x) for x in row.get("causes", [])],
            solutions=[str(x) for x in row.get("solutions", [])],
            anchor=f"fusion_errors[{idx}]",
        )

    for idx, row in enumerate(data.get("ora_errors", []), start=1):
        causes = [str(row.get("meaning", ""))] if row.get("meaning") else []
        add_issue(
            symptom=str(row.get("error", "")).strip(),
            causes=causes,
            solutions=[str(x) for x in row.get("solutions", [])],
            anchor=f"ora_errors[{idx}]",
            extra_keywords=[str(row.get("context", "")).strip()],
        )

    for idx, row in enumerate(data.get("workflow_errors", []), start=1):
        add_issue(
            symptom=str(row.get("error", "")).strip(),
            causes=[str(x) for x in row.get("causes", [])],
            solutions=[str(x) for x in row.get("solutions", [])],
            anchor=f"workflow_errors[{idx}]",
        )

    for idx, row in enumerate(data.get("common_issues", []), start=1):
        add_issue(
            symptom=str(row.get("issue", "")).strip(),
            causes=[],
            solutions=[str(x) for x in row.get("solutions", [])],
            anchor=f"common_issues[{idx}]",
        )

    return procedures, troubleshooting, rejected


def _parse_e2e_processes(source: SourceSpec) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    procedures: List[Dict[str, Any]] = []
    troubleshooting: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    data = _read_json(ORACLEWINGS_DIR / source.source_path)
    for idx, process in enumerate(data.get("processes", []), start=1):
        name = _normalize_text(process.get("name", ""))
        steps = process.get("steps", [])
        ordered_steps: List[Dict[str, Any]] = []
        for step in steps:
            step_no = int(step.get("step", len(ordered_steps) + 1))
            phase = _normalize_text(step.get("phase", ""))
            activity = _normalize_text(step.get("activity", ""))
            details = _normalize_text(step.get("details", ""))
            step_text = ". ".join([x for x in [phase, activity, details] if x])
            ordered_steps.append(
                {
                    "step_number": step_no,
                    "step_text": step_text,
                    "substeps": [],
                    "decision_notes": [f"phase: {phase}"] if phase else [],
                    "validation_checks": [f"Confirm completion of {activity}."],
                }
            )
        module = _infer_module_from_text(name, fallback="Procurement")
        procedure = _procedure_object(
            source_id=source.source_id,
            source_path=source.source_path,
            module=module,
            submodule="End-to-End Process",
            task_name=name,
            task_aliases=[name.lower(), _slugify(name).replace("-", " ")],
            prerequisites=["Relevant setup, approvals, and master data are configured."],
            ordered_steps=sorted(ordered_steps, key=lambda x: x["step_number"]),
            navigation_path=None,
            roles_or_personas=["Functional Consultant", "Process Owner"],
            warnings_or_constraints=["Ensure status transitions are completed before moving to the next phase."],
            expected_outcome=process.get("description", "Process completes end to end with expected transaction status."),
            keywords=[name, module, "process flow", "oracle fusion"],
            citation_anchor=f"processes[{idx}]",
            citation_excerpt=json.dumps(process, ensure_ascii=True),
            confidence_score=_confidence(0.9),
        )
        procedure["object_id"] = _stable_id("proc", [source.source_id, name, str(idx)])
        procedures.append(procedure)
    return procedures, troubleshooting, rejected


def _parse_functional_kb(source: SourceSpec) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    procedures: List[Dict[str, Any]] = []
    troubleshooting: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    data = _read_json(ORACLEWINGS_DIR / source.source_path)
    for topic_idx, topic in enumerate(data.get("topics", []), start=1):
        topic_name = _normalize_text(topic.get("name", ""))
        module = _infer_module_from_text(topic_name, fallback="Procurement")
        navs = [str(x) for x in topic.get("navigation_paths", [])]
        for sc_idx, scenario in enumerate(topic.get("common_scenarios", []), start=1):
            sc_name = _normalize_text(scenario.get("scenario", ""))
            sc_steps = [str(s) for s in scenario.get("steps", [])]
            if len(sc_steps) < 3:
                rejected.append(
                    {
                        "source_id": source.source_id,
                        "source_path": source.source_path,
                        "reason": "scenario_steps_too_short",
                        "raw_object": scenario,
                    }
                )
                continue
            procedure = _procedure_object(
                source_id=source.source_id,
                source_path=source.source_path,
                module=module,
                submodule=topic_name,
                task_name=sc_name or f"{topic_name} Scenario {sc_idx}",
                task_aliases=[topic_name.lower(), (sc_name or "").lower()],
                prerequisites=["Relevant configuration and user access are available."],
                ordered_steps=_list_to_steps(sc_steps),
                navigation_path=navs[0] if navs else None,
                roles_or_personas=["Requester", "Buyer", "Approver"],
                warnings_or_constraints=["Follow status and approval rules for each transaction stage."],
                expected_outcome=f"{topic_name} scenario is completed with expected status progression.",
                keywords=[topic_name, sc_name, "oracle fusion", module],
                citation_anchor=f"topics[{topic_idx}].common_scenarios[{sc_idx}]",
                citation_excerpt=json.dumps(scenario, ensure_ascii=True),
                confidence_score=_confidence(0.82),
            )
            procedure["object_id"] = _stable_id("proc", [source.source_id, topic_name, sc_name, str(sc_idx)])
            procedures.append(procedure)
    return procedures, troubleshooting, rejected


def _parse_hcm_docx_csv(source: SourceSpec) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    procedures: List[Dict[str, Any]] = []
    troubleshooting: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    content = _read_csv_content(ORACLEWINGS_DIR / source.source_path)
    flow = re.search(r"3\.\s*Extract Process Flow:(.+?)4\.\s*Types of Extracts:", content, flags=re.IGNORECASE | re.DOTALL)
    if flow:
        steps = _extract_numbered_steps(flow.group(1))
        if steps:
            obj = _procedure_object(
                source_id=source.source_id,
                source_path=source.source_path,
                module="HCM",
                submodule="HCM Extracts",
                task_name="HCM Extract Process Flow (Docx Converted)",
                task_aliases=["extract process flow", "hcm extracts flow"],
                prerequisites=["HCM Extract access and data model setup are available."],
                ordered_steps=_list_to_steps(steps),
                navigation_path="My Client Groups -> Data Exchange -> HCM Extracts",
                roles_or_personas=["HCM Extract Specialist"],
                warnings_or_constraints=["Monitor status in View Extract Results before downstream transfer."],
                expected_outcome="Extract process flow completes and output is ready for transfer.",
                keywords=["hcm extract", "process flow", "run extract", "view extract results"],
                citation_anchor="section=Extract Process Flow",
                citation_excerpt=flow.group(1)[:700],
                confidence_score=_confidence(0.84),
            )
            obj["object_id"] = _stable_id("proc", [source.source_id, obj["task_name"]])
            procedures.append(obj)

    tips = re.search(r"Troubleshooting Tips:(.+)$", content, flags=re.IGNORECASE | re.DOTALL)
    if tips:
        lines = [ln.strip(" -\u2022") for ln in tips.group(1).splitlines() if ln.strip()]
        if lines:
            symptom = "HCM extract run issues during setup or execution"
            obj = _troubleshooting_object(
                source_id=source.source_id,
                source_path=source.source_path,
                module="HCM",
                submodule="HCM Extracts",
                symptom=symptom,
                symptom_aliases=["hcm extract issue", "extract run error"],
                probable_causes=[
                    _cause_item(
                        label="Setup/configuration mismatch",
                        description="One or more extract setup elements are incomplete or inconsistent.",
                        evidence=symptom,
                        fix_steps=lines[:8],
                    )
                ],
                diagnostics=["Check extract definition, delivery option, and runtime parameters."],
                resolution_steps=lines[:8],
                prevention_notes=["Use validated template and check mandatory fields before run."],
                related_tasks=["Create Extract", "Submit Extract", "View Extract Results"],
                keywords=["hcm extract", "troubleshooting", "setup", "execution"],
                citation_anchor="section=Troubleshooting Tips",
                citation_excerpt=tips.group(1)[:700],
                confidence_score=_confidence(0.74),
            )
            obj["object_id"] = _stable_id("trb", [source.source_id, symptom])
            troubleshooting.append(obj)
    return procedures, troubleshooting, rejected


def _parse_hcm_info_csv(source: SourceSpec) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    procedures: List[Dict[str, Any]] = []
    troubleshooting: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    content = _read_csv_content(ORACLEWINGS_DIR / source.source_path)

    flow_block = re.search(
        r"How HCM Extract Works \(Simple Flow\)(.+?)Advantages",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if flow_block:
        lines = [_normalize_text(x) for x in flow_block.group(1).splitlines() if _normalize_text(x)]
        steps: List[str] = []
        for line in lines:
            if line.lower().startswith(
                (
                    "create extract",
                    "add attributes",
                    "create delivery option",
                    "submit extract",
                    "extract engine",
                    "file delivered",
                )
            ):
                steps.append(line)
        if len(steps) >= 4:
            obj = _procedure_object(
                source_id=source.source_id,
                source_path=source.source_path,
                module="HCM",
                submodule="HCM Extracts",
                task_name="HCM Extract Simple Flow (Info Guide)",
                task_aliases=["hcm extract simple flow", "extract workflow"],
                prerequisites=["Extract purpose, output format, and delivery target are identified."],
                ordered_steps=_list_to_steps(steps),
                navigation_path="Data Exchange -> HCM Extracts",
                roles_or_personas=["HCM Functional Analyst"],
                warnings_or_constraints=[
                    "Large-volume extracts should use threading and validated DBI mappings.",
                    "Delivery channel security must meet integration policy.",
                ],
                expected_outcome="Extract output is generated and delivered to configured target.",
                keywords=["hcm extract", "simple flow", "delivery", "submit extract"],
                citation_anchor="section=How HCM Extract Works (Simple Flow)",
                citation_excerpt=flow_block.group(1)[:700],
                confidence_score=_confidence(0.78, penalties=0.08),
            )
            obj["object_id"] = _stable_id("proc", [source.source_id, obj["task_name"]])
            procedures.append(obj)
        else:
            rejected.append(
                {
                    "source_id": source.source_id,
                    "source_path": source.source_path,
                    "reason": "insufficient_flow_steps_from_noisy_text",
                    "raw_excerpt": flow_block.group(1)[:700],
                }
            )
    return procedures, troubleshooting, rejected


def _procedure_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Normalized Procedure Grounding Object",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "object_type",
            "source_id",
            "source_path",
            "module",
            "task_name",
            "task_aliases",
            "intent_type",
            "prerequisites",
            "ordered_steps",
            "step_count",
            "roles_or_personas",
            "warnings_or_constraints",
            "expected_outcome",
            "keywords",
            "citations",
            "confidence_score",
            "provenance_metadata",
        ],
        "properties": {
            "object_id": {"type": "string"},
            "object_type": {"const": "procedure"},
            "source_id": {"type": "string"},
            "source_path": {"type": "string"},
            "module": {"type": "string"},
            "submodule": {"type": ["string", "null"]},
            "task_name": {"type": "string"},
            "task_aliases": {"type": "array", "items": {"type": "string"}},
            "intent_type": {"type": "string"},
            "prerequisites": {"type": "array", "items": {"type": "string"}},
            "ordered_steps": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["step_number", "step_text", "substeps", "decision_notes", "validation_checks"],
                    "properties": {
                        "step_number": {"type": "integer", "minimum": 1},
                        "step_text": {"type": "string"},
                        "substeps": {"type": "array", "items": {"type": "string"}},
                        "decision_notes": {"type": "array", "items": {"type": "string"}},
                        "validation_checks": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "step_count": {"type": "integer", "minimum": 1},
            "navigation_path": {"type": ["string", "null"]},
            "roles_or_personas": {"type": "array", "items": {"type": "string"}},
            "warnings_or_constraints": {"type": "array", "items": {"type": "string"}},
            "expected_outcome": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "citations": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["source_path", "source_anchor", "source_excerpt"],
                    "properties": {
                        "source_path": {"type": "string"},
                        "source_anchor": {"type": "string"},
                        "source_excerpt": {"type": "string"},
                    },
                },
            },
            "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
            "provenance_metadata": {"type": "object"},
            "validation_status": {"type": "string"},
            "validation_reasons": {"type": "array", "items": {"type": "string"}},
        },
    }


def _troubleshooting_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Normalized Troubleshooting Grounding Object",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "object_type",
            "source_id",
            "source_path",
            "module",
            "symptom",
            "symptom_aliases",
            "probable_causes",
            "diagnostics",
            "resolution_steps",
            "related_tasks",
            "keywords",
            "citations",
            "confidence_score",
            "provenance_metadata",
        ],
        "properties": {
            "object_id": {"type": "string"},
            "object_type": {"const": "troubleshooting"},
            "source_id": {"type": "string"},
            "source_path": {"type": "string"},
            "module": {"type": "string"},
            "submodule": {"type": ["string", "null"]},
            "symptom": {"type": "string"},
            "symptom_aliases": {"type": "array", "items": {"type": "string"}},
            "probable_causes": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": [
                        "cause_label",
                        "cause_description",
                        "evidence_text",
                        "fix_steps",
                        "validation_of_fix",
                    ],
                    "properties": {
                        "cause_label": {"type": "string"},
                        "cause_description": {"type": "string"},
                        "evidence_text": {"type": "string"},
                        "fix_steps": {"type": "array", "items": {"type": "string"}},
                        "validation_of_fix": {"type": "string"},
                    },
                },
            },
            "diagnostics": {"type": "array", "items": {"type": "string"}},
            "resolution_steps": {"type": "array", "items": {"type": "string"}},
            "prevention_notes": {"type": "array", "items": {"type": "string"}},
            "related_tasks": {"type": "array", "items": {"type": "string"}},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "citations": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["source_path", "source_anchor", "source_excerpt"],
                    "properties": {
                        "source_path": {"type": "string"},
                        "source_anchor": {"type": "string"},
                        "source_excerpt": {"type": "string"},
                    },
                },
            },
            "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
            "provenance_metadata": {"type": "object"},
            "validation_status": {"type": "string"},
            "validation_reasons": {"type": "array", "items": {"type": "string"}},
        },
    }


def _validate_procedure(obj: Dict[str, Any]) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    status = "accepted"

    module = obj.get("module")
    if not module or module not in KNOWN_MODULES:
        reasons.append("module_missing_or_ambiguous")
        status = "quarantined"

    task_name = _normalize_text(obj.get("task_name", ""))
    if not task_name or len(task_name) < 8:
        reasons.append("task_name_too_short")
        status = "quarantined"
    if task_name.lower() in GENERIC_TASK_TERMS:
        reasons.append("task_name_too_generic")
        status = "quarantined"

    steps = obj.get("ordered_steps", [])
    if not isinstance(steps, list) or len(steps) < 3:
        reasons.append("ordered_steps_missing_or_too_short")
        status = "quarantined"

    if obj.get("step_count") != len(steps):
        reasons.append("step_count_mismatch")
        if status == "accepted":
            status = "accepted_with_cleanup"

    if not obj.get("citations"):
        reasons.append("citations_missing")
        status = "rejected"

    if not obj.get("provenance_metadata"):
        reasons.append("provenance_missing")
        status = "rejected"

    if obj.get("confidence_score", 0) < 0.7 and status == "accepted":
        reasons.append("low_confidence_requires_cleanup")
        status = "accepted_with_cleanup"

    noise_hits = sum(1 for step in steps if re.search(r"\b(?:\.\s*){4,}", step.get("step_text", "")))
    if noise_hits > 0:
        reasons.append("ocr_noise_detected")
        if status == "accepted":
            status = "accepted_with_cleanup"

    if "tmp" in str(obj.get("source_path", "")).lower():
        reasons.append("temporary_source_not_supported")
        status = "rejected"

    return status, reasons


def _validate_troubleshooting(obj: Dict[str, Any]) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    status = "accepted"

    module = obj.get("module")
    if not module or module not in KNOWN_MODULES:
        reasons.append("module_missing_or_ambiguous")
        status = "quarantined"

    symptom = _normalize_text(obj.get("symptom", ""))
    if len(symptom) < 8:
        reasons.append("symptom_too_short")
        status = "quarantined"
    if symptom.lower() in {"issue", "error", "problem"}:
        reasons.append("symptom_too_generic")
        status = "quarantined"

    causes = obj.get("probable_causes", [])
    if not isinstance(causes, list) or not causes:
        reasons.append("cause_structure_missing")
        status = "quarantined"

    resolution_steps = obj.get("resolution_steps", [])
    if not isinstance(resolution_steps, list) or len(resolution_steps) < 1:
        reasons.append("resolution_steps_missing")
        status = "quarantined"

    if not obj.get("citations"):
        reasons.append("citations_missing")
        status = "rejected"

    if not obj.get("provenance_metadata"):
        reasons.append("provenance_missing")
        status = "rejected"

    if obj.get("confidence_score", 0) < 0.72 and status == "accepted":
        reasons.append("low_confidence_requires_cleanup")
        status = "accepted_with_cleanup"

    has_generic_fix = any("contact system administrator" in step.lower() for step in resolution_steps)
    if has_generic_fix and status == "accepted":
        reasons.append("contains_generic_fix_steps")
        status = "accepted_with_cleanup"

    return status, reasons


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _transform_rules_md() -> str:
    return """# Procedure + Troubleshooting Ingestion Transform Rules

## Scope
- This transform pass is limited to high-value Oraclewings sources for `doc-grounded procedure` and `troubleshooting`.
- No runtime behavior changes are applied in this pass.

## Deterministic Transform Rules
### Procedure extraction
1. Detect task title from structured fields first (`task_name`, `scenario`, `process name`), else from section headings.
2. Extract prerequisites from explicit fields first (`prerequisites`), else infer from setup/access lines.
3. Extract ordered steps from:
   - explicit step arrays in JSON,
   - numbered lines (`1.`, `Step 1`) in text,
   - structured CSV detail rows.
4. Preserve original order and numbering.
5. Preserve navigation/UI path strings when present (`Navigate to:`, `A -> B -> C`).
6. Preserve warnings and constraints from parameter or notes fields.
7. Dedupe repeated step blocks by normalized text hash.

### Troubleshooting extraction
1. Detect symptom from `error`, `issue`, or explicit symptom text.
2. Detect causes from explicit `causes` or `meaning`; if missing, synthesize one bounded cause using source evidence.
3. Detect remediation from `solutions` or solution sections.
4. Normalize to strict `symptom -> cause -> fix` object form.
5. Separate generic guidance from actionable fix steps; generic-only entries are downgraded or quarantined.

## Validation Rules
### Procedure
- Reject/quarantine when:
  - module missing/ambiguous,
  - task name generic,
  - fewer than 3 ordered steps,
  - missing provenance or citation anchors.
- `accepted_with_cleanup` when OCR/noise or low confidence is detected but structure is valid.

### Troubleshooting
- Reject/quarantine when:
  - module missing/ambiguous,
  - symptom too generic,
  - missing cause structure,
  - missing resolution steps,
  - missing provenance or citation anchors.
- `accepted_with_cleanup` when structure is valid but confidence is moderate or fixes are too generic.

## Safety and Provenance
- Every object must include:
  - source path,
  - source anchor,
  - source excerpt,
  - transform version metadata.
- Unsupported/noisy/generated assets are rejected or quarantined, never auto-accepted.
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    procedure_schema = _procedure_schema()
    troubleshooting_schema = _troubleshooting_schema()

    _write_json(OUTPUT_DIR / "normalized_procedure_schema.json", procedure_schema)
    _write_json(OUTPUT_DIR / "normalized_troubleshooting_schema.json", troubleshooting_schema)
    (OUTPUT_DIR / "ingestion_transform_rules.md").write_text(_transform_rules_md(), encoding="utf-8")

    parser_map = {
        "ow_hcm_hdl_extract_flow": _parse_hdl_extract_flow,
        "ow_hcm_extract_knowledge_compiled": _parse_extract_knowledge,
        "ow_hcm_extract_docx_csv": _parse_hcm_docx_csv,
        "ow_hcm_extract_info_csv": _parse_hcm_info_csv,
        "ow_scm_troubleshooting_kb": _parse_scm_troubleshooting,
        "ow_scm_e2e_processes_kb": _parse_e2e_processes,
        "ow_scm_functional_kb": _parse_functional_kb,
    }

    all_procedures: List[Dict[str, Any]] = []
    all_troubleshooting: List[Dict[str, Any]] = []
    raw_rejected: List[Dict[str, Any]] = []
    source_mapping_rows: List[Dict[str, Any]] = []

    for source in SOURCE_SPECS:
        parser = parser_map[source.source_id]
        procedures, troubleshooting, rejected = parser(source)
        all_procedures.extend(procedures)
        all_troubleshooting.extend(troubleshooting)
        raw_rejected.extend(rejected)

        source_mapping_rows.append(
            {
                "source_id": source.source_id,
                "source_path": source.source_path,
                "asset_type": source.asset_type,
                "lane_target": source.lane_target,
                "source_format": source.source_format,
                "likely_signal_quality": source.likely_signal_quality,
                "expected_cleanup_needs": source.expected_cleanup_needs,
                "transform_strategy": source.transform_strategy,
                "target_schema_objects": ["procedure", "troubleshooting"],
                "raw_counts": {
                    "procedures_extracted": len(procedures),
                    "troubleshooting_extracted": len(troubleshooting),
                    "rejected_during_transform": len(rejected),
                },
            }
        )

    accepted_rows: List[Dict[str, Any]] = []
    quarantined_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []

    by_type_counter = Counter()
    status_counter = Counter()
    seen_keys = set()

    for obj in sorted(all_procedures + all_troubleshooting, key=lambda x: x.get("object_id", "")):
        obj_type = obj.get("object_type")
        if obj_type == "procedure":
            status, reasons = _validate_procedure(obj)
            dedupe_key = _stable_id("dedupe", [obj.get("module", ""), obj.get("task_name", ""), obj.get("source_path", "")])
        else:
            status, reasons = _validate_troubleshooting(obj)
            dedupe_key = _stable_id("dedupe", [obj.get("module", ""), obj.get("symptom", ""), obj.get("source_path", "")])

        if dedupe_key in seen_keys:
            status = "rejected"
            reasons = reasons + ["duplicate_object"]
        seen_keys.add(dedupe_key)

        obj["dedupe_key"] = dedupe_key
        obj["validation_status"] = status
        obj["validation_reasons"] = reasons

        status_counter[status] += 1
        by_type_counter[obj_type] += 1

        if status in {"accepted", "accepted_with_cleanup"}:
            accepted_rows.append(obj)
        elif status == "quarantined":
            quarantined_rows.append(obj)
        else:
            rejected_rows.append(obj)

    for rr in raw_rejected:
        rejected_rows.append(
            {
                "object_type": "unknown",
                "source_id": rr.get("source_id", ""),
                "source_path": rr.get("source_path", ""),
                "validation_status": "rejected",
                "validation_reasons": [rr.get("reason", "transform_reject")],
                "raw_object": rr.get("raw_object") or rr.get("raw_excerpt", ""),
            }
        )
        status_counter["rejected"] += 1

    accepted_procedure_count = sum(1 for r in accepted_rows if r.get("object_type") == "procedure")
    accepted_troubleshooting_count = sum(1 for r in accepted_rows if r.get("object_type") == "troubleshooting")

    if accepted_procedure_count < 10 or accepted_troubleshooting_count < 10:
        raise RuntimeError(
            f"Insufficient accepted sample objects. procedure={accepted_procedure_count}, troubleshooting={accepted_troubleshooting_count}"
        )

    _write_json(OUTPUT_DIR / "source_to_schema_mapping.json", {"sources": source_mapping_rows})
    _write_jsonl(OUTPUT_DIR / "accepted_sample_objects.jsonl", accepted_rows)
    _write_jsonl(OUTPUT_DIR / "quarantined_objects.jsonl", quarantined_rows)
    _write_jsonl(OUTPUT_DIR / "rejected_objects.jsonl", rejected_rows)

    readiness_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "source_root": str(ORACLEWINGS_DIR),
        "discovery_inputs": {
            "folder_inventory": str(DISCOVERY_DIR / "folder_inventory.json"),
            "ingestion_candidates": str(DISCOVERY_DIR / "oraclewings_ingestion_candidates.json"),
            "risk_report": str(DISCOVERY_DIR / "oraclewings_risk_report.json"),
        },
        "selected_high_value_sources": [s.__dict__ for s in SOURCE_SPECS],
        "object_counts": {
            "total_extracted_before_validation": len(all_procedures) + len(all_troubleshooting),
            "procedure_extracted": len(all_procedures),
            "troubleshooting_extracted": len(all_troubleshooting),
            "accepted": status_counter["accepted"],
            "accepted_with_cleanup": status_counter["accepted_with_cleanup"],
            "quarantined": status_counter["quarantined"],
            "rejected": status_counter["rejected"],
            "accepted_procedure": accepted_procedure_count,
            "accepted_troubleshooting": accepted_troubleshooting_count,
        },
        "validation_status_definitions": {
            "accepted": "schema-valid, provenance retained, high-signal, ready for controlled ingestion",
            "accepted_with_cleanup": "schema-valid with moderate quality/noise requiring minor cleanup before scale ingestion",
            "quarantined": "potentially useful but blocked by ambiguity or missing structural requirements",
            "rejected": "unsupported, low-signal, duplicative, unsafe, or missing provenance/citation anchors",
        },
        "top_ingestion_ready_sources": [
            "backend/orawing_ai/data/sources/hcm/HDL_EXTRACT_FLOW_KNOWLEDGE.json",
            "backend/scm_bot_backend/knowledge_base/troubleshooting_kb.json",
            "backend/scm_bot_backend/knowledge_base/e2e_processes_kb.json",
        ],
        "top_sources_requiring_cleanup": [
            "backend/backend_hcm/HCM_Extract_Oracle_info.csv",
            "backend/backend_hcm/hcm_extracts_docx_converted.csv",
            "backend/backend_hcm/app/data/extract_knowledge.json",
        ],
        "next_batch_recommendation": {
            "batch_1": [
                "Ingest accepted objects from HDL_EXTRACT_FLOW_KNOWLEDGE.json and troubleshooting_kb.json first.",
                "Enable strict module/task filters and preserve source anchors in citation fields.",
            ],
            "batch_2": [
                "Promote accepted_with_cleanup objects from extract_knowledge and functional_kb after de-noise review.",
                "Re-run validation focusing on step completeness and symptom-cause-fix strictness.",
            ],
            "batch_3": [
                "Review quarantined objects manually for module disambiguation and generic symptom hardening.",
                "Reject unresolved items permanently if provenance or structure cannot be guaranteed.",
            ],
        },
    }
    _write_json(OUTPUT_DIR / "ingestion_readiness_summary.json", readiness_summary)

    print(json.dumps(readiness_summary["object_counts"], indent=2))


if __name__ == "__main__":
    main()
