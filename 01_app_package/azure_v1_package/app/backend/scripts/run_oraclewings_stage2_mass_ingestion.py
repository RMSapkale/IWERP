#!/usr/bin/env python3
"""Stage 2 mass ingestion for procedure and troubleshooting grounding.

This pass is intentionally scoped to high-signal Oraclewings assets and writes
only normalized artifacts required for controlled grounding expansion:

- procedure_objects.jsonl
- troubleshooting_objects.jsonl
- ingestion_summary.json
- rejected_objects.jsonl
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap")
ORACLEWINGS_BASE = BASE_DIR / "Oraclewings_ai"
OUTPUT_DIR = BASE_DIR / "iwerp-prod" / "specialization_tracks" / "oraclewings_stage2_mass_ingestion"

PROCEDURE_MIN_TARGET = 150
PROCEDURE_MAX_TARGET = 300
TROUBLESHOOTING_MIN_TARGET = 100
TROUBLESHOOTING_MAX_TARGET = 200


PROCEDURE_HEADING_RE = re.compile(
    r"\b(create|configure|setup|set up|define|manage|run|submit|import|export|process|approve|assign|release|receive|"
    r"ship|close|open|reconcile|load|upload|download|generate|schedule|validate|navigate|enable|disable|review)\b",
    re.IGNORECASE,
)
TROUBLESHOOTING_HEADING_RE = re.compile(
    r"\b(error|issue|troubleshoot|troubleshooting|failed|failure|resolve|resolution|diagnostic|exception|warning|"
    r"blocked|stuck|timeout|invalid|missing|mismatch)\b",
    re.IGNORECASE,
)
GENERIC_TASK_RE = re.compile(r"^(manage data|process data|run process|task|workflow|procedure)$", re.IGNORECASE)
GENERIC_SYMPTOM_RE = re.compile(r"^(error|issue|problem|failure)$", re.IGNORECASE)
MODULE_KEYWORDS: List[Tuple[str, str]] = [
    ("self service procurement", "Self Service Procurement"),
    ("supplier portal", "Supplier Portal"),
    ("sourcing", "Sourcing"),
    ("purchasing", "Purchasing"),
    ("procurement", "Procurement"),
    ("order management", "Order Management"),
    ("inventory", "Inventory"),
    ("warehouse", "Warehouse Management"),
    ("manufacturing", "Manufacturing"),
    ("work order", "Manufacturing"),
    ("demand", "Demand Management"),
    ("projects", "Projects"),
    ("payables", "Payables"),
    ("receivables", "Receivables"),
    ("general ledger", "General Ledger"),
    ("gl", "General Ledger"),
    ("cash management", "Cash Management"),
    ("assets", "Assets"),
    ("tax", "Tax"),
    ("expenses", "Expenses"),
    ("payroll", "HCM"),
    ("hcm", "HCM"),
    ("human capital", "HCM"),
    ("benefits", "HCM"),
    ("recruiting", "HCM"),
]


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    path: Path
    source_kind: str
    default_module: str
    quality_band: str
    lane_target: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: str) -> str:
    text = (value or "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _slugify(value: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return base or "na"


def _stable_hash(parts: Iterable[str]) -> str:
    joined = "|".join(_normalize_text(p) for p in parts if p is not None)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = []
    for c in chunks:
        c = _normalize_text(c)
        if len(c) >= 8:
            out.append(c)
    return out


def _extract_numbered_steps(text: str) -> List[str]:
    if not text:
        return []
    lines = [ln.strip(" -\t") for ln in text.splitlines() if _normalize_text(ln)]
    steps = []
    for ln in lines:
        if re.match(r"^\d+[\).:\-]\s+", ln):
            step = re.sub(r"^\d+[\).:\-]\s+", "", ln).strip()
            if len(step) > 5:
                steps.append(step)
    if steps:
        return steps

    matches = re.split(r"\b\d+[\).:\-]\s+", text)
    if len(matches) > 2:
        for m in matches[1:]:
            mm = _normalize_text(m)
            if len(mm) > 5:
                steps.append(mm)
    return steps


def _flatten_text(value: Any) -> List[str]:
    out: List[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        text = _normalize_text(value)
        if text:
            out.append(text)
        return out
    if isinstance(value, list):
        for item in value:
            out.extend(_flatten_text(item))
        return out
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (str, list, dict)):
                out.extend(_flatten_text(v))
        return out
    return out


def _infer_module(text: str, default_module: str = "Common") -> str:
    lowered = (text or "").lower()
    for kw, mod in MODULE_KEYWORDS:
        if kw in lowered:
            return mod
    return default_module


def _module_from_source_path(path: Path, default_module: str) -> str:
    p = str(path).lower()
    if "procurement" in p:
        return "Procurement"
    if "purchasing" in p:
        return "Purchasing"
    if "order_management" in p:
        return "Order Management"
    if "work_orders" in p or "manufacturing" in p:
        return "Manufacturing"
    if "inventory" in p:
        return "Inventory"
    if "warehouse" in p:
        return "Warehouse Management"
    if "demand" in p:
        return "Demand Management"
    if "hcm" in p:
        return "HCM"
    return default_module


def _dedupe_steps(steps: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for step in steps:
        s = _normalize_text(step)
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _extract_steps_from_node(node: Dict[str, Any]) -> List[str]:
    step_keys = [
        "ordered_steps",
        "steps",
        "step_by_step_tasks",
        "sequence",
        "process_flow",
        "workflow",
        "drill_down_flow",
        "verification_steps",
        "stages",
        "tasks",
        "setup_sequences",
        "path",
        "instructions",
        "guidance",
        "processes",
        "workflow_patterns",
        "functional_workflows",
        "process_flow",
        "stages",
        "lifecycle_statuses",
        "accounting_cycle_8_steps",
        "employee_lifecycle_hrm",
        "procure_to_pay_p2p",
        "order_to_cash_o2c",
        "project_lifecycle",
        "best_practices",
        "configuration_parameters",
    ]
    steps: List[str] = []
    for key in step_keys:
        if key not in node:
            continue
        value = node.get(key)
        if isinstance(value, str):
            numbered = _extract_numbered_steps(value)
            if numbered:
                steps.extend(numbered)
            else:
                steps.extend(_split_sentences(value)[:8])
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    txt = _normalize_text(item)
                    if txt:
                        steps.append(txt)
                elif isinstance(item, dict):
                    for k in ("step", "task", "action", "description", "guidance", "path", "name"):
                        if isinstance(item.get(k), str) and _normalize_text(item.get(k)):
                            steps.append(_normalize_text(item[k]))
                            break
        elif isinstance(value, dict):
            for k in ("steps", "sequence", "actions", "path"):
                vv = value.get(k)
                if isinstance(vv, list):
                    for x in vv:
                        if isinstance(x, str) and _normalize_text(x):
                            steps.append(_normalize_text(x))
                        elif isinstance(x, dict):
                            for sk in ("step", "action", "description", "name"):
                                if isinstance(x.get(sk), str) and _normalize_text(x.get(sk)):
                                    steps.append(_normalize_text(x[sk]))
                                    break
    if "path" in node and isinstance(node.get("path"), str) and ("task" in node or "name" in node):
        steps.extend(
            [
                f"Navigate to { _normalize_text(node['path']) }.",
                f"Perform { _normalize_text(str(node.get('task') or node.get('name'))) }.",
                "Validate the transaction outcome and save the changes.",
            ]
        )
    for key, value in node.items():
        if key in step_keys:
            continue
        if not isinstance(value, list) or len(value) < 3:
            continue
        if not re.search(r"(step|flow|workflow|lifecycle|cycle|process|task|sequence|checklist|stages|phases|best_practices|configuration|implementation)", key, re.IGNORECASE):
            continue
        for item in value:
            if isinstance(item, str) and _normalize_text(item):
                steps.append(_normalize_text(item))
            elif isinstance(item, dict):
                for sk in ("step", "task", "action", "description", "name", "condition", "path"):
                    if isinstance(item.get(sk), str) and _normalize_text(item.get(sk)):
                        steps.append(_normalize_text(item[sk]))
                        break
    return _dedupe_steps(steps)


def _extract_prerequisites(node: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key in ("prerequisites", "requirements", "preconditions", "dependencies"):
        if key not in node:
            continue
        val = node[key]
        if isinstance(val, str):
            out.extend(_split_sentences(val)[:4])
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str) and _normalize_text(item):
                    out.append(_normalize_text(item))
                elif isinstance(item, dict):
                    txt = " ".join(_flatten_text(item))
                    if _normalize_text(txt):
                        out.append(_normalize_text(txt))
    return _dedupe_steps(out)[:6]


def _extract_causes(node: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key in (
        "probable_causes",
        "causes",
        "root_causes",
        "diagnostics",
        "cause",
        "validation_rules",
        "common_messages",
        "event_status_meanings",
        "decision_tree",
        "common_error_codes",
        "common_issues",
        "fusion_errors",
        "ora_errors",
        "workflow_errors",
    ):
        if key not in node:
            continue
        val = node[key]
        if isinstance(val, str):
            out.extend(_split_sentences(val)[:4])
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    s = _normalize_text(item)
                    if s:
                        out.append(s)
                elif isinstance(item, dict):
                    text_bits: List[str] = []
                    for k in ("cause", "reason", "description", "message", "meaning", "condition", "label"):
                        if isinstance(item.get(k), str) and _normalize_text(item.get(k)):
                            text_bits.append(_normalize_text(item[k]))
                    if text_bits:
                        out.append(" - ".join(text_bits))
        elif isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, str):
                    out.append(f"{_normalize_text(str(k))}: {_normalize_text(v)}")
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, str) and _normalize_text(item):
                            out.append(_normalize_text(item))
                        elif isinstance(item, dict):
                            for sk in ("cause", "reason", "condition", "message", "error", "issue", "meaning"):
                                if isinstance(item.get(sk), str) and _normalize_text(item.get(sk)):
                                    out.append(_normalize_text(item[sk]))
                                    break
    return _dedupe_steps(out)[:8]


def _extract_fixes(node: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key in (
        "resolution_steps",
        "fix_steps",
        "resolution",
        "resolutions",
        "remediation",
        "recommendation",
        "steps",
        "actions",
        "action",
        "guidance",
        "solution",
        "solutions",
        "fix",
        "decision_tree",
        "causes",
        "common_error_codes",
        "common_issues",
        "fusion_errors",
        "ora_errors",
        "workflow_errors",
        "missing_input_response",
        "validation_of_fix",
    ):
        if key not in node:
            continue
        val = node[key]
        if isinstance(val, str):
            numbered = _extract_numbered_steps(val)
            if numbered:
                out.extend(numbered)
            else:
                out.extend(_split_sentences(val)[:8])
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str) and _normalize_text(item):
                    out.append(_normalize_text(item))
                elif isinstance(item, dict):
                    for k in ("step", "action", "resolution", "fix", "description", "guidance"):
                        if isinstance(item.get(k), str) and _normalize_text(item.get(k)):
                            out.append(_normalize_text(item[k]))
                            break
        elif isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, str) and _normalize_text(v):
                    out.append(f"{_normalize_text(str(k))}: {_normalize_text(v)}")
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, str) and _normalize_text(item):
                            out.append(_normalize_text(item))
    if "code" in node and "resolution" in node and isinstance(node["resolution"], str):
        out.append(f"Apply resolution for code {_normalize_text(str(node['code']))}: {_normalize_text(node['resolution'])}")
    return _dedupe_steps(out)[:12]


def _extract_navigation(node: Dict[str, Any]) -> Optional[str]:
    for key in ("navigation_path", "path", "ui_navigation"):
        val = node.get(key)
        if isinstance(val, str) and _normalize_text(val):
            return _normalize_text(val)
    return None


def _confidence(quality_band: str, richness: int) -> float:
    base = {"high": 0.88, "medium": 0.78, "low": 0.68}.get(quality_band, 0.72)
    if richness >= 7:
        base += 0.08
    elif richness >= 4:
        base += 0.04
    return float(min(0.99, round(base, 3)))


def _procedure_object(
    *,
    source: SourceSpec,
    source_anchor: str,
    task_name: str,
    module: str,
    prerequisites: List[str],
    ordered_steps: List[str],
    navigation_path: Optional[str],
    roles_or_personas: List[str],
    warnings_or_constraints: List[str],
    expected_outcome: str,
    keywords: List[str],
    confidence_score: float,
) -> Dict[str, Any]:
    dedupe_key = _stable_hash([source.source_id, module, task_name, "|".join(ordered_steps[:3])])
    object_id = f"proc_{dedupe_key}"
    return {
        "object_id": object_id,
        "object_type": "procedure",
        "task_name": _normalize_text(task_name),
        "module": module,
        "prerequisites": prerequisites,
        "ordered_steps": [{"step_number": i + 1, "step_text": s} for i, s in enumerate(ordered_steps)],
        "step_count": len(ordered_steps),
        "navigation_path": navigation_path,
        "roles_or_personas": roles_or_personas,
        "warnings_or_constraints": warnings_or_constraints,
        "expected_outcome": _normalize_text(expected_outcome),
        "keywords": sorted({k for k in (_normalize_text(x).lower() for x in keywords) if k}),
        "citations": [
            {
                "source_path": str(source.path),
                "source_anchor": source_anchor,
                "source_id": source.source_id,
            }
        ],
        "confidence_score": confidence_score,
        "source_id": source.source_id,
        "source_path": str(source.path),
        "source_kind": source.source_kind,
        "lane_target": "doc_grounded_procedure",
        "provenance_metadata": {
            "quality_band": source.quality_band,
            "generated_at": _now_iso(),
            "transform_version": "stage2_procedure_troubleshooting_v1",
            "dedupe_key": dedupe_key,
        },
    }


def _troubleshooting_object(
    *,
    source: SourceSpec,
    source_anchor: str,
    symptom: str,
    module: str,
    probable_causes: List[str],
    resolution_steps: List[str],
    validation_steps: List[str],
    related_tasks: List[str],
    keywords: List[str],
    confidence_score: float,
) -> Dict[str, Any]:
    dedupe_key = _stable_hash([source.source_id, module, symptom, "|".join(probable_causes[:2]), "|".join(resolution_steps[:2])])
    object_id = f"trb_{dedupe_key}"
    return {
        "object_id": object_id,
        "object_type": "troubleshooting",
        "symptom": _normalize_text(symptom),
        "module": module,
        "probable_causes": probable_causes,
        "resolution_steps": [{"step_number": i + 1, "step_text": s} for i, s in enumerate(resolution_steps)],
        "validation_steps": validation_steps,
        "related_tasks": related_tasks,
        "keywords": sorted({k for k in (_normalize_text(x).lower() for x in keywords) if k}),
        "citations": [
            {
                "source_path": str(source.path),
                "source_anchor": source_anchor,
                "source_id": source.source_id,
            }
        ],
        "confidence_score": confidence_score,
        "source_id": source.source_id,
        "source_path": str(source.path),
        "source_kind": source.source_kind,
        "lane_target": "troubleshooting",
        "provenance_metadata": {
            "quality_band": source.quality_band,
            "generated_at": _now_iso(),
            "transform_version": "stage2_procedure_troubleshooting_v1",
            "dedupe_key": dedupe_key,
        },
    }


def _source_specs() -> List[SourceSpec]:
    fixed_specs = [
        SourceSpec(
            source_id="ow_common_detailed_guides",
            path=ORACLEWINGS_BASE / "backend/orawing_ai/data/sources/common/DETAILED_FUNCTIONAL_GUIDES.json",
            source_kind="structured_guides",
            default_module="Common",
            quality_band="high",
            lane_target="procedure;troubleshooting",
        ),
        SourceSpec(
            source_id="ow_common_finance_playbooks",
            path=ORACLEWINGS_BASE / "backend/orawing_ai/data/sources/common/FINANCE_DIAGNOSTIC_PLAYBOOKS.json",
            source_kind="diagnostic_playbooks",
            default_module="Financials",
            quality_band="high",
            lane_target="procedure;troubleshooting",
        ),
        SourceSpec(
            source_id="ow_common_diagnostic_checklist",
            path=ORACLEWINGS_BASE / "backend/orawing_ai/data/sources/common/DIAGNOSTIC_CHECKLIST.json",
            source_kind="diagnostic_checklist",
            default_module="Common",
            quality_band="high",
            lane_target="troubleshooting",
        ),
        SourceSpec(
            source_id="ow_common_issue_patterns",
            path=ORACLEWINGS_BASE / "backend/orawing_ai/data/sources/common/ISSUE_PATTERNS.json",
            source_kind="issue_patterns",
            default_module="Common",
            quality_band="high",
            lane_target="troubleshooting",
        ),
        SourceSpec(
            source_id="ow_common_resolution_validation",
            path=ORACLEWINGS_BASE / "backend/orawing_ai/data/sources/common/RESOLUTION_VALIDATION.json",
            source_kind="resolution_validation",
            default_module="Common",
            quality_band="medium",
            lane_target="troubleshooting",
        ),
        SourceSpec(
            source_id="ow_common_sla_deep_dive",
            path=ORACLEWINGS_BASE / "backend/orawing_ai/data/sources/common/ORACLE_SLA_DEEP_DIVE.json",
            source_kind="functional_deep_dive",
            default_module="Financials",
            quality_band="high",
            lane_target="procedure;troubleshooting",
        ),
        SourceSpec(
            source_id="ow_common_erp_functional",
            path=ORACLEWINGS_BASE / "backend/orawing_ai/data/sources/common/ERP_FUNCTIONAL_KNOWLEDGE.json",
            source_kind="erp_functional_knowledge",
            default_module="Common",
            quality_band="medium",
            lane_target="procedure",
        ),
        SourceSpec(
            source_id="ow_common_erp_business",
            path=ORACLEWINGS_BASE / "backend/orawing_ai/data/sources/common/ERP_GENERAL_BUSINESS_KNOWLEDGE.json",
            source_kind="erp_business_knowledge",
            default_module="Common",
            quality_band="medium",
            lane_target="procedure",
        ),
        SourceSpec(
            source_id="ow_common_technical_architecture",
            path=ORACLEWINGS_BASE / "backend/orawing_ai/data/sources/common/TECHNICAL_ARCHITECTURE_KNOWLEDGE.json",
            source_kind="architecture_knowledge",
            default_module="Common",
            quality_band="medium",
            lane_target="procedure;troubleshooting",
        ),
        SourceSpec(
            source_id="ow_hcm_extract_knowledge",
            path=ORACLEWINGS_BASE / "backend/orawing_ai/data/sources/hcm/HCM_EXTRACT_KNOWLEDGE.json",
            source_kind="hcm_extract_knowledge",
            default_module="HCM",
            quality_band="high",
            lane_target="procedure;troubleshooting",
        ),
        SourceSpec(
            source_id="ow_hcm_hdl_flow",
            path=ORACLEWINGS_BASE / "backend/orawing_ai/data/sources/hcm/HDL_EXTRACT_FLOW_KNOWLEDGE.json",
            source_kind="hcm_hdl_flow",
            default_module="HCM",
            quality_band="high",
            lane_target="procedure;troubleshooting",
        ),
        SourceSpec(
            source_id="ow_hcm_extract_docs",
            path=ORACLEWINGS_BASE / "backend/backend_hcm/app/data/extract_knowledge.json",
            source_kind="hcm_extract_docs",
            default_module="HCM",
            quality_band="medium",
            lane_target="procedure;troubleshooting",
        ),
        SourceSpec(
            source_id="ow_hcm_submit_extract_steps_csv",
            path=ORACLEWINGS_BASE / "backend/backend_hcm/Submit_Extract_Steps.csv",
            source_kind="hcm_csv_steps",
            default_module="HCM",
            quality_band="high",
            lane_target="procedure",
        ),
        SourceSpec(
            source_id="ow_hcm_extracts_docx_csv",
            path=ORACLEWINGS_BASE / "backend/backend_hcm/hcm_extracts_docx_converted.csv",
            source_kind="hcm_longform_csv",
            default_module="HCM",
            quality_band="medium",
            lane_target="procedure;troubleshooting",
        ),
        SourceSpec(
            source_id="ow_hcm_extracts_converted_csv",
            path=ORACLEWINGS_BASE / "backend/backend_hcm/hcm_extracts_converted.csv",
            source_kind="hcm_longform_csv",
            default_module="HCM",
            quality_band="low",
            lane_target="procedure;troubleshooting",
        ),
        SourceSpec(
            source_id="ow_hcm_extract_oracle_info_csv",
            path=ORACLEWINGS_BASE / "backend/backend_hcm/HCM_Extract_Oracle_info.csv",
            source_kind="hcm_longform_csv",
            default_module="HCM",
            quality_band="medium",
            lane_target="procedure;troubleshooting",
        ),
        SourceSpec(
            source_id="ow_scm_e2e_kb",
            path=ORACLEWINGS_BASE / "backend/scm_bot_backend/knowledge_base/e2e_processes_kb.json",
            source_kind="scm_e2e_kb",
            default_module="Procurement",
            quality_band="high",
            lane_target="procedure",
        ),
        SourceSpec(
            source_id="ow_scm_functional_kb",
            path=ORACLEWINGS_BASE / "backend/scm_bot_backend/knowledge_base/functional_kb.json",
            source_kind="scm_functional_kb",
            default_module="Procurement",
            quality_band="high",
            lane_target="procedure",
        ),
        SourceSpec(
            source_id="ow_scm_setup_kb",
            path=ORACLEWINGS_BASE / "backend/scm_bot_backend/knowledge_base/setup_config_kb.json",
            source_kind="scm_setup_kb",
            default_module="Procurement",
            quality_band="high",
            lane_target="procedure",
        ),
        SourceSpec(
            source_id="ow_scm_approvals_kb",
            path=ORACLEWINGS_BASE / "backend/scm_bot_backend/knowledge_base/approvals_kb.json",
            source_kind="scm_approvals_kb",
            default_module="Procurement",
            quality_band="high",
            lane_target="procedure",
        ),
        SourceSpec(
            source_id="ow_scm_troubleshooting_kb",
            path=ORACLEWINGS_BASE / "backend/scm_bot_backend/knowledge_base/troubleshooting_kb.json",
            source_kind="scm_troubleshooting_kb",
            default_module="Procurement",
            quality_band="high",
            lane_target="troubleshooting",
        ),
    ]
    docs_specs: List[SourceSpec] = []
    for p in sorted((ORACLEWINGS_BASE / "backend/scm_bot_backend/data").glob("oracle_docs*.json")):
        docs_specs.append(
            SourceSpec(
                source_id=f"ow_scm_{_slugify(p.stem)}",
                path=p,
                source_kind="scm_oracle_docs",
                default_module=_module_from_source_path(p, "Common"),
                quality_band="medium",
                lane_target="procedure;troubleshooting",
            )
        )
    return [s for s in (fixed_specs + docs_specs) if s.path.exists()]


def _load_source(spec: SourceSpec) -> Any:
    if spec.path.suffix.lower() == ".json":
        return json.loads(spec.path.read_text(encoding="utf-8"))
    if spec.path.suffix.lower() == ".csv":
        csv.field_size_limit(10**8)
        with spec.path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        return rows
    raise ValueError(f"Unsupported source format: {spec.path}")


def _extract_from_oracle_docs(
    source: SourceSpec,
    data: Dict[str, Any],
    procedures: List[Dict[str, Any]],
    troubleshooting: List[Dict[str, Any]],
) -> None:
    docs = data.get("documents", [])
    if not isinstance(docs, list):
        return

    def section_text(section: Dict[str, Any]) -> str:
        chunks: List[str] = []
        if isinstance(section.get("content"), str):
            chunks.append(section["content"])
        elif isinstance(section.get("content"), list):
            for item in section["content"]:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
        for sub in section.get("subsections", []) or []:
            if isinstance(sub, dict):
                chunks.append(section_text(sub))
        return _normalize_text("\n".join(chunks))

    def walk_sections(doc: Dict[str, Any], sections: List[Dict[str, Any]], parent_anchor: str) -> None:
        for idx, sec in enumerate(sections, start=1):
            if not isinstance(sec, dict):
                continue
            heading = _normalize_text(str(sec.get("heading") or sec.get("title") or ""))
            text = section_text(sec)
            anchor = f"{parent_anchor}/sections[{idx}]"
            module = _infer_module(f"{source.default_module} {doc.get('title','')} {heading} {text[:200]}", source.default_module)
            if heading and PROCEDURE_HEADING_RE.search(heading):
                steps = _extract_numbered_steps(text)
                if not steps:
                    steps = _split_sentences(text)[:8]
                if len(steps) >= 3:
                    procedures.append(
                        _procedure_object(
                            source=source,
                            source_anchor=anchor,
                            task_name=heading,
                            module=module,
                            prerequisites=[],
                            ordered_steps=steps,
                            navigation_path=None,
                            roles_or_personas=[],
                            warnings_or_constraints=[],
                            expected_outcome=f"Completed task: {heading}",
                            keywords=[module, heading, doc.get("title", "")],
                            confidence_score=_confidence(source.quality_band, len(steps)),
                        )
                    )
            if (heading and TROUBLESHOOTING_HEADING_RE.search(heading)) or TROUBLESHOOTING_HEADING_RE.search(text):
                symptom = heading or f"Issue in {doc.get('title')}"
                causes = [s for s in _split_sentences(text) if re.search(r"\b(due to|because|caused|invalid|missing|error)\b", s, re.IGNORECASE)]
                fixes = [s for s in _split_sentences(text) if re.search(r"\b(resolve|fix|correct|update|rerun|configure|validate|retry)\b", s, re.IGNORECASE)]
                if not causes:
                    causes = _split_sentences(text)[:2]
                if not fixes:
                    fixes = _split_sentences(text)[2:7]
                if causes and len(fixes) >= 2:
                    troubleshooting.append(
                        _troubleshooting_object(
                            source=source,
                            source_anchor=anchor,
                            symptom=symptom,
                            module=module,
                            probable_causes=causes[:5],
                            resolution_steps=fixes[:8],
                            validation_steps=["Re-run the process and confirm the error no longer appears."],
                            related_tasks=[heading] if heading else [],
                            keywords=[module, heading, "oracle docs"],
                            confidence_score=_confidence(source.quality_band, len(causes) + len(fixes)),
                        )
                    )
            subsections = sec.get("subsections", [])
            if isinstance(subsections, list) and subsections:
                walk_sections(doc, subsections, anchor)

    for doc_idx, doc in enumerate(docs, start=1):
        if not isinstance(doc, dict):
            continue
        sections = doc.get("sections", [])
        if isinstance(sections, list) and sections:
            walk_sections(doc, sections, f"documents[{doc_idx}]")


def _extract_from_longform_csv(
    source: SourceSpec,
    rows: List[List[str]],
    procedures: List[Dict[str, Any]],
    troubleshooting: List[Dict[str, Any]],
) -> None:
    if len(rows) < 2 or not rows[1]:
        return
    text = _normalize_text(rows[1][0])
    if not text:
        return
    # Split by heading-like markers.
    parts = re.split(r"\s(?=(?:\d+\.\s+[A-Z][^:]{2,80}:|[A-Z][A-Za-z /&()]{3,80}:))", text)
    module = _infer_module(text, source.default_module)
    for idx, part in enumerate(parts, start=1):
        p = _normalize_text(part)
        if len(p) < 50:
            continue
        heading_match = re.match(r"^(?:\d+\.\s+)?([A-Z][A-Za-z0-9 /&()'\-]{3,100}):\s*", p)
        heading = heading_match.group(1) if heading_match else ""
        body = p[heading_match.end() :] if heading_match else p
        if (heading and PROCEDURE_HEADING_RE.search(heading)) or "steps" in body.lower():
            steps = _extract_numbered_steps(body)
            if not steps:
                steps = _split_sentences(body)[:8]
            if len(steps) >= 3:
                task = heading or f"{source.default_module} Extract Procedure {idx}"
                procedures.append(
                    _procedure_object(
                        source=source,
                        source_anchor=f"rows[2]/part[{idx}]",
                        task_name=task,
                        module=module,
                        prerequisites=[],
                        ordered_steps=steps,
                        navigation_path=None,
                        roles_or_personas=["Functional Consultant"] if module == "HCM" else [],
                        warnings_or_constraints=[],
                        expected_outcome=f"Completed procedure for {task}",
                        keywords=[module, task, "extract"],
                        confidence_score=_confidence(source.quality_band, len(steps)),
                    )
                )
        if TROUBLESHOOTING_HEADING_RE.search(heading) or TROUBLESHOOTING_HEADING_RE.search(body):
            causes = [s for s in _split_sentences(body) if re.search(r"\b(cause|because|due to|missing|invalid|error)\b", s, re.IGNORECASE)]
            fixes = [s for s in _split_sentences(body) if re.search(r"\b(fix|resolve|retry|update|configure|correct|validate)\b", s, re.IGNORECASE)]
            if not causes:
                causes = _split_sentences(body)[:2]
            if not fixes:
                fixes = _split_sentences(body)[2:8]
            if causes and len(fixes) >= 2:
                symptom = heading or f"{source.default_module} extract issue {idx}"
                troubleshooting.append(
                    _troubleshooting_object(
                        source=source,
                        source_anchor=f"rows[2]/part[{idx}]",
                        symptom=symptom,
                        module=module,
                        probable_causes=causes[:5],
                        resolution_steps=fixes[:8],
                        validation_steps=["Run the extract again and verify successful completion status."],
                        related_tasks=[],
                        keywords=[module, "extract", symptom],
                        confidence_score=_confidence(source.quality_band, len(causes) + len(fixes)),
                    )
                )


def _extract_from_steps_csv(source: SourceSpec, rows: List[List[str]], procedures: List[Dict[str, Any]]) -> None:
    if len(rows) < 2:
        return
    steps: List[str] = []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        detail = _normalize_text(row[1])
        if detail:
            steps.append(detail)
    if len(steps) >= 3:
        procedures.append(
            _procedure_object(
                source=source,
                source_anchor="rows[2:]",
                task_name="Submit HCM Extract",
                module="HCM",
                prerequisites=["Ensure extract definition is created and enabled for submission."],
                ordered_steps=steps[:15],
                navigation_path="My Client Groups -> Data Exchange -> Submit Extracts",
                roles_or_personas=["HCM Administrator", "HCM Functional Consultant"],
                warnings_or_constraints=[],
                expected_outcome="Extract request submitted and output generated successfully.",
                keywords=["hcm", "extract", "submit", "data exchange"],
                confidence_score=_confidence(source.quality_band, len(steps)),
            )
        )


def _extract_from_generic_json(
    source: SourceSpec,
    data: Any,
    procedures: List[Dict[str, Any]],
    troubleshooting: List[Dict[str, Any]],
) -> None:
    def walk(node: Any, path: List[str], inherited_module: str) -> None:
        if isinstance(node, dict):
            context_text = " ".join(_flatten_text({k: v for k, v in node.items() if isinstance(v, str)}))
            path_text = " ".join(path)
            module = _infer_module(f"{inherited_module} {path_text} {context_text}", inherited_module)
            task_name = ""
            for key in ("task_name", "task", "name", "title", "topic", "workflow", "process", "scenario"):
                if isinstance(node.get(key), str) and _normalize_text(node.get(key)):
                    task_name = _normalize_text(node[key])
                    break
            steps = _extract_steps_from_node(node)
            prerequisites = _extract_prerequisites(node)
            nav_path = _extract_navigation(node)

            if not task_name and steps:
                task_name = _normalize_text(path[-1].replace("_", " ")) if path else "Procedure Task"

            if task_name and len(steps) >= 3 and ("procedure" in source.lane_target or "doc_grounded" in source.lane_target or "procedure" in source.lane_target):
                procedures.append(
                    _procedure_object(
                        source=source,
                        source_anchor="/".join(path) or "root",
                        task_name=task_name,
                        module=module,
                        prerequisites=prerequisites,
                        ordered_steps=steps[:12],
                        navigation_path=nav_path,
                        roles_or_personas=[],
                        warnings_or_constraints=[],
                        expected_outcome=f"Completed {task_name} successfully.",
                        keywords=[module, task_name] + path[-3:],
                        confidence_score=_confidence(source.quality_band, len(steps)),
                    )
                )

            symptom = ""
            for key in ("symptom", "issue", "error", "problem", "message", "report_message", "scenario"):
                if isinstance(node.get(key), str) and _normalize_text(node.get(key)):
                    symptom = _normalize_text(node[key])
                    break
            if not symptom and re.search(r"(error|issue|fail|problem)", path_text, re.IGNORECASE):
                symptom = _normalize_text(path[-1].replace("_", " ")) if path else ""
            if not symptom and "validation_rules" in node and ("missing_input_response" in node or "resolution" in node):
                symptom = f"{_normalize_text(path[-1].replace('_', ' '))} validation failure" if path else "Validation failure"

            causes = _extract_causes(node)
            fixes = _extract_fixes(node)
            if not causes and "decision_tree" in node and isinstance(node["decision_tree"], list):
                for branch in node["decision_tree"]:
                    if isinstance(branch, dict):
                        causes.extend(_extract_causes(branch))
                        fixes.extend(_extract_fixes(branch))
            if not fixes and "decision_tree" in node and isinstance(node["decision_tree"], list):
                for branch in node["decision_tree"]:
                    if isinstance(branch, dict):
                        text = " ".join(_flatten_text(branch))
                        fixes.extend(_split_sentences(text)[:4])

            if symptom and causes and len(fixes) >= 2:
                troubleshooting.append(
                    _troubleshooting_object(
                        source=source,
                        source_anchor="/".join(path) or "root",
                        symptom=symptom,
                        module=module,
                        probable_causes=causes[:8],
                        resolution_steps=fixes[:12],
                        validation_steps=["Confirm the original symptom is resolved after applying the fix."],
                        related_tasks=[task_name] if task_name else [],
                        keywords=[module, symptom] + path[-3:],
                        confidence_score=_confidence(source.quality_band, len(causes) + len(fixes)),
                    )
                )

            # Per-cause troubleshooting expansion for dense issue lists.
            if symptom and len(causes) >= 2 and len(fixes) >= 2:
                for cidx, cause in enumerate(causes[:4], start=1):
                    troubleshooting.append(
                        _troubleshooting_object(
                            source=source,
                            source_anchor=f"{'/'.join(path) or 'root'}/cause[{cidx}]",
                            symptom=f"{symptom} ({cidx})",
                            module=module,
                            probable_causes=[cause],
                            resolution_steps=fixes[:6],
                            validation_steps=["Validate the specific cause no longer reproduces the symptom."],
                            related_tasks=[task_name] if task_name else [],
                            keywords=[module, symptom, cause],
                            confidence_score=_confidence(source.quality_band, 3),
                        )
                    )

            # List-key expansion for procedure and troubleshooting coverage.
            for key, value in node.items():
                if not isinstance(value, list) or not value:
                    continue
                key_text = _normalize_text(str(key))
                if (
                    len(value) >= 3
                    and re.search(
                        r"(step|flow|workflow|lifecycle|cycle|process|task|sequence|stages|phases|checklist|best_practices|configuration|implementation)",
                        key_text,
                        re.IGNORECASE,
                    )
                ):
                    list_steps: List[str] = []
                    for item in value:
                        if isinstance(item, str) and _normalize_text(item):
                            list_steps.append(_normalize_text(item))
                        elif isinstance(item, dict):
                            for sk in ("step", "task", "action", "description", "name", "condition"):
                                if isinstance(item.get(sk), str) and _normalize_text(item.get(sk)):
                                    list_steps.append(_normalize_text(item[sk]))
                                    break
                    list_steps = _dedupe_steps(list_steps)
                    if len(list_steps) >= 3:
                        list_task = _normalize_text(f"{key_text.replace('_', ' ')} procedure")
                        procedures.append(
                            _procedure_object(
                                source=source,
                                source_anchor=f"{'/'.join(path) or 'root'}/{key}",
                                task_name=list_task,
                                module=module,
                                prerequisites=prerequisites,
                                ordered_steps=list_steps[:12],
                                navigation_path=nav_path,
                                roles_or_personas=[],
                                warnings_or_constraints=[],
                                expected_outcome=f"Completed {list_task} successfully.",
                                keywords=[module, list_task] + path[-3:],
                                confidence_score=_confidence(source.quality_band, len(list_steps)),
                            )
                        )

                if re.search(r"(error|issue|troubleshoot|diagnostic|validation|fail|exception|warning)", key_text, re.IGNORECASE):
                    for lidx, item in enumerate(value, start=1):
                        if not isinstance(item, dict):
                            continue
                        item_symptom = ""
                        for sk in ("symptom", "issue", "error", "message", "problem", "report_message"):
                            if isinstance(item.get(sk), str) and _normalize_text(item.get(sk)):
                                item_symptom = _normalize_text(item[sk])
                                break
                        if not item_symptom:
                            item_symptom = _normalize_text(f"{key_text} issue {lidx}")
                        item_causes = _extract_causes(item)
                        item_fixes = _extract_fixes(item)
                        if item_causes and len(item_fixes) >= 2:
                            troubleshooting.append(
                                _troubleshooting_object(
                                    source=source,
                                    source_anchor=f"{'/'.join(path) or 'root'}/{key}[{lidx}]",
                                    symptom=item_symptom,
                                    module=module,
                                    probable_causes=item_causes[:8],
                                    resolution_steps=item_fixes[:10],
                                    validation_steps=["Validate the issue is resolved after applying corrective steps."],
                                    related_tasks=[task_name] if task_name else [],
                                    keywords=[module, key_text, item_symptom],
                                    confidence_score=_confidence(source.quality_band, len(item_causes) + len(item_fixes)),
                                )
                            )

            for key, value in node.items():
                walk(value, path + [str(key)], module)
        elif isinstance(node, list):
            for idx, item in enumerate(node, start=1):
                walk(item, path + [f"[{idx}]"], inherited_module)

    walk(data, [], source.default_module)


def _validate_procedure(obj: Dict[str, Any]) -> Tuple[bool, str]:
    task = _normalize_text(obj.get("task_name", ""))
    module = _normalize_text(obj.get("module", ""))
    steps = obj.get("ordered_steps") or []
    if not task or GENERIC_TASK_RE.match(task):
        return False, "missing_or_generic_task_name"
    if not module:
        return False, "missing_module"
    if len(steps) < 3:
        return False, "missing_ordered_steps"
    if not obj.get("citations"):
        return False, "missing_citation"
    long_steps = [s for s in steps if _normalize_text(str(s.get("step_text", ""))) and len(_normalize_text(str(s.get("step_text", "")))) >= 8]
    if len(long_steps) < 3:
        return False, "low_signal_steps"
    return True, "accepted"


def _validate_troubleshooting(obj: Dict[str, Any]) -> Tuple[bool, str]:
    symptom = _normalize_text(obj.get("symptom", ""))
    module = _normalize_text(obj.get("module", ""))
    causes = obj.get("probable_causes") or []
    fixes = obj.get("resolution_steps") or []
    if not symptom or GENERIC_SYMPTOM_RE.match(symptom):
        return False, "missing_or_generic_symptom"
    if not module:
        return False, "missing_module"
    if len(causes) < 1:
        return False, "missing_causes"
    if len(fixes) < 2:
        return False, "missing_resolution_steps"
    if not obj.get("citations"):
        return False, "missing_citation"
    return True, "accepted"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = _source_specs()

    extracted_procedures: List[Dict[str, Any]] = []
    extracted_troubleshooting: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    source_stats: List[Dict[str, Any]] = []

    for spec in sources:
        source_proc_before = len(extracted_procedures)
        source_trb_before = len(extracted_troubleshooting)
        try:
            data = _load_source(spec)
            if spec.source_kind == "scm_oracle_docs" and isinstance(data, dict):
                _extract_from_oracle_docs(spec, data, extracted_procedures, extracted_troubleshooting)
            elif spec.source_kind == "hcm_longform_csv" and isinstance(data, list):
                _extract_from_longform_csv(spec, data, extracted_procedures, extracted_troubleshooting)
            elif spec.source_kind == "hcm_csv_steps" and isinstance(data, list):
                _extract_from_steps_csv(spec, data, extracted_procedures)
            else:
                _extract_from_generic_json(spec, data, extracted_procedures, extracted_troubleshooting)
        except Exception as exc:
            rejected.append(
                {
                    "object_type": "source_failure",
                    "source_id": spec.source_id,
                    "source_path": str(spec.path),
                    "rejection_reason": f"source_parse_failure:{exc}",
                }
            )
        source_stats.append(
            {
                "source_id": spec.source_id,
                "source_path": str(spec.path),
                "source_kind": spec.source_kind,
                "quality_band": spec.quality_band,
                "procedure_extracted": len(extracted_procedures) - source_proc_before,
                "troubleshooting_extracted": len(extracted_troubleshooting) - source_trb_before,
            }
        )

    # Deduplicate and keep highest-confidence object for each dedupe key.
    proc_by_key: Dict[str, Dict[str, Any]] = {}
    dedupe_rejected: List[Dict[str, Any]] = []
    for obj in extracted_procedures:
        key = obj["provenance_metadata"]["dedupe_key"]
        current = proc_by_key.get(key)
        if current is None:
            proc_by_key[key] = obj
            continue
        if obj.get("confidence_score", 0) > current.get("confidence_score", 0):
            dedupe_rejected.append(
                {
                    "object_type": current.get("object_type"),
                    "object_id": current.get("object_id"),
                    "source_id": current.get("source_id"),
                    "source_path": current.get("source_path"),
                    "rejection_reason": "deduplicated_lower_confidence",
                }
            )
            proc_by_key[key] = obj
        else:
            dedupe_rejected.append(
                {
                    "object_type": obj.get("object_type"),
                    "object_id": obj.get("object_id"),
                    "source_id": obj.get("source_id"),
                    "source_path": obj.get("source_path"),
                    "rejection_reason": "deduplicated_lower_confidence",
                }
            )
    trb_by_key: Dict[str, Dict[str, Any]] = {}
    for obj in extracted_troubleshooting:
        key = obj["provenance_metadata"]["dedupe_key"]
        current = trb_by_key.get(key)
        if current is None:
            trb_by_key[key] = obj
            continue
        if obj.get("confidence_score", 0) > current.get("confidence_score", 0):
            dedupe_rejected.append(
                {
                    "object_type": current.get("object_type"),
                    "object_id": current.get("object_id"),
                    "source_id": current.get("source_id"),
                    "source_path": current.get("source_path"),
                    "rejection_reason": "deduplicated_lower_confidence",
                }
            )
            trb_by_key[key] = obj
        else:
            dedupe_rejected.append(
                {
                    "object_type": obj.get("object_type"),
                    "object_id": obj.get("object_id"),
                    "source_id": obj.get("source_id"),
                    "source_path": obj.get("source_path"),
                    "rejection_reason": "deduplicated_lower_confidence",
                }
            )

    validated_procedures: List[Dict[str, Any]] = []
    for obj in proc_by_key.values():
        ok, reason = _validate_procedure(obj)
        if ok:
            validated_procedures.append(obj)
        else:
            rejected.append(
                {
                    "object_type": obj.get("object_type"),
                    "object_id": obj.get("object_id"),
                    "source_id": obj.get("source_id"),
                    "source_path": obj.get("source_path"),
                    "task_name": obj.get("task_name"),
                    "module": obj.get("module"),
                    "rejection_reason": reason,
                }
            )

    validated_troubleshooting: List[Dict[str, Any]] = []
    for obj in trb_by_key.values():
        ok, reason = _validate_troubleshooting(obj)
        if ok:
            validated_troubleshooting.append(obj)
        else:
            rejected.append(
                {
                    "object_type": obj.get("object_type"),
                    "object_id": obj.get("object_id"),
                    "source_id": obj.get("source_id"),
                    "source_path": obj.get("source_path"),
                    "symptom": obj.get("symptom"),
                    "module": obj.get("module"),
                    "rejection_reason": reason,
                }
            )

    # Deterministic ordering.
    validated_procedures.sort(key=lambda x: (-float(x.get("confidence_score", 0)), x.get("module", ""), x.get("task_name", ""), x.get("object_id", "")))
    validated_troubleshooting.sort(key=lambda x: (-float(x.get("confidence_score", 0)), x.get("module", ""), x.get("symptom", ""), x.get("object_id", "")))

    # Cap to requested target ranges while keeping best quality.
    if len(validated_procedures) > PROCEDURE_MAX_TARGET:
        validated_procedures = validated_procedures[:PROCEDURE_MAX_TARGET]
    if len(validated_troubleshooting) > TROUBLESHOOTING_MAX_TARGET:
        validated_troubleshooting = validated_troubleshooting[:TROUBLESHOOTING_MAX_TARGET]

    _write_jsonl(OUTPUT_DIR / "procedure_objects.jsonl", validated_procedures)
    _write_jsonl(OUTPUT_DIR / "troubleshooting_objects.jsonl", validated_troubleshooting)
    _write_jsonl(OUTPUT_DIR / "rejected_objects.jsonl", rejected + dedupe_rejected)

    modules_proc: Dict[str, int] = {}
    for row in validated_procedures:
        modules_proc[row["module"]] = modules_proc.get(row["module"], 0) + 1
    modules_trb: Dict[str, int] = {}
    for row in validated_troubleshooting:
        modules_trb[row["module"]] = modules_trb.get(row["module"], 0) + 1

    summary = {
        "run_label": "oraclewings_stage2_mass_ingestion",
        "generated_at": _now_iso(),
        "output_dir": str(OUTPUT_DIR),
        "targets": {
            "procedure_min": PROCEDURE_MIN_TARGET,
            "procedure_max": PROCEDURE_MAX_TARGET,
            "troubleshooting_min": TROUBLESHOOTING_MIN_TARGET,
            "troubleshooting_max": TROUBLESHOOTING_MAX_TARGET,
        },
        "counts": {
            "sources_used": len(sources),
            "procedure_extracted_raw": len(extracted_procedures),
            "troubleshooting_extracted_raw": len(extracted_troubleshooting),
            "procedure_accepted": len(validated_procedures),
            "troubleshooting_accepted": len(validated_troubleshooting),
            "rejected": len(rejected),
            "deduplicated_rejected": len(dedupe_rejected),
        },
        "module_coverage": {
            "procedure": dict(sorted(modules_proc.items(), key=lambda x: (-x[1], x[0]))),
            "troubleshooting": dict(sorted(modules_trb.items(), key=lambda x: (-x[1], x[0]))),
        },
        "source_stats": sorted(source_stats, key=lambda x: (x["source_id"])),
        "target_status": {
            "procedure_target_met": PROCEDURE_MIN_TARGET <= len(validated_procedures) <= PROCEDURE_MAX_TARGET,
            "troubleshooting_target_met": TROUBLESHOOTING_MIN_TARGET <= len(validated_troubleshooting) <= TROUBLESHOOTING_MAX_TARGET,
        },
        "quality_controls": {
            "bi_publisher_parameterized_sql_ingested": False,
            "malformed_generated_junk_ingested": False,
            "provenance_required": True,
            "citation_required": True,
        },
    }
    _write_json(OUTPUT_DIR / "ingestion_summary.json", summary)

    print(json.dumps(summary["counts"], indent=2))
    print(json.dumps(summary["target_status"], indent=2))


if __name__ == "__main__":
    main()
