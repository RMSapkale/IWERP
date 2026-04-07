from __future__ import annotations

import json
import re
import sqlite3
import ssl
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.ingest.curation import CuratedIngestionValidator, stable_hash
from core.grounding.trusted_registry import get_default_registry
from core.schemas.curation import AuthorityTier, CorpusType, DocType, SourceSystem
from core.schemas.router import TaskType

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATA_ROOT = WORKSPACE_ROOT / "Data_Complete"
ORACLEWINGS_ROOT = WORKSPACE_ROOT / "oracle-fusion-slm"
OUTPUT_ROOT = PROJECT_ROOT / "coverage_expansion"
MANIFEST_ROOT = OUTPUT_ROOT / "manifests"
INDEXES_DIR = PROJECT_ROOT / "backend" / "core" / "retrieval" / "vectors"
DOCS_DB_PATH = INDEXES_DIR / "faiss" / "demo" / "docs_corpus" / "metadata.sqlite"
REAL_MODEL_VALIDATION = PROJECT_ROOT / "real_model_validation" / "real_model_validation_final.jsonl"
TENANT_ID = "demo"
SECOND_WAVE_ROOT = OUTPUT_ROOT / "second_wave"
SECOND_WAVE_MANIFEST_ROOT = SECOND_WAVE_ROOT / "manifests"
SECOND_WAVE_CACHE_ROOT = SECOND_WAVE_ROOT / "oracle_docs_cache"
SECOND_WAVE_CLASSIFICATION_PATH = SECOND_WAVE_ROOT / "repo_source_classification.json"
MIN_CONTENT_WORDS = 90
TARGET_CHUNK_WORDS = 450
MAX_CHUNK_WORDS = 550
OVERLAP_WORDS = 70
OBJECT_TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]{3,}\b")
RELEASE_PATTERN = re.compile(r"\b(?:\d{2}[a-z]|r13-update\d+[a-z])\b", re.IGNORECASE)
ACTIONABLE_VERBS = (
    "create",
    "manage",
    "submit",
    "approve",
    "review",
    "run",
    "import",
    "update",
    "process",
    "define",
    "configure",
    "set up",
    "assign",
    "upload",
    "validate",
    "enroll",
    "hire",
)
WEAK_TITLE_PATTERNS = (
    "videos",
    "bite-size",
    "what's new",
)
BOILERPLATE_MARKERS = (
    "help center home",
    "search help center home",
    "cloud readiness",
    "get started",
    "all books",
    "top tasks",
)
TASK_KEYWORDS = {
    TaskType.TROUBLESHOOTING.value: ("troubleshoot", "error", "issue", "fix", "reservation", "transfer order"),
    TaskType.INTEGRATION.value: ("integrate", "api", "rest", "soap", "fbdi", "endpoint"),
    TaskType.NAVIGATION.value: ("navigate", "navigation", "menu", "work area"),
    TaskType.PROCEDURE.value: ("implement", "configure", "administer", "use", "process", "change", "supplier", "catalog"),
}
MODULE_NORMALIZATION = {
    "Enterprise Resource Planning": "Financials",
    "Financials - General Ledger and Accounting": "General Ledger",
    "Financials - Payables": "Payables",
    "Financials - Receivables": "Receivables",
    "Supply Chain Management": "SCM",
    "Human Capital Management": "HCM",
    "Project Management": "Projects",
    "Applications Common": "Common",
    "Customer Experience": "CRM",
}
SECOND_WAVE_TARGETS: Dict[str, Dict[str, Any]] = {
    "Payables": {
        "target_chunks": 120,
        "topics": {
            "invoice_validation": {
                "keywords": ("invoice validation", "payables invoice", "invoice approval", "invoice processing"),
                "doc_type": DocType.PROCEDURE_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
            "payments": {
                "keywords": ("payment process", "payment method", "payments", "disbursement"),
                "doc_type": DocType.SETUP_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
        },
    },
    "Procurement": {
        "target_chunks": 150,
        "topics": {
            "purchase_order_lifecycle": {
                "keywords": ("purchase order", "purchase orders", "approved supplier", "requisition", "submit requisition"),
                "doc_type": DocType.PROCEDURE_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
            "purchase_order_change_orders": {
                "keywords": ("change order", "change orders", "modify purchase order", "revise order"),
                "doc_type": DocType.PROCEDURE_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
            "supplier_registration_onboarding": {
                "keywords": ("supplier registration", "prospective supplier", "supplier onboarding", "register supplier"),
                "doc_type": DocType.PROCEDURE_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
            "catalog_upload": {
                "keywords": ("catalog", "upload catalog", "procurement catalog", "punchout"),
                "doc_type": DocType.SETUP_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
            "requisition_processing": {
                "keywords": ("requisition", "shopping list", "noncatalog", "checkout", "purchase requisitions"),
                "doc_type": DocType.PROCEDURE_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
            "approval_workflows": {
                "keywords": ("approval", "approval rule", "workflow", "approval task", "document approvals"),
                "doc_type": DocType.SETUP_DOC,
                "task_type": TaskType.TROUBLESHOOTING.value,
            },
        },
    },
    "SCM": {
        "target_chunks": 250,
        "topics": {
            "transfer_orders": {
                "keywords": (
                    "transfer order",
                    "transfer orders",
                    "transfer flow",
                    "internal material transfer",
                    "interorganization transfer",
                    "intraorganization transfer",
                    "movement request",
                    "back-to-back",
                    "supply creation",
                    "intransit",
                    "source organization",
                    "destination organization",
                ),
                "doc_type": DocType.PROCEDURE_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
            "reservations": {
                "keywords": (
                    "reservation",
                    "reservations",
                    "reserve supply",
                    "reserved against",
                    "supply reservation",
                    "available to promise",
                    "promising",
                    "demand line",
                ),
                "doc_type": DocType.TROUBLESHOOTING_DOC,
                "task_type": TaskType.TROUBLESHOOTING.value,
            },
            "inventory_transactions": {
                "keywords": (
                    "inventory transaction",
                    "inventory transactions",
                    "inventory",
                    "subinventory",
                    "transaction interface",
                    "movement requests",
                    "inventory balance",
                    "miscellaneous issue",
                    "miscellaneous receipt",
                    "shipping",
                    "receiving",
                ),
                "doc_type": DocType.PROCEDURE_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
            "order_fulfillment": {
                "keywords": (
                    "order fulfillment",
                    "back-to-back fulfillment",
                    "fulfillment line",
                    "orchestration",
                    "orchestration process",
                    "order management",
                    "order promising",
                    "sales order",
                    "return sales order",
                    "cancel sales order",
                    "ship confirm",
                    "pick release",
                    "drop ship",
                ),
                "doc_type": DocType.PROCEDURE_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
        },
    },
    "HCM": {
        "target_chunks": 250,
        "topics": {
            "employee_lifecycle": {
                "keywords": ("hire employee", "worker lifecycle", "hire to retire", "employment record", "onboard"),
                "doc_type": DocType.PROCEDURE_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
            "recruiting": {
                "keywords": ("recruiting", "candidate", "job requisition", "job offer", "hiring"),
                "doc_type": DocType.SETUP_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
            "benefits": {
                "keywords": ("benefits", "benefit plan", "open enrollment", "benefits processing"),
                "doc_type": DocType.SETUP_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
            "absence_management": {
                "keywords": ("absence management", "absence type", "absence plan", "schedule and record absences"),
                "doc_type": DocType.PROCEDURE_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
            "payroll": {
                "keywords": ("payroll", "payment method", "payroll flow", "payroll process", "global payroll"),
                "doc_type": DocType.SETUP_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
        },
    },
    "Receivables": {
        "target_chunks": 120,
        "topics": {
            "payment_terms": {
                "keywords": ("payment terms", "transaction terms", "receivables", "invoice terms"),
                "doc_type": DocType.SETUP_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
            "invoice_processing": {
                "keywords": ("receivables", "transaction processing", "autoinvoice", "credit memo", "receipt"),
                "doc_type": DocType.PROCEDURE_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
        },
    },
    "General Ledger": {
        "target_chunks": 100,
        "topics": {
            "journal_approval": {
                "keywords": ("journal approval", "approve journals", "approval hierarchy", "journal batch"),
                "doc_type": DocType.SETUP_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
            "general_ledger_setup": {
                "keywords": ("general ledger", "ledger setup", "chart of accounts", "cross-validation", "journal"),
                "doc_type": DocType.SETUP_DOC,
                "task_type": TaskType.PROCEDURE.value,
            },
        },
    },
}
SECOND_WAVE_REMOTE_SOURCES: List[Dict[str, Any]] = [
    {
        "module": "Procurement",
        "title": "Using Procurement",
        "url": "https://docs.oracle.com/en/cloud/saas/procurement/25b/oaprc/using-procurement.pdf",
        "kind": "pdf",
        "source_system": SourceSystem.ORACLE_DOCS.value,
    },
    {
        "module": "SCM",
        "title": "Using Inventory Management",
        "url": "https://docs.oracle.com/en/cloud/saas/supply-chain-and-manufacturing/26a/famml/using-inventory-management.pdf",
        "kind": "pdf",
        "source_system": SourceSystem.ORACLE_DOCS.value,
    },
    {
        "module": "SCM",
        "title": "Using Order Management",
        "url": "https://docs.oracle.com/en/cloud/saas/supply-chain-and-manufacturing/26a/fauom/using-order-management.pdf",
        "kind": "pdf",
        "source_system": SourceSystem.ORACLE_DOCS.value,
    },
    {
        "module": "SCM",
        "title": "Using Order Promising",
        "url": "https://docs.oracle.com/en/cloud/saas/supply-chain-and-manufacturing/26a/fascp/using-order-promising.pdf",
        "kind": "pdf",
        "source_system": SourceSystem.ORACLE_DOCS.value,
    },
    {
        "module": "SCM",
        "title": "Using Supply Chain Orchestration",
        "url": "https://docs.oracle.com/en/cloud/saas/supply-chain-and-manufacturing/25d/fasco/index.html",
        "kind": "index_with_pdf",
        "source_system": SourceSystem.ORACLE_DOCS.value,
    },
    {
        "module": "SCM",
        "title": "Implementing Manufacturing and Supply Chain Materials Management",
        "url": "https://docs.oracle.com/en/cloud/saas/supply-chain-and-manufacturing/25b/faims/implementing-manufacturing-and-supply-chain-materials-management.pdf",
        "kind": "pdf",
        "source_system": SourceSystem.ORACLE_DOCS.value,
    },
    {
        "module": "HCM",
        "title": "Using Global Human Resources",
        "url": "https://docs.oracle.com/en/cloud/saas/human-resources/fawhr/index.html",
        "kind": "index_with_pdf",
        "source_system": SourceSystem.ORACLE_DOCS.value,
    },
    {
        "module": "HCM",
        "title": "Implementing Benefits",
        "url": "https://docs.oracle.com/en/cloud/saas/human-resources/faibf/index.html",
        "kind": "index_with_pdf",
        "source_system": SourceSystem.ORACLE_DOCS.value,
    },
    {
        "module": "HCM",
        "title": "Implementing Absence Management",
        "url": "https://docs.oracle.com/en/cloud/saas/human-resources/faiam/index.html",
        "kind": "index_with_pdf",
        "source_system": SourceSystem.ORACLE_DOCS.value,
    },
    {
        "module": "HCM",
        "title": "Using Absence Management",
        "url": "https://docs.oracle.com/en/cloud/saas/human-resources/fauam/index.html",
        "kind": "index_with_pdf",
        "source_system": SourceSystem.ORACLE_DOCS.value,
    },
    {
        "module": "HCM",
        "title": "Implementing Global Payroll",
        "url": "https://docs.oracle.com/en/cloud/saas/human-resources/faigp/index.html",
        "kind": "index_with_pdf",
        "source_system": SourceSystem.ORACLE_DOCS.value,
    },
    {
        "module": "HCM",
        "title": "Using Global Payroll for Employees",
        "url": "https://docs.oracle.com/en/cloud/saas/human-resources/oapay/index.html",
        "kind": "index_with_pdf",
        "source_system": SourceSystem.ORACLE_DOCS.value,
    },
    {
        "module": "HCM",
        "title": "Overview of Implementing Recruiting",
        "url": "https://docs.oracle.com/en/cloud/saas/talent-management/24c/faimh/overview-of-implementing-recruiting.html",
        "kind": "html",
        "source_system": SourceSystem.ORACLE_DOCS.value,
    },
]


@dataclass(frozen=True)
class SourceSpec:
    module: str
    root: Path
    pattern: str
    kind: str
    source_system: SourceSystem
    authority_tier: AuthorityTier
    priority: int
    quality_score: float
    max_matches: int = 1
    include_oracle_signal_only: bool = False


OFFICIAL_SOURCE_SPECS: List[SourceSpec] = [
    SourceSpec("Procurement", DATA_ROOT / "Procurement" / "txts", "https_docs.oracle.com_en_cloud_saas_procurement_25d_implement*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 120, 0.95),
    SourceSpec("Procurement", DATA_ROOT / "Procurement" / "txts", "https_docs.oracle.com_en_cloud_saas_procurement_25d_configure*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 118, 0.95),
    SourceSpec("Procurement", DATA_ROOT / "Procurement" / "txts", "https_docs.oracle.com_en_cloud_saas_procurement_25d_use*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 116, 0.95),
    SourceSpec("Procurement", DATA_ROOT / "Procurement" / "txts", "https_docs.oracle.com_en_cloud_saas_procurement_25d_administer*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 114, 0.94),
    SourceSpec("Procurement", DATA_ROOT / "Procurement" / "txts", "https_docs.oracle.com_en_cloud_saas_procurement_25d_secure*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 112, 0.93),
    SourceSpec("Procurement", DATA_ROOT / "Procurement" / "txts", "https_docs.oracle.com_en_cloud_saas_procurement_25d_integrate*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 110, 0.93),
    SourceSpec("Procurement", DATA_ROOT / "Procurement" / "txts", "https_docs.oracle.com_en_cloud_saas_procurement_25d_analyze-and-report*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 90, 0.9),
    SourceSpec("Procurement", DATA_ROOT / "Procurement" / "txts", "https_docs.oracle.com_en_cloud_saas_procurement_25d_oadpr_index*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 88, 0.9),
    SourceSpec("Procurement", DATA_ROOT / "Procurement" / "txts", "https_docs.oracle.com_en_cloud_saas_procurement_25d_faepp_index*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 86, 0.9),
    SourceSpec("Procurement", DATA_ROOT / "Procurement" / "jsons", "procurement_*.json", "oracle_json_record", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 40, 0.86, max_matches=2, include_oracle_signal_only=True),
    SourceSpec("SCM", DATA_ROOT / "SCM" / "jsons", "oracle_docs_inventory.json", "oracle_bundle_json", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 126, 0.94),
    SourceSpec("SCM", DATA_ROOT / "SCM" / "jsons", "oracle_docs_order_management.json", "oracle_bundle_json", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 124, 0.94),
    SourceSpec("SCM", DATA_ROOT / "SCM" / "jsons", "oracle_docs_work_orders.json", "oracle_bundle_json", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 122, 0.94),
    SourceSpec("SCM", DATA_ROOT / "SCM" / "jsons", "oracle_docs_purchasing.json", "oracle_bundle_json", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 120, 0.94),
    SourceSpec("SCM", DATA_ROOT / "SCM" / "jsons", "SCM_ORACLE_WEB_DATA.json", "oracle_web_data", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 118, 0.92),
    SourceSpec("SCM", DATA_ROOT / "SCM" / "jsons", "oracle_docs_manufacturing_25c.json", "oracle_bundle_json", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 116, 0.92),
    SourceSpec("SCM", DATA_ROOT / "SCM" / "jsons", "oracle_docs_demand.json", "oracle_bundle_json", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 92, 0.93),
    SourceSpec("SCM", DATA_ROOT / "SCM" / "jsons", "oracle_docs_demand_26a.json", "oracle_bundle_json", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 91, 0.93),
    SourceSpec("HCM", DATA_ROOT / "HCM" / "txts", "https_docs.oracle.com_en_cloud_saas_human-resources_implement*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 120, 0.95),
    SourceSpec("HCM", DATA_ROOT / "HCM" / "txts", "https_docs.oracle.com_en_cloud_saas_human-resources_configure*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 118, 0.95),
    SourceSpec("HCM", DATA_ROOT / "HCM" / "txts", "https_docs.oracle.com_en_cloud_saas_human-resources_use*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 116, 0.95),
    SourceSpec("HCM", DATA_ROOT / "HCM" / "txts", "https_docs.oracle.com_en_cloud_saas_human-resources_administer*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 114, 0.94),
    SourceSpec("HCM", DATA_ROOT / "HCM" / "txts", "https_docs.oracle.com_en_cloud_saas_human-resources_integrate*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 112, 0.94),
    SourceSpec("HCM", DATA_ROOT / "HCM" / "txts", "https_docs.oracle.com_en_cloud_saas_human-resources_secure*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 110, 0.93),
    SourceSpec("HCM", DATA_ROOT / "HCM" / "jsons", "HCM_ORACLE_WEB_DATA.json", "oracle_web_data", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 108, 0.91),
    SourceSpec("HCM", DATA_ROOT / "HCM" / "txts", "*_recr-*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 106, 0.9),
    SourceSpec("HCM", DATA_ROOT / "HCM" / "txts", "*_benf-*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 105, 0.9),
    SourceSpec("HCM", DATA_ROOT / "HCM" / "txts", "*_amg-*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 104, 0.9),
    SourceSpec("HCM", DATA_ROOT / "HCM" / "txts", "*_opma-*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 103, 0.9),
    SourceSpec("HCM", DATA_ROOT / "HCM" / "txts", "*_tama-*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 102, 0.9),
    SourceSpec("HCM", DATA_ROOT / "HCM" / "txts", "https_docs.oracle.com_en_cloud_saas_human-resources_books*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 72, 0.89),
    SourceSpec("HCM", DATA_ROOT / "HCM" / "txts", "https_docs.oracle.com_en_cloud_saas_human-resources_index*.txt", "oracle_txt", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 70, 0.89),
    SourceSpec("Payables", DATA_ROOT / "Finance" / "pdfs", "implementing-payables-invoice-to-pay.pdf", "oracle_pdf", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 120, 0.96),
    SourceSpec("Payables", DATA_ROOT / "Finance" / "pdfs", "using-payables-invoice-to-pay.pdf", "oracle_pdf", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 118, 0.96),
    SourceSpec("Receivables", DATA_ROOT / "Finance" / "pdfs", "implementing-receivables-credit-to-cash.pdf", "oracle_pdf", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 120, 0.96),
    SourceSpec("Receivables", DATA_ROOT / "Finance" / "pdfs", "using-receivables-credit-to-cash.pdf", "oracle_pdf", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 118, 0.96),
    SourceSpec("General Ledger", DATA_ROOT / "Finance" / "pdfs", "using-general-ledger.pdf", "oracle_pdf", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 116, 0.95),
    SourceSpec("Tax", DATA_ROOT / "Finance" / "pdfs", "using-tax.pdf", "oracle_pdf", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 108, 0.93),
    SourceSpec("General Ledger", DATA_ROOT / "Finance" / "pdfs", "using-subledger-accounting.pdf", "oracle_pdf", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 106, 0.93),
    SourceSpec("Financials", DATA_ROOT / "Finance" / "jsons", "ERP_ORACLE_WEB_DATA.json", "oracle_web_data", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 100, 0.92),
    SourceSpec("Common", DATA_ROOT / "Finance" / "jsons", "COMMON_APPS_ORACLE_WEB_DATA.json", "oracle_web_data", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 90, 0.9),
    SourceSpec("Financials", ORACLEWINGS_ROOT / "data" / "source_raw" / "Fin_Functional_Docs", "documentation_data.json", "documentation_list", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 89, 0.9),
    SourceSpec("Common", ORACLEWINGS_ROOT / "data" / "source_raw" / "Common_Architecture", "COMMON_APPS_ORACLE_WEB_DATA.json", "oracle_web_data", SourceSystem.ORACLE_DOCS, AuthorityTier.OFFICIAL, 88, 0.9),
]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_module(value: str) -> str:
    cleaned = normalize_text(value)
    return MODULE_NORMALIZATION.get(cleaned, cleaned or "Common")


def normalized_title(value: str) -> str:
    return normalize_text(value).lower()


def word_count(value: str) -> int:
    return len(re.findall(r"\b\w+\b", value or ""))


def extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:  # pragma: no cover - environment fallback
        try:
            from PyPDF2 import PdfReader
        except ModuleNotFoundError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("PDF ingestion requires pypdf or PyPDF2 to be installed.") from exc
    reader = PdfReader(str(path))
    return "\n".join(filter(None, (page.extract_text() for page in reader.pages))).strip()


def extract_trusted_objects(content: str, registry: Any) -> List[str]:
    return sorted({token for token in OBJECT_TOKEN_PATTERN.findall((content or "").upper()) if registry.has_object(token)})[:25]


def parse_release(value: str) -> Optional[str]:
    match = RELEASE_PATTERN.search(str(value or ""))
    return match.group(0).lower() if match else None


def release_sort_key(value: Optional[str]) -> Tuple[int, int]:
    if not value:
        return (0, 0)
    match = re.match(r"(\d{2})([a-z])", value)
    if match:
        return (int(match.group(1)), ord(match.group(2)) - ord("a") + 1)
    match = re.search(r"update(\d+)([a-z])", value)
    if match:
        return (int(match.group(1)), ord(match.group(2)) - ord("a") + 1)
    return (0, 0)


def infer_task_and_doc_type(text_hint: str) -> Tuple[str, DocType]:
    lowered = str(text_hint or "").lower()
    if any(token in lowered for token in TASK_KEYWORDS[TaskType.TROUBLESHOOTING.value]):
        return TaskType.TROUBLESHOOTING.value, DocType.TROUBLESHOOTING_DOC
    if any(token in lowered for token in TASK_KEYWORDS[TaskType.INTEGRATION.value]):
        return TaskType.INTEGRATION.value, DocType.FUNCTIONAL_DOC
    if any(token in lowered for token in ("navigate", "path", "menu")):
        return TaskType.NAVIGATION.value, DocType.NAVIGATION_DOC
    if any(token in lowered for token in ("implement", "configure", "administer", "secure", "setup")):
        return TaskType.PROCEDURE.value, DocType.SETUP_DOC
    if any(token in lowered for token in ("use", "process", "task", "catalog", "supplier", "change", "reservation", "transfer")):
        return TaskType.PROCEDURE.value, DocType.PROCEDURE_DOC
    return TaskType.GENERAL.value, DocType.FUNCTIONAL_DOC


def source_uri_from_path(path: Path) -> str:
    if path.name.startswith("https_docs.oracle.com_"):
        suffix = path.name[len("https_docs.oracle.com_") :]
        suffix = suffix.removesuffix(".txt")
        suffix = suffix.replace("_", "/")
        return f"https://docs.oracle.com/{suffix}"
    return str(path)


def safe_filename(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return normalized or "oracle_doc"


def official_oracle_source(source_uri: str) -> bool:
    return "docs.oracle.com" in str(source_uri or "").lower()


def is_index_like_source(title: str, source_uri: str) -> bool:
    joined = f"{title} {source_uri}".lower()
    return any(
        token in joined
        for token in (
            "index.html",
            "books.html",
            "videos.html",
            "get started",
            "what's new",
            "all books",
            "analyze-and-report",
        )
    )


def contains_actionable_signal(text: str) -> bool:
    lowered = str(text or "").lower()
    action_hits = sum(1 for verb in ACTIONABLE_VERBS if verb in lowered)
    return action_hits >= 1 or bool(re.search(r"\b(step|steps|task|tasks)\b", lowered))


def flatten_topic_profiles(module: str) -> List[Tuple[str, Dict[str, Any]]]:
    return list((SECOND_WAVE_TARGETS.get(module, {}).get("topics") or {}).items())


def score_target_block(module: str, title: str, content: str) -> Optional[Dict[str, Any]]:
    if module not in SECOND_WAVE_TARGETS:
        return None

    lowered_title = normalize_text(title).lower()
    lowered_content = normalize_text(content).lower()
    best: Optional[Dict[str, Any]] = None
    for topic_name, topic in flatten_topic_profiles(module):
        keywords = topic.get("keywords") or ()
        title_hits = sum(1 for keyword in keywords if keyword in lowered_title)
        content_hits = sum(1 for keyword in keywords if keyword in lowered_content)
        action_hits = sum(1 for verb in ACTIONABLE_VERBS if verb in lowered_content)
        score = (title_hits * 4) + (content_hits * 2) + min(action_hits, 3)
        if re.search(r"(^|\n)\s*(\d+\.|-)\s+\S+", content):
            score += 1
        if is_index_like_source(title, ""):
            score -= 2
        if "overview" in lowered_title and title_hits == 0:
            score -= 2
        if "what's new" in lowered_title or "readiness" in lowered_title:
            score -= 3
        if score < 4:
            continue
        if content_hits == 0 and title_hits == 0:
            continue
        if not contains_actionable_signal(content) and topic["task_type"] != TaskType.TROUBLESHOOTING.value:
            continue

        candidate = {
            "topic": topic_name,
            "score": score,
            "task_type": topic["task_type"],
            "doc_type": topic["doc_type"],
            "matched_keywords": [keyword for keyword in keywords if keyword in lowered_title or keyword in lowered_content],
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


def infer_financials_leaf_module(title: str, source_uri: str, content: str) -> Optional[str]:
    joined = f"{title}\n{source_uri}\n{content}".lower()
    if "payables" in joined or "invoice validation" in joined or "payments" in joined:
        return "Payables"
    if "receivables" in joined or "credit-to-cash" in joined or "payment terms" in joined or "autoinvoice" in joined:
        return "Receivables"
    if "general ledger" in joined or "journal approval" in joined or "ledger" in joined or "chart of accounts" in joined:
        return "General Ledger"
    if "tax" in joined:
        return "Tax"
    return None


def chunk_targeted_content(module: str, title: str, content: str) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunk_document_text(content) or [normalize_text(content)]):
        if word_count(chunk) < MIN_CONTENT_WORDS:
            continue
        scored = score_target_block(module, title, chunk)
        if not scored:
            continue
        documents.append(
            {
                "chunk_index": index,
                "content": chunk,
                "score": scored["score"],
                "task_type": scored["task_type"],
                "doc_type": scored["doc_type"],
                "matched_keywords": scored["matched_keywords"],
                "topic": scored["topic"],
            }
        )
    return documents


def iter_section_blocks(sections: List[Dict[str, Any]], ancestors: Optional[List[str]] = None) -> Iterable[Tuple[str, str]]:
    lineage = list(ancestors or [])
    for section in sections or []:
        heading = normalize_text(section.get("heading") or "")
        current_lineage = [item for item in (*lineage, heading) if item]
        lines: List[str] = []
        for block in section.get("content", []) or []:
            if block.get("type") == "paragraph":
                paragraph = normalize_text(block.get("text") or "")
                if paragraph:
                    lines.append(paragraph)
            elif block.get("type") == "list":
                for item in block.get("items", []) or []:
                    item_text = normalize_text(item)
                    if item_text:
                        lines.append(f"- {item_text}")
        text = "\n".join(lines).strip()
        if text:
            yield " > ".join(current_lineage) or heading or "Oracle Guide Section", text
        yield from iter_section_blocks(section.get("subsections", []) or [], current_lineage)


def extract_html_sections(html_content: str) -> List[Tuple[str, str]]:
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ModuleNotFoundError:
        text = re.sub(r"<[^>]+>", " ", html_content or "")
        normalized = normalize_text(text)
        return [("Oracle Guide", normalized)] if normalized else []

    soup = BeautifulSoup(html_content or "", "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    title_tag = soup.find("title")
    current_heading = normalize_text(title_tag.get_text(" ", strip=True) if title_tag else "Oracle Guide")
    lines: List[str] = []
    sections: List[Tuple[str, str]] = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = normalize_text(element.get_text(" ", strip=True))
        if not text:
            continue
        if element.name in {"h1", "h2", "h3", "h4"}:
            if lines:
                sections.append((current_heading, "\n".join(lines)))
                lines = []
            current_heading = text
            continue
        lines.append(text)
    if lines:
        sections.append((current_heading, "\n".join(lines)))
    return sections


def build_targeted_documents(
    *,
    module: str,
    title: str,
    source_path: str,
    source_uri: str,
    content_blocks: Sequence[Tuple[str, str]],
    quality_score: float,
    source_system: SourceSystem,
    authority_tier: AuthorityTier,
    registry: Any,
) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    for heading, block_text in content_blocks:
        combined_title = normalize_text(f"{title} - {heading}" if heading and heading != title else title)
        for chunk in chunk_targeted_content(module, combined_title, block_text):
            chunk_title = normalize_text(
                f"{combined_title} [{chunk['topic'].replace('_', ' ')} {chunk['chunk_index'] + 1}]"
            )
            content_hash = stable_hash("docs_corpus", normalize_text(chunk["content"]))
            documents.append(
                {
                    "source_path": source_path,
                    "source_uri": source_uri,
                    "canonical_uri": source_uri,
                    "title": chunk_title,
                    "module": module,
                    "task_type": chunk["task_type"],
                    "doc_type": chunk["doc_type"],
                    "quality_score": quality_score,
                    "content": chunk["content"],
                    "content_hash": content_hash,
                    "source_system": source_system,
                    "authority_tier": authority_tier,
                    "doc_release": parse_release(source_uri) or parse_release(source_path),
                    "trusted_schema_objects": extract_trusted_objects(chunk["content"], registry),
                    "metadata": {
                        "topic": chunk["topic"],
                        "matched_keywords": chunk["matched_keywords"],
                        "normalized_title": normalized_title(chunk_title),
                    },
                }
            )
    return documents


def classify_repo_source(path: Path) -> str:
    normalized = str(path).lower()
    name = path.name.lower()
    if "oracle_docs.jsonl" in normalized or "community_data" in normalized or "eval_" in normalized or "train" in normalized:
        return CorpusType.REJECT.value
    if name == "documentation_data.json":
        return SourceSystem.ORACLE_DOCS.value
    if name.endswith("_oracle_web_data.json"):
        return SourceSystem.ORACLEWINGS_REPO.value
    if name in {
        "detailed_functional_guides.json",
        "oracle_end_to_end_business_flows.json",
        "integration_reporting_oracle_web_data.json",
    }:
        return SourceSystem.ORACLEWINGS_REPO.value
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")[:5000].lower()
    except Exception:
        return CorpusType.REJECT.value
    if "docs.oracle.com" in text:
        return SourceSystem.ORACLE_DOCS.value
    if "oracle fusion" in text or "oracle cloud" in text:
        return SourceSystem.ORACLEWINGS_REPO.value
    return CorpusType.REJECT.value


def repo_source_classification() -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    source_root = ORACLEWINGS_ROOT / "data" / "source_raw"
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".docx", ".txt"}:
            continue
        candidates.append(
            {
                "path": str(path),
                "classification": classify_repo_source(path),
            }
        )
    return candidates


def fetch_bytes(url: str) -> Tuple[bytes, str]:
    context = ssl._create_unverified_context()
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=45, context=context) as response:
        return response.read(), str(response.headers.get("Content-Type") or "")


def cache_remote_document(url: str, suffix_hint: str = "") -> Path:
    SECOND_WAVE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = suffix_hint or Path(urllib.parse.urlparse(url).path).suffix or ".html"
    cache_name = f"{safe_filename(Path(urllib.parse.urlparse(url).path).stem)}_{stable_hash(url)[:12]}{suffix}"
    cache_path = SECOND_WAVE_CACHE_ROOT / cache_name
    if cache_path.exists():
        return cache_path
    payload, _ = fetch_bytes(url)
    cache_path.write_bytes(payload)
    return cache_path


def resolve_index_with_pdf(url: str) -> Tuple[str, Path]:
    html_path = cache_remote_document(url, suffix_hint=".html")
    html_content = html_path.read_text(encoding="utf-8", errors="ignore")
    pdf_links = sorted(set(re.findall(r'href=["\']([^"\']+\.pdf)["\']', html_content, flags=re.IGNORECASE)))
    if pdf_links:
        pdf_url = urllib.parse.urljoin(url, pdf_links[0])
        return pdf_url, cache_remote_document(pdf_url, suffix_hint=".pdf")
    return url, html_path


def remote_source_to_blocks(source: Dict[str, Any]) -> Tuple[str, str, List[Tuple[str, str]], str]:
    url = str(source["url"])
    kind = str(source["kind"])
    resolved_url = url
    cache_path: Path
    if kind == "index_with_pdf":
        resolved_url, cache_path = resolve_index_with_pdf(url)
    else:
        suffix = ".pdf" if kind == "pdf" else ".html"
        cache_path = cache_remote_document(url, suffix_hint=suffix)

    if cache_path.suffix.lower() == ".pdf":
        text = extract_pdf_text(cache_path)
        return str(cache_path), resolved_url, [(source["title"], text)], str(source["title"])

    html_content = cache_path.read_text(encoding="utf-8", errors="ignore")
    sections = extract_html_sections(html_content)
    return str(cache_path), resolved_url, sections, str(source["title"])


def dynamic_scm_remote_sources(limit: int = 24) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    source_paths = sorted((DATA_ROOT / "SCM" / "jsons").glob("oracle_docs_*.json"))
    seen: set[str] = set()
    for path in source_paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        for document in payload.get("documents", []) or []:
            url = str(document.get("url") or "")
            title = normalize_text(document.get("title") or "")
            if not official_oracle_source(url) or is_index_like_source(title, url) or url in seen:
                continue
            keywords_text = " ".join(document.get("keywords") or [])
            scored = score_target_block("SCM", title, f"{title}\n{keywords_text}")
            if not scored:
                continue
            priority = scored["score"] + len(document.get("sections") or [])
            candidates.append(
                {
                    "module": "SCM",
                    "title": title,
                    "url": url,
                    "kind": "html",
                    "source_system": SourceSystem.ORACLE_DOCS.value,
                    "priority": priority,
                }
            )
            seen.add(url)
    candidates.sort(key=lambda item: (item["priority"], item["title"]), reverse=True)
    return candidates[:limit]


def title_from_text_export(path: Path, text: str) -> str:
    first_line = normalize_text((text or "").splitlines()[0] if text else "")
    if "Go to main content" in first_line:
        first_line = normalize_text(first_line.split("Go to main content", 1)[0])
    if first_line and len(first_line.split()) >= 3:
        return first_line[:180]
    stem = path.stem.replace("https_docs.oracle.com_", "").replace("_", " ")
    return normalize_text(stem.title())


def is_oracle_signal(url_or_source: str, title: str) -> bool:
    joined = f"{url_or_source} {title}".lower()
    return "docs.oracle.com" in joined or joined.startswith("oracle ")


def is_weak_boilerplate(title: str, content: str) -> bool:
    lowered_title = normalized_title(title)
    lowered = normalize_text(content).lower()
    if any(token in lowered_title for token in WEAK_TITLE_PATTERNS):
        return True
    if word_count(content) < MIN_CONTENT_WORDS:
        return True
    marker_hits = sum(1 for marker in BOILERPLATE_MARKERS if marker in lowered)
    task_hits = sum(1 for words in TASK_KEYWORDS.values() for token in words if token in lowered)
    if marker_hits >= 3 and task_hits <= 2:
        return True
    return False


def flatten_section_nodes(sections: List[Dict[str, Any]], lines: List[str]) -> None:
    for section in sections or []:
        heading = normalize_text(section.get("heading") or "")
        if heading:
            lines.append(heading)
        for block in section.get("content", []) or []:
            if block.get("type") == "paragraph":
                paragraph = normalize_text(block.get("text") or "")
                if paragraph:
                    lines.append(paragraph)
            elif block.get("type") == "list":
                for item in block.get("items", []) or []:
                    item_text = normalize_text(item)
                    if item_text:
                        lines.append(f"- {item_text}")
        flatten_section_nodes(section.get("subsections", []) or [], lines)


def split_paragraphs(text: str) -> List[str]:
    raw_parts = re.split(r"\n\s*\n+", (text or "").replace("\r", "\n"))
    paragraphs: List[str] = []
    for raw in raw_parts:
        normalized = normalize_text(raw)
        if not normalized:
            continue
        if word_count(normalized) > MAX_CHUNK_WORDS * 2:
            sentences = re.split(r"(?<=[.!?])\s+", normalized)
            current = []
            current_words = 0
            for sentence in sentences:
                count = word_count(sentence)
                if current and current_words + count > MAX_CHUNK_WORDS:
                    paragraphs.append(" ".join(current))
                    current = [sentence]
                    current_words = count
                else:
                    current.append(sentence)
                    current_words += count
            if current:
                paragraphs.append(" ".join(current))
            continue
        paragraphs.append(normalized)
    return paragraphs


def chunk_document_text(text: str) -> List[str]:
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0
    for paragraph in paragraphs:
        paragraph_words = word_count(paragraph)
        if current and current_words + paragraph_words > MAX_CHUNK_WORDS:
            chunk_text = "\n\n".join(current).strip()
            if word_count(chunk_text) >= MIN_CONTENT_WORDS:
                chunks.append(chunk_text)
            overlap = " ".join(chunk_text.split()[-OVERLAP_WORDS:])
            current = [overlap, paragraph] if overlap else [paragraph]
            current_words = word_count(" ".join(current))
            continue
        current.append(paragraph)
        current_words += paragraph_words

    if current:
        chunk_text = "\n\n".join(current).strip()
        if word_count(chunk_text) >= MIN_CONTENT_WORDS:
            chunks.append(chunk_text)
    return chunks


def render_web_data_docs(path: Path, module: str, payload: List[Dict[str, Any]], quality_score: float) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    for entry in payload:
        raw_module = normalize_module(entry.get("module") or module)
        sub_module = normalize_text(entry.get("sub_module") or raw_module)
        source_uri = entry.get("url") or entry.get("source_uri") or str(path)

        grouped_sections = [
            (
                f"{sub_module} Overview",
                TaskType.GENERAL.value,
                DocType.FUNCTIONAL_DOC,
                ("business_purpose", "functional_concepts", "integration_points", "security_controls", "notes"),
            ),
            (
                f"{sub_module} Setup and Process",
                TaskType.PROCEDURE.value,
                DocType.PROCEDURE_DOC,
                ("key_processes", "setup_concepts", "dependencies", "decision_rules"),
            ),
            (
                f"{sub_module} Technical Notes",
                TaskType.TROUBLESHOOTING.value,
                DocType.TROUBLESHOOTING_DOC,
                ("technical_architecture",),
            ),
        ]

        for title, task_type, doc_type, keys in grouped_sections:
            lines = [title]
            for key in keys:
                value = entry.get(key)
                if not value:
                    continue
                lines.append(key.replace("_", " ").title())
                if isinstance(value, list):
                    lines.extend(f"- {normalize_text(item)}" for item in value if normalize_text(item))
                else:
                    lines.append(normalize_text(value))
            content = "\n".join(lines).strip()
            if is_weak_boilerplate(title, content):
                continue
            documents.append(
                {
                    "source_path": str(path),
                    "source_uri": str(source_uri),
                    "canonical_uri": str(source_uri),
                    "title": title,
                    "module": raw_module,
                    "task_type": task_type,
                    "doc_type": doc_type,
                    "quality_score": quality_score,
                    "content": content,
                    "doc_release": parse_release(str(source_uri) or path.name),
                }
            )
    return documents


def render_oracle_bundle_docs(path: Path, module: str, payload: Dict[str, Any], quality_score: float) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    metadata = payload.get("metadata") or {}
    base_release = parse_release(str(metadata.get("version") or "") or path.name)
    for document in payload.get("documents", []) or []:
        title = normalize_text(document.get("title") or path.stem.replace("_", " ").title())
        source_uri = document.get("url") or metadata.get("base_url") or str(path)
        lines = [title]
        flatten_section_nodes(document.get("sections", []) or [], lines)
        content = "\n".join(lines).strip()
        task_type, doc_type = infer_task_and_doc_type(f"{path.name} {title} {content[:200]}")
        if is_weak_boilerplate(title, content):
            continue
        documents.append(
            {
                "source_path": str(path),
                "source_uri": str(source_uri),
                "canonical_uri": str(source_uri),
                "title": title,
                "module": module,
                "task_type": task_type,
                "doc_type": doc_type,
                "quality_score": quality_score,
                "content": content,
                "doc_release": parse_release(str(source_uri)) or base_release,
            }
        )
    return documents


def render_doc_list(path: Path, module: str, payload: List[Dict[str, Any]], quality_score: float) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    for entry in payload:
        source_uri = entry.get("url") or str(path)
        title = normalize_text(entry.get("title") or path.stem.replace("_", " ").title())
        content = normalize_text(entry.get("content") or "")
        if is_weak_boilerplate(title, content):
            continue
        task_type, doc_type = infer_task_and_doc_type(f"{title} {source_uri}")
        documents.append(
            {
                "source_path": str(path),
                "source_uri": str(source_uri),
                "canonical_uri": str(source_uri),
                "title": title,
                "module": module,
                "task_type": task_type,
                "doc_type": doc_type,
                "quality_score": quality_score,
                "content": content,
                "doc_release": parse_release(str(source_uri)),
            }
        )
    return documents


def render_oracle_json_record(path: Path, module: str, payload: Dict[str, Any], quality_score: float) -> List[Dict[str, Any]]:
    source_uri = payload.get("source") or payload.get("url") or str(path)
    title = normalize_text(payload.get("title") or path.stem.replace("_", " ").title())
    content = normalize_text(payload.get("text") or payload.get("content") or "")
    if not is_oracle_signal(str(source_uri), title):
        return []
    if is_weak_boilerplate(title, content):
        return []
    task_type, doc_type = infer_task_and_doc_type(f"{title} {source_uri} {content[:200]}")
    return [
        {
            "source_path": str(path),
            "source_uri": str(source_uri),
            "canonical_uri": str(source_uri),
            "title": title,
            "module": module,
            "task_type": task_type,
            "doc_type": doc_type,
            "quality_score": quality_score,
            "content": content,
            "doc_release": parse_release(str(source_uri)),
        }
    ]


def load_documents_for_spec(spec: SourceSpec, path: Path) -> List[Dict[str, Any]]:
    if spec.kind == "oracle_pdf":
        content = extract_pdf_text(path)
        title = normalize_text(path.stem.replace("-", " ").replace("_", " ").title())
        task_type, doc_type = infer_task_and_doc_type(path.name)
        if is_weak_boilerplate(title, content):
            return []
        return [
            {
                "source_path": str(path),
                "source_uri": str(path),
                "canonical_uri": str(path),
                "title": title,
                "module": spec.module,
                "task_type": task_type,
                "doc_type": doc_type,
                "quality_score": spec.quality_score,
                "content": content,
                "doc_release": parse_release(path.name),
            }
        ]

    if spec.kind == "oracle_txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        title = title_from_text_export(path, text)
        source_uri = source_uri_from_path(path)
        task_type, doc_type = infer_task_and_doc_type(f"{path.name} {title}")
        if is_weak_boilerplate(title, text):
            return []
        return [
            {
                "source_path": str(path),
                "source_uri": source_uri,
                "canonical_uri": source_uri,
                "title": title,
                "module": spec.module,
                "task_type": task_type,
                "doc_type": doc_type,
                "quality_score": spec.quality_score,
                "content": text,
                "doc_release": parse_release(source_uri) or parse_release(path.name),
            }
        ]

    payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    if spec.kind == "oracle_web_data":
        if not isinstance(payload, list):
            return []
        return render_web_data_docs(path, spec.module, payload, spec.quality_score)
    if spec.kind == "oracle_bundle_json":
        if not isinstance(payload, dict):
            return []
        return render_oracle_bundle_docs(path, spec.module, payload, spec.quality_score)
    if spec.kind == "documentation_list":
        if not isinstance(payload, list):
            return []
        return render_doc_list(path, spec.module, payload, spec.quality_score)
    if spec.kind == "oracle_json_record":
        if not isinstance(payload, dict):
            return []
        return render_oracle_json_record(path, spec.module, payload, spec.quality_score)
    return []


def resolve_source_paths(spec: SourceSpec) -> List[Path]:
    matches = sorted(spec.root.glob(spec.pattern))
    if not matches:
        return []
    matches.sort(key=lambda path: (release_sort_key(parse_release(path.name)), path.name), reverse=True)
    return matches[: spec.max_matches]


def collect_source_batch(max_docs_per_module: int = 8) -> Tuple[List[Tuple[SourceSpec, Path]], List[Dict[str, Any]]]:
    by_module: Dict[str, List[Tuple[SourceSpec, Path]]] = defaultdict(list)
    rejected: List[Dict[str, Any]] = [
        {
            "path": str(ORACLEWINGS_ROOT / "data" / "crawler" / "oracle_docs.jsonl"),
            "reason": "stub_or_untrusted_repo_crawler_export",
        }
    ]
    for spec in OFFICIAL_SOURCE_SPECS:
        for path in resolve_source_paths(spec):
            if "COMMUNITY_DATA" in path.name.upper():
                rejected.append({"path": str(path), "reason": "community_data_rejected"})
                continue
            by_module[spec.module].append((spec, path))

    selected: List[Tuple[SourceSpec, Path]] = []
    for module, items in by_module.items():
        items.sort(key=lambda item: (item[0].priority, release_sort_key(parse_release(item[1].name))), reverse=True)
        selected.extend(items[:max_docs_per_module])
        for skipped_spec, skipped_path in items[max_docs_per_module:]:
            rejected.append({"path": str(skipped_path), "reason": f"controlled_batch_limit:{module}"})
    return selected, rejected


def dedupe_documents(documents: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    rejected: List[Dict[str, Any]] = []
    for document in documents:
        key = (
            str(document.get("canonical_uri") or document.get("source_uri") or document.get("source_path")),
            normalized_title(document["title"]),
        )
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = document
            continue
        current_release = release_sort_key(str(document.get("doc_release") or ""))
        existing_release = release_sort_key(str(existing.get("doc_release") or ""))
        if current_release > existing_release:
            rejected.append({"path": existing["source_path"], "reason": "deduped_by_uri_title"})
            deduped[key] = document
        else:
            rejected.append({"path": document["source_path"], "reason": "deduped_by_uri_title"})

    content_seen: Dict[str, Dict[str, Any]] = {}
    final_docs: List[Dict[str, Any]] = []
    for document in deduped.values():
        content_hash = stable_hash("docs_corpus", normalize_text(document["content"]))
        if content_hash in content_seen:
            rejected.append({"path": document["source_path"], "reason": "deduped_by_content_hash"})
            continue
        document["content_hash"] = content_hash
        content_seen[content_hash] = document
        final_docs.append(document)

    final_docs.sort(key=lambda item: (item["module"], normalized_title(item["title"])))
    return final_docs, rejected


def current_docs_counts() -> Dict[str, Dict[str, int]]:
    result = {
        "by_module": {},
        "by_task_type": {},
        "by_doc_type": {},
        "by_source_system": {},
    }
    if not DOCS_DB_PATH.exists():
        return result

    with sqlite3.connect(DOCS_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT metadata FROM chunks").fetchall()

    module_counter: Counter[str] = Counter()
    task_counter: Counter[str] = Counter()
    doc_type_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    for row in rows:
        metadata = json.loads(row["metadata"])
        module_counter[str(metadata.get("module") or "UNKNOWN")] += 1
        task_counter[str(metadata.get("task_type") or "UNKNOWN")] += 1
        doc_type_counter[str(metadata.get("doc_type") or "UNKNOWN")] += 1
        source_counter[str(metadata.get("source_system") or "UNKNOWN")] += 1

    result["by_module"] = dict(module_counter)
    result["by_task_type"] = dict(task_counter)
    result["by_doc_type"] = dict(doc_type_counter)
    result["by_source_system"] = dict(source_counter)
    return result


def validation_gap_summary() -> Dict[str, Any]:
    summary: Dict[str, Any] = {"failures": [], "by_bucket": {}}
    if not REAL_MODEL_VALIDATION.exists():
        return summary
    rows = [json.loads(line) for line in REAL_MODEL_VALIDATION.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("quality_tag") != "likely_correct":
            failure = {
                "id": row.get("id"),
                "bucket": row.get("benchmark_bucket"),
                "quality_tag": row.get("quality_tag"),
                "question": row.get("question"),
            }
            summary["failures"].append(failure)
            by_bucket[str(row.get("benchmark_bucket") or "UNKNOWN")].append(failure)
    summary["by_bucket"] = {
        bucket: {
            "failure_ids": [item["id"] for item in items],
            "quality_tags": sorted({item["quality_tag"] for item in items}),
            "count": len(items),
        }
        for bucket, items in by_bucket.items()
    }
    return summary


def build_task_coverage_map(selected_sources: List[Tuple[SourceSpec, Path]], rejected_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    current_counts = current_docs_counts()
    validation = validation_gap_summary()
    planned_by_module: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for spec, path in selected_sources:
        task_type, doc_type = infer_task_and_doc_type(path.name)
        planned_by_module[spec.module].append(
            {
                "path": str(path),
                "kind": spec.kind,
                "task_type_hint": task_type,
                "doc_type_hint": doc_type.value,
                "source_system": spec.source_system.value,
                "authority_tier": spec.authority_tier.value,
                "priority": spec.priority,
            }
        )

    gaps = [
        {
            "module": "Procurement",
            "missing_task_types": ["procedure", "troubleshooting"],
            "missing_doc_types": ["procedure_doc", "setup_doc", "troubleshooting_doc"],
            "benchmark_failure_ids": ["OF-0322", "OF-0323", "OF-0324"],
        },
        {
            "module": "SCM",
            "missing_task_types": ["procedure", "troubleshooting"],
            "missing_doc_types": ["procedure_doc", "functional_doc", "troubleshooting_doc"],
            "benchmark_failure_ids": ["OF-0841", "OF-0845"],
        },
        {
            "module": "HCM",
            "missing_task_types": ["procedure", "general", "integration"],
            "missing_doc_types": ["procedure_doc", "setup_doc", "functional_doc"],
            "benchmark_failure_ids": [],
        },
        {
            "module": "Receivables",
            "missing_task_types": ["procedure"],
            "missing_doc_types": ["procedure_doc", "setup_doc"],
            "benchmark_failure_ids": ["OF-0021", "OF-0026"],
        },
        {
            "module": "General Ledger",
            "missing_task_types": ["procedure"],
            "missing_doc_types": ["procedure_doc", "functional_doc"],
            "benchmark_failure_ids": ["OF-0031", "OF-0032"],
        },
    ]
    return {
        "current_counts": current_counts,
        "validation": validation,
        "gaps": gaps,
        "planned_sources": dict(planned_by_module),
        "rejected_sources": rejected_sources,
    }


def build_manifest_records(max_docs_per_module: int = 8) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    registry = get_default_registry()
    selected_sources, rejected_sources = collect_source_batch(max_docs_per_module=max_docs_per_module)
    coverage_map = build_task_coverage_map(selected_sources, rejected_sources)

    raw_documents: List[Dict[str, Any]] = []
    loader_rejections: List[Dict[str, Any]] = []
    for spec, path in selected_sources:
        try:
            docs = load_documents_for_spec(spec, path)
        except Exception as exc:
            loader_rejections.append({"path": str(path), "reason": f"loader_error:{exc}"})
            continue
        for document in docs:
            document["module"] = normalize_module(document["module"])
            document["source_system"] = spec.source_system
            document["authority_tier"] = spec.authority_tier
            document["trusted_schema_objects"] = extract_trusted_objects(document["content"], registry)
            raw_documents.append(document)

    curated_documents, dedupe_rejections = dedupe_documents(raw_documents)

    coverage_map["loader_rejections"] = loader_rejections
    coverage_map["dedupe_rejections"] = dedupe_rejections
    coverage_map["selected_document_count"] = len(curated_documents)

    manifest_records: List[Dict[str, Any]] = []
    for document in curated_documents:
        metadata = {
            "source_path": document["source_path"],
            "source_uri": document["source_uri"],
            "canonical_uri": document.get("canonical_uri") or document["source_uri"],
            "doc_release": document.get("doc_release"),
            "authority_tier": document["authority_tier"].value,
            "normalized_title": normalized_title(document["title"]),
        }
        manifest_records.append(
            {
                "source_path": document["source_path"],
                "source_uri": document["source_uri"],
                "title": document["title"],
                "module": document["module"],
                "task_type": document["task_type"],
                "doc_type": document["doc_type"].value,
                "trusted_schema_objects": document["trusted_schema_objects"],
                "quality_score": document["quality_score"],
                "content_hash": document["content_hash"],
                "source_system": document["source_system"].value,
                "corpus": "docs_corpus",
                "content": document["content"],
                "canonical_uri": metadata["canonical_uri"],
                "doc_release": metadata["doc_release"],
                "authority_tier": metadata["authority_tier"],
                "metadata": metadata,
            }
        )

    return manifest_records, coverage_map


def write_manifests(manifest_records: List[Dict[str, Any]]) -> Dict[str, str]:
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = {
        SourceSystem.ORACLE_DOCS.value: MANIFEST_ROOT / "oracle_docs_manifest.jsonl",
        SourceSystem.ORACLEWINGS_REPO.value: MANIFEST_ROOT / "oraclewings_repo_manifest.jsonl",
    }
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in manifest_records:
        grouped[record["source_system"]].append(record)

    for source_system, output_path in outputs.items():
        records = grouped.get(source_system, [])
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return {key: str(value) for key, value in outputs.items()}


def rebuild_docs_corpus(manifest_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    from core.retrieval.vectors.faiss_index import FaissIndex

    registry = get_default_registry()
    faiss = FaissIndex(tenant_id=TENANT_ID, indexes_dir=str(INDEXES_DIR), corpus="docs_corpus")
    faiss.reset()

    curated_chunks: List[Dict[str, Any]] = []
    documents_indexed = 0
    for record in manifest_records:
        document = CuratedIngestionValidator.build_document(
            source_path=record["source_path"],
            source_uri=record["source_uri"],
            title=record["title"],
            module=record["module"],
            task_type=record["task_type"],
            doc_type=DocType(record["doc_type"]),
            trusted_schema_objects=record["trusted_schema_objects"],
            quality_score=float(record["quality_score"]),
            source_system=SourceSystem(record["source_system"]),
            content=record["content"],
            metadata=dict(record.get("metadata") or {}),
        )
        documents_indexed += 1
        chunks = chunk_document_text(document.content)
        if not chunks:
            chunks = [document.content]
        for index, chunk_text in enumerate(chunks):
            chunk = CuratedIngestionValidator.build_chunk(document, chunk_text, index)
            payload = CuratedIngestionValidator.chunk_payload(chunk)
            payload["metadata"]["trusted_schema_objects"] = payload["metadata"].get("trusted_schema_objects") or extract_trusted_objects(chunk_text, registry)
            curated_chunks.append(payload)

    if curated_chunks:
        faiss.add_chunks_list(curated_chunks)

    by_module = Counter()
    by_source_system = Counter()
    by_task_type = Counter()
    by_doc_type = Counter()
    for chunk in curated_chunks:
        metadata = chunk.get("metadata") or {}
        by_module[str(metadata.get("module") or "UNKNOWN")] += 1
        by_source_system[str(metadata.get("source_system") or "UNKNOWN")] += 1
        by_task_type[str(metadata.get("task_type") or "UNKNOWN")] += 1
        by_doc_type[str(metadata.get("doc_type") or "UNKNOWN")] += 1

    return {
        "documents_indexed": documents_indexed,
        "chunks_indexed": len(curated_chunks),
        "by_module": dict(by_module),
        "by_source_system": dict(by_source_system),
        "by_task_type": dict(by_task_type),
        "by_doc_type": dict(by_doc_type),
        "faiss_stats": faiss.stats(),
    }


def run_controlled_bootstrap(max_docs_per_module: int = 8, rebuild: bool = True) -> Dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_records, coverage_map = build_manifest_records(max_docs_per_module=max_docs_per_module)
    manifest_paths = write_manifests(manifest_records)

    coverage_map_path = OUTPUT_ROOT / "first_wave_task_coverage_map.json"
    coverage_map_path.write_text(json.dumps(coverage_map, indent=2), encoding="utf-8")

    summary: Dict[str, Any] = {
        "max_docs_per_module": max_docs_per_module,
        "coverage_map_path": str(coverage_map_path),
        "manifest_paths": manifest_paths,
        "manifest_record_count": len(manifest_records),
        "rebuild_performed": rebuild,
    }
    if rebuild:
        summary["rebuild"] = rebuild_docs_corpus(manifest_records)

    summary_path = OUTPUT_ROOT / "first_wave_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def local_second_wave_documents(registry: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "repo_classification": repo_source_classification(),
        "accepted_sources": [],
        "rejected_sources": [],
    }

    finance_docs_path = ORACLEWINGS_ROOT / "data" / "source_raw" / "Fin_Functional_Docs" / "documentation_data.json"
    if finance_docs_path.exists():
        payload = json.loads(finance_docs_path.read_text(encoding="utf-8"))
        for entry in payload:
            source_uri = str(entry.get("url") or "")
            title = normalize_text(entry.get("title") or "")
            content = normalize_text(entry.get("content") or "")
            module = infer_financials_leaf_module(title, source_uri, content)
            if module not in SECOND_WAVE_TARGETS or not official_oracle_source(source_uri):
                continue
            if is_index_like_source(title, source_uri):
                continue
            docs = build_targeted_documents(
                module=module,
                title=title,
                source_path=str(finance_docs_path),
                source_uri=source_uri,
                content_blocks=[(title, content)],
                quality_score=0.96,
                source_system=SourceSystem.ORACLE_DOCS,
                authority_tier=AuthorityTier.OFFICIAL,
                registry=registry,
            )
            if docs:
                documents.extend(docs)
                report["accepted_sources"].append({"path": str(finance_docs_path), "title": title, "module": module, "classification": "oracle_docs"})

    finance_pdfs = [
        ("Payables", DATA_ROOT / "Finance" / "pdfs" / "using-payables-invoice-to-pay.pdf", 0.97),
        ("Receivables", DATA_ROOT / "Finance" / "pdfs" / "using-receivables-credit-to-cash.pdf", 0.97),
        ("General Ledger", DATA_ROOT / "Finance" / "pdfs" / "using-general-ledger.pdf", 0.97),
        ("General Ledger", DATA_ROOT / "Finance" / "pdfs" / "using-subledger-accounting.pdf", 0.94),
    ]
    for module, path, quality in finance_pdfs:
        if not path.exists():
            continue
        text = extract_pdf_text(path)
        title = normalize_text(path.stem.replace("-", " ").title())
        docs = build_targeted_documents(
            module=module,
            title=title,
            source_path=str(path),
            source_uri=str(path),
            content_blocks=[(title, text)],
            quality_score=quality,
            source_system=SourceSystem.ORACLE_DOCS,
            authority_tier=AuthorityTier.OFFICIAL,
            registry=registry,
        )
        if docs:
            documents.extend(docs)
            report["accepted_sources"].append({"path": str(path), "title": title, "module": module, "classification": "oracle_docs"})

    scm_bundles = [
        DATA_ROOT / "SCM" / "jsons" / "oracle_docs_inventory.json",
        DATA_ROOT / "SCM" / "jsons" / "oracle_docs_order_management.json",
        DATA_ROOT / "SCM" / "jsons" / "oracle_docs_work_orders.json",
        DATA_ROOT / "SCM" / "jsons" / "oracle_docs_purchasing.json",
        DATA_ROOT / "SCM" / "jsons" / "oracle_docs_demand.json",
        DATA_ROOT / "SCM" / "jsons" / "oracle_docs_manufacturing_25c.json",
    ]
    for path in scm_bundles:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        if not isinstance(payload, dict):
            continue
        for document in payload.get("documents", []) or []:
            source_uri = str(document.get("url") or path)
            title = normalize_text(document.get("title") or path.stem.replace("_", " ").title())
            blocks = list(iter_section_blocks(document.get("sections", []) or []))
            if not blocks:
                continue
            docs = build_targeted_documents(
                module="SCM",
                title=title,
                source_path=str(path),
                source_uri=source_uri,
                content_blocks=blocks,
                quality_score=0.95,
                source_system=SourceSystem.ORACLE_DOCS,
                authority_tier=AuthorityTier.OFFICIAL,
                registry=registry,
            )
            if docs:
                documents.extend(docs)
        report["accepted_sources"].append({"path": str(path), "title": path.name, "module": "SCM", "classification": "oracle_docs"})

    procurement_jsons = sorted((DATA_ROOT / "Procurement" / "jsons").glob("procurement_*.json"))
    for path in procurement_jsons:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        source_uri = str(payload.get("source") or payload.get("url") or path)
        title = normalize_text(payload.get("title") or path.stem)
        content = normalize_text(payload.get("text") or payload.get("content") or "")
        if not official_oracle_source(source_uri):
            continue
        docs = build_targeted_documents(
            module="Procurement",
            title=title,
            source_path=str(path),
            source_uri=source_uri,
            content_blocks=[(title, content)],
            quality_score=0.9,
            source_system=SourceSystem.ORACLE_DOCS,
            authority_tier=AuthorityTier.OFFICIAL,
            registry=registry,
        )
        if docs:
            documents.extend(docs)
            report["accepted_sources"].append({"path": str(path), "title": title, "module": "Procurement", "classification": "oracle_docs"})
        else:
            report["rejected_sources"].append({"path": str(path), "title": title, "reason": "low_signal_or_index_like"})

    return documents, report


def remote_second_wave_documents(registry: Any, current_counts: Dict[str, int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    documents: List[Dict[str, Any]] = []
    fetched_sources: List[Dict[str, Any]] = []
    remote_sources = [*SECOND_WAVE_REMOTE_SOURCES, *dynamic_scm_remote_sources()]
    for source in remote_sources:
        module = str(source["module"])
        if current_counts.get(module, 0) >= SECOND_WAVE_TARGETS[module]["target_chunks"]:
            continue
        source_path, source_uri, blocks, title = remote_source_to_blocks(source)
        docs = build_targeted_documents(
            module=module,
            title=title,
            source_path=source_path,
            source_uri=source_uri,
            content_blocks=blocks,
            quality_score=0.98,
            source_system=SourceSystem.ORACLE_DOCS,
            authority_tier=AuthorityTier.OFFICIAL,
            registry=registry,
        )
        if docs:
            documents.extend(docs)
            fetched_sources.append(
                {
                    "module": module,
                    "title": title,
                    "source_uri": source_uri,
                    "source_path": source_path,
                    "chunks_selected": len(docs),
                }
            )
            current_counts[module] = current_counts.get(module, 0) + len(docs)
    return documents, fetched_sources


def write_second_wave_manifests(manifest_records: List[Dict[str, Any]]) -> Dict[str, str]:
    SECOND_WAVE_MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = {
        SourceSystem.ORACLE_DOCS.value: SECOND_WAVE_MANIFEST_ROOT / "oracle_docs_manifest.jsonl",
        SourceSystem.ORACLEWINGS_REPO.value: SECOND_WAVE_MANIFEST_ROOT / "oraclewings_repo_manifest.jsonl",
    }
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in manifest_records:
        grouped[record["source_system"]].append(record)
    for source_system, output_path in outputs.items():
        with output_path.open("w", encoding="utf-8") as handle:
            for record in grouped.get(source_system, []):
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return {key: str(value) for key, value in outputs.items()}


def build_second_wave_manifest_records(allow_fetch: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    registry = get_default_registry()
    local_documents, local_report = local_second_wave_documents(registry)
    deduped_local_docs, local_dedupe_rejections = dedupe_documents(local_documents)

    local_counts = Counter(document["module"] for document in deduped_local_docs)
    fetched_documents: List[Dict[str, Any]] = []
    fetched_sources: List[Dict[str, Any]] = []
    if allow_fetch:
        fetched_documents, fetched_sources = remote_second_wave_documents(registry, dict(local_counts))

    curated_documents, all_dedupe_rejections = dedupe_documents([*deduped_local_docs, *fetched_documents])
    coverage_map = {
        "current_counts": current_docs_counts(),
        "validation": validation_gap_summary(),
        "target_chunks": {module: data["target_chunks"] for module, data in SECOND_WAVE_TARGETS.items()},
        "repo_source_classification_path": str(SECOND_WAVE_CLASSIFICATION_PATH),
        "repo_classification_summary": dict(Counter(item["classification"] for item in local_report["repo_classification"])),
        "local_candidate_counts": dict(local_counts),
        "fetched_sources": fetched_sources,
        "dedupe_rejections": [*local_dedupe_rejections, *all_dedupe_rejections],
        "accepted_local_sources": local_report["accepted_sources"],
        "rejected_local_sources": local_report["rejected_sources"],
    }

    manifest_records: List[Dict[str, Any]] = []
    for document in curated_documents:
        metadata = {
            "source_path": document["source_path"],
            "source_uri": document["source_uri"],
            "canonical_uri": document.get("canonical_uri") or document["source_uri"],
            "doc_release": document.get("doc_release"),
            "authority_tier": document["authority_tier"].value,
            "normalized_title": normalized_title(document["title"]),
            **dict(document.get("metadata") or {}),
        }
        manifest_records.append(
            {
                "source_path": document["source_path"],
                "source_uri": document["source_uri"],
                "title": document["title"],
                "module": document["module"],
                "task_type": document["task_type"],
                "doc_type": document["doc_type"].value,
                "trusted_schema_objects": document["trusted_schema_objects"],
                "quality_score": document["quality_score"],
                "content_hash": document["content_hash"],
                "source_system": document["source_system"].value,
                "corpus": "docs_corpus",
                "content": document["content"],
                "canonical_uri": metadata["canonical_uri"],
                "doc_release": metadata["doc_release"],
                "authority_tier": metadata["authority_tier"],
                "metadata": metadata,
            }
        )

    SECOND_WAVE_ROOT.mkdir(parents=True, exist_ok=True)
    SECOND_WAVE_CLASSIFICATION_PATH.write_text(
        json.dumps(local_report["repo_classification"], indent=2),
        encoding="utf-8",
    )
    coverage_map["selected_document_count"] = len(curated_documents)
    coverage_map["manifest_record_count"] = len(manifest_records)
    return manifest_records, coverage_map


def run_second_wave_bootstrap(rebuild: bool = True, allow_fetch: bool = True) -> Dict[str, Any]:
    SECOND_WAVE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_records, coverage_map = build_second_wave_manifest_records(allow_fetch=allow_fetch)
    manifest_paths = write_second_wave_manifests(manifest_records)
    coverage_map_path = SECOND_WAVE_ROOT / "second_wave_task_coverage_map.json"
    coverage_map_path.write_text(json.dumps(coverage_map, indent=2), encoding="utf-8")

    summary: Dict[str, Any] = {
        "coverage_map_path": str(coverage_map_path),
        "repo_classification_path": str(SECOND_WAVE_CLASSIFICATION_PATH),
        "manifest_paths": manifest_paths,
        "manifest_record_count": len(manifest_records),
        "allow_fetch": allow_fetch,
        "rebuild_performed": rebuild,
    }
    if rebuild:
        summary["rebuild"] = rebuild_docs_corpus(manifest_records)

    summary_path = SECOND_WAVE_ROOT / "second_wave_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
