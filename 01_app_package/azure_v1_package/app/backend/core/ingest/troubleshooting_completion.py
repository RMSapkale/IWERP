import json
import re
from pathlib import Path
from typing import Any, Dict, List

from core.ingest.curation import CuratedIngestionValidator, stable_hash
from core.retrieval.vectors.faiss_index import FaissIndex
from core.schemas.curation import AuthorityTier, DocType, IngestionManifestRecord, SourceSystem


BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
WORKSPACE_DIR = BASE_DIR.parent
OUTPUT_ROOT = BASE_DIR / "troubleshooting_completion"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
INDEXES_DIR = BASE_DIR / "backend" / "core" / "retrieval" / "vectors"
BENCHMARK_DIR = BASE_DIR / "production_benchmark"

VALIDATION_JSONL = WORKSPACE_DIR / "Data_Complete" / "Finance" / "jsonls" / "validation.jsonl"
ASSETS_PDF = WORKSPACE_DIR / "Data_Complete" / "Finance" / "pdfs" / "using-assets.pdf"
TAX_PDF = WORKSPACE_DIR / "Data_Complete" / "Finance" / "pdfs" / "using-tax.pdf"
RESIDUAL_BENCHMARK_RESULTS = BENCHMARK_DIR / "troubleshooting_completion_50" / "results.jsonl"
RESIDUAL_EXACT_RESULTS = BENCHMARK_DIR / "troubleshooting_completion_exact10" / "results.jsonl"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _preview(value: str, limit: int = 1800) -> str:
    return _normalize_text(value)[:limit]


def _read_validation_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with VALIDATION_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _validation_row_by_fragment(rows: List[Dict[str, Any]], fragment: str) -> Dict[str, Any]:
    fragment = fragment.lower()
    for row in rows:
        instruction = str(row.get("instruction") or "")
        output = str(row.get("output") or "")
        if fragment in instruction.lower() or fragment in output.lower():
            source_uri_match = re.search(r"Source:\s*(https?://\S+)", output)
            source_uri = source_uri_match.group(1) if source_uri_match else fragment
            title_match = re.search(r"#\s+([^\n]+)", output)
            title = title_match.group(1).strip() if title_match else fragment
            return {
                "source_path": str(VALIDATION_JSONL),
                "source_uri": source_uri,
                "title": title,
                "content": output,
                "source_kind": "validation_jsonl",
            }
    raise ValueError(f"Could not find validation row containing: {fragment}")


def _extract_pdf_pages(pdf_path: Path, page_numbers: List[int], title: str) -> Dict[str, Any]:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - exercised in venv at runtime
        raise RuntimeError("pypdf is required to extract troubleshooting PDF pages") from exc

    reader = PdfReader(str(pdf_path))
    extracted: List[str] = []
    for page_num in page_numbers:
        text = reader.pages[page_num - 1].extract_text() or ""
        extracted.append(f"[Page {page_num}]\n{text}")

    return {
        "source_path": str(pdf_path),
        "source_uri": f"{pdf_path}#pages={','.join(str(num) for num in page_numbers)}",
        "title": title,
        "content": "\n\n".join(extracted),
        "source_kind": "pdf_page_extract",
    }


PATTERN_SPECS: List[Dict[str, Any]] = [
    {
        "pattern_id": "cash_management_reconciliation_rules",
        "module": "Cash Management",
        "source": ("validation", "overview-of-reconciliation-rules-sets"),
        "title": "Cash Management troubleshooting: unmatched bank statement reconciliation lines",
        "symptom": "Automatic bank statement reconciliation leaves statement lines unmatched or rejected payments aren't reconciled.",
        "root_causes": [
            "Matching rules are sequenced poorly and broad rules run before precise one-to-one rules.",
            "Rules with bank reference IDs aren't prioritized, so duplicates or low-confidence matches win.",
            "Automatic reconciliation of rejected payments isn't enabled or the bank transaction code isn't set to Reversal.",
        ],
        "resolution": [
            "Review the assigned reconciliation rule set and place one-to-one rules before looser match types.",
            "Lower the sequence number for rules that use bank-provided reference IDs.",
            "Enable Automatic Reconciliation of Reject Payments and map the bank transaction code to Reversal.",
            "Rerun autoreconciliation and review the remaining exceptions on the reconciliation work area.",
        ],
        "symptom_terms": ["bank statement reconciliation", "unmatched lines", "rejected payments", "automatic reconciliation"],
        "error_keywords": ["reconciliation", "bank statement", "unmatched", "rejected", "rule set", "matching rule"],
        "task_signals": ["bank statement reconciliation"],
        "confidence_band": "high",
        "quality_score": 0.94,
    },
    {
        "pattern_id": "cash_management_transaction_visibility",
        "module": "Cash Management",
        "source": ("validation", "Cash-Management-External-Cash-Transactions-Real-Ti-SA-177"),
        "title": "Cash Management troubleshooting: bank statement transactions missing from reconciliation",
        "symptom": "Uploaded bank statement or reconciliation activity doesn't surface the expected external cash transactions.",
        "root_causes": [
            "The wrong bank account, business unit, legal entity, or date range is being reviewed.",
            "The external cash transaction wasn't generated through upload, reconciliation processing, or manual entry for the expected period.",
        ],
        "resolution": [
            "Review Manage Transactions for the bank account, business unit, legal entity, and date range tied to the statement.",
            "Confirm the transaction came from bank statement upload, reconciliation processing, or manual entry for the same period.",
            "Use the external cash transactions subject area or Manage Transactions page to verify whether the transaction exists before retrying reconciliation.",
        ],
        "symptom_terms": ["bank statement reconciliation", "missing transactions", "manage transactions", "external cash transaction"],
        "error_keywords": ["missing", "transactions", "bank statement", "reconciliation", "bank account", "business unit"],
        "task_signals": ["bank statement reconciliation"],
        "confidence_band": "high",
        "quality_score": 0.9,
    },
    {
        "pattern_id": "cash_management_receipt_matching",
        "module": "Cash Management",
        "source": ("validation", "overview-of-reconciliation-rules-sets"),
        "title": "Cash Management troubleshooting: receipt matching failures during reconciliation",
        "symptom": "Receipt-related cash transactions don't match during bank statement reconciliation and appear as receipt application errors.",
        "root_causes": [
            "The rule set is not optimized for receipt references provided by the bank.",
            "Transactions without stable reference IDs are evaluated too early and create duplicate or low-confidence matches.",
        ],
        "resolution": [
            "Adjust the reconciliation rule sequence so reference-based rules are evaluated before looser matching rules.",
            "Review the statement line reference values used by the bank and align the matching rule attributes to those values.",
            "Re-run reconciliation after rule sequencing changes and inspect the remaining receipt-related exceptions.",
        ],
        "symptom_terms": ["receipt application errors", "receipt matching", "bank statement reconciliation", "matching rules"],
        "error_keywords": ["receipt", "application", "matching", "reconciliation", "reference id"],
        "task_signals": ["bank statement reconciliation"],
        "confidence_band": "medium",
        "quality_score": 0.86,
    },
    {
        "pattern_id": "expenses_audit_actions",
        "module": "Expenses",
        "source": ("validation", "audit-actions"),
        "title": "Expenses troubleshooting: expense report stuck in audit or returned for action",
        "symptom": "An expense report remains in audit, is rejected, or repeatedly requests more information.",
        "root_causes": [
            "The selected audit action requires employee follow-up or receipt handling before the report can move forward.",
            "Manager approval and audit timing are misaligned with the Expense Report Audit Approval system option.",
            "The report is still on hold or waiting for the auditor to complete the audit action.",
        ],
        "resolution": [
            "Open Audit Expense Report and review the last audit action taken on the report.",
            "Use the appropriate audit action: Request More Information, Reject, Release Hold, Complete Audit, or Waive Receipts where policy allows.",
            "If manager approval is still pending, verify the Expense Report Audit Approval option and whether audit can run in parallel.",
            "After the required action is completed, confirm the report status changes toward Pending Payment or the next approval step.",
        ],
        "symptom_terms": ["expense report audit errors", "audit expense report", "request more information", "release hold"],
        "error_keywords": ["audit", "expense report", "hold", "reject", "request information", "status"],
        "task_signals": ["expense report audit"],
        "preferred_modules": ["Expenses"],
        "correction_note_required": True,
        "confidence_band": "high",
        "quality_score": 0.95,
    },
    {
        "pattern_id": "expenses_audit_list_rules",
        "module": "Expenses",
        "source": ("validation", "how-you-add-to-or-remove-employees-from-the-audit-list"),
        "title": "Expenses troubleshooting: user repeatedly selected for expense report audit",
        "symptom": "Expense reports keep getting selected for audit even after prior reviews are complete.",
        "root_causes": [
            "The employee remains on the audit list.",
            "Audit list rules keep re-adding the employee because thresholds or policy violations are still being hit.",
        ],
        "resolution": [
            "Review Manage Audit List Membership to confirm whether the employee is currently on the audit list.",
            "Check the configured audit list rules for receipt package delays, monthly amount limits, policy violations, and report-count thresholds.",
            "Remove the employee manually if appropriate, or adjust the audit list rule criteria if the current thresholds are too aggressive.",
        ],
        "symptom_terms": ["expense report audit errors", "audit list", "selected for audit", "policy violations"],
        "error_keywords": ["audit", "audit list", "expense report", "policy violation", "receipt package"],
        "task_signals": ["expense report audit"],
        "preferred_modules": ["Expenses"],
        "correction_note_required": True,
        "confidence_band": "high",
        "quality_score": 0.93,
    },
    {
        "pattern_id": "expenses_travel_card_upload",
        "module": "Expenses",
        "source": ("validation", "travel-card-processing"),
        "title": "Expenses troubleshooting: corporate card transactions not available for expense processing",
        "symptom": "Expense users can't complete report audit or receipt matching because corporate card transactions didn't upload cleanly.",
        "root_causes": [
            "Upload Corporate Card Transactions wasn't run or its upload results weren't reviewed.",
            "The transaction file upload contains errors that prevent usable card transactions from reaching Expenses.",
        ],
        "resolution": [
            "Run the Upload Corporate Card Transactions process and review the upload results.",
            "Open the transaction file upload details and correct the file-level errors before retrying.",
            "After transactions load successfully, continue audit or report processing on the affected expense reports.",
        ],
        "symptom_terms": ["receipt application errors", "corporate card", "travel card", "expense processing"],
        "error_keywords": ["card", "upload", "expense", "receipt", "audit", "transactions"],
        "task_signals": ["expense report audit"],
        "confidence_band": "medium",
        "quality_score": 0.84,
    },
    {
        "pattern_id": "assets_post_mass_additions",
        "module": "Assets",
        "source": ("pdf", str(ASSETS_PDF), [46]),
        "title": "Assets troubleshooting: Post Mass Additions fails",
        "symptom": "Mass additions or receipt-based asset lines fail during posting in Assets.",
        "root_causes": [
            "One or more source lines contain invalid data and fail during Post Mass Additions.",
            "The underlying source line needs correction before the mass addition can be posted.",
        ],
        "resolution": [
            "Open the log file from Post Mass Additions.",
            "Review the Post Mass Additions Execution Report to identify which source lines failed.",
            "Correct the failed source lines directly and resubmit Post Mass Additions.",
        ],
        "symptom_terms": ["asset capitalization errors", "post mass additions", "posting errors", "mass additions"],
        "error_keywords": ["post", "mass additions", "assets", "source line", "failed", "capitalization"],
        "task_signals": ["asset capitalization"],
        "confidence_band": "high",
        "quality_score": 0.95,
    },
    {
        "pattern_id": "assets_receiving_import",
        "module": "Assets",
        "source": ("pdf", str(ASSETS_PDF), [15, 28]),
        "title": "Assets troubleshooting: receipt or invoice lines not creating asset additions",
        "symptom": "Receipt-based or payables-based asset lines don't import cleanly into Assets and surface as receipt application or capitalization errors.",
        "root_causes": [
            "The invoice or receipt isn't fully approved/accounted before import.",
            "The asset or CIP clearing account isn't mapped to an existing asset category.",
            "Track as Asset or required ledger alignment isn't configured for the source line.",
        ],
        "resolution": [
            "Verify the invoice is approved, posted, and matched to the purchase order or receipt where required.",
            "Confirm the natural account is configured as an Asset Clearing or CIP Clearing account for the asset category.",
            "Check Track as Asset, business unit to ledger alignment, and the general ledger date before rerunning Create Mass Additions or Transfer Receipts to Mass Additions.",
        ],
        "symptom_terms": ["receipt application errors", "receipt-based asset lines", "mass additions", "capitalization"],
        "error_keywords": ["receipt", "invoice", "mass additions", "asset clearing", "cip clearing", "ledger"],
        "task_signals": ["asset capitalization"],
        "confidence_band": "high",
        "quality_score": 0.92,
    },
    {
        "pattern_id": "assets_reverse_capitalization",
        "module": "Assets",
        "source": ("pdf", str(ASSETS_PDF), [107]),
        "title": "Assets troubleshooting: capitalization posted incorrectly and needs reversal",
        "symptom": "A CIP asset was capitalized incorrectly or the capitalization transaction needs to be undone to correct the asset cost.",
        "root_causes": [
            "The wrong CIP asset or source lines were capitalized.",
            "A capitalization transaction was posted in error and needs a controlled reversal.",
        ],
        "resolution": [
            "Use the mass adjustment or transaction flow that supports Capitalization transactions and Reverse capitalization transactions.",
            "Reverse the incorrect capitalization first, then correct the source lines or CIP details.",
            "Repost the corrected capitalization after the source data and asset selection are verified.",
        ],
        "symptom_terms": ["asset capitalization errors", "reverse capitalization", "cip asset", "capitalization transaction"],
        "error_keywords": ["capitalize", "reverse capitalize", "cip", "source line", "posted incorrectly"],
        "task_signals": ["asset capitalization"],
        "confidence_band": "high",
        "quality_score": 0.91,
    },
    {
        "pattern_id": "assets_capitalization_approvals",
        "module": "Assets",
        "source": ("pdf", str(ASSETS_PDF), [116, 117, 118]),
        "title": "Assets troubleshooting: capitalization or asset transactions stuck in approval",
        "symptom": "Asset capitalization or related asset transactions remain pending, rejected, or blocked in the approval workflow.",
        "root_causes": [
            "Approvals are enabled for the book but workflow rules aren't configured correctly.",
            "There are pending transactions in the infotile or the spreadsheet workflow upload wasn't completed successfully.",
            "Excel macros are blocked when managing approval rules in spreadsheet templates.",
        ],
        "resolution": [
            "Verify approvals are enabled for the correct book and transaction types in Manage Asset Books.",
            "Review pending transactions before enabling approvals and confirm the Asset Transactions Approval Workflow rules were uploaded successfully.",
            "If spreadsheet rule maintenance is blocked, enable macros and trusted locations in Excel before regenerating and uploading the rule file.",
        ],
        "symptom_terms": ["asset capitalization errors", "approval workflow", "pending transactions", "rejected transaction"],
        "error_keywords": ["approval", "workflow", "pending", "rejected", "macros blocked", "book"],
        "task_signals": ["asset capitalization"],
        "confidence_band": "high",
        "quality_score": 0.9,
    },
    {
        "pattern_id": "tax_external_transaction_upload",
        "module": "Tax",
        "source": ("pdf", str(TAX_PDF), [86, 87, 89]),
        "title": "Tax troubleshooting: external taxable transaction upload ends in validation error",
        "symptom": "External taxable transaction imports fail with validation errors or stop during tax repository processing.",
        "root_causes": [
            "Required header or line data is missing in the spreadsheet.",
            "The source CSV contains invalid data and the load process deletes all rows after the failure.",
            "The tax setup prerequisites for external taxable transactions aren't complete.",
        ],
        "resolution": [
            "Review the Load File to Interface child process log and output to identify the failing data.",
            "Correct the spreadsheet data, regenerate the CSV or ZIP, and rerun Load Interface File for Import.",
            "If the batch is stuck in error, purge the tax transactions and reload after the data is corrected.",
            "Verify Subledger Accounting Methods, Configuration Owner Tax Options, liability and recoverable accounts, and active tax rate codes.",
        ],
        "symptom_terms": ["receipt application errors", "tax validation errors", "external taxable transactions", "import errors"],
        "error_keywords": ["validation", "error", "tax entry repository", "load interface", "purge tax transactions"],
        "task_signals": ["tax validation"],
        "confidence_band": "high",
        "quality_score": 0.95,
    },
    {
        "pattern_id": "tax_inclusive_treatment_error",
        "module": "Tax",
        "source": ("pdf", str(TAX_PDF), [88, 90, 91]),
        "title": "Tax troubleshooting: inclusive tax or exemption handling stops transaction processing",
        "symptom": "Tax processing stops because inclusive tax treatment or exemption data is inconsistent with the uploaded transaction.",
        "root_causes": [
            "Configuration Owner Tax Options use Inclusive Treatment = Error for the uploaded transaction shape.",
            "Manual exemption data conflicts with defined tax exemptions or required exemption handling.",
        ],
        "resolution": [
            "Review Configuration Owner Tax Options for the event class and confirm whether Inclusive Treatment should be Adjust or Error.",
            "If exemption data was entered manually, align the exemption handling value, certificate number, and exemption reason with the intended scenario.",
            "Reload the corrected external transaction after updating the spreadsheet or the tax option configuration.",
        ],
        "symptom_terms": ["tax validation errors", "inclusive tax", "receipt application errors", "tax exemption"],
        "error_keywords": ["inclusive", "error", "exemption", "tax option", "processing stops"],
        "task_signals": ["tax validation"],
        "confidence_band": "high",
        "quality_score": 0.9,
    },
    {
        "pattern_id": "tax_reconciliation_audit_reports",
        "module": "Tax",
        "source": ("pdf", str(TAX_PDF), [102, 103, 104, 105, 106]),
        "title": "Tax troubleshooting: reconciliation or audit reports missing expected tax activity",
        "symptom": "Tax reconciliation or audit reports don't show the expected transactions or totals for a reporting period.",
        "root_causes": [
            "The report is being run at the wrong reporting level or tax registration number.",
            "Only accounted or posted transactions are eligible for the selected report layout.",
            "The report parameters filter out the relevant account, tax rate code, or currency.",
        ],
        "resolution": [
            "Choose the reporting level that matches the use case: ledger, legal entity, or tax registration number.",
            "For reconciliation, verify the transactions are accounted or posted before expecting them in the report.",
            "Re-run the report with the correct registration number, tax point date, currency, and segment filters.",
        ],
        "symptom_terms": ["expense report audit errors", "tax audit trail report", "tax reconciliation report", "missing tax activity"],
        "error_keywords": ["audit", "reconciliation", "registration number", "posted", "accounted", "report parameters"],
        "task_signals": ["tax validation"],
        "confidence_band": "high",
        "quality_score": 0.92,
    },
    {
        "pattern_id": "tax_registration_validation",
        "module": "Tax",
        "source": ("validation", "czech-republic.html"),
        "title": "Tax troubleshooting: tax registration number fails validation",
        "symptom": "Tax setup or reporting fails because the tax registration number format doesn't validate.",
        "root_causes": [
            "The country code, validation type, digit count, or format doesn't match the configured registration rules.",
            "The registration number is too short, too long, or contains invalid characters.",
        ],
        "resolution": [
            "Verify the country code, validation type, allowed length, and format for the relevant tax registration type.",
            "Correct the registration number to satisfy the documented format before rerunning tax setup or reporting validation.",
            "If the jurisdiction differs, use the validation rules for the correct country-specific registration type.",
        ],
        "symptom_terms": ["tax validation errors", "tax registration", "invalid format", "registration number"],
        "error_keywords": ["tax registration", "validation", "country code", "format", "digits"],
        "task_signals": ["tax validation"],
        "confidence_band": "high",
        "quality_score": 0.9,
    },
]

RESIDUAL_PATTERN_SPECS: List[Dict[str, Any]] = [
    {
        "pattern_id": "assets_expense_report_audit_redirect",
        "module": "Assets",
        "source": ("validation", "audit-actions"),
        "title": "Assets troubleshooting: expense report audit issues must be resolved in Expenses",
        "symptom": "An Assets troubleshooting request references expense report audit errors, audit holds, or repeated audit actions on expense reports.",
        "root_causes": [
            "Expense report audit actions are handled in Expenses rather than Assets transaction processing.",
            "The report remains in audit because the required audit action, receipt action, or approval follow-up hasn't been completed in Expenses.",
        ],
        "resolution": [
            "Treat the issue as an Expenses audit workflow problem rather than an Assets processing problem.",
            "Open Audit Expense Report and review the last audit action taken on the report.",
            "Complete the required audit action or receipt action, then resubmit the report after the audit hold is cleared.",
        ],
        "symptom_terms": ["expense report audit errors", "audit expense report", "request more information", "release hold"],
        "error_keywords": ["expense", "audit", "report", "assets", "hold", "request information"],
        "task_signals": ["expense report audit"],
        "preferred_modules": ["Expenses"],
        "correction_note_required": True,
        "confidence_band": "high",
        "quality_score": 0.96,
    },
    {
        "pattern_id": "cash_management_expense_report_audit_redirect",
        "module": "Cash Management",
        "source": ("validation", "audit-actions"),
        "title": "Cash Management troubleshooting: expense report audit issues must be resolved in Expenses",
        "symptom": "A Cash Management troubleshooting request references expense report audit errors, request-more-information actions, or reports that remain on audit hold.",
        "root_causes": [
            "Expense report audit actions are executed in Expenses, not in Cash Management transaction processing.",
            "The report remains in audit because the required audit action, receipt action, or approval follow-up hasn't been completed in Expenses.",
        ],
        "resolution": [
            "Treat the issue as an Expenses audit workflow problem rather than a Cash Management processing problem.",
            "Open Audit Expense Report and review the last audit action taken on the report.",
            "Complete the required action such as Request More Information, Release Hold, Complete Audit, or Waive Receipts where policy allows.",
            "If the report keeps returning to audit, review the audit list membership and audit rules before resubmitting the report.",
        ],
        "symptom_terms": ["expense report audit errors", "audit expense report", "request more information", "release hold"],
        "error_keywords": ["expense", "audit", "report", "hold", "request information", "cash management"],
        "task_signals": ["expense report audit"],
        "preferred_modules": ["Expenses"],
        "correction_note_required": True,
        "confidence_band": "high",
        "quality_score": 0.97,
    },
    {
        "pattern_id": "general_ledger_expense_report_audit_redirect",
        "module": "General Ledger",
        "source": ("validation", "audit-actions"),
        "title": "General Ledger troubleshooting: expense report audit issues must be resolved in Expenses",
        "symptom": "A General Ledger troubleshooting request references expense report audit errors, audit holds, or request-more-information actions on expense reports.",
        "root_causes": [
            "Expense report audit actions are executed in Expenses rather than in General Ledger.",
            "The report remains in audit because the required audit action or approval follow-up hasn't been completed in Expenses.",
        ],
        "resolution": [
            "Treat the issue as an Expenses audit workflow problem rather than a General Ledger processing problem.",
            "Open Audit Expense Report and review the last audit action taken on the report.",
            "Complete the required action such as Request More Information, Release Hold, Complete Audit, or Waive Receipts where policy allows.",
        ],
        "symptom_terms": ["expense report audit errors", "audit expense report", "request more information", "release hold"],
        "error_keywords": ["expense", "audit", "report", "general ledger", "hold", "request information"],
        "task_signals": ["expense report audit"],
        "preferred_modules": ["Expenses"],
        "correction_note_required": True,
        "confidence_band": "high",
        "quality_score": 0.96,
    },
    {
        "pattern_id": "financials_supplier_site_setup_redirect",
        "module": "Financials",
        "source": ("validation", "how-you-approve-changes-Made-to-sites-in-a-supplier-profile"),
        "title": "Financials troubleshooting: supplier site setup issues usually resolve in Payables or Procurement supplier flows",
        "symptom": "A Financials troubleshooting request references supplier site setup failures, site changes pending approval, or supplier site updates that don't progress.",
        "root_causes": [
            "Supplier site changes are being controlled in supplier or procurement workflows and the change request is still pending.",
            "Required supplier site attributes under General, Purchasing, Receiving, or Invoicing are incomplete or inconsistent.",
        ],
        "resolution": [
            "Review the supplier profile change request or supplier site record and identify which site attributes triggered approval or validation failure.",
            "Verify the site purposes and required site attributes are complete for the intended supplier setup.",
            "Complete or approve the supplier site change request, then retry the supplier site update after the workflow is cleared.",
        ],
        "symptom_terms": ["supplier site setup errors", "supplier site", "site change request", "supplier profile"],
        "error_keywords": ["supplier site", "supplier profile", "approval", "site purpose", "financials"],
        "task_signals": ["supplier site setup"],
        "preferred_modules": ["Payables", "Procurement"],
        "correction_note_required": True,
        "confidence_band": "high",
        "quality_score": 0.95,
    },
    {
        "pattern_id": "financials_intercompany_redirect",
        "module": "Financials",
        "source": ("validation", "intercompany-balancing-rules"),
        "title": "Financials troubleshooting: intercompany issues usually resolve in ledger or subledger balancing setup",
        "symptom": "A Financials troubleshooting request references intercompany transaction errors, balancing failures, or intercompany accounts that can't be derived.",
        "root_causes": [
            "Intercompany balancing rules aren't defined at the right level for the participating ledgers, legal entities, or balancing segments.",
            "The transaction depends on intercompany account derivation, but the receiving subledger or ledger setup doesn't support the scenario.",
        ],
        "resolution": [
            "Review the intercompany balancing rules and template intercompany receivables or payables accounts used by the participating entities.",
            "Confirm the transaction is being processed in the correct subledger or ledger flow for the intercompany scenario.",
            "After the balancing setup is corrected, reprocess the intercompany transaction and verify that balancing lines can be derived.",
        ],
        "symptom_terms": ["intercompany transaction errors", "intercompany balancing rules", "balancing lines", "intercompany accounts"],
        "error_keywords": ["intercompany", "balancing", "receivables account", "payables account", "financials"],
        "task_signals": ["intercompany transaction"],
        "preferred_modules": ["Payables", "Receivables", "General Ledger", "Projects"],
        "correction_note_required": True,
        "confidence_band": "high",
        "quality_score": 0.95,
    },
    {
        "pattern_id": "procurement_supplier_site_change_control",
        "module": "Procurement",
        "source": ("validation", "how-you-approve-changes-Made-to-sites-in-a-supplier-profile"),
        "title": "Procurement troubleshooting: supplier site setup or site change request errors",
        "symptom": "Supplier site setup fails, site changes stay pending, or supplier site updates don't progress through approval.",
        "root_causes": [
            "Supplier site changes triggered internal change control and are waiting for approval.",
            "Core site attributes or site purposes changed and require approval routing before the site can be updated.",
            "Required site details under General, Purchasing, Receiving, or Invoicing are incomplete or inconsistent.",
        ],
        "resolution": [
            "Review the supplier profile change request for the affected site and confirm which attributes triggered approval.",
            "Verify the site purposes and core site attributes are complete and align with the intended supplier site setup.",
            "Complete or approve the pending site change request, then retry the supplier site update after the workflow is cleared.",
        ],
        "symptom_terms": ["supplier site setup errors", "supplier site", "site change request", "supplier profile"],
        "error_keywords": ["supplier site", "change control", "approval", "site purpose", "supplier profile"],
        "task_signals": ["supplier site setup"],
        "preferred_modules": ["Payables", "Procurement"],
        "correction_note_required": True,
        "confidence_band": "high",
        "quality_score": 0.93,
    },
    {
        "pattern_id": "payables_intercompany_invoice_validation",
        "module": "Payables",
        "source": ("validation", "what-is-the-intercompany-checkbox-on-the-ap-invoice-page"),
        "title": "Payables troubleshooting: intercompany invoice or cross-charge errors",
        "symptom": "An intercompany invoice in Payables doesn't validate as expected or posts to the wrong liability account.",
        "root_causes": [
            "The supplier doesn't have the required intercompany setup against the provider organization.",
            "Invoice validation overrides the liability account based on intercompany balancing rules, but the balancing setup is incomplete or incorrect.",
            "The invoice should use intercompany cross-charge handling but the setup doesn't support the intended accounting flow.",
        ],
        "resolution": [
            "Confirm the supplier is configured with the expected intercompany setup against the provider organization.",
            "Review the Intercompany Balancing Rules that derive the intercompany payables account during invoice validation.",
            "If the scenario is cross-charge rather than a standard intercompany invoice, verify the invoice is using the correct intercompany treatment before revalidating.",
        ],
        "symptom_terms": ["intercompany transaction errors", "intercompany invoice", "cross charge", "invoice validation"],
        "error_keywords": ["intercompany", "invoice validation", "liability account", "cross charge", "balancing rules"],
        "task_signals": ["intercompany transaction"],
        "preferred_modules": ["Payables", "Receivables", "General Ledger", "Projects"],
        "correction_note_required": True,
        "confidence_band": "high",
        "quality_score": 0.91,
    },
    {
        "pattern_id": "payables_journal_approval_triage",
        "module": "Payables",
        "source": ("validation", "workflow-rule-templates-for-journal-approval"),
        "title": "Payables troubleshooting: journal approval workflow errors require rule and hierarchy checks",
        "symptom": "A Payables troubleshooting request references journal approval failures, approval rules that don't route correctly, or approval hierarchy issues.",
        "root_causes": [
            "The configured approval rules or uploaded workflow template don't reflect the intended Payables routing policy.",
            "The approval hierarchy or rule conditions don't match the submitter, supervisor chain, or approval scope used by the Payables process.",
        ],
        "resolution": [
            "Review the approval rule template or configured approval rules tied to the journal approval flow.",
            "Confirm the hierarchy logic, rule conditions, and routing policy match the intended Payables approval path.",
            "Update and reupload the rule template or correct the approval setup, then resubmit the affected transaction.",
        ],
        "symptom_terms": ["journal approval errors", "journal approval", "workflow rules", "approval hierarchy"],
        "error_keywords": ["journal approval", "workflow rules", "approval routing", "payables", "hierarchy"],
        "task_signals": ["journal approval"],
        "preferred_modules": ["General Ledger", "Receivables", "Payables"],
        "correction_note_required": True,
        "confidence_band": "high",
        "quality_score": 0.95,
    },
    {
        "pattern_id": "payables_expense_report_audit_redirect",
        "module": "Payables",
        "source": ("validation", "audit-actions"),
        "title": "Payables troubleshooting: expense report audit issues must be resolved in Expenses",
        "symptom": "A Payables troubleshooting request references expense report audit errors, audit holds, or repeated audit actions on expense reports.",
        "root_causes": [
            "Expense report audit actions are handled in Expenses rather than Payables invoice processing.",
            "The report remains in audit because the required audit action or approval follow-up hasn't been completed in Expenses.",
        ],
        "resolution": [
            "Treat the issue as an Expenses audit workflow problem rather than a Payables processing problem.",
            "Open Audit Expense Report and review the last audit action taken on the report.",
            "Complete the required audit action or receipt action, then resubmit the report after the audit hold is cleared.",
        ],
        "symptom_terms": ["expense report audit errors", "audit expense report", "request more information", "release hold"],
        "error_keywords": ["expense", "audit", "report", "payables", "hold", "request information"],
        "task_signals": ["expense report audit"],
        "preferred_modules": ["Expenses"],
        "correction_note_required": True,
        "confidence_band": "high",
        "quality_score": 0.96,
    },
    {
        "pattern_id": "general_ledger_intercompany_balancing",
        "module": "General Ledger",
        "source": ("validation", "intercompany-balancing-rules"),
        "title": "General Ledger troubleshooting: intercompany transactions fail because balancing rules are incomplete",
        "symptom": "Intercompany transactions or journals fail because the balancing lines or intercompany accounts can't be derived correctly.",
        "root_causes": [
            "Intercompany balancing rules aren't defined at the right level for the ledger, legal entity, or balancing segment.",
            "The intercompany receivables or payables template accounts aren't set up correctly.",
            "The transaction relies on intercompany account generation, but the balancing setup doesn't support the entities in the transaction.",
        ],
        "resolution": [
            "Review Intercompany Balancing Rules at the primary balancing segment, legal entity, ledger, and chart of accounts levels.",
            "Verify the template intercompany receivables and payables accounts used to build balancing lines.",
            "Reprocess the intercompany transaction after correcting the rule level or template account setup.",
        ],
        "symptom_terms": ["intercompany transaction errors", "intercompany balancing rules", "balancing lines", "intercompany accounts"],
        "error_keywords": ["intercompany", "balancing rules", "receivables account", "payables account", "legal entity", "ledger"],
        "task_signals": ["intercompany transaction"],
        "preferred_modules": ["Payables", "Receivables", "General Ledger", "Projects"],
        "correction_note_required": True,
        "confidence_band": "high",
        "quality_score": 0.95,
    },
    {
        "pattern_id": "general_ledger_journal_approval_rules",
        "module": "General Ledger",
        "source": ("validation", "workflow-rule-templates-for-journal-approval"),
        "title": "General Ledger troubleshooting: journal approval workflow rules don't route correctly",
        "symptom": "Journal approval fails, routes incorrectly, or doesn't follow the expected hierarchy or approval policy.",
        "root_causes": [
            "The spreadsheet-based workflow rules don't reflect the intended journal approval policy.",
            "The journal approval rule template or sample rules weren't updated correctly before upload.",
            "The approval hierarchy in the configured rule doesn't match the intended submitter or supervisor chain.",
        ],
        "resolution": [
            "Review the Journal Approval Rules worksheet in the workflow rule template used for journal approval.",
            "Confirm the rule blocks, hierarchy logic, and approval policy conditions match the intended routing.",
            "Update or reupload the workflow rule template and resubmit the journal after the approval rules are corrected.",
        ],
        "symptom_terms": ["journal approval errors", "journal approval", "workflow rules", "approval hierarchy"],
        "error_keywords": ["journal approval", "workflow rules", "spreadsheet", "rule template", "approval routing"],
        "task_signals": ["journal approval"],
        "preferred_modules": ["General Ledger", "Receivables", "Payables"],
        "correction_note_required": True,
        "confidence_band": "high",
        "quality_score": 0.94,
    },
    {
        "pattern_id": "receivables_journal_approval_triage",
        "module": "Receivables",
        "source": ("validation", "workflow-rule-templates-for-journal-approval"),
        "title": "Receivables troubleshooting: journal approval workflow errors require rule and hierarchy checks",
        "symptom": "A Receivables troubleshooting request references journal approval failures, approval rules that don't route correctly, or approval hierarchy issues.",
        "root_causes": [
            "The approval rule configuration or uploaded workflow template doesn't reflect the intended routing policy for the transaction.",
            "The approval hierarchy or rule conditions don't match the submitter, supervisor chain, or rule scope used by the Receivables process.",
        ],
        "resolution": [
            "Review the approval rule template or configured approval rules tied to the journal approval flow.",
            "Confirm the hierarchy logic, rule conditions, and routing policy match the intended Receivables approval path.",
            "Update and reupload the rule template or correct the approval setup, then resubmit the affected transaction.",
        ],
        "symptom_terms": ["journal approval errors", "journal approval", "workflow rules", "approval hierarchy"],
        "error_keywords": ["journal approval", "workflow rules", "approval routing", "receivables", "hierarchy"],
        "task_signals": ["journal approval"],
        "preferred_modules": ["General Ledger", "Receivables", "Payables"],
        "correction_note_required": True,
        "confidence_band": "high",
        "quality_score": 0.95,
    },
    {
        "pattern_id": "receivables_expense_report_audit_redirect",
        "module": "Receivables",
        "source": ("validation", "audit-actions"),
        "title": "Receivables troubleshooting: expense report audit issues must be resolved in Expenses",
        "symptom": "A Receivables troubleshooting request references expense report audit errors, audit holds, or repeated audit actions on expense reports.",
        "root_causes": [
            "Expense report audit actions are handled in Expenses rather than Receivables processing.",
            "The report remains in audit because the required audit action or approval follow-up hasn't been completed in Expenses.",
        ],
        "resolution": [
            "Treat the issue as an Expenses audit workflow problem rather than a Receivables processing problem.",
            "Open Audit Expense Report and review the last audit action taken on the report.",
            "Complete the required audit action or receipt action, then resubmit the report after the audit hold is cleared.",
        ],
        "symptom_terms": ["expense report audit errors", "audit expense report", "request more information", "release hold"],
        "error_keywords": ["expense", "audit", "report", "receivables", "hold", "request information"],
        "task_signals": ["expense report audit"],
        "preferred_modules": ["Expenses"],
        "correction_note_required": True,
        "confidence_band": "high",
        "quality_score": 0.96,
    },
    {
        "pattern_id": "receivables_intercompany_balancing",
        "module": "Receivables",
        "source": ("validation", "intercompany-balancing-rules"),
        "title": "Receivables troubleshooting: intercompany transaction errors require balancing-rule checks",
        "symptom": "A Receivables troubleshooting request references intercompany transaction failures, balancing issues, or intercompany accounts that can't be derived.",
        "root_causes": [
            "The Receivables transaction depends on intercompany balancing rules or template accounts that are incomplete or incorrect.",
            "The participating legal entities or ledgers aren't supported by the current intercompany balancing configuration.",
        ],
        "resolution": [
            "Review the intercompany balancing rules and template receivables or payables accounts used for the transaction.",
            "Verify the entities, ledgers, and balancing segments involved in the Receivables flow are supported by the current setup.",
            "After correcting the balancing setup, reprocess the intercompany transaction and confirm the balancing lines are created correctly.",
        ],
        "symptom_terms": ["intercompany transaction errors", "intercompany balancing rules", "balancing lines", "intercompany accounts"],
        "error_keywords": ["intercompany", "balancing rules", "receivables", "legal entity", "ledger"],
        "task_signals": ["intercompany transaction"],
        "preferred_modules": ["Payables", "Receivables", "General Ledger", "Projects"],
        "correction_note_required": True,
        "confidence_band": "high",
        "quality_score": 0.95,
    },
    {
        "pattern_id": "procurement_catalog_upload_validation",
        "module": "Procurement",
        "source": ("validation", "how-agreement-lines-are-processed"),
        "title": "Procurement troubleshooting: catalog or agreement line upload fails validation",
        "symptom": "Catalog upload or agreement line upload fails because the loader raises parsing, formatting, or validation errors.",
        "root_causes": [
            "The upload file contains invalid language, line number, or required description values.",
            "The Agreement Loader detects parsing or formatting errors before the lines can be processed.",
            "External mapping translations, catalog attribute names, or data validations fail during preliminary checks.",
        ],
        "resolution": [
            "Review the Agreement Loader output and identify the exact parsing or validation errors in the upload file.",
            "Correct invalid line numbers, language values, descriptions, or other formatting issues in the source file.",
            "If external mappings or catalog attribute validations fail, fix those mappings and rerun the upload after the file passes preliminary validation.",
        ],
        "symptom_terms": ["catalog upload", "agreement loader", "upload errors", "validation errors"],
        "error_keywords": ["catalog", "upload", "agreement loader", "validation", "parsing", "formatting", "mapping"],
        "task_signals": ["catalog upload"],
        "preferred_modules": ["Procurement"],
        "correction_note_required": False,
        "confidence_band": "high",
        "quality_score": 0.96,
    },
]

PATTERN_SPECS.extend(RESIDUAL_PATTERN_SPECS)

RESIDUAL_FAILURE_BUCKETS = {
    "OF-0176": {
        "failure_bucket": "missing exact-module troubleshooting docs",
        "root_cause": "No retained Payables intercompany troubleshooting grounding survived the exact-module filter, so the request failed closed.",
    },
    "OF-0183": {
        "failure_bucket": "missing exact-module troubleshooting docs",
        "root_cause": "No retained Receivables journal approval troubleshooting grounding survived task-semantic filtering.",
    },
    "OF-0186": {
        "failure_bucket": "missing exact-module troubleshooting docs",
        "root_cause": "No retained Receivables intercompany troubleshooting grounding survived the exact-module and task-semantic gates.",
    },
    "OF-0196": {
        "failure_bucket": "missing exact-module troubleshooting docs",
        "root_cause": "No retained General Ledger intercompany troubleshooting grounding passed task-semantic matching.",
    },
    "OF-0210": {
        "failure_bucket": "weak symptom/root-cause mapping",
        "root_cause": "Cash Management only retrieved reconciliation troubleshooting, while the query actually needed Expenses audit guidance plus a correction note.",
    },
    "OF-0474": {
        "failure_bucket": "missing task-semantic coverage",
        "root_cause": "Procurement retrieved generic approval content instead of Agreement Loader catalog-upload validation guidance.",
    },
    "OF-0165": {
        "failure_bucket": "ambiguous business-task mismatch",
        "root_cause": "The system answered from Procurement content without an explicit correction note for the Financials family query.",
    },
    "OF-0166": {
        "failure_bucket": "overly strict refusal",
        "root_cause": "The task-semantic gate rejected the family-level intercompany query even though preferred-module intercompany grounding exists in sibling Financials modules.",
    },
}


def _source_payload(rows: List[Dict[str, Any]], spec: Dict[str, Any]) -> Dict[str, Any]:
    source = spec["source"]
    if source[0] == "validation":
        return _validation_row_by_fragment(rows, source[1])
    if source[0] == "pdf":
        return _extract_pdf_pages(Path(source[1]), list(source[2]), spec["title"])
    raise ValueError(f"Unsupported source type: {source[0]}")


def _pattern_content(spec: Dict[str, Any], source_doc: Dict[str, Any]) -> str:
    lines = [
        f"Symptom: {spec['symptom']}",
        "",
        "Root cause:",
    ]
    lines.extend(f"- {item}" for item in spec["root_causes"])
    lines.extend(
        [
            "",
            "Resolution:",
        ]
    )
    lines.extend(f"{index}. {step}" for index, step in enumerate(spec["resolution"], start=1))
    lines.extend(
        [
            "",
            "Grounding excerpt:",
            _preview(source_doc["content"], 1400),
        ]
    )
    return "\n".join(lines)


def _manifest_record(spec: Dict[str, Any], source_doc: Dict[str, Any]) -> Dict[str, Any]:
    content = _pattern_content(spec, source_doc)
    content_hash = stable_hash("troubleshooting_corpus", content)
    source_doc_key = stable_hash(source_doc["source_uri"], source_doc["title"])[:24]
    title = spec["title"]
    return IngestionManifestRecord(
        source_path=source_doc["source_path"],
        source_uri=source_doc["source_uri"],
        title=title,
        module=spec["module"],
        task_type="troubleshooting",
        doc_type=DocType.TROUBLESHOOTING_DOC,
        trusted_schema_objects=[],
        quality_score=float(spec["quality_score"]),
        content_hash=content_hash,
        source_system=SourceSystem.ORACLE_DOCS,
        corpus=None,
        content=content,
        canonical_uri=source_doc["source_uri"],
        authority_tier=AuthorityTier.OFFICIAL,
        metadata={
            "source_uri": source_doc["source_uri"],
            "canonical_uri": source_doc["source_uri"],
            "authority_tier": AuthorityTier.OFFICIAL.value,
            "symptom": spec["symptom"],
            "probable_cause": spec["root_causes"],
            "root_cause": spec["root_causes"],
            "resolution": spec["resolution"],
            "source_doc": {
                "title": source_doc["title"],
                "source_uri": source_doc["source_uri"],
                "source_kind": source_doc["source_kind"],
            },
            "source_doc_key": source_doc_key,
            "symptom_terms": spec["symptom_terms"],
            "error_keywords": spec["error_keywords"],
            "task_signals": spec.get("task_signals", []),
            "preferred_modules": spec.get("preferred_modules", []),
            "correction_note_required": bool(spec.get("correction_note_required")),
            "derived_from_doc": True,
            "confidence_band": spec["confidence_band"],
            "module_family": "Financials",
            "title": title,
            "summary": _preview(spec["symptom"]),
        },
    ).model_dump(mode="json")


def _write_json(path: Path, payload: Dict[str, Any] | List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _index_manifest_rows(manifest_rows: List[Dict[str, Any]], tenant_id: str, reset_index: bool) -> Dict[str, Any]:
    faiss = FaissIndex(tenant_id=tenant_id, indexes_dir=str(INDEXES_DIR), corpus="troubleshooting_corpus")
    if reset_index:
        faiss.reset()

    chunks: List[Dict[str, Any]] = []
    for payload in manifest_rows:
        record = IngestionManifestRecord(**payload)
        document = CuratedIngestionValidator.build_document(
            source_path=record.source_path,
            source_uri=record.source_uri or record.source_path,
            title=record.title,
            module=record.module,
            task_type=record.task_type,
            doc_type=record.doc_type,
            trusted_schema_objects=record.trusted_schema_objects,
            quality_score=record.quality_score,
            source_system=record.source_system,
            content=record.content or "",
            metadata=record.metadata,
        )
        chunk = CuratedIngestionValidator.build_chunk(document, document.content, 0)
        chunks.append(CuratedIngestionValidator.chunk_payload(chunk))

    if chunks:
        faiss.add_chunks_list(chunks)
    return faiss.stats()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_residual_failure_report(
    results_path: Path = RESIDUAL_BENCHMARK_RESULTS,
    exact_results_path: Path = RESIDUAL_EXACT_RESULTS,
    output_path: Path | None = None,
) -> Dict[str, Any]:
    output_path = output_path or (OUTPUT_ROOT / "residual_failure_report.json")
    results = _read_jsonl(results_path)
    exact_results = _read_jsonl(exact_results_path)
    exact_ids = {row.get("id") for row in exact_results}
    failures: List[Dict[str, Any]] = []

    for row in results:
        case_id = str(row.get("id") or "")
        if row.get("scoring_outcome") == "grounded_correct":
            continue
        metadata = RESIDUAL_FAILURE_BUCKETS.get(case_id)
        if not metadata:
            continue
        failures.append(
            {
                "case_id": case_id,
                "module": row.get("benchmark_module") or row.get("module"),
                "intent": row.get("benchmark_intent") or row.get("intent_detected"),
                "failure_type": row.get("scoring_outcome"),
                "retrieved_docs": [
                    {
                        "title": doc.get("title"),
                        "module": doc.get("module"),
                        "task_match_strength": doc.get("task_match_strength"),
                        "score": doc.get("score"),
                    }
                    for doc in (row.get("retrieved_docs") or [])[:3]
                ],
                "task_match": row.get("benchmark_task_signal") or "none",
                "module_match": (
                    "exact"
                    if (row.get("module_detected") or row.get("module")) == (row.get("benchmark_module") or row.get("module"))
                    else "fallback_or_mismatch"
                ),
                "refusal_or_wrong_answer": "refusal" if row.get("rejected") else "wrong_answer",
                "root_cause": metadata["root_cause"],
                "failure_bucket": metadata["failure_bucket"],
                "in_exact_slice": case_id in exact_ids,
                "question": row.get("question"),
            }
        )

    grouped: Dict[str, List[str]] = {}
    for failure in failures:
        grouped.setdefault(failure["failure_bucket"], []).append(failure["case_id"])

    report = {
        "results_path": str(results_path),
        "exact_results_path": str(exact_results_path),
        "failure_count": len(failures),
        "failures": failures,
        "grouped_failure_counts": {
            bucket: {
                "count": len(case_ids),
                "case_ids": sorted(case_ids),
            }
            for bucket, case_ids in sorted(grouped.items())
        },
    }
    _write_json(output_path, report)
    return report


def build_troubleshooting_completion(tenant_id: str = "demo", reset_index: bool = True) -> Dict[str, Any]:
    rows = _read_validation_rows()
    manifest_rows: List[Dict[str, Any]] = []
    source_docs: Dict[str, Dict[str, Any]] = {}
    modules = sorted({spec["module"] for spec in PATTERN_SPECS})

    for spec in PATTERN_SPECS:
        source_doc = _source_payload(rows, spec)
        source_key = stable_hash(source_doc["source_uri"], source_doc["title"])
        source_docs[source_key] = {
            "title": source_doc["title"],
            "source_uri": source_doc["source_uri"],
            "source_path": source_doc["source_path"],
            "source_kind": source_doc["source_kind"],
            "module": spec["module"],
        }
        manifest_rows.append(_manifest_record(spec, source_doc))

    manifest_rows.sort(key=lambda row: (row["module"], row["title"], row["content_hash"]))
    source_doc_rows = sorted(source_docs.values(), key=lambda row: (row["module"], row["title"], row["source_uri"]))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / "troubleshooting_corpus.jsonl"
    source_docs_path = OUTPUT_ROOT / "source_docs.json"
    summary_path = OUTPUT_ROOT / "summary.json"
    residual_report_path = OUTPUT_ROOT / "residual_failure_report.json"

    _write_jsonl(manifest_path, manifest_rows)
    _write_json(source_docs_path, source_doc_rows)
    index_stats = _index_manifest_rows(manifest_rows, tenant_id=tenant_id, reset_index=reset_index)
    residual_report = build_residual_failure_report(output_path=residual_report_path)

    summary = {
        "modules_covered": modules,
        "source_doc_count": len(source_doc_rows),
        "pattern_count": len(manifest_rows),
        "manifest_path": str(manifest_path),
        "source_docs_path": str(source_docs_path),
        "residual_pattern_count": len(RESIDUAL_PATTERN_SPECS),
        "residual_failure_report_path": str(residual_report_path),
        "residual_failure_count": residual_report["failure_count"],
        "index_stats": index_stats,
    }
    _write_json(summary_path, summary)
    return summary
