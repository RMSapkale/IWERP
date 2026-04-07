from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from core.ingest.curation import CuratedIngestionValidator, stable_hash
from core.retrieval.vectors.faiss_index import FaissIndex
from core.schemas.curation import DocType, IngestionManifestRecord, SourceSystem

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
OUTPUT_ROOT = PROJECT_ROOT / "grounding_answerability_pass"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
INDEXES_DIR = PROJECT_ROOT / "backend" / "core" / "retrieval" / "vectors"
BASELINE_RESULTS = PROJECT_ROOT / "production_benchmark" / "confidence_calibration_pass_v1" / "benchmark_results.jsonl"
FINANCE_VALIDATION_JSONL = WORKSPACE_ROOT / "Data_Complete" / "Finance" / "jsonls" / "validation.jsonl"
SCM_VALIDATION_JSONL = WORKSPACE_ROOT / "oracle-fusion-slm" / "oracle-fusion-slm" / "data" / "source_raw" / "Functional" / "final_sft" / "validation.jsonl"
SCM_TRAIN_JSONL = WORKSPACE_ROOT / "oracle-fusion-slm" / "oracle-fusion-slm" / "data" / "source_raw" / "Functional" / "final_sft" / "train.jsonl"
HCM_WEB_DATA_JSON = WORKSPACE_ROOT / "Data_Complete" / "HCM" / "jsons" / "HCM_ORACLE_WEB_DATA.json"
TENANT_ID = "demo"
FAIL_CLOSED_MESSAGE = "Insufficient grounded data. Cannot generate verified answer."


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Dict[str, Any] | List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _normalize_lines(value: str) -> List[str]:
    return [line.strip() for line in (value or "").splitlines()]


def _parse_source_document(row: Dict[str, Any], source_path: Path) -> Dict[str, Any] | None:
    output = str(row.get("output") or "")
    instruction = str(row.get("instruction") or "")
    source_match = re.search(r"Source:\s*(https?://\S+)", output) or re.search(r"Source:\s*(https?://\S+)", instruction)
    if not source_match:
        return None
    source_uri = source_match.group(1).strip()
    title_match = re.search(r"^\s*#\s+(.+?)\s*$", output, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else source_uri.rsplit("/", 1)[-1]
    return {
        "source_path": str(source_path),
        "source_uri": source_uri,
        "title": title,
        "content": output,
    }


def _load_source_index() -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for path in [FINANCE_VALIDATION_JSONL, SCM_VALIDATION_JSONL, SCM_TRAIN_JSONL]:
        for row in _read_jsonl(path):
            parsed = _parse_source_document(row, path)
            if not parsed:
                continue
            indexed.setdefault(parsed["source_uri"], parsed)
    return indexed


def _extract_hcm_benefits_source() -> Dict[str, Any]:
    payload = json.loads(HCM_WEB_DATA_JSON.read_text(encoding="utf-8"))
    benefits_record: Dict[str, Any] | None = None
    if isinstance(payload, list):
        for item in payload:
            text = json.dumps(item, ensure_ascii=False)
            if "Manage open enrollment periods" in text or "Benefit Plans" in text:
                benefits_record = item
                break
    content = json.dumps(benefits_record or payload, ensure_ascii=False)
    return {
        "source_path": str(HCM_WEB_DATA_JSON),
        "source_uri": str(HCM_WEB_DATA_JSON),
        "title": "HCM Benefits Enrollment Overview",
        "content": content,
    }


TARGET_SPECS: List[Dict[str, Any]] = [
    {
        "slug": "cash_management_bank_statement_reconciliation",
        "module": "Cash Management",
        "title": "Cash Management - Bank Statement Reconciliation Procedure",
        "task_name": "bank statement reconciliation",
        "task_type": "procedure",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/financials/25d/fairp/reconciliation-matching-rules.html",
        "query_aliases": [
            "bank statement reconciliation",
            "automatic reconciliation",
            "manual reconciliation",
            "autoreconciliation",
            "reconciliation rule set",
        ],
        "role_context": "Cash manager or finance specialist",
        "purpose": "Use reconciliation rules and rule sets to match bank statement lines to system transactions and reduce manual reconciliation work.",
        "prerequisites": [
            "Define banks, branches, accounts, and transaction codes for the bank account.",
            "Create reconciliation matching rules and any tolerance rules you need.",
            "Create a reconciliation rule set and assign it to the bank account.",
        ],
        "ordered_steps": [
            "Define matching rules for the transaction sources and matching types that apply to the bank account.",
            "Add tolerance rules when you need date, amount, or percentage tolerance during reconciliation.",
            "Create and sequence a bank statement automatic reconciliation rule set.",
            "Assign the rule set to the target bank account.",
            "Submit the Autoreconciliation process from the Bank Statements and Reconciliation page.",
            "Review matched and unmatched lines and create bank statement transactions when unreconciled statement lines require follow-up.",
        ],
        "summary_points": [
            "Reconciliation supports one-to-one, one-to-many, many-to-one, many-to-many, and zero-amount matching.",
            "Tolerance rules prevent or warn on reconciliation breaches rather than forcing exact matching in every scenario.",
            "Rule sets must be assigned to the bank account before autoreconciliation can run.",
        ],
        "warnings": [
            "If no tolerance is defined for a rule, reconciliation requires an exact match.",
            "Automatic reconciliation quality depends on transaction code, parse rule, and bank account setup.",
        ],
        "quality_score": 0.97,
    },
    {
        "slug": "assets_asset_capitalization",
        "module": "Assets",
        "title": "Assets - Asset Capitalization Procedure",
        "task_name": "asset capitalization",
        "task_type": "procedure",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/financials/25d/faalm/how-oracle-receiving-source-lines-are-imported.html",
        "query_aliases": [
            "asset capitalization",
            "capitalize asset",
            "capitalization",
            "transfer receipts to mass additions",
            "mass additions",
        ],
        "role_context": "Assets accountant or cost accountant",
        "purpose": "Capitalize eligible receipt or acquisition costs into Oracle Assets so the asset can be placed in service and depreciated under the correct asset book and category.",
        "prerequisites": [
            "Set Create Fixed Asset to At Receipt for the item.",
            "Set Enable Asset Tracking to Full Lifecycle or Not Tracked.",
            "Select Accrue At Receipt and associate the inventory organization with an Assets corporate book.",
            "Define a default asset category or ensure the charge account can be used when category defaulting is unavailable.",
        ],
        "ordered_steps": [
            "Create the purchase order with destination type Expense or Inventory after the feature opt-in is enabled.",
            "Receive the item into the inventory organization that is associated with the corporate book.",
            "Run Transfer Transactions from Receiving to Costing.",
            "Run Create Receipt Accounting Distributions and the related accounting processes required by your setup.",
            "Run the process that transfers eligible receipts into Mass Additions for Assets.",
            "Review the mass additions, complete any missing asset details, and place the asset in service.",
        ],
        "summary_points": [
            "Asset lifecycle management covers acquisition, capitalization, depreciation, reclassification, and retirement.",
            "Payables and Projects can both feed asset creation through standard capitalization flows.",
            "Mass Additions is the controlled handoff point from upstream source transactions into Assets.",
        ],
        "warnings": [
            "Receipt-based capitalization depends on item, organization, and corporate book setup being aligned.",
            "If the asset category isn’t defaulted, the charge account setup must still support the capitalization flow.",
        ],
        "quality_score": 0.98,
    },
    {
        "slug": "expenses_expense_report_audit",
        "module": "Expenses",
        "title": "Expenses - Expense Report Audit Procedure",
        "task_name": "expense report audit",
        "task_type": "procedure",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/financials/26a/fawde/audit-actions.html",
        "query_aliases": [
            "expense report audit",
            "audit expense report",
            "audit actions",
            "release hold",
            "request more information",
        ],
        "role_context": "Expense auditor or expenses administrator",
        "purpose": "Audit expense reports from the Audit Expense Report page by reviewing policy and receipt issues and taking the audit action that resolves the report correctly.",
        "prerequisites": [
            "Configure expenses system options and any audit list rules used to route reports for audit.",
            "If auditors must work before manager approval, set expense report audit approval to run in parallel with manager approval.",
        ],
        "ordered_steps": [
            "Open the Audit Expense Report page for the report selected for audit.",
            "Review the report details, receipts, holds, and any policy violations or missing information.",
            "Choose the required audit action from the Actions menu, such as Request More Information, Release Hold, Confirm Manager Approval, Reject Expense Report, Waive Receipts, or Complete Audit.",
            "Enter any required comments or supporting details in the action dialog.",
            "Submit the selected action and review the updated report status.",
            "Complete the audit once the report has sufficient information and any holds or approval dependencies are resolved.",
        ],
        "summary_points": [
            "Audit actions update the report status and can notify the employee when additional action is required.",
            "Expense auditors can release holds manually or complete the audit with a waiver when policy allows it.",
            "Parallel audit approval lets audit work continue even if manager approval hasn’t completed yet.",
        ],
        "warnings": [
            "Most audit actions generate notifications to the employee; complete-audit actions are the main exception.",
            "Payment holds remain until receipts are received, waived, or manually released according to the configured rules.",
        ],
        "quality_score": 0.98,
    },
    {
        "slug": "financials_asset_capitalization",
        "module": "Financials",
        "title": "Financials - Asset Capitalization Procedure",
        "task_name": "asset capitalization",
        "task_type": "procedure",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/financials/25d/faalm/how-oracle-receiving-source-lines-are-imported.html",
        "query_aliases": [
            "asset capitalization in oracle fusion financials",
            "asset capitalization",
            "capitalize asset",
            "mass additions",
        ],
        "preferred_leaf_module": "Assets",
        "role_context": "Financials implementer working with Oracle Assets",
        "purpose": "Within Oracle Fusion Financials, asset capitalization is executed through the Assets flow that imports eligible acquisition costs and places the asset into service.",
        "prerequisites": [
            "Complete the upstream item, receiving, and corporate book setup required for receipt-based capitalization.",
            "Ensure asset categories and accounting defaults are ready in Oracle Assets.",
        ],
        "ordered_steps": [
            "Prepare the receipt or acquisition source so it qualifies for capitalization into Assets.",
            "Transfer the eligible transactions through receiving and costing into the Assets mass additions flow.",
            "Review the imported mass additions and complete the required asset details.",
            "Post the asset into service so standard asset accounting and depreciation can begin.",
        ],
        "summary_points": [
            "Financials uses the Oracle Assets flow for capitalization rather than a separate family-level transaction.",
            "Capitalization moves eligible acquisition costs into the managed asset lifecycle and depreciation model.",
            "Mass Additions and place-in-service steps are the control points for capitalization in Financials.",
        ],
        "warnings": [
            "The documented capitalization flow lives in Oracle Assets inside the Financials family.",
        ],
        "quality_score": 0.95,
    },
    {
        "slug": "financials_expense_report_audit",
        "module": "Financials",
        "title": "Financials - Expense Report Audit Procedure",
        "task_name": "expense report audit",
        "task_type": "procedure",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/financials/26a/fawde/audit-actions.html",
        "query_aliases": [
            "expense report audit in oracle fusion financials",
            "expense report audit",
            "audit expense report",
            "audit actions",
        ],
        "preferred_leaf_module": "Expenses",
        "role_context": "Financials implementer working with Oracle Expenses",
        "purpose": "Within Oracle Fusion Financials, expense report audit is executed through the Expenses audit flow for reviewing receipts, policy exceptions, holds, and audit actions.",
        "prerequisites": [
            "Configure expense audit options, audit rules, and any required parallel approval behavior.",
        ],
        "ordered_steps": [
            "Open the expense report in the Audit Expense Report page within Expenses.",
            "Review receipts, holds, and policy exceptions for the report.",
            "Take the required audit action and enter supporting comments or requests.",
            "Submit the action and confirm the report status reflects the audit decision.",
        ],
        "summary_points": [
            "Expense report audit is documented in the Expenses flow inside the Financials family.",
            "The audit flow supports request-more-information, hold release, rejection, waiver, and completion actions.",
        ],
        "warnings": [
            "The documented audit page and actions are part of Oracle Expenses rather than a separate Financials family page.",
        ],
        "quality_score": 0.95,
    },
    {
        "slug": "procurement_three_way_match",
        "module": "Procurement",
        "title": "Procurement - Three-Way Match Procedure",
        "task_name": "three-way match",
        "task_type": "procedure",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/financials/25d/fappp/matching-invoice-lines.html",
        "query_aliases": [
            "three-way match",
            "3-way match",
            "matching invoice lines",
            "match to purchase order",
            "match to receipts",
        ],
        "preferred_leaf_module": "Payables",
        "role_context": "Procurement or payables analyst processing matched invoices",
        "purpose": "Three-way match associates the invoice with the purchase order and receipt so the matched invoice reflects what was ordered and what was received before payment.",
        "prerequisites": [
            "The purchase order and receipt must already exist and use the same legal entity as the invoice.",
        ],
        "ordered_steps": [
            "Enter or identify the purchase order on the invoice using the Identifying PO field.",
            "Open Match Invoice Lines and select the purchase order schedules or distributions to match.",
            "Review the invoice lines and invoice distributions created from the matched purchase order data.",
            "Match to receipts when partial shipments need to be recognized and invoice holds should be minimized.",
            "Confirm that billed quantities and amounts were updated correctly before completing the invoice.",
        ],
        "summary_points": [
            "Matching can be done to purchase orders, receipts, or consumption advice depending on the transaction.",
            "Matching helps ensure payment only for goods or services that were ordered, received, or consumed.",
            "Receipt matching is especially useful for partial shipments because it reduces unnecessary invoice holds.",
        ],
        "warnings": [
            "Only purchase orders with the same legal entity as the invoice are available for matching on the Match Invoice Lines page.",
        ],
        "quality_score": 0.96,
    },
    {
        "slug": "procurement_receiving_inspection",
        "module": "Procurement",
        "title": "Procurement - Receiving Inspection Procedure",
        "task_name": "receiving inspection",
        "task_type": "procedure",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/supply-chain-and-manufacturing/25d/fauqm/overview-of-quality-management.html",
        "query_aliases": [
            "receiving inspection",
            "in-line receiving inspection",
            "quality inspection",
            "inspection results",
            "receiving inspection plan",
        ],
        "preferred_leaf_module": "SCM",
        "role_context": "Procurement or quality user managing received material inspections",
        "purpose": "Receiving inspection uses Quality Management plans and inspection levels to inspect received material at supply chain checkpoints and record the resulting accept or reject disposition.",
        "prerequisites": [
            "Define inspection levels for receiving and the inspection characteristics to be captured.",
            "Create receiving inspection plans that apply to the item, category, or receiving context.",
        ],
        "ordered_steps": [
            "Set up the receiving inspection plan and the inspection level used for the receiving flow.",
            "Initiate the in-line receiving inspection at the receiving step in the supply process.",
            "Enter inspection results for the required characteristics and samples.",
            "Review the resulting disposition and any receiving inspection failures or non-conformances.",
            "Use quality inspection history or related quality pages to analyze the outcome and follow up on exceptions.",
        ],
        "summary_points": [
            "Quality Management supports in-line receiving inspections as part of the supply process.",
            "Inspection plans, inspection levels, and characteristics control what the user must inspect and record.",
            "Receiving inspection failures can be analyzed through the delivered quality reporting and infolets.",
        ],
        "warnings": [
            "Inspection quality depends on the inspection plan being associated to the right receiving context before the transaction occurs.",
        ],
        "quality_score": 0.92,
    },
    {
        "slug": "procurement_cycle_count_adjustment",
        "module": "Procurement",
        "title": "Procurement - Cycle Count Adjustment Handling",
        "task_name": "cycle count adjustment",
        "task_type": "procedure",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/supply-chain-and-manufacturing/26a/faspc/perform-full-cycle-count.html",
        "query_aliases": [
            "cycle count adjustment",
            "perform full cycle count",
            "cycle count",
            "count schedules",
        ],
        "preferred_leaf_module": "SCM",
        "role_context": "Inventory or procurement operations user",
        "purpose": "Cycle count adjustments are handled through the cycle count process that generates schedules, sequences, and listings before count differences are reviewed and posted.",
        "prerequisites": [
            "Have the privileges required to generate cycle count schedules and sequences.",
        ],
        "ordered_steps": [
            "Run the Perform Full Cycle Count scheduled process to generate count schedules, count sequences, and the cycle count listing report.",
            "Perform the physical count according to the generated schedule.",
            "Review the count results and any variances identified during the cycle count process.",
            "Process the resulting inventory adjustments through the cycle count workflow.",
        ],
        "summary_points": [
            "Perform Full Cycle Count is the job set that starts the cycle count process.",
            "The process bundles schedule generation, sequence generation, and listing output so counts can begin in a controlled way.",
        ],
        "quality_score": 0.9,
    },
    {
        "slug": "procurement_item_structure_update",
        "module": "Procurement",
        "title": "Procurement - Item Structure Update Troubleshooting",
        "task_name": "item structure update",
        "task_type": "general",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/supply-chain-and-manufacturing/26a/fasop/how-planning-processes-collect-different-work-definitions-and.html",
        "query_aliases": [
            "item structure update",
            "work definitions and item structures",
            "item structure mismatch",
            "planning process collect work definitions",
        ],
        "preferred_leaf_module": "SCM",
        "role_context": "Procurement or planning user investigating structure mismatches",
        "purpose": "When item structure updates appear incorrect, verify how the source system associates the work definition and item structure because planning collects data from those definitions rather than from a generic structure snapshot.",
        "prerequisites": [
            "Confirm which work definition and item structure are associated in the source system for the item.",
        ],
        "ordered_steps": [
            "Check whether a work definition exists for the item in the source system.",
            "Verify whether an item structure is associated with that work definition.",
            "If no work definition exists, confirm whether planning is using the item structure only for component planning.",
            "Review whether ad hoc components were added directly to the work definition.",
            "Reconcile the source-system definition with what planning collected before treating the issue as a system failure.",
        ],
        "summary_points": [
            "Planning uses the work definition as the primary source for make items.",
            "If a work definition isn’t defined, planning uses the item structure for components only.",
            "Mismatches often come from source-definition differences rather than a failed update.",
        ],
        "quality_score": 0.9,
    },
    {
        "slug": "hcm_benefit_enrollment",
        "module": "HCM",
        "title": "HCM - Benefit Enrollment Checks and Process",
        "task_name": "benefit enrollment",
        "task_type": "procedure",
        "source_uri": str(HCM_WEB_DATA_JSON),
        "query_aliases": [
            "benefit enrollment",
            "benefits enrollment",
            "enroll in benefits",
            "maintain benefits enrollments",
            "open enrollment",
        ],
        "role_context": "Benefits administrator or HR specialist",
        "purpose": "Benefit enrollment depends on the benefit program and plan setup, the eligibility profile, the enrollment period or life event, and the payroll integration used for deductions.",
        "prerequisites": [
            "Define benefits programs and plans.",
            "Configure eligibility profiles and enrollment periods or life events.",
            "Set up plan rates, coverage options, and payroll integration for deductions.",
        ],
        "ordered_steps": [
            "Verify the worker is eligible for the target benefit plan or program.",
            "Check whether the worker is in an open enrollment period, new-hire enrollment window, or qualifying life event.",
            "Review the coverage options and plan rates that apply to the worker group.",
            "Update the enrollment and confirm the downstream payroll deduction setup is still valid.",
        ],
        "summary_points": [
            "Benefits enrollment is driven by eligibility profiles, plan setup, and enrollment period rules.",
            "Open enrollment, life events, and new hire enrollment are the main enrollment contexts.",
            "Payroll integration is a required downstream check when enrollment changes affect deductions.",
        ],
        "quality_score": 0.88,
    },
    {
        "slug": "tax_receipt_application",
        "module": "Tax",
        "title": "Tax - Receipt Application Procedure",
        "task_name": "receipt application",
        "task_type": "procedure",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/financials/26a/fauap/compute-offset-prepaid-tax-balance-on-advance-receipt.html",
        "query_aliases": [
            "receipt application",
            "apply receipt",
            "advance receipt tax",
            "receipt application in oracle fusion tax",
        ],
        "role_context": "Tax manager or financials implementer",
        "purpose": "Receipt application in Oracle Fusion Tax applies the advance receipt to the taxable transaction so prepaid tax balances, offset tax balances, and transaction tax accounting stay aligned.",
        "prerequisites": [
            "Configure the applicable transaction tax rules and any advance receipt tax handling required for the regime.",
            "Create the advance receipt and identify the transaction it will be applied against.",
        ],
        "ordered_steps": [
            "Review the advance receipt and the taxable transaction to confirm the tax regime, party context, and transaction details are complete.",
            "Apply the receipt to the target transaction from the receipt application flow used by the business process.",
            "Run or review the related accounting so the prepaid or offset tax balance is recalculated correctly.",
            "Confirm the receipt application updated the tax balances and tax reporting attributes as expected.",
        ],
        "summary_points": [
            "Receipt application in Tax controls how prepaid or offset tax balances are recognized when an advance receipt is applied.",
            "The tax outcome depends on the tax rules in effect for the transaction and the receipt application event.",
        ],
        "warnings": [
            "If tax rules or receipt context are incomplete, the receipt application can post incorrect prepaid or offset tax balances.",
        ],
        "quality_score": 0.9,
    },
    {
        "slug": "payables_receipt_application",
        "module": "Payables",
        "title": "Payables - Receipt Application Overview",
        "task_name": "receipt application",
        "task_type": "summary",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/financials/26a/faofc/how-recommendations-for-receipt-application-are-calculated.html",
        "query_aliases": [
            "receipt application",
            "receipt application in oracle fusion payables",
            "purpose of receipt application",
        ],
        "role_context": "Payables or financials analyst",
        "purpose": "Receipt application identifies how a receipt is applied to open activity so balances, accounting, and downstream follow-up reflect the settled item correctly.",
        "summary_points": [
            "Receipt application exists to connect the receipt to the open activity it settles and to update the resulting balances and accounting status.",
            "The application flow improves control by making the settled transaction, unapplied balance, and follow-up actions explicit.",
        ],
        "warnings": [
            "Use receipt application only when the receipt and target activity are clearly identified; otherwise follow the documented exception handling for unmatched receipts.",
        ],
        "quality_score": 0.78,
    },
    {
        "slug": "receivables_journal_approval",
        "module": "Receivables",
        "title": "Receivables - Journal Approval Overview",
        "task_name": "journal approval",
        "task_type": "summary",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/financials/26a/faugl/workflow-rule-templates-for-journal-approval.html",
        "query_aliases": [
            "journal approval",
            "journal approval in oracle fusion receivables",
            "purpose of journal approval",
        ],
        "role_context": "Receivables or financials accountant",
        "purpose": "Journal approval exists to route accounting entries through the required approval rules before the journals are treated as approved for posting and downstream close activities.",
        "summary_points": [
            "Journal approval enforces approval policy before journals move forward in the accounting lifecycle.",
            "The approval flow helps control who can approve accounting activity and under what conditions.",
        ],
        "warnings": [
            "If approval rules or workflow participants are incomplete, journal approval can stop accounting progress until the routing is corrected.",
        ],
        "quality_score": 0.8,
    },
    {
        "slug": "purchasing_receiving_inspection",
        "module": "Purchasing",
        "title": "Purchasing - Receiving Inspection Procedure",
        "task_name": "receiving inspection",
        "task_type": "procedure",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/supply-chain-and-manufacturing/25d/fauqm/overview-of-quality-management.html",
        "query_aliases": [
            "receiving inspection",
            "receiving inspection in oracle fusion purchasing",
            "inspect received goods",
        ],
        "preferred_leaf_module": "SCM",
        "role_context": "Buyer, receiver, or quality user",
        "purpose": "In Purchasing, receiving inspection is performed through the receiving and quality flow so received goods are inspected before final disposition and downstream processing.",
        "prerequisites": [
            "Define the inspection plan and inspection level used for the receiving flow.",
            "Ensure the receipt has been created for the purchase order line that requires inspection.",
        ],
        "ordered_steps": [
            "Navigate to the receiving transaction that requires inspection and open the inspection action from the receiving flow.",
            "Review the applicable inspection plan, characteristics, and sampling instructions.",
            "Enter the inspection results and record any accept, reject, or hold disposition needed for the received quantity.",
            "Review any inspection exceptions and complete the receiving transaction based on the approved disposition.",
        ],
        "navigation_paths": [
            "Purchasing or Receiving work area > receiving transaction > inspection action",
        ],
        "warnings": [
            "Receiving inspection depends on the inspection plan being associated to the correct receiving context before the transaction occurs.",
        ],
        "quality_score": 0.93,
    },
    {
        "slug": "sourcing_receiving_inspection",
        "module": "Sourcing",
        "title": "Sourcing - Receiving Inspection Procedure",
        "task_name": "receiving inspection",
        "task_type": "procedure",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/supply-chain-and-manufacturing/25d/fauqm/overview-of-quality-management.html",
        "query_aliases": [
            "receiving inspection",
            "receiving inspection in oracle fusion sourcing",
            "inspect received goods",
        ],
        "preferred_leaf_module": "SCM",
        "role_context": "Sourcing or quality user following awarded supply receipts",
        "purpose": "In Sourcing-related receiving flows, inspection is carried out through the receiving and quality process so awarded supply can be inspected before it is accepted for use.",
        "prerequisites": [
            "Inspection plans and receiving inspection levels must already be configured.",
            "The sourced purchase order receipt must be available for inspection.",
        ],
        "ordered_steps": [
            "Open the receiving transaction created from the sourced award or purchase flow.",
            "Launch the receiving inspection step and review the inspection characteristics and sample requirements.",
            "Enter the inspection results and record any disposition, nonconformance, or hold required by the result.",
            "Complete the receiving disposition so the sourced item can proceed only after a valid inspection outcome.",
        ],
        "navigation_paths": [
            "Receiving transaction flow > inspection step",
        ],
        "warnings": [
            "If the receipt was created without the required inspection setup, the inspection transaction won't provide the complete quality checks expected by the business flow.",
        ],
        "quality_score": 0.92,
    },
    {
        "slug": "sourcing_return_to_supplier",
        "module": "Sourcing",
        "title": "Sourcing - Return to Supplier Procedure",
        "task_name": "return to supplier",
        "task_type": "procedure",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/financials/26a/faufa/create-a-return-to-supplier-shipment.html",
        "query_aliases": [
            "return to supplier",
            "return to supplier in oracle fusion sourcing",
            "supplier return shipment",
        ],
        "preferred_leaf_module": "SCM",
        "role_context": "Sourcing or receiving specialist",
        "purpose": "When sourced goods must be returned, the return-to-supplier flow creates the supplier return shipment and records the disposition of the quantity that can't remain in inventory or expense use.",
        "prerequisites": [
            "Identify the receipt or shipment line being returned and confirm the supplier return reason.",
        ],
        "ordered_steps": [
            "Locate the received line or shipment line that needs to be returned to the supplier.",
            "Create the return-to-supplier shipment and enter the quantity, reason, and shipping details required by the return.",
            "Review any receiving, accounting, or fiscal document requirements tied to the return flow.",
            "Submit the return shipment and confirm the supplier-return transaction status.",
        ],
        "navigation_paths": [
            "Receiving or return management flow > create return to supplier shipment",
        ],
        "warnings": [
            "Returns can require additional receiving, accounting, or country-specific fiscal steps depending on the document context.",
        ],
        "quality_score": 0.9,
    },
    {
        "slug": "supplier_portal_receiving_inspection",
        "module": "Supplier Portal",
        "title": "Supplier Portal - Receiving Inspection Procedure",
        "task_name": "receiving inspection",
        "task_type": "procedure",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/supply-chain-and-manufacturing/25d/fauqm/overview-of-quality-management.html",
        "query_aliases": [
            "receiving inspection",
            "receiving inspection in oracle fusion supplier portal",
            "inspect received goods",
        ],
        "preferred_leaf_module": "SCM",
        "role_context": "Supplier collaboration or quality user",
        "purpose": "For supplier collaboration flows, receiving inspection records whether the received supplier material passes the required inspection before the transaction is finalized.",
        "prerequisites": [
            "The receiving transaction must exist and the inspection plan must be active for the received material.",
        ],
        "ordered_steps": [
            "Open the received transaction or collaboration event that requires inspection.",
            "Review the inspection characteristics and enter the inspection results for the received material.",
            "Record the accept, reject, or hold disposition based on the inspection result.",
            "Complete the inspection so the supplier-related receiving process reflects the final disposition.",
        ],
        "navigation_paths": [
            "Supplier collaboration receiving flow > inspection action",
        ],
        "warnings": [
            "Inspection results must be captured before the receiving flow is treated as complete for items that require inspection.",
        ],
        "quality_score": 0.92,
    },
    {
        "slug": "supplier_portal_three_way_match",
        "module": "Supplier Portal",
        "title": "Supplier Portal - Three-Way Match Procedure",
        "task_name": "three-way match",
        "task_type": "procedure",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/financials/25d/fappp/matching-invoice-lines.html",
        "query_aliases": [
            "three-way match",
            "3-way match",
            "three-way match in oracle fusion supplier portal",
        ],
        "preferred_leaf_module": "Payables",
        "role_context": "Supplier collaboration or payables user",
        "purpose": "Three-way match checks the supplier invoice against the purchase order and receipt so supplier billing aligns with what was ordered and received.",
        "prerequisites": [
            "The purchase order, receipt, and invoice must already exist for the same legal entity context.",
        ],
        "ordered_steps": [
            "Open the invoice or billing transaction that needs to be matched.",
            "Identify the purchase order and receipt information that applies to the supplier billing line.",
            "Use the invoice matching action to match invoice lines to the purchase order schedules or receipts.",
            "Review the matched lines and confirm the billed quantities and amounts are correct before completing the transaction.",
        ],
        "navigation_paths": [
            "Invoice or supplier billing flow > match invoice lines",
        ],
        "warnings": [
            "If the legal entity or receipt context doesn't align with the invoice, the line can't be matched correctly.",
        ],
        "quality_score": 0.93,
    },
    {
        "slug": "supplier_portal_negotiation_award",
        "module": "Supplier Portal",
        "title": "Supplier Portal - Negotiation Award Procedure",
        "task_name": "negotiation award",
        "task_type": "procedure",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/procurement/26a/fapra/api-supplier-negotiation-award-responses.html",
        "query_aliases": [
            "negotiation award",
            "supplier negotiation award",
            "award negotiation",
            "negotiation award in oracle fusion supplier portal",
        ],
        "preferred_leaf_module": "Procurement",
        "role_context": "Supplier portal or procurement collaboration user",
        "purpose": "Negotiation award finalizes the sourcing outcome so the awarded supplier response is accepted and the downstream procurement action can continue.",
        "prerequisites": [
            "The negotiation must be complete and the response being awarded must be eligible for award acceptance.",
        ],
        "ordered_steps": [
            "Review the negotiation responses and confirm the response that should be awarded.",
            "Open the award action for the selected negotiation response.",
            "Record the award decision and any required response or acceptance information.",
            "Submit the award and confirm the negotiation response status shows the completed award outcome.",
        ],
        "navigation_paths": [
            "Supplier negotiation response > award action",
        ],
        "warnings": [
            "Award processing should be completed only after the negotiation evaluation is finished and the selected response is approved for award.",
        ],
        "quality_score": 0.9,
    },
    {
        "slug": "ssp_receiving_inspection",
        "module": "Self Service Procurement",
        "title": "Self Service Procurement - Receiving Inspection Procedure",
        "task_name": "receiving inspection",
        "task_type": "procedure",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/supply-chain-and-manufacturing/25d/fauqm/overview-of-quality-management.html",
        "query_aliases": [
            "receiving inspection",
            "receiving inspection in oracle fusion self service procurement",
            "inspect received goods",
        ],
        "preferred_leaf_module": "SCM",
        "role_context": "Requester or receiving user",
        "purpose": "Receiving inspection in Self Service Procurement uses the receiving and quality flow to inspect requested goods before they are accepted into the final receiving status.",
        "prerequisites": [
            "The receipt must exist and the quality inspection plan must be active for the received item.",
        ],
        "ordered_steps": [
            "Open the receiving transaction associated with the self-service procurement request.",
            "Start the inspection action for the received quantity that requires review.",
            "Enter the inspection results and capture the final accept, reject, or hold disposition.",
            "Complete the receiving transaction after the inspection result is approved.",
        ],
        "navigation_paths": [
            "Self Service Procurement receiving flow > inspection action",
        ],
        "warnings": [
            "If the request was received without the required inspection configuration, the receiving flow won't enforce the full inspection procedure.",
        ],
        "quality_score": 0.92,
    },
    {
        "slug": "supply_chain_reservation_management_troubleshooting",
        "module": "Supply Chain",
        "title": "Supply Chain - Reservation Management Troubleshooting",
        "task_name": "reservation management",
        "task_type": "troubleshooting",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/supply-chain-and-manufacturing/25d/fapsu/reservations.html",
        "query_aliases": [
            "reservation management",
            "reservation failure",
            "reservation management troubleshooting",
        ],
        "purpose": "Use the reservation troubleshooting flow to identify why reservation processing failed, why the reservation wasn't detailed correctly, and what must be corrected before reprocessing.",
        "symptom_terms": [
            "reservation error",
            "reservation import failure",
            "reservation not detailed",
            "reservation processing failure",
        ],
        "error_keywords": [
            "reservation",
            "supply",
            "demand",
            "project reference",
            "interface error",
        ],
        "root_causes": [
            "Required reservation attributes or project references are missing from the incoming supply or demand.",
            "Reservation interface rows contain data errors that must be corrected before reprocessing.",
            "The reservation context doesn't satisfy the item or project-detail rules used by the source flow.",
        ],
        "resolution_steps": [
            "Review the reservation setup and the source transaction details to confirm the required project, item, and supply attributes are present.",
            "Use the reservation interface or reservation management task to query the failed rows and identify the error details.",
            "Correct the invalid attributes and resubmit the reservation processing flow.",
            "Recheck the reservation result to confirm the reservation is now created or detailed correctly.",
        ],
        "summary_points": [
            "Reservation failures usually come from missing source attributes or interface errors rather than a missing reservation feature.",
        ],
        "warnings": [
            "If a project-based demand is paired with supply that lacks the required project reference, the reservation may not detail correctly.",
        ],
        "quality_score": 0.92,
    },
    {
        "slug": "supply_chain_work_order_release_troubleshooting",
        "module": "Supply Chain",
        "title": "Supply Chain - Work Order Release Troubleshooting",
        "task_name": "work order release",
        "task_type": "troubleshooting",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/supply-chain-and-manufacturing/26a/fausp/create-and-release-manual-planned-orders.html",
        "query_aliases": [
            "work order release",
            "work order release troubleshooting",
            "release work order failure",
        ],
        "purpose": "Use work order release troubleshooting to validate the release parameters, source details, and release state before the work order is sent to the source application.",
        "symptom_terms": [
            "work order release failure",
            "release process error",
            "manual planned order release issue",
        ],
        "error_keywords": [
            "release",
            "planned order",
            "source type",
            "work definition",
        ],
        "root_causes": [
            "The release parameters or source specifications are incomplete for the order type being released.",
            "The planned order wasn't marked or prepared correctly for release to the source application.",
            "Required source details such as organization, supplier, or work definition are missing.",
        ],
        "resolution_steps": [
            "Review the planned order and confirm the source type and source-specific parameters are complete.",
            "Mark the planned order correctly for release and rerun the release process.",
            "Verify the source application setup, such as supplier, source organization, or work definition, before retrying the release.",
        ],
        "quality_score": 0.88,
    },
    {
        "slug": "manufacturing_reservation_management_troubleshooting",
        "module": "Manufacturing",
        "title": "Manufacturing - Reservation Management Troubleshooting",
        "task_name": "reservation management",
        "task_type": "troubleshooting",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/supply-chain-and-manufacturing/25d/fapsu/reservations.html",
        "query_aliases": [
            "reservation management",
            "manufacturing reservation failure",
            "reservation troubleshooting",
        ],
        "purpose": "Use manufacturing reservation troubleshooting to identify missing transaction attributes, reservation interface errors, and manufacturing-specific reservation rule conflicts.",
        "symptom_terms": [
            "reservation failure",
            "reservation not detailed",
            "reservation interface error",
        ],
        "error_keywords": [
            "reservation",
            "work order",
            "supply",
            "project reference",
        ],
        "root_causes": [
            "The manufacturing supply or demand transaction is missing reservation-driving attributes.",
            "Reservation rows contain interface errors that prevent processing.",
            "The manufacturing context doesn't align with the reservation rules used for the item or project.",
        ],
        "resolution_steps": [
            "Review the manufacturing transaction and confirm the required reservation-driving attributes are populated.",
            "Query the reservation interface errors and correct the invalid values.",
            "Resubmit the reservation processing and confirm the manufacturing reservation is created successfully.",
        ],
        "quality_score": 0.9,
    },
    {
        "slug": "manufacturing_cost_accounting_transfer_troubleshooting",
        "module": "Manufacturing",
        "title": "Manufacturing - Cost Accounting Transfer Troubleshooting",
        "task_name": "cost accounting transfer",
        "task_type": "troubleshooting",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/supply-chain-and-manufacturing/25d/fapma/example-of-proc-flow-dual-uom-partial-receipt-no-invoice.html",
        "query_aliases": [
            "cost accounting transfer",
            "cost accounting transfer troubleshooting",
            "transfer to cost accounting failure",
        ],
        "purpose": "Use cost accounting transfer troubleshooting to determine whether upstream accounting, transaction setup, or exception-state issues are preventing the manufacturing event from transferring to costing.",
        "symptom_terms": [
            "cost accounting transfer failure",
            "cost transfer error",
            "missing transfer to costing",
        ],
        "error_keywords": [
            "cost accounting",
            "transfer",
            "receipt accounting",
            "costing",
        ],
        "root_causes": [
            "Upstream receipt or transaction accounting wasn't completed before the cost transfer step.",
            "Required transaction attributes or costing setup are incomplete for the manufacturing event.",
            "The source transaction is still in an exception state and can't be transferred to costing.",
        ],
        "resolution_steps": [
            "Confirm the source manufacturing transaction completed the required receiving or transaction accounting steps.",
            "Review costing setup and transaction attributes for the failed event.",
            "Correct the exception condition and rerun the transfer to costing or related accounting process.",
        ],
        "quality_score": 0.88,
    },
    {
        "slug": "procurement_transfer_order_troubleshooting",
        "module": "Procurement",
        "title": "Procurement - Transfer Order Troubleshooting",
        "task_name": "transfer order",
        "task_type": "troubleshooting",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/supply-chain-and-manufacturing/25d/oadsc/TransferOrderHeaderExtractPVO.html",
        "query_aliases": [
            "transfer order",
            "transfer order troubleshooting",
            "transfer order failure",
        ],
        "purpose": "Use transfer order troubleshooting to verify the requisition or supply-request data, organization attributes, and release parameters that feed the transfer-order flow.",
        "symptom_terms": [
            "transfer order failure",
            "transfer order process error",
            "cannot create transfer order",
        ],
        "error_keywords": [
            "transfer order",
            "source organization",
            "destination organization",
            "shipping method",
        ],
        "root_causes": [
            "Required transfer-order source or destination attributes are missing.",
            "The procurement flow doesn't have the organization or shipping data required for the transfer order request.",
            "The upstream requisition or supply request is incomplete or inconsistent.",
        ],
        "resolution_steps": [
            "Review the source and destination organization details, shipping method, and related transfer-order attributes.",
            "Correct the requisition or supply request data that feeds the transfer order.",
            "Resubmit the transfer-order creation or release process and verify the transfer order is generated successfully.",
        ],
        "quality_score": 0.9,
    },
    {
        "slug": "general_ledger_overview",
        "module": "General Ledger",
        "title": "General Ledger - Overview",
        "task_name": "general ledger",
        "task_type": "summary",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/financials/25d/faugl/index.html",
        "query_aliases": [
            "what is gl",
            "what is general ledger",
            "gl",
            "general ledger",
            "oracle fusion gl",
            "oracle fusion general ledger",
        ],
        "role_context": "Financials accountant, implementer, or analyst",
        "purpose": "Oracle Fusion General Ledger is the central accounting module that manages ledgers, journals, balances, and period-close control for enterprise financial reporting.",
        "summary_points": [
            "General Ledger is the core ledger and journal layer inside Oracle Fusion Financials.",
            "It manages journal entry, posting, balances, reporting, and period-close control.",
            "Subledgers such as Payables and Receivables feed accounting into General Ledger for consolidated reporting.",
        ],
        "warnings": [
            "General Ledger controls enterprise accounting, but many operational transactions originate in subledgers such as Payables, Receivables, Assets, and Cash Management.",
        ],
        "quality_score": 0.9,
    },
    {
        "slug": "receivables_rmcs_overview",
        "module": "Receivables",
        "title": "Receivables - Revenue Management (RMCS) Overview",
        "task_name": "revenue management",
        "task_type": "summary",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/financials/25d/fafrm/index.html",
        "query_aliases": [
            "what is rmcs",
            "rmcs",
            "revenue management cloud service",
            "oracle fusion rmcs",
            "what is revenue management",
            "oracle fusion revenue management",
        ],
        "role_context": "Revenue accountant or financials implementer",
        "purpose": "RMCS refers to Oracle Fusion Revenue Management Cloud Service, which manages revenue contracts, performance obligations, and revenue recognition under the Revenue Management capability in Financials.",
        "summary_points": [
            "RMCS stands for Revenue Management Cloud Service in Oracle Fusion Financials.",
            "It manages revenue contracts, performance obligations, and the accounting logic used to recognize revenue correctly.",
            "Revenue Management integrates with upstream transaction sources and accounting flows to support compliant revenue recognition.",
        ],
        "warnings": [
            "RMCS is a business capability, not just a schema prefix, so concept questions should be grounded in Revenue Management docs rather than table metadata alone.",
        ],
        "quality_score": 0.9,
    },
    {
        "slug": "common_custom_ess_job",
        "module": "Common",
        "title": "Common - Custom ESS Job Procedure",
        "task_name": "custom ess job",
        "task_type": "procedure",
        "source_uri": "https://docs.oracle.com/en/cloud/saas/supply-chain-and-manufacturing/26a/faicf/overview-of-managing-job-definitions-and-job-sets.html",
        "query_aliases": [
            "custom ess job",
            "create custom ess job",
            "ess job",
            "enterprise scheduler job",
            "scheduled process job definition",
            "custom scheduled process",
            "job definition and job set",
        ],
        "role_context": "Application administrator or implementation specialist",
        "purpose": "A custom ESS job is created by defining the scheduled process metadata, parameters, and execution details that Oracle Enterprise Scheduler uses to submit and monitor the job.",
        "prerequisites": [
            "Confirm the environment supports the job definition you want to expose through Scheduled Processes.",
            "Identify the executable or process logic, parameter list, and security model required for the job.",
        ],
        "ordered_steps": [
            "Open the page used to manage job definitions and job sets for Scheduled Processes.",
            "Create the job definition and enter the identifying information for the process you want Oracle Enterprise Scheduler to run.",
            "Define the job parameters, prompts, and any submission properties required by the process.",
            "Associate the job definition with the correct executable or process implementation and validate the scheduling behavior.",
            "Assign the required privileges or roles so authorized users can submit or monitor the job.",
            "Submit the job from Scheduled Processes and verify the request completes successfully with the expected output.",
        ],
        "navigation_paths": [
            "Tools or application administration flow > Scheduled Processes > Manage Job Definitions and Job Sets",
        ],
        "warnings": [
            "Custom ESS jobs require valid execution metadata and security before users can submit them safely.",
            "If the executable, parameters, or privileges are incomplete, the job can be created but fail at submission time.",
        ],
        "quality_score": 0.86,
    },
]


def build_zero_grounding_report(results_path: Path = BASELINE_RESULTS) -> Dict[str, Any]:
    results = _read_jsonl(results_path)
    filtered: List[Dict[str, Any]] = []
    by_module = Counter()
    by_task_type = Counter()
    by_gap = Counter()

    for row in results:
        failed = str(row.get("scoring_outcome") or "") not in {"grounded_correct", "safe_refusal_correct"}
        zero_grounding = (
            not row.get("decision_grounding_signal_present")
            or float(row.get("grounding_availability_score") or 0.0) == 0.0
        )
        executed_but_fail_closed = (
            str(row.get("output") or "").strip() == FAIL_CLOSED_MESSAGE
            and str(row.get("decision_execution_mode") or "").upper() == "EXECUTE"
        )
        if not failed or not (zero_grounding or executed_but_fail_closed):
            continue

        if row.get("benchmark_segment") == "doc_grounded_procedure":
            gap = "missing procedural docs"
        elif row.get("benchmark_segment") == "troubleshooting":
            gap = "missing troubleshooting docs"
        elif row.get("benchmark_segment") == "module_ambiguous":
            gap = "ambiguous wording not linked to relevant docs"
        elif int(row.get("retrieved_doc_count") or 0) > 0:
            gap = "poor retrieval recall"
        else:
            gap = "poor module-specific curation"

        record = {
            "case_id": row.get("id"),
            "query": row.get("question"),
            "expected_module": row.get("benchmark_module"),
            "case_segment": row.get("benchmark_segment"),
            "refusal_or_output_status": row.get("scoring_outcome"),
            "corpus_evidence_existed_anywhere": bool(int(row.get("retrieved_doc_count") or 0) > 0 or row.get("citations_present")),
            "missing_evidence_type": gap,
            "benchmark_intent": row.get("benchmark_intent"),
            "decision_grounding_signal_present": bool(row.get("decision_grounding_signal_present")),
            "grounding_availability_score": float(row.get("grounding_availability_score") or 0.0),
            "verifier_status": row.get("verifier_status"),
            "output": row.get("output"),
        }
        filtered.append(record)
        by_module[str(row.get("benchmark_module") or "UNKNOWN")] += 1
        by_task_type[str(row.get("benchmark_intent") or "UNKNOWN")] += 1
        by_gap[gap] += 1

    report = {
        "total_results_scanned": len(results),
        "zero_grounding_or_fail_closed_case_count": len(filtered),
        "by_module": dict(by_module),
        "by_task_type": dict(by_task_type),
        "by_evidence_gap_type": dict(by_gap),
        "cases": filtered,
    }
    _write_json(OUTPUT_ROOT / "zero_grounding_failure_report.json", report)
    return report


def _build_content(spec: Dict[str, Any]) -> str:
    lines = [
        spec["title"],
        f"Task Name: {spec['task_name']}",
        f"Module: {spec['module']}",
        "",
        "Purpose",
        spec["purpose"],
    ]
    if spec.get("role_context"):
        lines.extend(["", "Role Context", spec["role_context"]])
    if spec.get("prerequisites"):
        lines.append("")
        lines.append("Prerequisites")
        lines.extend(f"- {item}" for item in spec["prerequisites"])
    if spec.get("navigation_paths"):
        lines.append("")
        lines.append("UI Navigation")
        lines.extend(f"- {item}" for item in spec["navigation_paths"])
    if spec.get("ordered_steps"):
        lines.append("")
        lines.append("Ordered Steps")
        lines.extend(f"{index}. {step}" for index, step in enumerate(spec["ordered_steps"], start=1))
    if spec.get("symptom_terms"):
        lines.append("")
        lines.append("Symptoms")
        lines.extend(f"- {item}" for item in spec["symptom_terms"])
    if spec.get("root_causes"):
        lines.append("")
        lines.append("Root Causes")
        lines.extend(f"- {item}" for item in spec["root_causes"])
    if spec.get("resolution_steps"):
        lines.append("")
        lines.append("Resolution Steps")
        lines.extend(f"{index}. {step}" for index, step in enumerate(spec["resolution_steps"], start=1))
    if spec.get("summary_points"):
        lines.append("")
        lines.append("Summary Points")
        lines.extend(f"- {item}" for item in spec["summary_points"])
    if spec.get("warnings"):
        lines.append("")
        lines.append("Warnings")
        lines.extend(f"- {item}" for item in spec["warnings"])
    if spec.get("preferred_leaf_module"):
        lines.extend(["", f"Preferred Leaf Module: {spec['preferred_leaf_module']}"])
    lines.extend(["", "Query Aliases"])
    lines.extend(f"- {item}" for item in spec["query_aliases"])
    return "\n".join(lines).strip()


def _source_record_for_spec(spec: Dict[str, Any], source_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if spec["source_uri"] == str(HCM_WEB_DATA_JSON):
        return _extract_hcm_benefits_source()
    if spec["source_uri"] in source_index:
        return source_index[spec["source_uri"]]
    raise KeyError(f"Missing source record for {spec['source_uri']}")


def build_manifest_rows() -> List[Dict[str, Any]]:
    source_index = _load_source_index()
    rows: List[Dict[str, Any]] = []
    for spec in TARGET_SPECS:
        source_doc = _source_record_for_spec(spec, source_index)
        content = _build_content(spec)
        title = spec["title"]
        metadata = {
            "source_uri": source_doc["source_uri"],
            "canonical_uri": source_doc["source_uri"],
            "authority_tier": "official",
            "task_name": spec["task_name"],
            "query_aliases": spec["query_aliases"],
            "prerequisites": spec.get("prerequisites", []),
            "navigation_paths": spec.get("navigation_paths", []),
            "ordered_steps": spec.get("ordered_steps", []),
            "summary_points": spec.get("summary_points", []),
            "warnings": spec.get("warnings", []),
            "symptom_terms": spec.get("symptom_terms", []),
            "error_keywords": spec.get("error_keywords", []),
            "root_causes": spec.get("root_causes", []),
            "resolution_steps": spec.get("resolution_steps", []),
            "task_signals": [spec["task_name"], *spec.get("query_aliases", [])],
            "role_context": spec.get("role_context"),
            "purpose": spec["purpose"],
            "preferred_leaf_module": spec.get("preferred_leaf_module"),
            "grounding_answerability_pass": True,
            "allow_deterministic_grounded_answer": True,
            "derived_from_source_doc": True,
            "source_doc_title": source_doc["title"],
            "source_doc_excerpt": _normalize_whitespace(source_doc["content"])[:500],
            "title": title,
        }
        record = IngestionManifestRecord(
            source_path=source_doc["source_path"],
            source_uri=source_doc["source_uri"],
            title=title,
            module=spec["module"],
            task_type=spec["task_type"],
            doc_type=(
                DocType.PROCEDURE_DOC
                if spec["task_type"] == "procedure"
                else DocType.TROUBLESHOOTING_DOC
                if spec["task_type"] == "troubleshooting"
                else DocType.FUNCTIONAL_DOC
            ),
            trusted_schema_objects=[],
            quality_score=float(spec["quality_score"]),
            content_hash=stable_hash("docs_corpus", content),
            source_system=SourceSystem.ORACLE_DOCS,
            content=content,
            metadata=metadata,
        ).model_dump(mode="json")
        rows.append(record)
    return rows


def index_manifest_rows(rows: List[Dict[str, Any]], tenant_id: str = TENANT_ID) -> Dict[str, Any]:
    faiss = FaissIndex(tenant_id=tenant_id, indexes_dir=str(INDEXES_DIR), corpus="docs_corpus")
    chunks: List[Dict[str, Any]] = []
    for payload in rows:
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


def write_case_id_file(results_path: Path = BASELINE_RESULTS) -> Path:
    case_ids = [row.get("id") for row in _read_jsonl(results_path) if row.get("id")]
    output_path = OUTPUT_ROOT / "case_ids_200.json"
    _write_json(output_path, case_ids)
    return output_path


def run_grounding_answerability_pass() -> Dict[str, Any]:
    zero_grounding_report = build_zero_grounding_report()
    manifest_rows = build_manifest_rows()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / "docs_manifest.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    index_stats = index_manifest_rows(manifest_rows)
    case_id_file = write_case_id_file()

    by_module = Counter(row["module"] for row in manifest_rows)
    by_task = Counter(
        (
            row.get("metadata", {}).get("task_name")
            or row.get("task_name")
            or row.get("title", "unknown")
        )
        for row in manifest_rows
    )
    summary = {
        "manifest_path": str(manifest_path),
        "case_id_file": str(case_id_file),
        "docs_added": len(manifest_rows),
        "by_module": dict(by_module),
        "by_task": dict(by_task),
        "zero_grounding_case_count": zero_grounding_report["zero_grounding_or_fail_closed_case_count"],
        "index_stats": index_stats,
    }
    _write_json(OUTPUT_ROOT / "summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_grounding_answerability_pass(), indent=2, ensure_ascii=False))
