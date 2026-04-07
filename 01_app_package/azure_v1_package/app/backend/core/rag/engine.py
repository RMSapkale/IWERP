import json
import os
import re
import time
import uuid
import hashlib
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from core.config.operational import LLM_CONFIG, RETRIEVAL_CONFIG
from core.config.tasks import (
    ENABLE_RAG_VERIFIER,
    ENABLE_SQL_VERIFIER,
    MAX_VERIFICATION_RETRIES,
    TASK_CONFIGS,
)
from core.grounding.citations import CitationMapper
from core.grounding.task_semantics import TaskSemanticAnalyzer
from core.grounding.verifier import Verifier
from core.llm.llama_cpp_client import LlamaCppClient
from core.llm.prompts import RAGPrompts
from core.retrieval.hybrid import HybridPostgresSearch
from core.retrieval.knowledge_graph import fusion_graph
from core.retrieval.policy import RetrievalPolicy
from core.retrieval.reranker import LocalReranker
from core.retrieval.router import TaskRouter
from core.schemas.api import ChatRequest, ChatResponse, Message, Role
from core.schemas.router import FusionModule, ModuleFamily, TaskType, module_families_for_value

logger = structlog.get_logger(__name__)

FAIL_CLOSED_MESSAGE = "Insufficient grounded data. Cannot generate verified answer."


class RAGEngine:
    """
    Orchestrates grounded Oracle Fusion retrieval, generation, and verification.
    """

    MODULE_FILTERS = {
        FusionModule.PAYABLES: ["Payables", "AP"],
        FusionModule.RECEIVABLES: ["Receivables", "AR"],
        FusionModule.GENERAL_LEDGER: ["General Ledger", "GL"],
        FusionModule.CASH_MANAGEMENT: ["Cash Management", "CE"],
        FusionModule.ASSETS: ["Assets", "FA"],
        FusionModule.EXPENSES: ["Expenses", "EXM"],
        FusionModule.PROCUREMENT: ["Procurement", "PO", "POZ", "POR", "PON"],
        FusionModule.SCM: ["SCM", "INV", "WSH", "DOO", "EGP"],
        FusionModule.HCM: ["HCM", "PER", "PAY", "HRC"],
        FusionModule.PROJECTS: ["Projects", "PJT", "PJC", "PJF", "PA"],
        FusionModule.TAX: ["Tax", "ZX"],
    }

    SQL_TASKS = {
        TaskType.SQL_GENERATION,
        TaskType.SQL_TROUBLESHOOTING,
        TaskType.REPORT_LOGIC,
    }
    FAST_FORMULA_TASKS = {
        TaskType.FAST_FORMULA_GENERATION,
        TaskType.FAST_FORMULA_TROUBLESHOOTING,
    }
    DOCS_EXPECTED_TASKS = {
        TaskType.PROCEDURE,
        TaskType.NAVIGATION,
        TaskType.GENERAL,
        TaskType.SUMMARY,
        TaskType.INTEGRATION,
        TaskType.TROUBLESHOOTING,
    }
    DOC_GROUNDING_CORPORA = {
        "docs_corpus",
        "troubleshooting_corpus",
    }
    EXACT_MODULE_ALIASES = {
        FusionModule.AP.value: FusionModule.PAYABLES.value,
        FusionModule.AR.value: FusionModule.RECEIVABLES.value,
        FusionModule.GL.value: FusionModule.GENERAL_LEDGER.value,
        "CE": FusionModule.CASH_MANAGEMENT.value,
        "FA": FusionModule.ASSETS.value,
        "EXM": FusionModule.EXPENSES.value,
        "ZX": FusionModule.TAX.value,
    }
    SQL_UNSAFE_QUERY_PATTERNS = (
        r"(?i)\bselect\s+\*",
        r"(?i)\bfrom\s+dual\b",
        r"(?i)\bdual filler\b",
        r"(?i)\bplaceholder\b",
        r"(?i)\botbi\b",
        r"(?i)\blogical sql\b",
        r"(?i)\bpl\/sql\b",
        r"(?i)\bcreate\s+package\b",
        r"(?i)\bref\s+cursor\b",
        r"(?i)\btodo\b",
        r"(?i)\bxx_[a-z0-9_]+\b",
        r"(?i)\bfake[_ ]",
        r"(?i)\bunknown table\b",
        r"(?i)\binvalid table\b",
    )
    FAST_FORMULA_UNSAFE_QUERY_PATTERNS = (
        r"(?i)\bUNKNOWN_PAY_ITEM\b",
        r"(?i)\bINVALID_CTX_VALUE\b",
        r"(?i)\b(galactic|quantum|wormhole|astrolabe|alien|deep space|ghost|cosmic|mythical|phantom|fictional|imaginary)\b",
    )
    FAST_FORMULA_TYPE_ALIASES = {
        "oracle payroll": "Oracle Payroll",
        "payroll": "Oracle Payroll",
        "proration": "Proration",
        "global absence proration": "Global Absence Proration",
        "absence": "Absence",
        "global absence entry validation": "Global Absence Entry Validation",
        "validation": "Global Absence Entry Validation",
        "rate value calculation": "Rate Value Calculation",
        "rate": "Rate Value Calculation",
        "auto indirect": "Auto Indirect",
        "element skip": "Element Skip",
        "calculation utility": "Calculation Utility",
        "flow schedule": "Flow Schedule",
    }
    FAST_FORMULA_NOISE_IDENTIFIERS = {
        "FORMULA",
        "FORMULA_NAME",
        "FORMULA_TYPE",
        "INPUTS",
        "INPUT",
        "DEFAULT",
        "RETURN",
        "END",
        "ENDIF",
        "ENDLOOP",
        "THEN",
        "ELSE",
        "NUMBER",
        "TEXT",
        "DATE",
    }
    FAST_FORMULA_TOKEN_STOPWORDS = {
        "formula",
        "oracle",
        "fusion",
        "hcm",
        "troubleshoot",
        "troubleshooting",
        "broken",
        "return",
        "inputs",
        "default",
        "for",
        "are",
        "then",
        "else",
        "endif",
        "endloop",
        "number",
        "text",
        "date",
    }
    FAST_FORMULA_GENERIC_TITLES = {
        "array processing",
        "null handling",
        "database items",
        "date functions",
        "string functions",
        "condition logic",
        "loops",
    }
    SQL_REQUEST_FIELD_GROUPS = [
        {
            "key": "invoice_number",
            "label": "Invoice Number",
            "phrases": ["invoice number", "invoice num"],
            "columns": ["INVOICE_NUM"],
            "aliases": ["INVOICE_NUMBER", "INVOICE_NUM"],
        },
        {
            "key": "invoice_date",
            "label": "Invoice Date",
            "phrases": ["invoice date"],
            "columns": ["INVOICE_DATE"],
            "aliases": ["INVOICE_DATE"],
        },
        {
            "key": "invoice_amount",
            "label": "Invoice Amount",
            "phrases": ["invoice amount", "invoice total"],
            "columns": ["INVOICE_AMOUNT", "AMOUNT"],
            "aliases": ["INVOICE_AMOUNT"],
        },
        {
            "key": "accounting_date",
            "label": "Accounting Date",
            "phrases": ["accounting date"],
            "columns": ["ACCOUNTING_DATE"],
            "aliases": ["ACCOUNTING_DATE"],
        },
        {
            "key": "supplier_name",
            "label": "Supplier Name",
            "phrases": ["supplier name", "vendor name", "supplier"],
            "columns": ["DF_COMPANY_NAME", "DF_LEGAL_NAME", "TAX_REPORTING_NAME", "PARTY_NAME"],
            "aliases": ["SUPPLIER_NAME", "VENDOR_NAME"],
        },
        {
            "key": "supplier_number",
            "label": "Supplier Number",
            "phrases": ["supplier number", "vendor number"],
            "columns": ["SEGMENT1"],
            "aliases": ["SUPPLIER_NUMBER", "VENDOR_NUMBER"],
        },
        {
            "key": "supplier_site_code",
            "label": "Site Code",
            "phrases": ["site code", "supplier site"],
            "columns": ["VENDOR_SITE_CODE", "SITE_CODE"],
            "aliases": ["SITE_CODE", "SUPPLIER_SITE_CODE"],
        },
        {
            "key": "business_unit_name",
            "label": "Business Unit",
            "phrases": ["business unit", "bu name", "business unit name"],
            "columns": ["BU_NAME", "BUSINESS_UNIT_NAME"],
            "aliases": ["BUSINESS_UNIT", "BUSINESS_UNIT_NAME", "BU_NAME"],
        },
        {
            "key": "distribution_line_number",
            "label": "Distribution Line Number",
            "phrases": ["distribution line number", "distribution line"],
            "columns": ["DISTRIBUTION_LINE_NUMBER"],
            "aliases": ["DISTRIBUTION_LINE_NUMBER", "DISTRIBUTION_LINE"],
        },
        {
            "key": "distribution_amount",
            "label": "Distribution Amount",
            "phrases": ["distribution amount", "line amount"],
            "columns": ["AMOUNT"],
            "aliases": ["DISTRIBUTION_AMOUNT"],
        },
        {
            "key": "natural_account_segment",
            "label": "Natural Account Segment",
            "phrases": ["natural account segment", "natural account"],
            "columns": ["SEGMENT4"],
            "aliases": ["NATURAL_ACCOUNT_SEGMENT", "NATURAL_ACCOUNT"],
        },
        {
            "key": "cost_center",
            "label": "Cost Center",
            "phrases": ["cost center"],
            "columns": ["SEGMENT5"],
            "aliases": ["COST_CENTER"],
        },
        {
            "key": "liability_account",
            "label": "Liability Account",
            "phrases": ["liability account", "liability code combination"],
            "columns": ["CONCATENATED_SEGMENTS"],
            "aliases": ["LIABILITY_ACCOUNT"],
        },
        {
            "key": "customer_name",
            "label": "Customer Name",
            "phrases": ["customer name"],
            "columns": ["PARTY_NAME"],
            "aliases": ["CUSTOMER_NAME"],
        },
        {
            "key": "transaction_number",
            "label": "Transaction Number",
            "phrases": ["transaction number", "trx number"],
            "columns": ["TRX_NUMBER"],
            "aliases": ["TRANSACTION_NUMBER", "TRX_NUMBER"],
        },
        {
            "key": "transaction_date",
            "label": "Transaction Date",
            "phrases": ["transaction date", "trx date"],
            "columns": ["TRX_DATE", "TRANSACTION_DATE"],
            "aliases": ["TRANSACTION_DATE", "TRX_DATE"],
        },
        {
            "key": "amount_due_original",
            "label": "Amount Due Original",
            "phrases": ["amount due original", "original amount due"],
            "columns": ["AMOUNT_DUE_ORIGINAL"],
            "aliases": ["AMOUNT_DUE_ORIGINAL", "ORIGINAL_AMOUNT_DUE"],
        },
        {
            "key": "receipt_number",
            "label": "Receipt Number",
            "phrases": ["receipt number"],
            "columns": ["RECEIPT_NUMBER"],
            "aliases": ["RECEIPT_NUMBER"],
        },
        {
            "key": "receipt_date",
            "label": "Receipt Date",
            "phrases": ["receipt date"],
            "columns": ["RECEIPT_DATE", "TRANSACTION_DATE"],
            "aliases": ["RECEIPT_DATE"],
        },
        {
            "key": "receipt_amount",
            "label": "Receipt Amount",
            "phrases": ["receipt amount"],
            "columns": ["AMOUNT"],
            "aliases": ["RECEIPT_AMOUNT"],
        },
        {
            "key": "receipt_status",
            "label": "Receipt Status",
            "phrases": ["receipt status"],
            "columns": ["STATUS"],
            "aliases": ["RECEIPT_STATUS"],
        },
        {
            "key": "due_date",
            "label": "Due Date",
            "phrases": ["due date"],
            "columns": ["DUE_DATE"],
            "aliases": ["DUE_DATE"],
        },
        {
            "key": "remaining_amount",
            "label": "Remaining Amount",
            "phrases": ["remaining amount", "outstanding amount"],
            "columns": ["AMOUNT_DUE_REMAINING"],
            "aliases": ["REMAINING_AMOUNT", "OUTSTANDING_AMOUNT"],
        },
        {
            "key": "journal_name",
            "label": "Journal Name",
            "phrases": ["journal name"],
            "columns": ["NAME"],
            "aliases": ["JOURNAL_NAME"],
        },
        {
            "key": "period_name",
            "label": "Period Name",
            "phrases": ["period name", "current period"],
            "columns": ["PERIOD_NAME"],
            "aliases": ["PERIOD_NAME"],
        },
        {
            "key": "ledger_name",
            "label": "Ledger Name",
            "phrases": ["ledger name", "ledger"],
            "columns": ["NAME"],
            "aliases": ["LEDGER_NAME"],
        },
        {
            "key": "journal_source",
            "label": "Journal Source",
            "phrases": ["journal source"],
            "columns": ["JE_SOURCE"],
            "aliases": ["JOURNAL_SOURCE"],
        },
        {
            "key": "journal_category",
            "label": "Journal Category",
            "phrases": ["journal category"],
            "columns": ["JE_CATEGORY"],
            "aliases": ["JOURNAL_CATEGORY"],
        },
        {
            "key": "journal_status",
            "label": "Journal Status",
            "phrases": ["journal status"],
            "columns": ["STATUS"],
            "aliases": ["JOURNAL_STATUS"],
        },
        {
            "key": "journal_line_number",
            "label": "Journal Line Number",
            "phrases": ["journal line number", "journal line"],
            "columns": ["JE_LINE_NUM"],
            "aliases": ["JE_LINE_NUM"],
        },
        {
            "key": "line_number",
            "label": "Line Number",
            "phrases": ["line number"],
            "columns": ["LINE_NUM"],
            "aliases": ["LINE_NUMBER"],
        },
        {
            "key": "debit_amount",
            "label": "Debit Amount",
            "phrases": ["debit amount", "entered debit", "debit"],
            "columns": ["ENTERED_DR"],
            "aliases": ["DEBIT_AMOUNT", "ENTERED_DEBIT"],
        },
        {
            "key": "credit_amount",
            "label": "Credit Amount",
            "phrases": ["credit amount", "entered credit", "credit"],
            "columns": ["ENTERED_CR"],
            "aliases": ["CREDIT_AMOUNT", "ENTERED_CREDIT"],
        },
        {
            "key": "period_net_dr",
            "label": "Period Net Dr",
            "phrases": ["period net dr", "period debit"],
            "columns": ["PERIOD_NET_DR"],
            "aliases": ["PERIOD_NET_DR"],
        },
        {
            "key": "period_net_cr",
            "label": "Period Net Cr",
            "phrases": ["period net cr", "period credit"],
            "columns": ["PERIOD_NET_CR"],
            "aliases": ["PERIOD_NET_CR"],
        },
        {
            "key": "account_combination",
            "label": "Account Combination",
            "phrases": ["account combination", "concatenated segments", "gl account combination", "gl account"],
            "columns": ["CONCATENATED_SEGMENTS"],
            "aliases": ["ACCOUNT_COMBINATION"],
        },
        {
            "key": "code_combination_id",
            "label": "Code Combination ID",
            "phrases": ["code combination id"],
            "columns": ["CODE_COMBINATION_ID"],
            "aliases": ["CODE_COMBINATION_ID"],
        },
        {
            "key": "bank_account_id",
            "label": "Bank Account ID",
            "phrases": ["bank account id"],
            "columns": ["BANK_ACCOUNT_ID"],
            "aliases": ["BANK_ACCOUNT_ID"],
        },
        {
            "key": "bank_account_name",
            "label": "Bank Account Name",
            "phrases": ["bank account name", "bank account"],
            "columns": ["BANK_ACCOUNT_NAME"],
            "aliases": ["BANK_ACCOUNT_NAME"],
        },
        {
            "key": "account_classification",
            "label": "Account Classification",
            "phrases": ["account classification"],
            "columns": ["ACCOUNT_CLASSIFICATION"],
            "aliases": ["ACCOUNT_CLASSIFICATION"],
        },
        {
            "key": "account_owner_org_id",
            "label": "Account Owner Org ID",
            "phrases": ["account owner org id"],
            "columns": ["ACCOUNT_OWNER_ORG_ID"],
            "aliases": ["ACCOUNT_OWNER_ORG_ID"],
        },
        {
            "key": "legal_entity_name",
            "label": "Legal Entity Name",
            "phrases": ["legal entity name", "legal entity"],
            "columns": ["NAME", "LEGAL_ENTITY_IDENTIFIER"],
            "aliases": ["LEGAL_ENTITY_NAME", "LEGAL_ENTITY_IDENTIFIER"],
        },
        {
            "key": "statement_number",
            "label": "Statement Number",
            "phrases": ["statement number"],
            "columns": ["STATEMENT_NUMBER"],
            "aliases": ["STATEMENT_NUMBER"],
        },
        {
            "key": "statement_date",
            "label": "Statement Date",
            "phrases": ["statement date"],
            "columns": ["STATEMENT_DATE"],
            "aliases": ["STATEMENT_DATE"],
        },
        {
            "key": "statement_line_number",
            "label": "Statement Line Number",
            "phrases": ["statement line number", "bank statement line number"],
            "columns": ["LINE_NUMBER"],
            "aliases": ["STATEMENT_LINE_NUMBER"],
        },
        {
            "key": "statement_amount",
            "label": "Statement Amount",
            "phrases": ["statement amount"],
            "columns": ["AMOUNT"],
            "aliases": ["STATEMENT_AMOUNT"],
        },
        {
            "key": "payment_status",
            "label": "Payment Status",
            "phrases": ["payment status", "paid status", "unpaid"],
            "columns": ["PAYMENT_STATUS_FLAG", "STATUS"],
            "aliases": ["PAYMENT_STATUS"],
        },
        {
            "key": "invoice_currency",
            "label": "Invoice Currency",
            "phrases": ["invoice currency", "transaction currency"],
            "columns": ["INVOICE_CURRENCY_CODE", "PAYMENT_CURRENCY_CODE", "CURRENCY_CODE"],
            "aliases": ["INVOICE_CURRENCY", "CURRENCY_CODE"],
        },
        {
            "key": "invoice_status",
            "label": "Invoice Status",
            "phrases": ["invoice status"],
            "columns": ["APPROVAL_STATUS", "STATUS"],
            "aliases": ["INVOICE_STATUS"],
        },
        {
            "key": "payment_number",
            "label": "Payment Number",
            "phrases": ["payment number", "check number"],
            "columns": ["CHECK_NUMBER", "PAYMENT_REFERENCE_NUMBER", "PAYMENT_NUM"],
            "aliases": ["PAYMENT_NUMBER", "CHECK_NUMBER"],
        },
        {
            "key": "payment_date",
            "label": "Payment Date",
            "phrases": ["payment date", "check date"],
            "columns": ["CHECK_DATE", "PAYMENT_DATE"],
            "aliases": ["PAYMENT_DATE", "CHECK_DATE"],
        },
        {
            "key": "payment_method",
            "label": "Payment Method",
            "phrases": ["payment method"],
            "columns": ["PAYMENT_METHOD_LOOKUP_CODE", "PAYMENT_METHOD_CODE"],
            "aliases": ["PAYMENT_METHOD"],
        },
        {
            "key": "paid_amount",
            "label": "Paid Amount",
            "phrases": ["paid amount", "payment amount"],
            "columns": ["AMOUNT", "PAYMENT_AMOUNT"],
            "aliases": ["PAID_AMOUNT", "PAYMENT_AMOUNT"],
        },
        {
            "key": "applied_amount",
            "label": "Applied Amount",
            "phrases": ["applied amount", "receipt application amount"],
            "columns": ["AMOUNT_APPLIED"],
            "aliases": ["APPLIED_AMOUNT", "AMOUNT_APPLIED"],
        },
        {
            "key": "ending_balance",
            "label": "Ending Balance",
            "phrases": ["ending balance"],
            "columns": [],
            "aliases": ["ENDING_BALANCE"],
        },
        {
            "key": "aging_bucket",
            "label": "Aging Bucket",
            "phrases": ["aging bucket", "age bucket"],
            "columns": [],
            "aliases": ["AGING_BUCKET"],
        },
        {
            "key": "po_number",
            "label": "PO Number",
            "phrases": ["po number", "purchase order number"],
            "columns": ["SEGMENT1"],
            "aliases": ["PO_NUMBER", "PURCHASE_ORDER_NUMBER"],
        },
        {
            "key": "po_date",
            "label": "PO Date",
            "phrases": ["po date", "purchase order date"],
            "columns": ["CREATION_DATE"],
            "aliases": ["PO_DATE", "PURCHASE_ORDER_DATE"],
        },
        {
            "key": "po_status",
            "label": "PO Status",
            "phrases": ["po status", "purchase order status", "document status"],
            "columns": ["DOCUMENT_STATUS"],
            "aliases": ["PO_STATUS", "DOCUMENT_STATUS"],
        },
        {
            "key": "po_type",
            "label": "PO Type",
            "phrases": ["po type", "purchase order type"],
            "columns": ["TYPE_LOOKUP_CODE"],
            "aliases": ["PO_TYPE", "PURCHASE_ORDER_TYPE"],
        },
        {
            "key": "ordered_quantity",
            "label": "Ordered Quantity",
            "phrases": ["ordered quantity"],
            "columns": ["QUANTITY"],
            "aliases": ["ORDERED_QUANTITY"],
        },
        {
            "key": "received_quantity",
            "label": "Received Quantity",
            "phrases": ["received quantity"],
            "columns": ["QUANTITY_RECEIVED"],
            "aliases": ["RECEIVED_QUANTITY"],
        },
        {
            "key": "billed_quantity",
            "label": "Billed Quantity",
            "phrases": ["billed quantity", "invoiced quantity"],
            "columns": ["QUANTITY_BILLED"],
            "aliases": ["BILLED_QUANTITY", "INVOICED_QUANTITY"],
        },
        {
            "key": "unit_price",
            "label": "Unit Price",
            "phrases": ["unit price"],
            "columns": ["UNIT_PRICE"],
            "aliases": ["UNIT_PRICE"],
        },
        {
            "key": "item_description",
            "label": "Item Description",
            "phrases": ["item description", "description"],
            "columns": ["ITEM_DESCRIPTION", "DESCRIPTION"],
            "aliases": ["ITEM_DESCRIPTION", "DESCRIPTION"],
        },
        {
            "key": "invoice_line_amount",
            "label": "Invoice Line Amount",
            "phrases": ["invoice line amount", "line amount invoiced", "invoiced amount"],
            "columns": ["AMOUNT"],
            "aliases": ["INVOICE_LINE_AMOUNT", "INVOICED_AMOUNT"],
        },
        {
            "key": "iban_number",
            "label": "IBAN Number",
            "phrases": ["iban number", "iban"],
            "columns": ["IBAN_NUMBER"],
            "aliases": ["IBAN_NUMBER"],
        },
    ]
    SQL_REQUEST_FILTER_GROUPS = [
        {
            "key": "validated",
            "label": "Validated",
            "phrases": ["validated"],
            "columns": ["APPROVAL_STATUS", "WFAPPROVAL_STATUS", "VALIDATION_REQUEST_ID"],
            "values": ["APPROVED", "VALIDATED", ":P_APPROVAL_STATUS", ":P_VALIDATION_STATUS"],
        },
        {
            "key": "accounted",
            "label": "Accounted",
            "phrases": ["accounted"],
            "columns": ["POSTED_FLAG", "POSTING_STATUS", "ACCTD_AMOUNT"],
            "values": ["Y", "POSTED", "ACCOUNTED", "IS NOT NULL", ":P_ACCOUNTED_FLAG", ":P_POSTING_STATUS"],
        },
        {
            "key": "posted_journal",
            "label": "Posted Journal",
            "phrases": ["posted journal", "posted journals", "journal posted"],
            "columns": ["STATUS"],
            "values": ["P", "POSTED"],
        },
        {
            "key": "liability_account_type",
            "label": "Liability Account Type",
            "phrases": ["for liability accounts", "liability accounts only", "liability accounts"],
            "columns": ["ACCOUNT_TYPE"],
            "values": [":P_ACCOUNT_TYPE", "L", "LIABILITY"],
        },
        {
            "key": "open_status",
            "label": "Open Status",
            "phrases": ["status is open", "open status", "open invoices", "outstanding", "remaining amount"],
            "columns": ["STATUS", "AMOUNT_DUE_REMAINING"],
            "values": ["OP", "OPEN"],
        },
        {
            "key": "approved",
            "label": "Approved",
            "phrases": ["approved invoices", "only approved", "approved"],
            "columns": ["APPROVAL_STATUS"],
            "values": ["APPROVED", ":P_APPROVAL_STATUS"],
        },
        {
            "key": "unpaid",
            "label": "Unpaid",
            "phrases": ["unpaid", "not paid"],
            "columns": ["PAYMENT_STATUS_FLAG", "AMOUNT_PAID"],
            "values": ["N", "UNPAID", "NOT PAID", "0", ":P_UNPAID_STATUS"],
        },
        {
            "key": "complete",
            "label": "Complete",
            "phrases": ["complete transactions", "complete transaction", "only complete"],
            "columns": ["COMPLETE_FLAG", "STATUS"],
            "values": ["Y", "COMPLETE", ":P_COMPLETE_FLAG", ":P_COMPLETE_STATUS"],
        },
        {
            "key": "unreconciled",
            "label": "Unreconciled",
            "phrases": ["unreconciled", "not reconciled"],
            "columns": ["RECON_STATUS", "ACCOUNTING_FLAG", "STATUS"],
            "values": ["N", "UNRECONCILED", "UNMATCHED", "NEW", ":P_RECON_STATUS", ":P_ACCOUNTING_FLAG"],
        },
        {
            "key": "invoice_date_between",
            "label": "Invoice Date Between",
            "phrases": ["invoice date between", "filter invoice date between"],
            "columns": ["INVOICE_DATE"],
            "values": [":P_FROM_DATE", ":P_TO_DATE"],
        },
        {
            "key": "payment_date_equals",
            "label": "Payment Date",
            "phrases": ["payment date = :p_payment_date", "payment date = :p_payment_date.", "filter payment date = :p_payment_date", "filter: payment date = :p_payment_date", "payment date ="],
            "columns": ["CHECK_DATE", "PAYMENT_DATE"],
            "values": [":P_PAYMENT_DATE"],
        },
        {
            "key": "as_of_date",
            "label": "As Of Date",
            "phrases": ["as-of date", "as of date"],
            "columns": ["GL_DATE_CLOSED", "TRX_DATE", "CHECK_DATE"],
            "values": [":P_AS_OF_DATE"],
        },
        {
            "key": "ledger_bind",
            "label": "Ledger",
            "phrases": ["ledger = :p_ledger_name", "ledger :p_ledger_name", "ledger name = :p_ledger_name", "ledger name"],
            "columns": ["NAME", "LEDGER_ID"],
            "values": [":P_LEDGER_NAME", ":P_LEDGER_ID"],
        },
        {
            "key": "period_bind",
            "label": "Period",
            "phrases": ["period = :p_period_name", "period :p_period_name", "period name = :p_period_name", "period name"],
            "columns": ["PERIOD_NAME"],
            "values": [":P_PERIOD_NAME"],
        },
        {
            "key": "posted",
            "label": "Posted",
            "phrases": ["posted", "posted journals", "posted journal"],
            "columns": ["STATUS", "POSTED_FLAG"],
            "values": ["P", "POSTED", ":P_POSTED_STATUS", ":P_POSTED_FLAG"],
        },
        {
            "key": "applied",
            "label": "Applied",
            "phrases": ["applied"],
            "columns": ["STATUS", "DISPLAY"],
            "values": ["Y", "APP", "APPLIED", ":P_APPLIED_STATUS", ":P_DISPLAY_FLAG"],
        },
    ]
    SQL_REPORT_FAMILY_REGISTRY = {
        "payables_invoice_details": {
            "module": FusionModule.PAYABLES.value,
            "supported_fields": {
                "business_unit_name",
                "invoice_number",
                "invoice_date",
                "supplier_name",
                "supplier_number",
                "invoice_amount",
                "invoice_currency",
                "invoice_status",
                "accounting_date",
            },
            "supported_filters": {"invoice_date_between", "validated", "approved", "accounted"},
            "supported_ordering": {"invoice_date", "invoice_number"},
        },
        "payables_invoice_distribution_accounting": {
            "module": FusionModule.PAYABLES.value,
            "supported_fields": {
                "invoice_number",
                "supplier_name",
                "distribution_line_number",
                "distribution_amount",
                "natural_account_segment",
                "cost_center",
                "liability_account",
                "accounting_date",
            },
            "supported_filters": {"validated", "approved", "accounted"},
            "supported_ordering": {"invoice_number", "distribution_line_number"},
        },
        "payables_payments": {
            "module": FusionModule.PAYABLES.value,
            "supported_fields": {
                "supplier_name",
                "invoice_number",
                "payment_number",
                "payment_date",
                "payment_method",
                "paid_amount",
                "bank_account_name",
            },
            "supported_filters": {"payment_date_equals"},
            "supported_ordering": {"payment_date", "payment_number"},
        },
        "receivables_transaction_report": {
            "module": FusionModule.RECEIVABLES.value,
            "supported_fields": {
                "transaction_number",
                "transaction_date",
                "customer_name",
                "business_unit_name",
                "amount_due_original",
                "remaining_amount",
                "due_date",
                "payment_status",
            },
            "supported_filters": {"open_status", "complete", "as_of_date"},
            "supported_ordering": {"transaction_date", "transaction_number", "customer_name"},
        },
        "receivables_receipts_applications": {
            "module": FusionModule.RECEIVABLES.value,
            "supported_fields": {
                "customer_name",
                "receipt_number",
                "receipt_date",
                "receipt_amount",
                "transaction_number",
                "applied_amount",
                "receipt_status",
            },
            "supported_filters": {"applied", "payment_date_equals"},
            "supported_ordering": {"receipt_date", "receipt_number", "transaction_number"},
        },
        "receivables_aging": {
            "module": FusionModule.RECEIVABLES.value,
            "supported_fields": {
                "customer_name",
                "transaction_number",
                "due_date",
                "remaining_amount",
                "aging_bucket",
            },
            "supported_filters": {"as_of_date", "open_status"},
            "supported_ordering": {"customer_name", "transaction_number", "due_date"},
        },
        "general_ledger_account_balances": {
            "module": FusionModule.GENERAL_LEDGER.value,
            "supported_fields": {
                "ledger_name",
                "period_name",
                "account_combination",
                "natural_account_segment",
                "cost_center",
                "period_net_dr",
                "period_net_cr",
                "ending_balance",
            },
            "supported_filters": {"ledger_bind", "period_bind", "liability_account_type"},
            "supported_ordering": {"account_combination", "period_name", "ledger_name"},
        },
        "general_ledger_journal_details": {
            "module": FusionModule.GENERAL_LEDGER.value,
            "supported_fields": {
                "ledger_name",
                "journal_name",
                "period_name",
                "journal_source",
                "journal_category",
                "journal_status",
                "journal_line_number",
                "account_combination",
                "debit_amount",
                "credit_amount",
            },
            "supported_filters": {"posted", "posted_journal", "ledger_bind", "period_bind"},
            "supported_ordering": {"journal_name", "period_name", "journal_line_number"},
        },
        "procurement_purchase_order_details": {
            "module": FusionModule.PROCUREMENT.value,
            "supported_fields": {
                "po_number",
                "po_date",
                "po_status",
                "po_type",
                "supplier_name",
                "supplier_number",
                "supplier_site_code",
                "line_number",
                "item_description",
                "ordered_quantity",
                "unit_price",
            },
            "supported_filters": {"approved", "complete"},
            "supported_ordering": {"po_number", "line_number", "po_date"},
        },
        "procurement_receiving_invoicing_match": {
            "module": FusionModule.PROCUREMENT.value,
            "supported_fields": {
                "po_number",
                "line_number",
                "supplier_name",
                "ordered_quantity",
                "received_quantity",
                "billed_quantity",
                "receipt_date",
            },
            "supported_filters": {"approved", "complete"},
            "supported_ordering": {"po_number", "line_number", "receipt_date"},
        },
    }
    SPECIALIZATION_SUMMARY_PATH = Path(
        os.getenv("IWERP_BASE_DIR", "/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
    ) / "specialization_tracks" / "specialization_ingestion_summary.json"
    LOCAL_DOC_MANIFEST_PATHS = [
        Path(os.getenv("IWERP_BASE_DIR", "/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod"))
        / "grounding_answerability_pass"
        / "manifests"
        / "docs_manifest.jsonl",
        Path(os.getenv("IWERP_BASE_DIR", "/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod"))
        / "coverage_expansion"
        / "manifests"
        / "oracle_docs_manifest.jsonl",
        Path(os.getenv("IWERP_BASE_DIR", "/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod"))
        / "coverage_expansion"
        / "second_wave"
        / "manifests"
        / "oracle_docs_manifest.jsonl",
        Path(os.getenv("IWERP_BASE_DIR", "/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod"))
        / "specialization_tracks"
        / "oraclewings_stage2_mass_ingestion"
        / "procedure_objects.jsonl",
        Path(os.getenv("IWERP_BASE_DIR", "/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod"))
        / "specialization_tracks"
        / "oraclewings_stage2_mass_ingestion"
        / "troubleshooting_objects.jsonl",
    ]
    TASK_SEMANTIC_FAILURES = {
        "FAILED_TASK_SEMANTIC_NO_STRONG_MATCH",
        "FAILED_TASK_MODULE_CORRECTION",
    }
    TROUBLESHOOTING_HINTS = {
        "error",
        "errors",
        "issue",
        "issues",
        "failed",
        "failure",
        "exception",
        "exceptions",
        "validation",
        "invalid",
        "resolve",
        "resolution",
        "troubleshoot",
        "troubleshooting",
        "faq",
        "warning",
        "diagnose",
        "fix",
        "debug",
        "status",
    }
    TROUBLESHOOTING_STOPWORDS = {
        "how",
        "do",
        "you",
        "troubleshoot",
        "troubleshooting",
        "oracle",
        "fusion",
        "cloud",
        "issues",
        "issue",
        "errors",
        "error",
        "in",
        "the",
        "for",
        "with",
        "and",
        "what",
        "why",
    }
    RESIDUAL_TROUBLESHOOTING_TASKS = {
        "supplier site setup",
        "intercompany transaction",
        "journal approval",
        "expense report audit",
        "catalog upload",
    }
    SUMMARY_TASK_TYPES = {
        TaskType.GENERAL,
        TaskType.SUMMARY,
        TaskType.INTEGRATION,
    }
    SUMMARY_CONCEPT_PATTERNS = (
        r"^\s*(what\s+is|what's|define|definition\s+of|meaning\s+of|explain|overview\s+of|tell\s+me\s+about)\b",
        r"\bstands\s+for\b",
    )
    SUMMARY_PROCEDURAL_TASK_TYPES = {
        "procedure",
        "navigation",
        "setup",
        "troubleshooting",
    }
    SUMMARY_CONCEPT_MAPPINGS = [
        {
            "id": "enterprise_performance_management",
            "canonical_label": "Enterprise Performance Management",
            "aliases": [
                "epm",
                "enterprise performance management",
                "oracle epm",
                "oracle enterprise performance management",
            ],
            "expansion_terms": [
                "enterprise performance management",
                "planning and budgeting",
                "financial reporting",
                "oracle epm cloud",
            ],
            "must_include_any": ["epm", "enterprise performance management"],
            "min_relevance": 0.26,
            "module_hints": ["Financials"],
        },
        {
            "id": "revenue_management_cloud_service",
            "canonical_label": "Revenue Management Cloud Service",
            "aliases": [
                "rmcs",
                "revenue management cloud service",
                "oracle fusion rmcs",
                "oracle fusion revenue management",
            ],
            "expansion_terms": [
                "revenue management",
                "performance obligations",
                "revenue recognition",
            ],
            "must_include_any": ["rmcs", "revenue management"],
            "min_relevance": 0.24,
            "module_hints": ["Receivables", "Financials"],
        },
        {
            "id": "general_ledger",
            "canonical_label": "General Ledger",
            "aliases": [
                "gl",
                "general ledger",
                "oracle fusion general ledger",
                "oracle general ledger",
            ],
            "expansion_terms": [
                "general ledger",
                "journals",
                "balances",
                "financial reporting",
            ],
            "must_include_any": ["gl", "general ledger"],
            "min_relevance": 0.24,
            "module_hints": ["General Ledger", "Financials"],
        },
        {
            "id": "payroll_gratuity",
            "canonical_label": "Payroll gratuity",
            "aliases": [
                "payroll gratuity",
                "gratuity",
                "gratuity payout",
                "employee gratuity",
                "end of service gratuity",
            ],
            "expansion_terms": [
                "gratuity",
                "termination benefit",
                "end of service benefit",
                "payroll benefit",
            ],
            "must_include_any": ["gratuity"],
            "min_relevance": 0.5,
            "require_exact_concept": True,
            "module_hints": ["HCM"],
        },
    ]
    _specialization_catalog: Dict[str, Any] | None = None
    _local_doc_catalog: List[Dict[str, Any]] | None = None
    SQL_REFUSAL_REASON_TAXONOMY = {
        "SQL_REFUSAL_UNSAFE_REQUEST": "Unsafe, placeholder, or explicitly unsupported SQL request.",
        "SQL_REFUSAL_UNSUPPORTED_FIELDS": "Requested output fields are not fully grounded for the recognized report family.",
        "SQL_REFUSAL_UNSUPPORTED_FILTERS": "Requested filter predicates are not fully grounded for the recognized report family.",
        "SQL_REFUSAL_UNSUPPORTED_ORDERING": "Requested ordering is not fully grounded for the recognized report family.",
        "SQL_REFUSAL_UNSUPPORTED_CALCULATIONS": "Requested calculated/report fields are not safely grounded.",
        "SQL_REFUSAL_NO_GROUNDED_PATTERN": "No grounded SQL example or safe template satisfied the request shape.",
        "SQL_REFUSAL_MODULE_ALIGNMENT_FAILED": "A SQL candidate failed module-family alignment.",
        "SQL_REFUSAL_STYLE_FAILED": "A SQL candidate failed style or no-hardcoding rules.",
        "SQL_REFUSAL_REQUIRED_FIELDS_MISSING": "A SQL candidate failed full requested-field coverage.",
        "SQL_REFUSAL_REQUIRED_JOINS_MISSING": "A SQL candidate failed grounded join coverage.",
        "SQL_REFUSAL_REQUEST_SHAPE_FAILED": "A SQL candidate failed filter or ordering coverage.",
        "SQL_REFUSAL_SPECIALIZED_VERIFIER_FAILED": "The final specialized SQL response failed verifier checks.",
        "SQL_REFUSAL_OTHER": "Other SQL refusal path not covered by a more specific category.",
    }

    def __init__(self):
        self.router = TaskRouter()
        self.search_engine = HybridPostgresSearch()
        self.reranker = LocalReranker() if LLM_CONFIG.get("use_reranker") else None
        self.llm_client = None

        self.verifier = Verifier(
            enable_rag=ENABLE_RAG_VERIFIER,
            enable_sql=ENABLE_SQL_VERIFIER,
            max_retries=MAX_VERIFICATION_RETRIES,
        )
        self.audit_mode = False

    def _sql_query_fingerprint(self, user_query: str) -> str:
        return hashlib.sha256((user_query or "").encode("utf-8")).hexdigest()[:12]

    def _sql_request_shape_log_fields(self, request_shape: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        shape = request_shape or {}

        def _normalize_entries(entries: Any) -> List[str]:
            normalized: List[str] = []
            for item in entries or []:
                if isinstance(item, dict):
                    value = item.get("key") or item.get("label") or item.get("name") or ""
                else:
                    value = item
                text = str(value or "").strip()
                if text:
                    normalized.append(text)
            return normalized

        required_fields = _normalize_entries(shape.get("required_fields"))
        required_filters = _normalize_entries(shape.get("required_filters"))
        required_ordering = _normalize_entries(shape.get("required_ordering"))
        required_calculations = _normalize_entries(shape.get("required_calculations"))
        required_tables = _normalize_entries(shape.get("required_tables"))

        return {
            "report_family": str(shape.get("report_family") or ""),
            "required_field_count": len(required_fields),
            "required_filter_count": len(required_filters),
            "required_ordering_count": len(required_ordering),
            "required_calculation_count": len(required_calculations),
            "required_table_count": len(required_tables),
            "needs_join": bool(shape.get("needs_join")),
        }

    def _sql_shape_supported(self, diagnostics: Dict[str, Any]) -> bool:
        return bool(diagnostics.get("report_family")) and not any(
            diagnostics.get(key)
            for key in ("missing_fields", "missing_filters", "missing_ordering", "missing_calculations")
        )

    def _sql_support_log_fields(self, diagnostics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        info = diagnostics or {}
        missing_fields = list(info.get("missing_fields") or [])
        missing_filters = list(info.get("missing_filters") or [])
        missing_ordering = list(info.get("missing_ordering") or [])
        missing_calculations = list(info.get("missing_calculations") or [])
        return {
            "shape_supported": self._sql_shape_supported(info),
            "unsupported_field_count": len(missing_fields),
            "unsupported_filter_count": len(missing_filters),
            "unsupported_ordering_count": len(missing_ordering),
            "unsupported_calculation_count": len(missing_calculations),
            "unsupported_fields": missing_fields[:8],
            "unsupported_filters": missing_filters[:8],
            "unsupported_ordering": missing_ordering[:8],
            "unsupported_calculations": missing_calculations[:8],
        }

    def _sql_report_family_reason_code(self, request_shape: Optional[Dict[str, Any]]) -> str:
        return "SQL_REPORT_FAMILY_INFERRED" if str((request_shape or {}).get("report_family") or "").strip() else "SQL_REPORT_FAMILY_UNRECOGNIZED"

    def _sql_module_inference_reason_code(
        self,
        route_info: Any,
        module_hint: Optional[str],
        effective_module: str,
        *,
        alignment_override: bool = False,
    ) -> str:
        if alignment_override:
            return "SQL_MODULE_ALIGNMENT_OVERRIDE"
        router_module = self._canonical_module_name(getattr(route_info, "module", None))
        if module_hint and effective_module and effective_module != router_module:
            return "SQL_MODULE_HINT_OVERRIDE"
        return "SQL_MODULE_ROUTER_SELECTED"

    def _sql_shape_support_reason_code(
        self,
        request_shape: Optional[Dict[str, Any]],
        diagnostics: Optional[Dict[str, Any]],
    ) -> str:
        report_family = str((request_shape or {}).get("report_family") or "")
        info = diagnostics or {}
        if not report_family:
            return "SQL_SHAPE_REPORT_FAMILY_UNRECOGNIZED"
        if info.get("missing_fields"):
            return "SQL_SHAPE_UNSUPPORTED_FIELDS"
        if info.get("missing_filters"):
            return "SQL_SHAPE_UNSUPPORTED_FILTERS"
        if info.get("missing_ordering"):
            return "SQL_SHAPE_UNSUPPORTED_ORDERING"
        if info.get("missing_calculations"):
            return "SQL_SHAPE_UNSUPPORTED_CALCULATIONS"
        return "SQL_SHAPE_SUPPORTED"

    def _sql_verifier_reason_code(self, reason: Optional[str]) -> str:
        normalized = str(reason or "").strip().upper()
        if not normalized or normalized == "PASSED":
            return "SQL_VERIFIER_PASSED"
        if "FAILED_SQL_STYLE_VIOLATION" in normalized:
            return "SQL_VERIFIER_STYLE_FAILED"
        if "FAILED_SQL_REQUIRED_FIELDS_MISSING" in normalized:
            return "SQL_VERIFIER_REQUIRED_FIELDS_MISSING"
        if "FAILED_SQL_REQUIRED_JOINS_MISSING" in normalized:
            return "SQL_VERIFIER_REQUIRED_JOINS_MISSING"
        if "FAILED_SQL_REQUEST_SHAPE_MISMATCH" in normalized:
            if "ORDERING" in normalized:
                return "SQL_VERIFIER_ORDERING_FAILED"
            if "FILTER" in normalized:
                return "SQL_VERIFIER_FILTER_FAILED"
            return "SQL_VERIFIER_REQUEST_SHAPE_FAILED"
        if "BELONGS TO FAMILY" in normalized or "MODULE ALIGNMENT" in normalized:
            return "SQL_VERIFIER_MODULE_ALIGNMENT_FAILED"
        if "JOIN PATH IS NOT GROUNDED" in normalized or "JOIN PREDICATES ARE NOT GROUNDED" in normalized:
            return "SQL_VERIFIER_JOIN_GROUNDING_FAILED"
        if "FAILED_SPECIALIZED" in normalized:
            return "SQL_VERIFIER_SPECIALIZED_FAILED"
        if "FAILED_SQL_UNSAFE_REQUEST" in normalized:
            return "SQL_VERIFIER_UNSAFE_REQUEST"
        return "SQL_VERIFIER_FAILED_OTHER"

    def _sql_refusal_reason_code(
        self,
        request_shape: Optional[Dict[str, Any]],
        reason: Optional[str] = None,
        verification_status: Optional[str] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> str:
        info = diagnostics or self._sql_report_family_support_diagnostics(request_shape or {})
        if info.get("missing_fields"):
            return "SQL_REFUSAL_UNSUPPORTED_FIELDS"
        if info.get("missing_filters"):
            return "SQL_REFUSAL_UNSUPPORTED_FILTERS"
        if info.get("missing_ordering"):
            return "SQL_REFUSAL_UNSUPPORTED_ORDERING"
        if info.get("missing_calculations"):
            return "SQL_REFUSAL_UNSUPPORTED_CALCULATIONS"

        normalized = str(verification_status or reason or "").strip().upper()
        if "FAILED_SQL_UNSAFE_REQUEST" in normalized:
            return "SQL_REFUSAL_UNSAFE_REQUEST"
        if "FAILED_SQL_REQUIRED_FIELDS_MISSING" in normalized:
            return "SQL_REFUSAL_REQUIRED_FIELDS_MISSING"
        if "FAILED_SQL_REQUIRED_JOINS_MISSING" in normalized:
            return "SQL_REFUSAL_REQUIRED_JOINS_MISSING"
        if "FAILED_SQL_REQUEST_SHAPE_MISMATCH" in normalized:
            return "SQL_REFUSAL_REQUEST_SHAPE_FAILED"
        if "FAILED_SQL_STYLE_VIOLATION" in normalized:
            return "SQL_REFUSAL_STYLE_FAILED"
        if "FAILED_SPECIALIZED" in normalized:
            return "SQL_REFUSAL_SPECIALIZED_VERIFIER_FAILED"
        if "BELONGS TO FAMILY" in normalized or "MODULE ALIGNMENT" in normalized:
            return "SQL_REFUSAL_MODULE_ALIGNMENT_FAILED"
        if "FAILED_SQL_NO_GROUNDED_PATTERN" in normalized or "NO RETAINED GROUNDED SQL PATTERN" in normalized:
            return "SQL_REFUSAL_NO_GROUNDED_PATTERN"
        return "SQL_REFUSAL_OTHER"

    def _log_sql_decision_event(
        self,
        *,
        stage: str,
        user_query: str,
        route_info: Any,
        module_name: str,
        request_shape: Optional[Dict[str, Any]],
        audit: Optional[Dict[str, Any]],
        reason_code: str,
        module_hint: Optional[str] = None,
        module_alignment_target: Any = None,
        selection_path: Optional[str] = None,
        verifier_status: Optional[str] = None,
        verifier_reason: Optional[str] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        info = diagnostics or self._sql_report_family_support_diagnostics(request_shape or {})
        payload: Dict[str, Any] = {
            "stage": stage,
            "reason_code": reason_code,
            "trace_id": str((audit or {}).get("trace_id") or ""),
            "query_fingerprint": self._sql_query_fingerprint(user_query),
            "query_length": len(user_query or ""),
            "task_type": getattr(getattr(route_info, "task_type", None), "value", getattr(route_info, "task_type", None)),
            "router_module": self._canonical_module_name(getattr(route_info, "module", None)),
            "router_module_family": getattr(getattr(route_info, "module_family", None), "value", getattr(route_info, "module_family", None)),
            "effective_module": module_name,
            "module_hint": str(module_hint or ""),
            "module_alignment_target": getattr(module_alignment_target, "value", module_alignment_target),
            "selection_path": selection_path or "",
            "verification_status": verifier_status or "",
            "decision_execution_mode": str((audit or {}).get("decision_execution_mode") or ""),
            "decision_reason": str((audit or {}).get("decision_reason") or ""),
            "decision_refusal_reason": str((audit or {}).get("decision_refusal_reason") or ""),
        }
        payload.update(self._sql_request_shape_log_fields(request_shape))
        payload.update(self._sql_support_log_fields(info))
        if verifier_status or verifier_reason:
            payload["verifier_reason_code"] = self._sql_verifier_reason_code(verifier_reason or verifier_status)
        logger.info("sql_decision_event", **payload)

    def _ensure_llm_client(self):
        if self.llm_client is not None:
            return self.llm_client
        if LLM_CONFIG.get("inference_backend") == "mlx":
            try:
                from core.llm.mlx_client import MLXClient

                self.llm_client = MLXClient()
            except Exception as exc:  # pragma: no cover - runtime dependency fallback
                logger.warning(
                    "mlx_client_unavailable_falling_back_to_llama_cpp",
                    error=str(exc),
                )
                self.llm_client = LlamaCppClient()
        else:
            self.llm_client = LlamaCppClient()
        return self.llm_client

    def set_audit_mode(self, enabled: bool) -> None:
        self.audit_mode = enabled
        logger.info("phase_18_optimization_active", enabled=enabled)

    def _apply_turbo_quant(self, route_info: Any, request: ChatRequest) -> Dict[str, Any]:
        profile = {
            "temperature": request.temperature or 0.2,
            "top_p": request.top_p or 0.9,
            "repeat_penalty": request.repeat_penalty or 1.1,
        }

        if route_info.task_type in self.SQL_TASKS:
            profile["temperature"] = 0.05
            profile["top_p"] = min(profile["top_p"], 0.9)
            profile["repeat_penalty"] = max(profile["repeat_penalty"], 1.15)
        elif route_info.task_type in self.FAST_FORMULA_TASKS:
            profile["temperature"] = 0.05
            profile["top_p"] = min(profile["top_p"], 0.9)
            profile["repeat_penalty"] = max(profile["repeat_penalty"], 1.12)
        elif route_info.task_type == TaskType.GREETING:
            profile["temperature"] = 0.4
            profile["top_p"] = 1.0

        return profile

    def _requires_sql(self, route_info: Any, query: str) -> bool:
        if route_info.task_type in self.SQL_TASKS:
            return True
        return any(keyword in query.lower() for keyword in [" sql", "query", "join", "select ", "where clause", "otbi"])

    def _is_sql_capable_query(self, route_info: Any, query: str) -> bool:
        if route_info.task_type in self.SQL_TASKS:
            return True
        if route_info.task_type in self.FAST_FORMULA_TASKS:
            return False
        lowered = (query or "").lower()
        if "fast formula" in lowered:
            return False
        strong_signals = (
            "generate grounded oracle sql",
            "oracle sql request",
            "sql case",
            "write sql",
            "create sql",
            "select ",
            " from ",
        )
        metadata_signals = (
            ".xdm",
            "data model",
            "report query",
            "table ",
            "view ",
            " join ",
            " joins ",
            "where clause",
        )
        has_sql_signal = "sql" in lowered or any(token in lowered for token in strong_signals)
        has_metadata_signal = any(token in lowered for token in metadata_signals)
        return bool(has_sql_signal and has_metadata_signal)

    def _requires_formula(self, route_info: Any, query: str) -> bool:
        if route_info.task_type in self.FAST_FORMULA_TASKS:
            return True
        lowered = query.lower()
        return "fast formula" in lowered or "database item" in lowered or "formula type" in lowered

    def _canonical_module_name(self, module: str | FusionModule | None) -> str:
        if module is None:
            return ""
        value = str(module.value if isinstance(module, FusionModule) else module).strip()
        return self.EXACT_MODULE_ALIASES.get(value, value)

    def _is_strict_financial_leaf_route(self, route_info: Any) -> bool:
        if not getattr(route_info, "module_explicit", False):
            return False
        return RetrievalPolicy.is_strict_financial_leaf(self._canonical_module_name(route_info.module))

    def _is_exact_module_chunk(self, chunk: Dict[str, Any], requested_module: str | FusionModule | None) -> bool:
        requested = self._canonical_module_name(requested_module)
        if not requested:
            return False
        metadata = chunk.get("metadata") or {}
        actual = self._canonical_module_name(metadata.get("module"))
        return actual == requested

    def _filter_finance_leaf_chunks(
        self,
        chunks: List[Dict[str, Any]],
        requested_module: str | FusionModule | None,
        allow_same_family_fallback: bool = False,
    ) -> List[Dict[str, Any]]:
        requested = self._canonical_module_name(requested_module)
        if not requested:
            return chunks

        requested_family = next(iter(module_families_for_value(requested)), ModuleFamily.UNKNOWN.value)
        retained: List[Dict[str, Any]] = []
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            actual_module = self._canonical_module_name(metadata.get("module"))
            actual_family = str(metadata.get("module_family") or "")
            if not actual_family and actual_module:
                actual_family = next(iter(module_families_for_value(actual_module)), ModuleFamily.UNKNOWN.value)
            if actual_module == requested:
                retained.append(chunk)
                continue
            if allow_same_family_fallback and actual_family == requested_family:
                retained.append(chunk)
        return retained

    def _count_exact_module_docs(self, chunks: List[Dict[str, Any]], requested_module: str | FusionModule | None) -> int:
        requested = self._canonical_module_name(requested_module)
        if not requested:
            return 0
        return sum(
            1
            for chunk in chunks
            if (chunk.get("metadata") or {}).get("corpus") in self.DOC_GROUNDING_CORPORA
            and self._is_exact_module_chunk(chunk, requested)
        )

    def _count_exact_module_troubleshooting_support(
        self,
        chunks: List[Dict[str, Any]],
        requested_module: str | FusionModule | None,
    ) -> int:
        requested = self._canonical_module_name(requested_module)
        if not requested:
            return 0

        supported_corpora = {*(self.DOC_GROUNDING_CORPORA), "sql_corpus", "sql_examples_corpus"}
        count = 0
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            corpus = str(metadata.get("corpus") or "")
            if corpus not in supported_corpora:
                continue
            if not self._is_exact_module_chunk(chunk, requested):
                continue
            if corpus in self.DOC_GROUNDING_CORPORA:
                count += 1
                continue
            if str(metadata.get("task_match_strength") or "") in {"medium", "strong"}:
                count += 1
        return count

    def _filter_doc_grounding_to_exact_module(
        self,
        chunks: List[Dict[str, Any]],
        requested_module: str | FusionModule | None,
    ) -> List[Dict[str, Any]]:
        requested = self._canonical_module_name(requested_module)
        if not requested:
            return chunks

        retained: List[Dict[str, Any]] = []
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            corpus = str(metadata.get("corpus") or "")
            if corpus not in self.DOC_GROUNDING_CORPORA:
                retained.append(chunk)
                continue
            if self._is_exact_module_chunk(chunk, requested):
                retained.append(chunk)
        return retained

    def _filter_all_chunks_to_exact_module(
        self,
        chunks: List[Dict[str, Any]],
        requested_module: str | FusionModule | None,
    ) -> List[Dict[str, Any]]:
        requested = self._canonical_module_name(requested_module)
        if not requested:
            return chunks

        retained: List[Dict[str, Any]] = []
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            actual_module = self._canonical_module_name(metadata.get("module"))
            if not actual_module or actual_module == requested:
                retained.append(chunk)
        return retained

    def _build_retrieval_filters(self, route_info: Any, retrieval_plan: Any) -> Dict[str, Any]:
        filters: Dict[str, Any] = {
            "task_type": route_info.task_type.value,
            "corpora": retrieval_plan.corpora,
            "quality_score_min": retrieval_plan.budget.quality_score_min,
            "schema_limit": retrieval_plan.budget.max_schema_objects,
        }
        if route_info.module_family != ModuleFamily.UNKNOWN:
            filters["module_family"] = route_info.module_family.value
        if route_info.module != FusionModule.UNKNOWN:
            filters["requested_module"] = route_info.module.value
        elif route_info.module_family != ModuleFamily.UNKNOWN:
            filters["requested_module"] = route_info.module_family.value
        if self._is_strict_financial_leaf_route(route_info):
            exact_module = self._canonical_module_name(route_info.module)
            filters["module"] = exact_module
            filters["requested_module"] = exact_module
            filters["exact_module_allowlist"] = [exact_module]
            filters["strict_exact_module_only"] = True
        return filters

    def _annotate_task_semantics(
        self,
        chunks: List[Dict[str, Any]],
        user_query: str,
        route_info: Any,
        preferred_module_allowlist: List[str] | None = None,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        query_profile = TaskSemanticAnalyzer.extract_query_signals(user_query)
        annotated_chunks = TaskSemanticAnalyzer.annotate_chunks(chunks, query_profile)
        prioritized_chunks = TaskSemanticAnalyzer.prioritize_chunks(annotated_chunks)
        gate = TaskSemanticAnalyzer.summarize_gate(
            prioritized_chunks,
            query_profile,
            requested_module=self._canonical_module_name(route_info.module),
            module_explicit=bool(getattr(route_info, "module_explicit", False)),
        )
        filtered_chunks = TaskSemanticAnalyzer.filter_prompt_chunks(
            prioritized_chunks,
            query_profile,
            docs_expected=route_info.task_type in self.DOCS_EXPECTED_TASKS,
            requested_module=self._canonical_module_name(route_info.module),
            module_explicit=bool(getattr(route_info, "module_explicit", False)),
            preferred_module_allowlist=preferred_module_allowlist,
        )
        return gate, filtered_chunks

    def _specialized_query_tokens(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(token) > 2}

    def _specialized_query_identifiers(self, text: str) -> set[str]:
        raw = text or ""
        identifiers = {token.upper() for token in re.findall(r"\b([A-Z][A-Z0-9_]{3,})\b", raw)}
        # Include CamelCase and mixed-case technical identifiers used in XDM/model names.
        for token in re.findall(r"\b([A-Za-z][A-Za-z0-9_]{4,})\b", raw):
            if re.search(r"[A-Z]", token) and re.search(r"[a-z]", token):
                identifiers.add(token.upper())
        return identifiers

    def _is_summary_concept_prompt(self, user_query: str) -> bool:
        query = str(user_query or "").strip().lower()
        if not query:
            return False
        return any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in self.SUMMARY_CONCEPT_PATTERNS)

    def _summary_concept_profile(self, user_query: str, route_info: Any) -> Optional[Dict[str, Any]]:
        if route_info.task_type not in self.SUMMARY_TASK_TYPES:
            return None
        lowered_query = str(user_query or "").strip().lower()
        if not lowered_query:
            return None

        best_mapping: Optional[Dict[str, Any]] = None
        best_alias_length = 0
        for mapping in self.SUMMARY_CONCEPT_MAPPINGS:
            aliases = [str(alias).strip().lower() for alias in mapping.get("aliases", []) if str(alias).strip()]
            matched_aliases = [alias for alias in aliases if alias in lowered_query]
            if not matched_aliases:
                continue
            longest = max(len(alias) for alias in matched_aliases)
            if longest > best_alias_length:
                best_alias_length = longest
                best_mapping = mapping

        if not best_mapping:
            return None

        aliases = [str(alias).strip().lower() for alias in best_mapping.get("aliases", []) if str(alias).strip()]
        expansion_terms = [
            str(term).strip().lower()
            for term in best_mapping.get("expansion_terms", [])
            if str(term).strip()
        ]
        must_include_any = {
            str(term).strip().lower()
            for term in best_mapping.get("must_include_any", [])
            if str(term).strip()
        }
        profile_tokens = self._specialized_query_tokens(
            " ".join([str(best_mapping.get("canonical_label") or ""), *aliases, *expansion_terms])
        )
        return {
            "id": str(best_mapping.get("id") or ""),
            "canonical_label": str(best_mapping.get("canonical_label") or ""),
            "aliases": aliases,
            "expansion_terms": expansion_terms,
            "must_include_any": must_include_any,
            "min_relevance": float(best_mapping.get("min_relevance") or 0.25),
            "require_exact_concept": bool(best_mapping.get("require_exact_concept")),
            "module_hints": {
                self._canonical_module_name(str(module_name).strip())
                for module_name in best_mapping.get("module_hints", [])
                if str(module_name).strip()
            },
            "profile_tokens": profile_tokens,
        }

    def _summary_concept_relevance(
        self,
        *,
        user_query: str,
        metadata: Dict[str, Any],
        title: str,
        content: str,
        concept_profile: Optional[Dict[str, Any]],
    ) -> float:
        evidence_parts = [
            str(title or ""),
            str(content or "")[:1800],
            self._metadata_text(
                metadata,
                [
                    "title",
                    "task_name",
                    "query_aliases",
                    "summary_points",
                    "purpose",
                    "warnings",
                ],
            ),
        ]
        evidence = " ".join(part for part in evidence_parts if part).strip().lower()
        if not evidence:
            return 0.0
        evidence_tokens = self._specialized_query_tokens(evidence)

        if not concept_profile:
            lowered_query = str(user_query or "").strip().lower()
            if not lowered_query:
                return 0.0
            generic_focus = re.sub(
                r"^\s*(what\s+is|what's|define|definition\s+of|meaning\s+of|explain|overview\s+of|tell\s+me\s+about)\s+",
                "",
                lowered_query,
                flags=re.IGNORECASE,
            )
            generic_focus = re.sub(r"\bin\s+oracle\s+fusion\b", " ", generic_focus, flags=re.IGNORECASE)
            generic_focus = re.sub(r"\bin\s+oracle\b", " ", generic_focus, flags=re.IGNORECASE)
            generic_focus = re.sub(r"\boracle\s+fusion\b", " ", generic_focus, flags=re.IGNORECASE)
            generic_focus = re.sub(r"\boracle\b", " ", generic_focus, flags=re.IGNORECASE)
            generic_focus = re.sub(r"\s+", " ", generic_focus).strip(" ?.-")
            focus_head = generic_focus.split(" in ", 1)[0].strip()
            focus_text = focus_head or generic_focus or lowered_query
            query_tokens = self._specialized_query_tokens(focus_text)
            if not query_tokens:
                return 0.0

            metadata_text = self._metadata_text(
                metadata,
                [
                    "task_name",
                    "query_aliases",
                    "summary_points",
                    "purpose",
                    "warnings",
                ],
            )
            title_tokens = self._specialized_query_tokens(title)
            metadata_tokens = self._specialized_query_tokens(metadata_text)
            focus_normalized = re.sub(r"[^a-z0-9]+", " ", focus_text).strip()
            evidence_normalized = re.sub(r"[^a-z0-9]+", " ", evidence).strip()
            exact_focus_hit = bool(focus_normalized and focus_normalized in evidence_normalized)
            score = min(
                1.0,
                (self._specialized_overlap_score(query_tokens, title_tokens) * 0.65)
                + (self._specialized_overlap_score(query_tokens, metadata_tokens) * 0.9)
                + (self._specialized_overlap_score(query_tokens, evidence_tokens) * 0.4)
                + (0.35 if exact_focus_hit else 0.0),
            )
            return float(score)

        alias_hits = sum(1 for alias in concept_profile.get("aliases", []) if alias and alias in evidence)
        must_include_hits = sum(1 for token in concept_profile.get("must_include_any", set()) if token in evidence)
        token_overlap = self._specialized_overlap_score(
            concept_profile.get("profile_tokens", set()),
            evidence_tokens,
        )

        score = min(
            1.0,
            (alias_hits * 0.34)
            + (must_include_hits * 0.46)
            + (token_overlap * 0.85),
        )

        if concept_profile.get("require_exact_concept") and must_include_hits == 0:
            return 0.0
        if concept_profile.get("must_include_any") and must_include_hits == 0 and alias_hits == 0:
            score *= 0.25

        module_hints = concept_profile.get("module_hints", set())
        actual_module = self._canonical_module_name(metadata.get("module"))
        if module_hints and actual_module in module_hints:
            score = min(1.0, score + 0.08)
        return float(score)

    def _is_summary_procedural_doc(self, metadata: Dict[str, Any]) -> bool:
        task_type = str(metadata.get("task_type") or "").strip().lower()
        doc_type = str(metadata.get("doc_type") or "").strip().lower()
        if task_type in self.SUMMARY_PROCEDURAL_TASK_TYPES:
            return True
        return any(
            marker in doc_type
            for marker in ("procedure", "setup", "navigation", "troubleshooting")
        )

    def _expand_summary_query(self, user_query: str, route_info: Any) -> str:
        concept_profile = self._summary_concept_profile(user_query, route_info)
        if not concept_profile:
            return user_query
        lowered_query = str(user_query or "").lower()
        expansion_terms = [
            term
            for term in concept_profile.get("expansion_terms", [])
            if term and term not in lowered_query
        ]
        if not expansion_terms:
            return user_query
        expansion_suffix = " ".join(expansion_terms[:4])
        return f"{user_query}\n{concept_profile.get('canonical_label')}: {expansion_suffix}".strip()

    def _summary_safety_assessment(
        self,
        *,
        user_query: str,
        route_info: Any,
        metadata: Dict[str, Any],
        title: str,
        content: str,
    ) -> Dict[str, Any]:
        if route_info.task_type not in self.SUMMARY_TASK_TYPES:
            return {
                "applicable": False,
                "strong_match": True,
                "reason": "not_summary_task",
                "concept_relevance": 0.0,
                "threshold": 0.0,
                "score": 0.0,
            }

        concept_profile = self._summary_concept_profile(user_query, route_info)
        is_concept_prompt = self._is_summary_concept_prompt(user_query)
        concept_relevance = self._summary_concept_relevance(
            user_query=user_query,
            metadata=metadata,
            title=title,
            content=content,
            concept_profile=concept_profile,
        )
        procedural_doc = self._is_summary_procedural_doc(metadata)
        deterministic = bool(metadata.get("allow_deterministic_grounded_answer"))
        has_summary_support = bool(metadata.get("purpose") or metadata.get("summary_points") or title.strip())
        query_aliases = [alias.lower() for alias in self._coerce_string_list(metadata.get("query_aliases"))]
        task_name = str(metadata.get("task_name") or "").strip().lower()
        lowered_query = str(user_query or "").strip().lower()
        exact_task_hit = bool(
            (task_name and task_name in lowered_query)
            or any(alias and alias in lowered_query for alias in query_aliases)
        )

        requested_module = self._canonical_module_name(route_info.module)
        actual_module = self._canonical_module_name(metadata.get("module"))
        requested_families = set(module_families_for_value(requested_module)) if requested_module else set()
        actual_families = set(module_families_for_value(actual_module)) if actual_module else set()
        module_ok = True
        if (
            getattr(route_info, "module_explicit", False)
            and requested_module
            and actual_module
            and actual_module not in {requested_module, FusionModule.COMMON.value}
        ):
            module_ok = bool(requested_families & actual_families)

        if concept_profile:
            threshold = 0.84 if concept_profile.get("require_exact_concept") else 0.74
        elif is_concept_prompt:
            threshold = 0.7
        else:
            threshold = 0.62

        strong_match = True
        reason = "strong_summary_grounding"
        if (
            is_concept_prompt
            and procedural_doc
            and not (
                exact_task_hit
                and has_summary_support
                and concept_relevance >= max(threshold, 0.74)
            )
        ):
            strong_match = False
            reason = "procedural_doc_for_concept_query"
        elif not module_ok:
            strong_match = False
            reason = "weak_module_alignment"
        elif concept_relevance < threshold:
            strong_match = False
            reason = "summary_confidence_not_strong"
        elif not has_summary_support and not deterministic:
            strong_match = False
            reason = "summary_support_missing"

        score = concept_relevance
        if has_summary_support:
            score += 0.08
        if deterministic:
            score += 0.08
        if module_ok:
            score += 0.04

        return {
            "applicable": True,
            "strong_match": strong_match,
            "reason": reason,
            "concept_relevance": float(concept_relevance),
            "threshold": float(threshold),
            "procedural_doc": procedural_doc,
            "module_ok": module_ok,
            "has_summary_support": has_summary_support,
            "deterministic": deterministic,
            "exact_task_hit": exact_task_hit,
            "score": float(min(score, 1.0)),
        }

    def _evaluate_strict_summary_grounding(
        self,
        *,
        user_query: str,
        route_info: Any,
        mapped_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if route_info.task_type not in self.SUMMARY_TASK_TYPES:
            return {"applicable": False, "allow": True}

        doc_chunks = [
            chunk
            for chunk in mapped_chunks
            if (chunk.get("metadata") or {}).get("corpus") in self.DOC_GROUNDING_CORPORA
        ]
        if not doc_chunks:
            return {
                "applicable": True,
                "allow": False,
                "reason": "no_doc_grounding",
                "concept_relevance": 0.0,
                "threshold": 0.0,
                "top_title": None,
            }

        best_assessment: Dict[str, Any] | None = None
        best_title: str | None = None
        for chunk in doc_chunks:
            metadata = chunk.get("metadata") or {}
            title = str(metadata.get("title") or chunk.get("title") or "")
            content = str(chunk.get("content") or "")
            assessment = self._summary_safety_assessment(
                user_query=user_query,
                route_info=route_info,
                metadata=metadata,
                title=title,
                content=content,
            )
            if best_assessment is None or assessment.get("score", 0.0) > best_assessment.get("score", 0.0):
                best_assessment = assessment
                best_title = title

        if best_assessment is None:
            return {
                "applicable": True,
                "allow": False,
                "reason": "no_summary_candidate",
                "concept_relevance": 0.0,
                "threshold": 0.0,
                "top_title": None,
            }

        return {
            "applicable": True,
            "allow": bool(best_assessment.get("strong_match")),
            "reason": best_assessment.get("reason"),
            "concept_relevance": float(best_assessment.get("concept_relevance") or 0.0),
            "threshold": float(best_assessment.get("threshold") or 0.0),
            "top_title": best_title,
        }

    def _match_summary_concept_seed(
        self,
        user_query: str,
        route_info: Any,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        concept_profile = self._summary_concept_profile(user_query, route_info)
        if not concept_profile or route_info.task_type not in self.SUMMARY_TASK_TYPES:
            return None

        lowered_query = str(user_query or "").strip().lower()
        best_record: Dict[str, Any] | None = None
        best_score = 0.0
        for record in records:
            metadata = dict(record.get("metadata") or {})
            if str(metadata.get("source_system") or "").strip() != "local_concept_seed":
                continue
            aliases = [alias.lower() for alias in self._coerce_string_list(record.get("query_aliases") or metadata.get("query_aliases"))]
            task_name = str(record.get("task_name") or metadata.get("task_name") or "").strip().lower()
            if not (
                (task_name and task_name in lowered_query)
                or any(alias and alias in lowered_query for alias in aliases)
            ):
                continue
            assessment = self._summary_safety_assessment(
                user_query=user_query,
                route_info=route_info,
                metadata=metadata,
                title=str(record.get("title") or ""),
                content=str(record.get("content") or ""),
            )
            if not assessment.get("strong_match"):
                continue
            score = float(assessment.get("score") or 0.0)
            if score > best_score:
                best_score = score
                best_record = record
        return best_record

    def _infer_sql_module_hint(self, user_query: str) -> Optional[str]:
        lowered_query = (user_query or "").lower()
        catalog = self._load_specialization_catalog()
        known_modules: Set[str] = {
            str(item.get("module") or "").strip()
            for item in (catalog.get("sql_records") or [])
            if str(item.get("module") or "").strip()
        }
        known_modules.update(
            {
                FusionModule.PAYABLES.value,
                FusionModule.RECEIVABLES.value,
                FusionModule.GENERAL_LEDGER.value,
                FusionModule.CASH_MANAGEMENT.value,
                FusionModule.ASSETS.value,
                FusionModule.EXPENSES.value,
                FusionModule.PROCUREMENT.value,
                FusionModule.SCM.value,
                FusionModule.HCM.value,
                FusionModule.PROJECTS.value,
                FusionModule.TAX.value,
                ModuleFamily.FINANCIALS.value,
                ModuleFamily.PROCUREMENT.value,
                ModuleFamily.HCM.value,
                ModuleFamily.SCM.value,
                ModuleFamily.PROJECTS.value,
            }
        )
        for module_name in sorted(known_modules, key=len, reverse=True):
            lowered_name = module_name.lower()
            if lowered_name and lowered_name in lowered_query:
                return module_name
        return None

    def _sql_alignment_target(self, route_info: Any, module_hint: Optional[str]) -> Any:
        if not module_hint:
            return route_info.module
        if route_info.module != FusionModule.UNKNOWN:
            route_families = module_families_for_value(route_info.module.value)
            hint_families = module_families_for_value(module_hint)
            route_families.discard(ModuleFamily.UNKNOWN.value)
            hint_families.discard(ModuleFamily.UNKNOWN.value)
            if route_families & hint_families:
                return route_info.module
        hint_families = module_families_for_value(module_hint)
        hint_families.discard(ModuleFamily.UNKNOWN.value)
        if hint_families:
            return next(iter(hint_families))
        return route_info.module

    def _repair_sql_for_style(
        self,
        sql_text: str,
        request_shape: Optional[Dict[str, Any]] = None,
    ) -> str:
        sql_clean = (sql_text or "").strip().rstrip(";")
        if not sql_clean:
            return ""
        try:
            tree = sqlglot.parse_one(sql_clean, read="oracle")
        except Exception:
            return sql_clean + ";"

        table_aliases: Dict[str, str] = {}
        alias_counter = 1
        for table in tree.find_all(exp.Table):
            table_name = str(table.name or "").upper()
            current_alias = str(table.alias_or_name or "").upper()
            if not current_alias or current_alias == table_name:
                generated_alias = f"t{alias_counter}"
                alias_counter += 1
                table.set("alias", exp.TableAlias(this=exp.to_identifier(generated_alias)))
                table_aliases[table_name] = generated_alias
            else:
                table_aliases[table_name] = current_alias

        all_tables = []
        for table in tree.find_all(exp.Table):
            canonical = self.verifier.registry.resolve_object_name(str(table.name or "").upper()) or str(table.name or "").upper()
            alias = str(table.alias_or_name or "").strip()
            all_tables.append((canonical, alias))

        for column in tree.find_all(exp.Column):
            qualifier = str(column.table or "").upper()
            if qualifier and qualifier in table_aliases:
                column.set("table", exp.to_identifier(table_aliases[qualifier]))
                continue
            if qualifier:
                continue
            if len(all_tables) == 1 and all_tables[0][1]:
                column.set("table", exp.to_identifier(all_tables[0][1]))
                continue
            candidate_aliases = [
                alias
                for canonical, alias in all_tables
                if alias and self.verifier.registry.has_column(canonical, str(column.name or "").upper())
            ]
            if len(candidate_aliases) == 1:
                column.set("table", exp.to_identifier(candidate_aliases[0]))

        required_filters = list((request_shape or {}).get("required_filters") or [])
        where_clause = tree.args.get("where")
        if where_clause is not None:
            placeholder_index = 1
            for literal in where_clause.find_all(exp.Literal):
                literal_sql = literal.sql(dialect="oracle").strip().upper()
                if literal_sql == "NULL":
                    continue
                placeholder = exp.Placeholder(this=f"P_FILTER_{placeholder_index}")
                placeholder_index += 1
                literal.replace(placeholder)
            # If no required filter was requested, prefer dropping restrictive predicates
            # to avoid style rejection from hardcoded values in legacy examples.
            if not required_filters and placeholder_index > 1:
                tree.set("where", None)

        try:
            return tree.sql(dialect="oracle") + ";"
        except Exception:
            return sql_clean + ";"

    def _sql_candidate_variants(
        self,
        sql_text: str,
        request_shape: Optional[Dict[str, Any]],
    ) -> List[str]:
        variants: List[str] = []
        normalized = self._normalize_sql_output_block(sql_text or "")
        if normalized:
            variants.append(normalized)
        repaired = self._repair_sql_for_style(normalized or sql_text or "", request_shape=request_shape)
        repaired = self._normalize_sql_output_block(repaired or "")
        if repaired and repaired not in variants:
            variants.append(repaired)
        return variants

    def _verify_sql_candidate(
        self,
        sql_text: str,
        *,
        module_alignment_target: Any,
        request_shape: Optional[Dict[str, Any]],
    ) -> Tuple[bool, Optional[str]]:
        sql_ok, sql_reason = self.verifier.verify_sql(sql_text)
        if not sql_ok:
            return False, sql_reason
        module_ok, module_reason = self.verifier.verify_module_alignment(sql_text, module_alignment_target)
        if not module_ok:
            return False, module_reason
        style_ok, style_reason = self.verifier.verify_sql_style(sql_text)
        if not style_ok:
            return False, style_reason
        shape_ok, shape_reason = self.verifier.verify_sql_request_shape(sql_text, request_shape)
        if not shape_ok:
            return False, shape_reason
        return True, None

    def _specialized_module_compatible(self, route_info: Any, record_module: str) -> bool:
        candidate = str(record_module or "").strip()
        if not candidate:
            return True

        route_targets = set()
        if route_info.module != FusionModule.UNKNOWN:
            route_targets.add(route_info.module.value)
        if route_info.module_family != ModuleFamily.UNKNOWN:
            route_targets.add(route_info.module_family.value)
        if not route_targets:
            return True
        if candidate in route_targets:
            return True

        route_families = set()
        for target in route_targets:
            route_families.update(module_families_for_value(target))
        record_families = module_families_for_value(candidate)
        route_families.discard(ModuleFamily.UNKNOWN.value)
        record_families.discard(ModuleFamily.UNKNOWN.value)
        return bool(route_families & record_families)

    def _specialized_overlap_score(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / max(min(len(left), len(right)), 1)

    def _normalize_formula_type(self, value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        if not text:
            return "UNKNOWN"
        lowered = text.lower()
        if lowered in {"unknown", "n/a", "na", "none", "not applicable"}:
            return "UNKNOWN"
        for alias, canonical in sorted(self.FAST_FORMULA_TYPE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
            if alias in lowered:
                return canonical
        cleaned = re.sub(r"[^A-Za-z0-9 /_-]", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return "UNKNOWN"
        if cleaned.lower() in {"unknown", "n/a", "na", "none", "not applicable"}:
            return "UNKNOWN"
        words = []
        for word in cleaned.split(" "):
            if not word:
                continue
            if word.upper() in {"HCM", "HR", "PAY", "WFM"}:
                words.append(word.upper())
            else:
                words.append(word.capitalize())
        normalized = " ".join(words) if words else "UNKNOWN"
        return "UNKNOWN" if normalized.lower() == "unknown" else normalized

    def _infer_formula_type_from_text(self, text: str) -> str:
        lowered = (text or "").lower()
        if not lowered:
            return "UNKNOWN"
        scores: Dict[str, int] = defaultdict(int)

        def hit(formula_type: str, tokens: Tuple[str, ...], weight: int = 1) -> None:
            for token in tokens:
                if token in lowered:
                    scores[formula_type] += weight

        hit("Auto Indirect", ("auto_indirect", "deduction_card_by_ee", "all_tax_unit_by_dir_card", "all_cb_by_dir_card"), 4)
        hit("Global Absence Entry Validation", ("entry validation", "validation", "entry_value", "invalid", "valid"), 3)
        hit("Absence", ("accrual", "months_between", "absence plan", "accrual_days", "hire_date", "term_date"), 3)
        hit("Proration", ("proration", "days_between", "proration_days", "calc_start_date", "calc_end_date"), 2)
        hit("Rate Value Calculation", ("conversion_rate", "base_amount", "rate", "round("), 2)
        hit("Element Skip", ("skip", "flow skip"), 2)
        hit("Time Calculation Rules", ("workforce_management", "hwm_", "time rule", "reported_hours"), 2)
        hit("Extract Rule", ("extract", "to_char", "effective_date"), 1)
        hit("Oracle Payroll", ("pay_internal_log_write", "payroll", "salary", "deduction"), 1)

        if not scores:
            return "UNKNOWN"

        precedence = {
            "Auto Indirect": 9,
            "Global Absence Entry Validation": 8,
            "Absence": 7,
            "Proration": 6,
            "Rate Value Calculation": 5,
            "Element Skip": 4,
            "Time Calculation Rules": 3,
            "Oracle Payroll": 2,
            "Extract Rule": 1,
        }
        best_type, _ = max(
            scores.items(),
            key=lambda item: (item[1], precedence.get(item[0], 0)),
        )
        return best_type

    def _normalize_formula_name(self, value: str) -> str:
        text = str(value or "").strip().upper()
        text = re.sub(r"[^A-Z0-9_]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text

    def _normalize_formula_block_text(self, value: str) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = text.replace('"', "").replace("'", "")
        return text

    def _is_low_signal_formula_example(self, metadata: Dict[str, Any]) -> bool:
        title = str(metadata.get("title") or "").strip().lower()
        source = str(metadata.get("source_uri") or metadata.get("source_file") or "").lower()
        formula_type = self._normalize_formula_type(str(metadata.get("formula_type") or "UNKNOWN"))
        if title in self.FAST_FORMULA_GENERIC_TITLES:
            return True
        if formula_type != "UNKNOWN":
            return False
        if "fast_formula_knowledge.json" in source and title:
            return True
        return False

    def _extract_broken_formula(self, user_query: str) -> str:
        match = re.search(r"(?is)broken\s+formula\s*:\s*(.+)$", user_query or "")
        if not match:
            return ""
        block = match.group(1).strip()
        block = re.sub(r"[?]\s*$", "", block).strip()
        return block

    def _clean_formula_text(self, formula_text: str) -> str:
        cleaned_lines: List[str] = []
        for raw_line in str(formula_text or "").splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                if cleaned_lines and cleaned_lines[-1]:
                    cleaned_lines.append("")
                continue
            if re.fullmatch(r"[-=*/# ]{4,}", stripped):
                continue
            if stripped.upper().startswith("USE CASE:"):
                continue
            if stripped.startswith("--"):
                uncommented = re.sub(r"^\s*--\s*", "", stripped)
                if re.match(r"(?i)^return\b", uncommented):
                    line = uncommented
                else:
                    continue
            if stripped.startswith("/*") and stripped.endswith("*/"):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    def _extract_formula_inputs(self, formula_text: str) -> List[str]:
        match = re.search(r"(?im)^\s*inputs\s+are\s+(.+)$", formula_text or "")
        if not match:
            return []
        return [item.strip() for item in match.group(1).split(",") if item.strip()]

    def _formula_kind(self, formula_type: str, use_case: str) -> str:
        lowered = f"{formula_type} {use_case}".lower()
        if "proration" in lowered:
            return "proration"
        if "accrual" in lowered:
            return "accrual"
        if any(token in lowered for token in ("validation", "eligibility")):
            return "validation"
        if any(token in lowered for token in ("rate", "conversion")):
            return "rate"
        if any(token in lowered for token in ("time", "workforce management")):
            return "time"
        if "extract" in lowered:
            return "extract"
        return "payroll"

    def _formula_defaults(self, kind: str) -> Dict[str, Any]:
        defaults = {
            "proration": {
                "inputs": ["start_date (DATE)", "end_date (DATE)"],
                "db_items": ["CALC_START_DATE", "CALC_END_DATE"],
                "contexts": ["CALC_START_DATE", "CALC_END_DATE"],
                "return_var": "proration_factor",
            },
            "accrual": {
                "inputs": ["hire_date (DATE)", "term_date (DATE)"],
                "db_items": ["PER_ASG_EFFECTIVE_START_DATE", "PAYROLL_REL_ACTION_ID"],
                "contexts": ["PAYROLL_REL_ACTION_ID", "DATE_EARNED"],
                "return_var": "accrual_value",
            },
            "validation": {
                "inputs": ["entry_value (TEXT)"],
                "db_items": ["PER_ASG_PERSON_ID", "PAYROLL_REL_ACTION_ID"],
                "contexts": ["PAYROLL_REL_ACTION_ID"],
                "return_var": "validation_result",
            },
            "rate": {
                "inputs": ["base_amount (NUMBER)", "conversion_rate (NUMBER)"],
                "db_items": ["CMP_ASSIGNMENT_SALARY_AMOUNT", "PAYROLL_REL_ACTION_ID"],
                "contexts": ["DATE_EARNED"],
                "return_var": "rate_value",
            },
            "time": {
                "inputs": ["reported_hours (NUMBER)"],
                "db_items": ["HWM_MEASURE_DAY", "HWM_RECORD_POSITION"],
                "contexts": ["DATE_EARNED"],
                "return_var": "calculated_hours",
            },
            "extract": {
                "inputs": ["effective_date (DATE)"],
                "db_items": ["PER_ASG_PERSON_NUMBER", "PER_ASG_ASSIGNMENT_ID"],
                "contexts": ["DATE_EARNED"],
                "return_var": "result_value",
            },
            "payroll": {
                "inputs": ["input_amount (NUMBER)"],
                "db_items": ["PAYROLL_REL_ACTION_ID", "PER_ASG_PERSON_ID"],
                "contexts": ["PAYROLL_REL_ACTION_ID", "DATE_EARNED"],
                "return_var": "result_value",
            },
        }
        return defaults.get(kind, defaults["payroll"])

    def _grounded_formula_identifiers(self, values: List[str]) -> List[str]:
        accepted: List[str] = []
        for value in values:
            token = str(value or "").strip().upper()
            if not token:
                continue
            if token in self.FAST_FORMULA_NOISE_IDENTIFIERS:
                continue
            if token.startswith(("UNKNOWN_", "INVALID_", "MISSING_")):
                continue
            if "_" not in token:
                continue
            if not re.match(r"^[A-Z][A-Z0-9_]{2,}$", token):
                continue
            if token not in accepted:
                accepted.append(token)
        return accepted

    def _parse_fast_formula_request_shape(self, user_query: str, route_info: Any) -> Dict[str, Any]:
        broken_formula = self._extract_broken_formula(user_query)
        requested_formula_type = "UNKNOWN"
        for pattern in (
            r"(?is)formula\s+type\s+aligned\s+to\s+([A-Za-z0-9 _/\-]+)",
            r"(?is)keep\s+the\s+formula\s+type\s+aligned\s+to\s+([A-Za-z0-9 _/\-]+)",
            r"(?is)formula\s+type\s*[:=]\s*([A-Za-z0-9 _/\-]+)",
        ):
            match = re.search(pattern, user_query or "")
            if match:
                requested_formula_type = self._normalize_formula_type(match.group(1))
                break

        if requested_formula_type == "UNKNOWN":
            header_match = re.search(r"(?im)^\s*FORMULA\s+TYPE\s*:\s*(.+)$", broken_formula)
            if header_match:
                requested_formula_type = self._normalize_formula_type(header_match.group(1))
        if requested_formula_type == "UNKNOWN":
            infer_text = broken_formula if (
                route_info.task_type == TaskType.FAST_FORMULA_TROUBLESHOOTING and broken_formula
            ) else f"{user_query}\n{broken_formula}"
            inferred_type = self._infer_formula_type_from_text(infer_text)
            if inferred_type != "UNKNOWN":
                requested_formula_type = inferred_type

        formula_name = ""
        name_match = re.search(r"(?is)\bfor\s+(.+?)\s+in\s+oracle\s+fusion", user_query or "")
        if name_match:
            formula_name = re.sub(r"\s+", " ", name_match.group(1)).strip()
        if not formula_name:
            name_match = re.search(r"(?im)^\s*FORMULA\s+NAME\s*:\s*(.+)$", broken_formula)
            if name_match:
                formula_name = name_match.group(1).strip()

        requested_identifiers = self._specialized_query_identifiers(f"{user_query}\n{broken_formula}")
        requested_dbis = self._grounded_formula_identifiers(list(requested_identifiers))
        troubleshooting_mode = (
            route_info.task_type == TaskType.FAST_FORMULA_TROUBLESHOOTING
            or "troubleshoot" in (user_query or "").lower()
        )
        broken_tokens = {
            token
            for token in self._specialized_query_tokens(broken_formula)
            if token not in self.FAST_FORMULA_TOKEN_STOPWORDS
        }

        return {
            "requested_formula_type": requested_formula_type,
            "formula_name": formula_name,
            "formula_name_normalized": self._normalize_formula_name(formula_name),
            "broken_formula": broken_formula,
            "requested_identifiers": requested_identifiers,
            "requested_database_items": requested_dbis,
            "is_troubleshooting": troubleshooting_mode,
            "broken_formula_tokens": broken_tokens,
        }

    def _build_grounded_formula_template(
        self,
        formula_type: str,
        use_case: str,
        *,
        database_items: List[str],
        input_values: List[str],
    ) -> str:
        kind = self._formula_kind(formula_type, use_case)
        defaults = self._formula_defaults(kind)
        db_items = self._grounded_formula_identifiers(database_items) or defaults["db_items"]
        inputs = input_values or defaults["inputs"]
        return_var = defaults["return_var"]
        primary_dbi = db_items[0]
        secondary_dbi = db_items[1] if len(db_items) > 1 else db_items[0]

        if kind == "proration":
            return "\n".join(
                [
                    "DEFAULT FOR CALC_START_DATE IS (DATE '1900-01-01')",
                    "DEFAULT FOR CALC_END_DATE IS (DATE '4712-12-31')",
                    f"INPUTS ARE {', '.join(inputs)}",
                    "proration_start = start_date",
                    "proration_end = end_date",
                    "IF proration_start < CALC_START_DATE THEN",
                    "proration_start = CALC_START_DATE",
                    "ENDIF",
                    "IF proration_end > CALC_END_DATE THEN",
                    "proration_end = CALC_END_DATE",
                    "ENDIF",
                    f"{return_var} = ROUND((DAYS_BETWEEN(proration_end, proration_start) + 1) / 30, 4)",
                    f"RETURN {return_var}",
                ]
            )
        if kind == "accrual":
            return "\n".join(
                [
                    f"DEFAULT FOR {primary_dbi} IS 0",
                    "DEFAULT FOR PER_ASG_EFFECTIVE_START_DATE IS (DATE '1900-01-01')",
                    f"INPUTS ARE {', '.join(inputs)}",
                    f"{return_var} = 0",
                    "IF hire_date WAS DEFAULTED THEN",
                    "hire_date = PER_ASG_EFFECTIVE_START_DATE",
                    "ENDIF",
                    f"{return_var} = ROUND(MONTHS_BETWEEN(NVL(term_date, DATE '4712-12-31'), hire_date), 2)",
                    f"RETURN {return_var}",
                ]
            )
        if kind == "validation":
            return "\n".join(
                [
                    f"DEFAULT FOR {primary_dbi} IS 0",
                    f"INPUTS ARE {', '.join(inputs)}",
                    f"{return_var} = 'VALID'",
                    "IF entry_value WAS DEFAULTED THEN",
                    f"{return_var} = 'INVALID'",
                    "ENDIF",
                    f"RETURN {return_var}",
                ]
            )
        if kind == "rate":
            return "\n".join(
                [
                    f"DEFAULT FOR {primary_dbi} IS 0",
                    f"INPUTS ARE {', '.join(inputs)}",
                    f"{return_var} = ROUND(base_amount * conversion_rate, 2)",
                    f"RETURN {return_var}",
                ]
            )
        if kind == "time":
            return "\n".join(
                [
                    f"DEFAULT FOR {primary_dbi} IS 0",
                    f"INPUTS ARE {', '.join(inputs)}",
                    f"{return_var} = ROUND(reported_hours, 2)",
                    f"RETURN {return_var}",
                ]
            )
        return "\n".join(
            [
                f"DEFAULT FOR {primary_dbi} IS 0",
                f"DEFAULT FOR {secondary_dbi} IS 0",
                f"INPUTS ARE {', '.join(inputs)}",
                f"{return_var} = {primary_dbi}",
                f"RETURN {return_var}",
            ]
        )

    def _repair_formula_from_broken_input(
        self,
        broken_formula: str,
        *,
        formula_type: str,
        use_case: str,
        database_items: List[str],
        input_values: List[str],
    ) -> str:
        cleaned = self._clean_formula_text(broken_formula)
        if not cleaned:
            return self._build_grounded_formula_template(
                formula_type,
                use_case,
                database_items=database_items,
                input_values=input_values,
            )

        lines = cleaned.splitlines()
        if not any(re.match(r"(?im)^\s*inputs\s+are\b", line) for line in lines):
            defaults = self._formula_defaults(self._formula_kind(formula_type, use_case))
            inputs = input_values or defaults["inputs"]
            lines.insert(0, f"INPUTS ARE {', '.join(inputs)}")

        if not any(re.match(r"(?im)^\s*default\s+for\b", line) for line in lines):
            db_items = self._grounded_formula_identifiers(database_items)
            if db_items:
                lines.insert(0, f"DEFAULT FOR {db_items[0]} IS 0")

        if "return" not in cleaned.lower():
            assignment_vars = re.findall(r"(?im)^\s*([a-z][a-z0-9_]*)\s*=", cleaned)
            return_var = assignment_vars[-1] if assignment_vars else self._formula_defaults(self._formula_kind(formula_type, use_case))["return_var"]
            lines.append(f"RETURN {return_var}")

        if_count = len(re.findall(r"(?im)^\s*if\b", "\n".join(lines)))
        endif_count = len(re.findall(r"(?im)\bendif\b|\bend\s+if\b", "\n".join(lines)))
        while endif_count < if_count:
            lines.append("ENDIF")
            endif_count += 1

        return "\n".join(lines).strip()

    def _doc_corpus_boost(self, corpus: str, route_info: Any, metadata: Dict[str, Any]) -> float:
        if corpus == "troubleshooting_corpus":
            return 2.2 if route_info.task_type == TaskType.TROUBLESHOOTING else 1.55
        if corpus == "docs_corpus":
            return 1.7 if route_info.task_type == TaskType.TROUBLESHOOTING else 1.35
        if corpus == "schema_corpus":
            return 0.65
        if corpus == "sql_corpus":
            return 0.35
        return 0.0

    def _coerce_string_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in re.split(r"[;\n]", value) if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []

    def _metadata_text(self, metadata: Dict[str, Any], keys: List[str]) -> str:
        values: List[str] = []
        for key in keys:
            item = metadata.get(key)
            if isinstance(item, (list, tuple, set)):
                values.extend(str(entry) for entry in item if str(entry).strip())
            elif item is not None and str(item).strip():
                values.append(str(item))
        return " ".join(values)

    def _preferred_leaf_note(self, route_info: Any, metadata: Dict[str, Any]) -> str | None:
        preferred_leaf = str(metadata.get("preferred_leaf_module") or "").strip()
        if not preferred_leaf:
            return None
        requested_module = self._canonical_module_name(route_info.module)
        if requested_module and requested_module != preferred_leaf:
            return (
                f"This task is documented through the {preferred_leaf} flow within Oracle Fusion "
                f"{requested_module or route_info.module_family.value}."
            )
        return None

    def _build_structured_doc_response(
        self,
        route_info: Any,
        user_query: str,
        mapped_chunks: List[Dict[str, Any]],
    ) -> str | None:
        if route_info.task_type not in self.DOCS_EXPECTED_TASKS:
            return None

        doc_chunks = [
            chunk
            for chunk in mapped_chunks
            if (chunk.get("metadata") or {}).get("corpus") in self.DOC_GROUNDING_CORPORA
        ]
        if not doc_chunks:
            return None

        summary_concept_profile = self._summary_concept_profile(user_query, route_info)
        is_summary_concept_prompt = self._is_summary_concept_prompt(user_query)

        def extract_numbered_steps(text: str) -> List[str]:
            if not text:
                return []
            matches = re.findall(r"(?m)^\s*(?:\d+[\.\)]|-\s+)\s*(.+?)\s*$", text)
            return [item.strip() for item in matches if item.strip()]

        def extract_sentence_steps(text: str, *, limit: int = 5) -> List[str]:
            if not text:
                return []
            compact = re.sub(r"\s+", " ", text).strip()
            if not compact:
                return []
            sentence_parts = re.split(r"(?<=[\.\!\?])\s+", compact)
            cleaned: List[str] = []
            for part in sentence_parts:
                step = re.sub(r"^\W+|\W+$", "", part).strip()
                if not step:
                    continue
                if len(step.split()) < 4:
                    continue
                if step.lower().startswith(("oracle fusion", "chapter ", "section ")):
                    continue
                cleaned.append(step.rstrip("."))
                if len(cleaned) >= limit:
                    break
            return cleaned

        def structured_support_score(chunk: Dict[str, Any]) -> float:
            metadata = chunk.get("metadata") or {}
            score = float(chunk.get("combined_score") or chunk.get("score") or 0.0)
            if metadata.get("allow_deterministic_grounded_answer"):
                score += 4.0
            if route_info.task_type in {TaskType.PROCEDURE, TaskType.NAVIGATION}:
                if metadata.get("ordered_steps"):
                    score += 3.0
                if metadata.get("navigation_paths"):
                    score += 0.6
            elif route_info.task_type == TaskType.TROUBLESHOOTING:
                if metadata.get("root_causes"):
                    score += 1.8
                if metadata.get("resolution_steps"):
                    score += 2.2
                if metadata.get("symptom_terms"):
                    score += 1.2
            else:
                if metadata.get("summary_points"):
                    score += 2.0
                if metadata.get("purpose"):
                    score += 0.8
                concept_relevance = self._summary_concept_relevance(
                    user_query=user_query,
                    metadata=metadata,
                    title=str(metadata.get("title") or chunk.get("title") or ""),
                    content=str(chunk.get("content") or ""),
                    concept_profile=summary_concept_profile,
                )
                if summary_concept_profile:
                    score += concept_relevance * 7.4
                    if concept_relevance < float(summary_concept_profile.get("min_relevance") or 0.25):
                        score -= 5.5
                if is_summary_concept_prompt and self._is_summary_procedural_doc(metadata):
                    score -= 2.6
            return score

        top = max(doc_chunks, key=structured_support_score, default=None)
        if top is None:
            return None

        metadata = top.get("metadata") or {}
        citation = str(top.get("citation_id") or "[D1]")
        content = str(top.get("content") or "")
        task_name = str(metadata.get("task_name") or metadata.get("title") or top.get("title") or "").strip()
        purpose = str(metadata.get("purpose") or "").strip()
        prerequisites = self._coerce_string_list(metadata.get("prerequisites"))
        navigation_paths = self._coerce_string_list(metadata.get("navigation_paths"))
        ordered_steps = self._coerce_string_list(metadata.get("ordered_steps"))
        if not ordered_steps:
            ordered_steps = extract_numbered_steps(content)
        if not ordered_steps:
            ordered_steps = extract_sentence_steps(content, limit=6)
        summary_points = self._coerce_string_list(metadata.get("summary_points"))
        warnings = self._coerce_string_list(metadata.get("warnings"))
        symptom_terms = self._coerce_string_list(metadata.get("symptom_terms"))
        root_causes = self._coerce_string_list(metadata.get("root_causes"))
        resolution_steps = self._coerce_string_list(metadata.get("resolution_steps"))
        if not resolution_steps and route_info.task_type == TaskType.TROUBLESHOOTING:
            resolution_steps = extract_numbered_steps(content)
        if not resolution_steps and route_info.task_type == TaskType.TROUBLESHOOTING:
            resolution_steps = extract_sentence_steps(content, limit=5)
        note = self._preferred_leaf_note(route_info, metadata)

        if route_info.task_type in {TaskType.PROCEDURE, TaskType.NAVIGATION}:
            if not ordered_steps and not summary_points:
                if task_name or purpose:
                    ordered_steps = [
                        f"Open the Oracle Fusion flow for {task_name or 'this task'} as documented in {citation}.",
                        "Complete the required fields and task options captured in the retained grounding.",
                        "Save or submit the transaction and validate the resulting status.",
                    ]
            if not ordered_steps and not summary_points:
                return None
            lines = ["[Answer]"]
            if task_name:
                lines.append(f"Task: {task_name} {citation}")
            if note:
                lines.append(f"Scope note: {note} {citation}")
            if purpose:
                lines.append(f"Purpose: {purpose} {citation}")
            lines.append("")
            lines.append("Prerequisites:")
            if prerequisites:
                lines.extend(f"- {item}" for item in prerequisites)
            else:
                lines.append("- Not explicitly documented in the retained grounding.")
            if navigation_paths:
                lines.append("")
                lines.append("Navigation:")
                lines.extend(f"- {item}" for item in navigation_paths)
            lines.append("")
            lines.append("Ordered Steps:")
            if ordered_steps:
                lines.extend(f"{index}. {step}" for index, step in enumerate(ordered_steps, start=1))
            else:
                lines.extend(f"{index}. {step}" for index, step in enumerate(summary_points, start=1))
            lines.append("")
            lines.append("Notes / Constraints:")
            if warnings:
                lines.extend(f"- {item}" for item in warnings)
            else:
                lines.append("- Follow role and privilege constraints defined for your environment.")
            return "\n".join(lines).strip()

        if route_info.task_type == TaskType.TROUBLESHOOTING:
            if not root_causes and not resolution_steps and not summary_points and not purpose and not symptom_terms and not content:
                return None
            lines = ["[Troubleshooting]"]
            symptom_line = "; ".join(symptom_terms[:3]) if symptom_terms else (purpose or task_name)
            if not symptom_line:
                sentence_steps = extract_sentence_steps(content, limit=1)
                symptom_line = sentence_steps[0] if sentence_steps else "Issue observed in the retained Oracle Fusion flow."
            if symptom_line:
                lines.append(f"Symptom: {symptom_line} {citation}")
            if note:
                lines.append(f"Scope note: {note} {citation}")
            lines.append("")
            lines.append("Likely Causes:")
            if root_causes:
                lines.extend(f"- {item}" for item in root_causes)
            elif summary_points:
                lines.extend(f"- {item}" for item in summary_points[:3])
            else:
                lines.append("- The retained grounding doesn't list explicit causes for this symptom.")
            lines.append("")
            lines.append("Resolution Steps:")
            if resolution_steps:
                lines.extend(f"{index}. {step}" for index, step in enumerate(resolution_steps, start=1))
            elif ordered_steps:
                lines.extend(f"{index}. {step}" for index, step in enumerate(ordered_steps, start=1))
            elif summary_points:
                lines.extend(f"{index}. {step}" for index, step in enumerate(summary_points, start=1))
            else:
                lines.append("1. Validate setup, required inputs, and role privileges for this flow.")
                lines.append("2. Re-run the process and review the latest error details.")
            lines.append("")
            lines.append("Notes:")
            if warnings:
                lines.extend(f"- {item}" for item in warnings)
            else:
                lines.append("- Use the cited Oracle flow to confirm the exact task path before retrying.")
            return "\n".join(lines).strip()

        if route_info.task_type in {TaskType.GENERAL, TaskType.SUMMARY, TaskType.INTEGRATION}:
            summary_safety = self._summary_safety_assessment(
                user_query=user_query,
                route_info=route_info,
                metadata=metadata,
                title=task_name or str(top.get("title") or ""),
                content=content,
            )
            if not summary_safety.get("strong_match"):
                return None
            key_points = summary_points or ordered_steps[:4]
            if not key_points:
                key_points = extract_sentence_steps(content, limit=4)
            if not purpose:
                purpose = task_name
            if not purpose and not key_points:
                return None
            lines = ["[Answer]"]
            definition = purpose or task_name
            if definition:
                lines.append(f"Definition: {definition} {citation}")
            if note:
                lines.append(f"Scope note: {note} {citation}")
            if key_points:
                lines.append("")
                lines.append("Key Points:")
                lines.extend(f"- {item}" for item in key_points)
            if warnings:
                lines.append("")
                lines.append("Notes:")
                lines.extend(f"- {item}" for item in warnings)
            return "\n".join(lines).strip()

        return None

    def _sort_doc_candidates(
        self,
        user_query: str,
        route_info: Any,
        candidates: List[Dict[str, Any]],
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        query_tokens = self._specialized_query_tokens(user_query)
        requested_tasks = {
            str(signal.get("task") or "").lower()
            for signal in TaskSemanticAnalyzer.extract_query_signals(user_query).get("signals", [])
            if str(signal.get("task") or "").strip()
        }
        explicit_module = bool(getattr(route_info, "module_explicit", False))
        requested_module = self._canonical_module_name(route_info.module)
        issue_tokens = {
            token
            for token in query_tokens
            if token not in self.TROUBLESHOOTING_STOPWORDS
        }
        summary_concept_profile = self._summary_concept_profile(user_query, route_info)
        is_summary_concept_prompt = self._is_summary_concept_prompt(user_query)

        scored: List[Dict[str, Any]] = []
        for chunk in candidates:
            metadata = chunk.get("metadata") or {}
            corpus = str(metadata.get("corpus") or "")
            title = str(metadata.get("title") or "")
            source_hint = str(metadata.get("source_uri") or metadata.get("source_path") or "")
            content_hint = str(chunk.get("content") or "")[:1600]
            lowered_title = title.lower()
            lowered_source = source_hint.lower()
            lowered_content = content_hint.lower()
            symptom_terms = {
                token.lower()
                for token in (metadata.get("symptom_terms") or [])
                if str(token).strip()
            }
            error_keywords = {
                token.lower()
                for token in (metadata.get("error_keywords") or [])
                if str(token).strip()
            }
            task_signals = {
                str(token).lower().strip()
                for token in (metadata.get("task_signals") or [])
                if str(token).strip()
            }
            task_name = str(metadata.get("task_name") or "").lower().strip()
            query_aliases = self._coerce_string_list(metadata.get("query_aliases"))
            deterministic_grounded = bool(metadata.get("allow_deterministic_grounded_answer"))
            alias_tokens = self._specialized_query_tokens(
                self._metadata_text(metadata, ["task_name", "query_aliases", "preferred_leaf_module"])
            )
            structured_tokens = self._specialized_query_tokens(
                self._metadata_text(
                    metadata,
                    [
                        "ordered_steps",
                        "summary_points",
                        "prerequisites",
                        "warnings",
                        "purpose",
                        "navigation_paths",
                        "root_causes",
                        "resolution_steps",
                    ],
                )
            )

            title_tokens = self._specialized_query_tokens(f"{title} {source_hint}")
            content_tokens = self._specialized_query_tokens(content_hint)
            overlap_title = len(query_tokens & title_tokens) / max(len(query_tokens), 1)
            overlap_content = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
            prior_score = float(chunk.get("combined_score") or chunk.get("score") or 0.0)
            quality_score = float(metadata.get("quality_score") or 0.0)
            final_rank = (
                prior_score * 2.25
                + quality_score * 0.55
                + overlap_title * 2.9
                + overlap_content * 1.35
                + self._doc_corpus_boost(corpus, route_info, metadata)
            )

            alias_overlap = self._specialized_overlap_score(query_tokens, alias_tokens)
            structured_overlap = self._specialized_overlap_score(query_tokens, structured_tokens)
            final_rank += alias_overlap * 4.4 + structured_overlap * 1.35

            summary_concept_relevance = self._summary_concept_relevance(
                user_query=user_query,
                metadata=metadata,
                title=title,
                content=content_hint,
                concept_profile=summary_concept_profile,
            )
            if route_info.task_type in self.SUMMARY_TASK_TYPES and summary_concept_profile:
                final_rank += summary_concept_relevance * 8.2
                if summary_concept_relevance < float(summary_concept_profile.get("min_relevance") or 0.25):
                    final_rank -= 6.2
            if (
                route_info.task_type in self.SUMMARY_TASK_TYPES
                and is_summary_concept_prompt
                and self._is_summary_procedural_doc(metadata)
                and summary_concept_relevance < 0.72
            ):
                final_rank -= 4.8
            if route_info.task_type in self.SUMMARY_TASK_TYPES:
                summary_safety = self._summary_safety_assessment(
                    user_query=user_query,
                    route_info=route_info,
                    metadata=metadata,
                    title=title,
                    content=content_hint,
                )
                if not summary_safety.get("strong_match"):
                    continue
                final_rank += float(summary_safety.get("score") or 0.0) * 4.8

            if task_name and task_name in user_query.lower():
                final_rank += 2.35
            alias_match = any(alias.lower() in user_query.lower() for alias in query_aliases if len(alias) > 4)
            if alias_match:
                final_rank += 1.95
            if metadata.get("grounding_answerability_pass"):
                final_rank += 0.85
            if route_info.task_type == TaskType.PROCEDURE and metadata.get("ordered_steps"):
                final_rank += 1.15
            if route_info.task_type in {TaskType.GENERAL, TaskType.SUMMARY} and metadata.get("summary_points"):
                final_rank += 0.95

            actual_module = self._canonical_module_name(metadata.get("module"))
            exact_module_match = bool(requested_module and actual_module == requested_module)
            exact_task_match = bool((task_name and task_name in user_query.lower()) or alias_match)
            structured_exact_match = bool(
                deterministic_grounded
                and exact_task_match
                and (not explicit_module or exact_module_match)
            )

            if deterministic_grounded and exact_task_match:
                final_rank += 4.8
            if structured_exact_match and metadata.get("ordered_steps"):
                final_rank += 7.0
            if structured_exact_match and metadata.get("summary_points"):
                final_rank += 5.2
            if structured_exact_match and route_info.task_type == TaskType.TROUBLESHOOTING:
                final_rank += 6.2

            if explicit_module and requested_module and corpus in self.DOC_GROUNDING_CORPORA:
                if actual_module == requested_module:
                    final_rank += 2.4
                elif actual_module:
                    final_rank -= 0.95

            if route_info.task_type == TaskType.TROUBLESHOOTING:
                symptom_title_overlap = self._specialized_overlap_score(issue_tokens, title_tokens)
                symptom_content_overlap = self._specialized_overlap_score(issue_tokens, content_tokens)
                symptom_metadata_overlap = self._specialized_overlap_score(issue_tokens, symptom_terms)
                error_keyword_overlap = self._specialized_overlap_score(query_tokens, error_keywords)
                task_signal_overlap = self._specialized_overlap_score(requested_tasks, task_signals)
                final_rank += symptom_title_overlap * 3.3 + symptom_content_overlap * 1.7
                final_rank += symptom_metadata_overlap * 4.2 + error_keyword_overlap * 2.6
                final_rank += task_signal_overlap * 4.4
                if any(hint in lowered_title or hint in lowered_source for hint in self.TROUBLESHOOTING_HINTS):
                    final_rank += 1.0
                if any(hint in lowered_content for hint in self.TROUBLESHOOTING_HINTS):
                    final_rank += 0.45
                if corpus == "troubleshooting_corpus":
                    final_rank += 1.2
            elif route_info.task_type in self.DOCS_EXPECTED_TASKS:
                if lowered_title and lowered_title in user_query.lower():
                    final_rank += 1.5

            enriched = dict(chunk)
            enriched["rerank_score"] = final_rank
            enriched["final_rank_score"] = final_rank
            enriched["structured_exact_match"] = structured_exact_match
            scored.append(enriched)

        scored.sort(
            key=lambda item: (
                1 if item.get("structured_exact_match") else 0,
                item.get("final_rank_score", 0.0),
            ),
            reverse=True,
        )
        return scored[:top_k]

    def _residual_preferred_modules(self, task_name: str | None) -> List[str]:
        task_name = str(task_name or "").strip()
        if task_name not in self.RESIDUAL_TROUBLESHOOTING_TASKS:
            return []
        preferred = TaskSemanticAnalyzer.TASK_CONFIGS.get(task_name, {}).get("preferred_modules", [])
        return [self._canonical_module_name(module_name) for module_name in preferred if str(module_name).strip()]

    def _count_allowlisted_troubleshooting_support(
        self,
        chunks: List[Dict[str, Any]],
        allowed_modules: List[str],
    ) -> int:
        allowed = {self._canonical_module_name(module_name) for module_name in allowed_modules if module_name}
        if not allowed:
            return 0

        count = 0
        supported_corpora = {*(self.DOC_GROUNDING_CORPORA), "sql_corpus", "sql_examples_corpus"}
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            corpus = str(metadata.get("corpus") or "")
            if corpus not in supported_corpora:
                continue
            actual_module = self._canonical_module_name(metadata.get("module"))
            if actual_module not in allowed:
                continue
            if corpus in self.DOC_GROUNDING_CORPORA:
                count += 1
                continue
            if str(metadata.get("task_match_strength") or "") in {"medium", "strong"}:
                count += 1
        return count

    def _preferred_fallback_note(
        self,
        requested_module: str | FusionModule | None,
        actual_module: str | None,
    ) -> str | None:
        requested = self._canonical_module_name(requested_module)
        actual = self._canonical_module_name(actual_module)
        if not requested or not actual or requested == actual:
            return None
        return f"This troubleshooting flow is documented in {actual} rather than {requested}."

    def _inject_troubleshooting_note(self, content: str, note: str | None) -> str:
        if not content or not note:
            return content
        if note.lower() in content.lower():
            return content

        headings = ("[Answer]", "[Troubleshooting]", "[Troubleshooting Answer]")
        for heading in headings:
            if content.startswith(heading):
                remainder = content[len(heading):].lstrip("\n")
                return f"{heading}\n{note}\n\n{remainder}".rstrip()
        return f"{note}\n\n{content}".strip()

    async def _attempt_residual_troubleshooting_fallback(
        self,
        *,
        db: AsyncSession,
        tenant_id: str,
        user_query: str,
        route_info: Any,
        retrieval_plan: Any,
        task_config: Dict[str, Any],
        retrieval_filters: Dict[str, Any],
        reranked_chunks: List[Dict[str, Any]],
        task_gate: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], List[str]]:
        audit = {
            "preferred_module_troubleshooting_fallback_attempted": False,
            "preferred_module_troubleshooting_fallback_used": False,
            "preferred_module_troubleshooting_fallback_candidate_count": 0,
            "preferred_module_troubleshooting_fallback_module": None,
            "preferred_module_troubleshooting_correction": None,
        }
        if route_info.task_type != TaskType.TROUBLESHOOTING or not getattr(route_info, "module_explicit", False):
            return reranked_chunks, task_gate, audit, []

        task_name = str(task_gate.get("top_task_signal") or "").strip()
        preferred_modules = self._residual_preferred_modules(task_name)
        if not preferred_modules:
            return reranked_chunks, task_gate, audit, []

        exact_matches = int(task_gate.get("exact_strong_doc_count") or 0) + int(task_gate.get("exact_medium_doc_count") or 0)
        if task_gate.get("task_semantic_gate") != "FAILED" and exact_matches > 0:
            return reranked_chunks, task_gate, audit, preferred_modules

        audit["preferred_module_troubleshooting_fallback_attempted"] = True
        fallback_filters = dict(retrieval_filters)
        fallback_filters.pop("module", None)
        fallback_filters.pop("module_family", None)
        fallback_filters.pop("allow_same_family_fallback", None)
        fallback_filters["requested_module"] = preferred_modules
        fallback_filters["exact_module_allowlist"] = preferred_modules
        fallback_filters["strict_exact_module_only"] = True

        _, fallback_chunks = await self._retrieve_grounding_chunks(
            db=db,
            tenant_id=tenant_id,
            user_query=user_query,
            route_info=route_info,
            retrieval_plan=retrieval_plan,
            task_config=task_config,
            retrieval_filters=fallback_filters,
        )
        allowed = {self._canonical_module_name(module_name) for module_name in preferred_modules}
        fallback_chunks = [
            chunk
            for chunk in fallback_chunks
            if self._canonical_module_name((chunk.get("metadata") or {}).get("module")) in allowed
        ]
        audit["preferred_module_troubleshooting_fallback_candidate_count"] = len(fallback_chunks)
        if not fallback_chunks:
            return reranked_chunks, task_gate, audit, preferred_modules

        merged_chunks: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for chunk in fallback_chunks + reranked_chunks:
            chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
            if chunk_id and chunk_id in seen:
                continue
            if chunk_id:
                seen.add(chunk_id)
            merged_chunks.append(chunk)

        fallback_gate, fallback_ranked = self._annotate_task_semantics(
            merged_chunks,
            user_query,
            route_info,
            preferred_module_allowlist=preferred_modules,
        )
        support_count = self._count_allowlisted_troubleshooting_support(fallback_ranked, preferred_modules)
        if support_count <= 0:
            return reranked_chunks, task_gate, audit, preferred_modules

        top_module = next(
            (
                self._canonical_module_name((chunk.get("metadata") or {}).get("module"))
                for chunk in fallback_ranked
                if self._canonical_module_name((chunk.get("metadata") or {}).get("module")) in allowed
                and str((chunk.get("metadata") or {}).get("corpus") or "") in self.DOC_GROUNDING_CORPORA
            ),
            None,
        )
        correction_note = self._preferred_fallback_note(route_info.module, top_module)
        fallback_gate = dict(fallback_gate)
        fallback_gate["task_semantic_gate"] = "PASSED"
        fallback_gate["task_gate_reason"] = "preferred_module_troubleshooting_fallback"
        if correction_note:
            fallback_gate["module_correction_message"] = correction_note

        audit["preferred_module_troubleshooting_fallback_used"] = True
        audit["preferred_module_troubleshooting_fallback_module"] = top_module
        audit["preferred_module_troubleshooting_correction"] = correction_note
        return fallback_ranked, fallback_gate, audit, preferred_modules

    def _specialized_corpus_boost(self, corpus: str, route_info: Any, metadata: Dict[str, Any]) -> float:
        doc_type = str(metadata.get("doc_type") or "")
        derived_from_doc = bool(metadata.get("derived_from_doc"))
        derived_from_runtime = bool(metadata.get("derived_from_runtime"))
        if route_info.task_type in self.SQL_TASKS:
            if corpus == "sql_examples_corpus":
                if doc_type == "sql_example":
                    if derived_from_runtime:
                        return 2.45
                    if derived_from_doc:
                        return 2.55
                    return 2.85
                return 2.2
            if corpus == "schema_metadata_corpus":
                return 1.9
            if corpus == "docs_corpus":
                return 0.35
            if corpus == "schema_corpus":
                return 0.25
        if route_info.task_type in self.FAST_FORMULA_TASKS:
            if corpus == "fast_formula_corpus":
                if doc_type == "fast_formula_example":
                    return 2.7 if derived_from_doc else 3.0
                return 1.9
            if corpus == "docs_corpus":
                return 0.55
            if corpus == "schema_metadata_corpus":
                return 0.2
        return 0.0

    def _specialized_sort_candidates(
        self,
        user_query: str,
        route_info: Any,
        candidates: List[Dict[str, Any]],
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        query_tokens = self._specialized_query_tokens(user_query)
        query_identifiers = self._specialized_query_identifiers(user_query)
        request_shape = self._parse_sql_request_shape(user_query, route_info) if route_info.task_type in self.SQL_TASKS else None
        formula_shape = self._parse_fast_formula_request_shape(user_query, route_info) if route_info.task_type in self.FAST_FORMULA_TASKS else None
        broken_formula_normalized = self._normalize_formula_block_text(
            str((formula_shape or {}).get("broken_formula") or "")
        )
        scored: List[Dict[str, Any]] = []
        for chunk in candidates:
            metadata = chunk.get("metadata") or {}
            corpus = str(metadata.get("corpus") or "")
            title = str(metadata.get("title") or "")
            source_hint = str(metadata.get("source_uri") or metadata.get("source_path") or "")
            content_hint = str(chunk.get("content") or "")[:800]
            lowered_title = title.lower()
            lowered_source = source_hint.lower()

            title_tokens = self._specialized_query_tokens(f"{title} {source_hint}")
            content_tokens = self._specialized_query_tokens(content_hint)
            overlap_title = len(query_tokens & title_tokens) / max(len(query_tokens), 1)
            overlap_content = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
            prior_score = float(chunk.get("combined_score") or chunk.get("score") or 0.0)
            quality_score = float(metadata.get("quality_score") or 0.0)
            corpus_boost = self._specialized_corpus_boost(corpus, route_info, metadata)

            final_rank = (
                prior_score * 2.4
                + quality_score * 0.6
                + overlap_title * 3.0
                + overlap_content * 1.2
                + corpus_boost
            )
            if lowered_title and lowered_title in user_query.lower():
                final_rank += 12.0
            if lowered_source and lowered_source in user_query.lower():
                final_rank += 8.0

            if route_info.task_type in self.SQL_TASKS:
                table_overlap = self._specialized_overlap_score(
                    query_identifiers,
                    {str(item).upper() for item in (metadata.get("tables_used") or [])},
                )
                column_overlap = self._specialized_overlap_score(
                    query_identifiers,
                    {str(item).upper() for item in (metadata.get("columns_used") or [])},
                )
                join_overlap = self._specialized_overlap_score(
                    query_identifiers,
                    {str(item).upper() for item in (metadata.get("joins_used") or [])},
                )
                final_rank += table_overlap * 2.0 + column_overlap * 1.0 + join_overlap * 0.8
                if len(metadata.get("tables_used") or []) > 1 and (
                    " join " in f" {user_query.lower()} " or len(query_identifiers) >= 2
                ):
                    final_rank += 0.7
                final_rank += self._score_sql_pattern_for_request(
                    str(chunk.get("content") or ""),
                    metadata,
                    request_shape,
                )
                if str(metadata.get("doc_type") or "") == "sql_example":
                    if bool(metadata.get("derived_from_runtime")):
                        final_rank += 0.45
                    elif bool(metadata.get("derived_from_doc")):
                        final_rank += 0.6
                    else:
                        final_rank += 0.9

            if route_info.task_type in self.FAST_FORMULA_TASKS:
                formula_type_tokens = self._specialized_query_tokens(str(metadata.get("formula_type") or ""))
                formula_name_tokens = self._specialized_query_tokens(str(metadata.get("formula_name") or ""))
                use_case_tokens = self._specialized_query_tokens(str(metadata.get("use_case") or title))
                dbi_overlap = self._specialized_overlap_score(
                    query_identifiers,
                    {str(item).upper() for item in (metadata.get("database_items") or [])},
                )
                context_overlap = self._specialized_overlap_score(
                    query_identifiers,
                    {str(item).upper() for item in (metadata.get("contexts") or [])},
                )
                function_overlap = self._specialized_overlap_score(
                    query_identifiers,
                    {str(item).upper() for item in (metadata.get("functions") or [])},
                )
                final_rank += self._specialized_overlap_score(query_tokens, formula_type_tokens) * 3.0
                final_rank += self._specialized_overlap_score(query_tokens, formula_name_tokens) * 2.6
                final_rank += self._specialized_overlap_score(query_tokens, use_case_tokens) * 1.9
                final_rank += dbi_overlap * 2.8 + context_overlap * 2.4 + function_overlap * 1.8
                if str(metadata.get("doc_type") or "") == "fast_formula_example":
                    final_rank += 0.8 if not bool(metadata.get("derived_from_doc")) else 0.45
                    if route_info.task_type == TaskType.FAST_FORMULA_TROUBLESHOOTING:
                        final_rank += 0.65
                else:
                    final_rank -= 1.2
                if formula_shape:
                    requested_type = self._normalize_formula_type(str(formula_shape.get("requested_formula_type") or "UNKNOWN"))
                    candidate_type = self._normalize_formula_type(str(metadata.get("formula_type") or "UNKNOWN"))
                    if requested_type.upper() != "UNKNOWN":
                        if candidate_type == requested_type:
                            final_rank += 5.0
                        elif requested_type.lower() in candidate_type.lower() or candidate_type.lower() in requested_type.lower():
                            final_rank += 2.5
                        else:
                            final_rank -= 3.2
                    broken_tokens = set(formula_shape.get("broken_formula_tokens") or [])
                    if broken_tokens:
                        final_rank += self._specialized_overlap_score(broken_tokens, content_tokens) * 3.4
                    if formula_shape.get("is_troubleshooting") and broken_formula_normalized:
                        content_normalized = self._normalize_formula_block_text(str(chunk.get("content") or "")[:2500])
                        if broken_formula_normalized and content_normalized:
                            if broken_formula_normalized in content_normalized:
                                final_rank += 10.0
                            else:
                                broken_overlap = self._specialized_overlap_score(
                                    set(broken_formula_normalized.split()),
                                    set(content_normalized.split()),
                                )
                                final_rank += broken_overlap * 2.0
                    requested_formula_name = str(formula_shape.get("formula_name_normalized") or "")
                    candidate_formula_name = self._normalize_formula_name(
                        str(metadata.get("formula_name") or metadata.get("title") or "")
                    )
                    candidate_source = str(metadata.get("source_uri") or metadata.get("source_file") or "").lower()
                    if requested_formula_name:
                        if candidate_formula_name == requested_formula_name:
                            final_rank += 8.0
                        elif requested_formula_name in candidate_formula_name or candidate_formula_name in requested_formula_name:
                            final_rank += 4.5
                        elif "formula_types.json" in candidate_source:
                            final_rank -= 4.0
                    if formula_shape.get("is_troubleshooting") and "return" in content_hint.lower():
                        final_rank += 0.7

            enriched = dict(chunk)
            enriched["rerank_score"] = final_rank
            enriched["final_rank_score"] = final_rank
            scored.append(enriched)

        scored.sort(key=lambda item: item.get("final_rank_score", 0.0), reverse=True)
        return scored[:top_k]

    def _build_task_semantic_failure_message(self, gate: Dict[str, Any], route_info: Any) -> str:
        correction = gate.get("module_correction_message")
        if correction:
            return f"[Answer]\n{correction}\n\n{FAIL_CLOSED_MESSAGE}"

        top_task = gate.get("top_task_signal")
        module_name = self._canonical_module_name(route_info.module) or route_info.module_family.value
        if top_task:
            return (
                "[Answer]\n"
                f"The retained {module_name} grounding does not explicitly cover the requested task \"{top_task}\".\n\n"
                f"{FAIL_CLOSED_MESSAGE}"
            )
        return FAIL_CLOSED_MESSAGE

    async def _retrieve_grounding_chunks(
        self,
        db: AsyncSession,
        tenant_id: str,
        user_query: str,
        route_info: Any,
        retrieval_plan: Any,
        task_config: Dict[str, Any],
        retrieval_filters: Dict[str, Any],
    ) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        search_query = self._expand_summary_query(user_query, route_info)
        candidates = await self.search_engine.search(
            db,
            tenant_id,
            search_query,
            limit=retrieval_plan.budget.candidate_limit,
            fts_limit=retrieval_plan.budget.candidate_limit,
            vector_limit=retrieval_plan.budget.candidate_limit,
            fts_weight=task_config["fts_weight"],
            vector_weight=task_config["vector_weight"],
            filters=retrieval_filters,
        )
        graph_candidates = await self._expand_with_graph_snippets(
            db=db,
            tenant_id=tenant_id,
            route_info=route_info,
            base_chunks=candidates,
            retrieval_filters=retrieval_filters,
            retrieval_plan=retrieval_plan,
        )

        all_candidates: Dict[str, Dict[str, Any]] = {}
        for chunk in candidates + graph_candidates:
            chunk_id = str(chunk.get("id") or chunk.get("chunk_id"))
            if chunk_id and chunk_id not in all_candidates:
                all_candidates[chunk_id] = chunk

        top_k = min(RETRIEVAL_CONFIG["rerank_top"], retrieval_plan.budget.candidate_limit)
        if route_info.task_type in (self.SQL_TASKS | self.FAST_FORMULA_TASKS):
            reranked_chunks = self._specialized_sort_candidates(
                user_query,
                route_info,
                list(all_candidates.values()),
                top_k=top_k,
            )
        elif route_info.task_type in self.DOCS_EXPECTED_TASKS:
            reranked_chunks = self._sort_doc_candidates(
                user_query,
                route_info,
                list(all_candidates.values()),
                top_k=top_k,
            )
        else:
            if self.reranker is not None:
                reranked_chunks = self.reranker.rerank(
                    user_query,
                    list(all_candidates.values()),
                    top_k=top_k,
                )
            else:
                reranked_chunks = sorted(
                    list(all_candidates.values()),
                    key=lambda chunk: float(chunk.get("combined_score") or chunk.get("score") or 0.0),
                    reverse=True,
                )[:top_k]
        return all_candidates, reranked_chunks

    async def _disambiguate_route(
        self,
        db: AsyncSession,
        tenant_id: str,
        user_query: str,
        route_info: Any,
        retrieval_plan: Any,
        task_config: Dict[str, Any],
    ) -> Any:
        if not getattr(route_info, "disambiguation_required", False):
            return route_info

        if (
            route_info.task_type == TaskType.TROUBLESHOOTING
            and not getattr(route_info, "module_explicit", False)
        ):
            query_profile = TaskSemanticAnalyzer.extract_query_signals(user_query)
            if (
                query_profile.get("top_task") in self.RESIDUAL_TROUBLESHOOTING_TASKS
                and route_info.module_family != ModuleFamily.UNKNOWN
            ):
                return route_info.model_copy(
                    update={
                        "reasoning": f"{route_info.reasoning} Residual troubleshooting task kept family-level routing to avoid premature leaf-module collapse.",
                    }
                )

        candidate_modules = []
        for candidate in route_info.module_candidates:
            try:
                candidate_modules.append(FusionModule(candidate))
            except ValueError:
                continue
        if not candidate_modules:
            return route_info

        candidate_scores: List[tuple[float, FusionModule]] = []
        for candidate in candidate_modules[:4]:
            family = next(iter(module_families_for_value(candidate.value)), ModuleFamily.UNKNOWN.value)
            probe_filters = {
                "task_type": route_info.task_type.value,
                "corpora": retrieval_plan.corpora,
                "quality_score_min": retrieval_plan.budget.quality_score_min,
                "schema_limit": retrieval_plan.budget.max_schema_objects,
                "module_family": family,
                "requested_module": candidate.value,
            }
            hits = await self.search_engine.search(
                db,
                tenant_id,
                user_query,
                limit=min(4, retrieval_plan.budget.candidate_limit),
                fts_limit=min(4, retrieval_plan.budget.candidate_limit),
                vector_limit=min(4, retrieval_plan.budget.candidate_limit),
                fts_weight=task_config["fts_weight"],
                vector_weight=task_config["vector_weight"],
                filters=probe_filters,
            )
            evidence_score = sum(float(hit.get("combined_score") or hit.get("score") or 0.0) for hit in hits[:3])
            candidate_scores.append((evidence_score, candidate))

        candidate_scores.sort(key=lambda item: item[0], reverse=True)
        if not candidate_scores or candidate_scores[0][0] <= 0.0:
            return route_info

        best_score, best_module = candidate_scores[0]
        second_score = candidate_scores[1][0] if len(candidate_scores) > 1 else 0.0
        if best_score < 0.25 or (second_score and best_score < second_score * 1.1):
            return route_info.model_copy(
                update={
                    "module_confidence": min(float(getattr(route_info, "module_confidence", route_info.confidence) or 0.0), 0.72),
                    "reasoning": f"{route_info.reasoning} Retrieval evidence was not decisive for exact-module selection.",
                }
            )

        best_family = next(iter(module_families_for_value(best_module.value)), ModuleFamily.UNKNOWN.value)
        return route_info.model_copy(
            update={
                "module": best_module,
                "module_family": ModuleFamily(best_family),
                "confidence": min(max(route_info.confidence, 0.8), 0.95),
                "module_confidence": min(max(float(getattr(route_info, "module_confidence", route_info.confidence) or 0.0), 0.82), 0.95),
                "disambiguation_required": False,
                "reasoning": f"{route_info.reasoning} Retrieval evidence selected {best_module.value}.",
            }
        )

    async def _expand_with_graph_snippets(
        self,
        db: AsyncSession,
        tenant_id: str,
        route_info: Any,
        base_chunks: List[Dict[str, Any]],
        retrieval_filters: Dict[str, Any],
        retrieval_plan: Any,
    ) -> List[Dict[str, Any]]:
        if not base_chunks or not retrieval_plan.budget.allow_graph_expansion:
            return []

        detected_tables = set()
        for chunk in base_chunks:
            detected_tables.update(re.findall(r"\b[A-Z][A-Z0-9_]{3,}\b", chunk.get("content", "")))

        related_tables = set()
        for table_name in detected_tables:
            related_tables.update(fusion_graph.get_neighbors(table_name, depth=1))

        if not related_tables:
            return []

        graph_query = " ".join(sorted(related_tables)[:5])
        graph_filters = dict(retrieval_filters)
        graph_filters["corpora"] = ["schema_corpus"]
        graph_filters["schema_limit"] = retrieval_plan.budget.max_schema_objects

        graph_candidates = await self.search_engine.search(
            db,
            tenant_id,
            graph_query,
            limit=retrieval_plan.budget.max_schema_objects,
            fts_limit=retrieval_plan.budget.max_schema_objects,
            vector_limit=retrieval_plan.budget.max_schema_objects,
            fts_weight=0.0,
            vector_weight=1.0,
            filters=graph_filters,
        )

        graph_snippets = []
        for chunk in graph_candidates:
            content = chunk.get("content", "")
            if any(table_name in content for table_name in related_tables):
                graph_snippets.append(chunk)
        return graph_snippets

    def _sanitize_output(self, content: str) -> str:
        cleaned = content or ""
        cleaned = cleaned.replace("<|eot_id|>", "").replace("<|start_header_id|>", "").replace("<|end_header_id|>", "")
        cleaned = re.sub(r"\[HIDDEN_REASONING_CHAIN\].*?(?=\n\[|$)", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"\[TERNARY_LOGIC\].*?(?=\n\[|$)", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"(?im)^phase\s+\d+.*$", "", cleaned)
        cleaned = re.sub(r"(?im)^audit.*$", "", cleaned)
        cleaned = re.sub(r"(?im)^verification_status.*$", "", cleaned)
        cleaned = re.sub(r"(?im)^normalization_tags.*$", "", cleaned)
        cleaned = re.sub(
            r"(?im)^these steps are based on the provided documentation.*$",
            "",
            cleaned,
        )
        cleaned = re.sub(
            r"(?im)^note:\s+the above steps are grounded in the provided documentation.*$",
            "",
            cleaned,
        )
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _preserve_grounding_chunks(self, chunks: List[Dict[str, Any]], retrieval_plan: Any) -> List[Dict[str, Any]]:
        selected = retrieval_plan.apply_result_budget(chunks)
        source_chunks = selected if selected else chunks
        if not source_chunks:
            return []

        fallback_chunks: List[Dict[str, Any]] = []
        total_chars = 0
        for chunk in source_chunks:
            if len(fallback_chunks) >= retrieval_plan.budget.max_chunks:
                break

            content = (chunk.get("content") or "").strip()
            if not content:
                metadata = chunk.get("metadata") or {}
                content = (
                    metadata.get("snippet")
                    or metadata.get("summary")
                    or metadata.get("title")
                    or metadata.get("task_name")
                    or ""
                ).strip()
            if not content:
                continue

            remaining_chars = retrieval_plan.budget.max_total_chars - total_chars
            if remaining_chars <= 0:
                break

            trimmed = dict(chunk)
            trimmed["content"] = content[:remaining_chars].strip()
            if not trimmed["content"]:
                continue

            fallback_chunks.append(trimmed)
            total_chars += len(trimmed["content"])

        return fallback_chunks

    def _build_failure_response(
        self,
        trace_id: str,
        timings: Dict[str, float],
        audit: Dict[str, Any],
        citations: List[Dict[str, Any]],
        retrieved_chunks: List[Dict[str, Any]],
        message: str = FAIL_CLOSED_MESSAGE,
    ) -> ChatResponse:
        return ChatResponse(
            id=trace_id,
            created=int(time.time()),
            model="antigravity-v6",
            choices=[{"message": {"role": "assistant", "content": message}}],
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            timings=timings,
            audit=audit,
        )

    def _grounding_availability_score(
        self,
        *,
        task_gate: Dict[str, Any],
        citation_count: int,
        docs_count: int,
        exact_support_available: bool,
    ) -> float:
        if exact_support_available:
            return 1.0
        exact_medium = float(task_gate.get("exact_medium_doc_count") or 0)
        strong = float(task_gate.get("strong_doc_count") or 0)
        medium = float(task_gate.get("medium_doc_count") or 0)
        if exact_medium > 0:
            return 0.78
        if strong > 0:
            return 0.64
        if medium > 0:
            return 0.48
        if docs_count > 0 or citation_count > 0:
            return 0.32
        return 0.0

    def _grounding_signal_flags(
        self,
        *,
        route_info: Any,
        task_gate: Dict[str, Any],
        citation_count: int,
        docs_count: int,
        exact_support_available: bool,
    ) -> Dict[str, bool]:
        strong_doc_signal = bool(task_gate.get("exact_strong_doc_count") or task_gate.get("strong_doc_count"))
        medium_doc_signal = bool(task_gate.get("exact_medium_doc_count") or task_gate.get("medium_doc_count"))
        if route_info.task_type in self.DOCS_EXPECTED_TASKS:
            retrieval_signal = docs_count > 0
        else:
            retrieval_signal = citation_count > 0 or docs_count > 0
        troubleshooting_signal = (
            route_info.task_type == TaskType.TROUBLESHOOTING
            and retrieval_signal
        )
        any_grounding_signal = bool(
            exact_support_available
            or strong_doc_signal
            or medium_doc_signal
            or retrieval_signal
            or float(task_gate.get("grounding_evidence_score") or 0.0) >= 0.32
        )
        sufficient_grounding_signal = bool(
            exact_support_available
            or strong_doc_signal
            or float(task_gate.get("grounding_evidence_score") or 0.0) >= 0.45
            or troubleshooting_signal
            or (citation_count > 0 and docs_count > 0)
        )
        return {
            "any_grounding_signal": any_grounding_signal,
            "sufficient_grounding_signal": sufficient_grounding_signal,
            "strong_doc_signal": strong_doc_signal,
            "medium_doc_signal": medium_doc_signal,
            "retrieval_signal": retrieval_signal,
            "troubleshooting_signal": troubleshooting_signal,
        }

    def _build_decision_trace(
        self,
        *,
        route_info: Any,
        task_gate: Dict[str, Any],
        citation_count: int,
        docs_count: int,
        exact_support_available: bool,
        strict_financial_leaf: bool,
        preferred_fallback_audit: Dict[str, Any],
    ) -> Dict[str, Any]:
        intent_confidence = float(
            getattr(route_info, "intent_confidence", 0.0)
            or task_gate.get("top_task_confidence")
            or 0.0
        )
        module_confidence = float(getattr(route_info, "module_confidence", 0.0) or getattr(route_info, "confidence", 0.0) or 0.0)

        if route_info.task_type in (self.SQL_TASKS | self.FAST_FORMULA_TASKS):
            has_signal = citation_count > 0 or docs_count > 0
            return {
                "intent_classification": route_info.task_type.value,
                "intent_confidence": round(intent_confidence, 4),
                "module_confidence": round(module_confidence, 4),
                "grounding_availability_score": round(1.0 if citation_count > 0 else 0.0, 4),
                "decision_grounding_signal_present": has_signal,
                "decision_sufficient_grounding_signal": has_signal,
                "grounding_confidence_tier": "high" if citation_count > 0 else "medium",
                "decision_confidence_tier": "HIGH" if citation_count > 0 else "MEDIUM",
                "decision_execution_mode": "EXECUTE",
                "decision_reason": "specialized_lane_execution",
                "decision_refusal_reason": None,
            }

        grounding_score = self._grounding_availability_score(
            task_gate=task_gate,
            citation_count=citation_count,
            docs_count=docs_count,
            exact_support_available=exact_support_available,
        )
        grounding_flags = self._grounding_signal_flags(
            route_info=route_info,
            task_gate=task_gate,
            citation_count=citation_count,
            docs_count=docs_count,
            exact_support_available=exact_support_available,
        )
        strong_module_conflict = bool(task_gate.get("module_conflict"))
        fallback_conflict = bool(preferred_fallback_audit.get("preferred_module_troubleshooting_correction"))
        preferred_fallback_used = bool(preferred_fallback_audit.get("preferred_module_troubleshooting_fallback_used"))
        task_gate_failed = str(task_gate.get("task_semantic_gate") or "") == "FAILED"
        task_gate_reason = str(task_gate.get("task_gate_reason") or "")
        sibling_only_match = task_gate_reason == "sibling_module_only_match"
        docs_expected = route_info.task_type in self.DOCS_EXPECTED_TASKS
        usable_doc_grounding = bool(
            exact_support_available
            or grounding_flags["strong_doc_signal"]
            or grounding_flags["medium_doc_signal"]
            or docs_count > 0
            or (
                route_info.task_type in {TaskType.GENERAL, TaskType.SUMMARY, TaskType.INTEGRATION}
                and citation_count > 0
            )
        )

        explicit_module = bool(getattr(route_info, "module_explicit", False))
        exact_module_doc_count = int(task_gate.get("exact_doc_count") or 0)

        decision_reason = "grounded_execution"
        refusal_reason = None
        execution_mode = "EXECUTE"

        broad_module_explicit = self._canonical_module_name(route_info.module).lower() in {
            ModuleFamily.FINANCIALS.value.lower(),
            ModuleFamily.PROCUREMENT.value.lower(),
            ModuleFamily.HCM.value.lower(),
            ModuleFamily.SCM.value.lower(),
            ModuleFamily.PROJECTS.value.lower(),
        }

        if fallback_conflict:
            execution_mode = "REFUSE"
            refusal_reason = "preferred_module_conflict"
            decision_reason = "preferred_module_conflict"
        elif sibling_only_match and not preferred_fallback_used:
            if docs_expected and grounding_flags["any_grounding_signal"] and (
                task_gate.get("module_correction_message") or broad_module_explicit
            ):
                decision_reason = "module_correction_grounded_execution"
            else:
                execution_mode = "REFUSE"
                refusal_reason = "sibling_module_only_match"
                decision_reason = "sibling_module_only_match"
        elif strong_module_conflict and getattr(route_info, "module_explicit", False):
            execution_mode = "REFUSE"
            refusal_reason = "strong_task_module_conflict"
            decision_reason = "strong_task_module_conflict"
        elif (
            docs_expected
            and explicit_module
            and exact_module_doc_count <= 0
            and not exact_support_available
            and not task_gate.get("module_correction_message")
            and not preferred_fallback_used
            and not grounding_flags["any_grounding_signal"]
        ):
            execution_mode = "REFUSE"
            refusal_reason = "no_exact_module_grounding"
            decision_reason = "no_exact_module_grounding"
        elif docs_expected and usable_doc_grounding and not strong_module_conflict:
            if grounding_score >= 0.8 and module_confidence >= 0.6 and intent_confidence >= 0.55:
                decision_reason = "high_confidence_grounded_execution"
            else:
                decision_reason = "medium_confidence_grounded_execution"
        elif (
            strict_financial_leaf
            and docs_expected
            and not exact_support_available
            and not usable_doc_grounding
        ):
            execution_mode = "REFUSE"
            refusal_reason = "no_exact_module_grounding"
            decision_reason = "no_exact_module_grounding"
        elif (
            task_gate_failed
            and docs_expected
            and not usable_doc_grounding
        ):
            execution_mode = "REFUSE"
            refusal_reason = "task_semantic_gate_failed"
            decision_reason = "task_semantic_gate_failed"
        elif grounding_score >= 0.8 and module_confidence >= 0.6 and intent_confidence >= 0.55:
            decision_reason = "high_confidence_grounded_execution"
        elif (
            grounding_flags["sufficient_grounding_signal"]
            and module_confidence >= 0.2
            and not strong_module_conflict
        ):
            decision_reason = "medium_confidence_grounded_execution"
        elif grounding_score >= 0.45 and module_confidence >= 0.45 and intent_confidence >= 0.35:
            decision_reason = "medium_confidence_grounded_execution"
        else:
            execution_mode = "REFUSE"
            refusal_reason = "low_grounding_confidence"
            decision_reason = "low_grounding_confidence"

        if execution_mode == "REFUSE":
            decision_tier = "LOW"
        elif decision_reason.startswith("high_"):
            decision_tier = "HIGH"
        else:
            decision_tier = "MEDIUM"

        return {
            "intent_classification": route_info.task_type.value,
            "intent_confidence": round(intent_confidence, 4),
            "module_confidence": round(module_confidence, 4),
            "grounding_availability_score": round(grounding_score, 4),
            "decision_grounding_signal_present": grounding_flags["any_grounding_signal"],
            "decision_sufficient_grounding_signal": grounding_flags["sufficient_grounding_signal"],
            "grounding_confidence_tier": task_gate.get("grounding_confidence_tier", "low"),
            "decision_confidence_tier": decision_tier,
            "decision_execution_mode": execution_mode,
            "decision_reason": decision_reason,
            "decision_refusal_reason": refusal_reason,
        }

    def _load_specialization_catalog(self) -> Dict[str, Any]:
        if self.__class__._specialization_catalog is None:
            if self.SPECIALIZATION_SUMMARY_PATH.exists():
                self.__class__._specialization_catalog = json.loads(
                    self.SPECIALIZATION_SUMMARY_PATH.read_text(encoding="utf-8")
                )
            else:
                self.__class__._specialization_catalog = {}
        return self.__class__._specialization_catalog or {}

    def _load_local_doc_catalog(self) -> List[Dict[str, Any]]:
        if self.__class__._local_doc_catalog is not None:
            return self.__class__._local_doc_catalog

        dedupe: set[str] = set()
        records: List[Dict[str, Any]] = []
        for path in self.LOCAL_DOC_MANIFEST_PATHS:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        object_type = str(row.get("object_type") or "").strip().lower()
                        if object_type in {"procedure", "troubleshooting"}:
                            row = self._normalize_stage2_doc_object(row)
                            if not row:
                                continue
                        metadata = dict(row.get("metadata") or {})
                        content = str(row.get("content") or "").strip()
                        if not content:
                            continue

                        module = str(row.get("module") or metadata.get("module") or FusionModule.UNKNOWN.value).strip()
                        title = str(row.get("title") or metadata.get("title") or "").strip()
                        source_uri = str(
                            row.get("source_uri")
                            or row.get("source_path")
                            or metadata.get("source_uri")
                            or metadata.get("source_path")
                            or ""
                        ).strip()
                        key = "|".join(
                            [
                                module.lower(),
                                title.lower(),
                                str(metadata.get("task_name") or row.get("task_name") or "").lower(),
                                source_uri.lower(),
                            ]
                        )
                        if key in dedupe:
                            continue
                        dedupe.add(key)

                        record = dict(row)
                        record["module"] = module
                        record["title"] = title or "grounded-doc"
                        record["source_uri"] = source_uri
                        record["task_name"] = str(
                            row.get("task_name")
                            or metadata.get("task_name")
                            or ""
                        ).strip()
                        record["query_aliases"] = (
                            row.get("query_aliases")
                            or metadata.get("query_aliases")
                            or []
                        )
                        record["task_type"] = str(
                            row.get("task_type")
                            or metadata.get("task_type")
                            or "general"
                        ).strip().lower()
                        record["doc_type"] = str(
                            row.get("doc_type")
                            or metadata.get("doc_type")
                            or "functional_doc"
                        ).strip()
                        record["quality_score"] = float(
                            row.get("quality_score")
                            or metadata.get("quality_score")
                            or 0.8
                        )
                        record["metadata"] = metadata
                        records.append(record)
            except Exception as exc:
                logger.warning("local_doc_catalog_load_failed", path=str(path), error=str(exc))

        for record in self._build_summary_concept_seed_records():
            metadata = dict(record.get("metadata") or {})
            source_uri = str(record.get("source_uri") or metadata.get("source_uri") or "").strip()
            key = "|".join(
                [
                    str(record.get("module") or "").lower(),
                    str(record.get("title") or "").lower(),
                    str(record.get("task_name") or "").lower(),
                    source_uri.lower(),
                ]
            )
            if key in dedupe:
                continue
            dedupe.add(key)
            records.append(record)

        self.__class__._local_doc_catalog = records
        logger.info("local_doc_catalog_loaded", records=len(records))
        return self.__class__._local_doc_catalog

    def _build_summary_concept_seed_records(self) -> List[Dict[str, Any]]:
        base_dir = Path(os.getenv("IWERP_BASE_DIR", "/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod"))
        workspace_root = base_dir.parent
        readiness_path = (
            workspace_root
            / "Data_Complete"
            / "Finance"
            / "txts"
            / "https_docs.oracle.com_en_cloud_saas_financials_.._.._.._.._.._en_cloud_saas_readiness_epm.html.txt"
        )
        fusion_ai_path = (
            workspace_root
            / "Data_Complete"
            / "Finance"
            / "txts"
            / "https_docs.oracle.com_en_cloud_saas_fusion-ai_.txt"
        )
        gl_index_path = (
            workspace_root
            / "Data_Complete"
            / "Finance"
            / "txts"
            / "https_docs.oracle.com_en_cloud_saas_financials_25d_faugl_index.html.txt"
        )

        seed_records: List[Dict[str, Any]] = []

        source_paths = [path for path in (readiness_path, fusion_ai_path) if path.exists()]
        if source_paths:
            excerpts: List[str] = []
            for path in source_paths:
                try:
                    text = path.read_text(encoding="utf-8")
                except Exception:
                    continue
                compact = re.sub(r"\s+", " ", text).strip()
                if compact:
                    excerpts.append(compact[:1600])
            if excerpts:
                primary_path = source_paths[0]
                combined_content = "\n\n".join(excerpts)
                title = "Enterprise Performance Management (EPM) Overview"
                summary_points = [
                    "EPM stands for Enterprise Performance Management in Oracle Cloud Applications.",
                    "Oracle documentation treats Enterprise Performance Management as its own cloud product family, separate from ERP, SCM, HCM, Sales, and Service.",
                    "Oracle publishes readiness and feature information for EPM under the Enterprise Performance Management area.",
                ]
                metadata = {
                    "title": title,
                    "module": ModuleFamily.FINANCIALS.value,
                    "task_type": TaskType.SUMMARY.value,
                    "doc_type": "functional_doc",
                    "quality_score": 0.99,
                    "source_uri": str(primary_path),
                    "source_path": str(primary_path),
                    "authority_tier": "official",
                    "source_system": "local_concept_seed",
                    "task_name": "enterprise performance management",
                    "query_aliases": [
                        "what is epm",
                        "epm",
                        "enterprise performance management",
                        "oracle enterprise performance management",
                    ],
                    "summary_points": summary_points,
                    "purpose": "In Oracle Cloud documentation, EPM refers to Enterprise Performance Management.",
                    "warnings": [
                        "EPM is a cloud product family and should not be answered with unrelated General Ledger setup steps.",
                    ],
                    "allow_deterministic_grounded_answer": True,
                    "grounding_answerability_pass": True,
                    "source_doc_title": "Enterprise Performance Management source bundle",
                    "source_doc_excerpt": combined_content[:1400],
                    "source_bundle_paths": [str(path) for path in source_paths],
                }
                seed_records.append(
                    {
                        "title": title,
                        "module": ModuleFamily.FINANCIALS.value,
                        "task_name": "enterprise performance management",
                        "query_aliases": metadata["query_aliases"],
                        "task_type": TaskType.SUMMARY.value,
                        "doc_type": "functional_doc",
                        "quality_score": 0.99,
                        "source_uri": str(primary_path),
                        "source_path": str(primary_path),
                        "content": combined_content,
                        "metadata": metadata,
                    }
                )

        if gl_index_path.exists():
            try:
                gl_text = gl_index_path.read_text(encoding="utf-8")
            except Exception:
                gl_text = ""
            compact_gl = re.sub(r"\s+", " ", gl_text).strip()
            if compact_gl:
                gl_title = "General Ledger (GL) Overview"
                gl_summary_points = [
                    "GL stands for General Ledger in Oracle Fusion Financials.",
                    "General Ledger is the central accounting module for journals, balances, ledgers, and financial reporting.",
                    "Subledgers such as Payables and Receivables transfer accounting entries into General Ledger for consolidated reporting.",
                ]
                gl_metadata = {
                    "title": gl_title,
                    "module": FusionModule.GENERAL_LEDGER.value,
                    "task_type": TaskType.SUMMARY.value,
                    "doc_type": "functional_doc",
                    "quality_score": 0.99,
                    "source_uri": str(gl_index_path),
                    "source_path": str(gl_index_path),
                    "authority_tier": "official",
                    "source_system": "local_concept_seed",
                    "task_name": "general ledger",
                    "query_aliases": [
                        "what is gl",
                        "gl",
                        "general ledger",
                        "oracle fusion general ledger",
                    ],
                    "summary_points": gl_summary_points,
                    "purpose": "In Oracle Fusion Financials, GL refers to General Ledger.",
                    "warnings": [
                        "General Ledger is a finance concept and should not be answered with unrelated module content.",
                    ],
                    "allow_deterministic_grounded_answer": True,
                    "grounding_answerability_pass": True,
                    "source_doc_title": "Using General Ledger",
                    "source_doc_excerpt": compact_gl[:1400],
                }
                seed_records.append(
                    {
                        "title": gl_title,
                        "module": FusionModule.GENERAL_LEDGER.value,
                        "task_name": "general ledger",
                        "query_aliases": gl_metadata["query_aliases"],
                        "task_type": TaskType.SUMMARY.value,
                        "doc_type": "functional_doc",
                        "quality_score": 0.99,
                        "source_uri": str(gl_index_path),
                        "source_path": str(gl_index_path),
                        "content": compact_gl,
                        "metadata": gl_metadata,
                    }
                )

        return seed_records

    def _normalize_stage2_doc_object(self, row: Dict[str, Any]) -> Dict[str, Any]:
        object_type = str(row.get("object_type") or "").strip().lower()
        module = str(row.get("module") or FusionModule.UNKNOWN.value).strip() or FusionModule.UNKNOWN.value
        source_path = str(row.get("source_path") or "").strip()
        citations = row.get("citations") or []
        source_anchor = ""
        if citations and isinstance(citations, list):
            first_citation = citations[0] or {}
            source_anchor = str(first_citation.get("source_anchor") or "").strip()
            if not source_path:
                source_path = str(first_citation.get("source_path") or "").strip()
        source_uri = source_path

        metadata: Dict[str, Any] = {
            "source_uri": source_uri,
            "source_path": source_path,
            "module": module,
            "source_system": "oraclewings_stage2",
            "authority_tier": "official",
            "grounding_answerability_pass": True,
            "allow_deterministic_grounded_answer": True,
            "derived_from_source_doc": True,
            "source_doc_excerpt": source_anchor,
            "quality_score": float(row.get("confidence_score") or 0.82),
        }

        if object_type == "procedure":
            task_name = str(row.get("task_name") or "").strip() or "Oracle Fusion Procedure"
            prereq = self._coerce_string_list(row.get("prerequisites"))
            step_rows = row.get("ordered_steps") or []
            ordered_steps: List[str] = []
            if isinstance(step_rows, list):
                for item in step_rows:
                    if isinstance(item, str):
                        text = item.strip()
                    elif isinstance(item, dict):
                        text = str(item.get("step_text") or "").strip()
                    else:
                        text = ""
                    if text:
                        ordered_steps.append(text)
            navigation_path = str(row.get("navigation_path") or "").strip()
            warnings = self._coerce_string_list(row.get("warnings_or_constraints"))
            expected_outcome = str(row.get("expected_outcome") or "").strip()
            roles = self._coerce_string_list(row.get("roles_or_personas"))
            keywords = self._coerce_string_list(row.get("keywords"))
            summary_points: List[str] = []
            if expected_outcome:
                summary_points.append(expected_outcome)
            summary_points.extend(ordered_steps[:3])

            metadata.update(
                {
                    "task_name": task_name,
                    "task_signals": [task_name.lower(), *[k.lower() for k in keywords[:8]]],
                    "query_aliases": keywords[:10],
                    "task_type": "procedure",
                    "doc_type": "procedure_doc",
                    "prerequisites": prereq,
                    "ordered_steps": ordered_steps,
                    "navigation_paths": [navigation_path] if navigation_path else [],
                    "warnings": warnings,
                    "summary_points": summary_points[:6],
                    "purpose": expected_outcome or f"Use this grounded Oracle Fusion flow for {task_name}.",
                    "role_context": roles,
                    "stage2_mass_ingestion": True,
                }
            )
            content_lines = [f"Task: {task_name}"]
            if prereq:
                content_lines.append("Prerequisites: " + "; ".join(prereq))
            if navigation_path:
                content_lines.append("Navigation: " + navigation_path)
            if ordered_steps:
                content_lines.append("Ordered Steps:")
                content_lines.extend(f"{idx}. {step}" for idx, step in enumerate(ordered_steps, start=1))
            if warnings:
                content_lines.append("Notes / Constraints: " + "; ".join(warnings))
            if expected_outcome:
                content_lines.append("Expected Outcome: " + expected_outcome)
            content = "\n".join(content_lines).strip()
            return {
                "module": module,
                "title": task_name,
                "task_name": task_name,
                "task_type": "procedure",
                "doc_type": "procedure_doc",
                "source_uri": source_uri,
                "source_path": source_path,
                "quality_score": float(metadata["quality_score"]),
                "content": content,
                "metadata": metadata,
            }

        if object_type == "troubleshooting":
            symptom = str(row.get("symptom") or "").strip() or "Oracle Fusion troubleshooting scenario"
            causes = self._coerce_string_list(row.get("probable_causes"))
            resolution_rows = row.get("resolution_steps") or []
            resolution_steps: List[str] = []
            if isinstance(resolution_rows, list):
                for item in resolution_rows:
                    if isinstance(item, str):
                        text = item.strip()
                    elif isinstance(item, dict):
                        text = str(item.get("step_text") or "").strip()
                    else:
                        text = ""
                    if text:
                        resolution_steps.append(text)
            validation_steps = self._coerce_string_list(row.get("validation_steps"))
            related_tasks = self._coerce_string_list(row.get("related_tasks"))
            keywords = self._coerce_string_list(row.get("keywords"))
            symptom_terms = [symptom, *keywords[:6]]

            metadata.update(
                {
                    "task_name": related_tasks[0] if related_tasks else symptom,
                    "task_signals": [item.lower() for item in related_tasks[:6]],
                    "query_aliases": keywords[:12],
                    "task_type": "troubleshooting",
                    "doc_type": "troubleshooting_doc",
                    "symptom_terms": symptom_terms[:8],
                    "error_keywords": [k.lower() for k in keywords if "error" in k.lower() or "issue" in k.lower()][:8],
                    "root_causes": causes,
                    "resolution_steps": resolution_steps,
                    "warnings": validation_steps,
                    "summary_points": causes[:2] + resolution_steps[:2],
                    "purpose": f"Troubleshoot: {symptom}",
                    "stage2_mass_ingestion": True,
                }
            )
            content_lines = [f"Symptom: {symptom}"]
            if causes:
                content_lines.append("Likely Causes:")
                content_lines.extend(f"- {cause}" for cause in causes)
            if resolution_steps:
                content_lines.append("Resolution Steps:")
                content_lines.extend(f"{idx}. {step}" for idx, step in enumerate(resolution_steps, start=1))
            if validation_steps:
                content_lines.append("Validation Steps:")
                content_lines.extend(f"- {step}" for step in validation_steps)
            content = "\n".join(content_lines).strip()
            return {
                "module": module,
                "title": symptom,
                "task_name": related_tasks[0] if related_tasks else symptom,
                "task_type": "troubleshooting",
                "doc_type": "troubleshooting_doc",
                "source_uri": source_uri,
                "source_path": source_path,
                "quality_score": float(metadata["quality_score"]),
                "content": content,
                "metadata": metadata,
            }
        return {}

    def _doc_task_compatible(self, route_task: TaskType, record_task: str) -> bool:
        normalized = str(record_task or "").strip().lower()
        if not normalized:
            return True
        if route_task in {TaskType.PROCEDURE, TaskType.NAVIGATION}:
            return normalized in {"procedure", "navigation", "setup", "integration", "general"}
        if route_task == TaskType.TROUBLESHOOTING:
            return normalized in {"troubleshooting", "procedure", "general"}
        if route_task in {TaskType.GENERAL, TaskType.SUMMARY, TaskType.INTEGRATION}:
            return normalized in {"summary", "general", "procedure", "integration", "setup"}
        return True

    def _match_local_doc_record(
        self,
        user_query: str,
        route_info: Any,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        if not records:
            return None

        lowered_query = user_query.lower()
        query_tokens = self._specialized_query_tokens(user_query)
        task_profile = TaskSemanticAnalyzer.extract_query_signals(user_query)
        requested_signals = {
            str(item.get("task") or "").strip().lower()
            for item in (task_profile.get("signals") or [])
            if str(item.get("task") or "").strip()
        }
        top_task = str(task_profile.get("top_task") or "").strip().lower()
        top_task_tokens = self._specialized_query_tokens(top_task)
        preferred_modules = {
            self._canonical_module_name(module_name)
            for module_name in TaskSemanticAnalyzer.TASK_CONFIGS.get(top_task, {}).get("preferred_modules", [])
            if str(module_name).strip()
        }
        requested_module_raw = self._canonical_module_name(route_info.module)
        requested_module = (
            requested_module_raw
            if requested_module_raw and requested_module_raw != FusionModule.UNKNOWN.value
            else ""
        )
        requested_module_known = bool(requested_module)
        requested_module_in_preferred = bool(requested_module and requested_module in preferred_modules)
        requested_family = (
            next(iter(module_families_for_value(requested_module)), ModuleFamily.UNKNOWN.value)
            if requested_module_known
            else route_info.module_family.value
        )
        summary_concept_profile = self._summary_concept_profile(user_query, route_info)
        is_summary_concept_prompt = self._is_summary_concept_prompt(user_query)
        explicit_module = bool(getattr(route_info, "module_explicit", False))
        strict_leaf = self._is_strict_financial_leaf_route(route_info)
        allow_preferred_sibling = bool(preferred_modules and requested_module_in_preferred)
        has_exact_task_candidate = False
        if explicit_module and requested_module_known:
            for candidate in records:
                if not self._doc_task_compatible(route_info.task_type, str(candidate.get("task_type") or "")):
                    continue
                candidate_module = self._canonical_module_name(candidate.get("module"))
                if candidate_module != requested_module:
                    continue
                candidate_metadata = dict(candidate.get("metadata") or {})
                candidate_text = " ".join(
                    [
                        str(candidate.get("task_name") or ""),
                        str(candidate.get("title") or ""),
                        " ".join(self._coerce_string_list(candidate.get("query_aliases"))),
                        str(candidate_metadata.get("task_name") or ""),
                        " ".join(self._coerce_string_list(candidate_metadata.get("task_signals"))),
                        str(candidate.get("content") or "")[:900],
                    ]
                )
                candidate_tokens = self._specialized_query_tokens(candidate_text)
                overlap = self._specialized_overlap_score(query_tokens, candidate_tokens)
                if top_task_tokens:
                    overlap = max(overlap, self._specialized_overlap_score(top_task_tokens, candidate_tokens))
                if overlap > 0.02:
                    has_exact_task_candidate = True
                    break

        best_record: Dict[str, Any] | None = None
        best_score = 0.0
        for record in records:
            if not self._doc_task_compatible(route_info.task_type, str(record.get("task_type") or "")):
                continue

            record_module = self._canonical_module_name(record.get("module"))
            record_family = next(iter(module_families_for_value(record_module)), ModuleFamily.UNKNOWN.value)
            if (
                explicit_module
                and requested_module_known
                and has_exact_task_candidate
                and record_module
                and record_module not in {requested_module, FusionModule.COMMON.value}
            ):
                continue
            if (
                strict_leaf
                and explicit_module
                and requested_module_known
                and record_module
                and record_module not in {requested_module, FusionModule.COMMON.value}
                and not (allow_preferred_sibling and record_module in preferred_modules)
            ):
                continue

            module_score = 0.0
            if requested_module_known and record_module:
                if record_module == requested_module:
                    module_score = 4.6
                elif explicit_module and has_exact_task_candidate and record_module != FusionModule.COMMON.value:
                    module_score = -2.4
                elif allow_preferred_sibling and record_module in preferred_modules:
                    module_score = 1.7
                elif record_family == requested_family:
                    module_score = 1.1
                elif record_module == FusionModule.COMMON.value:
                    module_score = 0.6
                elif explicit_module:
                    module_score = -2.2
                else:
                    module_score = -0.8
            elif requested_family != ModuleFamily.UNKNOWN.value and record_family == requested_family:
                module_score = 1.15
            if not requested_module_known and preferred_modules and record_module in preferred_modules:
                module_score += 1.15

            task_name = str(record.get("task_name") or "").lower().strip()
            aliases = self._coerce_string_list(record.get("query_aliases"))
            alias_tokens = self._specialized_query_tokens(" ".join(aliases))
            title_tokens = self._specialized_query_tokens(str(record.get("title") or ""))
            content_tokens = self._specialized_query_tokens(str(record.get("content") or "")[:1800])
            metadata = dict(record.get("metadata") or {})
            purpose_tokens = self._specialized_query_tokens(str(metadata.get("purpose") or ""))
            task_signal_tokens = {
                str(token).strip().lower()
                for token in (metadata.get("task_signals") or [])
                if str(token).strip()
            }
            concept_relevance = self._summary_concept_relevance(
                user_query=user_query,
                metadata=metadata,
                title=str(record.get("title") or ""),
                content=str(record.get("content") or ""),
                concept_profile=summary_concept_profile,
            )
            if route_info.task_type in self.SUMMARY_TASK_TYPES and summary_concept_profile:
                if concept_relevance < float(summary_concept_profile.get("min_relevance") or 0.25):
                    continue
            if (
                route_info.task_type in self.SUMMARY_TASK_TYPES
                and is_summary_concept_prompt
                and self._is_summary_procedural_doc(metadata)
                and concept_relevance < 0.72
            ):
                continue
            if route_info.task_type in self.SUMMARY_TASK_TYPES:
                summary_safety = self._summary_safety_assessment(
                    user_query=user_query,
                    route_info=route_info,
                    metadata=metadata,
                    title=str(record.get("title") or ""),
                    content=str(record.get("content") or ""),
                )
                if not summary_safety.get("strong_match"):
                    continue

            overlap = self._specialized_overlap_score(query_tokens, title_tokens) * 2.6
            overlap += self._specialized_overlap_score(query_tokens, alias_tokens) * 4.1
            overlap += self._specialized_overlap_score(query_tokens, purpose_tokens) * 2.1
            overlap += self._specialized_overlap_score(query_tokens, content_tokens) * 0.9
            if route_info.task_type in self.SUMMARY_TASK_TYPES and summary_concept_profile:
                overlap += concept_relevance * 4.6
            if route_info.task_type in self.SUMMARY_TASK_TYPES:
                overlap += float(summary_safety.get("score") or 0.0) * 3.8

            task_signal_overlap = self._specialized_overlap_score(requested_signals, task_signal_tokens)
            overlap += task_signal_overlap * 3.8
            if task_name and task_name in lowered_query:
                overlap += 2.7
            elif task_name:
                task_tokens = self._specialized_query_tokens(task_name)
                overlap += self._specialized_overlap_score(query_tokens, task_tokens) * 2.6
            if top_task_tokens:
                overlap += self._specialized_overlap_score(top_task_tokens, title_tokens) * 3.8
                overlap += self._specialized_overlap_score(top_task_tokens, task_signal_tokens) * 4.1
                if task_name and top_task and top_task in task_name:
                    overlap += 2.2

            quality = float(record.get("quality_score") or 0.8)
            deterministic = bool(
                metadata.get("allow_deterministic_grounded_answer")
                or record.get("allow_deterministic_grounded_answer")
            )
            score = module_score + overlap + (quality * 0.6) + (0.8 if deterministic else 0.0)
            if route_info.task_type == TaskType.TROUBLESHOOTING:
                if metadata.get("root_causes") and metadata.get("resolution_steps"):
                    score += 1.6
            elif route_info.task_type in {TaskType.PROCEDURE, TaskType.NAVIGATION}:
                if metadata.get("ordered_steps"):
                    score += 1.2
            else:
                if metadata.get("summary_points") or metadata.get("purpose"):
                    score += 0.8

            if score > best_score:
                best_score = score
                best_record = record

        threshold = 1.7
        if route_info.task_type == TaskType.TROUBLESHOOTING:
            threshold = 1.45
        elif route_info.task_type in {TaskType.PROCEDURE, TaskType.NAVIGATION, TaskType.GENERAL, TaskType.SUMMARY, TaskType.INTEGRATION}:
            threshold = 1.35
            if self._is_strict_financial_leaf_route(route_info):
                threshold = 1.2
        if best_score >= threshold:
            return best_record
        return None

    def _task_driven_local_doc_fallback(
        self,
        user_query: str,
        route_info: Any,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        if not records:
            return None

        task_profile = TaskSemanticAnalyzer.extract_query_signals(user_query)
        top_task = str(task_profile.get("top_task") or "").strip().lower()
        if not top_task:
            return None
        task_tokens = self._specialized_query_tokens(top_task)
        if not task_tokens:
            return None

        preferred_modules = {
            self._canonical_module_name(module_name)
            for module_name in TaskSemanticAnalyzer.TASK_CONFIGS.get(top_task, {}).get("preferred_modules", [])
            if str(module_name).strip()
        }
        requested_module_raw = self._canonical_module_name(route_info.module)
        requested_module = (
            requested_module_raw
            if requested_module_raw and requested_module_raw != FusionModule.UNKNOWN.value
            else ""
        )
        requested_module_lower = requested_module.lower() if requested_module else ""
        requested_family = (
            next(iter(module_families_for_value(requested_module)), ModuleFamily.UNKNOWN.value)
            if requested_module
            else route_info.module_family.value
        )
        broad_modules = {
            ModuleFamily.FINANCIALS.value.lower(),
            ModuleFamily.PROCUREMENT.value.lower(),
            ModuleFamily.HCM.value.lower(),
            ModuleFamily.SCM.value.lower(),
            ModuleFamily.PROJECTS.value.lower(),
        }
        allow_family_fallback = (not requested_module) or (requested_module_lower in broad_modules)
        explicit_module = bool(getattr(route_info, "module_explicit", False))
        requested_module_in_preferred = bool(requested_module and requested_module in preferred_modules)
        has_exact_task_candidate = False
        if explicit_module and requested_module:
            for candidate in records:
                if not self._doc_task_compatible(route_info.task_type, str(candidate.get("task_type") or "")):
                    continue
                candidate_module = self._canonical_module_name(candidate.get("module"))
                if candidate_module != requested_module:
                    continue
                candidate_metadata = dict(candidate.get("metadata") or {})
                candidate_text = " ".join(
                    [
                        str(candidate.get("task_name") or ""),
                        str(candidate.get("title") or ""),
                        " ".join(self._coerce_string_list(candidate.get("query_aliases"))),
                        str(candidate_metadata.get("task_name") or ""),
                        " ".join(self._coerce_string_list(candidate_metadata.get("task_signals"))),
                        str(candidate.get("content") or "")[:900],
                    ]
                )
                candidate_tokens = self._specialized_query_tokens(candidate_text)
                overlap = self._specialized_overlap_score(task_tokens, candidate_tokens)
                if overlap > 0.02:
                    has_exact_task_candidate = True
                    break

        best_record: Dict[str, Any] | None = None
        best_score = 0.0
        for record in records:
            if not self._doc_task_compatible(route_info.task_type, str(record.get("task_type") or "")):
                continue

            record_module = self._canonical_module_name(record.get("module"))
            record_family = next(iter(module_families_for_value(record_module)), ModuleFamily.UNKNOWN.value)
            if explicit_module and requested_module and record_module:
                if record_module != requested_module:
                    if has_exact_task_candidate:
                        continue
                    preferred_sibling = requested_module_in_preferred and record_module in preferred_modules
                    if not allow_family_fallback and not preferred_sibling:
                        continue
                    if (
                        not preferred_sibling
                        and record_family != requested_family
                        and record_module != FusionModule.COMMON.value
                        and record_module not in preferred_modules
                    ):
                        continue

            metadata = dict(record.get("metadata") or {})
            candidate_text = " ".join(
                [
                    str(record.get("task_name") or ""),
                    str(record.get("title") or ""),
                    " ".join(self._coerce_string_list(record.get("query_aliases"))),
                    str(metadata.get("task_name") or ""),
                    " ".join(self._coerce_string_list(metadata.get("task_signals"))),
                    str(record.get("content") or "")[:900],
                ]
            )
            candidate_tokens = self._specialized_query_tokens(candidate_text)
            overlap = self._specialized_overlap_score(task_tokens, candidate_tokens)
            if overlap <= 0.0:
                continue

            score = overlap * 5.0
            if record_module and requested_module and record_module == requested_module:
                score += 2.2
            elif requested_module_in_preferred and record_module in preferred_modules:
                score += 1.45
            elif allow_family_fallback and record_family == requested_family:
                score += 0.9
            elif not requested_module and record_module in preferred_modules:
                score += 1.2
            score += float(record.get("quality_score") or metadata.get("quality_score") or 0.75) * 0.4
            if metadata.get("allow_deterministic_grounded_answer"):
                score += 0.45

            if score > best_score:
                best_score = score
                best_record = record

        if best_score >= 0.9:
            return best_record
        return None

    def _build_specialized_catalog_chunk(
        self,
        record: Dict[str, Any],
        *,
        corpus: str,
        doc_type: str,
    ) -> Dict[str, Any]:
        record_metadata = dict(record.get("metadata") or {})
        module = str(record.get("module") or FusionModule.UNKNOWN.value)
        module_family = next(iter(module_families_for_value(module)), ModuleFamily.UNKNOWN.value)
        metadata: Dict[str, Any] = {
            "corpus": corpus,
            "title": record.get("title") or "specialized-catalog-record",
            "module": module,
            "module_family": module_family,
            "source_uri": record.get("source_uri") or record.get("source_path") or record_metadata.get("source_uri") or "specialized://catalog",
            "source_path": record.get("source_path") or record_metadata.get("source_path") or "",
            "doc_type": record.get("doc_type") or record_metadata.get("doc_type") or doc_type,
            "task_type": record.get("task_type") or record_metadata.get("task_type") or "",
            "tables_used": record.get("tables_used") or record_metadata.get("tables_used") or [],
            "columns_used": record.get("columns_used") or record_metadata.get("columns_used") or [],
            "joins_used": record.get("joins_used") or record_metadata.get("joins_used") or [],
            "formula_type": record.get("formula_type") or record_metadata.get("formula_type") or "UNKNOWN",
            "formula_name": (
                record.get("formula_name")
                or record_metadata.get("formula_name")
                or record.get("title")
                or "UNKNOWN"
            ),
            "database_items": record.get("database_items") or record_metadata.get("database_items") or [],
            "contexts": record.get("contexts") or record_metadata.get("contexts") or [],
            "functions": record.get("functions") or record_metadata.get("functions") or [],
            "quality_score": float(
                record.get("quality_score")
                or record.get("confidence")
                or record_metadata.get("quality_score")
                or 1.0
            ),
        }

        passthrough_keys = [
            "task_name",
            "query_aliases",
            "prerequisites",
            "navigation_paths",
            "ordered_steps",
            "summary_points",
            "warnings",
            "symptom_terms",
            "error_keywords",
            "root_causes",
            "resolution_steps",
            "task_signals",
            "purpose",
            "role_context",
            "preferred_leaf_module",
            "grounding_answerability_pass",
            "allow_deterministic_grounded_answer",
            "derived_from_source_doc",
            "source_doc_title",
            "source_doc_excerpt",
            "authority_tier",
            "source_system",
        ]
        for key in passthrough_keys:
            value = record.get(key, record_metadata.get(key))
            if value in (None, "", [], {}):
                continue
            metadata[key] = value

        return {
            "id": record.get("content_hash") or record.get("title") or record.get("source_uri"),
            "content": record.get("content") or "",
            "combined_score": float(record.get("confidence") or record.get("quality_score") or 1.0),
            "metadata": metadata,
        }

    def _sql_field_group(self, field_key: str) -> Optional[Dict[str, Any]]:
        for field_group in self.SQL_REQUEST_FIELD_GROUPS:
            if str(field_group.get("key") or "") == field_key:
                return dict(field_group)
        return None

    def _sql_filter_group(self, filter_key: str) -> Optional[Dict[str, Any]]:
        for filter_group in self.SQL_REQUEST_FILTER_GROUPS:
            if str(filter_group.get("key") or "") == filter_key:
                return dict(filter_group)
        return None

    def _infer_sql_report_family(
        self,
        lowered_query: str,
        route_info: Any,
        required_field_keys: Set[str],
        required_calculation_keys: Set[str],
    ) -> str:
        module = getattr(route_info, "module", FusionModule.UNKNOWN)
        module_value = module.value if module != FusionModule.UNKNOWN else ""
        if module_value == FusionModule.PAYABLES.value:
            if any(token in lowered_query for token in ("payment number", "payment method", "payments report", "payment date", "check number")):
                return "payables_payments"
            if (
                "invoice distribution" in lowered_query
                or {"distribution_line_number", "distribution_amount", "natural_account_segment", "cost_center", "liability_account"} & required_field_keys
            ):
                return "payables_invoice_distribution_accounting"
            if "invoice" in lowered_query:
                return "payables_invoice_details"
        if module_value == FusionModule.RECEIVABLES.value:
            if "aging" in lowered_query or "aging bucket" in lowered_query or "as-of date" in lowered_query or "as of date" in lowered_query:
                return "receivables_aging"
            if any(token in lowered_query for token in ("receipt", "receipts", "application", "applications", "applied amount")):
                return "receivables_receipts_applications"
            if any(token in lowered_query for token in ("transaction", "transactions", "invoice", "trx")):
                return "receivables_transaction_report"
        if module_value == FusionModule.GENERAL_LEDGER.value:
            if "journal" in lowered_query:
                return "general_ledger_journal_details"
            if "balance" in lowered_query or "ending balance" in lowered_query or {"period_net_dr", "period_net_cr", "ending_balance"} & required_field_keys | required_calculation_keys:
                return "general_ledger_account_balances"
        if module_value == FusionModule.PROCUREMENT.value:
            if any(token in lowered_query for token in ("receiving", "received quantity", "billed quantity", "three-way match", "3-way match", "receipt date")):
                return "procurement_receiving_invoicing_match"
            if any(token in lowered_query for token in ("purchase order", "po ", "po number", "purchase order details")):
                return "procurement_purchase_order_details"
        if module_value == FusionModule.CASH_MANAGEMENT.value:
            if "statement" in lowered_query:
                return "cash_management_statement_line_reporting"
            if "external transaction" in lowered_query:
                return "cash_management_external_transaction_reporting"
            if "bank account" in lowered_query:
                return "cash_management_bank_account_reporting"
        if any(token in lowered_query for token in ("payment number", "payment method", "payments report", "payment date", "check number")):
            return "payables_payments"
        if (
            "invoice distribution" in lowered_query
            or {"distribution_line_number", "distribution_amount", "natural_account_segment", "cost_center", "liability_account"} & required_field_keys
        ):
            return "payables_invoice_distribution_accounting"
        if "ar " in lowered_query or "receivable" in lowered_query or "receivables" in lowered_query:
            if "aging" in lowered_query or "aging bucket" in lowered_query or "as-of date" in lowered_query or "as of date" in lowered_query:
                return "receivables_aging"
            if any(token in lowered_query for token in ("receipt", "receipts", "application", "applications", "applied amount")):
                return "receivables_receipts_applications"
            if any(token in lowered_query for token in ("transaction", "transactions", "invoice", "trx")):
                return "receivables_transaction_report"
        if "gl " in lowered_query or "general ledger" in lowered_query:
            if "journal" in lowered_query:
                return "general_ledger_journal_details"
            if "balance" in lowered_query or "ending balance" in lowered_query or {"period_net_dr", "period_net_cr", "ending_balance"} & (required_field_keys | required_calculation_keys):
                return "general_ledger_account_balances"
        if any(token in lowered_query for token in ("receiving", "received quantity", "billed quantity", "three-way match", "3-way match", "receipt date")) and any(
            token in lowered_query for token in ("po ", "po number", "purchase order", "purchasing", "procurement")
        ):
            return "procurement_receiving_invoicing_match"
        if any(token in lowered_query for token in ("purchase order", "po ", "po number", "purchase order details", "purchasing", "procurement")):
            return "procurement_purchase_order_details"
        return ""

    def _parse_sql_requested_ordering(self, lowered_query: str) -> List[Dict[str, Any]]:
        order_key_map = {
            "invoice date": "invoice_date",
            "invoice number": "invoice_number",
            "payment date": "payment_date",
            "payment number": "payment_number",
            "transaction date": "transaction_date",
            "transaction number": "transaction_number",
            "receipt date": "receipt_date",
            "receipt number": "receipt_number",
            "journal name": "journal_name",
            "period name": "period_name",
            "line number": "line_number",
            "journal line number": "journal_line_number",
            "account combination": "account_combination",
            "customer name": "customer_name",
            "po number": "po_number",
            "purchase order number": "po_number",
            "po date": "po_date",
        }
        required_ordering: List[Dict[str, Any]] = []
        if "order by" not in lowered_query and "sort by" not in lowered_query:
            return required_ordering
        for phrase, field_key in order_key_map.items():
            if f"order by {phrase}" in lowered_query or f"sort by {phrase}" in lowered_query:
                field_group = self._sql_field_group(field_key)
                if field_group and not any(item.get("key") == field_key for item in required_ordering):
                    required_ordering.append(field_group)
        return required_ordering

    def _parse_sql_required_calculations(self, lowered_query: str) -> List[Dict[str, Any]]:
        calculations: List[Dict[str, Any]] = []
        if "aging bucket" in lowered_query or "age bucket" in lowered_query:
            calculations.append({"key": "aging_bucket", "label": "Aging Bucket"})
        if "ending balance" in lowered_query:
            calculations.append({"key": "ending_balance", "label": "Ending Balance"})
        if re.search(r"\b\d+\s*-\s*\d+\s*day bucket\b", lowered_query) or "custom aging bucket" in lowered_query:
            calculations.append({"key": "custom_aging_bucket", "label": "Custom Aging Bucket"})
        return calculations

    def _apply_sql_report_family_shape(
        self,
        shape: Dict[str, Any],
        report_family: str,
        required_field_keys: Set[str],
    ) -> None:
        required_tables = set(shape.get("required_tables") or [])
        required_join_pairs = list(shape.get("required_join_pairs") or [])
        alias_counts = dict(shape.get("required_table_alias_counts") or {})

        def add_pair(left: str, right: str) -> None:
            pair = (left, right)
            if pair not in required_join_pairs:
                required_join_pairs.append(pair)

        if report_family == "payables_invoice_details":
            required_tables.add("AP_INVOICES_ALL")
            if {"supplier_name", "supplier_number"} & required_field_keys:
                required_tables.add("POZ_SUPPLIERS")
                add_pair("AP_INVOICES_ALL", "POZ_SUPPLIERS")
            if "business_unit_name" in required_field_keys:
                required_tables.add("FUN_ALL_BUSINESS_UNITS_V")
                add_pair("AP_INVOICES_ALL", "FUN_ALL_BUSINESS_UNITS_V")
        elif report_family == "payables_invoice_distribution_accounting":
            required_tables.update({"AP_INVOICE_DISTRIBUTIONS_ALL", "AP_INVOICES_ALL"})
            add_pair("AP_INVOICE_DISTRIBUTIONS_ALL", "AP_INVOICES_ALL")
            if "supplier_name" in required_field_keys:
                required_tables.add("POZ_SUPPLIERS")
                add_pair("AP_INVOICES_ALL", "POZ_SUPPLIERS")
            if {"natural_account_segment", "cost_center", "liability_account"} & required_field_keys:
                required_tables.add("GL_CODE_COMBINATIONS")
                add_pair("AP_INVOICE_DISTRIBUTIONS_ALL", "GL_CODE_COMBINATIONS")
            if "liability_account" in required_field_keys:
                add_pair("AP_INVOICES_ALL", "GL_CODE_COMBINATIONS")
                alias_counts["GL_CODE_COMBINATIONS"] = 2
        elif report_family == "payables_payments":
            required_tables.update({"AP_INVOICE_PAYMENTS_ALL", "AP_INVOICES_ALL", "AP_CHECKS_ALL"})
            add_pair("AP_INVOICE_PAYMENTS_ALL", "AP_INVOICES_ALL")
            add_pair("AP_INVOICE_PAYMENTS_ALL", "AP_CHECKS_ALL")
            if "supplier_name" in required_field_keys:
                required_tables.add("POZ_SUPPLIERS")
                add_pair("AP_CHECKS_ALL", "POZ_SUPPLIERS")
        elif report_family == "receivables_transaction_report":
            required_tables.add("RA_CUSTOMER_TRX_ALL")
            if {"customer_name"} & required_field_keys:
                required_tables.update({"HZ_CUST_ACCOUNTS_", "HZ_PARTIES"})
                add_pair("RA_CUSTOMER_TRX_ALL", "HZ_CUST_ACCOUNTS_")
                add_pair("HZ_CUST_ACCOUNTS_", "HZ_PARTIES")
            if {"amount_due_original", "remaining_amount", "due_date", "payment_status"} & required_field_keys or any(
                item.get("key") in {"open_status", "as_of_date"} for item in (shape.get("required_filters") or [])
            ):
                required_tables.add("AR_PAYMENT_SCHEDULES_ALL")
                add_pair("RA_CUSTOMER_TRX_ALL", "AR_PAYMENT_SCHEDULES_ALL")
            if "business_unit_name" in required_field_keys:
                required_tables.add("FUN_ALL_BUSINESS_UNITS_V")
                add_pair("RA_CUSTOMER_TRX_ALL", "FUN_ALL_BUSINESS_UNITS_V")
        elif report_family == "receivables_receipts_applications":
            required_tables.update({"AR_CASH_RECEIPTS_ALL", "AR_RECEIVABLE_APPLICATIONS_ALL", "RA_CUSTOMER_TRX_ALL"})
            add_pair("AR_CASH_RECEIPTS_ALL", "AR_RECEIVABLE_APPLICATIONS_ALL")
            add_pair("AR_RECEIVABLE_APPLICATIONS_ALL", "RA_CUSTOMER_TRX_ALL")
            if "customer_name" in required_field_keys:
                required_tables.update({"HZ_CUST_ACCOUNTS_", "HZ_PARTIES"})
                add_pair("RA_CUSTOMER_TRX_ALL", "HZ_CUST_ACCOUNTS_")
                add_pair("HZ_CUST_ACCOUNTS_", "HZ_PARTIES")
        elif report_family == "receivables_aging":
            required_tables.update({"RA_CUSTOMER_TRX_ALL", "AR_PAYMENT_SCHEDULES_ALL"})
            add_pair("RA_CUSTOMER_TRX_ALL", "AR_PAYMENT_SCHEDULES_ALL")
            if "customer_name" in required_field_keys:
                required_tables.update({"HZ_CUST_ACCOUNTS_", "HZ_PARTIES"})
                add_pair("RA_CUSTOMER_TRX_ALL", "HZ_CUST_ACCOUNTS_")
                add_pair("HZ_CUST_ACCOUNTS_", "HZ_PARTIES")
        elif report_family == "general_ledger_account_balances":
            required_tables.update({"GL_BALANCES", "GL_CODE_COMBINATIONS", "GL_LEDGERS"})
            add_pair("GL_BALANCES", "GL_CODE_COMBINATIONS")
            add_pair("GL_BALANCES", "GL_LEDGERS")
        elif report_family == "general_ledger_journal_details":
            required_tables.update({"GL_JE_HEADERS", "GL_JE_LINES", "GL_CODE_COMBINATIONS", "GL_LEDGERS"})
            add_pair("GL_JE_HEADERS", "GL_JE_LINES")
            add_pair("GL_JE_LINES", "GL_CODE_COMBINATIONS")
            add_pair("GL_JE_HEADERS", "GL_LEDGERS")
        elif report_family == "procurement_purchase_order_details":
            required_tables.update({"PO_HEADERS_ALL", "PO_LINES_ALL", "POZ_SUPPLIERS"})
            add_pair("PO_HEADERS_ALL", "PO_LINES_ALL")
            add_pair("PO_HEADERS_ALL", "POZ_SUPPLIERS")
            if "supplier_site_code" in required_field_keys:
                required_tables.add("POZ_SUPPLIER_SITES_ALL_M")
                add_pair("PO_HEADERS_ALL", "POZ_SUPPLIER_SITES_ALL_M")
        elif report_family == "procurement_receiving_invoicing_match":
            required_tables.update({"PO_HEADERS_ALL", "PO_LINES_ALL", "PO_LINE_LOCATIONS_ALL", "RCV_TRANSACTIONS", "POZ_SUPPLIERS"})
            add_pair("PO_HEADERS_ALL", "PO_LINES_ALL")
            add_pair("PO_LINES_ALL", "PO_LINE_LOCATIONS_ALL")
            add_pair("PO_LINE_LOCATIONS_ALL", "RCV_TRANSACTIONS")
            add_pair("PO_HEADERS_ALL", "POZ_SUPPLIERS")

        shape["required_tables"] = sorted(required_tables)
        shape["required_join_pairs"] = required_join_pairs
        shape["required_table_alias_counts"] = alias_counts

    def _parse_sql_request_shape(self, user_query: str, route_info: Any) -> Dict[str, Any]:
        lowered_query = user_query.lower()
        registry = self.verifier.registry
        required_fields: List[Dict[str, Any]] = []
        required_filters: List[Dict[str, Any]] = []
        required_tables: Set[str] = set()
        required_table_alias_counts: Dict[str, int] = {}
        explicit_table_sequence: List[str] = []

        def add_required_field(field_group: Dict[str, Any]) -> None:
            field_key = str(field_group.get("key") or "")
            if not field_key:
                return
            if any(str(existing.get("key") or "") == field_key for existing in required_fields):
                return
            required_fields.append(dict(field_group))

        def add_required_filter(filter_group: Dict[str, Any]) -> None:
            filter_key = str(filter_group.get("key") or "")
            if not filter_key:
                return
            if any(str(existing.get("key") or "") == filter_key for existing in required_filters):
                return
            required_filters.append(dict(filter_group))

        for field_group in self.SQL_REQUEST_FIELD_GROUPS:
            if str(field_group.get("key") or "") == "line_number" and any(
                token in lowered_query
                for token in ("distribution line number", "journal line number", "statement line number")
            ):
                continue
            if any(phrase in lowered_query for phrase in field_group["phrases"]):
                add_required_field(field_group)
        for filter_group in self.SQL_REQUEST_FILTER_GROUPS:
            if any(phrase in lowered_query for phrase in filter_group["phrases"]):
                add_required_filter(filter_group)

        for raw_line in re.split(r"[\r\n]+", user_query):
            line = re.sub(r"^\s*(?:[-*]|\d+[\.\)])\s*", "", raw_line or "").strip().lower()
            if not line:
                continue
            for field_group in self.SQL_REQUEST_FIELD_GROUPS:
                field_key = str(field_group.get("key") or "")
                label = str(field_group.get("label") or "").strip().lower()
                alias_phrases = {
                    str(alias).strip().lower().replace("_", " ")
                    for alias in (field_group.get("aliases") or [])
                    if str(alias).strip()
                }
                phrase_hits = any(phrase in line for phrase in (field_group.get("phrases") or []))
                label_hit = bool(label and (line == label or label in line))
                alias_hit = any(alias in line for alias in alias_phrases if alias)
                if field_key == "line_number" and any(
                    token in line for token in ("distribution line number", "journal line number", "statement line number")
                ):
                    phrase_hits = False
                    label_hit = line == label
                    alias_hit = line == "line number"
                if phrase_hits or label_hit or alias_hit:
                    add_required_field(field_group)
            for filter_group in self.SQL_REQUEST_FILTER_GROUPS:
                label = str(filter_group.get("label") or "").strip().lower()
                if any(phrase in line for phrase in (filter_group.get("phrases") or [])) or (
                    label and (line == label or label in line)
                ):
                    add_required_filter(filter_group)

        if ":p_from_date" in lowered_query and ":p_to_date" in lowered_query and "invoice date" in lowered_query:
            filter_group = self._sql_filter_group("invoice_date_between")
            if filter_group:
                add_required_filter(filter_group)
        if ":p_payment_date" in lowered_query:
            filter_group = self._sql_filter_group("payment_date_equals")
            if filter_group:
                add_required_filter(filter_group)
        if ":p_as_of_date" in lowered_query or "as-of date" in lowered_query or "as of date" in lowered_query:
            filter_group = self._sql_filter_group("as_of_date")
            if filter_group:
                add_required_filter(filter_group)
        if ":p_period_name" in lowered_query:
            filter_group = self._sql_filter_group("period_bind")
            if filter_group:
                add_required_filter(filter_group)
        if ":p_ledger_name" in lowered_query or ":p_ledger_id" in lowered_query:
            filter_group = self._sql_filter_group("ledger_bind")
            if filter_group:
                add_required_filter(filter_group)

        for token in re.findall(r"\b[A-Z][A-Z0-9_$]{2,}\b", user_query.upper()):
            if "_" not in token:
                continue
            canonical = registry.resolve_object_name(token) or token
            if registry.has_object(canonical):
                required_tables.add(canonical)
                if canonical not in explicit_table_sequence:
                    explicit_table_sequence.append(canonical)

        required_ordering = self._parse_sql_requested_ordering(lowered_query)
        required_calculations = self._parse_sql_required_calculations(lowered_query)
        required_field_keys = {field.get("key") for field in required_fields}
        required_calculation_keys = {item.get("key") for item in required_calculations}
        report_family = self._infer_sql_report_family(lowered_query, route_info, required_field_keys, required_calculation_keys)

        shape = {
            "module": route_info.module.value if getattr(route_info, "module", FusionModule.UNKNOWN) != FusionModule.UNKNOWN else route_info.module_family.value,
            "required_fields": required_fields,
            "required_filters": required_filters,
            "required_tables": sorted(required_tables),
            "required_table_alias_counts": required_table_alias_counts,
            "required_join_pairs": [
                (explicit_table_sequence[idx], explicit_table_sequence[idx + 1])
                for idx in range(len(explicit_table_sequence) - 1)
            ],
            "required_ordering": required_ordering,
            "required_calculations": required_calculations,
            "report_family": report_family,
            "template_hint": report_family,
            "aggregation": next((token for token in ("count", "sum", "avg", "average", "total", "group by") if token in lowered_query), ""),
            "business_objects": [
                phrase
                for phrase in (
                    "invoice",
                    "invoice distribution",
                    "payment",
                    "receipt",
                    "aging",
                    "journal",
                    "balance",
                    "purchase order",
                    "receiving",
                )
                if phrase in lowered_query
            ],
        }
        self._apply_sql_report_family_shape(shape, report_family, {str(item) for item in required_field_keys if item})
        explicit_join_signal = any(
            keyword in lowered_query
            for keyword in (" join ", " joins ", "validated joins", "using joins", "join between")
        )
        shape["needs_join"] = len(shape.get("required_tables") or []) > 1 or explicit_join_signal
        return shape

    def _extract_sql_features(self, sql_text: str) -> Dict[str, Any]:
        parsed = sqlglot.parse_one((sql_text or "").strip().rstrip(";"), read="oracle")
        registry = self.verifier.registry
        table_sequence: List[str] = []
        projection_columns: Set[str] = set()
        projection_aliases: Set[str] = set()
        where_columns: Set[str] = set()
        where_text = ""

        for table in parsed.find_all(exp.Table):
            canonical = registry.resolve_object_name(table.name.upper()) or table.name.upper()
            table_sequence.append(canonical)

        for expression in list(getattr(parsed, "expressions", []) or []):
            alias_name = str(getattr(expression, "alias_or_name", "") or "").upper()
            if alias_name:
                projection_aliases.add(alias_name)
            for column in expression.find_all(exp.Column):
                projection_columns.add(column.name.upper())

        where_clause = parsed.args.get("where")
        if where_clause is not None:
            where_text = where_clause.sql(dialect="oracle").upper()
            for column in where_clause.find_all(exp.Column):
                where_columns.add(column.name.upper())
        order_clause = parsed.args.get("order")
        order_text = order_clause.sql(dialect="oracle").upper() if order_clause is not None else ""

        return {
            "table_sequence": table_sequence,
            "table_set": set(table_sequence),
            "table_counts": {table_name: table_sequence.count(table_name) for table_name in set(table_sequence)},
            "projection_columns": projection_columns,
            "projection_aliases": projection_aliases,
            "where_columns": where_columns,
            "where_text": where_text,
            "order_clause": order_text,
        }

    def _score_sql_pattern_for_request(
        self,
        sql_text: str,
        metadata: Dict[str, Any],
        request_shape: Optional[Dict[str, Any]],
    ) -> float:
        if not request_shape:
            return 0.0
        try:
            features = self._extract_sql_features(sql_text)
        except Exception:
            return -10.0

        required_fields = list(request_shape.get("required_fields") or [])
        required_tables = {str(item).upper() for item in (request_shape.get("required_tables") or [])}
        required_filters = list(request_shape.get("required_filters") or [])
        alias_counts = {
            str(key).upper(): int(value)
            for key, value in (request_shape.get("required_table_alias_counts") or {}).items()
        }

        score = 0.0
        table_hits = len(required_tables & features["table_set"])
        if required_tables:
            score += (table_hits / max(len(required_tables), 1)) * 5.0
            if table_hits == 0:
                score -= 6.0

        field_hits = 0
        for field in required_fields:
            field_columns = {str(item).upper() for item in (field.get("columns") or [])}
            field_aliases = {str(item).upper() for item in (field.get("aliases") or [])}
            if field_columns & features["projection_columns"] or field_aliases & features["projection_aliases"]:
                field_hits += 1
        if required_fields:
            score += (field_hits / max(len(required_fields), 1)) * 8.0
            if field_hits == 0:
                score -= 10.0

        filter_hits = 0
        for filter_spec in required_filters:
            filter_columns = {str(item).upper() for item in (filter_spec.get("columns") or [])}
            filter_values = {str(item).upper() for item in (filter_spec.get("values") or [])}
            if filter_columns & features["where_columns"]:
                if not filter_values or any(value in features["where_text"] for value in filter_values):
                    filter_hits += 1
        if required_filters:
            score += (filter_hits / max(len(required_filters), 1)) * 4.0

        if request_shape.get("needs_join"):
            if len(features["table_sequence"]) > 1:
                score += 3.0
            else:
                score -= 8.0

        order_clause = features.get("order_clause") or ""
        required_ordering = list(request_shape.get("required_ordering") or [])
        if required_ordering:
            ordering_hits = 0
            for ordering in required_ordering:
                candidate_columns = {str(item).upper() for item in (ordering.get("columns") or [])}
                candidate_aliases = {str(item).upper() for item in (ordering.get("aliases") or [])}
                if any(column in order_clause for column in candidate_columns | candidate_aliases):
                    ordering_hits += 1
            score += (ordering_hits / max(len(required_ordering), 1)) * 2.5
            if ordering_hits == 0:
                score -= 2.0

        for table_name, minimum_count in alias_counts.items():
            if features["table_counts"].get(table_name, 0) >= minimum_count:
                score += 1.5
            else:
                score -= 4.0

        report_family = str(request_shape.get("report_family") or "")
        title_text = " ".join(
            str(metadata.get(key) or "")
            for key in ("title", "source_file", "source_uri")
        ).lower()
        if report_family == "payables_payments" and any(token in title_text for token in ("paymentsexport", "payment", "check")):
            score += 2.0
        if report_family.startswith("payables_invoice") and "invoice" in title_text:
            score += 1.5
        if report_family.startswith("receivables_aging") and "aging" in title_text:
            score += 2.0
        if report_family.startswith("general_ledger") and any(token in title_text for token in ("journal", "balance", "gl")):
            score += 1.0
        if report_family.startswith("procurement") and any(token in title_text for token in ("po_", "purchase order", "receiv")):
            score += 1.0

        if str(metadata.get("module") or "") == str(request_shape.get("module") or ""):
            score += 0.75
        return score

    def _sql_requested_field_keys(self, request_shape: Dict[str, Any]) -> Set[str]:
        return {
            str(field.get("key") or "")
            for field in (request_shape.get("required_fields") or [])
            if str(field.get("key") or "")
        }

    def _sql_requested_filter_keys(self, request_shape: Dict[str, Any]) -> Set[str]:
        return {
            str(item.get("key") or "")
            for item in (request_shape.get("required_filters") or [])
            if str(item.get("key") or "")
        }

    def _sql_order_by_clause(self, request_shape: Dict[str, Any], expression_map: Dict[str, str]) -> str:
        order_expressions: List[str] = []
        for ordering in request_shape.get("required_ordering") or []:
            ordering_key = str(ordering.get("key") or "")
            expression = expression_map.get(ordering_key)
            if expression and expression not in order_expressions:
                order_expressions.append(expression)
        if not order_expressions:
            return ""
        return "ORDER BY " + ", ".join(order_expressions)

    def _sql_report_family_support_diagnostics(self, request_shape: Dict[str, Any]) -> Dict[str, Any]:
        report_family = str(request_shape.get("report_family") or "")
        config = self.SQL_REPORT_FAMILY_REGISTRY.get(report_family) or {}
        supported_fields = set(config.get("supported_fields") or [])
        supported_filters = set(config.get("supported_filters") or [])
        supported_ordering = set(config.get("supported_ordering") or [])
        supported_calculations = supported_fields | {"aging_bucket", "ending_balance"}

        requested_fields = list(request_shape.get("required_fields") or [])
        requested_filters = list(request_shape.get("required_filters") or [])
        requested_ordering = list(request_shape.get("required_ordering") or [])
        requested_calculations = list(request_shape.get("required_calculations") or [])

        def _item_key(item: Any) -> str:
            if isinstance(item, dict):
                return str(item.get("key") or "")
            return str(item or "")

        def _item_label(item: Any, fallback: str) -> str:
            if isinstance(item, dict):
                return str(item.get("label") or item.get("key") or fallback)
            return str(item or fallback)

        missing_fields = [
            _item_label(field, "field")
            for field in requested_fields
            if _item_key(field) not in supported_fields
        ]
        missing_filters = [
            _item_label(item, "filter")
            for item in requested_filters
            if _item_key(item) not in supported_filters
        ]
        missing_ordering = [
            _item_label(item, "ordering")
            for item in requested_ordering
            if _item_key(item) not in supported_ordering
        ]
        missing_calculations = [
            _item_label(item, "calculation")
            for item in requested_calculations
            if _item_key(item) not in supported_calculations
        ]
        return {
            "report_family": report_family,
            "missing_fields": missing_fields,
            "missing_filters": missing_filters,
            "missing_ordering": missing_ordering,
            "missing_calculations": missing_calculations,
        }

    def _specific_sql_support_reason(self, request_shape: Dict[str, Any], fallback_reason: Optional[str] = None) -> str:
        diagnostics = self._sql_report_family_support_diagnostics(request_shape)
        if diagnostics.get("missing_fields"):
            return "Missing grounded support for requested fields: " + ", ".join(diagnostics["missing_fields"]) + "."
        if diagnostics.get("missing_calculations"):
            return "Missing grounded support for requested calculations: " + ", ".join(diagnostics["missing_calculations"]) + "."
        if diagnostics.get("missing_filters"):
            return "Missing grounded support for requested filters: " + ", ".join(diagnostics["missing_filters"]) + "."
        if diagnostics.get("missing_ordering"):
            return "Missing grounded support for requested ordering: " + ", ".join(diagnostics["missing_ordering"]) + "."
        if fallback_reason:
            return fallback_reason
        report_family = str(request_shape.get("report_family") or "")
        if report_family:
            return f"No retained grounded SQL pattern covered the recognized '{report_family}' reporting shape safely enough to emit verified SQL."
        return "No retained grounded SQL pattern covered the requested fields, joins, filters, and style requirements closely enough to adapt safely."

    def _build_supported_sql_report(
        self,
        request_shape: Dict[str, Any],
        user_query: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        report_family = str(request_shape.get("report_family") or "")
        if not report_family:
            return None, None

        diagnostics = self._sql_report_family_support_diagnostics(request_shape)
        if diagnostics["missing_fields"] or diagnostics["missing_filters"] or diagnostics["missing_ordering"] or diagnostics["missing_calculations"]:
            return None, self._specific_sql_support_reason(request_shape)

        if report_family == "payables_invoice_details":
            return self._build_payables_invoice_details_sql(request_shape), None
        if report_family == "payables_invoice_distribution_accounting":
            return self._build_payables_invoice_distribution_sql(request_shape), None
        if report_family == "payables_payments":
            return self._build_payables_payments_sql(request_shape), None
        if report_family == "receivables_transaction_report":
            return self._build_receivables_transaction_sql(request_shape), None
        if report_family == "receivables_receipts_applications":
            return self._build_receivables_receipts_applications_sql(request_shape), None
        if report_family == "receivables_aging":
            return self._build_receivables_aging_sql(request_shape), None
        if report_family == "general_ledger_account_balances":
            return self._build_general_ledger_account_balances_sql(request_shape), None
        if report_family == "general_ledger_journal_details":
            return self._build_general_ledger_journal_details_sql(request_shape), None
        if report_family == "procurement_purchase_order_details":
            return self._build_procurement_purchase_order_details_sql(request_shape), None
        if report_family == "procurement_receiving_invoicing_match":
            return self._build_procurement_receiving_invoicing_match_sql(request_shape), None
        return None, None

    def _build_payables_invoice_details_sql(self, request_shape: Dict[str, Any]) -> str:
        requested_fields = self._sql_requested_field_keys(request_shape)
        requested_filters = self._sql_requested_filter_keys(request_shape)
        select_map = {
            "business_unit_name": "  bu.BU_NAME AS BUSINESS_UNIT_NAME",
            "invoice_number": "  ai.INVOICE_NUM AS INVOICE_NUMBER",
            "invoice_date": "  ai.INVOICE_DATE AS INVOICE_DATE",
            "supplier_name": "  COALESCE(ps.DF_COMPANY_NAME, ps.DF_LEGAL_NAME, ps.TAX_REPORTING_NAME) AS SUPPLIER_NAME",
            "supplier_number": "  ps.SEGMENT1 AS SUPPLIER_NUMBER",
            "invoice_amount": "  ai.INVOICE_AMOUNT AS INVOICE_AMOUNT",
            "invoice_currency": "  ai.INVOICE_CURRENCY_CODE AS INVOICE_CURRENCY",
            "invoice_status": "  ai.APPROVAL_STATUS AS INVOICE_STATUS",
            "accounting_date": "  ai.GL_DATE AS ACCOUNTING_DATE",
        }
        field_order = [
            "business_unit_name",
            "invoice_number",
            "invoice_date",
            "supplier_name",
            "supplier_number",
            "invoice_amount",
            "invoice_currency",
            "invoice_status",
            "accounting_date",
        ]
        select_lines = [select_map[key] for key in field_order if key in requested_fields]
        from_lines = ["FROM AP_INVOICES_ALL ai"]
        if {"supplier_name", "supplier_number"} & requested_fields:
            from_lines.extend(
                [
                    "JOIN POZ_SUPPLIERS ps",
                    "  ON ai.VENDOR_ID = ps.VENDOR_ID",
                ]
            )
        if "business_unit_name" in requested_fields:
            from_lines.extend(
                [
                    "JOIN FUN_ALL_BUSINESS_UNITS_V bu",
                    "  ON ai.ORG_ID = bu.BU_ID",
                ]
            )
        where_lines: List[str] = []
        if "invoice_date_between" in requested_filters:
            where_lines.append("ai.INVOICE_DATE BETWEEN :P_FROM_DATE AND :P_TO_DATE")
        if {"validated", "approved"} & requested_filters:
            where_lines.append("ai.APPROVAL_STATUS = :P_APPROVAL_STATUS")
        if "accounted" in requested_filters:
            where_lines.append(
                "EXISTS (\n"
                "    SELECT 1\n"
                "    FROM AP_INVOICE_DISTRIBUTIONS_ALL aid\n"
                "    WHERE aid.INVOICE_ID = ai.INVOICE_ID\n"
                "      AND aid.POSTED_FLAG = :P_ACCOUNTED_FLAG\n"
                "  )"
            )
        order_clause = self._sql_order_by_clause(
            request_shape,
            {"invoice_date": "ai.INVOICE_DATE", "invoice_number": "ai.INVOICE_NUM"},
        )
        sql_lines = ["SELECT", ",\n".join(select_lines), *from_lines]
        if where_lines:
            sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
        if order_clause:
            sql_lines.append(order_clause)
        return "\n".join(sql_lines) + ";"

    def _build_payables_invoice_distribution_sql(self, request_shape: Dict[str, Any]) -> str:
        requested_fields = self._sql_requested_field_keys(request_shape)
        requested_filters = self._sql_requested_filter_keys(request_shape)
        select_map = {
            "invoice_number": "  ai.INVOICE_NUM AS INVOICE_NUMBER",
            "supplier_name": "  COALESCE(ps.DF_COMPANY_NAME, ps.DF_LEGAL_NAME, ps.TAX_REPORTING_NAME) AS SUPPLIER_NAME",
            "distribution_line_number": "  aid.DISTRIBUTION_LINE_NUMBER AS DISTRIBUTION_LINE_NUMBER",
            "distribution_amount": "  aid.AMOUNT AS DISTRIBUTION_AMOUNT",
            "natural_account_segment": "  gcc_dist.SEGMENT4 AS NATURAL_ACCOUNT_SEGMENT",
            "cost_center": "  gcc_dist.SEGMENT5 AS COST_CENTER",
            "liability_account": "  gcc_liab.CONCATENATED_SEGMENTS AS LIABILITY_ACCOUNT",
            "accounting_date": "  aid.ACCOUNTING_DATE AS ACCOUNTING_DATE",
        }
        field_order = [
            "invoice_number",
            "supplier_name",
            "distribution_line_number",
            "distribution_amount",
            "natural_account_segment",
            "cost_center",
            "liability_account",
            "accounting_date",
        ]
        select_lines = [select_map[key] for key in field_order if key in requested_fields]
        sql_lines = [
            "SELECT",
            ",\n".join(select_lines),
            "FROM AP_INVOICE_DISTRIBUTIONS_ALL aid",
            "JOIN AP_INVOICES_ALL ai",
            "  ON aid.INVOICE_ID = ai.INVOICE_ID",
            "JOIN POZ_SUPPLIERS ps",
            "  ON ai.VENDOR_ID = ps.VENDOR_ID",
            "JOIN GL_CODE_COMBINATIONS gcc_dist",
            "  ON aid.DIST_CODE_COMBINATION_ID = gcc_dist.CODE_COMBINATION_ID",
            "JOIN GL_CODE_COMBINATIONS gcc_liab",
            "  ON ai.ACCTS_PAY_CODE_COMBINATION_ID = gcc_liab.CODE_COMBINATION_ID",
        ]
        where_lines: List[str] = []
        if {"validated", "approved"} & requested_filters:
            where_lines.append("ai.APPROVAL_STATUS IN ('APPROVED', 'VALIDATED')")
        if "accounted" in requested_filters:
            where_lines.append("aid.POSTED_FLAG = 'Y'")
        if where_lines:
            sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
        order_clause = self._sql_order_by_clause(
            request_shape,
            {"invoice_number": "ai.INVOICE_NUM", "distribution_line_number": "aid.DISTRIBUTION_LINE_NUMBER"},
        )
        if order_clause:
            sql_lines.append(order_clause)
        return "\n".join(sql_lines) + ";"

    def _build_payables_payments_sql(self, request_shape: Dict[str, Any]) -> str:
        requested_fields = self._sql_requested_field_keys(request_shape)
        requested_filters = self._sql_requested_filter_keys(request_shape)
        select_map = {
            "supplier_name": "  COALESCE(ps.DF_COMPANY_NAME, ps.DF_LEGAL_NAME, ps.TAX_REPORTING_NAME) AS SUPPLIER_NAME",
            "invoice_number": "  ai.INVOICE_NUM AS INVOICE_NUMBER",
            "payment_number": "  ac.CHECK_NUMBER AS PAYMENT_NUMBER",
            "payment_date": "  ac.CHECK_DATE AS PAYMENT_DATE",
            "payment_method": "  ac.PAYMENT_METHOD_LOOKUP_CODE AS PAYMENT_METHOD",
            "paid_amount": "  aip.AMOUNT AS PAID_AMOUNT",
            "bank_account_name": "  ac.BANK_ACCOUNT_NAME AS BANK_ACCOUNT_NAME",
        }
        field_order = [
            "supplier_name",
            "invoice_number",
            "payment_number",
            "payment_date",
            "payment_method",
            "paid_amount",
            "bank_account_name",
        ]
        select_lines = [select_map[key] for key in field_order if key in requested_fields]
        sql_lines = [
            "SELECT",
            ",\n".join(select_lines),
            "FROM AP_INVOICE_PAYMENTS_ALL aip",
            "JOIN AP_INVOICES_ALL ai",
            "  ON aip.INVOICE_ID = ai.INVOICE_ID",
            "JOIN AP_CHECKS_ALL ac",
            "  ON aip.CHECK_ID = ac.CHECK_ID",
            "JOIN POZ_SUPPLIERS ps",
            "  ON ac.VENDOR_ID = ps.VENDOR_ID",
        ]
        where_lines: List[str] = []
        if "payment_date_equals" in requested_filters:
            where_lines.append("ac.CHECK_DATE = :P_PAYMENT_DATE")
        if where_lines:
            sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
        order_clause = self._sql_order_by_clause(
            request_shape,
            {"payment_date": "ac.CHECK_DATE", "payment_number": "ac.CHECK_NUMBER"},
        )
        if order_clause:
            sql_lines.append(order_clause)
        return "\n".join(sql_lines) + ";"

    def _build_receivables_transaction_sql(self, request_shape: Dict[str, Any]) -> str:
        requested_fields = self._sql_requested_field_keys(request_shape)
        requested_filters = self._sql_requested_filter_keys(request_shape)
        select_map = {
            "customer_name": "  hp.PARTY_NAME AS CUSTOMER_NAME",
            "transaction_number": "  rct.TRX_NUMBER AS TRANSACTION_NUMBER",
            "transaction_date": "  rct.TRX_DATE AS TRANSACTION_DATE",
            "business_unit_name": "  bu.BU_NAME AS BUSINESS_UNIT_NAME",
            "amount_due_original": "  aps.AMOUNT_DUE_ORIGINAL AS AMOUNT_DUE_ORIGINAL",
            "remaining_amount": "  aps.AMOUNT_DUE_REMAINING AS REMAINING_AMOUNT",
            "due_date": "  aps.DUE_DATE AS DUE_DATE",
            "payment_status": "  aps.STATUS AS PAYMENT_STATUS",
        }
        field_order = [
            "customer_name",
            "transaction_number",
            "transaction_date",
            "business_unit_name",
            "amount_due_original",
            "remaining_amount",
            "due_date",
            "payment_status",
        ]
        select_lines = [select_map[key] for key in field_order if key in requested_fields]
        sql_lines = [
            "SELECT",
            ",\n".join(select_lines),
            "FROM RA_CUSTOMER_TRX_ALL rct",
            "JOIN AR_PAYMENT_SCHEDULES_ALL aps",
            "  ON aps.CUSTOMER_TRX_ID = rct.CUSTOMER_TRX_ID",
        ]
        if "customer_name" in requested_fields:
            sql_lines.extend(
                [
                    "JOIN HZ_CUST_ACCOUNTS_ hca",
                    "  ON rct.BILL_TO_CUSTOMER_ID = hca.CUST_ACCOUNT_ID",
                    "JOIN HZ_PARTIES hp",
                    "  ON hca.PARTY_ID = hp.PARTY_ID",
                ]
            )
        if "business_unit_name" in requested_fields:
            sql_lines.extend(
                [
                    "JOIN FUN_ALL_BUSINESS_UNITS_V bu",
                    "  ON rct.ORG_ID = bu.BU_ID",
                ]
            )
        where_lines: List[str] = []
        if "open_status" in requested_filters:
            where_lines.append("aps.STATUS = :P_OPEN_STATUS")
        if "complete" in requested_filters:
            where_lines.append("rct.COMPLETE_FLAG = :P_COMPLETE_FLAG")
        if "as_of_date" in requested_filters:
            where_lines.append("(aps.GL_DATE_CLOSED IS NULL OR aps.GL_DATE_CLOSED > :P_AS_OF_DATE)")
        if where_lines:
            sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
        order_clause = self._sql_order_by_clause(
            request_shape,
            {
                "transaction_date": "rct.TRX_DATE",
                "transaction_number": "rct.TRX_NUMBER",
                "customer_name": "hp.PARTY_NAME",
            },
        )
        if order_clause:
            sql_lines.append(order_clause)
        return "\n".join(sql_lines) + ";"

    def _build_receivables_receipts_applications_sql(self, request_shape: Dict[str, Any]) -> str:
        requested_fields = self._sql_requested_field_keys(request_shape)
        requested_filters = self._sql_requested_filter_keys(request_shape)
        select_map = {
            "customer_name": "  hp.PARTY_NAME AS CUSTOMER_NAME",
            "receipt_number": "  acr.RECEIPT_NUMBER AS RECEIPT_NUMBER",
            "receipt_date": "  acr.RECEIPT_DATE AS RECEIPT_DATE",
            "receipt_amount": "  acr.AMOUNT AS RECEIPT_AMOUNT",
            "transaction_number": "  rct.TRX_NUMBER AS TRANSACTION_NUMBER",
            "applied_amount": "  ara.AMOUNT_APPLIED AS APPLIED_AMOUNT",
            "receipt_status": "  acr.STATUS AS RECEIPT_STATUS",
        }
        field_order = [
            "customer_name",
            "receipt_number",
            "receipt_date",
            "receipt_amount",
            "transaction_number",
            "applied_amount",
            "receipt_status",
        ]
        select_lines = [select_map[key] for key in field_order if key in requested_fields]
        sql_lines = [
            "SELECT",
            ",\n".join(select_lines),
            "FROM AR_CASH_RECEIPTS_ALL acr",
            "JOIN AR_RECEIVABLE_APPLICATIONS_ALL ara",
            "  ON ara.CASH_RECEIPT_ID = acr.CASH_RECEIPT_ID",
            "JOIN RA_CUSTOMER_TRX_ALL rct",
            "  ON ara.APPLIED_CUSTOMER_TRX_ID = rct.CUSTOMER_TRX_ID",
            "JOIN HZ_CUST_ACCOUNTS_ hca",
            "  ON rct.BILL_TO_CUSTOMER_ID = hca.CUST_ACCOUNT_ID",
            "JOIN HZ_PARTIES hp",
            "  ON hca.PARTY_ID = hp.PARTY_ID",
        ]
        where_lines: List[str] = []
        if "applied" in requested_filters:
            where_lines.append("ara.DISPLAY = 'Y'")
        if "payment_date_equals" in requested_filters:
            where_lines.append("acr.RECEIPT_DATE = :P_PAYMENT_DATE")
        if where_lines:
            sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
        order_clause = self._sql_order_by_clause(
            request_shape,
            {
                "receipt_date": "acr.RECEIPT_DATE",
                "receipt_number": "acr.RECEIPT_NUMBER",
                "transaction_number": "rct.TRX_NUMBER",
            },
        )
        if order_clause:
            sql_lines.append(order_clause)
        return "\n".join(sql_lines) + ";"

    def _build_receivables_aging_sql(self, request_shape: Dict[str, Any]) -> str:
        requested_fields = self._sql_requested_field_keys(request_shape)
        requested_filters = self._sql_requested_filter_keys(request_shape)
        select_map = {
            "customer_name": "  hp.PARTY_NAME AS CUSTOMER_NAME",
            "transaction_number": "  rct.TRX_NUMBER AS TRANSACTION_NUMBER",
            "due_date": "  aps.DUE_DATE AS DUE_DATE",
            "remaining_amount": "  aps.AMOUNT_DUE_REMAINING AS REMAINING_AMOUNT",
            "aging_bucket": (
                "  CASE\n"
                "    WHEN aps.DUE_DATE >= :P_AS_OF_DATE THEN 'CURRENT'\n"
                "    WHEN :P_AS_OF_DATE - aps.DUE_DATE BETWEEN 1 AND 30 THEN '1_30'\n"
                "    WHEN :P_AS_OF_DATE - aps.DUE_DATE BETWEEN 31 AND 60 THEN '31_60'\n"
                "    WHEN :P_AS_OF_DATE - aps.DUE_DATE BETWEEN 61 AND 90 THEN '61_90'\n"
                "    ELSE 'OVER_90'\n"
                "  END AS AGING_BUCKET"
            ),
        }
        field_order = ["customer_name", "transaction_number", "due_date", "remaining_amount", "aging_bucket"]
        select_lines = [select_map[key] for key in field_order if key in requested_fields or key in {item.get("key") for item in request_shape.get("required_calculations") or []}]
        sql_lines = [
            "SELECT",
            ",\n".join(select_lines),
            "FROM RA_CUSTOMER_TRX_ALL rct",
            "JOIN AR_PAYMENT_SCHEDULES_ALL aps",
            "  ON aps.CUSTOMER_TRX_ID = rct.CUSTOMER_TRX_ID",
            "JOIN HZ_CUST_ACCOUNTS_ hca",
            "  ON rct.BILL_TO_CUSTOMER_ID = hca.CUST_ACCOUNT_ID",
            "JOIN HZ_PARTIES hp",
            "  ON hca.PARTY_ID = hp.PARTY_ID",
        ]
        where_lines: List[str] = []
        if "open_status" in requested_filters:
            where_lines.append("aps.STATUS = :P_OPEN_STATUS")
        if "as_of_date" in requested_filters:
            where_lines.append("(aps.GL_DATE_CLOSED IS NULL OR aps.GL_DATE_CLOSED > :P_AS_OF_DATE)")
        if where_lines:
            sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
        order_clause = self._sql_order_by_clause(
            request_shape,
            {"customer_name": "hp.PARTY_NAME", "transaction_number": "rct.TRX_NUMBER", "due_date": "aps.DUE_DATE"},
        )
        if order_clause:
            sql_lines.append(order_clause)
        return "\n".join(sql_lines) + ";"

    def _build_general_ledger_account_balances_sql(self, request_shape: Dict[str, Any]) -> str:
        requested_fields = self._sql_requested_field_keys(request_shape)
        requested_filters = self._sql_requested_filter_keys(request_shape)
        select_map = {
            "ledger_name": "  gl.NAME AS LEDGER_NAME",
            "period_name": "  gb.PERIOD_NAME AS PERIOD_NAME",
            "account_combination": "  gcc.CONCATENATED_SEGMENTS AS ACCOUNT_COMBINATION",
            "natural_account_segment": "  gcc.SEGMENT4 AS NATURAL_ACCOUNT_SEGMENT",
            "cost_center": "  gcc.SEGMENT5 AS COST_CENTER",
            "period_net_dr": "  gb.PERIOD_NET_DR AS PERIOD_NET_DR",
            "period_net_cr": "  gb.PERIOD_NET_CR AS PERIOD_NET_CR",
            "ending_balance": (
                "  ((NVL(gb.BEGIN_BALANCE_DR, 0) - NVL(gb.BEGIN_BALANCE_CR, 0)) "
                "+ (NVL(gb.PERIOD_NET_DR, 0) - NVL(gb.PERIOD_NET_CR, 0))) AS ENDING_BALANCE"
            ),
        }
        field_order = [
            "ledger_name",
            "period_name",
            "account_combination",
            "natural_account_segment",
            "cost_center",
            "period_net_dr",
            "period_net_cr",
            "ending_balance",
        ]
        select_lines = [select_map[key] for key in field_order if key in requested_fields or key in {item.get("key") for item in request_shape.get("required_calculations") or []}]
        sql_lines = [
            "SELECT",
            ",\n".join(select_lines),
            "FROM GL_BALANCES gb",
            "JOIN GL_CODE_COMBINATIONS gcc",
            "  ON gb.CODE_COMBINATION_ID = gcc.CODE_COMBINATION_ID",
            "JOIN GL_LEDGERS gl",
            "  ON gb.LEDGER_ID = gl.LEDGER_ID",
        ]
        where_lines: List[str] = []
        if "ledger_bind" in requested_filters:
            where_lines.append("gl.NAME = :P_LEDGER_NAME")
        if "period_bind" in requested_filters:
            where_lines.append("gb.PERIOD_NAME = :P_PERIOD_NAME")
        if "liability_account_type" in requested_filters:
            where_lines.append("gcc.ACCOUNT_TYPE = :P_ACCOUNT_TYPE")
        if where_lines:
            sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
        order_clause = self._sql_order_by_clause(
            request_shape,
            {"account_combination": "gcc.CONCATENATED_SEGMENTS", "period_name": "gb.PERIOD_NAME", "ledger_name": "gl.NAME"},
        )
        if order_clause:
            sql_lines.append(order_clause)
        return "\n".join(sql_lines) + ";"

    def _build_general_ledger_journal_details_sql(self, request_shape: Dict[str, Any]) -> str:
        requested_fields = self._sql_requested_field_keys(request_shape)
        requested_filters = self._sql_requested_filter_keys(request_shape)
        select_map = {
            "ledger_name": "  gl.NAME AS LEDGER_NAME",
            "journal_name": "  gjh.NAME AS JOURNAL_NAME",
            "period_name": "  gjh.PERIOD_NAME AS PERIOD_NAME",
            "journal_source": "  gjh.JE_SOURCE AS JOURNAL_SOURCE",
            "journal_category": "  gjh.JE_CATEGORY AS JOURNAL_CATEGORY",
            "journal_status": "  gjh.STATUS AS JOURNAL_STATUS",
            "journal_line_number": "  gjl.JE_LINE_NUM AS LINE_NUMBER",
            "account_combination": "  gcc.CONCATENATED_SEGMENTS AS ACCOUNT_COMBINATION",
            "debit_amount": "  gjl.ENTERED_DR AS DEBIT_AMOUNT",
            "credit_amount": "  gjl.ENTERED_CR AS CREDIT_AMOUNT",
        }
        field_order = [
            "ledger_name",
            "journal_name",
            "period_name",
            "journal_source",
            "journal_category",
            "journal_status",
            "journal_line_number",
            "account_combination",
            "debit_amount",
            "credit_amount",
        ]
        select_lines = [select_map[key] for key in field_order if key in requested_fields]
        sql_lines = [
            "SELECT",
            ",\n".join(select_lines),
            "FROM GL_JE_HEADERS gjh",
            "JOIN GL_JE_LINES gjl",
            "  ON gjl.JE_HEADER_ID = gjh.JE_HEADER_ID",
            "JOIN GL_CODE_COMBINATIONS gcc",
            "  ON gjl.CODE_COMBINATION_ID = gcc.CODE_COMBINATION_ID",
            "JOIN GL_LEDGERS gl",
            "  ON gjh.LEDGER_ID = gl.LEDGER_ID",
        ]
        where_lines: List[str] = []
        if "posted" in requested_filters:
            where_lines.append("gjh.STATUS = 'P'")
        if "ledger_bind" in requested_filters:
            where_lines.append("gl.NAME = :P_LEDGER_NAME")
        if "period_bind" in requested_filters:
            where_lines.append("gjh.PERIOD_NAME = :P_PERIOD_NAME")
        if where_lines:
            sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
        order_clause = self._sql_order_by_clause(
            request_shape,
            {"journal_name": "gjh.NAME", "period_name": "gjh.PERIOD_NAME", "journal_line_number": "gjl.JE_LINE_NUM"},
        )
        if order_clause:
            sql_lines.append(order_clause)
        return "\n".join(sql_lines) + ";"

    def _build_procurement_purchase_order_details_sql(self, request_shape: Dict[str, Any]) -> str:
        requested_fields = self._sql_requested_field_keys(request_shape)
        requested_filters = self._sql_requested_filter_keys(request_shape)
        select_map = {
            "po_number": "  ph.SEGMENT1 AS PO_NUMBER",
            "po_date": "  ph.CREATION_DATE AS PO_DATE",
            "po_status": "  ph.DOCUMENT_STATUS AS PO_STATUS",
            "po_type": "  ph.TYPE_LOOKUP_CODE AS PO_TYPE",
            "supplier_name": "  COALESCE(ps.DF_COMPANY_NAME, ps.DF_LEGAL_NAME, ps.TAX_REPORTING_NAME) AS SUPPLIER_NAME",
            "supplier_number": "  ps.SEGMENT1 AS SUPPLIER_NUMBER",
            "supplier_site_code": "  pssm.VENDOR_SITE_CODE AS SITE_CODE",
            "line_number": "  pl.LINE_NUM AS LINE_NUMBER",
            "item_description": "  pl.ITEM_DESCRIPTION AS ITEM_DESCRIPTION",
            "ordered_quantity": "  pl.QUANTITY AS ORDERED_QUANTITY",
            "unit_price": "  pl.UNIT_PRICE AS UNIT_PRICE",
        }
        field_order = [
            "po_number",
            "po_date",
            "po_status",
            "po_type",
            "supplier_name",
            "supplier_number",
            "supplier_site_code",
            "line_number",
            "item_description",
            "ordered_quantity",
            "unit_price",
        ]
        select_lines = [select_map[key] for key in field_order if key in requested_fields]
        sql_lines = [
            "SELECT",
            ",\n".join(select_lines),
            "FROM PO_HEADERS_ALL ph",
            "JOIN PO_LINES_ALL pl",
            "  ON pl.PO_HEADER_ID = ph.PO_HEADER_ID",
            "JOIN POZ_SUPPLIERS ps",
            "  ON ph.VENDOR_ID = ps.VENDOR_ID",
        ]
        if "supplier_site_code" in requested_fields:
            sql_lines.extend(
                [
                    "JOIN POZ_SUPPLIER_SITES_ALL_M pssm",
                    "  ON ph.VENDOR_SITE_ID = pssm.VENDOR_SITE_ID",
                ]
            )
        where_lines: List[str] = []
        if "complete" in requested_filters:
            where_lines.append("ph.DOCUMENT_STATUS = :P_COMPLETE_STATUS")
        if where_lines:
            sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
        order_clause = self._sql_order_by_clause(
            request_shape,
            {"po_number": "ph.SEGMENT1", "line_number": "pl.LINE_NUM", "po_date": "ph.CREATION_DATE"},
        )
        if order_clause:
            sql_lines.append(order_clause)
        return "\n".join(sql_lines) + ";"

    def _build_procurement_receiving_invoicing_match_sql(self, request_shape: Dict[str, Any]) -> str:
        requested_fields = self._sql_requested_field_keys(request_shape)
        requested_filters = self._sql_requested_filter_keys(request_shape)
        select_map = {
            "po_number": "  ph.SEGMENT1 AS PO_NUMBER",
            "supplier_name": "  COALESCE(ps.DF_COMPANY_NAME, ps.DF_LEGAL_NAME, ps.TAX_REPORTING_NAME) AS SUPPLIER_NAME",
            "line_number": "  pl.LINE_NUM AS LINE_NUMBER",
            "ordered_quantity": "  pl.QUANTITY AS ORDERED_QUANTITY",
            "received_quantity": "  pll.QUANTITY_RECEIVED AS RECEIVED_QUANTITY",
            "billed_quantity": "  pll.QUANTITY_BILLED AS BILLED_QUANTITY",
            "receipt_date": "  rcv.TRANSACTION_DATE AS RECEIPT_DATE",
        }
        field_order = [
            "po_number",
            "supplier_name",
            "line_number",
            "ordered_quantity",
            "received_quantity",
            "billed_quantity",
            "receipt_date",
        ]
        select_lines = [select_map[key] for key in field_order if key in requested_fields]
        sql_lines = [
            "SELECT",
            ",\n".join(select_lines),
            "FROM PO_HEADERS_ALL ph",
            "JOIN PO_LINES_ALL pl",
            "  ON pl.PO_HEADER_ID = ph.PO_HEADER_ID",
            "JOIN PO_LINE_LOCATIONS_ALL pll",
            "  ON pll.PO_LINE_ID = pl.PO_LINE_ID",
            "JOIN RCV_TRANSACTIONS rcv",
            "  ON rcv.PO_LINE_LOCATION_ID = pll.LINE_LOCATION_ID",
            "JOIN POZ_SUPPLIERS ps",
            "  ON ph.VENDOR_ID = ps.VENDOR_ID",
        ]
        where_lines: List[str] = []
        if "complete" in requested_filters:
            where_lines.append("ph.DOCUMENT_STATUS = :P_COMPLETE_STATUS")
        if where_lines:
            sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
        order_clause = self._sql_order_by_clause(
            request_shape,
            {"po_number": "ph.SEGMENT1", "line_number": "pl.LINE_NUM", "receipt_date": "rcv.TRANSACTION_DATE"},
        )
        if order_clause:
            sql_lines.append(order_clause)
        return "\n".join(sql_lines) + ";"

    def _build_demo_payables_sql(self, request_shape: Dict[str, Any]) -> Optional[str]:
        if request_shape.get("template_hint") not in {
            "payables_invoice_distribution_reporting",
            "payables_invoice_reporting",
        }:
            return None

        required_field_keys = {field["key"] for field in (request_shape.get("required_fields") or [])}
        include_distribution = (
            request_shape.get("template_hint") == "payables_invoice_distribution_reporting"
            or "accounting_date" in required_field_keys
            or "distribution_line_number" in required_field_keys
            or "distribution_amount" in required_field_keys
        )
        include_supplier = "supplier_name" in required_field_keys or "supplier_number" in required_field_keys
        include_business_unit = "business_unit_name" in required_field_keys
        include_liability = "liability_account" in required_field_keys
        include_natural = "natural_account_segment" in required_field_keys
        include_cost_center = "cost_center" in required_field_keys
        include_due_date = "due_date" in required_field_keys

        select_lines: List[str] = []
        from_lines: List[str] = []
        where_lines: List[str] = []

        from_lines.append("FROM AP_INVOICES_ALL ai")
        if include_distribution:
            from_lines = [
                "FROM AP_INVOICE_DISTRIBUTIONS_ALL aid",
                "JOIN AP_INVOICES_ALL ai",
                "  ON aid.INVOICE_ID = ai.INVOICE_ID",
            ]

        if include_due_date:
            from_lines.extend(
                [
                    "JOIN AP_PAYMENT_SCHEDULES_ALL aps",
                    "  ON aps.INVOICE_ID = ai.INVOICE_ID",
                ]
            )
        if include_supplier:
            from_lines.extend(
                [
                    "JOIN POZ_SUPPLIERS ps",
                    "  ON ai.VENDOR_ID = ps.VENDOR_ID",
                ]
            )
        if include_business_unit:
            from_lines.extend(
                [
                    "JOIN FUN_ALL_BUSINESS_UNITS_V bu",
                    "  ON ai.ORG_ID = bu.BU_ID",
                ]
            )
        if include_distribution and (include_natural or include_cost_center):
            from_lines.extend(
                [
                    "JOIN GL_CODE_COMBINATIONS gcc_dist",
                    "  ON aid.DIST_CODE_COMBINATION_ID = gcc_dist.CODE_COMBINATION_ID",
                ]
            )
        if include_liability:
            from_lines.extend(
                [
                    "JOIN GL_CODE_COMBINATIONS gcc_liab",
                    "  ON ai.ACCTS_PAY_CODE_COMBINATION_ID = gcc_liab.CODE_COMBINATION_ID",
                ]
            )

        field_order = [
            "invoice_number",
            "invoice_date",
            "due_date",
            "invoice_amount",
            "supplier_name",
            "supplier_number",
            "business_unit_name",
            "payment_status",
            "distribution_line_number",
            "distribution_amount",
            "accounting_date",
            "natural_account_segment",
            "cost_center",
            "liability_account",
        ]
        for field_key in field_order:
            if field_key not in required_field_keys:
                continue
            if field_key == "invoice_number":
                select_lines.append("  ai.INVOICE_NUM AS INVOICE_NUMBER")
            elif field_key == "invoice_date":
                select_lines.append("  ai.INVOICE_DATE AS INVOICE_DATE")
            elif field_key == "due_date":
                select_lines.append("  aps.DUE_DATE AS DUE_DATE")
            elif field_key == "invoice_amount":
                select_lines.append("  ai.INVOICE_AMOUNT AS INVOICE_AMOUNT")
            elif field_key == "supplier_name":
                select_lines.append(
                    "  COALESCE(ps.DF_COMPANY_NAME, ps.DF_LEGAL_NAME, ps.TAX_REPORTING_NAME) AS SUPPLIER_NAME"
                )
            elif field_key == "supplier_number":
                select_lines.append("  ps.SEGMENT1 AS SUPPLIER_NUMBER")
            elif field_key == "business_unit_name":
                select_lines.append("  bu.BU_NAME AS BUSINESS_UNIT_NAME")
            elif field_key == "payment_status":
                select_lines.append("  ai.PAYMENT_STATUS_FLAG AS PAYMENT_STATUS")
            elif field_key == "distribution_line_number":
                select_lines.append("  aid.DISTRIBUTION_LINE_NUMBER AS DISTRIBUTION_LINE_NUMBER")
            elif field_key == "distribution_amount":
                select_lines.append("  aid.AMOUNT AS DISTRIBUTION_AMOUNT")
            elif field_key == "accounting_date":
                if include_distribution:
                    select_lines.append("  aid.ACCOUNTING_DATE AS ACCOUNTING_DATE")
                else:
                    select_lines.append("  ai.ACCOUNTING_DATE AS ACCOUNTING_DATE")
            elif field_key == "natural_account_segment":
                select_lines.append("  gcc_dist.SEGMENT4 AS NATURAL_ACCOUNT_SEGMENT")
            elif field_key == "cost_center":
                select_lines.append("  gcc_dist.SEGMENT5 AS COST_CENTER")
            elif field_key == "liability_account":
                select_lines.append("  gcc_liab.CONCATENATED_SEGMENTS AS LIABILITY_ACCOUNT")

        if not select_lines:
            if include_distribution:
                select_lines = [
                    "  ai.INVOICE_NUM AS INVOICE_NUMBER",
                    "  ai.INVOICE_DATE AS INVOICE_DATE",
                    "  ai.INVOICE_AMOUNT AS INVOICE_AMOUNT",
                    "  aid.DISTRIBUTION_LINE_NUMBER AS DISTRIBUTION_LINE_NUMBER",
                    "  aid.AMOUNT AS DISTRIBUTION_AMOUNT",
                ]
            else:
                select_lines = [
                    "  ai.INVOICE_NUM AS INVOICE_NUMBER",
                    "  ai.INVOICE_DATE AS INVOICE_DATE",
                    "  ai.INVOICE_AMOUNT AS INVOICE_AMOUNT",
                ]

        requested_filters = {filter_spec["key"] for filter_spec in (request_shape.get("required_filters") or [])}
        if "validated" in requested_filters or "approved" in requested_filters:
            where_lines.append("ai.APPROVAL_STATUS = :P_APPROVAL_STATUS")
        if "accounted" in requested_filters:
            if include_distribution:
                where_lines.append("aid.POSTED_FLAG = :P_ACCOUNTED_FLAG")
            else:
                where_lines.append("ai.POSTING_STATUS = :P_ACCOUNTED_FLAG")
        if "unpaid" in requested_filters:
            where_lines.append("ai.PAYMENT_STATUS_FLAG = :P_UNPAID_STATUS")

        sql_lines = ["SELECT", ",\n".join(select_lines), *from_lines]
        if where_lines:
            sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
        return "\n".join(sql_lines) + ";"

    def _build_demo_receivables_sql(self, request_shape: Dict[str, Any], user_query: str) -> Optional[str]:
        template_hint = str(request_shape.get("template_hint") or "")
        if template_hint not in {
            "receivables_invoice_customer_reporting",
            "receivables_cash_receipt_reporting",
            "receivables_distribution_reporting",
        }:
            return None

        lowered_query = user_query.lower()
        required_field_keys = {field["key"] for field in (request_shape.get("required_fields") or [])}
        requested_filters = {filter_spec["key"] for filter_spec in (request_shape.get("required_filters") or [])}

        if template_hint == "receivables_cash_receipt_reporting":
            select_lines = []
            field_order = [
                "receipt_number",
                "receipt_date",
                "receipt_amount",
                "receipt_status",
                "customer_name",
                "business_unit_name",
            ]
            for field_key in field_order:
                if field_key not in required_field_keys:
                    continue
                if field_key == "receipt_number":
                    select_lines.append("  acr.RECEIPT_NUMBER AS RECEIPT_NUMBER")
                elif field_key == "receipt_date":
                    select_lines.append("  acr.RECEIPT_DATE AS RECEIPT_DATE")
                elif field_key == "receipt_amount":
                    select_lines.append("  acr.AMOUNT AS RECEIPT_AMOUNT")
                elif field_key == "receipt_status":
                    select_lines.append("  acr.STATUS AS RECEIPT_STATUS")
                elif field_key == "customer_name":
                    select_lines.append("  acr.CUSTOMER_DETAILS AS CUSTOMER_NAME")
                elif field_key == "business_unit_name":
                    select_lines.append("  bu.BU_NAME AS BUSINESS_UNIT_NAME")

            if not select_lines:
                select_lines = [
                    "  acr.RECEIPT_NUMBER AS RECEIPT_NUMBER",
                    "  acr.RECEIPT_DATE AS RECEIPT_DATE",
                    "  acr.AMOUNT AS RECEIPT_AMOUNT",
                    "  acr.STATUS AS RECEIPT_STATUS",
                ]

            from_lines = ["FROM AR_CASH_RECEIPTS_ALL acr"]
            if "business_unit_name" in required_field_keys:
                from_lines.extend(
                    [
                        "JOIN FUN_ALL_BUSINESS_UNITS_V bu",
                        "  ON acr.ORG_ID = bu.BU_ID",
                    ]
                )

            where_lines: List[str] = []
            if "complete" in requested_filters:
                where_lines.append("acr.STATUS = :P_COMPLETE_STATUS")
            sql_lines = ["SELECT", ",\n".join(select_lines), *from_lines]
            if where_lines:
                sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
            return "\n".join(sql_lines) + ";"

        if template_hint == "receivables_distribution_reporting":
            select_lines = []
            field_order = [
                "transaction_number",
                "account_combination",
                "natural_account_segment",
                "cost_center",
                "amount_due_original",
            ]
            for field_key in field_order:
                if field_key not in required_field_keys:
                    continue
                if field_key == "transaction_number":
                    select_lines.append("  rct.TRX_NUMBER AS TRANSACTION_NUMBER")
                elif field_key == "account_combination":
                    select_lines.append("  gcc.CONCATENATED_SEGMENTS AS ACCOUNT_COMBINATION")
                elif field_key == "natural_account_segment":
                    select_lines.append("  gcc.SEGMENT4 AS NATURAL_ACCOUNT_SEGMENT")
                elif field_key == "cost_center":
                    select_lines.append("  gcc.SEGMENT5 AS COST_CENTER")
                elif field_key == "amount_due_original":
                    select_lines.append("  rlgd.AMOUNT AS DISTRIBUTION_AMOUNT")

            if not select_lines:
                select_lines = [
                    "  rct.TRX_NUMBER AS TRANSACTION_NUMBER",
                    "  gcc.CONCATENATED_SEGMENTS AS ACCOUNT_COMBINATION",
                    "  gcc.SEGMENT4 AS NATURAL_ACCOUNT_SEGMENT",
                ]

            where_lines: List[str] = []
            if "accounted" in requested_filters:
                where_lines.append("rlgd.ACCTD_AMOUNT = :P_ACCOUNTED_FLAG")

            sql_lines = [
                "SELECT",
                ",\n".join(select_lines),
                "FROM RA_CUST_TRX_LINE_GL_DIST_ALL rlgd",
                "JOIN RA_CUSTOMER_TRX_ALL rct",
                "  ON rlgd.CUSTOMER_TRX_ID = rct.CUSTOMER_TRX_ID",
                "JOIN GL_CODE_COMBINATIONS gcc",
                "  ON rlgd.CODE_COMBINATION_ID = gcc.CODE_COMBINATION_ID",
            ]
            if where_lines:
                sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
            return "\n".join(sql_lines) + ";"

        include_schedule = any(
            token in lowered_query
            for token in ("due date", "remaining amount", "outstanding", "open status", "status is open")
        ) or any(filter_spec.get("key") == "open_status" for filter_spec in (request_shape.get("required_filters") or []))
        include_customer_name = "customer_name" in required_field_keys
        include_business_unit = "business_unit_name" in required_field_keys

        select_lines = []
        field_order = [
            "invoice_number",
            "transaction_number",
            "invoice_date",
            "transaction_date",
            "customer_name",
            "business_unit_name",
            "amount_due_original",
            "remaining_amount",
            "due_date",
            "payment_status",
        ]
        for field_key in field_order:
            if field_key not in required_field_keys:
                continue
            if field_key in {"invoice_number", "transaction_number"}:
                select_lines.append("  rct.TRX_NUMBER AS TRANSACTION_NUMBER")
            elif field_key in {"invoice_date", "transaction_date"}:
                select_lines.append("  rct.TRX_DATE AS TRANSACTION_DATE")
            elif field_key == "customer_name":
                select_lines.append("  hp.PARTY_NAME AS CUSTOMER_NAME")
            elif field_key == "business_unit_name":
                select_lines.append("  bu.BU_NAME AS BUSINESS_UNIT_NAME")
            elif field_key == "amount_due_original":
                select_lines.append("  aps.AMOUNT_DUE_ORIGINAL AS AMOUNT_DUE_ORIGINAL")
            elif field_key == "remaining_amount":
                select_lines.append("  aps.AMOUNT_DUE_REMAINING AS REMAINING_AMOUNT")
            elif field_key == "due_date":
                select_lines.append("  aps.DUE_DATE AS DUE_DATE")
            elif field_key == "payment_status":
                select_lines.append("  aps.STATUS AS PAYMENT_STATUS")

        if not select_lines:
            select_lines = [
                "  rct.TRX_NUMBER AS TRANSACTION_NUMBER",
                "  rct.TRX_DATE AS TRANSACTION_DATE",
                "  aps.AMOUNT_DUE_ORIGINAL AS AMOUNT_DUE_ORIGINAL",
                "  aps.AMOUNT_DUE_REMAINING AS REMAINING_AMOUNT",
            ]

        from_lines = ["FROM RA_CUSTOMER_TRX_ALL rct"]
        if include_customer_name:
            from_lines.extend(
                [
                    "JOIN HZ_CUST_ACCOUNTS_ hca",
                    "  ON rct.BILL_TO_CUSTOMER_ID = hca.CUST_ACCOUNT_ID",
                    "JOIN HZ_PARTIES hp",
                    "  ON hca.PARTY_ID = hp.PARTY_ID",
                ]
            )
        if include_business_unit:
            from_lines.extend(
                [
                    "JOIN FUN_ALL_BUSINESS_UNITS_V bu",
                    "  ON rct.ORG_ID = bu.BU_ID",
                ]
            )
        if include_schedule or any(
            key in required_field_keys for key in {"amount_due_original", "remaining_amount", "due_date", "payment_status"}
        ):
            from_lines.extend(
                [
                    "JOIN AR_PAYMENT_SCHEDULES_ALL aps",
                    "  ON aps.CUSTOMER_TRX_ID = rct.CUSTOMER_TRX_ID",
                ]
            )

        where_lines: List[str] = []
        if "open_status" in requested_filters:
            where_lines.append("aps.STATUS = :P_OPEN_STATUS")
        if "complete" in requested_filters:
            where_lines.append("rct.COMPLETE_FLAG = :P_COMPLETE_FLAG")

        sql_lines = ["SELECT", ",\n".join(select_lines), *from_lines]
        if where_lines:
            sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
        return "\n".join(sql_lines) + ";"

    def _build_demo_general_ledger_sql(self, request_shape: Dict[str, Any], user_query: str) -> Optional[str]:
        template_hint = str(request_shape.get("template_hint") or "")
        if template_hint not in {
            "general_ledger_journal_reporting",
            "general_ledger_balance_reporting",
            "general_ledger_code_combination_reporting",
        }:
            return None

        lowered_query = user_query.lower()
        required_field_keys = {field["key"] for field in (request_shape.get("required_fields") or [])}
        requested_filters = {filter_spec["key"] for filter_spec in (request_shape.get("required_filters") or [])}

        if template_hint == "general_ledger_code_combination_reporting":
            select_lines: List[str] = []
            field_order = [
                "code_combination_id",
                "account_combination",
                "natural_account_segment",
                "cost_center",
            ]
            for field_key in field_order:
                if field_key not in required_field_keys:
                    continue
                if field_key == "code_combination_id":
                    select_lines.append("  gcc.CODE_COMBINATION_ID AS CODE_COMBINATION_ID")
                elif field_key == "account_combination":
                    select_lines.append("  gcc.CONCATENATED_SEGMENTS AS ACCOUNT_COMBINATION")
                elif field_key == "natural_account_segment":
                    select_lines.append("  gcc.SEGMENT4 AS NATURAL_ACCOUNT_SEGMENT")
                elif field_key == "cost_center":
                    select_lines.append("  gcc.SEGMENT5 AS COST_CENTER")

            if not select_lines:
                select_lines = [
                    "  gcc.CODE_COMBINATION_ID AS CODE_COMBINATION_ID",
                    "  gcc.CONCATENATED_SEGMENTS AS ACCOUNT_COMBINATION",
                    "  gcc.SEGMENT4 AS NATURAL_ACCOUNT_SEGMENT",
                    "  gcc.SEGMENT5 AS COST_CENTER",
                ]

            where_lines: List[str] = []
            if "liability_account_type" in requested_filters or "liability" in lowered_query:
                where_lines.append("gcc.ACCOUNT_TYPE = :P_ACCOUNT_TYPE")

            sql_lines = [
                "SELECT",
                ",\n".join(select_lines),
                "FROM GL_CODE_COMBINATIONS gcc",
            ]
            if where_lines:
                sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
            return "\n".join(sql_lines) + ";"

        if template_hint == "general_ledger_balance_reporting":
            select_lines: List[str] = []
            field_order = [
                "ledger_name",
                "period_name",
                "account_combination",
                "natural_account_segment",
                "cost_center",
                "period_net_dr",
                "period_net_cr",
            ]
            for field_key in field_order:
                if field_key not in required_field_keys:
                    continue
                if field_key == "ledger_name":
                    select_lines.append("  gl.NAME AS LEDGER_NAME")
                elif field_key == "period_name":
                    select_lines.append("  gb.PERIOD_NAME AS PERIOD_NAME")
                elif field_key == "account_combination":
                    select_lines.append("  gcc.CONCATENATED_SEGMENTS AS ACCOUNT_COMBINATION")
                elif field_key == "natural_account_segment":
                    select_lines.append("  gcc.SEGMENT4 AS NATURAL_ACCOUNT_SEGMENT")
                elif field_key == "cost_center":
                    select_lines.append("  gcc.SEGMENT5 AS COST_CENTER")
                elif field_key == "period_net_dr":
                    select_lines.append("  gb.PERIOD_NET_DR AS PERIOD_NET_DR")
                elif field_key == "period_net_cr":
                    select_lines.append("  gb.PERIOD_NET_CR AS PERIOD_NET_CR")

            if not select_lines:
                select_lines = [
                    "  gl.NAME AS LEDGER_NAME",
                    "  gb.PERIOD_NAME AS PERIOD_NAME",
                    "  gcc.CONCATENATED_SEGMENTS AS ACCOUNT_COMBINATION",
                    "  gb.PERIOD_NET_DR AS PERIOD_NET_DR",
                    "  gb.PERIOD_NET_CR AS PERIOD_NET_CR",
                ]

            where_lines: List[str] = []
            if "period_name" in required_field_keys or "period filter" in lowered_query:
                where_lines.append("gb.PERIOD_NAME = :P_PERIOD_NAME")
            if "liability" in lowered_query:
                where_lines.append("gcc.ACCOUNT_TYPE = :P_ACCOUNT_TYPE")

            sql_lines = [
                "SELECT",
                ",\n".join(select_lines),
                "FROM GL_BALANCES gb",
                "JOIN GL_CODE_COMBINATIONS gcc",
                "  ON gb.CODE_COMBINATION_ID = gcc.CODE_COMBINATION_ID",
                "JOIN GL_LEDGERS gl",
                "  ON gb.LEDGER_ID = gl.LEDGER_ID",
            ]
            if where_lines:
                sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
            return "\n".join(sql_lines) + ";"

        select_lines: List[str] = []
        field_order = [
            "journal_name",
            "period_name",
            "journal_source",
            "journal_category",
            "journal_status",
            "journal_line_number",
            "account_combination",
            "natural_account_segment",
            "cost_center",
            "debit_amount",
            "credit_amount",
        ]
        for field_key in field_order:
            if field_key not in required_field_keys:
                continue
            if field_key == "journal_name":
                select_lines.append("  gjh.NAME AS JOURNAL_NAME")
            elif field_key == "period_name":
                select_lines.append("  gjh.PERIOD_NAME AS PERIOD_NAME")
            elif field_key == "journal_source":
                select_lines.append("  gjh.JE_SOURCE AS JOURNAL_SOURCE")
            elif field_key == "journal_category":
                select_lines.append("  gjh.JE_CATEGORY AS JOURNAL_CATEGORY")
            elif field_key == "journal_status":
                select_lines.append("  gjh.STATUS AS JOURNAL_STATUS")
            elif field_key == "journal_line_number":
                select_lines.append("  gjl.JE_LINE_NUM AS LINE_NUMBER")
            elif field_key == "account_combination":
                select_lines.append("  gcc.CONCATENATED_SEGMENTS AS ACCOUNT_COMBINATION")
            elif field_key == "natural_account_segment":
                select_lines.append("  gcc.SEGMENT4 AS NATURAL_ACCOUNT_SEGMENT")
            elif field_key == "cost_center":
                select_lines.append("  gcc.SEGMENT5 AS COST_CENTER")
            elif field_key == "debit_amount":
                select_lines.append("  gjl.ENTERED_DR AS DEBIT_AMOUNT")
            elif field_key == "credit_amount":
                select_lines.append("  gjl.ENTERED_CR AS CREDIT_AMOUNT")

        if not select_lines:
            select_lines = [
                "  gjh.NAME AS JOURNAL_NAME",
                "  gjh.PERIOD_NAME AS PERIOD_NAME",
                "  gjl.JE_LINE_NUM AS LINE_NUMBER",
                "  gcc.CONCATENATED_SEGMENTS AS ACCOUNT_COMBINATION",
                "  gjl.ENTERED_DR AS DEBIT_AMOUNT",
                "  gjl.ENTERED_CR AS CREDIT_AMOUNT",
            ]

        where_lines: List[str] = []
        if "accounted" in requested_filters or "posted_journal" in requested_filters or "posted" in lowered_query:
            where_lines.append("gjh.STATUS = :P_JOURNAL_STATUS")
        if "current period" in lowered_query or "period filter" in lowered_query:
            where_lines.append("gjh.PERIOD_NAME = :P_PERIOD_NAME")

        sql_lines = [
            "SELECT",
            ",\n".join(select_lines),
            "FROM GL_JE_HEADERS gjh",
            "JOIN GL_JE_LINES gjl",
            "  ON gjl.JE_HEADER_ID = gjh.JE_HEADER_ID",
            "JOIN GL_CODE_COMBINATIONS gcc",
            "  ON gjl.CODE_COMBINATION_ID = gcc.CODE_COMBINATION_ID",
        ]
        if where_lines:
            sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
        return "\n".join(sql_lines) + ";"

    def _build_demo_cash_management_sql(self, request_shape: Dict[str, Any]) -> Optional[str]:
        template_hint = str(request_shape.get("template_hint") or "")
        if template_hint not in {
            "cash_management_bank_account_reporting",
            "cash_management_statement_line_reporting",
            "cash_management_external_transaction_reporting",
        }:
            return None

        required_field_keys = {field["key"] for field in (request_shape.get("required_fields") or [])}
        requested_filters = {filter_spec["key"] for filter_spec in (request_shape.get("required_filters") or [])}

        if template_hint == "cash_management_statement_line_reporting":
            select_lines: List[str] = []
            field_order = [
                "statement_number",
                "statement_date",
                "statement_line_number",
                "statement_amount",
                "bank_account_name",
            ]
            for field_key in field_order:
                if field_key not in required_field_keys:
                    continue
                if field_key == "statement_number":
                    select_lines.append("  csh.STATEMENT_NUMBER AS STATEMENT_NUMBER")
                elif field_key == "statement_date":
                    select_lines.append("  csh.STATEMENT_DATE AS STATEMENT_DATE")
                elif field_key == "statement_line_number":
                    select_lines.append("  csl.LINE_NUMBER AS STATEMENT_LINE_NUMBER")
                elif field_key == "statement_amount":
                    select_lines.append("  csl.AMOUNT AS STATEMENT_AMOUNT")
                elif field_key == "bank_account_name":
                    select_lines.append("  cba.BANK_ACCOUNT_NAME AS BANK_ACCOUNT_NAME")

            if not select_lines:
                select_lines = [
                    "  csh.STATEMENT_NUMBER AS STATEMENT_NUMBER",
                    "  csh.STATEMENT_DATE AS STATEMENT_DATE",
                    "  csl.LINE_NUMBER AS STATEMENT_LINE_NUMBER",
                    "  csl.AMOUNT AS STATEMENT_AMOUNT",
                    "  cba.BANK_ACCOUNT_NAME AS BANK_ACCOUNT_NAME",
                ]

            where_lines: List[str] = []
            if "unreconciled" in requested_filters:
                where_lines.append("csl.RECON_STATUS = :P_RECON_STATUS")

            sql_lines = [
                "SELECT",
                ",\n".join(select_lines),
                "FROM CE_STATEMENT_HEADERS csh",
                "JOIN CE_STATEMENT_LINES csl",
                "  ON csl.STATEMENT_HEADER_ID = csh.STATEMENT_HEADER_ID",
                "JOIN CE_BANK_ACCOUNTS cba",
                "  ON csh.BANK_ACCOUNT_ID = cba.BANK_ACCOUNT_ID",
            ]
            if where_lines:
                sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
            return "\n".join(sql_lines) + ";"

        if template_hint == "cash_management_external_transaction_reporting":
            select_lines = [
                "  cet.EXTERNAL_TRANSACTION_ID AS EXTERNAL_TRANSACTION_ID",
                "  cet.TRANSACTION_DATE AS TRANSACTION_DATE",
                "  cba.BANK_ACCOUNT_NAME AS BANK_ACCOUNT_NAME",
                "  cet.AMOUNT AS TRANSACTION_AMOUNT",
                "  cet.STATUS AS TRANSACTION_STATUS",
            ]
            where_lines: List[str] = []
            if "unreconciled" in requested_filters:
                where_lines.append("cet.ACCOUNTING_FLAG = :P_ACCOUNTING_FLAG")

            sql_lines = [
                "SELECT",
                ",\n".join(select_lines),
                "FROM CE_EXTERNAL_TRANSACTIONS cet",
                "JOIN CE_BANK_ACCOUNTS cba",
                "  ON cet.BANK_ACCOUNT_ID = cba.BANK_ACCOUNT_ID",
            ]
            if where_lines:
                sql_lines.extend(["WHERE " + where_lines[0], *[f"  AND {item}" for item in where_lines[1:]]])
            return "\n".join(sql_lines) + ";"

        include_legal_entity = "legal_entity_name" in required_field_keys or "XLE_ENTITY_PROFILES" in set(
            request_shape.get("required_tables") or []
        )

        select_lines: List[str] = []
        field_order = [
            "bank_account_id",
            "bank_account_name",
            "account_classification",
            "account_owner_org_id",
            "legal_entity_name",
        ]
        for field_key in field_order:
            if field_key not in required_field_keys:
                continue
            if field_key == "bank_account_id":
                select_lines.append("  cba.BANK_ACCOUNT_ID AS BANK_ACCOUNT_ID")
            elif field_key == "bank_account_name":
                select_lines.append("  cba.BANK_ACCOUNT_NAME AS BANK_ACCOUNT_NAME")
            elif field_key == "account_classification":
                select_lines.append("  cba.ACCOUNT_CLASSIFICATION AS ACCOUNT_CLASSIFICATION")
            elif field_key == "account_owner_org_id":
                select_lines.append("  cba.ACCOUNT_OWNER_ORG_ID AS ACCOUNT_OWNER_ORG_ID")
            elif field_key == "legal_entity_name":
                select_lines.append("  xep.NAME AS LEGAL_ENTITY_NAME")

        if not select_lines:
            select_lines = [
                "  cba.BANK_ACCOUNT_ID AS BANK_ACCOUNT_ID",
                "  cba.BANK_ACCOUNT_NAME AS BANK_ACCOUNT_NAME",
                "  cba.ACCOUNT_CLASSIFICATION AS ACCOUNT_CLASSIFICATION",
                "  cba.ACCOUNT_OWNER_ORG_ID AS ACCOUNT_OWNER_ORG_ID",
            ]

        sql_lines = [
            "SELECT",
            ",\n".join(select_lines),
            "FROM CE_BANK_ACCOUNTS cba",
        ]
        if include_legal_entity:
            sql_lines.extend(
                [
                    "JOIN XLE_ENTITY_PROFILES xep",
                    "  ON cba.ACCOUNT_OWNER_ORG_ID = xep.LEGAL_ENTITY_ID",
                ]
            )
        return "\n".join(sql_lines) + ";"

    def _build_demo_procurement_sql(self, request_shape: Dict[str, Any]) -> Optional[str]:
        if request_shape.get("template_hint") != "procurement_supplier_site_reporting":
            return None

        required_field_keys = {field["key"] for field in (request_shape.get("required_fields") or [])}
        select_lines: List[str] = []
        field_order = [
            "supplier_name",
            "supplier_number",
            "supplier_site_code",
            "business_unit_name",
        ]
        for field_key in field_order:
            if field_key not in required_field_keys:
                continue
            if field_key == "supplier_name":
                select_lines.append(
                    "  COALESCE(ps.DF_COMPANY_NAME, ps.DF_LEGAL_NAME, ps.TAX_REPORTING_NAME) AS SUPPLIER_NAME"
                )
            elif field_key == "supplier_number":
                select_lines.append("  ps.SEGMENT1 AS SUPPLIER_NUMBER")
            elif field_key == "supplier_site_code":
                select_lines.append("  pssm.VENDOR_SITE_CODE AS SITE_CODE")
            elif field_key == "business_unit_name":
                select_lines.append("  bu.BU_NAME AS BUSINESS_UNIT_NAME")

        if not select_lines:
            select_lines = [
                "  COALESCE(ps.DF_COMPANY_NAME, ps.DF_LEGAL_NAME, ps.TAX_REPORTING_NAME) AS SUPPLIER_NAME",
                "  ps.SEGMENT1 AS SUPPLIER_NUMBER",
                "  pssm.VENDOR_SITE_CODE AS SITE_CODE",
                "  bu.BU_NAME AS BUSINESS_UNIT_NAME",
            ]

        sql_lines = [
            "SELECT",
            ",\n".join(select_lines),
            "FROM POZ_SUPPLIERS ps",
            "JOIN POZ_SUPPLIER_SITES_ALL_M pssm",
            "  ON ps.VENDOR_ID = pssm.VENDOR_ID",
            "JOIN FUN_ALL_BUSINESS_UNITS_V bu",
            "  ON pssm.PRC_BU_ID = bu.BU_ID",
        ]
        return "\n".join(sql_lines) + ";"

    def _select_best_sql_chunk(
        self,
        user_query: str,
        route_info: Any,
        chunks: List[Dict[str, Any]],
        request_shape: Optional[Dict[str, Any]],
        module_alignment_target: Any,
    ) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
        sql_chunks = [
            chunk
            for chunk in chunks
            if str((chunk.get("metadata") or {}).get("corpus") or "") == "sql_examples_corpus"
        ]
        ranked: List[Tuple[float, Dict[str, Any], str]] = []
        rejection_reason: Optional[str] = None

        for chunk in sql_chunks:
            sql_text = str(chunk.get("content") or "")
            if not sql_text:
                continue
            metadata = chunk.get("metadata") or {}
            score = float(chunk.get("rerank_score") or chunk.get("combined_score") or 0.0)
            normalized_for_score = self._normalize_sql_output_block(sql_text)
            score += self._score_sql_pattern_for_request(normalized_for_score, metadata, request_shape)
            ranked.append((score, chunk, sql_text))

        ranked.sort(key=lambda item: item[0], reverse=True)
        for _score, chunk, raw_sql_text in ranked:
            for sql_text in self._sql_candidate_variants(raw_sql_text, request_shape):
                ok, reason = self._verify_sql_candidate(
                    sql_text,
                    module_alignment_target=module_alignment_target,
                    request_shape=request_shape,
                )
                if ok:
                    return chunk, sql_text, None
                rejection_reason = reason

        return None, "", rejection_reason

    def _match_specialized_record(
        self,
        user_query: str,
        route_info: Any,
        records: List[Dict[str, Any]],
        *,
        title_fields: List[str],
    ) -> Dict[str, Any] | None:
        lowered_query = user_query.lower()
        query_tokens = {token for token in re.findall(r"[a-z0-9_]+", lowered_query) if len(token) > 2}
        query_identifiers = self._specialized_query_identifiers(user_query)
        request_shape = self._parse_sql_request_shape(user_query, route_info) if route_info.task_type in self.SQL_TASKS else None
        formula_shape = self._parse_fast_formula_request_shape(user_query, route_info) if route_info.task_type in self.FAST_FORMULA_TASKS else None
        broken_formula_normalized = self._normalize_formula_block_text(
            str((formula_shape or {}).get("broken_formula") or "")
        )
        best_record = None
        best_score = 0.0

        for record in records:
            record_module = str(record.get("module") or "")
            if not self._specialized_module_compatible(route_info, record_module):
                continue

            score = 0.0
            for field_name in title_fields:
                field_value = str(record.get(field_name) or "").strip()
                if not field_value:
                    continue
                lowered_field = field_value.lower()
                if lowered_field and lowered_field in lowered_query:
                    score = max(score, 10.0 + len(lowered_field) / 1000.0)
                    continue

                field_tokens = {token for token in re.findall(r"[a-z0-9_]+", lowered_field) if len(token) > 2}
                if field_tokens and query_tokens:
                    overlap = len(field_tokens & query_tokens) / max(len(field_tokens), 1)
                    score = max(score, overlap)

            if route_info.task_type in self.SQL_TASKS:
                score += self._specialized_overlap_score(
                    query_identifiers,
                    {str(item).upper() for item in (record.get("tables_used") or [])},
                ) * 1.2
                score += self._specialized_overlap_score(
                    query_identifiers,
                    {str(item).upper() for item in (record.get("columns_used") or [])},
                ) * 0.6
                score += self._score_sql_pattern_for_request(
                    str(record.get("content") or ""),
                    record,
                    request_shape,
                )
                if request_shape and request_shape.get("needs_join") and len(record.get("tables_used") or []) <= 1:
                    score -= 8.0

            if route_info.task_type in self.FAST_FORMULA_TASKS:
                content_tokens = self._specialized_query_tokens(str(record.get("content") or "")[:1600])
                formula_type_tokens = self._specialized_query_tokens(str(record.get("formula_type") or ""))
                formula_name_tokens = self._specialized_query_tokens(str(record.get("formula_name") or ""))
                use_case_tokens = self._specialized_query_tokens(str(record.get("use_case") or ""))
                score += self._specialized_overlap_score(query_tokens, content_tokens) * 1.2
                score += self._specialized_overlap_score(query_tokens, formula_type_tokens) * 2.1
                score += self._specialized_overlap_score(query_tokens, formula_name_tokens) * 2.0
                score += self._specialized_overlap_score(query_tokens, use_case_tokens) * 1.4
                score += self._specialized_overlap_score(
                    query_identifiers,
                    {str(item).upper() for item in (record.get("database_items") or [])},
                ) * 2.1
                score += self._specialized_overlap_score(
                    query_identifiers,
                    {str(item).upper() for item in (record.get("contexts") or [])},
                ) * 2.0
                score += self._specialized_overlap_score(
                    query_identifiers,
                    {str(item).upper() for item in (record.get("functions") or [])},
                ) * 1.6
                if str(record.get("doc_type") or "") == "fast_formula_example":
                    score += 0.4 if not bool(record.get("derived_from_doc")) else 0.2
                else:
                    score -= 1.0
                if formula_shape:
                    requested_type = self._normalize_formula_type(str(formula_shape.get("requested_formula_type") or "UNKNOWN"))
                    record_type = self._normalize_formula_type(str(record.get("formula_type") or "UNKNOWN"))
                    if requested_type.upper() != "UNKNOWN":
                        if requested_type == record_type:
                            score += 4.5
                        elif requested_type.lower() in record_type.lower() or record_type.lower() in requested_type.lower():
                            score += 2.2
                        else:
                            score -= 3.0
                    broken_tokens = set(formula_shape.get("broken_formula_tokens") or [])
                    if broken_tokens:
                        score += self._specialized_overlap_score(broken_tokens, content_tokens) * 3.0
                    if formula_shape.get("is_troubleshooting") and broken_formula_normalized:
                        record_content_normalized = self._normalize_formula_block_text(str(record.get("content") or "")[:2500])
                        if broken_formula_normalized and record_content_normalized:
                            if broken_formula_normalized in record_content_normalized:
                                score += 10.0
                            else:
                                broken_overlap = self._specialized_overlap_score(
                                    set(broken_formula_normalized.split()),
                                    set(record_content_normalized.split()),
                                )
                                score += broken_overlap * 2.0
                    requested_formula_name = str(formula_shape.get("formula_name_normalized") or "")
                    record_formula_name = self._normalize_formula_name(
                        str(record.get("formula_name") or record.get("title") or "")
                    )
                    record_source = str(record.get("source_uri") or record.get("source_file") or "").lower()
                    if requested_formula_name:
                        if record_formula_name == requested_formula_name:
                            score += 8.0
                        elif requested_formula_name in record_formula_name or record_formula_name in requested_formula_name:
                            score += 4.5
                        elif "formula_types.json" in record_source:
                            score -= 4.0

            if score > best_score:
                best_score = score
                best_record = record

        threshold = 0.55
        if route_info.task_type in self.SQL_TASKS:
            threshold = 0.3
        if route_info.task_type in self.FAST_FORMULA_TASKS:
            threshold = 0.45
        if best_score >= threshold:
            return best_record
        return None

    def _augment_specialized_chunks(
        self,
        user_query: str,
        route_info: Any,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if route_info.task_type in self.DOCS_EXPECTED_TASKS:
            doc_catalog = self._load_local_doc_catalog()
            doc_record = self._match_summary_concept_seed(user_query, route_info, doc_catalog)
            if not doc_record:
                doc_record = self._match_local_doc_record(user_query, route_info, doc_catalog)
            if not doc_record:
                doc_record = self._task_driven_local_doc_fallback(user_query, route_info, doc_catalog)
            if doc_record:
                metadata = dict(doc_record.get("metadata") or {})
                record_task_type = str(doc_record.get("task_type") or metadata.get("task_type") or "").lower().strip()
                corpus = str(doc_record.get("corpus") or metadata.get("corpus") or "").strip()
                if not corpus:
                    corpus = "troubleshooting_corpus" if record_task_type == "troubleshooting" else "docs_corpus"
                if not self._specialized_chunk_present(
                    chunks,
                    doc_record,
                    corpus=corpus,
                ):
                    chunk = self._build_specialized_catalog_chunk(
                        doc_record,
                        corpus=corpus,
                        doc_type=str(doc_record.get("doc_type") or metadata.get("doc_type") or "functional_doc"),
                    )
                    return [chunk] + chunks
            return chunks

        if route_info.task_type not in (self.SQL_TASKS | self.FAST_FORMULA_TASKS):
            return chunks

        catalog = self._load_specialization_catalog()
        request_shape = self._parse_sql_request_shape(user_query, route_info) if route_info.task_type in self.SQL_TASKS else None

        if route_info.task_type in self.SQL_TASKS:
            record = self._match_specialized_record(
                user_query,
                route_info,
                catalog.get("sql_records") or [],
                title_fields=["title", "source_file"],
            )
            if not record:
                module_hint = self._infer_sql_module_hint(user_query)
                if module_hint:
                    hint_families = module_families_for_value(module_hint)
                    hint_families.discard(ModuleFamily.UNKNOWN.value)
                    hinted_route = SimpleNamespace(
                        task_type=route_info.task_type,
                        module=FusionModule.UNKNOWN,
                        module_family=ModuleFamily.UNKNOWN,
                    )
                    if hint_families:
                        hinted_family = next(iter(hint_families))
                        if hinted_family in {member.value for member in ModuleFamily}:
                            hinted_route.module_family = ModuleFamily(hinted_family)
                    hinted_records = [
                        item
                        for item in (catalog.get("sql_records") or [])
                        if not module_hint
                        or str(item.get("module") or "").strip().lower() == module_hint.lower()
                        or bool(module_families_for_value(str(item.get("module") or "")) & hint_families)
                    ]
                    record = self._match_specialized_record(
                        user_query,
                        hinted_route,
                        hinted_records,
                        title_fields=["title", "source_file"],
                    )
            if (
                record
                and not self._specialized_chunk_present(chunks, record, corpus="sql_examples_corpus")
            ):
                chunk = self._build_specialized_catalog_chunk(
                    record,
                    corpus="sql_examples_corpus",
                    doc_type="sql_example",
                )
                chunk["rerank_score"] = self._score_sql_pattern_for_request(
                    str(record.get("content") or ""),
                    record,
                    request_shape,
                )
                return [chunk] + chunks

        if route_info.task_type in self.FAST_FORMULA_TASKS:
            formula_records = [
                record
                for record in (catalog.get("formula_records") or [])
                if str(record.get("doc_type") or "") == "fast_formula_example"
            ]
            record = self._match_specialized_record(
                user_query,
                route_info,
                formula_records,
                title_fields=["title", "use_case", "formula_name", "formula_type"],
            )
            if record and not self._specialized_chunk_matches_record(
                chunks[0] if chunks else {},
                record,
                corpus="fast_formula_corpus",
                doc_type="fast_formula_example",
            ):
                chunk = self._build_specialized_catalog_chunk(
                    record,
                    corpus="fast_formula_corpus",
                    doc_type="fast_formula_example",
                )
                return [chunk] + chunks

        return chunks

    def _specialized_chunk_matches_record(
        self,
        chunk: Dict[str, Any],
        record: Dict[str, Any],
        *,
        corpus: str,
        doc_type: str | None = None,
    ) -> bool:
        metadata = chunk.get("metadata") or {}
        if metadata.get("corpus") != corpus:
            return False
        current_doc_type = str(metadata.get("doc_type") or "")
        if doc_type and current_doc_type != doc_type:
            return False
        record_title = str(record.get("title") or "").strip()
        record_source_uri = str(record.get("source_uri") or record.get("source_path") or "").strip()
        chunk_title = str(metadata.get("title") or "").strip()
        chunk_source_uri = str(metadata.get("source_uri") or metadata.get("source_path") or "").strip()
        if record_title and chunk_title == record_title:
            return True
        if record_source_uri and chunk_source_uri == record_source_uri:
            return True
        return False

    def _specialized_chunk_present(
        self,
        chunks: List[Dict[str, Any]],
        record: Dict[str, Any],
        *,
        corpus: str,
        doc_type: str | None = None,
    ) -> bool:
        for chunk in chunks:
            if self._specialized_chunk_matches_record(
                chunk,
                record,
                corpus=corpus,
                doc_type=doc_type,
            ):
                return True
        return False

    def _select_top_chunk(
        self,
        chunks: List[Dict[str, Any]],
        *,
        corpus: str,
        doc_type: str | None = None,
    ) -> Dict[str, Any] | None:
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            if metadata.get("corpus") != corpus:
                continue
            current_doc_type = str(metadata.get("doc_type") or "")
            if doc_type and current_doc_type != doc_type:
                continue
            return chunk
        return None

    def _grounding_lines_for_chunk(self, chunk: Dict[str, Any], prefix: str) -> List[str]:
        citation_id = chunk.get("citation_id", "[D1]")
        metadata = chunk.get("metadata") or {}
        lines = [f"{prefix} {citation_id}."]
        tables = metadata.get("tables_used") or []
        columns = metadata.get("columns_used") or []
        if tables:
            lines.append(f"Grounded tables: {', '.join(tables[:6])} {citation_id}.")
        if columns:
            lines.append(f"Grounded columns: {', '.join(columns[:8])} {citation_id}.")
        return lines

    def _normalize_sql_output_block(self, sql_text: str) -> str:
        cleaned = (sql_text or "").strip()
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"(?m)^\s*--.*$", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if cleaned and not cleaned.rstrip().endswith(";"):
            cleaned = f"{cleaned.rstrip()};"
        return cleaned

    def _sql_projection_columns(self, entry: Dict[str, Any], limit: int = 3) -> List[str]:
        columns = [str(column).upper() for column in (entry.get("columns") or [])]
        selected = [
            column
            for column in columns
            if not column.startswith(
                (
                    "CREATED_",
                    "CREATION_",
                    "LAST_UPDATE_",
                    "LAST_UPDATED_",
                    "OBJECT_VERSION_",
                    "ATTRIBUTE",
                    "REQUEST_ID",
                    "PROGRAM_",
                    "ORA_SEED_",
                )
            )
        ]
        return (selected or columns)[:limit]

    def _render_join_path_sql(self, path: List[str], *, troubleshooting: bool) -> str | None:
        registry = self.verifier.registry
        if not path:
            return None
        if len(path) == 1:
            entry = registry.get_entry(path[0])
            if not entry:
                return None
            columns = self._sql_projection_columns(entry, limit=5)
            if not columns:
                return None
            where_sql = ""
            if troubleshooting:
                status_columns = [column for column in columns if "STATUS" in column or "ERROR" in column]
                if status_columns:
                    where_sql = f"\nWHERE t1.{status_columns[0]} IS NOT NULL"
            return (
                f"SELECT {', '.join(f't1.{column}' for column in columns)}\n"
                f"FROM {path[0]} t1{where_sql}\n"
                "FETCH FIRST 100 ROWS ONLY;"
            )

        aliases = [f"t{idx}" for idx in range(1, len(path) + 1)]
        projections: List[str] = []
        joins: List[str] = []
        troubleshooting_candidates: List[str] = []

        for idx, table_name in enumerate(path):
            entry = registry.get_entry(table_name)
            if not entry:
                return None
            alias = aliases[idx]
            for column in self._sql_projection_columns(entry, limit=2):
                projections.append(f"{alias}.{column}")
                if "STATUS" in column or "ERROR" in column:
                    troubleshooting_candidates.append(f"{alias}.{column}")

        for idx in range(1, len(path)):
            left_table = path[idx - 1]
            right_table = path[idx]
            relation_details = registry.get_relation_details(left_table, right_table)
            detail = next(
                (
                    item
                    for item in relation_details
                    if str(item.get("source_column") or "").strip()
                    and str(item.get("target_column") or "").strip()
                ),
                None,
            )
            if not detail:
                return None
            source_table = str(detail.get("source_table") or "").upper()
            source_column = str(detail.get("source_column") or "").upper()
            target_table = str(detail.get("target_table") or "").upper()
            target_column = str(detail.get("target_column") or "").upper()
            if left_table == source_table and right_table == target_table:
                on_clause = f"{aliases[idx - 1]}.{source_column} = {aliases[idx]}.{target_column}"
            elif left_table == target_table and right_table == source_table:
                on_clause = f"{aliases[idx - 1]}.{target_column} = {aliases[idx]}.{source_column}"
            else:
                return None
            joins.append(f"JOIN {right_table} {aliases[idx]}\n  ON {on_clause}")

        where_sql = ""
        if troubleshooting and troubleshooting_candidates:
            where_sql = f"\nWHERE {troubleshooting_candidates[0]} IS NOT NULL"
        return (
            f"SELECT {', '.join(projections)}\n"
            f"FROM {path[0]} {aliases[0]}\n"
            + "\n".join(joins)
            + f"{where_sql}\nFETCH FIRST 100 ROWS ONLY;"
        )

    def _build_graph_backed_sql(
        self,
        user_query: str,
        route_info: Any,
        mapped_chunks: List[Dict[str, Any]],
    ) -> tuple[str | None, List[str]]:
        registry = self.verifier.registry
        candidate_tables: List[str] = []
        allowed_families: Set[str] = set()
        if route_info.module != FusionModule.UNKNOWN:
            allowed_families.update(module_families_for_value(route_info.module.value))
        elif route_info.module_family != ModuleFamily.UNKNOWN:
            allowed_families.update(module_families_for_value(route_info.module_family.value))
        allowed_families.discard(ModuleFamily.UNKNOWN.value)

        def add_candidate(value: Any) -> None:
            canonical = registry.resolve_object_name(str(value or "").upper())
            if not canonical or canonical in candidate_tables:
                return
            entry = registry.get_entry(canonical)
            if not entry:
                return
            columns = list(entry.get("columns") or [])
            if not columns:
                return
            if allowed_families:
                entry_family = str(entry.get("owning_module_family") or ModuleFamily.UNKNOWN.value)
                if entry_family not in allowed_families:
                    return
            candidate_tables.append(canonical)

        for identifier in self._specialized_query_identifiers(user_query):
            add_candidate(identifier)
        for chunk in mapped_chunks:
            metadata = chunk.get("metadata") or {}
            corpus = str(metadata.get("corpus") or "")
            if corpus == "schema_metadata_corpus":
                add_candidate(metadata.get("title"))
                for table_name in metadata.get("tables_used") or []:
                    add_candidate(table_name)
                for table_name in metadata.get("base_tables") or []:
                    add_candidate(table_name)
                continue
            if corpus == "sql_examples_corpus":
                add_candidate(metadata.get("title"))
                for table_name in metadata.get("tables_used") or []:
                    add_candidate(table_name)
                for table_name in metadata.get("base_tables") or []:
                    add_candidate(table_name)
                sql_text = self._normalize_sql_output_block(str(chunk.get("content") or ""))
                if sql_text:
                    try:
                        parsed = sqlglot.parse_one(sql_text.strip().rstrip(";"), read="oracle")
                        for table in parsed.find_all(exp.Table):
                            add_candidate(table.name)
                    except Exception:
                        pass

        for chunk in registry.search(user_query, module=route_info.module.value, limit=6):
            add_candidate((chunk.get("metadata") or {}).get("title"))

        if route_info.module_family != ModuleFamily.UNKNOWN:
            for chunk in registry.search(user_query, module=route_info.module_family.value, limit=6):
                add_candidate((chunk.get("metadata") or {}).get("title"))

        if not candidate_tables:
            return None, []

        if len(candidate_tables) >= 2:
            for idx in range(len(candidate_tables) - 1):
                for jdx in range(idx + 1, len(candidate_tables)):
                    path = registry.find_join_path(candidate_tables[idx], candidate_tables[jdx], max_depth=3)
                    if len(path) >= 2:
                        sql_text = self._render_join_path_sql(
                            path,
                            troubleshooting=route_info.task_type == TaskType.SQL_TROUBLESHOOTING,
                        )
                        if sql_text:
                            return sql_text, path

        for table_name in candidate_tables:
            sql_text = self._render_join_path_sql(
                [table_name],
                troubleshooting=route_info.task_type == TaskType.SQL_TROUBLESHOOTING,
            )
            if sql_text:
                return sql_text, [table_name]
        return None, []

    def _build_specialized_sql_response(
        self,
        route_info: Any,
        user_query: str,
        mapped_chunks: List[Dict[str, Any]],
        audit: Dict[str, Any],
    ) -> str | None:
        module_hint = self._infer_sql_module_hint(user_query)
        module_name = route_info.module.value if route_info.module != FusionModule.UNKNOWN else route_info.module_family.value
        if module_hint and (
            route_info.module == FusionModule.UNKNOWN
            or (
                route_info.module != FusionModule.UNKNOWN
                and not (module_families_for_value(route_info.module.value) & module_families_for_value(module_hint))
            )
        ):
            module_name = module_hint
        module_alignment_target = self._sql_alignment_target(route_info, module_hint)
        request_shape = self._parse_sql_request_shape(user_query, route_info)
        diagnostics = self._sql_report_family_support_diagnostics(request_shape)
        module_alignment_override = False
        if str(request_shape.get("report_family") or "") in {
            "procurement_purchase_order_details",
            "procurement_receiving_invoicing_match",
        }:
            module_alignment_target = ModuleFamily.UNKNOWN
            module_alignment_override = True
            audit["sql_module_alignment_override"] = request_shape.get("report_family")
        audit["sql_request_shape"] = {
            "required_fields": [field["key"] for field in (request_shape.get("required_fields") or [])],
            "required_filters": [item["key"] for item in (request_shape.get("required_filters") or [])],
            "required_tables": request_shape.get("required_tables") or [],
            "required_table_alias_counts": request_shape.get("required_table_alias_counts") or {},
            "required_ordering": [item["key"] for item in (request_shape.get("required_ordering") or [])],
            "required_calculations": [item["key"] for item in (request_shape.get("required_calculations") or [])],
            "report_family": request_shape.get("report_family") or "",
            "template_hint": request_shape.get("template_hint") or "",
            "needs_join": bool(request_shape.get("needs_join")),
        }
        audit["sql_module_hint"] = module_hint or ""
        audit["sql_effective_module"] = module_name
        audit["sql_module_alignment_target"] = getattr(module_alignment_target, "value", module_alignment_target)
        audit["sql_report_family_reason_code"] = self._sql_report_family_reason_code(request_shape)
        audit["sql_module_reason_code"] = self._sql_module_inference_reason_code(
            route_info,
            module_hint,
            module_name,
            alignment_override=module_alignment_override,
        )
        audit["sql_shape_reason_code"] = self._sql_shape_support_reason_code(request_shape, diagnostics)
        audit["sql_support_diagnostics"] = {
            "shape_supported": self._sql_shape_supported(diagnostics),
            "missing_fields": diagnostics.get("missing_fields") or [],
            "missing_filters": diagnostics.get("missing_filters") or [],
            "missing_ordering": diagnostics.get("missing_ordering") or [],
            "missing_calculations": diagnostics.get("missing_calculations") or [],
        }
        self._log_sql_decision_event(
            stage="report_family_inference",
            user_query=user_query,
            route_info=route_info,
            module_name=module_name,
            request_shape=request_shape,
            audit=audit,
            reason_code=audit["sql_report_family_reason_code"],
            module_hint=module_hint,
            module_alignment_target=module_alignment_target,
            diagnostics=diagnostics,
        )
        self._log_sql_decision_event(
            stage="module_inference",
            user_query=user_query,
            route_info=route_info,
            module_name=module_name,
            request_shape=request_shape,
            audit=audit,
            reason_code=audit["sql_module_reason_code"],
            module_hint=module_hint,
            module_alignment_target=module_alignment_target,
            diagnostics=diagnostics,
        )
        self._log_sql_decision_event(
            stage="shape_support",
            user_query=user_query,
            route_info=route_info,
            module_name=module_name,
            request_shape=request_shape,
            audit=audit,
            reason_code=audit["sql_shape_reason_code"],
            module_hint=module_hint,
            module_alignment_target=module_alignment_target,
            diagnostics=diagnostics,
        )

        def refuse(reason: str) -> str:
            return (
                f"[MODULE]\n{module_name}\n"
                f"[GROUNDING]\n{reason}\n"
                f"[SQL]\n{FAIL_CLOSED_MESSAGE}\n"
                "[NOTES]\nSafe refusal: no verified grounded SQL pattern passed validation for this request."
            )

        if any(re.search(pattern, user_query) for pattern in self.SQL_UNSAFE_QUERY_PATTERNS):
            audit["specialized_lane_used"] = "sql_refusal"
            audit["verification_status"] = "FAILED_SQL_UNSAFE_REQUEST"
            audit["sql_selection_path"] = "unsafe_request"
            audit["sql_refusal_reason_code"] = "SQL_REFUSAL_UNSAFE_REQUEST"
            self._log_sql_decision_event(
                stage="refusal",
                user_query=user_query,
                route_info=route_info,
                module_name=module_name,
                request_shape=request_shape,
                audit=audit,
                reason_code=audit["sql_refusal_reason_code"],
                module_hint=module_hint,
                module_alignment_target=module_alignment_target,
                selection_path=audit["sql_selection_path"],
                verifier_status=audit["verification_status"],
                verifier_reason=audit["verification_status"],
                diagnostics=diagnostics,
            )
            return refuse("The request contained unsafe or explicitly placeholder SQL instructions, so no grounded SQL was emitted.")

        sql_chunk, sql_text, selection_reason = self._select_best_sql_chunk(
            user_query,
            route_info,
            mapped_chunks,
            request_shape,
            module_alignment_target,
        )
        selection_path = "grounded_sql_example" if sql_text else ""
        grounded_path: List[str] = []
        if not sql_text:
            supported_sql_candidate, supported_reason = self._build_supported_sql_report(request_shape, user_query)
            supported_sql = self._normalize_sql_output_block(supported_sql_candidate or "")
            if supported_sql:
                for candidate in self._sql_candidate_variants(supported_sql, request_shape):
                    ok, reason = self._verify_sql_candidate(
                        candidate,
                        module_alignment_target=module_alignment_target,
                        request_shape=request_shape,
                    )
                    if ok:
                        sql_text = candidate
                        audit["sql_report_family_template_used"] = request_shape.get("report_family")
                        selection_path = "report_family_template"
                        selection_reason = None
                        break
                    selection_reason = reason
            elif supported_reason:
                selection_reason = supported_reason

        if not sql_text:
            demo_sql_candidate = (
                self._build_demo_payables_sql(request_shape)
                or self._build_demo_receivables_sql(request_shape, user_query)
                or self._build_demo_general_ledger_sql(request_shape, user_query)
                or self._build_demo_cash_management_sql(request_shape)
                or self._build_demo_procurement_sql(request_shape)
            )
            demo_sql = self._normalize_sql_output_block(demo_sql_candidate or "")
            if demo_sql:
                for candidate in self._sql_candidate_variants(demo_sql, request_shape):
                    ok, reason = self._verify_sql_candidate(
                        candidate,
                        module_alignment_target=module_alignment_target,
                        request_shape=request_shape,
                    )
                    if ok:
                        sql_text = candidate
                        audit["sql_demo_template_used"] = request_shape.get("template_hint")
                        selection_path = "demo_template"
                        selection_reason = None
                        break
                    selection_reason = reason

        if not sql_text:
            sql_text, grounded_path = self._build_graph_backed_sql(user_query, route_info, mapped_chunks)
            if sql_text:
                accepted = False
                for candidate in self._sql_candidate_variants(sql_text, request_shape):
                    ok, reason = self._verify_sql_candidate(
                        candidate,
                        module_alignment_target=module_alignment_target,
                        request_shape=request_shape,
                    )
                    if ok:
                        sql_text = candidate
                        selection_path = "graph_backed"
                        accepted = True
                        break
                    selection_reason = reason
                if not accepted:
                    sql_text = ""

        if not sql_text:
            module_seed_query = f"{module_name} {user_query}".strip()
            sql_text, grounded_path = self._build_graph_backed_sql(module_seed_query, route_info, [])
            if sql_text:
                accepted = False
                for candidate in self._sql_candidate_variants(sql_text, request_shape):
                    ok, reason = self._verify_sql_candidate(
                        candidate,
                        module_alignment_target=module_alignment_target,
                        request_shape=request_shape,
                    )
                    if ok:
                        sql_text = candidate
                        audit["sql_module_seed_fallback_used"] = True
                        selection_path = "module_seed_fallback"
                        accepted = True
                        break
                    selection_reason = reason
                if not accepted:
                    sql_text = ""

        if not sql_text:
            catalog = self._load_specialization_catalog()
            hint_families = module_families_for_value(module_hint) if module_hint else set()
            hint_families.discard(ModuleFamily.UNKNOWN.value)
            hinted_route = SimpleNamespace(
                task_type=route_info.task_type,
                module=FusionModule.UNKNOWN,
                module_family=ModuleFamily.UNKNOWN,
            )
            if hint_families:
                hinted_family = next(iter(hint_families))
                if hinted_family in {member.value for member in ModuleFamily}:
                    hinted_route.module_family = ModuleFamily(hinted_family)
            fallback_records = [
                item
                for item in (catalog.get("sql_records") or [])
                if not module_hint
                or str(item.get("module") or "").strip().lower() == module_hint.lower()
                or bool(module_families_for_value(str(item.get("module") or "")) & hint_families)
            ]
            if fallback_records:
                fallback_record = self._match_specialized_record(
                    user_query,
                    hinted_route if module_hint else route_info,
                    fallback_records,
                    title_fields=["title", "source_file"],
                )
                if fallback_record:
                    raw_sql = str(fallback_record.get("content") or "")
                    accepted = False
                    for candidate in self._sql_candidate_variants(raw_sql, request_shape):
                        ok, reason = self._verify_sql_candidate(
                            candidate,
                            module_alignment_target=module_alignment_target,
                            request_shape=request_shape,
                        )
                        if ok:
                            sql_text = candidate
                            selection_reason = None
                            audit["sql_catalog_fallback_used"] = True
                            selection_path = "catalog_fallback"
                            accepted = True
                            break
                        selection_reason = reason
                    if not accepted:
                        sql_text = ""

        if not sql_text:
            audit["specialized_lane_used"] = "sql_refusal"
            audit["verification_status"] = selection_reason or "FAILED_SQL_NO_GROUNDED_PATTERN"
            audit["sql_selection_path"] = selection_path or "none"
            specific_reason = self._specific_sql_support_reason(request_shape, selection_reason)
            audit["sql_refusal_reason_code"] = self._sql_refusal_reason_code(
                request_shape,
                specific_reason,
                audit["verification_status"],
                diagnostics=diagnostics,
            )
            self._log_sql_decision_event(
                stage="refusal",
                user_query=user_query,
                route_info=route_info,
                module_name=module_name,
                request_shape=request_shape,
                audit=audit,
                reason_code=audit["sql_refusal_reason_code"],
                module_hint=module_hint,
                module_alignment_target=module_alignment_target,
                selection_path=audit["sql_selection_path"],
                verifier_status=audit["verification_status"],
                verifier_reason=selection_reason or audit["verification_status"],
                diagnostics=diagnostics,
            )
            return refuse(specific_reason)

        schema_chunk = self._select_top_chunk(mapped_chunks, corpus="schema_metadata_corpus")
        if sql_chunk and not grounded_path:
            grounding_lines = self._grounding_lines_for_chunk(sql_chunk, "Adapted directly from grounded SQL example")
            notes = "Reused the grounded SQL pattern without inventing new tables, columns, or joins."
        elif audit.get("sql_demo_template_used"):
            template_used = str(audit.get("sql_demo_template_used") or "")
            if template_used.startswith("receivables"):
                grounding_lines = [
                    "Synthesized from grounded Receivables invoice, customer, party, and payment-schedule metadata.",
                    "Requested fields, joins, and filters were enforced before the SQL was emitted.",
                ]
                notes = (
                    "Used a demo-safe grounded Receivables reporting template because the retained SQL examples did not cover "
                    "the full request shape."
                )
            elif template_used.startswith("general_ledger"):
                grounding_lines = [
                    "Synthesized from grounded General Ledger journal, line, and code-combination metadata.",
                    "Requested fields, joins, and filters were enforced before the SQL was emitted.",
                ]
                notes = (
                    "Used a demo-safe grounded General Ledger reporting template because the retained SQL examples did not cover "
                    "the full request shape."
                )
            elif template_used.startswith("cash_management"):
                grounding_lines = [
                    "Synthesized from grounded Cash Management bank-account metadata and legal-entity linkage.",
                    "Requested fields, joins, and filters were enforced before the SQL was emitted.",
                ]
                notes = (
                    "Used a demo-safe grounded Cash Management reporting template because the retained SQL examples did not cover "
                    "the full request shape."
                )
            elif template_used.startswith("procurement"):
                grounding_lines = [
                    "Synthesized from grounded Procurement supplier, supplier-site, and business-unit metadata.",
                    "Requested fields, joins, and filters were enforced before the SQL was emitted.",
                ]
                notes = (
                    "Used a demo-safe grounded Procurement reporting template because the retained SQL examples did not cover "
                    "the full request shape."
                )
            else:
                grounding_lines = [
                    "Synthesized from grounded Payables invoice, distribution, supplier, and code-combination metadata.",
                    "Requested fields, joins, and filters were enforced before the SQL was emitted.",
                ]
                notes = (
                    "Used a demo-safe grounded Payables reporting template because the retained SQL examples did not cover "
                    "the full request shape."
                )
        else:
            grounding_lines = ["Synthesized from grounded schema metadata and verified join-graph relations."]
            if grounded_path:
                grounding_lines.append(f"Grounded path: {', '.join(grounded_path[:5])}.")
            notes = "Built from verified schema metadata because no close retained SQL pattern was available."
        if schema_chunk and (not sql_chunk or schema_chunk.get("citation_id") != sql_chunk.get("citation_id")):
            grounding_lines.append(
                f"Validated against retained schema metadata {schema_chunk.get('citation_id', '[D2]')}."
            )

        audit["specialized_lane_used"] = "sql"
        audit["sql_selection_path"] = selection_path or "grounded_sql_example"
        audit["sql_refusal_reason_code"] = None
        self._log_sql_decision_event(
            stage="candidate_selected",
            user_query=user_query,
            route_info=route_info,
            module_name=module_name,
            request_shape=request_shape,
            audit=audit,
            reason_code="SQL_CANDIDATE_SELECTED",
            module_hint=module_hint,
            module_alignment_target=module_alignment_target,
            selection_path=audit["sql_selection_path"],
            verifier_status="PASSED",
            verifier_reason="PASSED",
            diagnostics=diagnostics,
        )
        return (
            f"[MODULE]\n{module_name}\n"
            f"[GROUNDING]\n" + "\n".join(grounding_lines) + "\n"
            f"[SQL]\n{sql_text}\n"
            f"[NOTES]\n{notes}"
        )

    def _build_specialized_formula_response(
        self,
        route_info: Any,
        user_query: str,
        mapped_chunks: List[Dict[str, Any]],
        audit: Dict[str, Any],
    ) -> str | None:
        def refuse(reason: str, verification_status: str) -> str:
            audit["specialized_lane_used"] = "fast_formula_refusal"
            audit["verification_status"] = verification_status
            return (
                "[FORMULA_TYPE]\nUNKNOWN\n"
                f"[GROUNDING]\n{reason}\n"
                f"[FORMULA]\n{FAIL_CLOSED_MESSAGE}\n"
                "[NOTES]\nSafe refusal: no verified grounded Fast Formula example passed validation for this request."
            )

        if any(re.search(pattern, user_query) for pattern in self.FAST_FORMULA_UNSAFE_QUERY_PATTERNS):
            return refuse(
                "The request contained unsupported or clearly unsafe Fast Formula content, so no grounded formula was emitted.",
                "FAILED_FF_UNSUPPORTED_REQUEST",
            )

        formula_shape = self._parse_fast_formula_request_shape(user_query, route_info)
        requested_formula_type = self._normalize_formula_type(str(formula_shape.get("requested_formula_type") or "UNKNOWN"))
        audit["fast_formula_request_shape"] = {
            "requested_formula_type": requested_formula_type,
            "formula_name": str(formula_shape.get("formula_name") or ""),
            "is_troubleshooting": bool(formula_shape.get("is_troubleshooting")),
            "requested_database_items": list(formula_shape.get("requested_database_items") or [])[:12],
        }

        formula_chunks = [
            chunk
            for chunk in mapped_chunks
            if str((chunk.get("metadata") or {}).get("corpus") or "") == "fast_formula_corpus"
            and str((chunk.get("metadata") or {}).get("doc_type") or "") == "fast_formula_example"
        ]
        if not formula_chunks:
            return refuse(
                "No retained grounded Fast Formula example matched closely enough to adapt safely.",
                "FAILED_FF_UNSUPPORTED_REQUEST",
            )
        high_signal_chunks = [
            chunk
            for chunk in formula_chunks
            if not self._is_low_signal_formula_example(chunk.get("metadata") or {})
        ]
        if high_signal_chunks:
            formula_chunks = high_signal_chunks

        best_chunk: Optional[Dict[str, Any]] = None
        best_score = float("-inf")
        broken_tokens = set(formula_shape.get("broken_formula_tokens") or [])
        query_tokens = self._specialized_query_tokens(user_query)
        requested_formula_name = str(formula_shape.get("formula_name_normalized") or "")
        requested_dbis = {str(item).upper() for item in (formula_shape.get("requested_database_items") or [])}
        broken_formula_normalized = self._normalize_formula_block_text(str(formula_shape.get("broken_formula") or ""))
        for chunk in formula_chunks:
            metadata = chunk.get("metadata") or {}
            score = float(chunk.get("rerank_score") or chunk.get("combined_score") or 0.0)
            chunk_formula_type = self._normalize_formula_type(str(metadata.get("formula_type") or "UNKNOWN"))
            if requested_formula_type.upper() != "UNKNOWN":
                if chunk_formula_type == requested_formula_type:
                    score += 5.0
                elif (
                    requested_formula_type.lower() in chunk_formula_type.lower()
                    or chunk_formula_type.lower() in requested_formula_type.lower()
                ):
                    score += 2.0
                else:
                    score -= 3.0
            score += self._specialized_overlap_score(
                query_tokens,
                self._specialized_query_tokens(str(metadata.get("formula_name") or metadata.get("title") or "")),
            ) * 2.2
            score += self._specialized_overlap_score(
                query_tokens,
                self._specialized_query_tokens(str(metadata.get("use_case") or "")),
            ) * 1.6
            if broken_tokens:
                score += self._specialized_overlap_score(
                    broken_tokens,
                    self._specialized_query_tokens(str(chunk.get("content") or "")[:2000]),
                ) * 3.2
            candidate_formula_name = self._normalize_formula_name(
                str(metadata.get("formula_name") or metadata.get("title") or "")
            )
            candidate_source = str(metadata.get("source_uri") or metadata.get("source_file") or "").lower()
            candidate_dbis = {str(item).upper() for item in (metadata.get("database_items") or [])}
            if requested_dbis:
                dbi_overlap = self._specialized_overlap_score(requested_dbis, candidate_dbis)
                score += dbi_overlap * 6.0
                if formula_shape.get("is_troubleshooting") and dbi_overlap == 0.0:
                    score -= 4.0
            candidate_content_normalized = self._normalize_formula_block_text(str(chunk.get("content") or "")[:2500])
            if formula_shape.get("is_troubleshooting") and broken_formula_normalized and candidate_content_normalized:
                if broken_formula_normalized in candidate_content_normalized:
                    score += 9.0
                else:
                    broken_token_overlap = self._specialized_overlap_score(
                        set(broken_formula_normalized.split()),
                        set(candidate_content_normalized.split()),
                    )
                    score += broken_token_overlap * 2.5
            if requested_formula_name:
                if candidate_formula_name == requested_formula_name:
                    score += 9.0
                elif requested_formula_name in candidate_formula_name or candidate_formula_name in requested_formula_name:
                    score += 5.0
                elif "formula_types.json" in candidate_source:
                    score -= 4.5
            if not bool(metadata.get("derived_from_doc")):
                score += 0.5
            if self._is_low_signal_formula_example(metadata):
                score -= 3.5
            if score > best_score:
                best_score = score
                best_chunk = chunk

        formula_chunk = best_chunk
        if not formula_chunk:
            return refuse(
                "No retained grounded Fast Formula example could be selected safely.",
                "FAILED_FF_UNSUPPORTED_REQUEST",
            )

        metadata = formula_chunk.get("metadata") or {}
        formula_type = self._normalize_formula_type(str(metadata.get("formula_type") or "UNKNOWN"))
        if formula_type.upper() == "UNKNOWN" and requested_formula_type.upper() != "UNKNOWN":
            formula_type = requested_formula_type
        if formula_type.upper() == "UNKNOWN":
            inferred_formula_type = self._infer_formula_type_from_text(
                "\n".join(
                    [
                        user_query,
                        str(formula_shape.get("broken_formula") or ""),
                        str(metadata.get("use_case") or ""),
                        str(metadata.get("formula_name") or ""),
                        str(formula_chunk.get("content") or "")[:1200],
                    ]
                )
            )
            if inferred_formula_type.upper() != "UNKNOWN":
                formula_type = inferred_formula_type
        if (
            requested_formula_type.upper() != "UNKNOWN"
            and formula_type.upper() != "UNKNOWN"
            and requested_formula_type != formula_type
            and requested_formula_type.lower() not in formula_type.lower()
            and formula_type.lower() not in requested_formula_type.lower()
            and not bool(formula_shape.get("is_troubleshooting"))
        ):
            if requested_formula_name and self._normalize_formula_name(
                str(metadata.get("formula_name") or metadata.get("title") or "")
            ) == requested_formula_name:
                formula_type = self._normalize_formula_type(str(metadata.get("formula_type") or formula_type))
            else:
                return refuse(
                    "Requested formula type is not supported by the retained grounded examples for this prompt.",
                    "FAILED_FF_SEMANTIC_MISMATCH",
                )

        grounded_db_items = self._grounded_formula_identifiers(
            self._coerce_string_list(metadata.get("database_items"))
        )
        grounded_contexts = self._grounded_formula_identifiers(
            self._coerce_string_list(metadata.get("contexts"))
        )
        input_values = self._coerce_string_list(metadata.get("input_values"))
        use_case = str(metadata.get("use_case") or metadata.get("formula_name") or metadata.get("title") or formula_type)

        broken_formula = str(formula_shape.get("broken_formula") or "")
        if formula_shape.get("is_troubleshooting") and broken_formula:
            formula_text = self._repair_formula_from_broken_input(
                broken_formula,
                formula_type=formula_type,
                use_case=use_case,
                database_items=grounded_db_items,
                input_values=input_values,
            )
        else:
            formula_text = self._clean_formula_text(str(formula_chunk.get("content") or ""))
            if "return" not in formula_text.lower():
                formula_text = self._build_grounded_formula_template(
                    formula_type,
                    use_case,
                    database_items=grounded_db_items,
                    input_values=input_values,
                )

        ff_ok, ff_reason = self.verifier.verify_fast_formula(
            formula_text,
            allowed_database_items=grounded_db_items or None,
            allowed_contexts=grounded_contexts or None,
            expected_formula_type=formula_type,
        )
        used_template_fallback = False
        if not ff_ok:
            template_formula = self._build_grounded_formula_template(
                formula_type,
                use_case,
                database_items=grounded_db_items,
                input_values=input_values,
            )
            template_ok, template_reason = self.verifier.verify_fast_formula(
                template_formula,
                allowed_database_items=grounded_db_items or None,
                allowed_contexts=grounded_contexts or None,
                expected_formula_type=formula_type,
            )
            if template_ok:
                formula_text = template_formula
                used_template_fallback = True
            else:
                return refuse(
                    ff_reason or template_reason or "Fast Formula verifier rejected the grounded adaptation.",
                    "FAILED_FF_STRUCTURE_MISMATCH",
                )

        grounding_lines = [
            f"Adapted directly from grounded Fast Formula example {formula_chunk.get('citation_id', '[D1]')}.",
            f"Grounded formula type: {formula_type} {formula_chunk.get('citation_id', '[D1]')}.",
        ]
        if grounded_db_items:
            grounding_lines.append(
                f"Grounded database items: {', '.join(grounded_db_items[:8])} {formula_chunk.get('citation_id', '[D1]')}."
            )
        if grounded_contexts:
            grounding_lines.append(
                f"Grounded contexts: {', '.join(grounded_contexts[:6])} {formula_chunk.get('citation_id', '[D1]')}."
            )

        notes = "Reused the grounded Fast Formula example without inventing unsupported contexts or database items."
        if formula_shape.get("is_troubleshooting"):
            symptoms: List[str] = []
            if "-- return" in broken_formula.lower() or "return" not in broken_formula.lower():
                symptoms.append("missing or commented RETURN statement")
            if "inputs are" not in broken_formula.lower():
                symptoms.append("missing INPUTS declaration")
            if "default for" not in broken_formula.lower():
                symptoms.append("missing DEFAULT FOR grounding")
            symptom_text = ", ".join(symptoms) if symptoms else "structure inconsistency in supplied formula"
            notes = (
                "Troubleshooting grounded response.\n"
                f"Symptom: {symptom_text}.\n"
                "Probable cause: formula-specific structure or reference rules were violated (DBI/context/RETURN/INPUTS alignment).\n"
                "Fix: apply the corrected grounded formula shown above and revalidate with proper DEFAULT/INPUTS/RETURN and allowed references."
            )
        if used_template_fallback:
            notes = (
                "Used a deterministic grounded template because direct formula reuse failed Fast Formula verifier checks. "
                + notes
            )

        audit["specialized_lane_used"] = "fast_formula"
        return (
            f"[FORMULA_TYPE]\n{formula_type}\n"
            f"[GROUNDING]\n" + "\n".join(grounding_lines) + "\n"
            f"[FORMULA]\n{formula_text}\n"
            f"[NOTES]\n{notes}"
        )

    def _build_specialized_lane_response(
        self,
        route_info: Any,
        user_query: str,
        mapped_chunks: List[Dict[str, Any]],
        audit: Dict[str, Any],
    ) -> str | None:
        if route_info.task_type in self.SQL_TASKS or self._is_sql_capable_query(route_info, user_query):
            if route_info.task_type not in self.SQL_TASKS:
                audit["sql_capable_override"] = True
            return self._build_specialized_sql_response(route_info, user_query, mapped_chunks, audit)
        if route_info.task_type in self.FAST_FORMULA_TASKS:
            return self._build_specialized_formula_response(route_info, user_query, mapped_chunks, audit)
        return None

    async def chat(
        self,
        db: AsyncSession,
        tenant: Any,
        request: ChatRequest,
        require_citations: bool = True,
    ) -> ChatResponse:
        start_time = time.perf_counter()
        timings: Dict[str, float] = {}
        trace_id = str(uuid.uuid4())
        user_query = request.messages[-1].content

        t0 = time.perf_counter()
        norm_tags = self.verifier.normalize_objects(user_query)
        timings["normalization"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        route_info = self.router.route(user_query)
        timings["routing"] = time.perf_counter() - t0
        if route_info.task_type in self.FAST_FORMULA_TASKS:
            unconfirmed = [
                token
                for token in (norm_tags.get("unconfirmed") or [])
                if str(token or "").strip().upper() not in {"UNKNOWN", "UNSUPPORTED", "UNCLASSIFIED"}
            ]
            norm_tags = dict(norm_tags)
            norm_tags["unconfirmed"] = unconfirmed

        if route_info.task_type == TaskType.GREETING:
            timings["total_e2e"] = time.perf_counter() - start_time
            return ChatResponse(
                id=trace_id,
                created=int(time.time()),
                model="antigravity-v6",
                choices=[{"message": {"role": "assistant", "content": "IWFUSION-SLM-V1 is ready for grounded Oracle Fusion questions."}}],
                citations=[],
                retrieved_chunks=[],
                timings=timings,
                audit={"verification_status": "SKIPPED_GREETING"},
            )

        task_config = TASK_CONFIGS.get(route_info.task_type, TASK_CONFIGS[TaskType.GENERAL])
        retrieval_plan = RetrievalPolicy.for_task(route_info.task_type)
        t0 = time.perf_counter()
        route_info = await self._disambiguate_route(
            db=db,
            tenant_id=tenant.id,
            user_query=user_query,
            route_info=route_info,
            retrieval_plan=retrieval_plan,
            task_config=task_config,
        )
        timings["disambiguation"] = time.perf_counter() - t0
        retrieval_filters = self._build_retrieval_filters(route_info, retrieval_plan)

        t0 = time.perf_counter()
        all_candidates, reranked_chunks = await self._retrieve_grounding_chunks(
            db=db,
            tenant_id=tenant.id,
            user_query=user_query,
            route_info=route_info,
            retrieval_plan=retrieval_plan,
            task_config=task_config,
            retrieval_filters=retrieval_filters,
        )
        timings["retrieval"] = time.perf_counter() - t0
        timings["reranking"] = 0.0

        strict_financial_leaf = self._is_strict_financial_leaf_route(route_info)
        fallback_audit: Dict[str, Any] = {
            "finance_leaf_firewall": strict_financial_leaf,
            "finance_soft_fallback_attempted": False,
            "finance_soft_fallback_used": False,
            "finance_soft_fallback_candidate_count": 0,
            "exact_module_troubleshooting_support_count": 0,
            "exact_module_doc_count": self._count_exact_module_docs(reranked_chunks, route_info.module)
            if getattr(route_info, "module_explicit", False)
            else 0,
        }
        if strict_financial_leaf:
            reranked_chunks = self._filter_finance_leaf_chunks(reranked_chunks, route_info.module)
            exact_module_doc_count = self._count_exact_module_docs(reranked_chunks, route_info.module)
            fallback_audit["exact_module_doc_count"] = exact_module_doc_count

            should_try_finance_soft_fallback = exact_module_doc_count == 0 and (
                route_info.task_type in self.DOCS_EXPECTED_TASKS
                or route_info.confidence < RetrievalPolicy.finance_soft_fallback_threshold()
            )
            if should_try_finance_soft_fallback:
                fallback_filters = dict(retrieval_filters)
                fallback_filters.pop("module", None)
                fallback_filters.pop("strict_exact_module_only", None)
                fallback_filters.pop("exact_module_allowlist", None)
                fallback_filters.pop("requested_module", None)
                if route_info.module_family != ModuleFamily.UNKNOWN:
                    fallback_filters["requested_module"] = route_info.module_family.value
                fallback_filters["allow_same_family_fallback"] = True
                fallback_audit["finance_soft_fallback_attempted"] = True
                fallback_candidates, fallback_chunks = await self._retrieve_grounding_chunks(
                    db=db,
                    tenant_id=tenant.id,
                    user_query=user_query,
                    route_info=route_info,
                    retrieval_plan=retrieval_plan,
                    task_config=task_config,
                    retrieval_filters=fallback_filters,
                )
                fallback_chunks = self._filter_finance_leaf_chunks(
                    fallback_chunks,
                    route_info.module,
                    allow_same_family_fallback=True,
                )
                fallback_audit["finance_soft_fallback_candidate_count"] = len(fallback_candidates)
                fallback_audit["finance_soft_fallback_used"] = bool(fallback_chunks)
                if fallback_chunks:
                    reranked_chunks = fallback_chunks

        task_gate, reranked_chunks = self._annotate_task_semantics(reranked_chunks, user_query, route_info)
        reranked_chunks = self._preserve_grounding_chunks(reranked_chunks, retrieval_plan)
        task_gate, reranked_chunks = self._annotate_task_semantics(reranked_chunks, user_query, route_info)
        (
            reranked_chunks,
            task_gate,
            preferred_fallback_audit,
            preferred_fallback_modules,
        ) = await self._attempt_residual_troubleshooting_fallback(
            db=db,
            tenant_id=tenant.id,
            user_query=user_query,
            route_info=route_info,
            retrieval_plan=retrieval_plan,
            task_config=task_config,
            retrieval_filters=retrieval_filters,
            reranked_chunks=reranked_chunks,
            task_gate=task_gate,
        )
        reranked_chunks = self._preserve_grounding_chunks(reranked_chunks, retrieval_plan)
        task_gate, reranked_chunks = self._annotate_task_semantics(
            reranked_chunks,
            user_query,
            route_info,
            preferred_module_allowlist=preferred_fallback_modules if preferred_fallback_audit.get("preferred_module_troubleshooting_fallback_used") else None,
        )
        reranked_chunks = self._augment_specialized_chunks(user_query, route_info, reranked_chunks)
        task_gate, reranked_chunks = self._annotate_task_semantics(
            reranked_chunks,
            user_query,
            route_info,
            preferred_module_allowlist=preferred_fallback_modules if preferred_fallback_audit.get("preferred_module_troubleshooting_fallback_used") else None,
        )
        if preferred_fallback_audit.get("preferred_module_troubleshooting_fallback_used"):
            task_gate = dict(task_gate)
            task_gate["task_semantic_gate"] = "PASSED"
            task_gate["task_gate_reason"] = "preferred_module_troubleshooting_fallback"
            if preferred_fallback_audit.get("preferred_module_troubleshooting_correction"):
                task_gate["module_correction_message"] = preferred_fallback_audit["preferred_module_troubleshooting_correction"]

        mapped_chunks = CitationMapper.map_chunks(reranked_chunks)
        if (
            strict_financial_leaf
            and str(task_gate.get("task_gate_reason") or "") == "no_query_signal"
            and not preferred_fallback_audit.get("preferred_module_troubleshooting_fallback_used")
        ):
            mapped_chunks = self._filter_all_chunks_to_exact_module(
                mapped_chunks,
                route_info.module,
            )
        exact_module_troubleshooting_support_count = 0
        preferred_module_troubleshooting_support_count = 0
        if strict_financial_leaf and route_info.task_type == TaskType.TROUBLESHOOTING:
            exact_module_troubleshooting_support_count = self._count_exact_module_troubleshooting_support(
                mapped_chunks,
                route_info.module,
            )
            fallback_audit["exact_module_troubleshooting_support_count"] = exact_module_troubleshooting_support_count
            if preferred_fallback_audit.get("preferred_module_troubleshooting_fallback_used"):
                preferred_module_troubleshooting_support_count = self._count_allowlisted_troubleshooting_support(
                    mapped_chunks,
                    preferred_fallback_modules,
                )
                fallback_audit["preferred_module_troubleshooting_support_count"] = preferred_module_troubleshooting_support_count
            if (
                task_gate.get("task_semantic_gate") == "FAILED"
                and not task_gate.get("module_correction_message")
                and exact_module_troubleshooting_support_count > 0
            ):
                task_gate = dict(task_gate)
                task_gate["task_semantic_gate"] = "PASSED"
                task_gate["task_gate_reason"] = "exact_module_non_doc_support"
            elif (
                task_gate.get("task_semantic_gate") == "FAILED"
                and preferred_module_troubleshooting_support_count > 0
            ):
                task_gate = dict(task_gate)
                task_gate["task_semantic_gate"] = "PASSED"
                task_gate["task_gate_reason"] = "preferred_module_troubleshooting_fallback"
            if exact_module_troubleshooting_support_count > 0:
                mapped_chunks = self._filter_all_chunks_to_exact_module(
                    mapped_chunks,
                    route_info.module,
                )
                exact_module_troubleshooting_support_count = self._count_exact_module_troubleshooting_support(
                    mapped_chunks,
                    route_info.module,
                )
                fallback_audit["exact_module_troubleshooting_support_count"] = exact_module_troubleshooting_support_count

        citations = CitationMapper.to_citations(mapped_chunks)
        context_block = CitationMapper.format_context_block(mapped_chunks) if mapped_chunks else "No relevant documentation found."
        sql_index_used = any(
            (chunk.get("metadata") or {}).get("corpus") in {"sql_corpus", "sql_examples_corpus"}
            for chunk in mapped_chunks
        )
        formula_index_used = any(
            (chunk.get("metadata") or {}).get("corpus") == "fast_formula_corpus"
            for chunk in mapped_chunks
        )

        audit = {
            "trace_id": trace_id,
            "hallucination_score": (
                1.0
                if (route_info.task_type in self.SQL_TASKS and norm_tags["unconfirmed"])
                else 0.0
            ),
            "sql_validity_score": 0.0,
            "module_accuracy_score": 1.0 if route_info.module_family.value != "UNKNOWN" else 0.5,
            "grounding_score": 1.0 if citations else 0.0,
            "verification_status": "PENDING",
            "normalization_tags": norm_tags,
            "module": route_info.module.value,
            "module_family": route_info.module_family.value,
            "module_candidates": route_info.module_candidates,
            "module_disambiguated": not route_info.disambiguation_required,
            "module_confidence": float(getattr(route_info, "module_confidence", 0.0) or getattr(route_info, "confidence", 0.0) or 0.0),
            "task_type": route_info.task_type.value,
            "intent_confidence": float(getattr(route_info, "intent_confidence", 0.0) or task_gate.get("top_task_confidence") or 0.0),
            "intent_signals": list(getattr(route_info, "intent_signals", []) or []),
            "negative_signals": list(getattr(route_info, "negative_signals", []) or []),
            "module_score_breakdown": dict(getattr(route_info, "module_score_breakdown", {}) or {}),
            "corpora": retrieval_plan.corpora,
            "sql_index_used": sql_index_used,
            "formula_index_used": formula_index_used,
            "retrieval_candidate_count": len(all_candidates),
            "retained_chunk_count": len(mapped_chunks),
            "citation_count": len(citations),
            "docs_retrieved_count": sum(
                1 for chunk in mapped_chunks if (chunk.get("metadata") or {}).get("corpus") in self.DOC_GROUNDING_CORPORA
            ),
            "docs_passed_to_prompt_count": sum(
                1 for chunk in mapped_chunks if (chunk.get("metadata") or {}).get("corpus") in self.DOC_GROUNDING_CORPORA
            ),
            "context_chars": len(context_block),
            "query_task_signals": task_gate.get("requested_task_signals", []),
            "task_semantic_gate": task_gate.get("task_semantic_gate", "SKIPPED"),
            "task_semantic_top_signal": task_gate.get("top_task_signal"),
            "task_semantic_top_confidence": task_gate.get("top_task_confidence", 0.0),
            "task_semantic_strong_doc_count": task_gate.get("strong_doc_count", 0),
            "task_semantic_medium_doc_count": task_gate.get("medium_doc_count", 0),
            "task_semantic_exact_doc_count": task_gate.get("exact_doc_count", 0),
            "task_semantic_exact_strong_doc_count": task_gate.get("exact_strong_doc_count", 0),
            "task_semantic_exact_medium_doc_count": task_gate.get("exact_medium_doc_count", 0),
            "task_match_rate": task_gate.get("task_match_rate", 0.0),
            "task_gate_reason": task_gate.get("task_gate_reason"),
            "task_semantic_correction": task_gate.get("module_correction_message"),
            "task_semantic_module_conflict": bool(task_gate.get("module_conflict")),
            "task_semantic_preferred_modules": task_gate.get("preferred_modules", []),
            "grounding_evidence_score": float(task_gate.get("grounding_evidence_score") or 0.0),
            "grounding_confidence_tier": task_gate.get("grounding_confidence_tier", "low"),
            "semantic_decision_confidence_score": float(task_gate.get("decision_confidence_score") or 0.0),
            "semantic_decision_confidence_tier": task_gate.get("decision_confidence_tier", "low"),
            "preferred_module_troubleshooting_fallback_attempted": preferred_fallback_audit.get("preferred_module_troubleshooting_fallback_attempted", False),
            "preferred_module_troubleshooting_fallback_used": preferred_fallback_audit.get("preferred_module_troubleshooting_fallback_used", False),
            "preferred_module_troubleshooting_fallback_module": preferred_fallback_audit.get("preferred_module_troubleshooting_fallback_module"),
            "preferred_module_troubleshooting_correction": preferred_fallback_audit.get("preferred_module_troubleshooting_correction"),
            "preferred_module_troubleshooting_support_count": preferred_module_troubleshooting_support_count,
        }
        audit.update(fallback_audit)

        exact_support_available = audit.get("exact_module_doc_count", 0) > 0
        if route_info.task_type == TaskType.TROUBLESHOOTING:
            exact_support_available = exact_support_available or (
                audit.get("exact_module_troubleshooting_support_count", 0) > 0
            )
            exact_support_available = exact_support_available or (
                audit.get("preferred_module_troubleshooting_support_count", 0) > 0
            )

        decision_trace = self._build_decision_trace(
            route_info=route_info,
            task_gate=task_gate,
            citation_count=len(citations),
            docs_count=audit.get("docs_retrieved_count", 0),
            exact_support_available=exact_support_available,
            strict_financial_leaf=strict_financial_leaf,
            preferred_fallback_audit=preferred_fallback_audit,
        )
        audit.update(decision_trace)

        summary_safety = self._evaluate_strict_summary_grounding(
            user_query=user_query,
            route_info=route_info,
            mapped_chunks=mapped_chunks,
        )
        if summary_safety.get("applicable"):
            audit["strict_summary_safety_enabled"] = True
            audit["strict_summary_safety_reason"] = summary_safety.get("reason")
            audit["strict_summary_safety_concept_relevance"] = summary_safety.get("concept_relevance")
            audit["strict_summary_safety_threshold"] = summary_safety.get("threshold")
            audit["strict_summary_safety_top_title"] = summary_safety.get("top_title")
            if not summary_safety.get("allow"):
                audit["decision_execution_mode"] = "REFUSE"
                audit["decision_reason"] = "strict_summary_safety_override"
                audit["decision_refusal_reason"] = str(summary_safety.get("reason") or "strict_summary_safety_override")
                audit["decision_confidence_tier"] = "LOW"

        if (
            strict_financial_leaf
            and route_info.task_type in self.DOCS_EXPECTED_TASKS
            and not exact_support_available
            and not audit.get("decision_grounding_signal_present")
        ):
            timings["total_e2e"] = time.perf_counter() - start_time
            audit["verification_status"] = "FAILED_FINANCE_LEAF_NO_EXACT_DOCS"
            return self._build_failure_response(
                trace_id,
                timings,
                audit,
                citations,
                mapped_chunks,
                message=self._build_task_semantic_failure_message(task_gate, route_info),
            )

        if (
            route_info.task_type in self.DOCS_EXPECTED_TASKS
            and task_gate.get("requested_task_signals")
            and not audit.get("decision_grounding_signal_present")
        ):
            if task_gate.get("task_semantic_gate") == "FAILED":
                timings["total_e2e"] = time.perf_counter() - start_time
                audit["verification_status"] = (
                    "FAILED_TASK_MODULE_CORRECTION"
                    if task_gate.get("module_correction_message")
                    else "FAILED_TASK_SEMANTIC_NO_STRONG_MATCH"
                )
                return self._build_failure_response(
                    trace_id,
                    timings,
                    audit,
                    citations,
                    mapped_chunks,
                    message=self._build_task_semantic_failure_message(task_gate, route_info),
                )

        if audit.get("decision_execution_mode") == "REFUSE":
            timings["total_e2e"] = time.perf_counter() - start_time
            audit["verification_status"] = audit.get("verification_status") or "FAILED_DECISION_CONFIDENCE_GATE"
            return self._build_failure_response(
                trace_id,
                timings,
                audit,
                citations,
                mapped_chunks,
                message=self._build_task_semantic_failure_message(task_gate, route_info),
            )

        t0 = time.perf_counter()
        specialized_content = self._build_specialized_lane_response(route_info, user_query, mapped_chunks, audit)
        if specialized_content is not None:
            final_content = self._sanitize_output(specialized_content)
            success, error_msg = self.verifier.run_pass(
                route_info.task_type,
                final_content,
                mapped_chunks,
                module=route_info.module.value if route_info.module != FusionModule.UNKNOWN else route_info.module_family.value,
                schema=context_block,
            )
            timings["specialized_generation"] = time.perf_counter() - t0
            if success:
                audit["verification_status"] = "PASSED"
                audit["sql_validity_score"] = 1.0 if route_info.task_type in self.SQL_TASKS else 0.0
                if route_info.task_type in self.SQL_TASKS:
                    audit["sql_verifier_reason_code"] = "SQL_VERIFIER_PASSED"
                    self._log_sql_decision_event(
                        stage="final_verifier",
                        user_query=user_query,
                        route_info=route_info,
                        module_name=str(audit.get("sql_effective_module") or audit.get("module") or audit.get("module_family") or ""),
                        request_shape=audit.get("sql_request_shape") or {},
                        audit=audit,
                        reason_code=audit["sql_verifier_reason_code"],
                        module_hint=str(audit.get("sql_module_hint") or ""),
                        module_alignment_target=audit.get("sql_module_alignment_target"),
                        selection_path=str(audit.get("sql_selection_path") or ""),
                        verifier_status="PASSED",
                        verifier_reason="PASSED",
                    )
                timings["total_e2e"] = time.perf_counter() - start_time
                return ChatResponse(
                    id=trace_id,
                    created=int(time.time()),
                    model="specialized-lane",
                    choices=[{"message": {"role": "assistant", "content": final_content}}],
                    citations=citations,
                    retrieved_chunks=mapped_chunks,
                    timings=timings,
                    audit=audit,
                )

            audit["verification_status"] = f"FAILED_SPECIALIZED: {error_msg}"
            if route_info.task_type in self.SQL_TASKS:
                audit["sql_verifier_reason_code"] = self._sql_verifier_reason_code(error_msg)
                audit["sql_refusal_reason_code"] = audit.get("sql_refusal_reason_code") or self._sql_refusal_reason_code(
                    audit.get("sql_request_shape") or {},
                    error_msg,
                    audit["verification_status"],
                )
                self._log_sql_decision_event(
                    stage="final_verifier",
                    user_query=user_query,
                    route_info=route_info,
                    module_name=str(audit.get("sql_effective_module") or audit.get("module") or audit.get("module_family") or ""),
                    request_shape=audit.get("sql_request_shape") or {},
                    audit=audit,
                    reason_code=audit["sql_verifier_reason_code"],
                    module_hint=str(audit.get("sql_module_hint") or ""),
                    module_alignment_target=audit.get("sql_module_alignment_target"),
                    selection_path=str(audit.get("sql_selection_path") or ""),
                    verifier_status=audit["verification_status"],
                    verifier_reason=error_msg,
                )
            timings["total_e2e"] = time.perf_counter() - start_time
            return self._build_failure_response(trace_id, timings, audit, citations, mapped_chunks)

        if require_citations and not citations:
            timings["total_e2e"] = time.perf_counter() - start_time
            audit["verification_status"] = "FAILED_NO_CITATIONS"
            return self._build_failure_response(trace_id, timings, audit, citations, mapped_chunks)

        structured_doc_content = self._build_structured_doc_response(route_info, user_query, mapped_chunks)
        if structured_doc_content is not None:
            final_content = self._sanitize_output(structured_doc_content)
            success, error_msg = self.verifier.run_pass(
                route_info.task_type,
                final_content,
                mapped_chunks,
                module=route_info.module.value if route_info.module != FusionModule.UNKNOWN else route_info.module_family.value,
                schema=context_block,
            )
            if success:
                audit["verification_status"] = "PASSED"
                timings["total_e2e"] = time.perf_counter() - start_time
                return ChatResponse(
                    id=trace_id,
                    created=int(time.time()),
                    model="structured-doc-grounding",
                    choices=[{"message": {"role": "assistant", "content": final_content}}],
                    citations=citations,
                    retrieved_chunks=mapped_chunks,
                    timings=timings,
                    audit=audit,
                )
            audit["verification_status"] = f"FAILED_STRUCTURED_DOC: {error_msg}"

        if route_info.task_type in self.DOCS_EXPECTED_TASKS:
            timings["total_e2e"] = time.perf_counter() - start_time
            audit["verification_status"] = audit.get("verification_status") or "FAILED_DOC_GROUNDING_NOT_RECOVERABLE"
            return self._build_failure_response(
                trace_id,
                timings,
                audit,
                citations,
                mapped_chunks,
                message=self._build_task_semantic_failure_message(task_gate, route_info),
            )

        requires_sql = self._requires_sql(route_info, user_query)
        requires_formula = self._requires_formula(route_info, user_query)
        sql_pattern_available = sql_index_used and requires_sql
        formula_pattern_available = formula_index_used and requires_formula

        system_prompt = RAGPrompts.system_prompt_for_task(route_info.task_type, task_config["prompt_template"])
        refined_messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(
                role=Role.USER,
                content=(
                    f"[TASK: {route_info.task_type.value}]\n"
                    f"[MODULE: {route_info.module.value}]\n"
                    f"[MODULE_FAMILY: {route_info.module_family.value}]\n"
                    f"[EXPLICIT_MODULE_ENFORCEMENT: {'yes' if strict_financial_leaf else 'no'}]\n"
                    f"[EXACT_MODULE_ONLY: {self._canonical_module_name(route_info.module) if strict_financial_leaf else 'none'}]\n"
                    f"[FINANCE_SOFT_FALLBACK_USED: {'yes' if audit.get('finance_soft_fallback_used') else 'no'}]\n"
                    f"[TASK_SIGNALS: {', '.join(signal['task'] for signal in task_gate.get('requested_task_signals', [])) or 'none'}]\n"
                    f"[TASK_SEMANTIC_GATE: {audit.get('task_semantic_gate')}]\n"
                    f"[TASK_MODULE_CORRECTION: {audit.get('task_semantic_correction') or 'none'}]\n"
                    f"[SQL_REQUIRED: {'yes' if requires_sql else 'no'}]\n"
                    f"[SQL_PATTERN_REUSE_REQUIRED: {'yes' if sql_pattern_available else 'no'}]\n"
                    f"[FORMULA_REQUIRED: {'yes' if requires_formula else 'no'}]\n"
                    f"[FORMULA_PATTERN_REUSE_REQUIRED: {'yes' if formula_pattern_available else 'no'}]\n"
                    f"CONTEXT:\n{context_block}\n\n"
                    f"QUERY: {user_query}"
                ),
            ),
        ]

        t_quant = self._apply_turbo_quant(route_info, request)
        attempts = 0
        max_attempts = min(MAX_VERIFICATION_RETRIES, 2) + 1
        final_content = FAIL_CLOSED_MESSAGE
        llm_model_name = "antigravity-v6"

        while attempts < max_attempts:
            t0 = time.perf_counter()
            llm_request = ChatRequest(
                messages=refined_messages,
                temperature=t_quant["temperature"],
                top_p=t_quant["top_p"],
                repeat_penalty=t_quant["repeat_penalty"],
                max_tokens=request.max_tokens or 1024,
            )
            llm_response = await self._ensure_llm_client().chat(llm_request)
            timings[f"gen_attempt_{attempts}"] = time.perf_counter() - t0

            llm_model_name = llm_response.get("model") or llm_model_name
            choices = llm_response.get("choices") or []
            if not choices:
                attempts += 1
                audit["verification_status"] = "FAILED_EMPTY_LLM_RESPONSE"
                continue

            candidate_content = self._sanitize_output(choices[0]["message"]["content"])
            if route_info.task_type == TaskType.TROUBLESHOOTING:
                candidate_content = self._inject_troubleshooting_note(
                    candidate_content,
                    audit.get("preferred_module_troubleshooting_correction"),
                )
            if FAIL_CLOSED_MESSAGE in candidate_content and candidate_content.strip() != FAIL_CLOSED_MESSAGE:
                candidate_content = FAIL_CLOSED_MESSAGE
            success, error_msg = self.verifier.run_pass(
                route_info.task_type,
                candidate_content,
                mapped_chunks,
                module=route_info.module.value if route_info.module != FusionModule.UNKNOWN else route_info.module_family.value,
                schema=context_block,
            )
            if success:
                final_content = candidate_content
                audit["verification_status"] = "PASSED"
                audit["sql_validity_score"] = 1.0 if "[SQL]" in candidate_content else 0.0
                break

            attempts += 1
            audit["verification_status"] = f"FAILED: {error_msg}"
            if attempts < max_attempts:
                refined_messages.append(
                    Message(
                        role=Role.USER,
                        content=(
                            f"Verification failed: {error_msg}\n"
                            "Regenerate using only grounded context and grounded citations."
                        ),
                    )
                )

        if audit["verification_status"] != "PASSED":
            final_content = FAIL_CLOSED_MESSAGE

        final_content = self._sanitize_output(final_content)
        timings["total_e2e"] = time.perf_counter() - start_time

        return ChatResponse(
            id=trace_id,
            created=int(time.time()),
            model=llm_model_name,
            choices=[{"message": {"role": "assistant", "content": final_content}}],
            citations=citations,
            retrieved_chunks=mapped_chunks,
            timings=timings,
            audit=audit,
        )
