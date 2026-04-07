from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

class TaskType(str, Enum):
    QA = "qa"
    SUMMARY = "summary"
    TABLE_LOOKUP = "table_lookup"
    SQL_GENERATION = "sql_generation"
    SQL_TROUBLESHOOTING = "sql_troubleshooting"
    TROUBLESHOOTING = "troubleshooting"
    FAST_FORMULA_GENERATION = "fast_formula_generation"
    FAST_FORMULA_TROUBLESHOOTING = "fast_formula_troubleshooting"
    NAVIGATION = "navigation"
    PROCEDURE = "procedure"
    INTEGRATION = "integration"
    REPORT_LOGIC = "report_logic"
    FUSION_NAV = "fusion_nav"  # Backwards compatibility
    FUSION_PROC = "fusion_proc"
    FUSION_TROUBLESHOOT = "fusion_troubleshoot"
    FUSION_INTEGRATION = "fusion_integration"
    GENERAL = "general"
    GREETING = "greeting"

class FusionModule(str, Enum):
    PAYABLES = "Payables"
    RECEIVABLES = "Receivables"
    GENERAL_LEDGER = "General Ledger"
    CASH_MANAGEMENT = "Cash Management"
    ASSETS = "Assets"
    EXPENSES = "Expenses"
    PROCUREMENT = "Procurement"
    SCM = "SCM"
    HCM = "HCM"
    PROJECTS = "Projects"
    TAX = "Tax"
    COMMON = "Common"
    AP = "AP" # Backwards compatibility
    AR = "AR"
    GL = "GL"
    UNKNOWN = "UNKNOWN"

class ModuleFamily(str, Enum):
    CRM = "CRM"
    HCM = "HCM"
    FINANCIALS = "Financials"
    PROJECTS = "Projects"
    PROCUREMENT = "Procurement"
    INVOICING = "Invoicing"
    SCM = "SCM"
    COMMON = "Common"
    UNKNOWN = "UNKNOWN"


EXACT_MODULE_TO_FAMILY = {
    FusionModule.PAYABLES.value: ModuleFamily.FINANCIALS.value,
    FusionModule.AP.value: ModuleFamily.FINANCIALS.value,
    FusionModule.RECEIVABLES.value: ModuleFamily.FINANCIALS.value,
    FusionModule.AR.value: ModuleFamily.FINANCIALS.value,
    FusionModule.GENERAL_LEDGER.value: ModuleFamily.FINANCIALS.value,
    FusionModule.GL.value: ModuleFamily.FINANCIALS.value,
    FusionModule.CASH_MANAGEMENT.value: ModuleFamily.FINANCIALS.value,
    FusionModule.ASSETS.value: ModuleFamily.FINANCIALS.value,
    FusionModule.EXPENSES.value: ModuleFamily.FINANCIALS.value,
    FusionModule.TAX.value: ModuleFamily.FINANCIALS.value,
    FusionModule.PROCUREMENT.value: ModuleFamily.PROCUREMENT.value,
    FusionModule.SCM.value: ModuleFamily.SCM.value,
    FusionModule.HCM.value: ModuleFamily.HCM.value,
    FusionModule.PROJECTS.value: ModuleFamily.PROJECTS.value,
    FusionModule.COMMON.value: ModuleFamily.COMMON.value,
    FusionModule.UNKNOWN.value: ModuleFamily.UNKNOWN.value,
}

MODULE_FAMILY_ALIASES = {
    ModuleFamily.CRM.value: {
        "CRM",
        "Sales",
        "Customer_Relationship",
        "Customer Relationship",
        "Channel_Revenue_Management",
        "Channel Revenue Management",
    },
    ModuleFamily.HCM.value: {
        "HCM",
        "Core HR",
        "Recruiting",
        "Performance Management",
        "Benefits",
        "Absence Management",
        "Talent Management",
        "Payroll Interface",
        "Learning",
        "Payroll",
    },
    ModuleFamily.FINANCIALS.value: {
        "Financials",
        "Payables",
        "AP",
        "Receivables",
        "AR",
        "General Ledger",
        "GL",
        "Cash Management",
        "Assets",
        "Tax",
        "Expenses",
        "Payments",
        "Subledger Accounting",
        "Invoicing",
        "Foundation Services",
        "Lease Accounting",
        "Revenue Management",
    },
    ModuleFamily.PROJECTS.value: {
        "Projects",
        "PPM",
        "Billing",
        "Costing",
        "Grants",
    },
    ModuleFamily.PROCUREMENT.value: {
        "Procurement",
        "Purchasing",
        "Sourcing",
        "Supplier Portal",
        "Self Service Procurement",
        "Contracts",
    },
    ModuleFamily.INVOICING.value: {
        "Invoicing",
    },
    ModuleFamily.SCM.value: {
        "SCM",
        "Supply Chain",
        "Inventory Management",
        "Order Management",
        "Manufacturing",
        "Shipping",
        "Planning",
        "Product Management",
        "Cost Management",
        "Pricing",
    },
    ModuleFamily.COMMON.value: {
        "Common",
        "Shared",
        "Foundation",
        "Application Toolkit",
        "Application Development Framework",
        "Application Dictionary",
    },
}


def module_families_for_value(value: Optional[str]) -> Set[str]:
    if not value:
        return {ModuleFamily.UNKNOWN.value}

    text = str(value).strip()
    if not text:
        return {ModuleFamily.UNKNOWN.value}

    if text in MODULE_FAMILY_ALIASES:
        return {text}

    family = EXACT_MODULE_TO_FAMILY.get(text)
    if family:
        return {family}

    matched = {family_name for family_name, aliases in MODULE_FAMILY_ALIASES.items() if text in aliases}
    return matched or {ModuleFamily.UNKNOWN.value}


class RouterResponse(BaseModel):
    task_type: TaskType = Field(..., description="Categorized task type")
    module: FusionModule = Field(FusionModule.UNKNOWN, description="Identified Oracle Fusion module")
    module_family: ModuleFamily = Field(ModuleFamily.UNKNOWN, description="Top-level taxonomy family")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    module_confidence: float = Field(0.0, ge=0.0, le=1.0)
    intent_confidence: float = Field(0.0, ge=0.0, le=1.0)
    module_candidates: List[str] = Field(default_factory=list, description="Candidate exact modules ordered by score")
    intent_signals: List[str] = Field(default_factory=list, description="Positive intent evidence matched in the query")
    negative_signals: List[str] = Field(default_factory=list, description="Negative intent/module evidence that reduced confidence")
    module_score_breakdown: Dict[str, float] = Field(default_factory=dict, description="Module score snapshot for traceability")
    module_explicit: bool = Field(False, description="Whether an explicit module cue was found in the query")
    disambiguation_required: bool = Field(False, description="Whether retrieval evidence should choose the final exact module")
    reasoning: Optional[str] = None
