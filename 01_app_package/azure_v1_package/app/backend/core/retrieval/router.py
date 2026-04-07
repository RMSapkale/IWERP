import re
from typing import Dict, List

from core.grounding.task_semantics import TaskSemanticAnalyzer
from core.schemas.router import FusionModule, ModuleFamily, RouterResponse, TaskType, module_families_for_value


class TaskRouter:
    """
    Module-first router for constrained Oracle Fusion retrieval.
    """

    MODULE_KEYWORDS: Dict[FusionModule, List[str]] = {
        FusionModule.PAYABLES: ["accounts payable", "invoice", "payment", "supplier", "ap", "poz_", "invoice validation"],
        FusionModule.RECEIVABLES: ["accounts receivable", "receipt", "customer", "bill", "ar", "ra_", "cash receipt", "rmcs", "revenue management cloud service"],
        FusionModule.GENERAL_LEDGER: ["general ledger", "journal", "ledger", "period", "gl", "code combination"],
        FusionModule.CASH_MANAGEMENT: ["cash management", "bank", "statement", "reconciliation", "ce_"],
        FusionModule.ASSETS: ["assets", "depreciation", "retirement", "fa_"],
        FusionModule.EXPENSES: ["expenses", "reimbursement", "audit", "exm_"],
        FusionModule.PROCUREMENT: ["procurement", "purchase order", "po ", "requisition", "poz_", "por_", "supplier site"],
        FusionModule.SCM: ["inventory", "shipping", "logistics", "order management", "inv_", "wsh_", "doo_", "egp_"],
        FusionModule.HCM: ["hcm", "payroll", "employee", "assignment", "fast formula", "database item", "formula type", "per_", "pay_"],
        FusionModule.PROJECTS: ["projects", "billing", "revenue", "pjt", "pjc", "pjf"],
        FusionModule.TAX: ["tax", "zx_"],
        FusionModule.COMMON: ["fnd_", "gl_code_combinations", "ess", "enterprise scheduler", "scheduled process", "scheduled processes", "job definition", "job set"],
    }

    MODULE_EXPLICIT_TERMS: Dict[FusionModule, List[str]] = {
        FusionModule.PAYABLES: ["payables", "accounts payable", "ap", "ap_", "oracle fusion payables"],
        FusionModule.RECEIVABLES: ["receivables", "accounts receivable", "ar", "ar_", "oracle fusion receivables", "rmcs", "revenue management cloud service"],
        FusionModule.GENERAL_LEDGER: ["general ledger", "gl", "gl_", "oracle fusion general ledger"],
        FusionModule.CASH_MANAGEMENT: ["cash management", "ce_", "oracle fusion cash management"],
        FusionModule.ASSETS: ["assets", "fa_", "oracle fusion assets"],
        FusionModule.EXPENSES: ["expenses", "expense report", "exm_", "oracle fusion expenses"],
        FusionModule.PROCUREMENT: ["procurement", "purchasing", "sourcing", "supplier portal", "oracle fusion purchasing"],
        FusionModule.SCM: ["supply chain", "inventory management", "order management", "shipping", "planning", "oracle fusion scm"],
        FusionModule.HCM: ["hcm", "core hr", "recruiting", "benefits", "absence management", "talent management", "fast formula", "oracle fusion hcm"],
        FusionModule.PROJECTS: ["projects", "ppm", "oracle fusion projects"],
        FusionModule.TAX: ["tax", "oracle fusion tax"],
        FusionModule.COMMON: ["oracle fusion common", "shared", "shared services", "fnd_", "ess", "enterprise scheduler", "scheduled process", "scheduled processes", "job definition", "job set"],
    }

    FAMILY_KEYWORDS: Dict[ModuleFamily, List[str]] = {
        ModuleFamily.CRM: ["crm", "sales", "opportunity", "lead", "channel", "customer relationship", "zca_", "moo_", "mkt_", "psc_", "cso_"],
        ModuleFamily.HCM: ["hcm", "human capital", "recruiting", "performance", "benefits", "absence", "talent", "payroll", "employee", "worker"],
        ModuleFamily.FINANCIALS: ["financials", "payables", "receivables", "general ledger", "cash management", "tax", "assets", "expense", "subledger", "payments"],
        ModuleFamily.PROJECTS: ["projects", "ppm", "billing", "costing", "grant"],
        ModuleFamily.PROCUREMENT: ["procurement", "purchasing", "sourcing", "supplier portal", "self service procurement", "requisition", "purchase order"],
        ModuleFamily.INVOICING: ["invoicing", "invoice extract", "invoice interface"],
        ModuleFamily.SCM: ["supply chain", "inventory", "order management", "manufacturing", "shipping", "planning", "product management", "scm"],
        ModuleFamily.COMMON: ["shared", "fnd_"],
    }

    FAMILY_EXPLICIT_TERMS: Dict[ModuleFamily, List[str]] = {
        ModuleFamily.CRM: ["crm", "oracle fusion crm", "sales cloud"],
        ModuleFamily.HCM: ["hcm", "oracle fusion hcm", "human capital management"],
        ModuleFamily.FINANCIALS: ["financials", "oracle fusion financials"],
        ModuleFamily.PROJECTS: ["projects", "oracle fusion projects", "ppm"],
        ModuleFamily.PROCUREMENT: ["procurement", "oracle fusion procurement"],
        ModuleFamily.INVOICING: ["invoicing", "oracle fusion invoicing"],
        ModuleFamily.SCM: ["scm", "oracle fusion scm", "supply chain"],
        ModuleFamily.COMMON: ["oracle fusion common", "shared services"],
    }

    FAMILY_MODULE_CANDIDATES: Dict[ModuleFamily, List[FusionModule]] = {
        ModuleFamily.CRM: [FusionModule.UNKNOWN],
        ModuleFamily.HCM: [FusionModule.HCM],
        ModuleFamily.FINANCIALS: [
            FusionModule.PAYABLES,
            FusionModule.RECEIVABLES,
            FusionModule.GENERAL_LEDGER,
            FusionModule.CASH_MANAGEMENT,
            FusionModule.ASSETS,
            FusionModule.EXPENSES,
            FusionModule.TAX,
        ],
        ModuleFamily.PROJECTS: [FusionModule.PROJECTS],
        ModuleFamily.PROCUREMENT: [FusionModule.PROCUREMENT],
        ModuleFamily.INVOICING: [FusionModule.PAYABLES, FusionModule.RECEIVABLES],
        ModuleFamily.SCM: [FusionModule.SCM],
        ModuleFamily.COMMON: [FusionModule.COMMON],
    }

    TASK_KEYWORDS: Dict[TaskType, List[str]] = {
        TaskType.TABLE_LOOKUP: [
            "table",
            "column",
            "view",
            "schema",
            "pk",
            "primary key",
            "fk",
            "foreign key",
            "metadata",
            "which object",
        ],
        TaskType.SQL_TROUBLESHOOTING: [
            "sql error",
            "query error",
            "fix sql",
            "debug sql",
            "invalid identifier",
            "missing expression",
            "ora-",
            "sql troubleshooting",
        ],
        TaskType.SQL_GENERATION: ["exact sql", "sql", "query", "select", "join", "where clause", "report query"],
        TaskType.FAST_FORMULA_TROUBLESHOOTING: [
            "fast formula error",
            "formula error",
            "compile formula",
            "compile error",
            "database item error",
            "context error",
            "fix formula",
            "debug formula",
        ],
        TaskType.FAST_FORMULA_GENERATION: [
            "fast formula",
            "formula type",
            "database item",
            "contexts",
            "ff formula",
            "write a formula",
        ],
        TaskType.TROUBLESHOOTING: ["error", "fix", "issue", "debug", "failed", "warning", "not working"],
        TaskType.NAVIGATION: ["navigate", "menu", "where is", "path", "navigator", "click"],
        TaskType.PROCEDURE: ["how to", "steps", "procedure", "create", "setup", "configure"],
        TaskType.INTEGRATION: ["api", "rest", "soap", "fbdi", "integration", "endpoint", "web service"],
        TaskType.REPORT_LOGIC: ["report query", "bi publisher", "otbi", "extract", "subject area", "data model", "xdm", "analysis"],
        TaskType.SUMMARY: ["summarize", "summary", "overview", "what is"],
        TaskType.GREETING: ["hi", "hello", "hey", "greetings", "who are you"],
        TaskType.QA: ["question", "answer", "tell me"],
    }

    TASK_PRIORITIES: Dict[TaskType, float] = {
        TaskType.TABLE_LOOKUP: 1.6,
        TaskType.SQL_TROUBLESHOOTING: 1.8,
        TaskType.SQL_GENERATION: 1.7,
        TaskType.FAST_FORMULA_TROUBLESHOOTING: 1.8,
        TaskType.FAST_FORMULA_GENERATION: 1.75,
        TaskType.TROUBLESHOOTING: 1.4,
        TaskType.NAVIGATION: 1.2,
        TaskType.PROCEDURE: 1.2,
        TaskType.INTEGRATION: 1.2,
        TaskType.REPORT_LOGIC: 1.3,
        TaskType.SUMMARY: 1.0,
        TaskType.QA: 0.9,
        TaskType.GREETING: 2.0,
    }

    INTENT_REQUIRED_SIGNALS: Dict[TaskType, List[List[str]]] = {
        TaskType.SQL_TROUBLESHOOTING: [
            ["sql", "query", "select", "join"],
            ["error", "ora-", "fix", "debug", "troubleshoot", "repair", "invalid identifier"],
        ],
        TaskType.SQL_GENERATION: [
            ["sql", "query", "select", "join", "report query"],
        ],
        TaskType.FAST_FORMULA_TROUBLESHOOTING: [
            ["fast formula", "formula"],
            ["error", "compile", "debug", "fix", "troubleshoot", "context", "database item"],
        ],
        TaskType.FAST_FORMULA_GENERATION: [
            ["fast formula", "formula"],
            ["write", "generate", "create", "formula type", "database item", "contexts"],
        ],
        TaskType.TROUBLESHOOTING: [
            ["troubleshoot", "troubleshooting", "debug", "resolve", "resolution", "root cause", "what causes", "why is", "why are", "common failures"],
            ["error", "errors", "issue", "issues", "failure", "failures", "failed", "warning", "problem", "problems", "common failures"],
        ],
        TaskType.PROCEDURE: [
            ["how do you", "what steps", "steps", "procedure", "create", "setup", "configure", "process"],
        ],
        TaskType.NAVIGATION: [
            ["navigate", "where is", "path", "navigator", "menu", "click"],
        ],
        TaskType.INTEGRATION: [
            ["api", "rest", "soap", "fbdi", "integration", "endpoint", "web service"],
        ],
        TaskType.REPORT_LOGIC: [
            ["bi publisher", "otbi", "subject area", "report query", "data model", "xdm", "analysis", "extract"],
        ],
        TaskType.SUMMARY: [
            ["what is", "what are", "purpose of", "summary", "overview", "summarize"],
        ],
    }
    INTENT_OPTIONAL_SIGNALS: Dict[TaskType, List[str]] = {
        TaskType.TROUBLESHOOTING: ["status", "diagnose", "cause", "causes", "resolution", "faq"],
        TaskType.PROCEDURE: ["required", "workflow", "task", "steps are required", "prerequisite"],
        TaskType.SUMMARY: ["purpose", "important", "overview"],
        TaskType.GENERAL: ["when should", "before updating", "what should be checked"],
    }
    INTENT_NEGATIVE_SIGNALS: Dict[TaskType, List[str]] = {
        TaskType.TROUBLESHOOTING: ["what is", "what are", "purpose of"],
        TaskType.PROCEDURE: ["error", "errors", "issue", "issues", "failed", "debug", "troubleshoot", "common failures"],
        TaskType.SUMMARY: ["how do you", "what steps", "troubleshoot", "debug", "fix"],
    }

    def _count_hits(self, query_lower: str, keywords: List[str]) -> int:
        hits = 0
        for keyword in keywords:
            if " " in keyword:
                if keyword in query_lower:
                    hits += 1
            elif re.search(rf"\b{re.escape(keyword)}\b", query_lower):
                hits += 1
        return hits

    def _match_patterns(self, query_lower: str, patterns: List[str]) -> List[str]:
        matches: List[str] = []
        for pattern in patterns:
            if " " in pattern:
                if pattern in query_lower:
                    matches.append(pattern)
            elif re.search(rf"\b{re.escape(pattern)}\b", query_lower):
                matches.append(pattern)
        return matches

    def _intent_cluster_score(self, query_lower: str, task_type: TaskType) -> tuple[float, List[str], List[str]]:
        required_groups = self.INTENT_REQUIRED_SIGNALS.get(task_type, [])
        optional_patterns = self.INTENT_OPTIONAL_SIGNALS.get(task_type, [])
        negative_patterns = self.INTENT_NEGATIVE_SIGNALS.get(task_type, [])

        matched_required: List[str] = []
        if required_groups:
            for group in required_groups:
                group_matches = self._match_patterns(query_lower, group)
                if not group_matches:
                    return 0.0, [], []
                matched_required.extend(group_matches[:1])

        optional_matches = self._match_patterns(query_lower, optional_patterns)
        negative_matches = self._match_patterns(query_lower, negative_patterns)
        score = 0.35 * len(matched_required) + 0.12 * len(optional_matches) - 0.18 * len(negative_matches)
        if matched_required:
            score += 0.15
        return max(score, 0.0), list(dict.fromkeys(matched_required + optional_matches)), list(dict.fromkeys(negative_matches))

    def _preferred_modules_for_query(self, query: str) -> List[str]:
        profile = TaskSemanticAnalyzer.extract_query_signals(query)
        top_task = str(profile.get("top_task") or "").strip()
        if not top_task:
            return []
        config = TaskSemanticAnalyzer.TASK_CONFIGS.get(top_task, {})
        return [str(module_name).strip() for module_name in config.get("preferred_modules", []) if str(module_name).strip()]

    def route(self, query: str) -> RouterResponse:
        query_lower = query.lower()
        preferred_modules = self._preferred_modules_for_query(query)
        negative_signals: List[str] = []
        explicit_sql_request = bool(
            re.search(r"\b(sql|query|select|join)\b", query_lower)
            and re.search(r"\b(generate|write|create|fix|debug|troubleshoot|repair|optimi[sz]e)\b", query_lower)
        )
        explicit_sql_troubleshooting = bool(
            re.search(r"\b(sql|query|select|join)\b", query_lower)
            and re.search(r"\b(fix|debug|troubleshoot|repair|error|ora-)\b", query_lower)
        )
        explicit_formula_request = bool(
            re.search(r"\b(fast formula|formula)\b", query_lower)
            and re.search(r"\b(write|generate|create|fix|debug|troubleshoot)\b", query_lower)
        )
        explicit_formula_troubleshooting = bool(
            re.search(r"\b(fast formula|formula)\b", query_lower)
            and re.search(r"\b(fix|debug|troubleshoot|error|compile)\b", query_lower)
        )
        explicit_troubleshooting_request = bool(
            re.search(r"\b(troubleshoot|troubleshooting|debug|resolve|resolution|root cause|why is|why are)\b", query_lower)
            or "issues with" in query_lower
            or "issue with" in query_lower
            or "not working" in query_lower
            or "what causes" in query_lower
            or "common failure" in query_lower
            or "common failures" in query_lower
        )
        explicit_report_logic_request = bool(
            re.search(
                r"\b(bi publisher|otbi|subject area|report query|data model|xdm|analysis|extract)\b",
                query_lower,
            )
        )

        module_scores: Dict[FusionModule, float] = {}
        explicit_module_hits: Dict[FusionModule, int] = {}
        for module, keywords in self.MODULE_KEYWORDS.items():
            generic_hits = self._count_hits(query_lower, keywords)
            explicit_hits = self._count_hits(query_lower, self.MODULE_EXPLICIT_TERMS.get(module, []))
            module_scores[module] = (explicit_hits * 3.0) + generic_hits
            explicit_module_hits[module] = explicit_hits

        if "common failures" in query_lower or "common failure" in query_lower:
            module_scores[FusionModule.COMMON] = max(0.0, module_scores.get(FusionModule.COMMON, 0.0) - 3.5)
            family_scores_hint = "common_failures_phrase_demoted_common_module"
            negative_signals.append(family_scores_hint)

        family_scores: Dict[ModuleFamily, float] = {}
        explicit_family_hits: Dict[ModuleFamily, int] = {}
        for family, keywords in self.FAMILY_KEYWORDS.items():
            generic_hits = self._count_hits(query_lower, keywords)
            explicit_hits = self._count_hits(query_lower, self.FAMILY_EXPLICIT_TERMS.get(family, []))
            family_scores[family] = (explicit_hits * 3.0) + generic_hits
            explicit_family_hits[family] = explicit_hits

        ranked_modules = sorted(
            ((score, module) for module, score in module_scores.items() if score > 0),
            key=lambda item: (-item[0], item[1].value),
        )
        ranked_families = sorted(
            ((score, family) for family, score in family_scores.items() if score > 0),
            key=lambda item: (-item[0], item[1].value),
        )

        identified_module = ranked_modules[0][1] if ranked_modules else FusionModule.UNKNOWN
        max_module_hits = float(ranked_modules[0][0]) if ranked_modules else 0.0
        module_explicit = any(explicit_module_hits.values())

        identified_family = (
            ranked_families[0][1].value
            if ranked_families
            else next(iter(module_families_for_value(identified_module.value)))
        )
        max_family_hits = float(ranked_families[0][0]) if ranked_families else 0.0

        if module_explicit:
            explicit_ranked_modules = sorted(
                ((hits, module) for module, hits in explicit_module_hits.items() if hits > 0),
                key=lambda item: (-item[0], item[1].value),
            )
            identified_module = explicit_ranked_modules[0][1]
            identified_family = next(iter(module_families_for_value(identified_module.value)))

        if identified_module != FusionModule.UNKNOWN:
            module_family = next(iter(module_families_for_value(identified_module.value)))
        else:
            module_family = identified_family

        family_explicit = any(explicit_family_hits.values())
        module_candidates = [module.value for _, module in ranked_modules[:3]]
        disambiguation_required = False
        preferred_family_resolution = False

        if family_explicit and not module_explicit:
            family_key = ModuleFamily(module_family)
            module_candidates = [
                candidate.value
                for candidate in self.FAMILY_MODULE_CANDIDATES.get(family_key, [])
                if candidate != FusionModule.UNKNOWN
            ]
            preferred_candidates: List[str] = []
            for candidate in preferred_modules:
                candidate_family = next(iter(module_families_for_value(candidate)), ModuleFamily.UNKNOWN.value)
                if candidate in module_candidates or candidate_family == family_key.value:
                    preferred_candidates.append(candidate)
            if preferred_candidates:
                deduped_preferred = list(dict.fromkeys(preferred_candidates))
                module_candidates = deduped_preferred
                if len(deduped_preferred) == 1:
                    try:
                        identified_module = FusionModule(deduped_preferred[0])
                    except ValueError:
                        identified_module = FusionModule.UNKNOWN
                    disambiguation_required = False
                    preferred_family_resolution = True
                else:
                    disambiguation_required = True
                    identified_module = FusionModule.UNKNOWN
            elif module_candidates:
                disambiguation_required = len(module_candidates) > 1
                identified_module = FusionModule.UNKNOWN if disambiguation_required else self.FAMILY_MODULE_CANDIDATES[family_key][0]

        if not module_explicit and len(ranked_modules) >= 2 and not preferred_family_resolution:
            top_score = ranked_modules[0][0]
            second_score = ranked_modules[1][0]
            if (top_score - second_score) <= 1.0:
                disambiguation_required = True
                module_candidates = [module.value for _, module in ranked_modules[:3]]

        identified_task = TaskType.GENERAL
        best_task_score = 0.0
        best_task_hits = 0
        best_intent_signals: List[str] = []
        best_negative_signals: List[str] = []

        explicit_schema_object = bool(re.search(r"\b[A-Z][A-Z0-9_]{3,}\b", query))

        for task_type, keywords in self.TASK_KEYWORDS.items():
            hits = self._count_hits(query_lower, keywords)
            cluster_score, intent_signals, cluster_negatives = self._intent_cluster_score(query_lower, task_type)
            score = hits * self.TASK_PRIORITIES.get(task_type, 1.0)
            score += cluster_score
            if explicit_schema_object and task_type == TaskType.TABLE_LOOKUP and not explicit_sql_request and not explicit_formula_request:
                score += 1.0
            if score > best_task_score:
                best_task_score = score
                best_task_hits = hits
                identified_task = task_type
                best_intent_signals = intent_signals
                best_negative_signals = cluster_negatives

        if explicit_formula_troubleshooting:
            identified_task = TaskType.FAST_FORMULA_TROUBLESHOOTING
            best_intent_signals = ["explicit_formula_troubleshooting"]
        elif explicit_formula_request:
            identified_task = TaskType.FAST_FORMULA_GENERATION
            best_intent_signals = ["explicit_formula_request"]
        elif explicit_sql_troubleshooting:
            identified_task = TaskType.SQL_TROUBLESHOOTING
            best_intent_signals = ["explicit_sql_troubleshooting"]
        elif explicit_sql_request:
            identified_task = TaskType.SQL_GENERATION
            best_intent_signals = ["explicit_sql_request"]
        elif explicit_troubleshooting_request:
            identified_task = TaskType.TROUBLESHOOTING
            best_intent_signals = ["explicit_troubleshooting_request"]
        elif explicit_report_logic_request:
            identified_task = TaskType.REPORT_LOGIC
            best_intent_signals = ["explicit_report_logic_request"]

        if identified_task == TaskType.GENERAL and explicit_schema_object:
            identified_task = TaskType.TABLE_LOOKUP
            best_intent_signals = ["explicit_schema_object"]

        confidence = 0.25
        if identified_task != TaskType.GENERAL:
            confidence += 0.2
        if identified_module != FusionModule.UNKNOWN:
            confidence += 0.2
        confidence += min(0.25, (best_task_hits + max(max_module_hits, max_family_hits)) * 0.08)
        if disambiguation_required:
            confidence = min(confidence, 0.7)
        elif preferred_family_resolution:
            confidence = max(confidence, 0.82)

        top_module_score = ranked_modules[0][0] if ranked_modules else 0.0
        second_module_score = ranked_modules[1][0] if len(ranked_modules) > 1 else 0.0
        module_confidence = 0.2
        if identified_module != FusionModule.UNKNOWN or module_candidates:
            module_confidence += 0.2
        if top_module_score > 0.0:
            module_confidence += min(0.35, top_module_score * 0.08)
            module_confidence += min(0.2, max(top_module_score - second_module_score, 0.0) * 0.08)
        if disambiguation_required:
            module_confidence = min(module_confidence, 0.72)
        elif preferred_family_resolution:
            module_confidence = max(module_confidence, 0.78)

        intent_confidence = 0.2
        if identified_task != TaskType.GENERAL:
            intent_confidence += 0.2
        if best_intent_signals:
            intent_confidence += min(0.35, 0.12 * len(best_intent_signals))
        if best_task_hits:
            intent_confidence += min(0.15, best_task_hits * 0.06)
        if best_negative_signals:
            intent_confidence = max(0.15, intent_confidence - min(0.25, 0.08 * len(best_negative_signals)))

        confidence = min(confidence, max(module_confidence, intent_confidence, confidence))
        module_score_breakdown = {
            module.value: round(float(score), 4)
            for score, module in ranked_modules[:5]
        }

        return RouterResponse(
            task_type=identified_task,
            module=identified_module,
            module_family=ModuleFamily(module_family),
            confidence=min(confidence, 0.95),
            module_confidence=min(module_confidence, 0.95),
            intent_confidence=min(intent_confidence, 0.95),
            module_candidates=module_candidates,
            intent_signals=list(dict.fromkeys(best_intent_signals)),
            negative_signals=list(dict.fromkeys(negative_signals + best_negative_signals)),
            module_score_breakdown=module_score_breakdown,
            module_explicit=module_explicit,
            disambiguation_required=disambiguation_required,
            reasoning=(
                f"Module-first routing matched {max_module_hits} module cues, {max_family_hits} family cues and "
                f"{best_task_hits} task cues. Explicit module={module_explicit}. "
                f"Module confidence={min(module_confidence, 0.95):.2f}. Intent confidence={min(intent_confidence, 0.95):.2f}. "
                f"Disambiguation required={disambiguation_required}."
            ),
        )
