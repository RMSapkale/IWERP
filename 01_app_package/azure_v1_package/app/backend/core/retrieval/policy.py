from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from core.schemas.curation import CorpusType
from core.schemas.router import FusionModule, TaskType


@dataclass(frozen=True)
class RetrievalBudget:
    max_chunks: int
    max_total_chars: int
    per_corpus_chunks: Dict[str, int]
    per_corpus_chars: Dict[str, int]
    max_schema_objects: int
    candidate_limit: int
    quality_score_min: float = 0.75
    allow_graph_expansion: bool = False


@dataclass(frozen=True)
class RetrievalPlan:
    corpora: List[str]
    budget: RetrievalBudget

    def apply_result_budget(self, chunks: List[dict]) -> List[dict]:
        selected = []
        per_corpus_counts = defaultdict(int)
        per_corpus_chars = defaultdict(int)
        total_chars = 0
        schema_objects = 0

        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            corpus = str(metadata.get("corpus") or "")
            if corpus not in self.corpora:
                continue

            if len(selected) >= self.budget.max_chunks:
                break
            if per_corpus_counts[corpus] >= self.budget.per_corpus_chunks.get(corpus, 0):
                continue

            content = (chunk.get("content") or "").strip()
            remaining_corpus_chars = self.budget.per_corpus_chars.get(corpus, 0) - per_corpus_chars[corpus]
            remaining_total_chars = self.budget.max_total_chars - total_chars
            allowed_chars = min(len(content), remaining_corpus_chars, remaining_total_chars)
            if allowed_chars <= 0:
                continue

            trimmed = dict(chunk)
            trimmed["content"] = content[:allowed_chars].strip()
            if not trimmed["content"]:
                continue

            trusted_objects = metadata.get("trusted_schema_objects") or []
            if corpus == CorpusType.SCHEMA.value:
                projected_schema_count = schema_objects + len(trusted_objects or [trimmed.get("id", "")])
                if projected_schema_count > self.budget.max_schema_objects:
                    continue
                schema_objects = projected_schema_count

            per_corpus_counts[corpus] += 1
            per_corpus_chars[corpus] += len(trimmed["content"])
            total_chars += len(trimmed["content"])
            selected.append(trimmed)

        return selected


class RetrievalPolicy:
    STRICT_FINANCIAL_LEAF_MODULES = {
        FusionModule.PAYABLES.value,
        FusionModule.RECEIVABLES.value,
        FusionModule.GENERAL_LEDGER.value,
        FusionModule.CASH_MANAGEMENT.value,
        FusionModule.ASSETS.value,
        FusionModule.EXPENSES.value,
        FusionModule.TAX.value,
        FusionModule.AP.value,
        FusionModule.AR.value,
        FusionModule.GL.value,
        "CE",
        "FA",
        "EXM",
        "ZX",
    }
    FINANCE_SOFT_FALLBACK_CONFIDENCE_THRESHOLD = 0.78

    TASK_BUDGETS: Dict[TaskType, RetrievalPlan] = {
        TaskType.TABLE_LOOKUP: RetrievalPlan(
            corpora=[CorpusType.SCHEMA.value],
            budget=RetrievalBudget(
                max_chunks=4,
                max_total_chars=1600,
                per_corpus_chunks={CorpusType.SCHEMA.value: 4},
                per_corpus_chars={CorpusType.SCHEMA.value: 1600},
                max_schema_objects=4,
                candidate_limit=8,
                quality_score_min=0.9,
                allow_graph_expansion=False,
            ),
        ),
        TaskType.SQL_GENERATION: RetrievalPlan(
            corpora=[CorpusType.SQL_EXAMPLES.value, CorpusType.SCHEMA_METADATA.value, CorpusType.DOCS.value],
            budget=RetrievalBudget(
                max_chunks=5,
                max_total_chars=2600,
                per_corpus_chunks={
                    CorpusType.SQL_EXAMPLES.value: 3,
                    CorpusType.SCHEMA_METADATA.value: 2,
                    CorpusType.DOCS.value: 1,
                },
                per_corpus_chars={
                    CorpusType.SQL_EXAMPLES.value: 1400,
                    CorpusType.SCHEMA_METADATA.value: 900,
                    CorpusType.DOCS.value: 300,
                },
                max_schema_objects=4,
                candidate_limit=10,
                quality_score_min=0.85,
                allow_graph_expansion=True,
            ),
        ),
        TaskType.SQL_TROUBLESHOOTING: RetrievalPlan(
            corpora=[CorpusType.SQL_EXAMPLES.value, CorpusType.SCHEMA_METADATA.value, CorpusType.DOCS.value],
            budget=RetrievalBudget(
                max_chunks=6,
                max_total_chars=2800,
                per_corpus_chunks={
                    CorpusType.SQL_EXAMPLES.value: 3,
                    CorpusType.SCHEMA_METADATA.value: 2,
                    CorpusType.DOCS.value: 1,
                },
                per_corpus_chars={
                    CorpusType.SQL_EXAMPLES.value: 1400,
                    CorpusType.SCHEMA_METADATA.value: 900,
                    CorpusType.DOCS.value: 500,
                },
                max_schema_objects=4,
                candidate_limit=12,
                quality_score_min=0.85,
                allow_graph_expansion=True,
            ),
        ),
        TaskType.REPORT_LOGIC: RetrievalPlan(
            corpora=[CorpusType.SQL_EXAMPLES.value, CorpusType.SCHEMA_METADATA.value, CorpusType.DOCS.value],
            budget=RetrievalBudget(
                max_chunks=5,
                max_total_chars=2600,
                per_corpus_chunks={
                    CorpusType.SQL_EXAMPLES.value: 3,
                    CorpusType.SCHEMA_METADATA.value: 2,
                    CorpusType.DOCS.value: 1,
                },
                per_corpus_chars={
                    CorpusType.SQL_EXAMPLES.value: 1400,
                    CorpusType.SCHEMA_METADATA.value: 900,
                    CorpusType.DOCS.value: 300,
                },
                max_schema_objects=4,
                candidate_limit=10,
                quality_score_min=0.85,
                allow_graph_expansion=True,
            ),
        ),
        TaskType.FAST_FORMULA_GENERATION: RetrievalPlan(
            corpora=[CorpusType.FAST_FORMULA.value],
            budget=RetrievalBudget(
                max_chunks=4,
                max_total_chars=2400,
                per_corpus_chunks={CorpusType.FAST_FORMULA.value: 4},
                per_corpus_chars={CorpusType.FAST_FORMULA.value: 2400},
                max_schema_objects=0,
                candidate_limit=10,
                quality_score_min=0.75,
                allow_graph_expansion=False,
            ),
        ),
        TaskType.FAST_FORMULA_TROUBLESHOOTING: RetrievalPlan(
            corpora=[CorpusType.FAST_FORMULA.value],
            budget=RetrievalBudget(
                max_chunks=4,
                max_total_chars=2400,
                per_corpus_chunks={CorpusType.FAST_FORMULA.value: 4},
                per_corpus_chars={CorpusType.FAST_FORMULA.value: 2400},
                max_schema_objects=0,
                candidate_limit=10,
                quality_score_min=0.75,
                allow_graph_expansion=False,
            ),
        ),
        TaskType.TROUBLESHOOTING: RetrievalPlan(
            corpora=[CorpusType.TROUBLESHOOTING.value, CorpusType.DOCS.value],
            budget=RetrievalBudget(
                max_chunks=6,
                max_total_chars=3200,
                per_corpus_chunks={
                    CorpusType.TROUBLESHOOTING.value: 4,
                    CorpusType.DOCS.value: 3,
                },
                per_corpus_chars={
                    CorpusType.TROUBLESHOOTING.value: 1800,
                    CorpusType.DOCS.value: 1400,
                },
                max_schema_objects=3,
                candidate_limit=18,
                quality_score_min=0.65,
                allow_graph_expansion=False,
            ),
        ),
        TaskType.PROCEDURE: RetrievalPlan(
            corpora=[CorpusType.DOCS.value],
            budget=RetrievalBudget(
                max_chunks=5,
                max_total_chars=2300,
                per_corpus_chunks={CorpusType.DOCS.value: 5},
                per_corpus_chars={CorpusType.DOCS.value: 2300},
                max_schema_objects=1,
                candidate_limit=16,
                quality_score_min=0.6,
                allow_graph_expansion=False,
            ),
        ),
        TaskType.NAVIGATION: RetrievalPlan(
            corpora=[CorpusType.DOCS.value],
            budget=RetrievalBudget(
                max_chunks=5,
                max_total_chars=2200,
                per_corpus_chunks={CorpusType.DOCS.value: 5},
                per_corpus_chars={CorpusType.DOCS.value: 2200},
                max_schema_objects=1,
                candidate_limit=16,
                quality_score_min=0.6,
                allow_graph_expansion=False,
            ),
        ),
        TaskType.GENERAL: RetrievalPlan(
            corpora=[CorpusType.DOCS.value],
            budget=RetrievalBudget(
                max_chunks=4,
                max_total_chars=2200,
                per_corpus_chunks={CorpusType.DOCS.value: 4},
                per_corpus_chars={CorpusType.DOCS.value: 2200},
                max_schema_objects=1,
                candidate_limit=14,
                quality_score_min=0.6,
                allow_graph_expansion=False,
            ),
        ),
        TaskType.INTEGRATION: RetrievalPlan(
            corpora=[CorpusType.DOCS.value],
            budget=RetrievalBudget(
                max_chunks=4,
                max_total_chars=2200,
                per_corpus_chunks={CorpusType.DOCS.value: 4},
                per_corpus_chars={CorpusType.DOCS.value: 2200},
                max_schema_objects=2,
                candidate_limit=14,
                quality_score_min=0.6,
                allow_graph_expansion=False,
            ),
        ),
    }

    @classmethod
    def for_task(cls, task_type: TaskType) -> RetrievalPlan:
        if task_type in {TaskType.FUSION_NAV}:
            task_type = TaskType.NAVIGATION
        elif task_type in {TaskType.FUSION_PROC}:
            task_type = TaskType.PROCEDURE
        elif task_type in {TaskType.FUSION_TROUBLESHOOT}:
            task_type = TaskType.TROUBLESHOOTING
        elif task_type in {TaskType.FUSION_INTEGRATION}:
            task_type = TaskType.INTEGRATION
        return cls.TASK_BUDGETS.get(task_type, cls.TASK_BUDGETS[TaskType.GENERAL])

    @classmethod
    def is_strict_financial_leaf(cls, module: str | FusionModule | None) -> bool:
        if module is None:
            return False
        return str(module) in cls.STRICT_FINANCIAL_LEAF_MODULES

    @classmethod
    def finance_soft_fallback_threshold(cls) -> float:
        return cls.FINANCE_SOFT_FALLBACK_CONFIDENCE_THRESHOLD

    @staticmethod
    def task_filters_for_corpus(task_type: TaskType, corpus: str) -> List[str]:
        if corpus == CorpusType.SQL.value:
            if task_type in {TaskType.SQL_GENERATION, TaskType.REPORT_LOGIC, TaskType.TROUBLESHOOTING}:
                return [
                    TaskType.SQL_GENERATION.value,
                    TaskType.REPORT_LOGIC.value,
                    TaskType.TROUBLESHOOTING.value,
                ]
            return []

        if corpus == CorpusType.SQL_EXAMPLES.value:
            if task_type in {
                TaskType.SQL_GENERATION,
                TaskType.SQL_TROUBLESHOOTING,
                TaskType.REPORT_LOGIC,
                TaskType.TROUBLESHOOTING,
            }:
                return [
                    TaskType.SQL_GENERATION.value,
                    TaskType.SQL_TROUBLESHOOTING.value,
                    TaskType.REPORT_LOGIC.value,
                    TaskType.TROUBLESHOOTING.value,
                ]
            return []

        if corpus == CorpusType.SCHEMA_METADATA.value:
            if task_type in {
                TaskType.TABLE_LOOKUP,
                TaskType.SQL_GENERATION,
                TaskType.SQL_TROUBLESHOOTING,
                TaskType.REPORT_LOGIC,
                TaskType.TROUBLESHOOTING,
            }:
                return [
                    TaskType.TABLE_LOOKUP.value,
                    TaskType.SQL_GENERATION.value,
                    TaskType.SQL_TROUBLESHOOTING.value,
                    TaskType.REPORT_LOGIC.value,
                    TaskType.TROUBLESHOOTING.value,
                ]
            return [TaskType.TABLE_LOOKUP.value]

        if corpus == CorpusType.FAST_FORMULA.value:
            if task_type in {
                TaskType.FAST_FORMULA_GENERATION,
                TaskType.FAST_FORMULA_TROUBLESHOOTING,
            }:
                return [
                    TaskType.FAST_FORMULA_GENERATION.value,
                    TaskType.FAST_FORMULA_TROUBLESHOOTING.value,
                ]
            return []

        if corpus == CorpusType.TROUBLESHOOTING.value:
            if task_type == TaskType.TROUBLESHOOTING:
                return [TaskType.TROUBLESHOOTING.value]
            return []

        if corpus == CorpusType.DOCS.value:
            if task_type in {TaskType.GENERAL, TaskType.SUMMARY}:
                return [
                    TaskType.PROCEDURE.value,
                    TaskType.NAVIGATION.value,
                    TaskType.TROUBLESHOOTING.value,
                    TaskType.INTEGRATION.value,
                    TaskType.GENERAL.value,
                    TaskType.SUMMARY.value,
                ]
            if task_type in {TaskType.SQL_GENERATION, TaskType.SQL_TROUBLESHOOTING, TaskType.REPORT_LOGIC}:
                return [
                    TaskType.TROUBLESHOOTING.value,
                    TaskType.PROCEDURE.value,
                    TaskType.REPORT_LOGIC.value,
                ]
            if task_type == TaskType.TROUBLESHOOTING:
                return [
                    TaskType.TROUBLESHOOTING.value,
                    TaskType.PROCEDURE.value,
                    TaskType.GENERAL.value,
                    TaskType.NAVIGATION.value,
                    TaskType.INTEGRATION.value,
                    TaskType.SUMMARY.value,
                ]
            return [task_type.value]

        if corpus == CorpusType.SCHEMA.value:
            if task_type in {TaskType.SQL_GENERATION, TaskType.REPORT_LOGIC, TaskType.TROUBLESHOOTING}:
                return [
                    TaskType.TABLE_LOOKUP.value,
                    TaskType.SQL_GENERATION.value,
                    TaskType.REPORT_LOGIC.value,
                    TaskType.TROUBLESHOOTING.value,
                ]
            return [TaskType.TABLE_LOOKUP.value]

        return [task_type.value]
