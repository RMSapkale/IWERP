from typing import Any, Dict

from core.schemas.router import TaskType


TASK_CONFIGS: Dict[TaskType, Dict[str, Any]] = {
    TaskType.TABLE_LOOKUP: {
        "fts_weight": 0.0,
        "vector_weight": 1.0,
        "prompt_template": "Return compact grounded schema facts only. No SQL unless requested.",
    },
    TaskType.SQL_GENERATION: {
        "fts_weight": 0.2,
        "vector_weight": 0.8,
        "prompt_template": "Use curated SQL examples first, then compact schema evidence. Reuse grounded patterns and keep the explanation short.",
    },
    TaskType.SQL_TROUBLESHOOTING: {
        "fts_weight": 0.25,
        "vector_weight": 0.75,
        "prompt_template": "Use grounded SQL examples and schema metadata first. Diagnose the SQL issue, fix it only when grounded, and fail closed if the schema support is weak.",
    },
    TaskType.REPORT_LOGIC: {
        "fts_weight": 0.2,
        "vector_weight": 0.8,
        "prompt_template": "Use curated SQL or report logic examples first, then compact schema evidence.",
    },
    TaskType.FAST_FORMULA_GENERATION: {
        "fts_weight": 0.25,
        "vector_weight": 0.75,
        "prompt_template": "Use grounded Fast Formula examples first. Adapt a close formula pattern, keep the output syntactically plausible, and refuse instead of inventing unsupported formula structures.",
    },
    TaskType.FAST_FORMULA_TROUBLESHOOTING: {
        "fts_weight": 0.3,
        "vector_weight": 0.7,
        "prompt_template": "Use grounded Fast Formula examples and troubleshooting notes first. Explain the error briefly, provide a grounded fix when possible, and refuse unsupported repairs.",
    },
    TaskType.TROUBLESHOOTING: {
        "fts_weight": 0.45,
        "vector_weight": 0.55,
        "prompt_template": "Use troubleshooting docs first, then grounded SQL patterns if present, then minimal schema checks.",
    },
    TaskType.PROCEDURE: {
        "fts_weight": 0.55,
        "vector_weight": 0.45,
        "prompt_template": "Use procedure/setup docs only, keep the answer concise, and refuse instead of inferring missing steps.",
    },
    TaskType.NAVIGATION: {
        "fts_weight": 0.65,
        "vector_weight": 0.35,
        "prompt_template": "Use navigation docs only, keep the path concise, and do not add inferred menu steps.",
    },
    TaskType.INTEGRATION: {
        "fts_weight": 0.4,
        "vector_weight": 0.6,
        "prompt_template": "Use curated integration docs with minimal schema evidence.",
    },
    TaskType.SUMMARY: {
        "fts_weight": 0.5,
        "vector_weight": 0.5,
        "prompt_template": "Use curated docs only and keep the summary grounded and short.",
    },
    TaskType.QA: {
        "fts_weight": 0.45,
        "vector_weight": 0.55,
        "prompt_template": "Use the smallest grounded evidence set available.",
    },
    TaskType.GENERAL: {
        "fts_weight": 0.4,
        "vector_weight": 0.6,
        "prompt_template": "Prefer curated docs, add schema only when necessary, keep the answer short, and refuse when evidence is thin.",
    },
    TaskType.GREETING: {
        "fts_weight": 0.0,
        "vector_weight": 0.0,
        "prompt_template": "Greeting only. No retrieval required.",
    },
}

TASK_CONFIGS[TaskType.FUSION_NAV] = TASK_CONFIGS[TaskType.NAVIGATION]
TASK_CONFIGS[TaskType.FUSION_PROC] = TASK_CONFIGS[TaskType.PROCEDURE]
TASK_CONFIGS[TaskType.FUSION_TROUBLESHOOT] = TASK_CONFIGS[TaskType.TROUBLESHOOTING]
TASK_CONFIGS[TaskType.FUSION_INTEGRATION] = TASK_CONFIGS[TaskType.INTEGRATION]

ENABLE_RAG_VERIFIER = True
ENABLE_SQL_VERIFIER = True
MAX_VERIFICATION_RETRIES = 2
