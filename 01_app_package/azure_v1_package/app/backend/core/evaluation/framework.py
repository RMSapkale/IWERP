from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional


PRIMARY_METRIC_KEYS = [
    "trusted_outcome_pct",
    "over_refusal_pct",
    "refusal_correctness_pct",
    "hallucination_pct",
    "wrong_module_pct",
    "semantic_correctness_pct",
    "citation_presence_pct",
    "citation_correctness_pct",
    "verifier_approved_pct",
    "grounding_supported_answer_rate_pct",
]

TASK_SPECIFIC_METRIC_KEYS = {
    "sql": [
        "verifier_pass_pct",
        "table_correctness_pct",
        "column_correctness_pct",
        "join_correctness_pct",
        "semantic_sql_match_pct",
        "refusal_correctness_pct",
        "hallucinated_schema_rate_pct",
    ],
    "fast_formula": [
        "verifier_parser_pass_pct",
        "variable_reference_correctness_pct",
        "semantic_formula_correctness_pct",
        "grounded_derivation_correctness_pct",
        "refusal_correctness_pct",
        "hallucination_pct",
    ],
    "procedure": [
        "step_completeness_pct",
        "step_order_correctness_pct",
        "prerequisite_presence_pct",
        "role_context_correctness_pct",
        "citation_coverage_pct",
        "semantic_correctness_pct",
    ],
    "troubleshooting": [
        "symptom_match_pct",
        "root_cause_relevance_pct",
        "resolution_usefulness_pct",
        "resolution_completeness_pct",
        "citation_coverage_pct",
        "semantic_correctness_pct",
    ],
    "summary": [
        "semantic_correctness_pct",
        "citation_correctness_pct",
        "answer_completeness_pct",
        "groundedness_support_pct",
    ],
    "refusal": [
        "refusal_correctness_pct",
        "safe_refusal_rate_pct",
        "false_refusal_rate_pct",
        "unsafe_answer_rate_pct",
    ],
}

SCORING_RUBRIC: Dict[str, Any] = {
    "version": "2026-04-02.production_scale_v1",
    "primary_metrics": PRIMARY_METRIC_KEYS,
    "task_specific_metrics": TASK_SPECIFIC_METRIC_KEYS,
    "primary_verdict_rules": {
        "grounded_correct": "pass",
        "safe_refusal_correct": "pass",
        "wrong-module answer": "fail",
        "hallucination_error": "fail",
        "semantic_mismatch": "fail",
        "missing-citation": "fail",
        "verifier_failure": "fail",
        "sql_failure": "fail",
    },
    "policy": {
        "lexical_metrics_secondary_only": True,
        "lexical_metrics_never_primary_for_sql": True,
        "lexical_metrics_never_primary_for_fast_formula": True,
        "citation_rich_but_semantically_wrong_fails": True,
        "semantic_correct_with_wording_variation_can_pass": True,
        "refusal_cases_scored_independently": True,
    },
    "task_thresholds": {
        "sql": {"pass_score_min": 0.85},
        "fast_formula": {"pass_score_min": 0.8},
        "procedure": {"pass_score_min": 0.8},
        "troubleshooting": {"pass_score_min": 0.8},
        "summary": {"pass_score_min": 0.75},
        "refusal": {"pass_score_min": 0.9},
    },
}


def clamp(value: float, floor: float = 0.0, ceiling: float = 1.0) -> float:
    return max(floor, min(ceiling, value))


def normalize_text(text: Optional[str]) -> str:
    value = (text or "").lower()
    value = re.sub(r"\[[^\]]+\]", " ", value)
    value = re.sub(r"```[\s\S]*?```", " ", value)
    value = re.sub(r"[^a-z0-9\s_/-]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def tokenize(text: Optional[str]) -> List[str]:
    return [token for token in normalize_text(text).split() if token]


def exact_match(reference: Optional[str], candidate: Optional[str]) -> float:
    return 1.0 if normalize_text(reference) == normalize_text(candidate) and normalize_text(reference) else 0.0


def token_f1(reference: Optional[str], candidate: Optional[str]) -> float:
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    overlap = sum(min(ref_counts[token], cand_counts[token]) for token in ref_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(reference: Optional[str], candidate: Optional[str]) -> float:
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    dp = [[0] * (len(cand_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
    for i, ref_token in enumerate(ref_tokens, start=1):
        for j, cand_token in enumerate(cand_tokens, start=1):
            if ref_token == cand_token:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu4(reference: Optional[str], candidate: Optional[str]) -> float:
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    precisions: List[float] = []
    for n in range(1, 5):
        if len(cand_tokens) < n:
            precisions.append(0.0)
            continue
        cand_ngrams = Counter(tuple(cand_tokens[i : i + n]) for i in range(len(cand_tokens) - n + 1))
        ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
        overlap = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
        total = sum(cand_ngrams.values()) or 1
        precisions.append(overlap / total)
    if min(precisions) == 0:
        return 0.0
    brevity_penalty = 1.0
    if len(cand_tokens) < len(ref_tokens):
        brevity_penalty = math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1))
    return brevity_penalty * math.exp(sum(math.log(value) for value in precisions) / 4)


def ordered_step_count(output: Optional[str]) -> int:
    if not output:
        return 0
    numbered = re.findall(r"(?m)^\s*(?:\d+\.|\*|-)\s+", output)
    return len(numbered)


def extract_sql_text(output: Optional[str]) -> str:
    text = output or ""
    block = re.search(r"\[SQL\]\s*(.*?)(?:\n\[[A-Za-z_ ]+\]|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
    if block:
        return block.group(1).strip()
    fenced = re.search(r"```sql\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return ""


def extract_formula_text(output: Optional[str]) -> str:
    text = output or ""
    block = re.search(r"\[FORMULA\]\s*(.*?)(?:\n\[[A-Za-z_ ]+\]|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
    if block:
        return block.group(1).strip()
    fenced = re.search(r"```(?:ff|formula)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def infer_task_bucket(record: Dict[str, Any]) -> str:
    expected = str(record.get("expected_behavior") or "")
    intent = str(record.get("benchmark_intent") or record.get("intent_detected") or "").lower()

    if expected == "sql" or intent in {"sql_generation", "report_logic", "sql_troubleshooting"}:
        return "sql"
    if intent in {"fast_formula_generation", "fast_formula_troubleshooting"}:
        return "fast_formula"
    if expected in {"refusal", "correction_then_refusal"} or record.get("safe_refusal_expected"):
        return "refusal"
    if intent == "troubleshooting":
        return "troubleshooting"
    if intent in {"procedure", "navigation", "integration"}:
        return "procedure"
    return "summary"


def semantic_correct(record: Dict[str, Any]) -> float:
    return 1.0 if str(record.get("scoring_outcome")) in {"grounded_correct", "safe_refusal_correct"} else 0.0


def citation_presence(record: Dict[str, Any]) -> float:
    return 1.0 if record.get("citations_present") else 0.0


def citation_correctness(record: Dict[str, Any]) -> float:
    citations = record.get("citations") or []
    if not citations:
        return 0.0
    if str(record.get("scoring_outcome")) in {"wrong-module answer", "hallucination_error"}:
        return 0.0
    for citation in citations:
        if not citation.get("citation_id") or not citation.get("title") or not citation.get("source_uri"):
            return 0.0
    return 1.0


def grounding_supported_answer(record: Dict[str, Any]) -> float:
    if record.get("rejected"):
        return 0.0
    if record.get("citations_present") or int(record.get("docs_passed_to_prompt_count") or 0) > 0:
        return 1.0
    if record.get("sql_generated"):
        return 1.0
    return 0.0


def hallucination_flag(record: Dict[str, Any]) -> float:
    return 1.0 if str(record.get("scoring_outcome")) == "hallucination_error" else 0.0


def wrong_module_flag(record: Dict[str, Any]) -> float:
    return 1.0 if str(record.get("scoring_outcome")) == "wrong-module answer" else 0.0


def refusal_flag(record: Dict[str, Any]) -> bool:
    return bool(record.get("rejected"))


def refusal_correct(record: Dict[str, Any]) -> float:
    return 1.0 if str(record.get("scoring_outcome")) == "safe_refusal_correct" else 0.0


def over_refusal(record: Dict[str, Any]) -> float:
    if not record.get("rejected"):
        return 0.0
    expected = str(record.get("expected_behavior") or "")
    return 0.0 if expected in {"refusal", "correction_then_refusal"} else 1.0


def _strong_query_overlap(record: Dict[str, Any], terms: Iterable[str]) -> float:
    query_tokens = set(tokenize(record.get("question")))
    target = {normalize_text(term) for term in terms if normalize_text(term)}
    if not query_tokens or not target:
        return 0.0
    overlap = query_tokens & target
    return len(overlap) / max(len(target), 1)


def sql_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    sql_text = extract_sql_text(record.get("output"))
    has_join = bool(re.search(r"(?i)\bjoin\b", sql_text)) or sql_text.lower().count(" from ") > 1
    verifier_pass = 1.0 if record.get("verifier_status") == "PASSED" and sql_text else 0.0
    safe_schema = 1.0 if not record.get("unknown_schema_usage") else 0.0
    schema_objects = list(record.get("schema_objects_used") or [])
    table_correctness = 1.0 if verifier_pass and safe_schema and schema_objects else 0.0
    column_correctness = 1.0 if verifier_pass and "select *" not in sql_text.lower() else 0.0
    join_correctness = 1.0 if (not has_join or verifier_pass) else 0.0
    refusal_correctness = 1.0 if str(record.get("scoring_outcome")) == "safe_refusal_correct" else 0.0
    hallucinated_schema_rate = 1.0 if record.get("unknown_schema_usage") else 0.0
    return {
        "verifier_pass_pct": verifier_pass * 100,
        "table_correctness_pct": table_correctness * 100,
        "column_correctness_pct": column_correctness * 100,
        "join_correctness_pct": join_correctness * 100,
        "semantic_sql_match_pct": semantic_correct(record) * 100,
        "refusal_correctness_pct": refusal_correctness * 100,
        "hallucinated_schema_rate_pct": hallucinated_schema_rate * 100,
    }


def fast_formula_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    formula_text = extract_formula_text(record.get("output"))
    has_inputs = "inputs are" in formula_text.lower()
    has_default = "default for" in formula_text.lower()
    has_return = "return" in formula_text.lower()
    verifier_pass = 1.0 if record.get("verifier_status") == "PASSED" and formula_text else 0.0
    variable_reference = 1.0 if (has_inputs or has_default) and has_return else 0.0
    grounded_derivation = 1.0 if record.get("citations_present") else 0.0
    return {
        "verifier_parser_pass_pct": verifier_pass * 100,
        "variable_reference_correctness_pct": variable_reference * 100,
        "semantic_formula_correctness_pct": semantic_correct(record) * 100,
        "grounded_derivation_correctness_pct": grounded_derivation * 100,
        "refusal_correctness_pct": refusal_correct(record) * 100,
        "hallucination_pct": hallucination_flag(record) * 100,
    }


def procedure_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    output = record.get("output") or ""
    steps = ordered_step_count(output)
    prerequisite_present = 1.0 if re.search(r"(?i)\bprerequisite|before you begin|must have|required\b", output) else 0.0
    role_context = 1.0 if re.search(r"(?i)\bwork area|navigator|task|role|privilege|responsibility\b", output) else 0.0
    step_completeness = 1.0 if semantic_correct(record) and steps >= 2 else 0.5 if semantic_correct(record) else 0.0
    step_order = 1.0 if re.search(r"(?m)^\s*1\.", output) else 0.5 if steps else 0.0
    return {
        "step_completeness_pct": step_completeness * 100,
        "step_order_correctness_pct": step_order * 100,
        "prerequisite_presence_pct": prerequisite_present * 100,
        "role_context_correctness_pct": role_context * 100,
        "citation_coverage_pct": citation_presence(record) * 100,
        "semantic_correctness_pct": semantic_correct(record) * 100,
    }


def troubleshooting_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    output = record.get("output") or ""
    symptom_match = 1.0 if _strong_query_overlap(record, tokenize(output)) >= 0.15 else 0.0
    cause_present = 1.0 if re.search(r"(?i)\broot cause|probable cause|causes?|because|reason\b", output) else 0.0
    resolution_steps = 1.0 if re.search(r"(?i)\bresolution|fix|resolve|correct|resubmit|review\b", output) else 0.0
    completeness = clamp((cause_present + resolution_steps + semantic_correct(record)) / 3.0)
    return {
        "symptom_match_pct": symptom_match * 100,
        "root_cause_relevance_pct": cause_present * 100,
        "resolution_usefulness_pct": resolution_steps * 100,
        "resolution_completeness_pct": completeness * 100,
        "citation_coverage_pct": citation_presence(record) * 100,
        "semantic_correctness_pct": semantic_correct(record) * 100,
    }


def summary_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    completeness = max(token_f1(record.get("benchmark_answer"), record.get("output")), semantic_correct(record))
    return {
        "semantic_correctness_pct": semantic_correct(record) * 100,
        "citation_correctness_pct": citation_correctness(record) * 100,
        "answer_completeness_pct": completeness * 100,
        "groundedness_support_pct": grounding_supported_answer(record) * 100,
    }


def refusal_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    safe_refusal = refusal_correct(record)
    false_refusal = over_refusal(record)
    unsafe_answer = 1.0 if str(record.get("expected_behavior") or "") in {"refusal", "correction_then_refusal"} and not record.get("rejected") else 0.0
    return {
        "refusal_correctness_pct": safe_refusal * 100,
        "safe_refusal_rate_pct": safe_refusal * 100,
        "false_refusal_rate_pct": false_refusal * 100,
        "unsafe_answer_rate_pct": unsafe_answer * 100,
    }


def lexical_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    reference = record.get("benchmark_answer") or ""
    candidate = record.get("output") or ""
    return {
        "exact_match_pct": exact_match(reference, candidate) * 100,
        "token_f1_pct": token_f1(reference, candidate) * 100,
        "rouge_l_pct": rouge_l_f1(reference, candidate) * 100,
        "bleu_pct": bleu4(reference, candidate) * 100,
    }


def primary_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    trusted = 1.0 if str(record.get("scoring_outcome")) in {"grounded_correct", "safe_refusal_correct"} else 0.0
    verifier_approved = 1.0 if record.get("verifier_status") == "PASSED" else 0.0
    return {
        "trusted_outcome_pct": trusted * 100,
        "over_refusal_pct": over_refusal(record) * 100,
        "refusal_correctness_pct": refusal_correct(record) * 100,
        "hallucination_pct": hallucination_flag(record) * 100,
        "wrong_module_pct": wrong_module_flag(record) * 100,
        "semantic_correctness_pct": semantic_correct(record) * 100,
        "citation_presence_pct": citation_presence(record) * 100,
        "citation_correctness_pct": citation_correctness(record) * 100,
        "verifier_approved_pct": verifier_approved * 100,
        "grounding_supported_answer_rate_pct": grounding_supported_answer(record) * 100,
    }


def task_specific_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    bucket = infer_task_bucket(record)
    if bucket == "sql":
        return sql_metrics(record)
    if bucket == "fast_formula":
        return fast_formula_metrics(record)
    if bucket == "procedure":
        return procedure_metrics(record)
    if bucket == "troubleshooting":
        return troubleshooting_metrics(record)
    if bucket == "refusal":
        return refusal_metrics(record)
    return summary_metrics(record)


def primary_verdict(record: Dict[str, Any]) -> str:
    outcome = str(record.get("scoring_outcome") or "")
    return "pass" if SCORING_RUBRIC["primary_verdict_rules"].get(outcome) == "pass" else "fail"


def pass_fail_reason(record: Dict[str, Any]) -> str:
    outcome = str(record.get("scoring_outcome") or "unknown")
    if outcome in {"grounded_correct", "safe_refusal_correct"}:
        return "Task-aware scoring accepted the response."
    if outcome == "semantic_mismatch":
        return "Response was grounded but semantically wrong or incomplete for the expected task."
    if outcome == "wrong-module answer":
        return "Response used the wrong module interpretation."
    if outcome == "hallucination_error":
        return "Response used unsupported or hallucinated grounding."
    if outcome == "missing-citation":
        return "Response lacked required citation support."
    if outcome == "verifier_failure":
        return "Response failed verifier or was conservatively fail-closed."
    if outcome == "sql_failure":
        return "SQL-specific validation failed."
    return f"Scoring outcome: {outcome}."


def evaluator_notes(record: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    if record.get("rejected"):
        notes.append("Response was refused.")
    if record.get("citations_present"):
        notes.append(f"Citations attached: {len(record.get('citations') or [])}.")
    if record.get("decision_grounding_signal_present") is False:
        notes.append("No grounding signal was available at decision time.")
    if record.get("decision_sufficient_grounding_signal") is False and not record.get("rejected"):
        notes.append("Execution proceeded with weak grounding support.")
    if record.get("same_family_bleed_through"):
        notes.append("Same-family bleed-through was detected.")
    if record.get("unknown_schema_usage"):
        notes.append("Unknown schema object usage detected.")
    return notes


def build_case_scorecard(record: Dict[str, Any]) -> Dict[str, Any]:
    bucket = infer_task_bucket(record)
    lexical = lexical_metrics(record) if bucket in {"procedure", "troubleshooting", "summary"} else {}
    return {
        "case_id": str(record.get("id") or ""),
        "module": str(record.get("benchmark_module") or record.get("module") or "UNKNOWN"),
        "task_type": bucket,
        "difficulty": str(record.get("difficulty") or "unknown"),
        "primary_verdict": primary_verdict(record),
        "primary_metrics": primary_metrics(record),
        "task_specific_metrics": task_specific_metrics(record),
        "lexical_metrics": lexical,
        "pass_fail_reason": pass_fail_reason(record),
        "failure_category": str(record.get("failure_category") or ""),
        "evaluator_notes": evaluator_notes(record),
    }


def _average_metric(items: List[Dict[str, Any]], key: str) -> float:
    if not items:
        return 0.0
    values = [float(item.get(key) or 0.0) for item in items]
    return round(sum(values) / max(len(values), 1), 2)


def aggregate_scorecards(scorecards: List[Dict[str, Any]], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not scorecards:
        return {
            "primary_metrics": {key: 0.0 for key in PRIMARY_METRIC_KEYS},
            "per_task_type": {},
            "per_module": {},
            "per_difficulty": {},
            "failure_categories": {},
        }

    primary_summary = {
        key: _average_metric([card["primary_metrics"] for card in scorecards], key)
        for key in PRIMARY_METRIC_KEYS
    }

    def build_breakdown(group_key: str) -> Dict[str, Any]:
        grouped_cards: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for card in scorecards:
            grouped_cards[str(card.get(group_key) or "UNKNOWN")].append(card)
        breakdown: Dict[str, Any] = {}
        for name, items in sorted(grouped_cards.items()):
            breakdown[name] = {
                "count": len(items),
                "primary_metrics": {
                    key: _average_metric([item["primary_metrics"] for item in items], key)
                    for key in PRIMARY_METRIC_KEYS
                },
                "task_specific_metrics": {
                    metric: _average_metric(
                        [item["task_specific_metrics"] for item in items if metric in item["task_specific_metrics"]],
                        metric,
                    )
                    for metric in sorted(
                        {metric for item in items for metric in item["task_specific_metrics"].keys()}
                    )
                },
            }
        return breakdown

    failure_counter: Counter[str] = Counter()
    for record in records:
        category = str(record.get("failure_category") or "")
        if category and str(record.get("scoring_outcome")) not in {"grounded_correct", "safe_refusal_correct"}:
            failure_counter[category] += 1

    return {
        "primary_metrics": primary_summary,
        "per_task_type": build_breakdown("task_type"),
        "per_module": build_breakdown("module"),
        "per_difficulty": build_breakdown("difficulty"),
        "failure_categories": dict(sorted(failure_counter.items(), key=lambda item: item[1], reverse=True)),
        "scoring_rubric": SCORING_RUBRIC,
    }
