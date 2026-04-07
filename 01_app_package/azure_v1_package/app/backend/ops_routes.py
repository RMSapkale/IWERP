from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from .dependencies import get_current_tenant, get_db
from .routes import get_rag_engine
from core.evaluation.benchmark_ops import (
    BENCHMARK_ROOT,
    benchmark_cases,
    benchmark_summary,
    health_readiness_payload,
    launch_benchmark_run,
    list_benchmark_runs,
    load_trace,
    persist_trace,
    resolve_dataset_path,
)
from core.evaluation.framework import SCORING_RUBRIC, aggregate_scorecards, build_case_scorecard
from core.rag.engine import FAIL_CLOSED_MESSAGE
from core.schemas.api import ChatRequest, Message, Role
from core.schemas.ops_api import (
    BenchmarkCaseResponse,
    BenchmarkHistoryItem,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    BenchmarkSummaryResponse,
    DecisionTrace,
    EvaluateRequest,
    EvaluateResponse,
    GroundingTrace,
    HealthReadinessResponse,
    OpenAIStyleChoice,
    OpenAIStyleChoiceMessage,
    OpenAIStyleRequest,
    OpenAIStyleResponse,
    TraceResponse,
)


ops_router = APIRouter()
legacy_openai_router = APIRouter()


def _request_to_chat_request(payload: OpenAIStyleRequest) -> ChatRequest:
    messages = list(payload.messages)
    if payload.system_prompt:
        messages = [Message(role=Role.SYSTEM, content=payload.system_prompt), *messages]
    if not messages and payload.input is not None:
        if isinstance(payload.input, str):
            messages = [Message(role=Role.USER, content=payload.input)]
        elif isinstance(payload.input, list):
            normalized: List[Message] = []
            for item in payload.input:
                if isinstance(item, str):
                    normalized.append(Message(role=Role.USER, content=item))
                elif isinstance(item, dict):
                    normalized.append(
                        Message(
                            role=Role(str(item.get("role") or "user")),
                            content=str(item.get("content") or ""),
                        )
                    )
            messages = normalized
    if not messages:
        raise HTTPException(status_code=400, detail="messages or input is required")
    return ChatRequest(
        messages=messages,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        top_p=payload.top_p,
        top_k=payload.top_k,
        repeat_penalty=payload.repeat_penalty,
    )


def _decision_trace_from_audit(audit: Dict[str, Any]) -> DecisionTrace:
    return DecisionTrace(
        intent_classification=audit.get("intent_classification") or audit.get("task_type"),
        intent_confidence=float(audit.get("intent_confidence") or 0.0),
        module_confidence=float(audit.get("module_confidence") or 0.0),
        grounding_availability_score=float(audit.get("grounding_availability_score") or 0.0),
        grounding_confidence_tier=audit.get("grounding_confidence_tier"),
        decision_confidence_tier=audit.get("decision_confidence_tier"),
        decision_execution_mode=audit.get("decision_execution_mode"),
        decision_reason=audit.get("decision_reason"),
        decision_refusal_reason=audit.get("decision_refusal_reason"),
    )


def _grounding_trace_from_response(response: Any) -> GroundingTrace:
    chunks = response.retrieved_chunks or []
    audit = response.audit or {}
    corpora = sorted(
        {
            str((chunk.get("metadata") or {}).get("corpus") or "")
            for chunk in chunks
            if (chunk.get("metadata") or {}).get("corpus")
        }
    )
    return GroundingTrace(
        docs_retrieved_count=len(chunks),
        docs_passed_to_prompt_count=int(audit.get("docs_passed_to_prompt_count") or 0),
        citation_count=len(response.citations or []),
        exact_module_doc_count=int(audit.get("exact_module_doc_count") or 0),
        task_semantic_gate=audit.get("task_semantic_gate"),
        task_match_rate=float(audit.get("task_match_rate") or 0.0),
        selected_corpora=corpora,
        retrieved_chunks=chunks,
    )


def _response_to_openai_style(response: Any, *, debug: bool, object_type: str) -> OpenAIStyleResponse:
    output_text = response.choices[0]["message"]["content"] if response.choices else ""
    refusal = FAIL_CLOSED_MESSAGE in output_text
    decision_trace = _decision_trace_from_audit(response.audit or {})
    grounding_trace = _grounding_trace_from_response(response)
    payload = OpenAIStyleResponse(
        id=response.id,
        object=object_type,
        created=response.created,
        model=response.model,
        output_text=output_text,
        choices=[
            OpenAIStyleChoice(
                index=0,
                message=OpenAIStyleChoiceMessage(content=output_text),
            )
        ],
        citations=response.citations or [],
        selected_module=(response.audit or {}).get("module"),
        task_type=(response.audit or {}).get("task_type"),
        grounded=bool(response.citations or []) or bool((response.audit or {}).get("docs_passed_to_prompt_count")),
        refusal=refusal,
        verifier_status=(response.audit or {}).get("verification_status"),
        decision_trace=decision_trace if debug else None,
        grounding_trace=grounding_trace if debug else None,
        usage=response.usage or {},
    )
    trace_payload = {
        "id": payload.id,
        "object": payload.object,
        "created": payload.created,
        "model": payload.model,
        "output_text": payload.output_text,
        "citations": [citation.model_dump() for citation in payload.citations],
        "selected_module": payload.selected_module,
        "task_type": payload.task_type,
        "grounded": payload.grounded,
        "refusal": payload.refusal,
        "verifier_status": payload.verifier_status,
        "decision_trace": payload.decision_trace.model_dump() if payload.decision_trace else decision_trace.model_dump(),
        "grounding_trace": payload.grounding_trace.model_dump() if payload.grounding_trace else grounding_trace.model_dump(),
        "audit": response.audit or {},
    }
    payload_dict = payload.model_dump()
    payload_dict["_trace_payload"] = trace_payload
    return OpenAIStyleResponse(**payload_dict)

async def _run_compat_chat(
    payload: OpenAIStyleRequest,
    *,
    tenant: Any,
    db: AsyncSession,
    object_type: str,
) -> OpenAIStyleResponse:
    chat_request = _request_to_chat_request(payload)
    response = await get_rag_engine().chat(db, tenant, chat_request, require_citations=True)
    wrapped = _response_to_openai_style(response, debug=payload.debug, object_type=object_type)
    persist_trace(
        trace_id=wrapped.id,
        tenant_id=str(getattr(tenant, "id", "")),
        request_payload=payload.model_dump(),
        response_payload={
            **wrapped.model_dump(exclude={"decision_trace", "grounding_trace"}),
            "decision_trace": _decision_trace_from_audit(response.audit or {}).model_dump(),
            "grounding_trace": _grounding_trace_from_response(response).model_dump(),
            "audit": response.audit or {},
        },
    )
    return wrapped


def _live_eval_record(case: Dict[str, Any], response: OpenAIStyleResponse) -> Dict[str, Any]:
    refusal = bool(response.refusal)
    expected_behavior = "answer"
    scoring_outcome = "grounded_correct"
    if refusal and case.get("expected_answer"):
        scoring_outcome = "verifier_failure"
    elif refusal and not case.get("expected_answer"):
        expected_behavior = "refusal"
        scoring_outcome = "safe_refusal_correct"
    elif case.get("expected_module") and response.selected_module and response.selected_module != case.get("expected_module"):
        scoring_outcome = "wrong-module answer"
    elif response.verifier_status != "PASSED" and not refusal:
        scoring_outcome = "verifier_failure"

    return {
        "id": case.get("case_id") or f"eval_{uuid.uuid4().hex[:8]}",
        "question": case.get("query"),
        "benchmark_answer": case.get("expected_answer"),
        "benchmark_module": case.get("expected_module") or response.selected_module or "UNKNOWN",
        "difficulty": case.get("difficulty") or "unknown",
        "benchmark_intent": case.get("expected_task_type") or response.task_type or "general",
        "expected_behavior": expected_behavior,
        "output": response.output_text,
        "rejected": refusal,
        "citations_present": bool(response.citations),
        "citations": [citation.model_dump() for citation in response.citations],
        "citation_count": len(response.citations),
        "module_detected": response.selected_module,
        "intent_detected": response.task_type,
        "verifier_status": response.verifier_status,
        "verifier_passed": response.verifier_status == "PASSED",
        "runtime_error": None,
        "docs_passed_to_prompt_count": response.grounding_trace.docs_passed_to_prompt_count if response.grounding_trace else 0,
        "retrieved_doc_count": response.grounding_trace.docs_retrieved_count if response.grounding_trace else 0,
        "retrieved_docs": response.grounding_trace.retrieved_chunks if response.grounding_trace else [],
        "same_family_bleed_through": False,
        "unknown_schema_usage": False,
        "hallucination_score": 0.0,
        "sql_generated": None,
        "safe_refusal_expected": expected_behavior == "refusal",
        "decision_grounding_signal_present": bool(
            response.decision_trace and response.decision_trace.grounding_availability_score > 0.0
        ),
        "decision_sufficient_grounding_signal": bool(
            response.decision_trace and response.decision_trace.grounding_availability_score >= 0.45
        ),
        "scoring_outcome": scoring_outcome,
        "failure_category": "" if scoring_outcome in {"grounded_correct", "safe_refusal_correct"} else scoring_outcome,
    }


@legacy_openai_router.post(
    "/openai/chat/completions",
    response_model=OpenAIStyleResponse,
    tags=["compat"],
    include_in_schema=False,
)
async def openai_style_chat_completions(
    payload: OpenAIStyleRequest,
    tenant=Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    return await _run_compat_chat(payload, tenant=tenant, db=db, object_type="chat.completion")


@legacy_openai_router.post(
    "/openai/responses",
    response_model=OpenAIStyleResponse,
    tags=["compat"],
    include_in_schema=False,
)
async def openai_style_responses(
    payload: OpenAIStyleRequest,
    tenant=Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    return await _run_compat_chat(payload, tenant=tenant, db=db, object_type="response")


@ops_router.post("/sovereign/chat/completions", response_model=OpenAIStyleResponse, tags=["inference"])
async def sovereign_chat_completions(
    payload: OpenAIStyleRequest,
    tenant=Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    return await _run_compat_chat(payload, tenant=tenant, db=db, object_type="chat.completion")


@ops_router.post("/sovereign/responses", response_model=OpenAIStyleResponse, tags=["inference"])
async def sovereign_responses(
    payload: OpenAIStyleRequest,
    tenant=Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    return await _run_compat_chat(payload, tenant=tenant, db=db, object_type="response")


@ops_router.post("/evaluate", response_model=EvaluateResponse, tags=["evaluation"])
async def evaluate_cases(
    payload: EvaluateRequest,
    tenant=Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    case_results: List[Dict[str, Any]] = []
    scorecards: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    for case in payload.cases:
        chat_payload = OpenAIStyleRequest(
            messages=[Message(role=Role.USER, content=case.query)],
            max_tokens=120,
            debug=payload.debug,
        )
        chat_request = _request_to_chat_request(chat_payload)
        response = await get_rag_engine().chat(db, tenant, chat_request, require_citations=True)
        wrapped = _response_to_openai_style(response, debug=True, object_type="response")
        record = _live_eval_record(case.model_dump(), wrapped)
        scorecard = build_case_scorecard(record)
        case_results.append(
            {
                "request": case.model_dump(),
                "response": wrapped.model_dump(),
                "evaluation": scorecard,
            }
        )
        scorecards.append(scorecard)
        records.append(record)
    aggregate = aggregate_scorecards(scorecards, records)
    return EvaluateResponse(cases=case_results, aggregate=aggregate, scoring_rubric=SCORING_RUBRIC)


@ops_router.post("/benchmarks", response_model=BenchmarkRunResponse, tags=["benchmark"])
async def start_benchmark_run(
    payload: BenchmarkRunRequest,
    tenant=Depends(get_current_tenant),
):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    label = payload.label or f"ops_{payload.dataset}_{timestamp}"
    dataset_path = resolve_dataset_path(
        payload.dataset,
        label=label,
        module_filters=payload.module_filters,
        task_filters=payload.task_filters,
        case_ids=payload.case_ids,
        custom_input_path=payload.custom_input_path,
    )
    metadata = launch_benchmark_run(
        label=label,
        dataset_path=dataset_path,
        max_tokens=payload.max_tokens,
        flush_every=payload.flush_every,
        case_timeout_sec=payload.case_timeout_sec,
        case_retries=payload.case_retries,
        recycle_worker_every=payload.recycle_worker_every,
    )
    return BenchmarkRunResponse(
        run_id=metadata["run_id"],
        label=metadata["label"],
        status=metadata["status"],
        dataset_path=metadata["dataset_path"],
        command=metadata["command"],
        output_dir=str(BENCHMARK_ROOT / label),
        pid=metadata.get("pid"),
        summary_path=str(BENCHMARK_ROOT / label / "benchmark_summary.json"),
    )


@ops_router.get("/benchmarks", response_model=List[BenchmarkHistoryItem], tags=["benchmark"])
async def get_benchmark_runs(
    _tenant=Depends(get_current_tenant),
):
    return [BenchmarkHistoryItem(**item) for item in list_benchmark_runs(limit=50)]


@ops_router.get("/benchmarks/{run_id}", response_model=BenchmarkSummaryResponse, tags=["benchmark"])
async def get_benchmark_summary(
    run_id: str,
    _tenant=Depends(get_current_tenant),
):
    try:
        summary = benchmark_summary(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown benchmark run: {run_id}") from exc
    return BenchmarkSummaryResponse(**summary)


@ops_router.get("/benchmarks/{run_id}/cases", response_model=BenchmarkCaseResponse, tags=["benchmark"])
async def get_benchmark_case_rows(
    run_id: str,
    module: str | None = Query(default=None),
    task_type: str | None = Query(default=None),
    failure_category: str | None = Query(default=None),
    primary_verdict: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    _tenant=Depends(get_current_tenant),
):
    return BenchmarkCaseResponse(
        **benchmark_cases(
            run_id,
            module=module,
            task_type=task_type,
            failure_category=failure_category,
            primary_verdict=primary_verdict,
            limit=limit,
        )
    )


@ops_router.get("/traces/{trace_id}", response_model=TraceResponse, tags=["trace"])
async def get_trace(
    trace_id: str,
    _tenant=Depends(get_current_tenant),
):
    try:
        trace = load_trace(trace_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown trace: {trace_id}") from exc
    return TraceResponse(**trace)


@ops_router.get("/health/readiness", response_model=HealthReadinessResponse, tags=["health"])
async def get_health_readiness(
    _tenant=Depends(get_current_tenant),
):
    return HealthReadinessResponse(**health_readiness_payload())
