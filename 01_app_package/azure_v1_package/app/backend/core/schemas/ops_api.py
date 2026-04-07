from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from core.schemas.api import Citation, Message


class OpenAIStyleRequest(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    input: Optional[str | List[Dict[str, Any]] | List[str]] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repeat_penalty: Optional[float] = None
    debug: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DecisionTrace(BaseModel):
    intent_classification: Optional[str] = None
    intent_confidence: float = 0.0
    module_confidence: float = 0.0
    grounding_availability_score: float = 0.0
    grounding_confidence_tier: Optional[str] = None
    decision_confidence_tier: Optional[str] = None
    decision_execution_mode: Optional[str] = None
    decision_reason: Optional[str] = None
    decision_refusal_reason: Optional[str] = None


class GroundingTrace(BaseModel):
    docs_retrieved_count: int = 0
    docs_passed_to_prompt_count: int = 0
    citation_count: int = 0
    exact_module_doc_count: int = 0
    task_semantic_gate: Optional[str] = None
    task_match_rate: float = 0.0
    selected_corpora: List[str] = Field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)


class OpenAIStyleChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class OpenAIStyleChoice(BaseModel):
    index: int = 0
    message: OpenAIStyleChoiceMessage
    finish_reason: Literal["stop"] = "stop"


class OpenAIStyleResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    output_text: str
    choices: List[OpenAIStyleChoice] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    selected_module: Optional[str] = None
    task_type: Optional[str] = None
    grounded: bool = False
    refusal: bool = False
    verifier_status: Optional[str] = None
    evaluation: Optional[Dict[str, Any]] = None
    decision_trace: Optional[DecisionTrace] = None
    grounding_trace: Optional[GroundingTrace] = None
    usage: Dict[str, int] = Field(default_factory=dict)


class EvaluateCaseInput(BaseModel):
    case_id: Optional[str] = None
    query: str
    expected_module: Optional[str] = None
    expected_answer: Optional[str] = None
    expected_task_type: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluateRequest(BaseModel):
    cases: List[EvaluateCaseInput]
    debug: bool = False


class EvaluateResponse(BaseModel):
    cases: List[Dict[str, Any]] = Field(default_factory=list)
    aggregate: Dict[str, Any] = Field(default_factory=dict)
    scoring_rubric: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkRunRequest(BaseModel):
    dataset: Literal["200", "1000", "5000", "custom"] = "1000"
    label: Optional[str] = None
    custom_input_path: Optional[str] = None
    module_filters: List[str] = Field(default_factory=list)
    task_filters: List[str] = Field(default_factory=list)
    case_ids: List[str] = Field(default_factory=list)
    max_tokens: int = 120
    flush_every: int = 25
    case_timeout_sec: int = 120
    case_retries: int = 1
    recycle_worker_every: int = 50
    debug: bool = False


class BenchmarkRunResponse(BaseModel):
    run_id: str
    label: str
    status: str
    dataset_path: str
    command: List[str] = Field(default_factory=list)
    output_dir: str
    pid: Optional[int] = None
    summary_path: Optional[str] = None


class BenchmarkHistoryItem(BaseModel):
    run_id: str
    label: str
    status: str
    sample_size: int = 0
    created_at: Optional[int] = None
    updated_at: Optional[int] = None
    summary_path: Optional[str] = None
    primary_metrics: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkCaseResponse(BaseModel):
    run_id: str
    total_cases: int
    returned_cases: int
    cases: List[Dict[str, Any]] = Field(default_factory=list)


class BenchmarkSummaryResponse(BaseModel):
    run_id: str
    label: str
    status: str
    sample_size: int = 0
    summary: Dict[str, Any] = Field(default_factory=dict)
    history: List[BenchmarkHistoryItem] = Field(default_factory=list)


class TraceResponse(BaseModel):
    trace_id: str
    request: Dict[str, Any] = Field(default_factory=dict)
    response: Dict[str, Any] = Field(default_factory=dict)
    decision_trace: Dict[str, Any] = Field(default_factory=dict)
    grounding_trace: Dict[str, Any] = Field(default_factory=dict)
    audit: Dict[str, Any] = Field(default_factory=dict)


class HealthReadinessResponse(BaseModel):
    status: str
    version: str
    model_backend: str
    corpus_status: Dict[str, Any] = Field(default_factory=dict)
    latest_benchmark: Dict[str, Any] = Field(default_factory=dict)
    scoring_rubric: Dict[str, Any] = Field(default_factory=dict)
    safety_flags: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int
