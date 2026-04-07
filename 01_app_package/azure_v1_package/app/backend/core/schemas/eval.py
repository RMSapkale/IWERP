from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from core.schemas.router import TaskType, FusionModule

class EvalItem(BaseModel):
    id: str = Field(default_factory=lambda: "eval_" + str(uuid.uuid4())[:8])
    question: str
    expected_answer: Optional[str] = None
    expected_keypoints: List[str] = Field(default_factory=list)
    expected_citations: List[str] = Field(default_factory=list)
    task_type: TaskType = TaskType.GENERAL
    module: FusionModule = FusionModule.UNKNOWN

class EvalMetricResult(BaseModel):
    faithfulness: float = 0.0
    relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    keypoint_alignment: float = 0.0
    citation_rate: float = 0.0
    total_score: float = 0.0

class EvalSampleResult(BaseModel):
    item: EvalItem
    actual_answer: str
    retrieved_chunks: List[Dict[str, Any]]
    metrics: EvalMetricResult
    latency_seconds: float
    trace_id: str

class EvalReport(BaseModel):
    samples: List[EvalSampleResult]
    avg_metrics: EvalMetricResult
    total_time_seconds: float
