from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import uuid
from core.schemas.router import TaskType, FusionModule
from core.schemas.api import Role

class SFTMessage(BaseModel):
    role: Role
    content: str

class SFTSampleSchema(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    module: FusionModule
    task_type: TaskType
    messages: List[SFTMessage]
    difficulty: str = "medium"
    source_doc_ids: List[str] = Field(default_factory=list)

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        if v not in ["easy", "medium", "hard"]:
            raise ValueError("Difficulty must be easy, medium, or hard")
        return v
