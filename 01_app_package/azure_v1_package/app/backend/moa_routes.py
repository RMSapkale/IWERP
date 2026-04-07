from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from .dependencies import get_db, get_current_tenant
from core.moe.agent_orchestrator import MoAOrchestrator
from core.schemas.api import ChatRequest, ChatResponse, Message, Role
from core.database.models import Tenant
from typing import Dict, Any
import uuid
import time

moa_router = APIRouter()
orchestrator = MoAOrchestrator()

@moa_router.post("/chat/expert", response_model=ChatResponse)
async def moa_expert_chat(
    request: ChatRequest,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """
    Expert-assisted chat endpoint using Mixture of Agents (MoA).
    Routes query to a specialized Oracle Fusion expert (HCM, SCM, SQL, etc).
    """
    start_time = time.time()
    query = request.messages[-1].content
    
    # 1. Routing phase (In production, this is done by the 8B model)
    # For now, we use a simple routing heuristic while 8B is training
    intent = "hcm_functional"
    if "sql" in query.lower() or "table" in query.lower():
        intent = "sql"
    elif "formula" in query.lower():
        intent = "hcm_formula"
    elif "integration" in query.lower() or "oic" in query.lower():
        intent = "oic_architect"
    
    # 2. Dispatch to Expert
    expert_res = orchestrator.get_expert_response(intent, query)
    
    # 3. Synthesis phase
    final_text = orchestrator.synthesize("Expert mapping confirmed.", expert_res)
    
    end_time = time.time()
    
    return ChatResponse(
        id=str(uuid.uuid4()),
        object="chat.completion",
        created=int(start_time),
        model="fusion-slm-8b-moa",
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": final_text
            },
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": len(query.split()),
            "completion_tokens": len(final_text.split()),
            "total_tokens": len(query.split()) + len(final_text.split())
        },
        system_fingerprint="moa-v1"
    )
