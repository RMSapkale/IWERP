import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .dependencies import get_current_tenant, get_db
from core.database.models import Feedback, IngestJob, Tenant
from core.rag.engine import RAGEngine
from core.schemas.api import ChatRequest, ChatResponse, Message, Role

router = APIRouter()
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


def _coerce_chat_request(payload: Any) -> ChatRequest:
    if isinstance(payload, ChatRequest):
        return payload

    if isinstance(payload, dict) and "messages" in payload:
        return ChatRequest(**payload)

    if isinstance(payload, dict) and payload.get("message"):
        return ChatRequest(messages=[Message(role=Role.USER, content=str(payload["message"]))])

    raise HTTPException(status_code=400, detail="Request must include either 'messages' or 'message'.")


@router.get("/whoami")
async def whoami(tenant: Tenant = Depends(get_current_tenant)):
    return {"tenant_id": tenant.id, "status": "authenticated", "moe_group": tenant.moe_experiment_group}


@router.post("/expert/chat", response_model=ChatResponse)
async def expert_chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Expert auditing endpoint with RAG grounding enforced.
    """
    stmt = select(Tenant).limit(1)
    result = await db.execute(stmt)
    tenant = result.scalar_one_or_none()

    if not tenant:
        raise HTTPException(status_code=404, detail="No valid tenant found for technical grounding.")

    return await get_rag_engine().chat(db, tenant, request)


@router.post("/rag/chat", response_model=ChatResponse)
async def hybrid_rag_chat(
    request: ChatRequest,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """
    Main grounded chat endpoint.
    """
    return await get_rag_engine().chat(db, tenant, request)


@router.post("/chat/completions", response_model=ChatResponse)
async def openai_compatible_chat(
    payload: Dict[str, Any],
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """
    Compatibility endpoint routed through the same grounded RAG engine.
    """
    request = _coerce_chat_request(payload)
    return await get_rag_engine().chat(db, tenant, request)


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: int = Field(..., description="1 for thumbs up, -1 for thumbs down")
    issue_type: Optional[str] = None
    comment: Optional[str] = None
    corrected_answer: Optional[str] = None


@router.post("/rag/feedback")
async def submit_feedback(
    payload: FeedbackRequest,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """
    Submits user feedback for a specific RAG trace.
    """
    try:
        trace_uuid = uuid.UUID(payload.trace_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid trace_id format.") from exc

    feedback = Feedback(
        trace_id=trace_uuid,
        tenant_id=tenant.id,
        rating=payload.rating,
        issue_type=payload.issue_type,
        comment=payload.comment,
        corrected_answer=payload.corrected_answer,
    )
    db.add(feedback)
    await db.commit()
    return {"status": "success", "feedback_id": str(feedback.id)}


@router.post("/ingest/jobs")
async def create_ingest_job(
    file: UploadFile = File(...),
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """
    Starts an asynchronous ingestion job for the uploaded file.
    """
    job_id = uuid.uuid4()
    job = IngestJob(
        id=job_id,
        tenant_id=tenant.id,
        filename=file.filename,
        status="pending",
    )
    db.add(job)
    await db.commit()

    return {
        "job_id": str(job_id),
        "status": "pending",
        "filename": file.filename,
    }


@router.get("/ingest/jobs/{job_id}")
async def get_ingest_job(
    job_id: str,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieves the status of a specific ingestion job.
    """
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid job_id format.") from exc

    result = await db.execute(
        select(IngestJob).where(IngestJob.id == job_uuid, IngestJob.tenant_id == tenant.id)
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")

    return {
        "job_id": str(job.id),
        "status": job.status,
        "filename": job.filename,
        "error": job.error_message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


@router.post("/sync/oraclewings")
async def sync_oraclewings(payload: Dict[str, Any], tenant: Tenant = Depends(get_current_tenant)):
    """
    Webhook connector for oraclewings.ai to sync external data or signals.
    """
    return {"status": "synced", "tenant": tenant.id, "received": len(payload)}


@router.post("/ingest/text")
async def ingest_plain_text(
    filename: str,
    text_content: str,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    from apps.ingest.worker import IngestWorker

    worker = IngestWorker()
    await worker.ingest_file(tenant.id, filename, text_content)
    return {"status": "success", "filename": filename}
