from fastapi import APIRouter, Depends, HTTPException
from typing import List, Union, Dict, Any
from pydantic import BaseModel
from .dependencies import get_current_tenant
from core.database.models import Tenant
import structlog
from fastembed import TextEmbedding

logger = structlog.get_logger(__name__)
router = APIRouter()

# Singleton for local embeddings
class LocalEmbedder:
    _instance = None
    def __init__(self):
        logger.info("initializing_fastembed", model="BAAI/bge-small-en-v1.5")
        self.model = TextEmbedding()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "titan-v2-embed"

@router.post("/embeddings")
async def generate_embeddings(
    request: EmbeddingRequest,
    tenant: Tenant = Depends(get_current_tenant)
):
    """
    Sovereign-Pure Embeddings API for 3rd party RAG.
    Secured by multi-tenant API keys or JWT.
    """
    logger.info("generate_embeddings_request", tenant_id=tenant.id, input_type=type(request.input))
    
    try:
        embedder = LocalEmbedder.get_instance()
        texts = [request.input] if isinstance(request.input, str) else request.input
        
        # FastEmbed is a generator
        embeddings_gen = embedder.model.embed(texts)
        embeddings_list = [list(e) for e in embeddings_gen]
        
        # Return OpenAI-compatible format
        data = []
        for i, emb in enumerate(embeddings_list):
            data.append({
                "object": "embedding",
                "embedding": emb,
                "index": i
            })
            
        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
                "prompt_tokens": 0, # Local engine doesn't charge tokens
                "total_tokens": 0
            }
        }
    except Exception as e:
        logger.error("embeddings_generation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal embedding error: {str(e)}")
