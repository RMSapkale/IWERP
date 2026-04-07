import asyncio
import structlog
from typing import List, Optional
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from core.database.session import AsyncSessionLocal
from core.database.models import Chunk
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)

class BatchEmbedder:
    """
    Asynchronous batch embedder for processing chunks missing embeddings.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    async def embed_tenant_chunks(self, tenant_id: str):
        """
        Embeds all chunks for a specific tenant that are missing embeddings.
        """
        logger.info("batch_embedding_started", tenant_id=tenant_id)
        
        async with AsyncSessionLocal() as db:
            while True:
                # 1. Select a batch of chunks missing embeddings
                result = await db.execute(
                    select(Chunk).where(
                        Chunk.tenant_id == tenant_id,
                        Chunk.embedding == None
                    ).limit(self.batch_size)
                )
                chunks = result.scalars().all()
                
                if not chunks:
                    logger.info("batch_embedding_finished", tenant_id=tenant_id)
                    break
                
                # 2. Compute embeddings
                texts = [c.content for c in chunks]
                embeddings = self.model.encode(texts).tolist()
                
                # 3. Update chunks
                for chunk, emb in zip(chunks, embeddings):
                    chunk.embedding = emb
                
                await db.commit()
                logger.info("batch_embedded", tenant_id=tenant_id, count=len(chunks))

if __name__ == "__main__":
    import sys
    tenant = sys.argv[1] if len(sys.argv) > 1 else "demo"
    embedder = BatchEmbedder()
    asyncio.run(embedder.embed_tenant_chunks(tenant))
