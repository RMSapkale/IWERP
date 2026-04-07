from typing import List, Dict, Any, Optional
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from core.database.models import Chunk, Document

class PostgresVectorSearch:
    """
    Advanced Postgres Vector Search using pgvector cosine similarity.
    Supports tenant isolation and metadata filtering.
    """
    async def search(
        self,
        db: AsyncSession,
        tenant_id: str,
        query_embedding: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs a vector search using cosine distance (<=>).
        """
        # pgvector cosine distance: <=>
        # lower distance = higher similarity
        distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
        
        # 1. Build base query
        stmt = (
            select(
                Chunk,
                Document.filename,
                Document.metadata_json,
                distance
            )
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.tenant_id == tenant_id)
        )
        
        # 2. Apply Metadata Filters
        if filters:
            from sqlalchemy.dialects.postgresql import JSONB
            metadata = func.cast(Document.metadata_json, JSONB)

            if "module" in filters:
                stmt = stmt.where(metadata["module"].astext == filters["module"])
            if "doc_id" in filters:
                stmt = stmt.where(Document.id == filters["doc_id"])
            if "tags" in filters:
                stmt = stmt.where(metadata["tags"].contains([filters["tags"]]))

        # 3. Order and Limit
        # pgvector uses distance, so we order by distance ASC
        stmt = stmt.order_by(distance).limit(limit)
        
        result = await db.execute(stmt)
        rows = result.all()
        
        return [
            {
                "id": str(row.Chunk.id),
                "content": row.Chunk.content,
                "filename": row.filename,
                # Convert distance to a similarity score (1 - distance)
                "score": 1.0 - float(row.distance),
                "metadata": row.metadata_json or {}
            }
            for row in rows
        ]
