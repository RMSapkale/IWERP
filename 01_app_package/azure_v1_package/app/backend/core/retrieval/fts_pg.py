from typing import List, Dict, Any, Optional
from sqlalchemy import Float, select, func, text, and_
from sqlalchemy.ext.asyncio import AsyncSession
from core.database.models import Chunk, Document

class PostgresFTS:
    """
    Advanced Postgres Full-Text Search using ts_rank_cd and websearch_to_tsquery.
    Supports tenant isolation and metadata filtering.
    """
    async def search(
        self,
        db: AsyncSession,
        tenant_id: str,
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs a full-text search with ranking and filters.
        """
        # 1. Convert websearch query to tsquery
        ts_query = func.websearch_to_tsquery('english', query)
        
        # 2. Build base query
        stmt = (
            select(
                Chunk,
                Document.filename,
                Document.metadata_json,
                func.ts_rank_cd(Chunk.content_tsvector, ts_query).label("rank")
            )
            .join(Document, Chunk.document_id == Document.id)
            .where(
                and_(
                    Chunk.tenant_id == tenant_id,
                    Chunk.content_tsvector.op('@@')(ts_query)
                )
            )
        )
        
        # 3. Apply Metadata Filters
        if filters:
            from sqlalchemy.dialects.postgresql import JSONB
            metadata = func.cast(Document.metadata_json, JSONB)
            
            if "module" in filters:
                module_value = filters["module"]
                if isinstance(module_value, (list, tuple, set)):
                    stmt = stmt.where(metadata["module"].astext.in_([str(v) for v in module_value]))
                else:
                    stmt = stmt.where(metadata["module"].astext == str(module_value))
            if "corpora" in filters:
                corpora_value = filters["corpora"]
                if isinstance(corpora_value, (list, tuple, set)):
                    stmt = stmt.where(metadata["corpus"].astext.in_([str(v) for v in corpora_value]))
                else:
                    stmt = stmt.where(metadata["corpus"].astext == str(corpora_value))
            if "corpus" in filters:
                corpus_value = filters["corpus"]
                if isinstance(corpus_value, (list, tuple, set)):
                    stmt = stmt.where(metadata["corpus"].astext.in_([str(v) for v in corpus_value]))
                else:
                    stmt = stmt.where(metadata["corpus"].astext == str(corpus_value))
            if "task_type" in filters:
                task_value = filters["task_type"]
                if isinstance(task_value, (list, tuple, set)):
                    stmt = stmt.where(metadata["task_type"].astext.in_([str(v) for v in task_value]))
                else:
                    stmt = stmt.where(metadata["task_type"].astext == str(task_value))
            if "quality_score_min" in filters:
                stmt = stmt.where(
                    func.coalesce(func.cast(metadata["quality_score"].astext, Float), 0.0) >= float(filters["quality_score_min"])
                )
            if "doc_id" in filters:
                stmt = stmt.where(Document.id == filters["doc_id"])
            if "tags" in filters:
                # contains works on JSONB
                stmt = stmt.where(metadata["tags"].contains([filters["tags"]]))

        # 4. Order and Limit
        stmt = stmt.order_by(text("rank DESC")).limit(limit)
        
        result = await db.execute(stmt)
        rows = result.all()
        
        return [
            {
                "id": str(row.Chunk.id),
                "content": row.Chunk.content,
                "filename": row.filename,
                "score": float(row.rank),
                "metadata": row.metadata_json or {}
            }
            for row in rows
        ]
