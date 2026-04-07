import asyncio
import structlog
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from core.retrieval.hybrid import HybridPostgresSearch
from core.llm.llama_cpp_client import LlamaCppClient
from core.schemas.api import ChatRequest, Message, Role

logger = structlog.get_logger(__name__)

class RAGFusionEngine:
    """
    Experimental retrieval upgrade for Version 2.
    Implements:
    1. Multi-query generation (RAG-Fusion pattern)
    2. Parallel search execution
    3. Reciprocal Rank Fusion (RRF) for robust merging
    """
    def __init__(
        self, 
        base_search: Optional[HybridPostgresSearch] = None,
        llm: Optional[LlamaCppClient] = None
    ):
        self.search_engine = base_search or HybridPostgresSearch()
        self.llm = llm or LlamaCppClient()

    async def generate_queries(self, original_query: str, n: int = 3) -> List[str]:
        """
        Uses the SLM to generate n semantic variations of the original query.
        """
        prompt = (
            f"You are an AI assistant helping a user perform deep research on Oracle Fusion Cloud. "
            f"Generate {n} different variations/rephrasings of the following user query to help "
            f"retrieve more relevant documentation. Output only the variations, one per line.\n"
            f"Original Query: {original_query}"
        )
        
        request = ChatRequest(
            messages=[Message(role=Role.USER, content=prompt)],
            temperature=0.7, # Higher temperature for variety
            max_tokens=200
        )
        
        try:
            response = await self.llm.chat(request)
            content = response["choices"][0]["message"]["content"]
            variations = [line.strip("- ").strip() for line in content.split("\n") if line.strip()]
            
            # Ensure we have at least the original and meaningful variations
            final_queries = [original_query] + variations[:n]
            logger.info("rag_fusion_queries_generated", count=len(final_queries))
            return list(set(final_queries)) # Deduplicate
        except Exception as e:
            logger.error("rag_fusion_query_gen_failed", error=str(e))
            return [original_query]

    def reciprocal_rank_fusion(self, results_list: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        """
        Combines multiple ranked lists using the RRF algorithm.
        Score = sum(1 / (k + rank))
        """
        fused_scores = {} # chunk_id -> score
        chunk_data = {}   # chunk_id -> metadata
        
        for results in results_list:
            for rank, hit in enumerate(results, 1):
                chunk_id = hit["id"]
                if chunk_id not in fused_scores:
                    fused_scores[chunk_id] = 0.0
                    chunk_data[chunk_id] = hit
                
                fused_scores[chunk_id] += 1.0 / (k + rank)
        
        # Convert back to list and sort by fused score
        final_results = []
        for chunk_id, score in fused_scores.items():
            item = chunk_data[chunk_id].copy()
            item["rrf_score"] = score
            final_results.append(item)
            
        return sorted(final_results, key=lambda x: x["rrf_score"], reverse=True)

    async def search(
        self,
        db: AsyncSession,
        tenant_id: str,
        query: str,
        limit: int = 10,
        num_queries: int = 3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Orchestrates the RAG-Fusion search flow.
        """
        # 1. Generate variations
        queries = await self.generate_queries(query, n=num_queries)
        
        # 2. Parallel search (Sequential awaits to keep DB session safe)
        all_results = []
        for q in queries:
            results = await self.search_engine.search(db, tenant_id, q, limit=40, filters=filters)
            all_results.append(results)
            
        # 3. Fuse results
        fused_results = self.reciprocal_rank_fusion(all_results)
        
        logger.info("rag_fusion_complete", query=query, variations=len(queries), total_hits=len(fused_results))
        
        return fused_results[:limit]
