import structlog
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder

logger = structlog.get_logger(__name__)

class LocalReranker:
    """
    Second-stage reranker using a local Cross-Encoder model.
    Implements a singleton pattern for the model instance.
    """
    _model = None

    def __init__(self, model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        if LocalReranker._model is None:
            logger.info("loading_reranker_model", model_name=model_name)
            LocalReranker._model = CrossEncoder(model_name, max_length=max_length)

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate documents against a query.
        Returns the top_k most relevant documents.
        """
        if not documents:
            return []

        # 1. Prepare pairs for cross-encoding
        # Cross-encoders take (query, text) pairs
        pairs = []
        for doc in documents:
            # Safe truncation: CrossEncoder handles this via max_length, 
            # but we can also do a pre-truncation if needed.
            pairs.append([query, doc["content"]])

        # 2. Compute relevance scores
        logger.info("reranking_candidates", count=len(documents), query=query[:50])
        scores = LocalReranker._model.predict(pairs)

        # 3. Update documents with new scores and sort
        for i, score in enumerate(scores):
            prior_score = float(documents[i].get("combined_score") or documents[i].get("score") or 0.0)
            documents[i]["rerank_score"] = float(score)
            # Blend semantic relevance with retrieval prior so module/corpus routing survives reranking.
            documents[i]["final_rank_score"] = float(score) + (prior_score * 0.75)

        reranked_docs = sorted(
            documents, 
            key=lambda x: x.get("final_rank_score", x["rerank_score"]), 
            reverse=True
        )

        # 4. Return top K
        return reranked_docs[:top_k]
