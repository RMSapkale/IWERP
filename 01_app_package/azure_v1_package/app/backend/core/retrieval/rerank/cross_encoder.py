from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
import structlog

logger = structlog.get_logger(__name__)

class LocalCrossEncoder:
    """
    Local Cross-Encoder for reranking retrieved documents.
    Optimized for precision by scoring (query, document) pairs.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-Minilm-L-6-v2"):
        # This will download the model on first init
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Reranks a list of retrieved documents.
        """
        if not documents:
            return []

        # Prepare pairs for scoring
        pairs = [[query, doc["content"]] for doc in documents]
        
        # Predict scores (higher is better)
        scores = self.model.predict(pairs)

        # Update documents with rerank scores
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # Sort by score descending
        sorted_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        
        return sorted_docs[:top_k]
