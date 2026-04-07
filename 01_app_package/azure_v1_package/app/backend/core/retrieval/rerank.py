from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from core.schemas.config import RetrievalSettings

class CrossEncoderReranker:
    """
    Reranks candidate documents from BM25 and Vector search to improve precision.
    Must be instantiated only once to avoid memory bloat.
    """
    def __init__(self, settings: RetrievalSettings):
        self.model = CrossEncoder(settings.reranker_model, max_length=512)

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks a unified list of document dictionaries.
        Documents must contain a 'content' field.
        """
        if not documents:
            return []
            
        # Prepare inputs for cross encoder (query, document) pairs
        pairs = [[query, doc["content"]] for doc in documents]
        
        # Predict relevancy scores
        scores = self.model.predict(pairs)
        
        # Zip scores, update documents, and sort
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
            
        # Sort descending by newly acquired scores
        ranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]
