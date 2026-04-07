import tantivy
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

class TantivyBM25:
    """
    Manages multi-field BM25 keyword retrieval using Tantivy.
    Supports weighted searching across title, headings, and body.
    Each tenant has strictly isolated indexes.
    """
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir) / "bm25"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Schema definition
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("chunk_id", stored=True)
        schema_builder.add_text_field("document_id", stored=True)
        schema_builder.add_text_field("title", stored=True, index=True)
        schema_builder.add_text_field("headings", stored=True, index=True)
        schema_builder.add_text_field("body", stored=True, index=True)
        schema_builder.add_text_field("section_path", stored=True)
        schema_builder.add_text_field("module", stored=True, index=True)
        self.schema = schema_builder.build()
        
        try:
            self.index = tantivy.Index(self.schema, path=str(self.index_dir))
        except ValueError:
            # Create if doesn't exist
            self.index = tantivy.Index(self.schema, path=str(self.index_dir))

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Adds multiple chunks to the index.
        """
        writer = self.index.writer()
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            writer.add_document(
                tantivy.Document(
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    title=metadata.get("title", ""),
                    headings=metadata.get("headings", ""),
                    body=chunk["content"],
                    section_path=metadata.get("section_path", ""),
                    module=metadata.get("module", "")
                )
            )
        writer.commit()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs a weighted search across fields.
        Weights: title: 2.0, headings: 1.5, body: 1.0
        """
        self.index.reload()
        searcher = self.index.searcher()
        
        # Weighted query parsing
        # Note: tantivy-py supports weighted fields in parse_query via a dict or similar if configured,
        # but standard way is often boosting in the query string or separate sub-queries.
        # Here we use the boost syntax or field-specific weights if the API allows.
        # As of current tantivy-py, we can pass weights to the query parser.
        weights = {
            "title": 2.0,
            "headings": 1.5,
            "body": 1.0
        }
        
        q = self.index.parse_query(query, ["title", "headings", "body"], weights=weights)
        
        results = searcher.search(q, top_k).hits
        
        retrieved = []
        for score, doc_address in results:
            doc = searcher.doc(doc_address)
            retrieved.append({
                "chunk_id": doc["chunk_id"][0],
                "document_id": doc["document_id"][0],
                "title": doc["title"][0],
                "headings": doc["headings"][0],
                "content": doc["body"][0],
                "section_path": doc["section_path"][0],
                "module": doc["module"][0],
                "score": float(score)
            })
        return retrieved

    def clear(self):
        """
        Clears the index for a fresh rebuild.
        """
        writer = self.index.writer()
        writer.delete_all_documents()
        writer.commit()
