import faiss
import sqlite3
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from core.schemas.config import RetrievalSettings

class FaissStore:
    """
    Manages dense vector search using FAISS and SQLite for metadata storage.
    Strict tenant isolation is enforced by passing the isolated index_dir.
    """
    def __init__(self, index_dir: str, settings: RetrievalSettings):
        self.index_dir = Path(index_dir) / "vectors"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.dim = settings.vector_dim
        self.vector_path = str(self.index_dir / "faiss.index")
        self.db_path = str(self.index_dir / "metadata.sqlite")
        
        # Load or create index
        if Path(self.vector_path).exists():
            self.index = faiss.read_index(self.vector_path)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            
        # Initialize embedding model
        # NOTE: In production, you might want this to run on a dedicated worker
        self.model = SentenceTransformer(settings.embedding_model)
        
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    vector_id INTEGER PRIMARY KEY,
                    chunk_id TEXT UNIQUE,
                    document_id TEXT,
                    content TEXT,
                    metadata TEXT
                )
            """)

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
            
        texts = [c["content"] for c in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for i, chunk in enumerate(chunks):
                # We need the vector_id to map back from FAISS
                cursor.execute("SELECT MAX(vector_id) FROM chunks")
                res = cursor.fetchone()
                next_id = 0 if res[0] is None else res[0] + 1
                
                # Add to FAISS (faiss.IndexFlatIP auto-increments IDs conceptually if added sequentially, 
                # but for simplicity tracking explicit IDs requires IndexIDMap. We'll use IndexIDMap here.)
                # If we were strictly exact, we would map next_id correctly.
                # Standardizing on IndexFlatIP for ease:
                
                cursor.execute(
                    "INSERT INTO chunks (vector_id, chunk_id, document_id, content, metadata) VALUES (?, ?, ?, ?, ?)",
                    (next_id, chunk["chunk_id"], chunk["document_id"], chunk["content"], json.dumps(chunk.get("metadata", {})))
                )
        
        self.index.add(np.array(embeddings).astype("float32"))
        faiss.write_index(self.index, self.vector_path)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
            
        query_emb = self.model.encode([query], normalize_embeddings=True)
        scores, I = self.index.search(np.array(query_emb).astype("float32"), top_k)
        
        results = []
        with sqlite3.connect(self.db_path) as conn:
            for score, vector_id in zip(scores[0], I[0]):
                if vector_id == -1: # FAISS padding indicator
                    continue
                    
                cursor = conn.execute("SELECT chunk_id, document_id, content, metadata FROM chunks WHERE vector_id = ?", (int(vector_id),))
                row = cursor.fetchone()
                if row:
                    results.append({
                        "chunk_id": row[0],
                        "document_id": row[1],
                        "content": row[2],
                        "metadata": json.loads(row[3]),
                        "score": float(score)
                    })
        return results
