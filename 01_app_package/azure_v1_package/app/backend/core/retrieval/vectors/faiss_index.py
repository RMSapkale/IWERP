import json
import os
import sqlite3
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from core.ingest.curation import CuratedIngestionValidator

logger = structlog.get_logger(__name__)

try:
    import faiss  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in integration runtime
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:  # pragma: no cover - exercised in integration runtime
    SentenceTransformer = None


class _HashingEmbedder:
    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, texts: List[str], normalize_embeddings: bool = True) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for row_idx, text in enumerate(texts):
            tokens = str(text or "").lower().split()
            if not tokens:
                continue
            for token in tokens:
                slot = hash(token) % self.dim
                vectors[row_idx, slot] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        return vectors


class _NumpyInnerProductIndex:
    def __init__(self, dim: int, vectors: Optional[np.ndarray] = None):
        self.dim = dim
        self.vectors = (
            np.array(vectors, dtype="float32")
            if vectors is not None and len(vectors)
            else np.empty((0, dim), dtype="float32")
        )

    @property
    def ntotal(self) -> int:
        return int(self.vectors.shape[0])

    def add(self, batch: np.ndarray) -> None:
        batch = np.array(batch, dtype="float32")
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        self.vectors = np.vstack([self.vectors, batch]) if self.vectors.size else batch

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        query = np.array(query, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        batch_size = query.shape[0]
        if self.ntotal == 0:
            return (
                np.full((batch_size, k), -1.0, dtype="float32"),
                np.full((batch_size, k), -1, dtype="int64"),
            )

        scores = np.matmul(query, self.vectors.T)
        top_k = min(k, self.ntotal)
        order = np.argsort(-scores, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(scores, order, axis=1)

        if top_k < k:
            padded_scores = np.full((batch_size, k), -1.0, dtype="float32")
            padded_indices = np.full((batch_size, k), -1, dtype="int64")
            padded_scores[:, :top_k] = top_scores
            padded_indices[:, :top_k] = order
            return padded_scores, padded_indices
        return top_scores.astype("float32"), order.astype("int64")


class FaissIndex:
    """
    Manages per-tenant, per-corpus FAISS indexes with SQLite metadata storage.
    """

    INTERNAL_FILTER_KEYS = {
        "corpora",
        "quality_score_min",
        "schema_limit",
        "requested_module",
        "exact_module_allowlist",
        "strict_exact_module_only",
        "allow_same_family_fallback",
    }
    SQLITE_MAX_VARS = 900

    def __init__(
        self,
        tenant_id: str,
        indexes_dir: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        vector_dim: int = 384,
        corpus: str = "docs",
    ):
        self.tenant_id = tenant_id
        self.corpus = corpus
        self.faiss_dir = Path(indexes_dir) / "faiss" / tenant_id / corpus
        self.faiss_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.faiss_dir / "faiss.index"
        self.fallback_index_path = self.faiss_dir / "faiss_fallback.npy"
        self.db_path = self.faiss_dir / "metadata.sqlite"
        self.dim = vector_dim
        force_hash = os.getenv("IWFUSION_FORCE_HASH_EMBEDDER", "false").lower() == "true"
        offline_mode = os.getenv("HF_HUB_OFFLINE", "").lower() in {"1", "true"} or os.getenv("TRANSFORMERS_OFFLINE", "").lower() in {"1", "true"}
        if force_hash or SentenceTransformer is None:
            self.model = _HashingEmbedder(self.dim)
        else:
            try:
                self.model = SentenceTransformer(embedding_model, local_files_only=offline_mode)
            except Exception as exc:  # pragma: no cover - fallback path depends on runtime setup
                logger.warning(
                    "faiss_embedder_init_fallback_hash",
                    tenant_id=tenant_id,
                    corpus=corpus,
                    embedding_model=embedding_model,
                    error=str(exc),
                )
                self.model = _HashingEmbedder(self.dim)

        self._init_db()
        self.index = self._load_index()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    vector_id INTEGER PRIMARY KEY,
                    chunk_id TEXT NOT NULL UNIQUE,
                    tenant_id TEXT NOT NULL,
                    corpus TEXT NOT NULL,
                    document_id TEXT,
                    content_hash TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
            if "content_hash" not in columns:
                conn.execute("ALTER TABLE chunks ADD COLUMN content_hash TEXT")
                conn.execute("UPDATE chunks SET content_hash = chunk_id WHERE content_hash IS NULL OR content_hash = ''")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_chunk_id ON chunks(chunk_id)")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_content_hash ON chunks(tenant_id, corpus, content_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tenant_corpus ON chunks(tenant_id, corpus)")

    def _load_index(self):
        if faiss is not None:
            if self.index_path.exists():
                return faiss.read_index(str(self.index_path))
            return faiss.IndexFlatIP(self.dim)

        if self.fallback_index_path.exists():
            vectors = np.load(self.fallback_index_path)
            return _NumpyInnerProductIndex(self.dim, vectors=vectors)
        return _NumpyInnerProductIndex(self.dim)

    def _normalize_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(chunk.get("metadata") or {})
        filename = (
            metadata.get("filename")
            or metadata.get("title")
            or metadata.get("file")
            or metadata.get("source")
            or chunk.get("document_id")
            or "grounding-source"
        )
        metadata.setdefault("filename", str(filename))
        metadata.setdefault("title", str(metadata.get("title") or metadata["filename"]))
        metadata.setdefault("source_uri", str(metadata.get("source_uri") or metadata.get("file") or metadata.get("source") or metadata["filename"]))
        metadata["tenant_id"] = self.tenant_id
        metadata["corpus"] = self.corpus
        return metadata

    def _existing_chunk_ids(self, chunk_ids: List[str]) -> set[str]:
        if not chunk_ids:
            return set()
        found: set[str] = set()
        with sqlite3.connect(self.db_path) as conn:
            for start in range(0, len(chunk_ids), self.SQLITE_MAX_VARS):
                batch = chunk_ids[start:start + self.SQLITE_MAX_VARS]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT chunk_id FROM chunks WHERE chunk_id IN ({placeholders})",
                    batch,
                ).fetchall()
                found.update(row[0] for row in rows)
        return found

    def _existing_content_hashes(self, content_hashes: List[str]) -> set[str]:
        if not content_hashes:
            return set()
        found: set[str] = set()
        with sqlite3.connect(self.db_path) as conn:
            max_batch = max(self.SQLITE_MAX_VARS - 2, 1)
            for start in range(0, len(content_hashes), max_batch):
                batch = content_hashes[start:start + max_batch]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT content_hash FROM chunks WHERE tenant_id = ? AND corpus = ? AND content_hash IN ({placeholders})",
                    [self.tenant_id, self.corpus, *batch],
                ).fetchall()
                found.update(row[0] for row in rows if row and row[0])
        return found

    def reset(self) -> None:
        if self.faiss_dir.exists():
            shutil.rmtree(self.faiss_dir)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.index = self._load_index()

    def stats(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total_chunks, COUNT(DISTINCT content_hash) AS unique_content_hashes FROM chunks"
            ).fetchone()
        return {
            "tenant_id": self.tenant_id,
            "corpus": self.corpus,
            "faiss_dir": str(self.faiss_dir),
            "index_path": str(self.index_path),
            "db_path": str(self.db_path),
            "total_vectors": int(self.index.ntotal),
            "total_chunks": int(row[0]) if row else 0,
            "unique_content_hashes": int(row[1]) if row else 0,
        }

    def build_from_chunks(self, chunks_jsonl_path: str, batch_size: int = 64) -> None:
        """
        Builds the FAISS index from a JSONL file of chunks.
        """
        if not os.path.exists(chunks_jsonl_path):
            logger.error("chunks_file_not_found", path=chunks_jsonl_path)
            return

        chunks = []
        with open(chunks_jsonl_path, "r") as f:
            for line in f:
                chunks.append(json.loads(line))

        self.add_chunks_list(chunks, batch_size)

    def add_chunks_list(self, chunks: List[Dict[str, Any]], batch_size: int = 64) -> None:
        """
        Adds a list of chunks to the FAISS index and metadata store without duplicating chunk IDs.
        """
        if not chunks:
            logger.info("no_chunks_to_index", tenant_id=self.tenant_id, corpus=self.corpus)
            return

        deduped_chunks: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
            if not chunk_id:
                continue
            normalized = dict(chunk)
            normalized["chunk_id"] = chunk_id
            normalized["metadata"] = self._normalize_metadata(normalized)
            normalized["content_hash"] = str(
                normalized.get("content_hash")
                or normalized["metadata"].get("content_hash")
                or chunk_id
            )
            deduped_chunks.setdefault(chunk_id, normalized)

        existing_ids = self._existing_chunk_ids(list(deduped_chunks.keys()))
        existing_hashes = self._existing_content_hashes([chunk["content_hash"] for chunk in deduped_chunks.values()])
        pending_chunks = []
        pending_hashes = set()
        for chunk_id, chunk in deduped_chunks.items():
            content_hash = chunk["content_hash"]
            if chunk_id in existing_ids:
                continue
            if content_hash in existing_hashes or content_hash in pending_hashes:
                continue
            pending_hashes.add(content_hash)
            pending_chunks.append(chunk)
        if not pending_chunks:
            logger.info("faiss_index_no_new_chunks", tenant_id=self.tenant_id, corpus=self.corpus)
            return

        for i in range(0, len(pending_chunks), batch_size):
            batch = pending_chunks[i:i + batch_size]
            contents = [chunk["content"] for chunk in batch]
            embeddings = self.model.encode(contents, normalize_embeddings=True)

            start_idx = self.index.ntotal
            self.index.add(np.array(embeddings).astype("float32"))

            with sqlite3.connect(self.db_path) as conn:
                for j, chunk in enumerate(batch):
                    vector_id = start_idx + j
                    conn.execute(
                        """
                        INSERT INTO chunks (vector_id, chunk_id, tenant_id, corpus, document_id, content_hash, content, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            vector_id,
                            chunk["chunk_id"],
                            self.tenant_id,
                            self.corpus,
                            chunk.get("document_id"),
                            chunk["content_hash"],
                            chunk["content"],
                            json.dumps(chunk.get("metadata", {})),
                        ),
                    )

        if faiss is not None:
            faiss.write_index(self.index, str(self.index_path))
        else:
            np.save(self.fallback_index_path, getattr(self.index, "vectors", np.empty((0, self.dim), dtype="float32")))
        logger.info(
            "indexing_completed",
            tenant_id=self.tenant_id,
            corpus=self.corpus,
            total=self.index.ntotal,
            added=len(pending_chunks),
        )

    def _matches_filters(self, metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return CuratedIngestionValidator.is_curated_metadata(metadata, metadata.get("corpus"))

        quality_score_min = filters.get("quality_score_min")
        if quality_score_min is not None:
            if float(metadata.get("quality_score") or 0.0) < float(quality_score_min):
                return False

        corpora = filters.get("corpora")
        if corpora:
            if metadata.get("corpus") not in {str(value) for value in corpora}:
                return False

        if not CuratedIngestionValidator.is_curated_metadata(metadata, metadata.get("corpus")):
            return False

        for key, expected in filters.items():
            if expected in (None, "", [], (), {}):
                continue
            if key == "corpus":
                if metadata.get("corpus") != expected:
                    return False
                continue
            if key in self.INTERNAL_FILTER_KEYS:
                continue

            actual = metadata.get(key)
            if actual is None:
                return False
            if isinstance(expected, (list, tuple, set)):
                if str(actual) not in {str(v) for v in expected}:
                    return False
            elif str(actual) != str(expected):
                return False
        return True

    def query(self, text: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Queries the index and returns top_k results with metadata.
        """
        if self.index.ntotal == 0:
            return []

        embedding = self.model.encode([text], normalize_embeddings=True)
        candidate_k = min(max(top_k * 5, top_k), self.index.ntotal)
        if filters:
            candidate_k = min(max(candidate_k, top_k * 20, 200), self.index.ntotal)

        results = []
        seen_vector_ids = set()
        while True:
            scores, indices = self.index.search(np.array(embedding).astype("float32"), candidate_k)

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1 or int(idx) in seen_vector_ids:
                        continue
                    seen_vector_ids.add(int(idx))

                    row = conn.execute("SELECT * FROM chunks WHERE vector_id = ?", (int(idx),)).fetchone()
                    if not row:
                        continue

                    payload = dict(row)
                    metadata = json.loads(payload["metadata"])
                    if not self._matches_filters(metadata, filters):
                        continue

                    payload["metadata"] = metadata
                    payload["score"] = float(score)
                    payload["filename"] = metadata.get("filename")
                    payload["title"] = metadata.get("title")
                    payload["source_uri"] = metadata.get("source_uri")
                    results.append(payload)

                    if len(results) >= top_k:
                        break

            if len(results) >= top_k or candidate_k >= self.index.ntotal:
                break
            candidate_k = min(candidate_k * 2, self.index.ntotal)

        return results[:top_k]
