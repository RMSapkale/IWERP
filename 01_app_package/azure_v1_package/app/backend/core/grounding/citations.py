from typing import List, Dict, Any

class CitationMapper:
    """
    Maps retrieved chunks to citation identifiers like [D1], [D2], etc.
    Ensures consistent numbering and provides a formatted block for context.
    """
    @staticmethod
    def _normalize_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
        metadata = chunk.get("metadata") or {}
        filename = (
            chunk.get("filename")
            or metadata.get("filename")
            or metadata.get("title")
            or metadata.get("file")
            or metadata.get("source")
            or "grounding-source"
        )
        source_uri = (
            metadata.get("source_uri")
            or metadata.get("source_path")
            or metadata.get("file")
            or metadata.get("source")
            or filename
        )
        title = metadata.get("title") or filename
        module = metadata.get("module") or metadata.get("module_family") or "UNKNOWN"
        source = metadata.get("source") or metadata.get("source_path") or source_uri
        corpus = metadata.get("corpus") or "unknown"
        snippet = (chunk.get("content") or metadata.get("snippet") or title or "").strip()
        task_signals = metadata.get("task_signals") or []
        task_confidence = float(metadata.get("task_confidence") or 0.0)
        task_match_score = float(metadata.get("task_match_score") or 0.0)
        task_match_strength = str(metadata.get("task_match_strength") or "none")

        normalized = dict(chunk)
        normalized["metadata"] = metadata
        normalized["filename"] = str(filename)
        normalized["title"] = str(title)
        normalized["source_uri"] = str(source_uri)
        normalized["source"] = str(source)
        normalized["module"] = str(module)
        normalized["corpus"] = str(corpus)
        normalized["snippet"] = snippet
        normalized["task_signals"] = list(task_signals)
        normalized["task_confidence"] = task_confidence
        normalized["task_match_score"] = task_match_score
        normalized["task_match_strength"] = task_match_strength
        normalized["document_id"] = str(
            chunk.get("document_id")
            or metadata.get("document_id")
            or metadata.get("doc_id")
            or filename
        )
        return normalized

    @staticmethod
    def map_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Adds a 'citation_id' to each chunk.
        """
        mapped_chunks = []
        for i, chunk in enumerate(chunks):
            normalized = CitationMapper._normalize_chunk(chunk)
            normalized["citation_id"] = f"[D{i+1}]"
            mapped_chunks.append(normalized)
        return mapped_chunks

    @staticmethod
    def to_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        citations = []
        for chunk in CitationMapper.map_chunks(chunks):
            snippet = (chunk.get("snippet") or chunk.get("content") or "").strip()[:300]
            if not snippet:
                snippet = chunk["title"]
            citations.append(
                {
                    "citation_id": chunk["citation_id"],
                    "document_id": chunk["document_id"],
                    "title": chunk["title"],
                    "snippet": snippet,
                    "module": chunk.get("module") or "UNKNOWN",
                    "source": chunk.get("source") or chunk["source_uri"],
                    "corpus": chunk.get("corpus") or "unknown",
                    "score": float(chunk.get("combined_score") or chunk.get("score") or 0.0),
                    "source_uri": chunk["source_uri"],
                }
            )
        return citations

    @staticmethod
    def format_context_block(chunks: List[Dict[str, Any]]) -> str:
        """
        Formats chunks into a readable block for the LLM prompt.
        """
        if not chunks:
            return "No relevant documentation found."

        lines = []
        for chunk in CitationMapper.map_chunks(chunks):
            cid = chunk.get("citation_id", "[UNK]")
            title = chunk.get("title", "grounding-source")
            module = chunk.get("module", "UNKNOWN")
            corpus = chunk.get("corpus", "unknown")
            source = chunk.get("source_uri", "grounding-source")
            content = (chunk.get("content") or chunk.get("snippet") or title or "").strip()[:1200]
            task_signals = ", ".join(chunk.get("task_signals") or []) or "none"
            task_match = chunk.get("task_match_strength") or "none"
            lines.append(
                f"DOCUMENT {cid}\n"
                f"Title: {title}\n"
                f"Module: {module}\n"
                f"Corpus: {corpus}\n"
                f"Source: {source}\n"
                f"TaskSignals: {task_signals}\n"
                f"TaskMatch: {task_match}\n"
                f"Excerpt:\n{content}\n"
            )

        return "\n".join(lines)
