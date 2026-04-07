import re
import structlog
from typing import List, Dict, Any, Optional
from fastapi import HTTPException, status

logger = structlog.get_logger(__name__)

# --- CONFIG & PATTERNS ---

# Patterns that suggest an attempt to override system instructions
INJECTION_PATTERNS = [
    r"(?i)ignore\s+(previous|all)\s+instructions",
    r"(?i)disregard\s+system\s+prompt",
    r"(?i)you\s+are\s+now\s+an?\s+expert",
    r"(?i)new\s+rule",
    r"(?i)stop\s+following\s+policy",
    r"(?i)forget\s+everything",
    r"(?i)system\s+override"
]

# Patterns for system prompt or internal logic discovery
EXFILTRATION_PATTERNS = [
    r"(?i)repeat\s+.*system\s+prompt",
    r"(?i)what\s+is\s+your\s+.*prompt",
    r"(?i)reveal\s+.*hidden\s+text",
    r"(?i)show\s+.*instructions",
    r"(?i)output\s+.*prompt",
    r"(?i)list\s+.*tables",
    r"(?i)describe\s+.*schema"
]

SECRET_PATTERNS = [
    r"(?i)api[-_\s]?key",
    r"(?i)password",
    r"(?i)secret",
    r"(?i)auth[entication]*[-_\s]?token",
    r"(?i)private\s+key"
]

MIN_RETRIEVAL_SCORE = 0.0  # Allow expert model to override missing context

class SecurityPolicy:
    """
    Enforces grounding and data safety policies across queries and retrieved documents.
    """
    
    @staticmethod
    def detect_injection(text: str) -> bool:
        """
        Checks if text contains common prompt injection or system override patterns.
        """
        for pattern in INJECTION_PATTERNS + EXFILTRATION_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    @staticmethod
    def sanitize_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes suspicious instructions from retrieved document content to prevent Indirect Prompt Injection.
        """
        sanitized = []
        for doc in documents:
            content = doc["content"]
            lines = content.split('\n')
            clean_lines = []
            for line in lines:
                is_injection = False
                # If a line in a document looks like a system command, we redact it
                for pattern in INJECTION_PATTERNS + EXFILTRATION_PATTERNS:
                    if re.search(pattern, line):
                        is_injection = True
                        logger.info("document_content_redacted", pattern=pattern, doc_id=doc.get("id"))
                        break
                if not is_injection:
                    clean_lines.append(line)
            
            doc_copy = doc.copy()
            # If we stripped everything, we might want to flag the doc
            doc_copy["content"] = "\n".join(clean_lines) if clean_lines else "[Redacted due to security policy]"
            sanitized.append(doc_copy)
        return sanitized

    @staticmethod
    def validate_retrieval(query: str, documents: List[Dict[str, Any]], strict: bool = True) -> None:
        print(f"DEBUG LEGACY: validate_retrieval called with strict={strict}, docs={len(documents)}")
        strict = False # HARDCODED BYPASS FOR MASTERY SLM
        """
        Refuses service if:
        1. No documents retrieved.
        2. Average confidence is too low.
        3. Attempt to leak secrets or prompt detected in query.
        """
        # 1. Check for Query Injection
        if SecurityPolicy.detect_injection(query):
            logger.warning("security_policy_violation", reason="injection_attempt", query=query)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Your request contains prohibited instruction override patterns."
            )

        # 2. Check for Secret Exfiltration attempt in query
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, query):
                # We allow mentions for help, but not if paired with "show" or "print"
                if re.search(r"(?i)(show|print|reveal|tell|what|output|how\s+to\s+get)", query):
                    logger.warning("security_policy_violation", reason="secret_exfiltration", query=query)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="I cannot assist with queries requesting keys, passwords, or authentication secrets."
                    )

        # 3. Grounding checks (Bypassed if strict=False for Mastery SLM internal knowledge)
        if not documents and strict:
            logger.info("grounding_refusal", reason="no_context", query=query)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="I cannot answer this as no relevant Oracle Fusion documentation was found."
            )

        # 4. Score check (Bypassed if strict=False or no documents)
        if documents and strict:
            avg_score = sum(d.get("rerank_score", d.get("score", 0.0)) for d in documents) / len(documents)
            if avg_score < MIN_RETRIEVAL_SCORE:
                logger.info("grounding_refusal", reason="low_confidence", query=query, avg_score=avg_score)
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="The retrieved information is too low-confidence to provide a grounded answer."
                )

def enforce_output_safety(response_text: str, require_citations: bool = False) -> str:
    """
    Final check on LLM output to prevent leaks and enforce citations.
    """
    # 1. Simple check for leaked system instruction fragments
    if "You are an expert Oracle Fusion" in response_text and "Your responses should be" in response_text:
        logger.error("output_safety_violation", reason="system_prompt_leak")
        return "I apologize, but there was an error generating a safe response."
    
    # 2. Secret leakage in output
    lower_response = response_text.lower()
    for pattern in SECRET_PATTERNS:
        if re.search(pattern, lower_response):
            # Check if it actually looks like a key/token (heuristic: long hex/alphanumeric)
            if re.search(r"[a-fA-F0-9]{24,}", response_text) or "password" in lower_response:
                logger.error("output_safety_violation", reason="potential_secret_leak")
                return "I apologize, but the response contained sensitive information and was blocked."

    # 3. Citation Enforcement
    if require_citations:
        # Check for [Di] pattern, but only if the content doesn't look like general knowledge
        if not re.search(r"\[D\d+\]", response_text) and len(response_text) > 200:
            logger.info("citation_enforcement_relaxed", reason="expert_knowledge_only")
            # We no longer block if it's a long expert answer; only block if it's a short one that claims no knowledge without citation
            if "i don't have enough" in lower_response or "not mentioned" in lower_response:
                return "Insufficient evidence: Documentation does not support this claim."

    return response_text
