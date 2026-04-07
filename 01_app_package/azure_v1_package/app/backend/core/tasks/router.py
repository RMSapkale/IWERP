import json
import structlog
from typing import Any, Dict, List
from pydantic import ValidationError
from core.tasks.schemas import FusionNavigation, FusionProcedure, FusionTroubleshoot, FusionIntegration
from core.schemas.api import UserIntent

logger = structlog.get_logger(__name__)

# Mapping from intent to Pydantic model
SCHEMA_MAP = {
    UserIntent.FUSION_NAV: FusionNavigation,
    UserIntent.FUSION_PROC: FusionProcedure,
    UserIntent.FUSION_TROUBLESHOOT: FusionTroubleshoot,
    UserIntent.FUSION_INTEGRATION: FusionIntegration
}

def intent_router(query: str) -> UserIntent:
    """
    Routes the query to a specific task type based on keywords and patterns.
    """
    q = query.lower()
    
    # 1. Navigation Tasks
    nav_keywords = ["how to go to", "where is", "target page", "navigation", "screen path", "find page"]
    if any(k in q for k in nav_keywords):
        return UserIntent.FUSION_NAV
        
    # 2. Procedure Tasks
    proc_keywords = ["how to", "procedure", "steps to", "guide", "process", "sop", "standard operating"]
    if any(k in q for k in proc_keywords) and not any(k in q for k in nav_keywords):
        return UserIntent.FUSION_PROC
        
    # 3. Troubleshooting Tasks
    trouble_keywords = ["error", "issue", "failed", "bug", "broken", "troubleshoot", "why is", "diagnostic"]
    if any(k in q for k in trouble_keywords):
        return UserIntent.FUSION_TROUBLESHOOT
        
    # 4. Integration Tasks
    integ_keywords = ["integration", "rest api", "soap", "fbdi", "bip", "endpoint", "mapping", "payload", "webhook"]
    if any(k in q for k in integ_keywords):
        return UserIntent.FUSION_INTEGRATION
        
    # Legacy Heuristics
    code_keywords = ["write", "code", "generate", "script", "sql", "pl/sql", "python", "api"]
    if any(keyword in q for keyword in code_keywords) and "?" not in query:
        return UserIntent.CODE_GEN
        
    summary_keywords = ["summarize", "tl;dr", "summary", "brief"]
    if any(keyword in q for keyword in summary_keywords):
        return UserIntent.SUMMARY
        
    return UserIntent.QA

def validate_task_output(intent: UserIntent, content: str, citations: List) -> Dict[str, Any]:
    """
    Validates the LLM output against the intent-specific schema.
    Also enforces citation presence for specialized tasks.
    """
    if intent not in SCHEMA_MAP:
        return {"content": content, "structured": False}
        
    # 1. Enforcement: Citation presence
    if not citations:
        logger.error("validation_failed", intent=intent, reason="citations_missing")
        return {
            "content": "Error: This task requires information from the Oracle Fusion documentation, but no relevant context was found.", 
            "structured": False, 
            "valid": False
        }

    # 2. Validation: JSON parsing and Pydantic check
    try:
        json_str = content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
            
        data = json.loads(json_str)
        model = SCHEMA_MAP[intent]
        validated = model(**data)
        
        return {
            "content": validated.dict(),
            "structured": True,
            "valid": True
        }
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error("validation_failed", intent=intent, error=str(e))
        return {
            "content": f"The response was generated but did not match the required structured format: {str(e)}",
            "raw_content": content,
            "structured": True,
            "valid": False
        }
