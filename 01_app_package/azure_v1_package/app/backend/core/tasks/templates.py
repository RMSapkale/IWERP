import json
from typing import List, Dict, Any
from core.schemas.api import UserIntent, Citation

# --- BASE PROMPTS ---

BASE_SYSTEM_PROMPT = """You are the Antigravity Modification Engine. 
Directly grounded Oracle Fusion ERP expert answers only.
NO FILLER. NO APOLOGIES. NO OUTRO. 
START IMMEDIATELY WITH [Answer]."""

STRUCTURED_INSTRUCTION = """\nIMPORTANT: Return your answer in exactly these three blocks: [Answer], [SQL], [Notes]. Do not use JSON. Do not use conversational filler."""

# --- TASK TEMPLATES ---

TASK_INSTRUCTIONS = {
    TaskType.QA: "Directly answer the question using facts from the context. Cite the context conceptually if helpful.",
    TaskType.SUMMARY: "Provide a comprehensive but concise summary of the provided context relating to the user's prompt.",
    TaskType.SQL_GENERATION: "Generate exact Oracle Fusion SQL queries. Adhere to the SQL GENERATION POLICY: Never invent tables, use confirmed schema only.",
    TaskType.NAVIGATION: "Provide precise navigation steps to reach a specific screen or page in Oracle Fusion Cloud Application.",
    TaskType.PROCEDURE: "Detail a standard operating procedure (SOP) or task-based guide based on the documentation.",
    TaskType.TROUBLESHOOTING: "Identify symptoms, diagnostic steps, and possible resolutions for the technical issue.",
    TaskType.INTEGRATION: "Detail technical integration specifications, including REST/SOAP endpoints and FBDI templates.",
    TaskType.REPORT_LOGIC: "Explain the underlying logic, tables, and joins required for a specific BI Publisher or OTBI report."
}

TASK_SCHEMAS = {
    TaskType.NAVIGATION: {
        "target_page": "string",
        "module": "string",
        "navigation_path": ["string"],
        "task_name": "string"
    },
    TaskType.PROCEDURE: {
        "title": "string",
        "prerequisites": ["string"],
        "steps": ["string"],
        "expected_result": "string"
    },
    TaskType.TROUBLESHOOTING: {
        "issue_description": "string",
        "diagnostic_sql": "string",
        "potential_causes": ["string"],
        "resolutions": ["string"]
    },
    TaskType.INTEGRATION: {
        "integration_style": "string",
        "endpoint": "string",
        "payload_structure": "string",
        "mapping_logic": "string"
    },
    TaskType.SQL_GENERATION: {
        "primary_tables": ["string"],
        "sql_query": "string",
        "explanation": "string"
    }
}

# --- HELPERS ---

def _format_context(citations: List[Citation]) -> str:
    if not citations:
        return "No relevant context found."
        
    formatted = []
    for i, cit in enumerate(citations):
        title = cit.title or "Unknown Source"
        formatted.append(f"--- Context {i+1}: {title} ---\n{cit.snippet}\n")
    return "\n".join(formatted)

def get_prompt_template(intent: UserIntent, query: str, citations: List[Citation]) -> str:
    """
    Returns the fully formatted system prompt with specialized task instructions and schema.
    """
    context_str = _format_context(citations)
    instruction = TASK_INSTRUCTIONS.get(intent, TASK_INSTRUCTIONS[UserIntent.QA])
    
    full_prompt = f"{BASE_SYSTEM_PROMPT}\n\nTask Type: {intent}\nInstruction: {instruction}\n\n"
    
    if intent in TASK_SCHEMAS:
        schema_json = json.dumps(TASK_SCHEMAS[intent], indent=2)
        full_prompt += f"{STRUCTURED_INSTRUCTION}\n\nSCHEMA:\n{schema_json}\n\n"
    
    full_prompt += f"Retrieved Context:\n{context_str}\n\n"
    full_prompt += f"Original Query: {query}"
    
    return full_prompt
