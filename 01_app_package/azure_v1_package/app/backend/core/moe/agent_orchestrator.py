import os
import sys
import json
import re
import importlib.util
from typing import Dict, Any, List, Optional
from pathlib import Path

# Insert project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MoAOrchestrator:
    """
    Mixture of Agents (MoA) Orchestrator.
    Routes queries to 16+ specialized experts from the oracle_fusion_agents_bundle.
    """
    
    def __init__(self, bundle_dir: str = "oracle_fusion_agents_bundle"):
        self.bundle_dir = Path(bundle_dir)
        self.registry = self._load_experts()
        
    def _load_experts(self) -> Dict[str, str]:
        """Maps intents to their respective agent scripts based on AGENTS_OVERVIEW.md."""
        return {
            "sql": "sql_engine.py",
            "sql_validate": "sql_validator_and_repair.py",
            "hcm_functional": "hcm_functional_expert.py",
            "hcm_setup": "hcm_setup_navigator.py",
            "hcm_formula": "hcm_fast_formula_expert.py",
            "oic_consultant": "oic_consultant.py",
            "oic_architect": "oic_architect.py",
            "oci": "oci_infrastructure_agent.py",
            "scm_functional": "scm_functional_assistant.py",
            "procurement_rag": "procurement_knowledge_rag.py",
            "scm_rag": "scm_knowledge_rag.py",
            "intent": "intent_analyzer_router.py",
            "planner": "multi_hop_planner.py",
            "workflow": "agent_workflow_definition.py"
        }

    def parse_routing_token(self, model_output: str) -> Optional[str]:
        """
        Extracts a routing intent from the 8B model's output.
        Expects format: [ROUTE: INTENT]
        """
        match = re.search(r'\[ROUTE:\s*([A-Z0-9_]+)\]', model_output.upper())
        if match:
            intent = match.group(1).lower()
            return intent if intent in self.registry else None
        return None

    def get_expert_response(self, intent: str, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Dispatches a query to a specific expert.
        """
        agent_file = self.registry.get(intent)
        if not agent_file:
            return {"error": f"No expert registered for intent: {intent}", "status": "failed"}
            
        agent_path = self.bundle_dir / agent_file
        if not agent_path.exists():
            return {"error": f"Agent script not found at {agent_path}", "status": "failed"}

        try:
            # Dynamic loading of the expert module
            spec = importlib.util.spec_from_file_location(agent_file.replace(".py", ""), str(agent_path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Convention: Many bundle agents have an LLMService or a main entry method
            # We'll check for 'classify_and_generate' or 'handle_query'
            if hasattr(module, "LLMService"):
                service = module.LLMService()
                if hasattr(service, "classify_and_generate"):
                    # Use the standard interface found in hcm_fast_formula_expert.py
                    import asyncio
                    if asyncio.iscoroutinefunction(service.classify_and_generate):
                        result = asyncio.run(service.classify_and_generate(query))
                    else:
                        result = service.classify_and_generate(query)
                    
                    return {
                        "intent": intent,
                        "expert_script": agent_file,
                        "status": "success",
                        "output": result
                    }
            
            return {
                "intent": intent,
                "expert_script": agent_file,
                "status": "partial_success",
                "message": f"Expert {agent_file} loaded but missing standard interface."
            }
            
        except Exception as e:
            return {
                "intent": intent,
                "expert_script": agent_file,
                "status": "error",
                "error": str(e)
            }

    def synthesize(self, router_output: str, expert_output: Dict[str, Any]) -> str:
        """
        Uses the 8B model's routing instructions to wrap the expert's technical 
        output into a professional Oracle Fusion response.
        """
        if expert_output.get("status") != "success":
            return f"I encountered an issue connecting to the {expert_output.get('intent', 'specialist')} expert. Internal Error: {expert_output.get('error', 'Unknown')}"
            
        # Placeholder for 8B synthesis logic
        technical_data = expert_output.get("output", "")
        return f"### [Expert Analysis: {expert_output['intent'].upper()}]\n\n{technical_data}\n\n*This response was synthesized by the 8B Specialist using domain experts.*"

if __name__ == "__main__":
    # Quick Test
    orchestrator = MoAOrchestrator()
    print(json.dumps(orchestrator.get_expert_response("hcm_formula", "How do I calculate accruals?"), indent=2))
