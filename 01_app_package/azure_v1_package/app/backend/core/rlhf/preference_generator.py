import asyncio
import json
import os
import structlog
from typing import List, Dict, Any
from core.database.session import AsyncSessionLocal
from core.rag.engine import RAGEngine
from core.schemas.api import ChatRequest, Message, Role

logger = structlog.get_logger(__name__)

class PreferenceGenerator:
    """
    Generates multiple candidate answers for RLHF preference data collection.
    """
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.rag = RAGEngine()

    async def generate_candidates(
        self, 
        questions: List[str], 
        temps: List[float] = [0.2, 0.7, 1.0],
        output_path: str = "data/rlhf/raw_candidates.json"
    ):
        results = []
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        async with AsyncSessionLocal() as db:
            for q_idx, query in enumerate(questions):
                logger.info("generating_candidates", query_idx=q_idx, query=query[:50])
                candidates = []
                
                for t in temps:
                    req = ChatRequest(
                        messages=[Message(role=Role.USER, content=query)],
                        temperature=t
                    )
                    # Run RAG for each temperature
                    resp = await self.rag.chat(db, self.tenant_id, req)
                    
                    candidates.append({
                        "text": resp.choices[0]["message"]["content"],
                        "temp": t,
                        "latency": resp.timings.get("total_e2e", 0)
                    })

                results.append({
                    "id": f"q_{q_idx}",
                    "query": query,
                    "candidates": candidates
                })

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info("candidates_saved", path=output_path, count=len(results))
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--input", required=True, help="Path to JSON questions file")
    parser.add_argument("--output", default="data/rlhf/raw_candidates.json")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)
        # Extract questions from list of strings or dicts
        questions = [item["question"] if isinstance(item, dict) else item for item in data]

    generator = PreferenceGenerator(args.tenant)
    asyncio.run(generator.generate_candidates(questions, output_path=args.output))
