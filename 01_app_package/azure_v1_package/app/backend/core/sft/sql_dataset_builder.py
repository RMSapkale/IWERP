import json
import random
import os
import uuid
import structlog
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
import sqlglot
from pathlib import Path

from core.schemas.sft import SFTSampleSchema, SFTMessage
from core.schemas.router import TaskType, FusionModule
from core.schemas.api import Role

logger = structlog.get_logger(__name__)

class SQLPlan(BaseModel):
    query_intent: str
    target_tables: List[str]
    join_logic: str
    complexity_level: str

class SQLPart(BaseModel):
    part_id: int
    depends_on: List[int] = []
    sql_content: str
    explanation: str

class SQLCompilerBuilder:
    """
    Dedicated SQL Dataset Builder for 'SQL Compiler Mode'.
    Generates multi-turn samples with PLAN_JSON and SQL_PART blocks.
    """
    
    BUCKETS = [50, 200, 500, 1000, 2000, 3000]

    def __init__(self, tenant_id: str, max_part_tokens: int = 400):
        self.tenant_id = tenant_id
        self.max_part_tokens = max_part_tokens

    def _validate_sql(self, sql_part: str) -> bool:
        """Validates SQL syntax using sqlglot."""
        try:
            # We use 'oracle' dialect as default for Fusion
            sqlglot.transpile(sql_part, read="oracle")
            return True
        except Exception:
            return False

    def _chunk_sql(self, full_sql: str) -> List[str]:
        """
        Naive chunking by token budget (simulated by chars for now).
        In a real scenario, this would use a proper tokenizer.
        """
        # Roughly 4 chars per token
        char_limit = self.max_part_tokens * 4
        parts = []
        lines = full_sql.splitlines()
        current_part = []
        current_len = 0
        
        for line in lines:
            if current_len + len(line) > char_limit and current_part:
                parts.append("\n".join(current_part))
                current_part = []
                current_len = 0
            current_part.append(line)
            current_len += len(line)
            
        if current_part:
            parts.append("\n".join(current_part))
        return parts

    def create_sample(self, request: str, full_sql: str, module: FusionModule) -> Dict[str, Any]:
        """Creates a multi-turn SFT sample."""
        
        # 1. Create Plan
        # Heuristic/Mock extraction for demo purposes
        plan = SQLPlan(
            query_intent=request,
            target_tables=["PO_HEADERS_ALL", "PO_LINES_ALL"],
            join_logic="Inner join on header_id",
            complexity_level="medium"
        )
        
        messages = [
            {"role": Role.USER, "content": request},
            {"role": Role.ASSISTANT, "content": f"PLAN_JSON: {plan.model_dump_json()}"}
        ]
        
        # 2. Chunk SQL into Parts
        sql_chunks = self._chunk_sql(full_sql)
        valid_count = 0
        
        for i, chunk in enumerate(sql_chunks):
            # Ensure stable aliases and comments
            commented_chunk = f"-- Part {i+1} of {len(sql_chunks)}\n{chunk}"
            
            # Register validation
            if self._validate_sql(chunk):
                valid_count += 1
            
            part = SQLPart(
                part_id=i + 1,
                depends_on=[i] if i > 0 else [],
                sql_content=commented_chunk,
                explanation=f"Generating logical block {i+1}."
            )
            messages.append({"role": Role.ASSISTANT, "content": f"SQL_PART: {part.model_dump_json()}"})
            
        return {
            "id": str(uuid.uuid4()),
            "tenant_id": self.tenant_id,
            "module": module,
            "task_type": TaskType.FUSION_PROC, # Default to Procurement
            "messages": messages,
            "num_parts": len(sql_chunks),
            "total_lines": len(full_sql.splitlines()),
            "parse_success": valid_count == len(sql_chunks)
        }

    def build_dataset(self, samples_input: List[Tuple[str, str, FusionModule]], output_dir: str = "data/sft_sql"):
        """Builds, bucketizes, and exports dataset."""
        processed_samples = []
        for req, sql, mod in samples_input:
            sample = self.create_sample(req, sql, mod)
            if sample:
                processed_samples.append(sample)
                
        # 1. Bucketize
        bucketed_data = {b: [] for b in self.BUCKETS}
        for s in processed_samples:
            lines = s["total_lines"]
            for b in self.BUCKETS:
                if lines <= b:
                    bucketed_data[b].append(s)
                    break
        
        # 2. Curriculum Sorting (Smallest buckets first)
        final_ordered = []
        for b in self.BUCKETS:
            final_ordered.extend(bucketed_data[b])
            
        # 3. Export
        tenant_path = Path(output_dir) / self.tenant_id
        tenant_path.mkdir(parents=True, exist_ok=True)
        
        # Split 90/10
        random.shuffle(final_ordered)
        split_idx = int(len(final_ordered) * 0.9)
        train_samples = final_ordered[:split_idx]
        val_samples = final_ordered[split_idx:]
        
        self._write_jsonl(tenant_path / "train.jsonl", train_samples)
        self._write_jsonl(tenant_path / "val.jsonl", val_samples)
        
        # 4. Report
        report = {
            "total_samples": len(processed_samples),
            "parse_success_rate": sum(1 for s in processed_samples if s["parse_success"]) / len(processed_samples) if processed_samples else 0,
            "avg_parts_per_query": sum(s["num_parts"] for s in processed_samples) / len(processed_samples) if processed_samples else 0,
            "bucket_distribution": {str(b): len(bucketed_data[b]) for b in self.BUCKETS}
        }
        
        with open(tenant_path / "report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        logger.info("sql_sft_build_finished", report=report)
        return tenant_path

    def _write_jsonl(self, path: Path, data: List[Dict]):
        with open(path, "w") as f:
            for item in data:
                # Remove meta fields before final SFT export
                clean_item = {k: v for k, v in item.items() if k not in ["num_parts", "total_lines", "parse_success"]}
                f.write(json.dumps(clean_item) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", required=True)
    args = parser.parse_args()
    
    # Mock data for builder run
    builder = SQLCompilerBuilder(args.tenant)
    mock_samples = [
        ("Get all open purchase orders", "SELECT * FROM PO_HEADERS_ALL WHERE STATUS = 'OPEN';", FusionModule.SCM),
        ("List lines for PO 123", "SELECT * FROM PO_LINES_ALL WHERE HEADER_ID = 123 ORDER BY LINE_NUM;", FusionModule.SCM)
    ]
    builder.build_dataset(mock_samples)
