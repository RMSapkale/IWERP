import json
import random
import os
import structlog
from typing import List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pathlib import Path

from core.database.models import SFTSample
from core.schemas.sft import SFTSampleSchema

logger = structlog.get_logger(__name__)

class SFTSampleBuilder:
    """
    Builds SFT datasets from curated Postgres samples.
    """
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id

    async def export_dataset(
        self, 
        db: AsyncSession, 
        output_dir: str = "data/sft_export",
        val_split: float = 0.2,
        seed: int = 42
    ) -> Tuple[str, str]:
        """
        Queries samples, validates, splits, and exports to JSONL.
        """
        # 1. Query Samples
        result = await db.execute(
            select(SFTSample).where(SFTSample.tenant_id == self.tenant_id)
        )
        samples = result.scalars().all()
        
        logger.info("sft_export_started", count=len(samples), tenant=self.tenant_id)
        
        if not samples:
            logger.warning("sft_export_empty", tenant=self.tenant_id)
            return "", ""

        # 2. Validate and Convert to Schema
        processed_samples = []
        for s in samples:
            try:
                sample_data = SFTSampleSchema(
                    id=str(s.id),
                    tenant_id=s.tenant_id,
                    module=s.module,
                    task_type=s.task_type,
                    messages=s.messages,
                    difficulty=s.difficulty,
                    source_doc_ids=s.source_doc_ids
                )
                processed_samples.append(sample_data.model_dump())
            except Exception as e:
                logger.error("sft_validation_error", id=str(s.id), error=str(e))

        # 3. Shuffle and Split
        random.seed(seed)
        random.shuffle(processed_samples)
        
        split_idx = int(len(processed_samples) * (1 - val_split))
        train_samples = processed_samples[:split_idx]
        val_samples = processed_samples[split_idx:]
        
        # 4. Export to JSONL
        out_path = Path(output_dir) / self.tenant_id
        out_path.mkdir(parents=True, exist_ok=True)
        
        train_file = out_path / "train.jsonl"
        val_file = out_path / "val.jsonl"
        
        self._write_jsonl(train_file, train_samples)
        self._write_jsonl(val_file, val_samples)
        
        logger.info("sft_export_finished", 
                    train_count=len(train_samples), 
                    val_count=len(val_samples),
                    output_dir=str(out_path))
        
        return str(train_file), str(val_file)

    def _write_jsonl(self, path: Path, data: List[Dict[str, Any]]):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    import asyncio
    import argparse
    from core.database.session import AsyncSessionLocal
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--output", default="data/sft_export")
    args = parser.parse_args()
    
    async def run():
        async with AsyncSessionLocal() as db:
            builder = SFTSampleBuilder(args.tenant)
            await builder.export_dataset(db, output_dir=args.output)
            
    asyncio.run(run())
