import json
import random
import os
import re
import uuid
import structlog
from typing import List, Dict, Any, Tuple, Set
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pathlib import Path
from pydantic import ValidationError

from core.database.models import Chunk, Document, SFTSample
from core.schemas.sft import SFTSampleSchema, SFTMessage
from core.schemas.router import TaskType, FusionModule

logger = structlog.get_logger(__name__)

class SFTSampleBuilderV2:
    """
    Advanced SFT Dataset Builder focused on High Accuracy.
    Includes cleaning, deduplication, and quality scoring.
    """
    
    def __init__(self, tenant_id: str, min_content_length: int = 150):
        self.tenant_id = tenant_id
        self.min_content_length = min_content_length
        self.boilerplate_patterns = [
            r"Navigation Menu.*",
            r"Privacy Policy.*",
            r"Terms of Use.*",
            r"Copyright © \d{4}.*",
            r"All rights reserved.*",
            r"Contact Us.*",
            r"Skip to content.*",
            r"Header\d+.*",
            r"Footer\d+.*",
        ]

    def _clean_content(self, text: str) -> str:
        """Removes boilerplate and redundant whitespace."""
        for pattern in self.boilerplate_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _get_shingles(self, text: str, k: int = 5) -> Set[str]:
        """Generates shingles for near-duplicate detection."""
        text = re.sub(r"[^\w\s]", " ", text.lower())
        words = text.split()
        if len(words) < k:
            return {text}
        return {" ".join(words[i:i+k]) for i in range(len(words)-k+1)}

    def _jaccard_similarity(self, s1: Set[str], s2: Set[str]) -> float:
        """Computes Jaccard similarity between two sets of shingles."""
        if not s1 or not s2:
            return 0.0
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def _score_sample(self, sample: Dict[str, Any]) -> float:
        """
        Quality Scoring Logic:
        - objective clarity (+0.3)
        - module relevance (+0.2)
        - structure (bullet points/numbered lists) (+0.3)
        - length penalty/bonus (up to +0.2)
        """
        score = 0.0
        content = sample["messages"][-1]["content"]
        
        # 1. Structure check
        if bool(re.search(r"^\s*[\-\*\d\.]+\s+", content, re.MULTILINE)):
            score += 0.3
            
        # 2. Objective check (has explicit verbs like 'Click', 'Go to', 'Run')
        if bool(re.search(r"\b(Click|Go to|Select|Run|Execute|Configure|Navigate)\b", content, re.IGNORECASE)):
            score += 0.3
            
        # 3. Length check (sweet spot 300-2000 chars)
        length = len(content)
        if 300 < length < 2000:
            score += 0.2
        elif length > 2000:
            score += 0.1
            
        # 4. Context check (references Fusion terms)
        if "fusion" in content.lower() or "oracle" in content.lower():
            score += 0.2
            
        return score

    async def fetch_and_process(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """Reads from Postgres and applies cleaning/dedupe."""
        # For simplicity in this implementation, we simulate fetching chunks
        # and converting them to instruction samples.
        result = await db.execute(
            select(Chunk).where(Chunk.tenant_id == self.tenant_id)
        )
        chunks = result.scalars().all()
        
        logger.info("processing_chunks", count=len(chunks), tenant=self.tenant_id)
        
        cleaned_samples = []
        seen_shingles = []
        
        for chunk in chunks:
            # 1. Cleaning
            content = self._clean_content(chunk.content)
            if len(content) < self.min_content_length:
                continue
                
            # 2. Deduplication (Near-duplicate check)
            shingles = self._get_shingles(content)
            is_duplicate = False
            for existing in seen_shingles:
                if self._jaccard_similarity(shingles, existing) > 0.8:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue
            
            seen_shingles.append(shingles)
            
            # 3. Create instruction sample (heuristic-based)
            # In a real scenario, this would use a more sophisticated template or LLM-based extraction
            module = FusionModule.SCM # Procurement falls under SCM in the enum
            task_type = TaskType.FUSION_PROC
            
            messages = [
                {"role": "user", "content": f"How do I perform tasks related to {self.tenant_id} in Oracle Fusion?"},
                {"role": "assistant", "content": content}
            ]
            
            sample = {
                "id": str(uuid.uuid4()),
                "tenant_id": self.tenant_id,
                "module": module,
                "task_type": task_type,
                "messages": messages,
                "difficulty": "medium",
                "source_doc_ids": [str(chunk.document_id)]
            }
            
            # 4. Score and Filter
            quality_score = self._score_sample(sample)
            if quality_score < 0.5:
                continue
                
            sample["quality_score"] = quality_score
            cleaned_samples.append(sample)
            
        return cleaned_samples

    async def build_dataset(
        self, 
        db: AsyncSession, 
        output_dir: str = "data/sft_export",
        val_split: float = 0.1
    ):
        samples = await self.fetch_and_process(db)
        
        # Sort by quality
        samples.sort(key=lambda x: x["quality_score"], reverse=True)
        
        # Split
        random.shuffle(samples)
        split_idx = int(len(samples) * (1 - val_split))
        train_data = samples[:split_idx]
        val_data = samples[split_idx:]
        
        tenant_path = Path(output_dir) / self.tenant_id / "sft"
        tenant_path.mkdir(parents=True, exist_ok=True)
        
        # Write files
        self._write_jsonl(tenant_path / "train.jsonl", train_data)
        self._write_jsonl(tenant_path / "val.jsonl", val_data)
        
        # Generate Report
        report = {
            "total_samples": len(samples),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "avg_quality_score": sum(s["quality_score"] for s in samples) / len(samples) if samples else 0,
            "module_coverage": self._get_coverage(samples, "module"),
            "task_coverage": self._get_coverage(samples, "task_type")
        }
        
        with open(tenant_path / "report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        logger.info("dataset_build_complete", report=report)
        return tenant_path

    def _get_coverage(self, samples: List[Dict], key: str) -> Dict[str, int]:
        coverage = {}
        for s in samples:
            val = str(s[key])
            coverage[val] = coverage.get(val, 0) + 1
        return coverage

    def _write_jsonl(self, path: Path, data: List[Dict]):
        with open(path, "w") as f:
            for item in data:
                # Remove quality_score before writing to final JSONL
                clean_item = {k: v for k, v in item.items() if k != "quality_score"}
                f.write(json.dumps(clean_item) + "\n")

if __name__ == "__main__":
    import asyncio
    import argparse
    from core.database.session import AsyncSessionLocal
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", required=True)
    args = parser.parse_args()
    
    async def main():
        async with AsyncSessionLocal() as db:
            builder = SFTSampleBuilderV2(args.tenant)
            await builder.build_dataset(db)
            
    asyncio.run(main())
