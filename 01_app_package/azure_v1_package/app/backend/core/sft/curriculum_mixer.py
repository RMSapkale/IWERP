import json
import random
import os
import structlog
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from pathlib import Path
from datasets import Dataset, load_dataset, concatenate_datasets

logger = structlog.get_logger(__name__)

class StageConfig(BaseModel):
    name: str
    step_threshold: int
    ratios: Dict[str, float]  # e.g. {"general": 0.6, "sql": 0.3, "hardneg": 0.1}

class CurriculumConfig(BaseModel):
    total_steps: int
    stages: List[StageConfig]

class CurriculumMixer:
    """
    Dynamically mixes multiple datasets based on current training step.
    Implements stage-based curriculum learning.
    """

    def __init__(self, config: CurriculumConfig, dataset_paths: Dict[str, str], seed: int = 42):
        self.config = config
        self.dataset_paths = dataset_paths
        self.seed = seed
        self._datasets = {}
        self._load_datasets()

    def _load_datasets(self):
        """Loads all involved datasets into memory or streaming mode."""
        for key, path in self.dataset_paths.items():
            if os.path.exists(path):
                logger.info("loading_dataset", key=key, path=path)
                self._datasets[key] = load_dataset("json", data_files=path, split="train")
            else:
                logger.warning("dataset_not_found", key=key, path=path)
                # Create empty dataset if missing to avoid crashes
                self._datasets[key] = Dataset.from_dict({"messages": []})

    def get_stage_for_step(self, global_step: int) -> StageConfig:
        """Returns the appropriate stage config for the given global step."""
        for stage in self.config.stages:
            if global_step <= stage.step_threshold:
                return stage
        return self.config.stages[-1]

    def sample_batch(self, batch_size: int, global_step: int) -> Dataset:
        """
        Samples a mixed batch based on stage ratios.
        Note: This is a simplified version for demonstration.
        In real Trainer integration, we might use a custom Sampler or 
        re-shuffle the concat dataset periodically.
        """
        stage = self.get_stage_for_step(global_step)
        batch_samples = []
        
        for key, ratio in stage.ratios.items():
            num_samples = int(batch_size * ratio)
            if num_samples > 0 and key in self._datasets and len(self._datasets[key]) > 0:
                indices = [random.randint(0, len(self._datasets[key]) - 1) for _ in range(num_samples)]
                batch_samples.append(self._datasets[key].select(indices))
                
        # Fill remaining if any due to rounding
        remaining = batch_size - sum(int(batch_size * r) for r in stage.ratios.values())
        if remaining > 0 and len(self._datasets["general"]) > 0:
            indices = [random.randint(0, len(self._datasets["general"]) - 1) for _ in range(remaining)]
            batch_samples.append(self._datasets["general"].select(indices))
            
        return concatenate_datasets(batch_samples).shuffle(seed=self.seed + global_step)

    def get_full_mixed_dataset(self) -> Dataset:
        """
        Returns a roughly mixed full dataset for standard Trainer use.
        Note: True curriculum requires per-step sampling, but standard Trainer 
        often prefers a static Dataset object.
        """
        # For simple integration, we return a balanced mix based on Stage B (middle ground)
        stage_b = self.config.stages[1] if len(self.config.stages) > 1 else self.config.stages[0]
        logger.info("creating_static_mix", stage=stage_b.name)
        
        all_ds = []
        for key, ratio in stage_b.ratios.items():
            if key in self._datasets and len(self._datasets[key]) > 0:
                all_ds.append(self._datasets[key])
        
        return concatenate_datasets(all_ds).shuffle(seed=self.seed)

def get_default_curriculum(total_steps: int) -> CurriculumConfig:
    """Returns the user-requested A/B/C curriculum."""
    return CurriculumConfig(
        total_steps=total_steps,
        stages=[
            StageConfig(
                name="Stage A: Foundation",
                step_threshold=int(total_steps * 0.25),
                ratios={"general": 0.6, "sql": 0.3, "hardneg": 0.1}
            ),
            StageConfig(
                name="Stage B: Mastery",
                step_threshold=int(total_steps * 0.75),
                ratios={"general": 0.45, "sql": 0.45, "hardneg": 0.1}
            ),
            StageConfig(
                name="Stage C: Specialization",
                step_threshold=total_steps,
                ratios={"general": 0.35, "sql": 0.55, "hardneg": 0.1}
            )
        ]
    )

if __name__ == "__main__":
    # Test logic
    paths = {
        "general": "data/demo/sft/train.jsonl",
        "sql": "data/demo/sft_sql/train.jsonl",
        "hardneg": "data/demo/sft_hardneg/train.jsonl"
    }
    config = get_default_curriculum(1000)
    mixer = CurriculumMixer(config, paths)
    
    stage = mixer.get_stage_for_step(100)
    print(f"Step 100 Stage: {stage.name} - Ratios: {stage.ratios}")
    
    stage = mixer.get_stage_for_step(500)
    print(f"Step 500 Stage: {stage.name} - Ratios: {stage.ratios}")
    
    stage = mixer.get_stage_for_step(900)
    print(f"Step 900 Stage: {stage.name} - Ratios: {stage.ratios}")
