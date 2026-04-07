import torch
import json
import os
import structlog
from typing import List, Dict, Any, Optional
from core.rag.engine import RAGEngine
from core.schemas.api import ChatRequest, Message, Role

logger = structlog.get_logger(__name__)

class PPOUtils:
    """
    Helpers for PPO training: safety filters and evaluation-based rollback.
    """
    @staticmethod
    def apply_safety_filters(text: str, stop_sequences: List[str] = []) -> str:
        """
        Ensures model doesn't generate forbidden patterns or system leaks.
        """
        for seq in stop_sequences:
            if seq in text:
                text = text.split(seq)[0]
        
        # Avoid leaked system prompt signatures
        if "You are the Oracle Fusion Navigation Assistant" in text:
             logger.warning("safety_filter_triggered", reason="system_prompt_leak")
             return "Filtered: Potential system prompt leak detected."
        
        return text

    @staticmethod
    async def run_golden_eval(
        model, 
        tokenizer, 
        golden_set_path: str,
        tenant_id: str
    ) -> float:
        """
        Runs a quick evaluation against a golden set to detect drift.
        Returns a 'quality score' (e.g. mean reward or expert metric).
        """
        if not os.path.exists(golden_set_path):
            return 0.0
            
        with open(golden_set_path, "r") as f:
            golden_data = json.load(f)

        # Simplified logic: just check if it can generate coherent responses
        # In production, this would call the RAGEngine or Reward Model.
        return 1.0 # Placeholder for actual metric

class RollbackManager:
    """
    Tracks performance and manages checkpoint rollbacks.
    """
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.history = []

    def log_eval(self, step: int, score: float):
        self.history.append({"step": step, "score": score})
        
    def should_rollback(self, tolerance: float = 0.2) -> bool:
        """
        Returns True if current score is significantly worse than best.
        """
        if len(self.history) < 2:
            return False
            
        best_score = max(h["score"] for h in self.history[:-1])
        current_score = self.history[-1]["score"]
        
        if current_score < best_score * (1 - tolerance):
            logger.error("ppo_performance_regression", current=current_score, best=best_score)
            return True
        return False
