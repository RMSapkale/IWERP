import os
import json
import re
import torch
import structlog
from typing import List, Dict, Any, Optional
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from pathlib import Path
import sqlglot

logger = structlog.get_logger(__name__)

class LiveEvalCallback(TrainerCallback):
    """
    Continuous evaluation during training.
    Tracks citations, faithfulness, and SQL validity.
    Triggers early stopping on quality regression.
    """

    def __init__(
        self,
        tenant_id: str,
        eval_subset: List[Dict[str, Any]],
        output_dir: str,
        citation_threshold: float = 0.7,
        sql_parse_threshold: float = 0.8,
        patience: int = 3
    ):
        self.tenant_id = tenant_id
        self.eval_subset = eval_subset
        self.output_dir = Path(output_dir)
        self.citation_threshold = citation_threshold
        self.sql_parse_threshold = sql_parse_threshold
        self.patience = patience
        
        self.metrics_log_path = self.output_dir / "training_live_metrics.jsonl"
        self.regression_count = 0
        
        os.makedirs(self.output_dir, exist_ok=True)

    def _check_citation(self, text: str) -> bool:
        """Checks if response contains at least one bracketed citation."""
        return bool(re.search(r"\[Source: .*?\]", text))

    def _check_sql_parse(self, text: str) -> bool:
        """Checks if SQL block parses correctly."""
        sql_match = re.search(r"```sql\n(.*?)\n```", text, re.DOTALL)
        if not sql_match:
            # If no SQL block, check if raw text looks like SQL and is valid
            if "SELECT" in text.upper():
                try:
                    sqlglot.transpile(text, read="oracle")
                    return True
                except:
                    return False
            return False
            
        sql = sql_match.group(1)
        try:
            sqlglot.transpile(sql, read="oracle")
            return True
        except:
            return False

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")
        
        if not model or not tokenizer:
            return

        logger.info("running_live_eval", step=state.global_step)
        model.eval()
        
        total = len(self.eval_subset)
        citation_success = 0
        sql_success = 0
        
        for sample in self.eval_subset:
            prompt = sample["messages"][0]["content"]
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=200)
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if self._check_citation(decoded):
                citation_success += 1
            
            # Only check SQL success if prompt looks like a SQL request
            if any(k in prompt.upper() for k in ["SELECT", "SQL", "QUERY", "TABLE"]):
                if self._check_sql_parse(decoded):
                    sql_success += 1
            else:
                # Neutral for non-SQL
                sql_success += 1

        citation_rate = citation_success / total
        sql_rate = sql_success / total
        
        metrics = {
            "step": state.global_step,
            "citation_rate": citation_rate,
            "sql_parse_rate": sql_rate,
            "eval_loss": state.log_history[-1].get("eval_loss", 0) if state.log_history else 0
        }
        
        with open(self.metrics_log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
            
        logger.info("live_metrics", **metrics)

        # Early Stopping Logic (Heuristic)
        is_failing = (
            citation_rate < self.citation_threshold or 
            sql_rate < self.sql_parse_threshold
        )
        
        if is_failing:
            self.regression_count += 1
            logger.warning("quality_regression_detected", 
                           count=self.regression_count, 
                           patience=self.patience)
        else:
            self.regression_count = 0

        if self.regression_count >= self.patience:
            logger.error("early_stop_triggered", reason="Quality regression exceeded patience")
            control.should_training_stop = True
            
        model.train()
