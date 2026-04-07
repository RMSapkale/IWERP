import os
import subprocess
import time
import json
import logging
import torch
from typing import Dict, Any, Optional
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

logger = logging.getLogger(__name__)

class StabilityGovernor(TrainerCallback):
    """
    Mac-only Training Stability Governor.
    Monitors memory pressure and thermal throttling to adjust training parameters.
    """

    def __init__(
        self,
        config_path: str,
        log_path: str = "logs/training_stability.jsonl",
    ):
        with open(config_path, "r") as f:
            import yaml
            self.config = yaml.safe_load(f)
        
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        self.last_step_time = time.time()
        self.step_durations = []
        self.pause_count = 0
        
        # Initial state from config
        self.min_batch_size = self.config.get("min_batch_size", 1)
        self.cooldown_seconds = self.config.get("cooldown_seconds", 30)
        self.mem_threshold_pages = self.config.get("mem_threshold_pages", 50000) # example threshold
        self.spike_factor = self.config.get("spike_factor", 2.0)

    def _get_vm_stat(self) -> Dict[str, int]:
        """Parses vm_stat output."""
        import re
        try:
            output = subprocess.check_output(["vm_stat"], text=True)
            stats = {}
            for line in output.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    # Extract only the leading integer, skip lines like "(page size of N bytes)"
                    m = re.search(r"(\d+)", val.strip())
                    if m:
                        stats[key.strip()] = int(m.group(1))
            return stats
        except Exception as e:
            logger.error(f"Error calling vm_stat: {e}")
            return {}

    def _get_thermal_state(self) -> str:
        """Best effort thermal state check."""
        try:
            output = subprocess.check_output(["pmset", "-g", "therm"], text=True)
            return output.strip()
        except:
            return "Unknown"

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        with open(self.log_path, "a") as f:
            entry = {
                "timestamp": time.time(),
                "event": event_type,
                **data
            }
            f.write(json.dumps(entry) + "\n")

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Check status before each step."""
        stats = self._get_vm_stat()
        free_pages = stats.get("Pages free", 0)
        
        pressure_high = free_pages < self.mem_threshold_pages
        
        # Check for time spike
        now = time.time()
        step_time = now - self.last_step_time
        self.last_step_time = now
        
        spike_detected = False
        if len(self.step_durations) > 5:
            avg_time = sum(self.step_durations[-5:]) / 5
            if step_time > avg_time * self.spike_factor:
                spike_detected = True
        
        if pressure_high or spike_detected:
            self.pause_count += 1
            reason = "memory" if pressure_high else "thermal_spike"
            print(f"\n[StabilityGovernor] Pressure detected ({reason}). Pausing for {self.cooldown_seconds}s...")
            
            self._log_event("pause", {
                "step": state.global_step,
                "reason": reason,
                "free_pages": free_pages,
                "step_time": step_time,
                "pause_duration": self.cooldown_seconds
            })
            
            time.sleep(self.cooldown_seconds)
            
            # Reduce resources
            if args.per_device_train_batch_size > self.min_batch_size:
                args.per_device_train_batch_size -= 1
                args.gradient_accumulation_steps += 1
                print(f"[StabilityGovernor] Reduced batch size to {args.per_device_train_batch_size}, increased grad_accum to {args.gradient_accumulation_steps}")
            
            # Reduce dataloader workers
            if args.dataloader_num_workers > 0:
                args.dataloader_num_workers = 0
                print("[StabilityGovernor] Set dataloader workers to 0")

        self.step_durations.append(step_time)
        if len(self.step_durations) > 100:
            self.step_durations.pop(0)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Check status before evaluation."""
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        stats = self._get_vm_stat()
        free_pages = stats.get("Pages free", 0)
        
        # Stricter threshold for evaluation
        if free_pages < self.mem_threshold_pages:
            print(f"\n[StabilityGovernor] Memory pressure detected BEFORE evaluation. Pausing for {self.cooldown_seconds * 2}s...")
            time.sleep(self.cooldown_seconds * 2)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Log training state and system health."""
        if logs is None:
            logs = {}
        
        stats = self._get_vm_stat()
        health_data = {
            "step": state.global_step,
            "loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "free_pages": stats.get("Pages free"),
            "active_pages": stats.get("Pages active"),
            "thermal": self._get_thermal_state()
        }
        self._log_event("metrics", health_data)
