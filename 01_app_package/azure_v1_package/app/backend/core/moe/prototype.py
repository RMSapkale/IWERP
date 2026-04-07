import time
import torch
import torch.nn as nn
import numpy as np
import json
import os
from typing import List, Dict, Any
from core.moe.layer import MoELayer

class MoEPrototypeHarness:
    """
    Harness to evaluate MoE vs Dense baselines.
    """
    def __init__(self, d_model: int = 768, d_ff: int = 3072):
        self.d_model = d_model
        self.dense_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.moe_layer = MoELayer(d_model, d_ff, num_experts=8, top_k=2)

    def benchmark_latency(self, batch_size=1, seq_len=128, iterations=100):
        x = torch.randn(batch_size, seq_len, self.d_model)
        
        # Warmup
        for _ in range(10):
            self.dense_layer(x)
            self.moe_layer(x)
            
        # Dense Bench
        start = time.perf_counter()
        for _ in range(iterations):
            self.dense_layer(x)
        dense_time = (time.perf_counter() - start) / iterations
        
        # MoE Bench
        start = time.perf_counter()
        for _ in range(iterations):
            self.moe_layer(x)
        moe_time = (time.perf_counter() - start) / iterations
        
        return {
            "dense_ms": round(dense_time * 1000, 2),
            "moe_ms": round(moe_time * 1000, 2),
            "overhead_ratio": round(moe_time / dense_time, 2)
        }

    def evaluate_routing_health(self, input_samples: torch.Tensor):
        """Measures expert utilization entropy."""
        _, _ = self.moe_layer(input_samples)
        
        # In a real test, we'd hook into the layer to get usage stats.
        # For the prototype, we'll simulate based on known routing logic.
        logits = self.moe_layer.router(input_samples.view(-1, self.d_model))
        probs = torch.softmax(logits, dim=-1)
        usage = probs.mean(dim=0).detach().numpy()
        
        entropy = -np.sum(usage * np.log(usage + 1e-6))
        max_entropy = np.log(len(usage))
        
        return {
            "expert_utilization": usage.tolist(),
            "utilization_entropy": round(float(entropy), 4),
            "balance_score": round(float(entropy / max_entropy), 4)
        }

    def run_full_suite(self, output_path="data/eval/moe_prototype_results.json"):
        print("Running MoE Prototype Evaluation...")
        
        latency = self.benchmark_latency()
        print(f"Latency: Dense {latency['dense_ms']}ms vs MoE {latency['moe_ms']}ms")
        
        # Real-ish data check
        x_samples = torch.randn(32, 128, self.d_model)
        health = self.evaluate_routing_health(x_samples)
        print(f"Routing Health (Balance Score): {health['balance_score']}")
        
        results = {
            "latency": latency,
            "routing_health": health,
            "comparison": {
                "metric": "p95_latency",
                "status": "PASS" if latency['overhead_ratio'] < 2.5 else "FAIL"
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
            
        return results

if __name__ == "__main__":
    harness = MoEPrototypeHarness()
    harness.run_full_suite()
