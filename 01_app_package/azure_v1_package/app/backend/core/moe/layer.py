import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Expert(nn.Module):
    """A simple MLP expert."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MoELayer(nn.Module):
    """
    Sparse Mixture of Experts Layer with Top-2 Routing.
    """
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, d_model)
        Returns (output, aux_loss)
        """
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1]) # (total_tokens, d_model)
        
        # 1. Routing
        router_logits = self.router(x) # (total_tokens, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # 2. Top-K Selection
        top_weights, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        # Normalize weights for the selected experts
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
        # 3. Expert Execution
        # In a real sparse implementation, we would group tokens by expert.
        # For the prototype, we'll do a weighted sum.
        output = torch.zeros_like(x)
        
        # Utilization tracking for Aux Loss
        expert_mask = F.one_hot(top_indices, num_classes=self.num_experts) # (tokens, k, experts)
        utilization = expert_mask.float().mean(dim=(0, 1)) # (experts)
        
        # 4. Loop over experts (inefficient but clear for prototype)
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            token_mask = (top_indices == i).any(dim=-1)
            if token_mask.any():
                # For simplicity in prototype, we just apply to all and mask
                # In production, this would be a specialized kernel.
                expert_output = expert(x[token_mask])
                
                # Assign with weighted contribution
                # tokens x k -> weights
                for k in range(self.top_k):
                    k_mask = (top_indices[:, k] == i)
                    if k_mask.any():
                        output[k_mask] += top_weights[k_mask, k].unsqueeze(-1) * expert(x[k_mask])

        # 5. Load Balancing Loss (Squared CV of utilization)
        # Higher CV means more imbalance.
        cv = utilization.std() / (utilization.mean() + 1e-6)
        aux_loss = cv.pow(2)
        
        return output.view(*orig_shape), aux_loss
