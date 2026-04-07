import os
from typing import Dict, Any

# Mac mini operational defaults (Apple Silicon optimized)

LLM_CONFIG = {
    "inference_backend": os.getenv("IWFUSION_INFERENCE_BACKEND", "mlx"),  # "mlx" or "llama_cpp"
    "model_path": os.getenv(
        "IWFUSION_MODEL_PATH",
        "/Users/integrationwings/.cache/huggingface/hub/models--mlx-community--Meta-Llama-3-8B-Instruct-4bit/snapshots/c38b3b1f03cce0ce0ccd235e5c97b0d3d255e651",
    ),
    "adapter_path": os.getenv(
        "IWFUSION_ADAPTER_PATH",
        "/Users/integrationwings/Desktop/LLM_Wrap/iwerp-azure-deploy/models/checkpoint-177849/mlx_adapter",
    ),  # IWFUSION-SLM-V1 Master Adapter
    "model_name": os.getenv("IWFUSION_MODEL_NAME", "IWFUSION-SLM-V1"),  # Global Production Identity
    "ctx_size": int(os.getenv("IWFUSION_CTX_SIZE", "8192")),  # Expanded for Infini-attention
    "threads": int(os.getenv("IWFUSION_THREADS", "8")),
    "temperature": float(os.getenv("IWFUSION_TEMPERATURE", "0.0")),  # Greedy for audit precision
    "top_p": float(os.getenv("IWFUSION_TOP_P", "0.9")),
    "use_reranker": os.getenv("IWFUSION_USE_RERANKER", "false").lower() == "true",
    "use_embeddings": os.getenv("IWFUSION_USE_EMBEDDINGS", "true").lower() == "true",
    "max_new_tokens": int(os.getenv("IWFUSION_MAX_NEW_TOKENS", "1024")),
    "repeat_penalty": float(os.getenv("IWFUSION_REPEAT_PENALTY", "1.15")),
}

RETRIEVAL_CONFIG = {
    "fts_top": 40,
    "vec_top": 40,
    "rerank_top": 15,
    "final_chunks": 8
}

POLICY_CONFIG = {
    "REQUIRE_CITATIONS": False,
    "REFUSE_ON_INJECTION": True
}
