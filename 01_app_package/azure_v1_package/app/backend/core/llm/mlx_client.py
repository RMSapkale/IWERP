import mlx_lm
import structlog
from typing import List, Dict, Any, Optional
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from core.schemas.api import ChatRequest

logger = structlog.get_logger(__name__)

class MLXModelLoader:
    _instance = None
    _model = None
    _tokenizer = None

    @classmethod
    def get_instance(cls, model_path: str, adapter_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = cls()
            logger.info("loading_mlx_model_with_adapter", path=model_path, adapter=adapter_path)
            # Load the base model with the adapter applied
            cls._model, cls._tokenizer = mlx_lm.load(model_path, adapter_path=adapter_path)
        return cls._model, cls._tokenizer

class MLXClient:
    """
    Local MLX client that performs inference directly in-process.
    """
    def __init__(self, model_path: Optional[str] = None):
        from core.config.operational import LLM_CONFIG
        self.model_path = model_path or LLM_CONFIG.get("model_path", "mlx-community/Meta-Llama-3-8B-Instruct-4bit")
        self.adapter_path = LLM_CONFIG.get("adapter_path")
        self.model_name = LLM_CONFIG.get("model_name", "IWFUSION-SLM-V1")
        self.model = None
        self.tokenizer = None

    def _ensure_model(self):
        if self.model is None:
            self.model, self.tokenizer = MLXModelLoader.get_instance(self.model_path, self.adapter_path)

    async def chat(self, request: ChatRequest) -> Dict[str, Any]:
        self._ensure_model()
        
        # Convert Pydantic request to MLX format
        messages = [{"role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role), "content": msg.content} for msg in request.messages]
        # VITAL: tokenize=True enforces returning the strict integer model tokens instead of strings. 
        # This prevents MLX from encoding `<|start_header_id|>` as literal text characters.
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        print(f"====== IWFUSION-SLM-V1 DENSITY: {len(prompt)} RAW TOKENS ======")
        
        # Stable parameters as identified in testing
        # Enforcing greedy decoding (temp=0.0) to match the flawless CLI behavior
        sampler = make_sampler(temp=0.0)
        
        # Pass through the repetition penalty if provided in config or request
        repetition_penalty = request.repeat_penalty or 1.1 # Default mild penalty
        
        # New MLX-LM versions use logits_processors for repetition_penalty
        logits_processors = make_logits_processors(logit_bias=None, repetition_penalty=repetition_penalty)
        
        # Use stream_generate for O(N) suffix-based stopping and stable memory building
        full_response = ""
        stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "User:", "Assistant:"]
        
        token_count = 0
        for response in mlx_lm.stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=request.max_tokens or 1024,
            sampler=sampler,
            logits_processors=logits_processors
        ):
            full_response += response.text
            token_count += 1
            
            # Periodic heartbeat for log observability
            if token_count % 50 == 0:
                logger.debug("generation_progress", tokens=token_count, text_peek=response.text)
                
            # Efficient O(N) suffix check: only look at the end of the string
            # We check the last 20 characters to capture the longest stop-token safely
            tail = full_response[-20:]
            if any(stop in tail for stop in stop_tokens):
                # Final cleanup
                for stop in stop_tokens:
                    if stop in full_response:
                        full_response = full_response.split(stop)[0]
                break
                
        response_text = full_response.strip()
        
        return {
            "id": "mlx-local",
            "object": "chat.completion",
            "created": 0,
            "model": self.model_path,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
