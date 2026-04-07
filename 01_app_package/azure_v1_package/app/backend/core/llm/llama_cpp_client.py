import asyncio
import json
import os
import time
import uuid
from typing import List, Dict, Any, AsyncGenerator, Optional
import httpx
import structlog
from core.schemas.api import ChatRequest, Message

logger = structlog.get_logger(__name__)

class LlamaCppClient:
    """
    Robust HTTP client for llama-server (OpenAI-compatible /v1/chat/completions).
    Features:
    - Non-streaming and Streaming support
    - Exponential backoff retries
    - Latency tracking and structured logging
    """
    def __init__(
        self, 
        base_url: str = "http://localhost:8080", 
        timeout: float = 300.0,
        max_retries: int = 3
    ):
        self.base_url = os.getenv("LOCAL_SLM_BASE_URL", os.getenv("LLAMA_CPP_BASE_URL", base_url)).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    async def chat(
        self, 
        request: ChatRequest, 
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Non-streaming chat completion."""
        trace_id = trace_id or str(uuid.uuid4())
        url = f"{self.base_url}/v1/chat/completions"
        payload = request.model_dump(exclude_none=True)
        
        start_time = time.time()
        
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    
                    latency = time.time() - start_time
                    result = response.json()
                    
                    logger.info(
                        "llm_request_success",
                        trace_id=trace_id,
                        latency_ms=int(latency * 1000),
                        attempt=attempt + 1,
                        stream=False
                    )
                    return result

            except (httpx.HTTPError, httpx.TimeoutException) as e:
                if attempt == self.max_retries:
                    logger.error(
                        "llm_request_failed_final",
                        trace_id=trace_id,
                        error=str(e),
                        attempt=attempt + 1
                    )
                    raise
                
                wait_time = 2 ** attempt # Exponential backoff
                logger.warning(
                    "llm_request_retry",
                    trace_id=trace_id,
                    error=str(e),
                    attempt=attempt + 1,
                    next_retry_sec=wait_time
                )
                await asyncio.sleep(wait_time)

    async def chat_stream(
        self, 
        request: ChatRequest, 
        trace_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming chat completion."""
        trace_id = trace_id or str(uuid.uuid4())
        url = f"{self.base_url}/v1/chat/completions"
        request.stream = True
        payload = request.model_dump(exclude_none=True)
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if not line or line == "data: [DONE]":
                            continue
                        
                        if line.startswith("data: "):
                            raw_data = line[6:]
                            try:
                                yield json.loads(raw_data)
                            except json.JSONDecodeError:
                                continue

            latency = time.time() - start_time
            logger.info(
                "llm_stream_finished",
                trace_id=trace_id,
                latency_ms=int(latency * 1000)
            )

        except Exception as e:
            logger.error(
                "llm_stream_failed",
                trace_id=trace_id,
                error=str(e)
            )
            raise
