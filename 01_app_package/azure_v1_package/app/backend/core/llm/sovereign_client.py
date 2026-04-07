import os
import json
import structlog
import httpx
import hashlib
from typing import Dict, Any, List, Union, Optional

logger = structlog.get_logger(__name__)

class SovereignLLMClient:
    """
    A strictly localized LLM client for IWERP SLM (Titan V2).
    Adapted for the iwerp-prod environment with Redis caching.
    """
    def __init__(self):
        # Default to local SLM server base URL (e.g. LM Studio or local vLLM)
        self.base_url = os.getenv("LOCAL_SLM_BASE_URL", "http://localhost:8080/v1")
        self.chat_url = f"{self.base_url}/chat/completions"
        self.embedding_url = f"{self.base_url}/embeddings"
        
        # Redis Caching Setup
        self.redis_url = os.getenv("REDIS_URL")
        self.redis = None
        if self.redis_url:
            try:
                import redis
                self.redis = redis.from_url(self.redis_url, decode_responses=True)
                logger.info("connected_to_redis", url=self.redis_url)
            except Exception as e:
                logger.warning("redis_connection_failed", error=str(e))

        logger.info("sovereign_client_initialized", base_url=self.base_url)

    def _get_cache_key(self, payload: Dict[str, Any]) -> str:
        """Generates a stable cache key from the request payload."""
        payload_str = json.dumps(payload, sort_keys=True)
        return f"slm_cache:{hashlib.md5(payload_str.encode()).hexdigest()}"

    async def chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Routes the chat request exclusively to the local SLM, with Redis caching.
        """
        cache_key = None
        if self.redis is not None:
            cache_key = self._get_cache_key(payload)
            cached_resp = self.redis.get(cache_key)
            if cached_resp:
                logger.info("cache_hit", service="slm")
                return json.loads(cached_resp)

        logger.info("routing_chat_to_local_slm", url=self.chat_url)
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                self.chat_url,
                headers={"Content-Type": "application/json"},
                json=payload
            )
            resp.raise_for_status()
            result = resp.json()
            
            if self.redis is not None and cache_key is not None:
                # Cache for 24 hours
                self.redis.setex(cache_key, 86400, json.dumps(result))
                
            return result

    async def embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generates embeddings using the local SLM model.
        """
        logger.info("routing_embeddings_to_local_slm", url=self.embedding_url)
        if isinstance(texts, str):
            texts = [texts]
            
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                self.embedding_url,
                headers={"Content-Type": "application/json"},
                json={"input": texts, "model": "titan-v2-embed"}
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data.get("data", [])]
