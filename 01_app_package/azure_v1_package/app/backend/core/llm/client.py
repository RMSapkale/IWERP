import httpx
import json
from typing import AsyncGenerator, Iterator
from core.schemas.config import LlmSettings
from core.schemas.api import ChatRequest, Message

class LlamaClient:
    """
    Client for interacting with a local llama.cpp HTTP server conforming to the OAI spec.
    """
    def __init__(self, settings: LlmSettings):
        self.base_url = settings.base_url
        self.model = settings.model_name
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)

    async def chat_completion(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """
        Sends a chat completion request to the llama.cpp server.
        Yields chunks of the response if stream=True, else yields the entire response at once.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
            "temperature": request.temperature if request.temperature is not None else self.temperature,
            "max_tokens": request.max_tokens if request.max_tokens is not None else self.max_tokens,
            "stream": request.stream
        }

        if request.stream:
            async with self._client.stream("POST", "/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            # Parse the JSON but yield the delta string directly for simplicity
                            # In a real app, you might want a proper SSE streaming model
                            chunk = json.loads(data_str)
                            if chunk["choices"][0]["delta"].get("content"):
                                yield chunk["choices"][0]["delta"]["content"]
                        except json.JSONDecodeError:
                            continue
        else:
            response = await self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            yield data["choices"][0]["message"]["content"]
