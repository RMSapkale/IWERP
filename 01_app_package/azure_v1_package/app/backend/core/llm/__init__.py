"""
Client for communicating with the local llama.cpp HTTP server.
"""

try:
    from .client import LlamaClient
except ModuleNotFoundError:
    LlamaClient = None

__all__ = ["LlamaClient"]
