"""
Task routing and templating for query processing.
"""
from .router import intent_router
from .templates import get_prompt_template

__all__ = ["intent_router", "get_prompt_template"]
