from typing import Any, Dict, List, Optional

from core.grounding.trusted_registry import get_default_registry


class SchemaIndex:
    """
    Small, deterministic schema index backed by the trusted object registry.
    """

    def __init__(self):
        self.registry = get_default_registry()

    def search(
        self,
        query: str,
        module: Optional[str] = None,
        top_k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        requested_module = module
        if filters:
            module_filter = (
                filters.get("requested_module")
                or filters.get("module_family")
                or filters.get("module")
            )
            if isinstance(module_filter, (list, tuple, set)):
                requested_module = next(iter(module_filter), module)
            elif module_filter:
                requested_module = str(module_filter)
        return self.registry.search(query, module=requested_module, limit=top_k)
