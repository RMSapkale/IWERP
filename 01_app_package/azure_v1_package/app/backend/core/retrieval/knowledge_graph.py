from typing import Dict, Set

import structlog

from core.grounding.trusted_registry import get_default_registry

logger = structlog.get_logger(__name__)


class TableJoinGraph:
    """
    Graph wrapper backed by the trusted object registry.
    """

    def __init__(self):
        self.registry = get_default_registry()

    def add_relationship(self, table_u: str, table_v: str) -> None:
        logger.warning("knowledge_graph_add_relationship_ignored", table_u=table_u, table_v=table_v)

    def get_neighbors(self, table_name: str, depth: int = 1) -> Set[str]:
        if depth <= 0:
            return set()

        table_name = table_name.upper()
        visited = {table_name}
        frontier = {table_name}

        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.registry.get_related_objects(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            if not next_frontier:
                break
            frontier = next_frontier

        visited.discard(table_name)
        return visited


fusion_graph = TableJoinGraph()
