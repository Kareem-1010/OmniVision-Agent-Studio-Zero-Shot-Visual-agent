"""
Scene graph aliases and compatibility mappings for geometry and spatial reasoning.
"""

from __future__ import annotations

from typing import List, Optional

from src.memory.scene_graph import (
    Persistent4DSceneGraph,
    SceneGraphEdge,
    SceneGraphNode,
    compute_graph_diff,
)


class PersistentSceneGraph(Persistent4DSceneGraph):
    """Alias and helper wrapper for Persistent4DSceneGraph with spatial convenience methods."""

    def get_neighbors(
        self,
        node_id: str,
        relation_type: Optional[str] = None,
    ) -> List[SceneGraphNode]:
        """Return target nodes connected to node_id via incoming or outgoing edges."""
        neighbors: List[SceneGraphNode] = []
        for edge in self._edges_list:
            if edge.source_id == node_id:
                if relation_type is None or edge.relation_type == relation_type:
                    target_node = self.get_node(edge.target_id)
                    if target_node:
                        neighbors.append(target_node)
            elif edge.target_id == node_id:
                if relation_type is None or edge.relation_type == relation_type:
                    src_node = self.get_node(edge.source_id)
                    if src_node:
                        neighbors.append(src_node)
        return neighbors


__all__ = [
    "PersistentSceneGraph",
    "Persistent4DSceneGraph",
    "SceneGraphNode",
    "SceneGraphEdge",
    "compute_graph_diff",
]
