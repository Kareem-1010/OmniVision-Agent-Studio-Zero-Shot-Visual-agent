"""
Persistent 4D (3D + Time) Scene Graph with Query DSL and Graph-Diff Event Detection.
Represents physical instances as temporal 3D nodes and time-varying spatial/functional relations as edges.
Provides NetworkX integration, JSON serialization, a declarative query interpreter, and graph diffing.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger("OmniVision.Memory.SceneGraph")


class SceneGraphNode(BaseModel):
    """Temporal 3D instance node in the 4D scene graph."""
    node_id: str
    label: str
    track_id: Optional[int] = None
    time_span: Tuple[int, int] = (0, 0)  # (start_frame, end_frame)
    bbox_2d: Optional[List[float]] = None
    centroid_3d: Optional[List[float]] = None  # [X, Y, Z] in metres
    dimensions_3d: Optional[List[float]] = None  # [w, h, d] in metres
    attributes: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    mask_rle: Optional[str] = None


class SceneGraphEdge(BaseModel):
    """Directed temporal edge capturing a spatial or semantic relation between instances."""
    source_id: str
    target_id: str
    relation_type: str  # e.g., 'left_of', 'supports', 'inside', 'holding', 'touching'
    time_span: Tuple[int, int] = (0, 0)
    metric_distance_meters: Optional[float] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    attributes: Dict[str, Any] = Field(default_factory=dict)


class GraphDiffEvent(BaseModel):
    """Discrete perceptual or physical event detected via graph-diff across timestamps."""
    event_type: str  # 'INSTANCE_APPEARED', 'INSTANCE_DISAPPEARED', 'RELATION_FORMED', 'RELATION_BROKEN', 'ATTRIBUTE_CHANGED'
    subject_id: str
    object_id: Optional[str] = None
    frame_index: int
    details: Dict[str, Any] = Field(default_factory=dict)


class Persistent4DSceneGraph:
    """
    Persistent 4D Scene Graph holding spatiotemporal instance nodes and relational edges.
    Built atop NetworkX MultiDiGraph with graph query DSL and JSON persistence.
    """

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()
        self._nodes_dict: Dict[str, SceneGraphNode] = {}
        self._edges_list: List[SceneGraphEdge] = []

    def add_node(self, node: SceneGraphNode) -> None:
        """Insert or update a spatiotemporal instance node."""
        self._nodes_dict[node.node_id] = node
        self._graph.add_node(node.node_id, **node.model_dump())

    def get_node(self, node_id: str) -> Optional[SceneGraphNode]:
        """Fetch node by identifier."""
        return self._nodes_dict.get(node_id)

    def add_edge(self, edge: SceneGraphEdge) -> None:
        """Insert a relational edge between two existing nodes."""
        if edge.source_id not in self._nodes_dict:
            raise KeyError(f"Source node '{edge.source_id}' does not exist in scene graph.")
        if edge.target_id not in self._nodes_dict:
            raise KeyError(f"Target node '{edge.target_id}' does not exist in scene graph.")

        self._edges_list.append(edge)
        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=f"{edge.relation_type}_{edge.time_span}",
            **edge.model_dump(),
        )

    def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relation_type: Optional[str] = None,
    ) -> List[SceneGraphEdge]:
        """Query edges matching source, target, or relation filters."""
        res: List[SceneGraphEdge] = []
        for e in self._edges_list:
            if source_id and e.source_id != source_id:
                continue
            if target_id and e.target_id != target_id:
                continue
            if relation_type and e.relation_type != relation_type:
                continue
            res.append(e)
        return res

    def list_nodes(self) -> List[SceneGraphNode]:
        """List all registered nodes."""
        return list(self._nodes_dict.values())

    def list_edges(self) -> List[SceneGraphEdge]:
        """List all registered edges."""
        return list(self._edges_list)

    def get_active_subgraph_at_frame(self, frame_index: int) -> Persistent4DSceneGraph:
        """Extract slice of scene graph active at a specific frame timestamp."""
        subgraph = Persistent4DSceneGraph()
        for node in self._nodes_dict.values():
            if node.time_span[0] <= frame_index <= node.time_span[1]:
                subgraph.add_node(node)

        for edge in self._edges_list:
            if edge.time_span[0] <= frame_index <= edge.time_span[1]:
                if (
                    subgraph.get_node(edge.source_id) is not None
                    and subgraph.get_node(edge.target_id) is not None
                ):
                    subgraph.add_edge(edge)

        return subgraph

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scene graph into JSON-compatible dictionary."""
        return {
            "nodes": [n.model_dump() for n in self._nodes_dict.values()],
            "edges": [e.model_dump() for e in self._edges_list],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize scene graph to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Persistent4DSceneGraph:
        """Instantiate scene graph from dictionary."""
        sg = cls()
        for nd in data.get("nodes", []):
            sg.add_node(SceneGraphNode.model_validate(nd))
        for ed in data.get("edges", []):
            sg.add_edge(SceneGraphEdge.model_validate(ed))
        return sg

    @classmethod
    def from_json(cls, json_str: str) -> Persistent4DSceneGraph:
        """Instantiate scene graph from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute Scene Graph Query DSL.
        Syntax:
        - 'FIND ?x WHERE ?x.label == "person"'
        - 'FIND ?x WHERE ?x.label == "car" AND ?x.color == "red"'
        - 'FIND ?x, ?y WHERE ?x.label == "person" AND ?y.label == "umbrella" AND ?x.holding == ?y'
        - 'FIND ?x WHERE ?x.distance < 5.0'
        """
        query = query.strip()
        pattern = r"^FIND\s+([\?\w\s,]+)\s+WHERE\s+(.+)$"
        match = re.match(pattern, query, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid DSL query syntax: '{query}'. Expected 'FIND ?var WHERE ...'")

        vars_str, conditions_str = match.groups()
        target_vars = [v.strip().lstrip("?") for v in vars_str.split(",") if v.strip()]

        # Split conditions by AND
        raw_clauses = [c.strip() for c in conditions_str.split("AND") if c.strip()]

        # Node-level attribute filters: var -> list of (key_path, operator, value)
        node_filters: Dict[str, List[Tuple[str, str, Any]]] = {v: [] for v in target_vars}
        edge_filters: List[Tuple[str, str, str]] = []  # (src_var, relation, target_var)

        for clause in raw_clauses:
            # Check edge relation clause: ?x.relation == ?y or ?x relation ?y
            edge_match = re.match(r"^\?(\w+)\.([\w_]+)\s*==\s*\?(\w+)$", clause)
            if edge_match:
                s, rel, t = edge_match.groups()
                edge_filters.append((s, rel, t))
                continue

            # Check node attribute condition: ?x.field == "value" or ?x.field < value
            attr_match = re.match(r"^\?(\w+)\.([\w\._]+)\s*(==|!=|<=|>=|<|>)\s*(.+)$", clause)
            if attr_match:
                v, field_name, op, val_str = attr_match.groups()
                clean_val_str = val_str.strip().strip('"\'')
                # Parse numeric if possible
                try:
                    parsed_val = float(clean_val_str) if "." in clean_val_str else int(clean_val_str)
                except ValueError:
                    parsed_val = clean_val_str

                if v in node_filters:
                    node_filters[v].append((field_name, op, parsed_val))
                continue

        # Single variable query resolution
        if len(target_vars) == 1:
            var_name = target_vars[0]
            matches: List[Dict[str, Any]] = []
            for node in self._nodes_dict.values():
                if self._evaluate_node_filters(node, node_filters[var_name]):
                    matches.append({var_name: node.model_dump()})
            return matches

        # Two variable query resolution with relational joins
        elif len(target_vars) == 2:
            v1, v2 = target_vars[0], target_vars[1]
            matches = []
            cand_1 = [n for n in self._nodes_dict.values() if self._evaluate_node_filters(n, node_filters[v1])]
            cand_2 = [n for n in self._nodes_dict.values() if self._evaluate_node_filters(n, node_filters[v2])]

            for n1 in cand_1:
                for n2 in cand_2:
                    if n1.node_id == n2.node_id:
                        continue
                    # Check edge filters
                    edges_satisfied = True
                    for src_v, rel_type, tgt_v in edge_filters:
                        src_node = n1 if src_v == v1 else n2
                        tgt_node = n2 if tgt_v == v2 else n1
                        matching_edges = self.get_edges(
                            source_id=src_node.node_id,
                            target_id=tgt_node.node_id,
                            relation_type=rel_type,
                        )
                        if not matching_edges:
                            edges_satisfied = False
                            break
                    if edges_satisfied:
                        matches.append({v1: n1.model_dump(), v2: n2.model_dump()})

            return matches

        return []

    def _evaluate_node_filters(self, node: SceneGraphNode, filters: List[Tuple[str, str, Any]]) -> bool:
        """Check if a node satisfies all specified property constraints."""
        for field_path, op, target_val in filters:
            # Resolve field value
            val = None
            if field_path == "label":
                val = node.label
            elif field_path in ("color", "posture", "material"):
                val = node.attributes.get(field_path)
            elif field_path.startswith("attributes."):
                attr_key = field_path.split(".", 1)[1]
                val = node.attributes.get(attr_key)
            elif field_path == "distance" and node.centroid_3d:
                # Euclidean distance to camera
                val = float(np.linalg.norm(node.centroid_3d))
            elif hasattr(node, field_path):
                val = getattr(node, field_path)

            if val is None:
                return False

            if op == "==" and str(val).lower() != str(target_val).lower():
                return False
            elif op == "!=" and str(val).lower() == str(target_val).lower():
                return False
            elif op == "<" and not (val < target_val):
                return False
            elif op == "<=" and not (val <= target_val):
                return False
            elif op == ">" and not (val > target_val):
                return False
            elif op == ">=" and not (val >= target_val):
                return False

        return True


def compute_graph_diff(
    graph_t1: Persistent4DSceneGraph,
    graph_t2: Persistent4DSceneGraph,
    frame_index: int,
) -> List[GraphDiffEvent]:
    """
    Compute graph difference between two timestamps (t1 -> t2).
    Emits formal events for instance appearances, disappearances, attribute transitions, and relation changes.
    """
    events: List[GraphDiffEvent] = []

    nodes_1 = {n.node_id: n for n in graph_t1.list_nodes()}
    nodes_2 = {n.node_id: n for n in graph_t2.list_nodes()}

    # 1. Instance Appearances
    for nid, node in nodes_2.items():
        if nid not in nodes_1:
            events.append(
                GraphDiffEvent(
                    event_type="INSTANCE_APPEARED",
                    subject_id=nid,
                    frame_index=frame_index,
                    details={"label": node.label, "centroid_3d": node.centroid_3d},
                )
            )

    # 2. Instance Disappearances
    for nid, node in nodes_1.items():
        if nid not in nodes_2:
            events.append(
                GraphDiffEvent(
                    event_type="INSTANCE_DISAPPEARED",
                    subject_id=nid,
                    frame_index=frame_index,
                    details={"label": node.label},
                )
            )

    # 3. Attribute Transitions
    for nid in nodes_1.keys() & nodes_2.keys():
        n1 = nodes_1[nid]
        n2 = nodes_2[nid]
        diff_attrs = {}
        for k, v in n2.attributes.items():
            if k not in n1.attributes or n1.attributes[k] != v:
                diff_attrs[k] = {"before": n1.attributes.get(k), "after": v}
        if diff_attrs:
            events.append(
                GraphDiffEvent(
                    event_type="ATTRIBUTE_CHANGED",
                    subject_id=nid,
                    frame_index=frame_index,
                    details={"changed_attributes": diff_attrs},
                )
            )

    # 4. Relations Formed & Broken
    edges_1 = {(e.source_id, e.target_id, e.relation_type) for e in graph_t1.list_edges()}
    edges_2 = {(e.source_id, e.target_id, e.relation_type) for e in graph_t2.list_edges()}

    for src, tgt, rel in edges_2 - edges_1:
        events.append(
            GraphDiffEvent(
                event_type="RELATION_FORMED",
                subject_id=src,
                object_id=tgt,
                frame_index=frame_index,
                details={"relation_type": rel},
            )
        )

    for src, tgt, rel in edges_1 - edges_2:
        events.append(
            GraphDiffEvent(
                event_type="RELATION_BROKEN",
                subject_id=src,
                object_id=tgt,
                frame_index=frame_index,
                details={"relation_type": rel},
            )
        )

    return events
