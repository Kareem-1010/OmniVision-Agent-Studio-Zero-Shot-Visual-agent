"""
OmniVision Geometry and Multi-View Spatial Engine Package.
"""

from src.geometry.relations import (
    GeometryEngine as BaseGeometryEngine,
    SpatialRelation,
    EgocentricDirection,
    OcclusionPair,
    LineSegment,
    LineCrossingEvent,
)
from src.geometry.slam_lite import (
    SLAMVisualOdometryLite,
    CameraPose,
    WorldInstance,
)
from src.geometry.engine import (
    Point2D,
    Point3D,
    BoundingBox2D,
    BoundingBox3D,
    GeometryEngine,
)
from src.geometry.scene_graph import (
    PersistentSceneGraph,
    Persistent4DSceneGraph,
    SceneGraphNode,
    SceneGraphEdge,
)

__all__ = [
    "GeometryEngine",
    "BaseGeometryEngine",
    "SpatialRelation",
    "EgocentricDirection",
    "OcclusionPair",
    "LineSegment",
    "LineCrossingEvent",
    "SLAMVisualOdometryLite",
    "CameraPose",
    "WorldInstance",
    "Point2D",
    "Point3D",
    "BoundingBox2D",
    "BoundingBox3D",
    "PersistentSceneGraph",
    "Persistent4DSceneGraph",
    "SceneGraphNode",
    "SceneGraphEdge",
]
