"""
Geometric primitives and computational geometry engine.
Defines Point2D, Point3D, BoundingBox2D, BoundingBox3D, and extends GeometryEngine.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field, model_validator

from src.geometry.relations import (
    GeometryEngine as BaseGeometryEngine,
    LineSegment,
    LineCrossingEvent,
    SpatialRelation,
    EgocentricDirection,
    OcclusionPair,
)


class Point2D(BaseModel):
    """2D spatial point in normalized image coordinates [0, 1]."""
    x: float
    y: float


class Point3D(BaseModel):
    """3D spatial coordinate in camera or world coordinate frame (metres)."""
    x: float
    y: float
    z: float


class BoundingBox2D(BaseModel):
    """2D bounding box with strict geometric sanity checks."""
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)
    label: str = ""
    score: float = 1.0

    @model_validator(mode="after")
    def validate_bounds(self) -> BoundingBox2D:
        if self.x2 < self.x1:
            raise ValueError(f"Invalid bounding box: x2 ({self.x2}) must be >= x1 ({self.x1})")
        if self.y2 < self.y1:
            raise ValueError(f"Invalid bounding box: y2 ({self.y2}) must be >= y1 ({self.y1})")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


class BoundingBox3D(BaseModel):
    """Oriented 3D bounding box with center, extent dimensions, and orientation."""
    center: Point3D
    dimensions: Point3D  # (width, height, depth) in metres
    orientation: Point3D  # (yaw, pitch, roll) in radians
    label: str = ""
    confidence: float = 1.0

    @property
    def volume(self) -> float:
        return abs(self.dimensions.x * self.dimensions.y * self.dimensions.z)


class GeometryEngine(BaseGeometryEngine):
    """Extended geometry engine incorporating primitive point and bounding box types."""

    @classmethod
    def check_line_crossing(
        cls,
        track_id: str,
        prev_pos: Any,
        curr_pos: Any,
        line: Any,
        frame_idx: int = 0,
    ) -> Optional[Any]:
        # Handle Point2D or Tuple
        p1 = (prev_pos.x, prev_pos.y) if hasattr(prev_pos, "x") else (prev_pos[0], prev_pos[1])
        p2 = (curr_pos.x, curr_pos.y) if hasattr(curr_pos, "x") else (curr_pos[0], curr_pos[1])

        line_name = getattr(line, "name", "tripwire")
        if hasattr(line, "line"):
            line_seg = line.line
            lp1 = (line_seg.p1.x, line_seg.p1.y) if hasattr(line_seg.p1, "x") else (line_seg.p1[0], line_seg.p1[1])
            lp2 = (line_seg.p2.x, line_seg.p2.y) if hasattr(line_seg.p2, "x") else (line_seg.p2[0], line_seg.p2[1])
        elif hasattr(line, "p1"):
            lp1 = (line.p1.x, line.p1.y) if hasattr(line.p1, "x") else (line.p1[0], line.p1[1])
            lp2 = (line.p2.x, line.p2.y) if hasattr(line.p2, "x") else (line.p2[0], line.p2[1])
        else:
            lp1 = (0.0, 0.5)
            lp2 = (1.0, 0.5)

        base_seg = LineSegment(name=line_name, p1=lp1, p2=lp2)
        evt = BaseGeometryEngine.check_line_crossing(track_id, p1, p2, base_seg, frame_idx)
        if evt:
            # Map direction
            evt.direction = evt.crossing_direction
        return evt
