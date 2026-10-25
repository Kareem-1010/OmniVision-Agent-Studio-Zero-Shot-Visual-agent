"""
Temporal Event Detection and Boundary Segmentation Engine.
Fuses multi-object tracklets, tripwire crossings, and 4D scene graph diffs into
discrete, bounded semantic events (onset, peak, offset).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from src.geometry.relations import LineCrossingEvent
from src.memory.scene_graph import GraphDiffEvent

logger = logging.getLogger("OmniVision.Video.Events")


class SemanticEvent(BaseModel):
    """Discrete semantic action or interaction with temporal start/peak/end bounds."""
    event_id: str = Field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:8]}")
    event_type: str  # e.g., 'instance_entered', 'instance_exited', 'object_manipulation', 'fall_incident', 'proximity_interaction'
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    peak_frame: int
    peak_time_sec: Optional[float] = None
    primary_track_id: str
    secondary_track_id: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    description: str = ""
    evidence_tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.peak_time_sec is None:
            self.peak_time_sec = round((self.start_time_sec + self.end_time_sec) / 2.0, 3)


class TemporalEventSegmenter:
    """
    Temporal Event Detector & Boundary Segmenter.
    Aggregates multi-frame tracking trajectories, line crossings, and scene graph diffs
    to isolate and segment coherent physical events.
    """

    def __init__(self, fps: float = 30.0, min_event_duration_frames: int = 3) -> None:
        self.fps = fps
        self.min_duration = min_event_duration_frames

    def detect_events_from_tracks_and_diffs(
        self,
        tracks: List[Dict[str, Any]],
        graph_diffs: Optional[List[GraphDiffEvent]] = None,
        line_crossings: Optional[List[LineCrossingEvent]] = None,
    ) -> List[SemanticEvent]:
        """
        Segment continuous tracking history and graph transitions into discrete SemanticEvents.
        """
        events: List[SemanticEvent] = []

        # 1. Process Line Crossing Events (e.g. tripwires, perimeter entry/exit)
        if line_crossings:
            for lc in line_crossings:
                f_idx = lc.frame_index
                t_sec = round(f_idx / self.fps, 3)
                ev_type = f"boundary_{lc.crossing_direction}"
                desc = f"Track '{lc.track_id}' crossed '{lc.line_name}' ({lc.crossing_direction}) at {t_sec}s."

                events.append(
                    SemanticEvent(
                        event_type=ev_type,
                        start_frame=max(0, f_idx - 2),
                        end_frame=f_idx + 2,
                        start_time_sec=round(max(0.0, t_sec - 0.1), 3),
                        end_time_sec=round(t_sec + 0.1, 3),
                        peak_frame=f_idx,
                        primary_track_id=lc.track_id,
                        confidence=0.95,
                        description=desc,
                        evidence_tags=[f"line_crossing:{lc.line_name}", f"direction:{lc.crossing_direction}"],
                        metadata={"intersection_point": lc.intersection_point},
                    )
                )

        # 2. Process Graph Diff Events (appearances, attribute shifts, relation changes)
        if graph_diffs:
            for gd in graph_diffs:
                f_idx = gd.frame_index
                t_sec = round(f_idx / self.fps, 3)

                if gd.event_type == "INSTANCE_APPEARED":
                    lbl = gd.details.get("label", "instance")
                    events.append(
                        SemanticEvent(
                            event_type="instance_appeared",
                            start_frame=f_idx,
                            end_frame=f_idx + 5,
                            start_time_sec=t_sec,
                            end_time_sec=round(t_sec + 5.0 / self.fps, 3),
                            peak_frame=f_idx,
                            primary_track_id=gd.subject_id,
                            confidence=0.90,
                            description=f"New instance '{lbl}' ({gd.subject_id}) appeared in scene at {t_sec}s.",
                            evidence_tags=["graph_diff:appearance", f"label:{lbl}"],
                        )
                    )
                elif gd.event_type == "RELATION_FORMED":
                    rel = gd.details.get("relation_type", "related")
                    events.append(
                        SemanticEvent(
                            event_type=f"relation_{rel}",
                            start_frame=max(0, f_idx - 3),
                            end_frame=f_idx + 3,
                            start_time_sec=round(max(0.0, t_sec - 0.1), 3),
                            end_time_sec=round(t_sec + 0.1, 3),
                            peak_frame=f_idx,
                            primary_track_id=gd.subject_id,
                            secondary_track_id=gd.object_id,
                            confidence=0.88,
                            description=f"Relation formed: '{gd.subject_id}' is now {rel} '{gd.object_id}' at {t_sec}s.",
                            evidence_tags=["graph_diff:relation", f"relation:{rel}"],
                        )
                    )
                elif gd.event_type == "ATTRIBUTE_CHANGED":
                    changed = gd.details.get("changed_attributes", {})
                    if "posture" in changed:
                        p_after = changed["posture"].get("after")
                        ev_name = "fall_incident" if p_after in ("fallen", "lying") else "posture_transition"
                        events.append(
                            SemanticEvent(
                                event_type=ev_name,
                                start_frame=max(0, f_idx - 5),
                                end_frame=f_idx + 10,
                                start_time_sec=round(max(0.0, t_sec - 0.15), 3),
                                end_time_sec=round(t_sec + 0.3, 3),
                                peak_frame=f_idx,
                                primary_track_id=gd.subject_id,
                                confidence=0.92,
                                description=f"Posture changed for '{gd.subject_id}' to {p_after} at {t_sec}s.",
                                evidence_tags=["attribute_change:posture", f"posture:{p_after}"],
                            )
                        )

        # 3. Process Kinematic Track Trajectories (Stationary -> Moving or High Velocity)
        for trk in tracks:
            tid = str(trk.get("track_id", "unknown"))
            pts = trk.get("trajectory", [])
            frames = trk.get("frames_seen", [])
            if len(pts) >= 4 and len(frames) == len(pts):
                speeds = []
                for k in range(1, len(pts)):
                    dx = pts[k][0] - pts[k-1][0]
                    dy = pts[k][1] - pts[k-1][1]
                    dist_px = (dx**2 + dy**2)**0.5
                    speeds.append(dist_px)

                # Detect sudden acceleration / motion burst
                max_speed = max(speeds) if speeds else 0.0
                if max_speed > 25.0:  # significant movement threshold
                    peak_idx = speeds.index(max_speed)
                    peak_frame = frames[peak_idx]
                    t_peak = round(peak_frame / self.fps, 3)

                    events.append(
                        SemanticEvent(
                            event_type="rapid_movement",
                            start_frame=frames[0],
                            end_frame=frames[-1],
                            start_time_sec=round(frames[0] / self.fps, 3),
                            end_time_sec=round(frames[-1] / self.fps, 3),
                            peak_frame=peak_frame,
                            primary_track_id=tid,
                            confidence=0.85,
                            description=f"Track '{tid}' exhibited rapid movement (speed: {max_speed:.1f}px/frame) around {t_peak}s.",
                            evidence_tags=["trajectory:speed_burst"],
                        )
                    )

        # Sort all events chronologically by start timestamp
        events.sort(key=lambda e: (e.start_time_sec, e.peak_frame))
        logger.info(f"Segmented {len(events)} discrete semantic events from video history.")
        return events
