"""
Streaming Video Ingestion and Sliding Window Online Buffer.
Enables real-time continuous video processing over live camera feeds, maintaining a
FIFO sliding window, online scene graph updates, and low-latency anomaly alert emission.
"""

from __future__ import annotations

import collections
import logging
import uuid
from typing import Any, Callable, Deque, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from src.memory.scene_graph import Persistent4DSceneGraph, SceneGraphNode, compute_graph_diff
from src.tools.detector_ensemble import DetectorEnsembleTool
from src.tools.pose_action import PoseActionTool

logger = logging.getLogger("OmniVision.Video.Streaming")


class StreamAlert(BaseModel):
    """Real-time event or safety hazard notification emitted by the streaming engine."""
    alert_id: str = Field(default_factory=lambda: f"alt_{uuid.uuid4().hex[:8]}")
    alert_type: str  # e.g., 'fall_detected', 'restricted_zone_breach', 'ppe_violation', 'speed_anomaly'
    severity: Literal["info", "warning", "critical"] = "warning"
    timestamp: float
    frame_index: int
    description: str
    source_track_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BufferedFrame(BaseModel):
    """Single frame stored in the active sliding window buffer."""
    frame_index: int
    timestamp: float
    image_uri: str
    detections: List[Dict[str, Any]] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)


class StreamingVideoProcessor:
    """
    Streaming Video Ingestion & Sliding Window Engine.
    Processes live frame feeds, updates a continuous 4D scene graph,
    and dispatches instant alerts on physical or perceptual anomalies.
    """

    def __init__(
        self,
        window_size_frames: int = 60,
        fps: float = 30.0,
        alert_callback: Optional[Callable[[StreamAlert], None]] = None,
    ) -> None:
        self.window_size = window_size_frames
        self.fps = fps
        self.alert_callback = alert_callback

        # Sliding Window FIFO Queue
        self._buffer: Deque[BufferedFrame] = collections.deque(maxlen=window_size_frames)
        self._frame_counter = 0

        # Persistent Scene Graph & Perception tools
        self.scene_graph = Persistent4DSceneGraph()
        self._prev_graph_slice = Persistent4DSceneGraph()
        self._detector = DetectorEnsembleTool()
        self._pose_tool = PoseActionTool()

        # Alert history
        self._emitted_alerts: List[StreamAlert] = []

    def ingest_frame(
        self,
        image_uri: str,
        timestamp: Optional[float] = None,
        check_ppe_and_falls: bool = True,
    ) -> List[StreamAlert]:
        """
        Ingest a new incoming video frame into the sliding window.
        Executes online perception, updates scene graph, and checks alert conditions.
        """
        f_idx = self._frame_counter
        self._frame_counter += 1
        t_sec = timestamp if timestamp is not None else round(f_idx / self.fps, 3)

        new_alerts: List[StreamAlert] = []

        # 1. Perception on Frame: Fast detection
        det_res = self._detector.execute({"image_uri": image_uri})
        dets = []
        if det_res.success and det_res.data:
            for d in det_res.data.detections:
                b = d.box
                dets.append({
                    "label": d.label,
                    "box": [b.x1, b.y1, b.x2, b.y2],
                    "confidence": d.confidence,
                })

        # 2. Safety / Fall check if requested
        if check_ppe_and_falls:
            pose_res = self._pose_tool.execute({"image_uri": image_uri, "detect_falls": True, "check_ppe": True})
            if pose_res.success and pose_res.data:
                for person in pose_res.data.persons:
                    is_fall = getattr(person, "is_fallen", False) or getattr(person, "is_fall_detected", False)
                    if is_fall or pose_res.data.fall_detected_in_scene:
                        alert = StreamAlert(
                            alert_type="fall_detected",
                            severity="critical",
                            timestamp=t_sec,
                            frame_index=f_idx,
                            description=f"Person fall detected in frame {f_idx} at {t_sec:.2f}s!",
                            metadata={"bbox": person.box if hasattr(person, "box") else []},
                        )
                        new_alerts.append(alert)

                    # Check PPE
                    ppe = person.ppe_compliance
                    missing = [k for k, v in ppe.items() if not v]
                    if missing:
                        alert = StreamAlert(
                            alert_type="ppe_violation",
                            severity="warning",
                            timestamp=t_sec,
                            frame_index=f_idx,
                            description=f"PPE compliance violation: missing {', '.join(missing)} at {t_sec:.2f}s.",
                            metadata={"missing_ppe": missing},
                        )
                        new_alerts.append(alert)

        # 3. Update Sliding Window Buffer
        buf_frame = BufferedFrame(
            frame_index=f_idx,
            timestamp=t_sec,
            image_uri=image_uri,
            detections=dets,
        )
        self._buffer.append(buf_frame)

        # 4. Online 4D Scene Graph Update
        curr_graph_slice = Persistent4DSceneGraph()
        for i, d in enumerate(dets):
            node_id = f"stream_inst_{d['label']}_{i}"
            curr_graph_slice.add_node(
                SceneGraphNode(
                    node_id=node_id,
                    label=d["label"],
                    bbox_2d=d["box"],
                    time_span=(f_idx, f_idx),
                    attributes={"confidence": d["confidence"]},
                )
            )

        # Compute graph diff against previous frame slice
        diff_events = compute_graph_diff(self._prev_graph_slice, curr_graph_slice, frame_index=f_idx)
        self._prev_graph_slice = curr_graph_slice

        # Trigger callbacks and store alerts
        for alt in new_alerts:
            self._emitted_alerts.append(alt)
            if self.alert_callback:
                try:
                    self.alert_callback(alt)
                except Exception as ex:
                    logger.warning(f"Error in streaming alert callback: {ex}")

        return new_alerts

    def get_window_frames(self) -> List[BufferedFrame]:
        """Fetch all frames currently in the sliding window buffer."""
        return list(self._buffer)

    def get_alert_history(self) -> List[StreamAlert]:
        """Fetch complete log of emitted alerts."""
        return list(self._emitted_alerts)

    def clear(self) -> None:
        """Reset streaming buffer and state."""
        self._buffer.clear()
        self._frame_counter = 0
        self._emitted_alerts.clear()
        self.scene_graph = Persistent4DSceneGraph()
        self._prev_graph_slice = Persistent4DSceneGraph()
