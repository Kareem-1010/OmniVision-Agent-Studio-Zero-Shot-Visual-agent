"""
Retail Analytics Domain Preset.
Implements retail intelligence:
1. Footfall counting & bidirectional tripwire entry/exit tracking.
2. Queue length monitoring and Little's Law / departure-based wait-time estimation.
3. Shelf gap & out-of-stock (OOS) empty slot detection.
4. Shopper dwell time profiling and 2D spatial dwell heatmap generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from src.geometry.relations import GeometryEngine, LineCrossingEvent, LineSegment
from src.presets.base import BaseDomainPreset, DomainPresetConfig, get_preset_registry

logger = logging.getLogger("OmniVision.Presets.Retail")


class ShelfGapItem(BaseModel):
    """Detected out-of-stock shelf gap or empty slot."""
    shelf_id: str
    gap_box: List[float] = Field(description="Normalized [x1, y1, x2, y2] gap coordinates")
    gap_width_ratio: float = Field(ge=0.0, le=1.0)
    category_label: str = "general_merchandise"
    restock_priority: str = "normal"  # normal, urgent, critical


class QueueStatus(BaseModel):
    """Real-time service counter or checkout queue metrics."""
    queue_id: str
    people_count: int = Field(ge=0)
    queue_head_pos: Optional[List[float]] = None
    queue_tail_pos: Optional[List[float]] = None
    estimated_wait_time_sec: float = Field(ge=0.0)
    congestion_level: str = "low"  # low, moderate, severe


class RetailKPIReport(BaseModel):
    """Consolidated Retail Intelligence KPI Report."""
    media_uri: str
    timestamp: float
    footfall_in: int = 0
    footfall_out: int = 0
    net_store_occupancy: int = 0
    average_dwell_time_sec: float = 0.0
    queues: Dict[str, QueueStatus] = Field(default_factory=dict)
    shelf_gaps: List[ShelfGapItem] = Field(default_factory=list)
    out_of_stock_percentage: float = 0.0
    heatmap_resolution: Tuple[int, int] = (50, 50)
    dwell_heatmap_matrix: List[List[float]] = Field(default_factory=list)
    summary_text: str = ""


class RetailPreset(BaseDomainPreset):
    """Domain Preset for Retail Analytics and Store Optimization."""

    def _default_config(self) -> DomainPresetConfig:
        return DomainPresetConfig(
            name="retail",
            description="Retail analytics preset for footfall, queue wait times, shelf gaps, and dwell heatmaps.",
            system_prompt=(
                "You are OmniVision Retail Specialist. Analyze retail environments to measure footfall, "
                "shopper dwell time, checkout queue congestion, and shelf out-of-stock gaps. "
                "Produce precise quantitative counts, spatial heatmaps, and restock alerts."
            ),
            enabled_tools=[
                "detector_ensemble",
                "open_vocab",
                "grounded_segmenter",
                "video_tracker",
                "scene_ocr",
                "metric_depth",
                "visualizer",
            ],
            confidence_threshold=0.45,
            max_iterations=8,
            budget_limit_usd=0.25,
            custom_parameters={
                "tripwire_entry": [(0.2, 0.8), (0.8, 0.8)],
                "dwell_bins": 50,
                "service_rate_per_min": 1.5,  # shoppers served per minute per register
                "shelf_gap_threshold_ratio": 0.15,
            },
        )

    def analyze(self, media_uri: str, query: Optional[str] = None, **kwargs: Any) -> RetailKPIReport:
        """Execute complete retail scene or video analytics pipeline."""
        tracks = kwargs.get("tracks", [])
        shelves = kwargs.get("shelves", [])
        tripwire = kwargs.get("tripwire", self.config.custom_parameters.get("tripwire_entry"))
        return self.analyze_retail_scene(media_uri=media_uri, tracks=tracks, shelves=shelves, tripwire=tripwire)

    def compute_footfall(
        self,
        tracks: List[Dict[str, Any]],
        tripwire: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[int, int, Dict[str, float]]:
        """
        Compute bidirectional footfall (in / out) across entrance tripwires and per-shopper dwell time.
        """
        wire = tripwire or [(0.2, 0.8), (0.8, 0.8)]
        p1 = wire[0]
        p2 = wire[1]
        line_seg = LineSegment(name="tripwire", p1=p1, p2=p2)

        footfall_in = 0
        footfall_out = 0
        dwell_times: Dict[str, float] = {}

        for trk in tracks:
            tid = str(trk.get("track_id", "trk_0"))
            pts = trk.get("trajectory", [])  # list of [x, y, timestamp]
            if len(pts) < 2:
                continue

            # Calculate dwell duration
            t_start = pts[0][2] if len(pts[0]) > 2 else 0.0
            t_end = pts[-1][2] if len(pts[-1]) > 2 else 0.0
            dwell_times[tid] = max(0.0, t_end - t_start)

            # Check for line crossing across trajectory
            for i in range(len(pts) - 1):
                pt_a = (pts[i][0], pts[i][1])
                pt_b = (pts[i + 1][0], pts[i + 1][1])
                crossing = GeometryEngine.check_line_crossing(
                    track_id=tid,
                    prev_pt=pt_a,
                    curr_pt=pt_b,
                    line=line_seg,
                    frame_idx=i + 1,
                )
                if crossing:
                    # For horizontal entrance tripwires, moving up (decreasing y) is entering
                    if pt_b[1] < pt_a[1]:
                        footfall_in += 1
                    else:
                        footfall_out += 1
                    break

        return footfall_in, footfall_out, dwell_times

    def monitor_queues(
        self,
        shopper_positions: List[Tuple[float, float]],
        checkout_zones: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
        service_rate_per_min: float = 1.5,
    ) -> Dict[str, QueueStatus]:
        """
        Detect queue lengths in checkout zones and estimate wait times using Little's law / service rates.
        """
        zones = checkout_zones or {
            "register_01": (0.6, 0.6, 0.95, 0.95),
            "customer_service": (0.05, 0.5, 0.35, 0.9),
        }
        queues: Dict[str, QueueStatus] = {}

        for q_name, (x1, y1, x2, y2) in zones.items():
            in_zone = [p for p in shopper_positions if x1 <= p[0] <= x2 and y1 <= p[1] <= y2]
            count = len(in_zone)

            # Estimate wait time: Wait = Count / ServiceRate
            rate_per_sec = max(0.01, service_rate_per_min / 60.0)
            wait_time_sec = count / rate_per_sec

            congestion = "low"
            if count >= 6:
                congestion = "severe"
            elif count >= 3:
                congestion = "moderate"

            head = in_zone[0] if in_zone else None
            tail = in_zone[-1] if in_zone else None

            queues[q_name] = QueueStatus(
                queue_id=q_name,
                people_count=count,
                queue_head_pos=list(head) if head else None,
                queue_tail_pos=list(tail) if tail else None,
                estimated_wait_time_sec=round(wait_time_sec, 1),
                congestion_level=congestion,
            )

        return queues

    def detect_shelf_gaps(
        self,
        shelves_data: List[Dict[str, Any]],
        gap_threshold_ratio: float = 0.15,
    ) -> Tuple[List[ShelfGapItem], float]:
        """
        Analyze planogram shelf compartments for out-of-stock voids or gaps.
        """
        gaps: List[ShelfGapItem] = []
        total_shelf_slots = 0
        empty_slots = 0

        for shelf in shelves_data:
            shelf_id = str(shelf.get("shelf_id", "shelf_main"))
            slots = shelf.get("slots", [])
            total_shelf_slots += max(1, len(slots))

            for s_idx, slot in enumerate(slots):
                # slot has [x1, y1, x2, y2], is_occupied: bool, width_ratio: float, category: str
                occupied = slot.get("is_occupied", True)
                if not occupied:
                    empty_slots += 1
                    w_ratio = float(slot.get("width_ratio", 0.2))
                    priority = "normal"
                    if w_ratio >= gap_threshold_ratio * 2.0:
                        priority = "critical"
                    elif w_ratio >= gap_threshold_ratio:
                        priority = "urgent"

                    gaps.append(
                        ShelfGapItem(
                            shelf_id=f"{shelf_id}_slot_{s_idx}",
                            gap_box=slot.get("box", [0.2, 0.4, 0.35, 0.6]),
                            gap_width_ratio=round(w_ratio, 3),
                            category_label=slot.get("category", "beverages"),
                            restock_priority=priority,
                        )
                    )

        oos_pct = (empty_slots / max(1, total_shelf_slots)) * 100.0
        return gaps, round(oos_pct, 1)

    def generate_dwell_heatmap(
        self,
        trajectories: List[List[Tuple[float, float]]],
        grid_size: Tuple[int, int] = (50, 50),
    ) -> np.ndarray:
        """
        Accumulate 2D trajectory points into normalized spatial dwell heatmap matrix [H, W].
        """
        h, w = grid_size
        heatmap = np.zeros((h, w), dtype=np.float32)

        for traj in trajectories:
            for pt in traj:
                x = min(w - 1, max(0, int(pt[0] * w)))
                y = min(h - 1, max(0, int(pt[1] * h)))
                # Gaussian splatting onto grid
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny = y + dy
                        nx = x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            dist_sq = dx * dx + dy * dy
                            weight = np.exp(-dist_sq / 2.0)
                            heatmap[ny, nx] += float(weight)

        max_val = float(heatmap.max())
        if max_val > 0.0:
            heatmap = heatmap / max_val
        return heatmap

    def analyze_retail_scene(
        self,
        media_uri: str,
        tracks: Optional[List[Dict[str, Any]]] = None,
        shelves: Optional[List[Dict[str, Any]]] = None,
        tripwire: Optional[List[Tuple[float, float]]] = None,
    ) -> RetailKPIReport:
        """
        Consolidate all retail metrics into RetailKPIReport.
        """
        track_list = tracks or []
        shelf_list = shelves or []

        # 1. Footfall and Dwell Times
        foot_in, foot_out, dwell_dict = self.compute_footfall(track_list, tripwire)
        net_occ = max(0, foot_in - foot_out)
        avg_dwell = float(np.mean(list(dwell_dict.values()))) if dwell_dict else 0.0

        # 2. Queue Analysis
        shopper_pts = [
            (t["trajectory"][-1][0], t["trajectory"][-1][1])
            for t in track_list
            if "trajectory" in t and len(t["trajectory"]) > 0
        ]
        queues = self.monitor_queues(shopper_pts)

        # 3. Shelf Gap Analysis
        gaps, oos_pct = self.detect_shelf_gaps(shelf_list)

        # 4. Spatial Dwell Heatmap
        trajs = [
            [(p[0], p[1]) for p in t.get("trajectory", [])]
            for t in track_list
        ]
        heatmap_matrix = self.generate_dwell_heatmap(trajs, grid_size=(30, 30))

        summary = (
            f"Retail Intelligence: Footfall In: {foot_in}, Out: {foot_out} (Net Occupancy: {net_occ}). "
            f"Avg Dwell: {avg_dwell:.1f}s. Queues Active: {len(queues)} with peak wait time: "
            f"{max((q.estimated_wait_time_sec for q in queues.values()), default=0.0):.1f}s. "
            f"Shelf Out-Of-Stock: {oos_pct:.1f}% ({len(gaps)} empty slots)."
        )

        return RetailKPIReport(
            media_uri=media_uri,
            timestamp=1000.0,
            footfall_in=foot_in,
            footfall_out=foot_out,
            net_store_occupancy=net_occ,
            average_dwell_time_sec=round(avg_dwell, 1),
            queues=queues,
            shelf_gaps=gaps,
            out_of_stock_percentage=oos_pct,
            heatmap_resolution=(30, 30),
            dwell_heatmap_matrix=heatmap_matrix.tolist(),
            summary_text=summary,
        )


# Register retail preset
get_preset_registry().register(RetailPreset())


class TripwireConfig(BaseModel):
    """Configuration for virtual tripwire line and crossing semantics."""
    line: Any
    direction: str = "down"
    label: str = "tripwire"


# Backward compatibility alias
RetailAnalyticsPipeline = RetailPreset

