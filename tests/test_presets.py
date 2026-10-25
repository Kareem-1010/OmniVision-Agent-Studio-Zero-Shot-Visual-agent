"""
Unit and Integration Tests for Phase 7 Domain Presets and Demo Modes.
Tests:
- Preset Registry and Base configuration
- Retail Mode: Footfall, Queue Wait-Times, Shelf Gaps, Dwell Heatmaps
- Accessibility Mode: Egocentric Clock-face Bearings, OCR Narration, Obstacle Alerts, TTS Audio
- Hazard/Safety Mode: PPE Compliance, Environmental Hazards, 3D Proximity Risk, PDF Export
"""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import wave
import pytest
import numpy as np

from src.presets import (
    AccessibilityNarrationReport,
    AccessibilityPreset,
    BaseDomainPreset,
    DomainPresetConfig,
    HazardSafetyPreset,
    PresetRegistry,
    RetailKPIReport,
    RetailPreset,
    SafetyAuditReport,
    TTSSynthesizer,
    get_preset_registry,
)


# ---------------------------------------------------------------------------
# Test Registry & Base Architecture
# ---------------------------------------------------------------------------

def test_preset_registry():
    registry = get_preset_registry()
    names = registry.list_presets()
    assert "retail" in names
    assert "accessibility" in names
    assert "hazard_safety" in names

    retail_preset = registry.get("retail")
    assert isinstance(retail_preset, RetailPreset)
    assert "detector_ensemble" in retail_preset.get_toolset()
    assert "Retail Specialist" in retail_preset.get_system_prompt()

    # Unknown preset
    with pytest.raises(KeyError):
        registry.get("space_exploration")


# ---------------------------------------------------------------------------
# Test Retail Mode Preset
# ---------------------------------------------------------------------------

def test_retail_footfall_and_queues():
    preset = RetailPreset()

    # Synthetic shopper tracks crossing tripwire at y=0.8
    tracks = [
        {
            "track_id": "shopper_1",
            "trajectory": [
                [0.5, 0.9, 10.0],  # below line
                [0.5, 0.7, 12.0],  # crossed line upwards (entering)
                [0.5, 0.6, 25.0],
            ],
        },
        {
            "track_id": "shopper_2",
            "trajectory": [
                [0.4, 0.5, 15.0],
                [0.4, 0.75, 20.0],
                [0.4, 0.85, 30.0],  # crossed line downwards (exiting)
            ],
        },
    ]

    foot_in, foot_out, dwell = preset.compute_footfall(tracks, tripwire=[(0.2, 0.8), (0.8, 0.8)])
    assert foot_in >= 1
    assert foot_out >= 1
    assert dwell["shopper_1"] == 15.0  # 25.0 - 10.0
    assert dwell["shopper_2"] == 15.0  # 30.0 - 15.0

    # Test Queue monitoring
    shopper_positions = [
        (0.7, 0.7),
        (0.72, 0.75),
        (0.75, 0.80),
        (0.78, 0.85),
    ]
    queues = preset.monitor_queues(
        shopper_positions=shopper_positions,
        checkout_zones={"register_01": (0.6, 0.6, 0.9, 0.9)},
        service_rate_per_min=2.0,  # 2 per min = 1 every 30s
    )

    assert "register_01" in queues
    q1 = queues["register_01"]
    assert q1.people_count == 4
    assert q1.estimated_wait_time_sec == 120.0  # 4 * 30s
    assert q1.congestion_level in ["moderate", "severe"]


def test_retail_shelf_gaps_and_heatmap():
    preset = RetailPreset()

    shelves_data = [
        {
            "shelf_id": "aisle_4_shelf_b",
            "slots": [
                {"is_occupied": True, "category": "cereal"},
                {"is_occupied": False, "category": "cereal", "width_ratio": 0.25, "box": [0.3, 0.4, 0.5, 0.6]},
                {"is_occupied": True, "category": "cereal"},
                {"is_occupied": False, "category": "cereal", "width_ratio": 0.45, "box": [0.6, 0.4, 0.85, 0.6]},
            ],
        }
    ]

    gaps, oos_pct = preset.detect_shelf_gaps(shelves_data, gap_threshold_ratio=0.20)
    assert len(gaps) == 2
    assert oos_pct == 50.0
    assert gaps[0].restock_priority in ["urgent", "critical"]

    # Dwell heatmap test
    trajectories = [
        [(0.5, 0.5), (0.51, 0.52), (0.52, 0.51)],
        [(0.2, 0.8), (0.22, 0.79)],
    ]
    heatmap = preset.generate_dwell_heatmap(trajectories, grid_size=(20, 20))
    assert heatmap.shape == (20, 20)
    assert 0.0 <= heatmap.min()
    assert heatmap.max() == 1.0  # normalized


def test_retail_scene_analysis_pipeline():
    preset = RetailPreset()
    report = preset.analyze("store_camera_02.mp4")
    assert isinstance(report, RetailKPIReport)
    assert report.media_uri == "store_camera_02.mp4"
    assert "Retail Intelligence" in report.summary_text


# ---------------------------------------------------------------------------
# Test Accessibility Assistant Preset
# ---------------------------------------------------------------------------

def test_accessibility_clockface_and_elevation():
    preset = AccessibilityPreset()

    # Center object (straight ahead)
    assert preset.calculate_clock_face([0.45, 0.4, 0.55, 0.6], user_heading_deg=0.0) == "12 o'clock"
    # Far right object
    assert preset.calculate_clock_face([0.85, 0.4, 0.95, 0.6], user_heading_deg=0.0) in ["1 o'clock", "2 o'clock"]
    # Far left object
    assert preset.calculate_clock_face([0.05, 0.4, 0.15, 0.6], user_heading_deg=0.0) in ["10 o'clock", "11 o'clock"]

    # Elevation
    assert preset.estimate_elevation([0.4, 0.8, 0.6, 0.95]) == "ground_level"
    assert preset.estimate_elevation([0.4, 0.05, 0.6, 0.18]) == "overhead"


def test_accessibility_tts_and_alerts():
    tts = TTSSynthesizer(sample_rate=16000)
    ssml = tts.to_ssml("Door ahead at 12 o'clock", rate="fast")
    assert "<speak" in ssml
    assert 'rate="fast"' in ssml
    assert "Door ahead at 12 o'clock" in ssml

    wav_bytes = tts.synthesize_wav_tone("Alert obstacle")
    assert len(wav_bytes) > 1000

    # Verify wave header validity
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() > 0


def test_accessibility_scene_narration():
    preset = AccessibilityPreset()
    report = preset.narrate_scene(
        media_uri="user_glasses_cam.jpg",
        user_heading_deg=0.0,
        detections=[
            {"label": "tripod cord", "box": [0.48, 0.75, 0.56, 0.95], "depth_meters": 1.0, "is_hazard": True},
            {"label": "exit door", "box": [0.65, 0.20, 0.90, 0.85], "depth_meters": 3.2, "is_landmark": True},
        ],
        ocr_texts=["FIRE EXIT"],
    )

    assert isinstance(report, AccessibilityNarrationReport)
    assert len(report.obstacle_alerts) >= 1
    assert report.obstacle_alerts[0].urgency == "critical"
    assert "FIRE EXIT" in report.speech_transcript
    assert report.audio_wav_bytes_length > 0


# ---------------------------------------------------------------------------
# Test Hazard & Safety Preset
# ---------------------------------------------------------------------------

def test_hazard_safety_ppe_audit():
    preset = HazardSafetyPreset()

    workers = [
        {"worker_id": "W1", "box": [0.1, 0.2, 0.3, 0.8], "ppe": {"hard_hat": True, "high_vis_vest": True, "safety_boots": True}},
        {"worker_id": "W2", "box": [0.4, 0.2, 0.6, 0.8], "ppe": {"hard_hat": False, "high_vis_vest": True, "safety_boots": True}},
        {"worker_id": "W3", "box": [0.7, 0.2, 0.9, 0.8], "ppe": {"hard_hat": False, "high_vis_vest": False, "safety_boots": False}},
    ]

    audits, rate = preset.audit_ppe(workers, required_ppe=["hard_hat", "high_vis_vest", "safety_boots"])
    assert len(audits) == 3
    assert audits[0].is_fully_compliant is True
    assert audits[1].is_fully_compliant is False
    assert audits[1].missing_gear == ["hard_hat"]
    assert len(audits[2].missing_gear) == 3
    assert rate == 33.3  # 1 out of 3 compliant


def test_hazard_safety_3d_proximity_risk():
    preset = HazardSafetyPreset()

    # Worker close to active forklift
    workers = [
        preset.audit_ppe([{"worker_id": "W1", "box": [0.5, 0.2, 0.6, 0.8]}])[0][0]
    ]
    machinery = [
        {"name": "forklift", "box": [0.52, 0.2, 0.7, 0.8], "depth_meters": 3.5, "is_active": True}
    ]

    risk_high = preset.compute_3d_proximity_risk(workers, machinery)
    assert risk_high >= 0.70  # Very close in 3D

    # Machinery inactive or far away
    machinery_far = [
        {"name": "crane", "box": [0.1, 0.1, 0.2, 0.3], "depth_meters": 12.0, "is_active": True}
    ]
    risk_low = preset.compute_3d_proximity_risk(workers, machinery_far)
    assert risk_low == 0.0


def test_hazard_safety_pdf_report_generation():
    preset = HazardSafetyPreset()

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "audit_report.pdf"
        report = preset.audit_scene(
            media_uri="warehouse_drone_01.jpg",
            site_id="CONSTRUCTION-BAY-4",
            export_pdf_path=str(pdf_path),
        )

        assert isinstance(report, SafetyAuditReport)
        assert report.overall_safety_rating in ["PASS", "WARNING", "CRITICAL"]
        assert report.pdf_report_path is not None
        assert pdf_path.exists()
        # Verify valid non-empty PDF file
        assert pdf_path.stat().st_size > 1000
        # Check PDF header signature
        with open(pdf_path, "rb") as f:
            header = f.read(5)
            assert header == b"%PDF-"
