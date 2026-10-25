"""
Phase 7 Domain Presets and Demo Modes Demonstration Script.
Demonstrates:
1. Retail Analytics Mode:
   - Tripwire bidirectional footfall tracking & net store occupancy.
   - Queue length monitoring & Little's Law wait-time estimation.
   - Shelf gap & planogram out-of-stock (OOS) empty slot detection.
   - 2D spatial shopper dwell time heatmap generation.
2. Accessibility Assistant Mode:
   - Metric egocentric scene narration with 12-hour clock-face bearings.
   - Environmental OCR scene-text decoding.
   - Immediate walking path collision & tripping obstacle warnings.
   - Text-To-Speech (TTS) audio synthesis (WAV PCM) and SSML generation.
3. Hazard & Safety Mode:
   - Worker PPE compliance audit (hard hat, high-vis vest, safety footwear).
   - Environmental hazard detection (liquid spill, blocked fire exit).
   - 3D-aware physical proximity risk scoring between personnel and machinery.
   - Professional exportable PDF compliance report generation using ReportLab.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Add project root to path for direct invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.presets import (
    AccessibilityNarrationReport,
    AccessibilityPreset,
    HazardSafetyPreset,
    RetailKPIReport,
    RetailPreset,
    SafetyAuditReport,
    get_preset_registry,
)


def run_phase7_demo():
    print("=" * 80)
    print(" OmniVision Agent - Phase 7 Domain Presets & Vertical Demo Modes")
    print("=" * 80)

    registry = get_preset_registry()
    print(f"\nRegistered Vertical Presets: {', '.join(repr(p) for p in registry.list_presets())}")

    # =========================================================================
    # 1. Retail Analytics Mode
    # =========================================================================
    print("\n" + "-" * 80)
    print(" [1] RETAIL ANALYTICS PRESET (Footfall, Queues, Shelf Gaps, Dwell Heatmaps)")
    print("-" * 80)

    retail_preset: RetailPreset = registry.get("retail")  # type: ignore
    print(f"  System Instruction: {retail_preset.get_system_prompt()[:85]}...")
    print(f"  Enabled Perception Tools: {retail_preset.get_toolset()}")

    # Simulate realistic shopper trajectories
    shopper_tracks = [
        {"track_id": "sh_101", "trajectory": [[0.35, 0.95, 10.0], [0.35, 0.72, 18.0], [0.38, 0.45, 65.0]]},  # entered
        {"track_id": "sh_102", "trajectory": [[0.55, 0.90, 15.0], [0.55, 0.68, 22.0], [0.70, 0.50, 80.0]]},  # entered
        {"track_id": "sh_103", "trajectory": [[0.45, 0.60, 20.0], [0.45, 0.78, 35.0], [0.45, 0.92, 45.0]]},  # exited
        {"track_id": "sh_104", "trajectory": [[0.80, 0.70, 30.0], [0.82, 0.72, 60.0], [0.85, 0.75, 95.0]]},  # in queue
    ]
    # Shelf planogram data
    shelf_inventory = [
        {
            "shelf_id": "aisle_03_dairy",
            "slots": [
                {"is_occupied": True, "category": "organic_milk"},
                {"is_occupied": False, "category": "organic_milk", "width_ratio": 0.28, "box": [0.25, 0.35, 0.42, 0.58]},
                {"is_occupied": True, "category": "yogurt"},
                {"is_occupied": False, "category": "yogurt", "width_ratio": 0.40, "box": [0.60, 0.35, 0.85, 0.58]},
                {"is_occupied": True, "category": "cheese"},
            ],
        }
    ]

    retail_kpis: RetailKPIReport = retail_preset.analyze(
        media_uri="supermarket_camera_02.mp4",
        tracks=shopper_tracks,
        shelves=shelf_inventory,
        tripwire=[(0.2, 0.8), (0.8, 0.8)],
    )

    print(f"\n  Footfall Metrics:")
    print(f"    - Inbound Shoppers     : {retail_kpis.footfall_in}")
    print(f"    - Outbound Shoppers    : {retail_kpis.footfall_out}")
    print(f"    - Net Store Occupancy  : {retail_kpis.net_store_occupancy}")
    print(f"    - Average Dwell Time   : {retail_kpis.average_dwell_time_sec} seconds")

    print(f"\n  Checkout Queue Intelligence:")
    for q_id, q_stat in retail_kpis.queues.items():
        print(f"    - Zone '{q_id}': {q_stat.people_count} in line | Est. Wait: {q_stat.estimated_wait_time_sec}s | Congestion: {q_stat.congestion_level.upper()}")

    print(f"\n  Inventory & Shelf Out-Of-Stock (OOS):")
    print(f"    - Out-Of-Stock Rate    : {retail_kpis.out_of_stock_percentage}%")
    print(f"    - Detected Empty Gaps  : {len(retail_kpis.shelf_gaps)}")
    for g in retail_kpis.shelf_gaps:
        print(f"      * Slot: {g.shelf_id} | Gap Width: {g.gap_width_ratio*100:.1f}% | Category: {g.category_label} | Priority: {g.restock_priority.upper()}")

    print(f"  Dwell Heatmap Grid: {len(retail_kpis.dwell_heatmap_matrix)}x{len(retail_kpis.dwell_heatmap_matrix[0])} spatial accumulation matrix synthesized.")

    # =========================================================================
    # 2. Accessibility Assistant Mode
    # =========================================================================
    print("\n" + "-" * 80)
    print(" [2] ACCESSIBILITY ASSISTANT PRESET (3D Narration, OCR, Obstacles, TTS)")
    print("-" * 80)

    access_preset: AccessibilityPreset = registry.get("accessibility")  # type: ignore
    print(f"  System Instruction: {access_preset.get_system_prompt()[:85]}...")
    print(f"  Enabled Perception Tools: {access_preset.get_toolset()}")

    scene_detections = [
        {"label": "sliding glass door", "box": [0.45, 0.20, 0.55, 0.85], "depth_meters": 3.4, "is_landmark": True},
        {"label": "low curb step", "box": [0.42, 0.78, 0.58, 0.96], "depth_meters": 1.1, "is_hazard": True},
        {"label": "information kiosk", "box": [0.80, 0.35, 0.95, 0.75], "depth_meters": 2.8, "is_landmark": True},
        {"label": "bench", "box": [0.08, 0.45, 0.28, 0.78], "depth_meters": 2.2, "is_landmark": False},
    ]
    scene_texts = ["PLATFORM 2 - DOWNTOWN EXPRESS", "WATCH YOUR STEP"]

    access_report: AccessibilityNarrationReport = access_preset.narrate_scene(
        media_uri="user_smart_glasses_feed.jpg",
        user_heading_deg=0.0,
        detections=scene_detections,
        ocr_texts=scene_texts,
    )

    print(f"\n  Egocentric 3D Surroundings:")
    for obj in access_report.clockface_objects:
        landmark_tag = " [Landmark]" if obj.is_landmark else ""
        print(f"    - {obj.clock_face:>11} | {obj.distance_meters:.1f}m away | {obj.elevation_level:<12} | {obj.label}{landmark_tag}")

    print(f"\n  Immediate Obstacle Alerts:")
    for alt in access_report.obstacle_alerts:
        print(f"    - [{alt.urgency.upper()}] {alt.hazard_label} at {alt.direction} ({alt.distance_meters:.1f}m) -> {alt.guidance}")

    print(f"\n  Speech Transcript:")
    print(f'    "{access_report.speech_transcript}"')

    print(f"\n  Synthesized Audio:")
    print(f"    - Format: 16-bit PCM Mono WAV @ {access_report.audio_sample_rate_hz} Hz")
    print(f"    - WAV Audio Payload Size: {access_report.audio_wav_bytes_length} bytes")
    print(f"    - SSML Length: {len(access_report.ssml_markup)} characters")

    # =========================================================================
    # 3. Hazard & Safety Mode
    # =========================================================================
    print("\n" + "-" * 80)
    print(" [3] HAZARD & SAFETY PRESET (PPE Compliance, 3D Risk, Exportable PDF)")
    print("-" * 80)

    safety_preset: HazardSafetyPreset = registry.get("hazard_safety")  # type: ignore
    print(f"  System Instruction: {safety_preset.get_system_prompt()[:85]}...")
    print(f"  Enabled Perception Tools: {safety_preset.get_toolset()}")

    workers_data = [
        {"worker_id": "W-01", "box": [0.15, 0.25, 0.30, 0.85], "ppe": {"hard_hat": True, "high_vis_vest": True, "safety_boots": True}},
        {"worker_id": "W-02", "box": [0.38, 0.22, 0.52, 0.82], "ppe": {"hard_hat": False, "high_vis_vest": True, "safety_boots": True}},
        {"worker_id": "W-03", "box": [0.65, 0.30, 0.78, 0.88], "ppe": {"hard_hat": True, "high_vis_vest": False, "safety_boots": True}},
    ]
    hazards_data = [
        {
            "hazard_id": "HAZ-001",
            "hazard_type": "chemical_spill",
            "box": [0.30, 0.70, 0.50, 0.90],
            "depth_meters": 2.2,
            "severity": "warning",
            "remediation": "Deploy neutralizing powder, cordon off area.",
        },
        {
            "hazard_id": "HAZ-002",
            "hazard_type": "blocked_emergency_exit",
            "box": [0.82, 0.15, 0.98, 0.92],
            "depth_meters": 4.5,
            "severity": "critical",
            "remediation": "Remove stacked wooden pallets from emergency egress corridor immediately.",
        },
    ]
    machinery_data = [
        {"name": "forklift_alpha", "box": [0.40, 0.20, 0.65, 0.75], "depth_meters": 3.8, "is_active": True}
    ]

    pdf_out = Path("reports") / "safety_audit_report.pdf"
    safety_report: SafetyAuditReport = safety_preset.audit_scene(
        media_uri="warehouse_cctv_zone_b.mp4",
        site_id="LOGISTICS-HUB-NORTH",
        workers=workers_data,
        hazards=hazards_data,
        machinery=machinery_data,
        export_pdf_path=str(pdf_out),
    )

    print(f"\n  Workplace Safety Audit Summary:")
    print(f"    - Facility Site ID       : {safety_report.site_id}")
    print(f"    - Overall Safety Status  : {safety_report.overall_safety_rating}")
    print(f"    - PPE Compliance Rate    : {safety_report.ppe_compliance_rate}% ({safety_report.compliant_personnel_count}/{safety_report.personnel_count} compliant)")
    print(f"    - 3D Proximity Risk Score: {safety_report.max_3d_risk_score:.3f} / 1.000")

    print(f"\n  Worker PPE Violations:")
    for w in safety_report.workers:
        v_str = f"Missing: {', '.join(w.missing_gear)}" if w.missing_gear else "Fully Compliant"
        print(f"    - {w.worker_id}: {'[PASS]' if w.is_fully_compliant else '[VIOLATION]'} | {v_str}")

    print(f"\n  Active Environmental Hazards:")
    for h in safety_report.environmental_hazards:
        print(f"    - [{h.severity.upper()}] {h.hazard_type.replace('_', ' ').title()} @ {h.estimated_3d_distance_m:.1f}m -> {h.remediation_action}")

    print(f"\n  Exported PDF Audit Report:")
    print(f"    - PDF Report Path        : {safety_report.pdf_report_path}")
    print(f"    - File Size On Disk      : {Path(safety_report.pdf_report_path).stat().st_size if safety_report.pdf_report_path else 0} bytes")

    print("\n" + "=" * 80)
    print(" Phase 7 Demonstration Successfully Completed!")
    print("=" * 80)


if __name__ == "__main__":
    run_phase7_demo()
