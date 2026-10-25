"""
Phase 5 Memory and Long Video Reasoning End-to-End Demonstration Script.
Demonstrates:
1. Hierarchical Episodic Vector Memory indexing (Frames -> Clips -> Events -> Episode).
2. Adaptive Content-Aware Keyframe Sampling across long video.
3. Temporal Event Boundary Detection fusing tripwires, graph diffs, and kinematics.
4. Long-Video Temporal QA Engine (localization, causality, recurrence, summary).
5. Real-Time Streaming Ingestion & Sliding Window buffer with anomaly alerts.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for direct invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.geometry.relations import LineCrossingEvent
from src.memory.episodic import (
    ClipNode,
    EpisodeNode,
    EpisodicVectorMemory,
    EventNode,
    FrameNode,
)
from src.memory.scene_graph import GraphDiffEvent
from src.video.events import SemanticEvent, TemporalEventSegmenter
from src.video.sampler import AdaptiveKeyframeSampler
from src.video.streaming import StreamAlert, StreamingVideoProcessor
from src.video.temporal_qa import TemporalQAEngine


def run_phase5_demo():
    print("=" * 75)
    print(" OmniVision Agent - Phase 5 Memory & Long Video Reasoning Demonstration")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Hierarchical Episodic Vector Memory
    # ---------------------------------------------------------
    print("\n[1] Hierarchical Episodic Vector Memory Indexing:")
    mem = EpisodicVectorMemory(embedding_dim=4)

    # Ingest frames spanning a simulated 60-second video
    for i in range(12):
        t_sec = i * 5.0
        # Simulated embedding shifting over time
        emb = [round(float(i) / 12.0, 3), 0.5, 0.2, 0.1]
        ents = ["worker"] if i < 6 else ["worker", "forklift"]
        if i == 8:
            ents.append("pallet")

        mem.add_frame(
            FrameNode(
                frame_id=f"frame_{i*150:05d}",
                frame_index=i * 150,
                timestamp=t_sec,
                image_uri=f"cctv_frame_{i:03d}.jpg",
                embedding=emb,
                detected_entities=ents,
                scene_summary=f"Warehouse scene at {t_sec}s with {', '.join(ents)}.",
            )
        )

    # Consolidate frames into clips
    clips = mem.consolidate_clips_from_frames(frames_per_clip=4)
    print(f"  Ingested 12 frames across 60.0s.")
    print(f"  Consolidated into {len(clips)} canonical ClipNodes:")
    for c in clips:
        print(f"    - [{c.clip_id}] {c.start_time}s - {c.end_time}s: {c.activity_summary} (entities: {c.prominent_entities})")

    # Vector similarity search over episodic memory
    query_vec = [0.75, 0.5, 0.2, 0.1]  # points to later half
    search_results = mem.search_frames(query_embedding=query_vec, top_k=2)
    print(f"\n  Vector search for embedding {query_vec}:")
    for fn, sim in search_results:
        print(f"    -> Matched '{fn.frame_id}' at {fn.timestamp}s (cosine sim: {sim:.4f}, entities: {fn.detected_entities})")

    # ---------------------------------------------------------
    # 2. Adaptive Content-Aware Keyframe Sampler
    # ---------------------------------------------------------
    print("\n[2] Adaptive Content-Aware Keyframe Sampler:")
    sampler = AdaptiveKeyframeSampler(
        motion_threshold=0.25,
        semantic_threshold=0.30,
        min_temporal_gap_frames=10,
    )
    raw_frame_uris = [f"video_raw_frame_{idx:04d}.jpg" for idx in range(120)]
    keyframes = sampler.sample_keyframes(
        frame_uris=raw_frame_uris,
        fps=30.0,
        frame_budget=6,
    )
    print(f"  Sampled {len(keyframes)} salient keyframes from 120 raw frames (budget: 6):")
    for kf in keyframes:
        print(f"    - Frame {kf.frame_index:03d} ({kf.timestamp:.2f}s): score={kf.salience_score:.2f}, reason='{kf.selection_reason}'")

    # ---------------------------------------------------------
    # 3. Temporal Event Detection & Boundary Segmentation
    # ---------------------------------------------------------
    print("\n[3] Temporal Event Detection & Boundary Segmentation:")
    segmenter = TemporalEventSegmenter(fps=30.0)

    # Simulate multi-source perceptual inputs
    line_cross = LineCrossingEvent(
        track_id="worker_42",
        line_name="loading_bay_perimeter",
        frame_index=90,
        crossing_direction="entering",
        intersection_point=(250.0, 300.0),
    )
    graph_diff = GraphDiffEvent(
        event_type="RELATION_FORMED",
        subject_id="forklift_01",
        object_id="pallet_03",
        frame_index=240,
        details={"relation_type": "supports"},
    )
    fast_trajectory = {
        "track_id": "forklift_01",
        "frames_seen": [360, 361, 362, 363, 364],
        "trajectory": [(100, 100), (120, 100), (180, 100), (220, 100), (240, 100)],
    }

    segmented_events = segmenter.detect_events_from_tracks_and_diffs(
        tracks=[fast_trajectory],
        graph_diffs=[graph_diff],
        line_crossings=[line_cross],
    )
    print(f"  Segmented {len(segmented_events)} distinct physical events:")
    for ev in segmented_events:
        print(f"    - [{ev.event_type}] {ev.start_time_sec:.1f}s - {ev.end_time_sec:.1f}s (peak: {ev.peak_time_sec:.1f}s): {ev.description}")

    # ---------------------------------------------------------
    # 4. Long-Video Temporal QA Engine
    # ---------------------------------------------------------
    print("\n[4] Long-Video Temporal QA Engine:")
    qa_engine = TemporalQAEngine(memory=mem, fps=30.0)

    # Q1: Localization
    q1 = "When did the worker enter the loading bay perimeter?"
    ans1 = qa_engine.answer_query(q1, segmented_events=segmented_events)
    print(f"  Query (Localization): '{q1}'")
    print(f"  -> Answer: {ans1.answer}")
    print(f"  -> Grounded Interval: [{ans1.start_time_sec}s, {ans1.end_time_sec}s] (Confidence: {ans1.confidence*100:.0f}%)")

    # Q2: Causality
    q2 = "What happened after the worker entered the perimeter?"
    ans2 = qa_engine.answer_query(q2, segmented_events=segmented_events)
    print(f"\n  Query (Causality): '{q2}'")
    print(f"  -> Answer: {ans2.answer}")

    # Q3: Recurrence
    q3 = "How many times was rapid movement detected?"
    ans3 = qa_engine.answer_query(q3, segmented_events=segmented_events)
    print(f"\n  Query (Recurrence): '{q3}'")
    print(f"  -> Answer: {ans3.answer}")

    # ---------------------------------------------------------
    # 5. Real-Time Streaming Ingestion & Sliding Window
    # ---------------------------------------------------------
    print("\n[5] Real-Time Streaming Video Ingestion & Sliding Window:")
    alerts_captured: list[StreamAlert] = []

    def on_alert(alert: StreamAlert):
        alerts_captured.append(alert)

    streaming_processor = StreamingVideoProcessor(
        window_size_frames=6,
        fps=10.0,
        alert_callback=on_alert,
    )

    # Stream 10 frames
    print("  Ingesting simulated live camera feed (window size: 6 frames)...")
    for f in range(10):
        t = round(f * 0.1, 2)
        # Check PPE and falls on frame 5
        check_safety = (f == 5)
        emitted = streaming_processor.ingest_frame(
            image_uri=f"live_feed_{f:03d}.jpg",
            timestamp=t,
            check_ppe_and_falls=check_safety,
        )

    buf = streaming_processor.get_window_frames()
    print(f"  Sliding window buffer size: {len(buf)} frames (indices: {[b.frame_index for b in buf]})")
    print(f"  Total hazard alerts emitted: {len(streaming_processor.get_alert_history())}")
    for alt in streaming_processor.get_alert_history():
        print(f"    ! [{alt.severity.upper()}] {alt.alert_type} at {alt.timestamp:.2f}s: {alt.description}")

    print("\n" + "=" * 75)
    print(" [OK] Phase 5 Memory & Long Video Demonstration Completed Successfully!")
    print("=" * 75)


if __name__ == "__main__":
    run_phase5_demo()
