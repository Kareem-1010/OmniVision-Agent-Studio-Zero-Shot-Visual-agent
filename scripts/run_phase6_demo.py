"""
Phase 6 Training and Self-Improvement Pipeline Demonstration Script.
Demonstrates:
1. Trajectory Harvesting & Reward Scoring from agent execution states.
2. SFT Dataset synthesis and DPO Contrastive Preference Pair extraction with JSONL export.
3. Planner Self-Improvement via Direct Preference Optimization (Bradley-Terry loss).
4. Student Object Detector Knowledge Distillation (Soft KL + Hard CE + Smooth L1).
5. SigLIP / CLIP Linear Probing with learnable temperature scaling & Expected Calibration Error (ECE).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import torch

# Add project root to path for direct invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.state import (
    ClaimAttribution,
    ClaimsVerificationResult,
    CritiqueVerdict,
    GroundedSynthesis,
    OmniVisionState,
    PlanStep,
    QueryAnalysis,
    TraceEvent,
    VisionProgramPlan,
)
from src.training import (
    AgentTrajectory,
    DatasetBuilder,
    DistillationLossOutput,
    DPOPreferencePair,
    LinearProbeHead,
    PlannerSelfImprovementTrainer,
    ProbeEpochMetrics,
    SFTSample,
    SigLIPLinearProbeTrainer,
    StudentDetectorDistiller,
    StudentDetectorNetwork,
    TrainingStepMetrics,
    compute_dpo_loss,
    compute_ece,
)


def _build_sample_agent_state(
    query: str,
    passed: bool = True,
    critique_score: float = 0.92,
    hallucination_rate: float = 0.0,
    cost: float = 0.015,
) -> OmniVisionState:
    """Helper to simulate an OmniVisionState with realistic trace, plan, and critique."""
    state = OmniVisionState(
        query=query,
        media_uri="warehouse_camera_stream_04.jpg",
        media_type="image",
        accumulated_cost_usd=cost,
        accumulated_latency_ms=115.0,
    )
    state.query_analysis = QueryAnalysis(
        original_query=query,
        modality="image",
        reasoning_type=["detection", "spatial_relation", "safety"],
        sub_goals=["Detect personnel", "Detect forklift", "Verify clearance boundary"],
        key_targets=["worker", "forklift"],
    )
    state.plan = VisionProgramPlan(
        strategy_rationale="decompose_and_verify_clearance",
        steps=[
            PlanStep(
                step_id="step_1",
                sub_goal="Detect personnel and equipment",
                tool_name="detector_ensemble",
                arguments={"classes": ["person", "forklift"], "confidence_threshold": 0.55},
                status="completed" if passed else "failed",
                cost_estimate_usd=0.008,
                latency_estimate_ms=65.0,
            ),
            PlanStep(
                step_id="step_2",
                sub_goal="Calculate pairwise 3D distance and zone containment",
                tool_name="metric_depth",
                arguments={"reference_box": [120, 80, 240, 320]},
                status="completed" if passed else "failed",
                cost_estimate_usd=0.007,
                latency_estimate_ms=50.0,
            ),
        ],
    )
    state.trace_log.append(
        TraceEvent(
            node_name="detector_ensemble",
            action="execute_subgoal",
            details={"detections": 2, "labels": ["worker", "forklift"]},
            cost_usd=0.008,
            latency_ms=65.0,
        )
    )
    state.critique = CritiqueVerdict(
        passed=passed,
        score=critique_score,
        should_replan=not passed,
        feedback="Verified 3D clearance satisfied." if passed else "Insufficient evidence; hallucinated presence.",
    )
    state.claims_verification = ClaimsVerificationResult(
        all_verified=passed,
        verified_claims=[
            ClaimAttribution(statement="Worker is 2.8m from forklift", supported=True, grounded_box=[120, 80, 240, 320])
        ] if passed else [],
        unsupported_claims=[
            ClaimAttribution(statement="Clearance is maintained", supported=False)
        ] if not passed else [],
        hallucination_rate=hallucination_rate,
    )
    state.final_synthesis = GroundedSynthesis(
        query=query,
        answer="The worker is located 2.8 meters away from the forklift, outside the active hazard corridor."
        if passed
        else "Worker appears in safe zone without confirmed distance.",
        calibrated_confidence=0.94 if passed else 0.45,
        citations=["step_1", "step_2"] if passed else [],
    )
    return state


def run_phase6_demo():
    print("=" * 78)
    print(" OmniVision Agent - Phase 6 Training & Self-Improvement Pipeline Demo")
    print("=" * 78)

    # ---------------------------------------------------------
    # 1. Trajectory Harvesting & Reward Scoring
    # ---------------------------------------------------------
    print("\n[1] Trajectory Harvesting & Multi-Factor Reward Scoring:")
    builder = DatasetBuilder(reward_pass_threshold=0.75)

    q1 = "Is the warehouse worker within the forklift hazard perimeter?"
    state_chosen = _build_sample_agent_state(q1, passed=True, critique_score=0.95, hallucination_rate=0.0, cost=0.015)
    state_rejected = _build_sample_agent_state(q1, passed=False, critique_score=0.35, hallucination_rate=0.5, cost=0.040)

    traj_win = builder.harvest_from_state(state_chosen)
    traj_loss = builder.harvest_from_state(state_rejected)

    print(f"  Harvested Chosen Trajectory ID   : {traj_win.trajectory_id}")
    print(f"  Chosen Reward Score              : {traj_win.reward_score:.4f} (Critique: 0.95, Halluc: 0.0%)")
    print(f"  Harvested Rejected Trajectory ID : {traj_loss.trajectory_id}")
    print(f"  Rejected Reward Score            : {traj_loss.reward_score:.4f} (Critique: 0.35, Halluc: 50.0%)")
    print(f"  Trajectory Reward Margin         : {traj_win.reward_score - traj_loss.reward_score:.4f}")

    # ---------------------------------------------------------
    # 2. SFT & DPO Dataset Synthesis
    # ---------------------------------------------------------
    print("\n[2] SFT & DPO Dataset Synthesis & JSONL Export:")
    sft_samples = builder.build_sft_dataset(min_reward=0.75)
    dpo_pairs = builder.build_dpo_pairs(min_margin=0.20)

    print(f"  Synthesized SFT Samples          : {len(sft_samples)}")
    print(f"  Extracted DPO Preference Pairs   : {len(dpo_pairs)}")

    if sft_samples:
        print(f"  Sample SFT Prompt Instruction    : '{sft_samples[0].instruction}'")
        print(f"  Sample SFT Context Snippet       : {sft_samples[0].input_context[:60]}...")
    if dpo_pairs:
        pair = dpo_pairs[0]
        print(f"  DPO Pair ID                      : {pair.pair_id}")
        print(f"  DPO Prompt                       : '{pair.prompt}'")
        print(f"  Chosen vs Rejected Rewards       : {pair.chosen_reward:.4f} vs {pair.rejected_reward:.4f} (Margin: {pair.reward_margin:.4f})")

    # Export demo
    with tempfile.TemporaryDirectory() as tmpdir:
        sft_path = Path(tmpdir) / "sft_planner.jsonl"
        dpo_path = Path(tmpdir) / "dpo_planner.jsonl"
        builder.export_to_jsonl(sft_samples, str(sft_path))
        builder.export_to_jsonl(dpo_pairs, str(dpo_path))
        print(f"  Successfully exported to JSONL   : {sft_path.name} & {dpo_path.name}")

    # ---------------------------------------------------------
    # 3. Planner Self-Improvement via DPO
    # ---------------------------------------------------------
    print("\n[3] Planner Policy Optimization via Direct Preference Optimization (DPO):")
    trainer = PlannerSelfImprovementTrainer(beta=0.1, learning_rate=2e-3, feature_dim=32)

    # Train policy on preference pairs
    metrics_history = trainer.run_dpo_training_loop(dpo_pairs, num_epochs=5)
    for m in metrics_history:
        print(f"  Epoch {m.step + 1:02d} | Loss: {m.loss:.5f} | Chosen R: {m.chosen_reward:+.4f} | "
              f"Rejected R: {m.rejected_reward:+.4f} | Margin: {m.reward_margin:+.4f} | Win Rate: {m.accuracy_or_win_rate * 100:.1f}%")

    print(f"  Policy optimization complete. Win rate shifted to: {metrics_history[-1].accuracy_or_win_rate * 100:.1f}%")

    # ---------------------------------------------------------
    # 4. Student Detector Distillation
    # ---------------------------------------------------------
    print("\n[4] Knowledge Distillation to Lightweight Student Detector:")
    distiller = StudentDetectorDistiller(
        num_classes=4,
        temperature=3.0,
        alpha_hard=0.3,
        beta_bbox=1.5,
        learning_rate=1e-3,
    )

    torch.manual_seed(101)
    batch_size = 4
    dummy_imgs = torch.randn(batch_size, 3, 48, 48)
    # Teacher ensemble pseudo-labels (soft logits + calibrated bounding boxes)
    teacher_logits = torch.randn(batch_size, 4) * 2.0
    teacher_boxes = torch.rand(batch_size, 4)
    true_labels = torch.tensor([0, 1, 2, 3])

    print("  Distilling teacher ensemble (YOLO11 + RT-DETR + WBF) into student network...")
    for step in range(4):
        step_metrics = distiller.train_step(
            images=dummy_imgs,
            teacher_logits=teacher_logits,
            teacher_boxes=teacher_boxes,
            true_labels=true_labels,
        )
        print(f"  Step {step + 1} | Total Loss: {step_metrics.total_loss:.4f} "
              f"(Soft KL: {step_metrics.soft_teacher_loss:.4f}, Hard CE: {step_metrics.hard_label_loss:.4f}, "
              f"BBox L1: {step_metrics.box_regression_loss:.4f})")

    # Student model verification
    distiller.student.eval()
    with torch.no_grad():
        s_logits, s_boxes = distiller.student(dummy_imgs[:1])
    print(f"  Student output shapes: Logits {list(s_logits.shape)}, Bounding Boxes {list(s_boxes.shape)}")
    print(f"  Normalized Box prediction: [{', '.join(f'{coord:.3f}' for coord in s_boxes[0].tolist())}]")

    # ---------------------------------------------------------
    # 5. SigLIP / CLIP Linear Probing with Calibration (ECE)
    # ---------------------------------------------------------
    print("\n[5] SigLIP / CLIP Linear Probing & Calibration (ECE):")
    embed_dim = 64
    num_classes = 3
    classes = ["safe_pedestrian", "forklift_transit", "restricted_zone"]
    probe_trainer = SigLIPLinearProbeTrainer(
        embed_dim=embed_dim,
        num_classes=num_classes,
        class_names=classes,
        learning_rate=8e-3,
    )

    # Synthesize cluster embeddings
    torch.manual_seed(42)
    n_per_class = 25
    c0 = torch.randn(n_per_class, embed_dim) * 0.4 + torch.tensor([5.0] + [0.0] * (embed_dim - 1))
    c1 = torch.randn(n_per_class, embed_dim) * 0.4 + torch.tensor([0.0, 5.0] + [0.0] * (embed_dim - 2))
    c2 = torch.randn(n_per_class, embed_dim) * 0.4 + torch.tensor([0.0, 0.0, 5.0] + [0.0] * (embed_dim - 3))

    all_embs = torch.cat([c0, c1, c2], dim=0)
    all_targets = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class), torch.full((n_per_class,), 2)]).long()

    # Pre-training baseline evaluation
    pre_eval = probe_trainer.evaluate(all_embs, all_targets)
    print(f"  Untrained Probe Baseline       : Accuracy: {pre_eval['accuracy']*100:.1f}%, Loss: {pre_eval['loss']:.4f}, ECE: {pre_eval['ece']:.4f}")

    # Train probe
    probe_history = probe_trainer.fit(
        train_embeddings=all_embs,
        train_labels=all_targets,
        val_embeddings=all_embs,
        val_labels=all_targets,
        num_epochs=8,
        batch_size=16,
    )

    post_eval = probe_trainer.evaluate(all_embs, all_targets)
    temp_val = torch.exp(probe_trainer.model.log_temp).item()
    print(f"  Trained Probe Final Metrics    : Accuracy: {post_eval['accuracy']*100:.1f}%, Loss: {post_eval['loss']:.4f}, ECE: {post_eval['ece']:.4f}")
    print(f"  Calibrated Learned Temperature : {temp_val:.3f}")

    # Single-sample prediction
    preds, probs = probe_trainer.predict(all_embs[:1])
    pred_class = classes[preds[0].item()]
    print(f"  Sample Prediction              : Predicted Class: '{pred_class}' (Confidence: {probs[0].max().item()*100:.1f}%)")

    print("\n" + "=" * 78)
    print(" Phase 6 Demonstration Successfully Completed!")
    print("=" * 78)


if __name__ == "__main__":
    run_phase6_demo()
