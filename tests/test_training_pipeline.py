"""
Unit and Integration Tests for Phase 6 Training & Self-Improvement Pipeline.
Covers DatasetBuilder (SFT/DPO), Bradley-Terry DPO optimization,
Student Detector Distillation, and SigLIP Linear Probing with ECE.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest
import torch
import numpy as np

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


def _create_mock_state(
    query: str = "Is the red car parked near the curb?",
    critique_score: float = 0.9,
    hallucination_rate: float = 0.0,
    cost: float = 0.02,
) -> OmniVisionState:
    """Helper to construct a realistic completed OmniVisionState."""
    state = OmniVisionState(
        query=query,
        media_uri="tests/data/sample.jpg",
        media_type="image",
        accumulated_cost_usd=cost,
        accumulated_latency_ms=120.0,
    )
    state.query_analysis = QueryAnalysis(
        original_query=query,
        modality="image",
        reasoning_type=["detection", "spatial_relation"],
        sub_goals=["Detect red car", "Check curb proximity"],
        key_targets=["red car", "curb"],
    )
    state.plan = VisionProgramPlan(
        strategy_rationale="decompose_and_detect",
        steps=[
            PlanStep(
                step_id="step_1",
                sub_goal="Detect red car",
                tool_name="detector_ensemble",
                arguments={"confidence_threshold": 0.5},
                status="completed",
                cost_estimate_usd=0.01,
                latency_estimate_ms=85.0,
            )
        ],
    )
    state.trace_log.append(
        TraceEvent(
            node_name="detector_ensemble",
            action="executed",
            details={"summary": "Detected 1 car at [100, 150, 300, 400]"},
            cost_usd=0.01,
            latency_ms=85.0,
        )
    )
    state.critique = CritiqueVerdict(
        passed=True,
        score=critique_score,
        should_replan=False,
        feedback="Grounding verified",
    )
    state.claims_verification = ClaimsVerificationResult(
        all_verified=(hallucination_rate == 0.0),
        verified_claims=[
            ClaimAttribution(
                statement="Red car is present",
                supported=True,
                grounded_box=[100.0, 150.0, 300.0, 400.0],
            )
        ] if hallucination_rate == 0.0 else [],
        unsupported_claims=[
            ClaimAttribution(
                statement="Curb is present",
                supported=False,
            )
        ] if hallucination_rate > 0.0 else [],
        hallucination_rate=hallucination_rate,
    )
    state.final_synthesis = GroundedSynthesis(
        query=query,
        answer="Yes, the red car is parked adjacent to the curb.",
        calibrated_confidence=0.92,
        citations=["step_1"],
    )
    return state


# ---------------------------------------------------------------------------
# Test Dataset Builder
# ---------------------------------------------------------------------------

def test_dataset_builder_harvest_and_reward():
    builder = DatasetBuilder(reward_pass_threshold=0.7)

    # High quality state
    state_good = _create_mock_state(critique_score=0.95, hallucination_rate=0.0, cost=0.01)
    traj_good = builder.harvest_from_state(state_good)

    assert traj_good.query == state_good.query
    assert traj_good.reward_score >= 0.85
    assert len(traj_good.trace_events) == 1
    assert traj_good.final_synthesis["calibrated_confidence"] == 0.92

    # Poor quality state with hallucination
    state_bad = _create_mock_state(critique_score=0.4, hallucination_rate=0.5, cost=0.05)
    traj_bad = builder.harvest_from_state(state_bad)
    assert traj_bad.reward_score < traj_good.reward_score
    assert traj_bad.hallucination_rate == 0.5


def test_dataset_builder_sft_and_dpo_generation():
    builder = DatasetBuilder(reward_pass_threshold=0.70)

    # Add two states for the same query to create a contrast pair
    good_state = _create_mock_state(query="Find the dog", critique_score=0.9, hallucination_rate=0.0)
    poor_state = _create_mock_state(query="Find the dog", critique_score=0.3, hallucination_rate=0.6)

    builder.harvest_from_state(good_state)
    builder.harvest_from_state(poor_state)

    # 1. SFT generation
    sft_samples = builder.build_sft_dataset(min_reward=0.70)
    assert len(sft_samples) == 1
    assert "Find the dog" in sft_samples[0].input_context
    assert "decompose_and_detect" in sft_samples[0].output_response

    # 2. DPO pair generation
    dpo_pairs = builder.build_dpo_pairs(min_margin=0.25)
    assert len(dpo_pairs) == 1
    pair = dpo_pairs[0]
    assert pair.chosen_reward > pair.rejected_reward
    assert pair.reward_margin >= 0.25
    assert "Find the dog" in pair.prompt

    # 3. Export to JSONL
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "dpo_test.jsonl"
        count = builder.export_to_jsonl(dpo_pairs, str(jsonl_path))
        assert count == 1
        assert jsonl_path.exists()
        line = jsonl_path.read_text(encoding="utf-8").strip()
        data = json.loads(line)
        assert data["pair_id"] == pair.pair_id
        assert data["reward_margin"] == pair.reward_margin


# ---------------------------------------------------------------------------
# Test DPO Loss & Planner Policy Trainer
# ---------------------------------------------------------------------------

def test_compute_dpo_loss_mechanics():
    # Construct synthetic logps where policy prefers chosen over rejected
    policy_chosen = torch.tensor([1.5, 2.0])
    policy_rejected = torch.tensor([-1.0, -0.5])
    ref_chosen = torch.tensor([0.0, 0.0])
    ref_rejected = torch.tensor([0.0, 0.0])

    loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
        policy_chosen_logps=policy_chosen,
        policy_rejected_logps=policy_rejected,
        reference_chosen_logps=ref_chosen,
        reference_rejected_logps=ref_rejected,
        beta=0.1,
    )

    assert loss.dim() == 0  # scalar
    assert loss.item() > 0.0
    assert torch.all(chosen_rewards > rejected_rewards)
    assert (chosen_rewards - rejected_rewards).mean().item() > 0.0


def test_planner_dpo_trainer_optimization():
    trainer = PlannerSelfImprovementTrainer(beta=0.1, learning_rate=1e-3, feature_dim=16)

    # Create dummy DPO pairs
    dummy_pairs = [
        DPOPreferencePair(
            prompt="Plan query 1",
            chosen='{"action": "detect"}',
            rejected='{"action": "guess"}',
            chosen_reward=0.9,
            rejected_reward=0.2,
            reward_margin=0.7,
        )
        for _ in range(4)
    ]

    metrics = trainer.run_dpo_training_loop(dummy_pairs, num_epochs=4)
    assert len(metrics) == 4
    for m in metrics:
        assert isinstance(m, TrainingStepMetrics)
        assert m.loss >= 0.0
        assert 0.0 <= m.accuracy_or_win_rate <= 1.0

    # Policy weights should have updated
    history = trainer.get_metrics_history()
    assert len(history) == 4


# ---------------------------------------------------------------------------
# Test Student Detector Distillation
# ---------------------------------------------------------------------------

def test_student_detector_network_shapes():
    model = StudentDetectorNetwork(in_channels=3, num_classes=5, base_dim=16)
    x = torch.randn(2, 3, 64, 64)
    logits, boxes = model(x)

    assert logits.shape == (2, 5)
    assert boxes.shape == (2, 4)
    # Boxes bounded in [0, 1] due to Sigmoid
    assert torch.all(boxes >= 0.0) and torch.all(boxes <= 1.0)


def test_student_detector_distiller_step():
    distiller = StudentDetectorDistiller(
        num_classes=4,
        temperature=2.0,
        alpha_hard=0.4,
        beta_bbox=1.0,
        learning_rate=1e-3,
    )

    batch_size = 4
    images = torch.randn(batch_size, 3, 32, 32)
    teacher_logits = torch.randn(batch_size, 4)
    teacher_boxes = torch.rand(batch_size, 4)
    true_labels = torch.randint(0, 4, (batch_size,))

    metrics = distiller.train_step(
        images=images,
        teacher_logits=teacher_logits,
        teacher_boxes=teacher_boxes,
        true_labels=true_labels,
    )

    assert isinstance(metrics, DistillationLossOutput)
    assert metrics.total_loss > 0.0
    assert metrics.hard_label_loss >= 0.0
    assert metrics.soft_teacher_loss >= 0.0
    assert metrics.box_regression_loss >= 0.0


# ---------------------------------------------------------------------------
# Test Linear Probe & ECE Calibration
# ---------------------------------------------------------------------------

def test_compute_ece():
    # Perfectly calibrated confident predictions
    probs_perfect = torch.tensor([
        [0.9, 0.1],
        [0.85, 0.15],
        [0.95, 0.05],
    ])
    targets_perfect = torch.tensor([0, 0, 0])
    ece_good = compute_ece(probs_perfect, targets_perfect, n_bins=5)
    assert 0.0 <= ece_good <= 0.2

    # Completely wrong confident predictions (badly calibrated)
    targets_wrong = torch.tensor([1, 1, 1])
    ece_bad = compute_ece(probs_perfect, targets_wrong, n_bins=5)
    assert ece_bad > ece_good


def test_siglip_linear_probe_trainer():
    embed_dim = 32
    num_classes = 3
    trainer = SigLIPLinearProbeTrainer(embed_dim=embed_dim, num_classes=num_classes, learning_rate=5e-3)

    # Synthetic normalized embeddings for 3 distinct clusters
    torch.manual_seed(42)
    c0 = torch.randn(20, embed_dim) * 0.3 + torch.tensor([4.0] + [0.0] * (embed_dim - 1))
    c1 = torch.randn(20, embed_dim) * 0.3 + torch.tensor([0.0, 4.0] + [0.0] * (embed_dim - 2))
    c2 = torch.randn(20, embed_dim) * 0.3 + torch.tensor([0.0, 0.0, 4.0] + [0.0] * (embed_dim - 3))

    train_x = torch.cat([c0, c1, c2], dim=0)
    train_y = torch.cat([torch.zeros(20), torch.ones(20), torch.full((20,), 2)]).long()

    val_x = train_x[:15]
    val_y = train_y[:15]

    history = trainer.fit(
        train_embeddings=train_x,
        train_labels=train_y,
        val_embeddings=val_x,
        val_labels=val_y,
        num_epochs=8,
        batch_size=16,
    )

    assert len(history) == 8
    assert history[-1].train_accuracy >= 0.70
    assert history[-1].expected_calibration_error is not None

    # Inference test
    preds, probs = trainer.predict(val_x)
    assert preds.shape == (15,)
    assert probs.shape == (15, num_classes)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(15), atol=1e-4)

    # Evaluation test
    eval_res = trainer.evaluate(val_x, val_y)
    assert "accuracy" in eval_res
    assert "loss" in eval_res
    assert "ece" in eval_res
    assert eval_res["accuracy"] >= 0.60
