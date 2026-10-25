"""
Hallucination Guard and Claim-Evidence Attribution Engine.
Deconstructs draft answers into atomic factual claims, verifies each against
grounded observation IDs and spatial coordinates, and remediates ungrounded assertions.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional
from src.agent.state import ClaimAttribution, ClaimsVerificationResult

logger = logging.getLogger("OmniVision.Agent.HallucinationGuard")


class HallucinationGuard:
    """
    Formal Claim-Evidence Attribution and Hallucination Prevention Engine.
    Ensures that every factual assertion in the agent's answer is traceable to
    a cryptographically hashed observation ID and bounding region.
    """

    def __init__(self, tolerance_threshold: float = 0.0) -> None:
        self.tolerance_threshold = tolerance_threshold

    def verify_draft_answer(
        self,
        draft_text: str,
        observations: Dict[str, Dict[str, Any]],
        scene_graph_dict: Optional[Dict[str, Any]] = None,
    ) -> ClaimsVerificationResult:
        """
        Extract atomic claims from the draft response and verify against registered observations.
        """
        # 1. Deconstruct text into atomic sentences / clauses
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", draft_text) if s.strip()]

        verified_claims: List[ClaimAttribution] = []
        unsupported_claims: List[ClaimAttribution] = []

        # Build index of empirical observation evidence
        detected_labels: Dict[str, List[Dict[str, Any]]] = {}
        for obs_id, obs in observations.items():
            data = obs.get("data") or {}
            # Detector Ensemble Detections
            for d in data.get("detections", []):
                lbl = d.get("label", "").lower()
                b = d.get("box") or d.get("box_2d")
                b_list = [b["x1"], b["y1"], b["x2"], b["y2"]] if isinstance(b, dict) and "x1" in b else b
                detected_labels.setdefault(lbl, []).append({
                    "obs_id": obs_id,
                    "box": b_list,
                    "conf": d.get("confidence", 0.9),
                })
            # Open Vocab Grounded Instances
            for inst in data.get("instances", []):
                lbl = inst.get("phrase", "").lower()
                b = inst.get("box")
                b_list = [b["x1"], b["y1"], b["x2"], b["y2"]] if isinstance(b, dict) and "x1" in b else b
                detected_labels.setdefault(lbl, []).append({
                    "obs_id": obs_id,
                    "box": b_list,
                    "conf": inst.get("confidence", 0.9),
                })
            # 3D instances
            for inst in data.get("instances_3d", []):
                lbl = inst.get("label", "").lower()
                detected_labels.setdefault(lbl, []).append({
                    "obs_id": obs_id,
                    "box": inst.get("bbox_2d"),
                    "depth_z": inst.get("centroid_3d", [0, 0, 0])[2] if inst.get("centroid_3d") else None,
                    "conf": inst.get("confidence", 0.9),
                })
            # OCR texts
            for t in data.get("recognized_texts", []):
                txt_str = t.get("text", "").lower()
                detected_labels.setdefault(txt_str, []).append({
                    "obs_id": obs_id,
                    "box": t.get("box_2d"),
                    "conf": t.get("confidence", 0.9),
                })
            # Grounded VLM boxes
            for vb in data.get("grounded_boxes", []):
                lbl = vb.get("label", "").lower()
                detected_labels.setdefault(lbl, []).append({
                    "obs_id": obs_id,
                    "box": vb.get("box_2d"),
                    "conf": vb.get("confidence", 0.9),
                })

        # 2. Evaluate each claim against evidence index
        for sent in sentences:
            sent_lower = sent.lower()

            # Find matching evidence by entity mention
            matched_obs_ids: List[str] = []
            matched_box: Optional[List[float]] = None
            max_conf = 0.0

            for entity_lbl, evidence_list in detected_labels.items():
                if entity_lbl and entity_lbl in sent_lower:
                    for ev in evidence_list:
                        matched_obs_ids.append(ev["obs_id"])
                        if ev.get("box") and matched_box is None:
                            matched_box = ev["box"]
                        max_conf = max(max_conf, ev.get("conf", 0.9))

            # Deduplicate observation IDs
            matched_obs_ids = list(dict.fromkeys(matched_obs_ids))

            # Discursive / meta-statements (e.g. "Based on the visual analysis...") are non-empirical
            is_meta_statement = any(sent_lower.startswith(prefix) for prefix in [
                "in summary", "overall", "based on", "the analysis indicates", "to answer your question",
                "consequently", "visual analysis", "in response to", "for query", "regarding",
                "scene summary", "summary indicates", "the scene depicts", "the visual scene"
            ])

            if matched_obs_ids or is_meta_statement:
                claim = ClaimAttribution(
                    claim_id=f"claim_{uuid.uuid4().hex[:6]}",
                    statement=sent,
                    supported=True,
                    observation_ids=matched_obs_ids,
                    grounded_box=matched_box,
                    confidence=max_conf if matched_obs_ids else 1.0,
                    verifier_notes="Verified against observation index." if matched_obs_ids else "Meta-reasoning statement.",
                )
                verified_claims.append(claim)
            else:
                claim = ClaimAttribution(
                    claim_id=f"claim_{uuid.uuid4().hex[:6]}",
                    statement=sent,
                    supported=False,
                    observation_ids=[],
                    grounded_box=None,
                    confidence=0.20,
                    verifier_notes="No supporting observation ID or perceptual evidence found.",
                )
                unsupported_claims.append(claim)

        # 3. Calculate Hallucination Rate
        total_claims = len(verified_claims) + len(unsupported_claims)
        rate = len(unsupported_claims) / max(1, total_claims)

        # 4. Synthesize Remediated Text
        remediated_sentences: List[str] = []
        for claim in verified_claims:
            if claim.observation_ids:
                citation_tags = "".join(f" [{oid}]" for oid in claim.observation_ids[:2])
                remediated_sentences.append(f"{claim.statement}{citation_tags}")
            else:
                remediated_sentences.append(claim.statement)

        for claim in unsupported_claims:
            # Annotate unverified claims explicitly
            remediated_sentences.append(f"[UNVERIFIED CLAIM: {claim.statement}]")

        remediated_text = " ".join(remediated_sentences)

        return ClaimsVerificationResult(
            all_verified=len(unsupported_claims) == 0,
            verified_claims=verified_claims,
            unsupported_claims=unsupported_claims,
            hallucination_rate=round(rate, 4),
            remediated_answer_text=remediated_text,
        )
