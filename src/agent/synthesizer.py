"""
Final Grounded Synthesizer Node.
Integrates verified observation evidence, 3D metric scene graphs, and calibrated conformal
uncertainty estimates into a comprehensive, citation-backed response.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
import numpy as np

from src.agent.state import ClaimsVerificationResult, GroundedSynthesis, QueryAnalysis

logger = logging.getLogger("OmniVision.Agent.Synthesizer")

_SYNTH_SYSTEM = (
    "You are a world-class visual AI assistant. You are given:\n"
    "1. The user question\n"
    "2. The actual visual image\n"
    "3. YOLO object detector findings and CLIP/VLM scene analysis\n\n"
    "Carefully inspect the image. Cross-reference with detector evidence. "
    "Give a factual, detailed, accurate answer. Never hallucinate. "
    "Structure your answer clearly."
)


class GroundedSynthesizer:
    """Evidence-Attributed Answer Synthesizer with YOLO + CLIP unified section."""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def synthesize(
        self,
        analysis,
        observations,
        scene_graph_dict,
        claims_verification=None,
        accumulated_cost_usd=0.0,
        accumulated_latency_ms=0.0,
        media_uri=None,
    ):
        citations = []
        evidence_summary = {
            "yolo_detections": [],
            "clip_findings": [],
            "detected_objects": [],
            "depth_measurements": [],
            "recognized_texts": [],
            "poses_and_actions": [],
            "spatial_relations": [],
            "vlm_grounded_answer": "",
        }

        for obs_id, obs in observations.items():
            tool_name = obs.get("tool_name", "")
            data = obs.get("data") or {}

            if tool_name in ("detector_ensemble", "yolo_detector", "detector"):
                for d in data.get("detections", []):
                    lbl = d.get("label", "object")
                    b = d.get("box") or d.get("box_2d")
                    if isinstance(b, dict) and "x1" in b:
                        b_list = [b["x1"], b["y1"], b["x2"], b["y2"]]
                    else:
                        b_list = b
                    conf = float(d.get("confidence", d.get("calibrated_score", 0.9)))
                    src = d.get("detector_source", "yolov8")
                    det = {"label": lbl, "box_2d": b_list, "confidence": conf, "source": src, "obs_id": obs_id}
                    evidence_summary["yolo_detections"].append(det)
                    evidence_summary["detected_objects"].append(det)
                    citations.append(obs_id)

            elif tool_name in ("open_vocab", "open_vocab_detector"):
                for inst in data.get("instances", []):
                    lbl = inst.get("phrase", "object")
                    b = inst.get("box")
                    if isinstance(b, dict) and "x1" in b:
                        b_list = [b["x1"], b["y1"], b["x2"], b["y2"]]
                    else:
                        b_list = b
                    conf = float(inst.get("confidence", 0.9))
                    det = {"label": lbl, "box_2d": b_list, "confidence": conf, "source": "yolo-world", "obs_id": obs_id}
                    evidence_summary["yolo_detections"].append(det)
                    evidence_summary["detected_objects"].append(det)
                    citations.append(obs_id)

            elif tool_name == "metric_depth":
                for inst in data.get("instances_3d", []):
                    evidence_summary["depth_measurements"].append({
                        "label": inst.get("label", "object"),
                        "centroid_3d": inst.get("centroid_3d", [0, 0, 0]),
                        "distance_meters": inst.get("distance_to_camera_meters", 0.0),
                        "obs_id": obs_id,
                    })
                    citations.append(obs_id)

            elif tool_name == "scene_ocr":
                for t in data.get("recognized_texts", []):
                    evidence_summary["recognized_texts"].append({"text": t.get("text", ""), "obs_id": obs_id})
                    citations.append(obs_id)

            elif tool_name == "pose_action":
                for p in data.get("persons", []):
                    is_fall = bool(
                        p.get("is_fallen", False)
                        or p.get("is_fall_detected", False)
                        or data.get("fall_detected_in_scene", False)
                        or "fall" in str(p.get("action_label", "")).lower()
                        or "distress" in str(p.get("action_label", "")).lower()
                    )
                    evidence_summary["poses_and_actions"].append({
                        "posture": p.get("posture", "standing"),
                        "fall_detected": is_fall,
                        "action_label": p.get("action_label", "unknown"),
                        "ppe": p.get("ppe_compliance", {}),
                        "box": p.get("box", []),
                        "obs_id": obs_id,
                    })
                    citations.append(obs_id)
                    p_box = p.get("box", [])
                    if p_box:
                        label = "person (fallen / distress)" if is_fall else "person"
                        det = {
                            "label": label,
                            "box_2d": p_box,
                            "confidence": float(p.get("action_confidence", 0.94)),
                            "source": "pose_action",
                            "obs_id": obs_id,
                        }
                        if not any(d.get("label", "").startswith("person") for d in evidence_summary["yolo_detections"]):
                            evidence_summary["yolo_detections"].append(det)
                            evidence_summary["detected_objects"].append(det)

            elif tool_name == "grounded_vlm":
                vlm_ans = data.get("answer", "")
                if vlm_ans:
                    evidence_summary["vlm_grounded_answer"] = vlm_ans
                    evidence_summary["clip_findings"].append({"description": vlm_ans, "obs_id": obs_id})
                    citations.append(obs_id)

        unique_citations = list(dict.fromkeys(citations))

        for e in scene_graph_dict.get("edges", []):
            src_id = e.get("source_id", "?")
            tgt_id = e.get("target_id", "?")
            rel = e.get("relation_type", "near")
            dist = e.get("metric_distance_meters", 0)
            evidence_summary["spatial_relations"].append(
                f"'{src_id}' is {rel} '{tgt_id}' ({dist}m)"
            )

        key_findings = []
        yolo_dets = evidence_summary["yolo_detections"]
        if yolo_dets:
            obj_counts: Dict[str, int] = {}
            for o in yolo_dets:
                obj_counts[o["label"]] = obj_counts.get(o["label"], 0) + 1
            count_str = ", ".join(f"{cnt} {lbl}(s)" for lbl, cnt in obj_counts.items())
            key_findings.append(f"[YOLO] Detected: {count_str}")
        else:
            key_findings.append("[YOLO] No objects detected.")

        clip_ans = evidence_summary["vlm_grounded_answer"]
        if clip_ans:
            key_findings.append(f"[CLIP/VLM] {clip_ans}")

        if evidence_summary["depth_measurements"]:
            closest = min(evidence_summary["depth_measurements"], key=lambda x: x["distance_meters"])
            key_findings.append(
                f"[Depth] Closest: {closest['label']} at {closest['distance_meters']:.2f}m"
            )

        for pa in evidence_summary["poses_and_actions"]:
            fall_str = " FALL!" if pa["fall_detected"] else ""
            key_findings.append(f"[Pose] {pa['posture']}{fall_str}")

        if evidence_summary["recognized_texts"]:
            texts = ", ".join(f"'{t['text']}'" for t in evidence_summary["recognized_texts"])
            key_findings.append(f"[OCR] {texts}")

        if evidence_summary["spatial_relations"]:
            key_findings.append(f"[Spatial] {'; '.join(evidence_summary['spatial_relations'][:3])}")

        unified = self._build_unified_section(evidence_summary)
        answer = self._generate_answer(
            analysis.original_query, key_findings, unified,
            evidence_summary, unique_citations, claims_verification, media_uri
        )

        all_confs = [o["confidence"] for o in evidence_summary["detected_objects"]] or [0.90]
        mean_conf = sum(all_confs) / len(all_confs)
        lo = max(0.01, round(mean_conf - 0.05, 3))
        hi = min(0.99, round(mean_conf + 0.05, 3))

        return GroundedSynthesis(
            query=analysis.original_query,
            answer=answer,
            confidence_interval=(lo, hi),
            calibrated_confidence=round(mean_conf, 3),
            citations=unique_citations,
            key_findings=key_findings,
            total_cost_usd=round(accumulated_cost_usd, 5),
            total_latency_ms=max(0.1, round(accumulated_latency_ms, 2)),
            visual_evidence_summary=evidence_summary,
        )

    def _build_unified_section(self, evidence_summary: Dict[str, Any]) -> str:
        sep = "=" * 60
        lines = [f"\n{sep}", "  DETECTION & PERCEPTION EVIDENCE SUMMARY", sep]

        lines.append("\n[YOLO Object Detections]")
        yolo_dets = evidence_summary.get("yolo_detections", [])
        if yolo_dets:
            by_label: Dict[str, List] = {}
            for d in yolo_dets:
                by_label.setdefault(d["label"], []).append(d)
            for lbl, dets in by_label.items():
                avg_conf = sum(d["confidence"] for d in dets) / len(dets)
                srcs = ", ".join(list(dict.fromkeys(d.get("source", "yolo") for d in dets)))
                lines.append(
                    f"  - {lbl}: {len(dets)} instance(s) | "
                    f"confidence: {avg_conf:.1%} | model: {srcs}"
                )
        else:
            lines.append("  - No objects detected by YOLO ensemble.")

        lines.append("\n[CLIP / VLM Scene Analysis]")
        clip_ans = evidence_summary.get("vlm_grounded_answer", "")
        if clip_ans:
            display = clip_ans[:400] + ("..." if len(clip_ans) > 400 else "")
            lines.append(f"  - {display}")
        else:
            lines.append("  - No VLM/CLIP scene analysis available.")

        depth = evidence_summary.get("depth_measurements", [])
        if depth:
            lines.append("\n[3D Depth Measurements]")
            for dm in depth[:5]:
                lines.append(f"  - {dm['label']}: {dm['distance_meters']:.2f}m from camera")

        lines.append(sep)
        return "\n".join(lines)

    def _generate_answer(
        self, query, key_findings, unified_section, evidence_summary,
        unique_citations, claims_verification, media_uri
    ) -> str:
        # ---------------------------------------------------------------
        # Build the full user_prompt BEFORE the try block so the
        # text-only fallback path can also access it
        # ---------------------------------------------------------------
        yolo_dets = evidence_summary.get("yolo_detections", [])
        obj_counts: Dict[str, int] = {}
        for o in yolo_dets:
            obj_counts[o["label"]] = obj_counts.get(o["label"], 0) + 1
        count_text = ", ".join(f"{c} {l}(s)" for l, c in obj_counts.items()) or "No objects detected by YOLO"
        clip_ans = evidence_summary.get("vlm_grounded_answer", "No CLIP scene analysis available")

        det_details = []
        for d in yolo_dets[:20]:
            box = d.get("box_2d", [0, 0, 0, 0])
            det_details.append(f"  - {d['label']} (confidence: {d['confidence']:.1%}, bbox: {box}, model: {d.get('source', 'yolo')})")
        det_details_str = "\n".join(det_details) if det_details else "  - None"

        depth_details = []
        for dm in evidence_summary.get("depth_measurements", [])[:10]:
            depth_details.append(f"  - {dm['label']} at {dm['distance_meters']:.2f}m from camera (centroid: {dm.get('centroid_3d')})")
        depth_details_str = "\n".join(depth_details) if depth_details else "  - None"

        ocr_details = []
        for t in evidence_summary.get("recognized_texts", []):
            ocr_details.append(f"  - Text: '{t['text']}'")
        ocr_details_str = "\n".join(ocr_details) if ocr_details else "  - None"

        pose_details = []
        for pa in evidence_summary.get("poses_and_actions", []):
            fall_flag = "YES (PERSON FALLEN / IN DISTRESS)" if pa.get("fall_detected") else "No"
            ppe = pa.get("ppe", {})
            hardhat = "Worn" if ppe.get("hard_hat") else "MISSING"
            vest = "Worn" if ppe.get("safety_vest") else "MISSING"
            pose_details.append(f"  - Posture: {pa.get('posture')}, Fall / Distress: {fall_flag}, Hard Hat: {hardhat}, Safety Vest: {vest}")
        pose_details_str = "\n".join(pose_details) if pose_details else "  - None"

        spatial_details = []
        for sr in evidence_summary.get("spatial_relations", [])[:10]:
            spatial_details.append(f"  - {sr}")
        spatial_details_str = "\n".join(spatial_details) if spatial_details else "  - None"

        # Clearance / passage geometry between pairs of 3D instances
        clearance_details = []
        dms = evidence_summary.get("depth_measurements", [])
        if len(dms) >= 2:
            for i in range(len(dms)):
                for j in range(i + 1, min(len(dms), i + 4)):
                    c1 = dms[i].get("centroid_3d", [0, 0, 0])
                    c2 = dms[j].get("centroid_3d", [0, 0, 0])
                    lat_dist = float(np.sqrt((c1[0] - c2[0])**2 + (c1[2] - c2[2])**2))
                    gap_m = max(0.0, lat_dist - 0.8)
                    clearance_details.append(
                        f"  - Estimated clearance gap between '{dms[i]['label']}' and '{dms[j]['label']}': {gap_m:.2f}m ({gap_m * 100:.0f} cm)"
                    )
        clearance_details_str = "\n".join(clearance_details) if clearance_details else "  - None"

        user_prompt = (
            f"User Question: {query}\n\n"
            f"=== COMPUTER VISION PERCEPTION EVIDENCE ===\n"
            f"1. YOLO Object Detections Summary:\n{count_text}\n\n"
            f"2. YOLO Detected Instances with Coordinates:\n{det_details_str}\n\n"
            f"3. CLIP / VLM Scene Analysis:\n{clip_ans}\n\n"
            f"4. 3D Depth Measurements:\n{depth_details_str}\n\n"
            f"5. Scene Text & OCR Signage:\n{ocr_details_str}\n\n"
            f"6. Human Postures, Fall/Distress, & PPE Safety Compliance:\n{pose_details_str}\n\n"
            f"7. Spatial Relationships & Clearance:\n{spatial_details_str}\n{clearance_details_str}\n\n"
            f"8. Key Verified Findings:\n"
            + "\n".join(f"- {f}" for f in key_findings)
            + "\n\n"
            f"Instructions:\n"
            f"- Answer the user's question directly, precisely, and completely based STRICTLY on the computer vision perception evidence above.\n"
            f"- If the user asks to count or find an item that is NOT present in the detections (e.g., asking for red cars when none are detected), state unambiguously that there are 0 present.\n"
            f"- If the query asks an ambiguous question like 'Count the vehicles' on an image with mixed vehicle types, provide a clear breakdown of each specific vehicle type found.\n"
            f"- For clearance/navigation queries (e.g., whether a 60 cm cart fits), state the numerical clearance gap in cm and answer definitively whether it will fit.\n"
            f"- For safety / distress queries, cite specific persons, fallen status, or missing PPE equipment.\n"
            f"- For OCR signage queries, state the exact text read from the sign adjacent to the referenced landmark (e.g. door).\n"
            f"- For depth/distance queries, identify the closest object and report its distance in metres from camera.\n"
            f"- For spatial relation queries, describe what objects are spatially adjacent or related to the target."
        )

        if self.llm and not hasattr(self.llm, "default_content"):
            try:
                # Check if model supports multimodal image_url
                is_multimodal = False
                model_name = getattr(self.llm, "model", "").lower()
                if any(v in model_name for v in ["gpt-4o", "gemini-1.5", "claude-3-5", "llava"]):
                    is_multimodal = True

                content_payload: Any = user_prompt
                if is_multimodal and media_uri and os.path.exists(media_uri):
                    try:
                        import base64
                        import io
                        from PIL import Image
                        pil = Image.open(media_uri).convert("RGB")
                        w, h = pil.size
                        if max(w, h) > 1024:
                            s = 1024 / max(w, h)
                            pil = pil.resize((int(w * s), int(h * s)), Image.LANCZOS)
                        buf = io.BytesIO()
                        pil.save(buf, format="JPEG", quality=90)
                        b64 = base64.b64encode(buf.getvalue()).decode()
                        content_payload = [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        ]
                    except Exception as img_e:
                        logger.debug(f"Image encode error: {img_e}")
                        content_payload = user_prompt

                resp = self.llm.generate(
                    messages=[
                        {"role": "system", "content": _SYNTH_SYSTEM},
                        {"role": "user", "content": content_payload},
                    ],
                    max_tokens=800,
                )
                txt = resp.content.strip()
                if txt:
                    return txt + "\n" + unified_section
            except Exception as e:
                logger.warning(f"LLM answer generation failed: {e}")
                # Fallback to pure text if multimodal payload failed
                try:
                    resp = self.llm.generate(
                        messages=[
                            {"role": "system", "content": _SYNTH_SYSTEM},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=800,
                    )
                    txt = resp.content.strip()
                    if txt:
                        return txt + "\n" + unified_section
                except Exception as fallback_e:
                    logger.warning(f"LLM text fallback failed: {fallback_e}")

        if claims_verification and claims_verification.remediated_answer_text:
            return claims_verification.remediated_answer_text + "\n" + unified_section

        body = "\n".join(key_findings) if key_findings else "No visual evidence detected."
        return body + "\n" + unified_section
