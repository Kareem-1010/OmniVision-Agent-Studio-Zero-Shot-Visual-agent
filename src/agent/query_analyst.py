"""
Query Analyst Agent Node.
Decomposes user queries into multimodal reasoning requirements, sub-goals,
target entity classes, and ambiguity assessments.

Uses Groq LLM (llama-3.3-70b-versatile) when GROQ_API_KEY is set for accurate
natural-language understanding; falls back to heuristic parsing otherwise.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from src.agent.state import QueryAnalysis
from src.utils.llm_factory import BaseLLMClient, LLMFactory

logger = logging.getLogger("OmniVision.Agent.QueryAnalyst")


# ---------------------------------------------------------------------------
# COCO-80 vocabulary for routing decisions
# ---------------------------------------------------------------------------
COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
}

PLURAL_TO_SINGULAR: Dict[str, str] = {
    "people": "person", "peoples": "person", "persons": "person",
    "humans": "person", "pedestrians": "person", "workers": "person",
    "cars": "car", "vehicles": "car", "trucks": "truck", "buses": "bus",
    "bicycles": "bicycle", "bikes": "bicycle", "motorcycles": "motorcycle",
    "apples": "apple", "oranges": "orange", "bananas": "banana",
    "bottles": "bottle", "cups": "cup", "books": "book",
    "chairs": "chair", "tables": "dining table", "sofas": "couch",
    "dogs": "dog", "cats": "cat", "birds": "bird",
    "phones": "cell phone", "laptops": "laptop",
    "backpacks": "backpack", "umbrellas": "umbrella",
    "sandwiches": "sandwich", "pizzas": "pizza", "donuts": "donut",
    "cakes": "cake", "hotdogs": "hot dog",
}


# ---------------------------------------------------------------------------
# LLM-Powered Analysis Prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an expert computer vision query analyst.
Given a natural language question about an image or video, extract:
1. reasoning_type: list from [detection, counting, spatial_relation, 3d_depth, temporal, ocr, action_pose, attributes, summary, vqa]
2. key_targets: specific object entities the user is asking about. Preserve descriptive color and attribute modifiers when present (e.g. "red car", "yellow bus", "blue truck", "apple", "tomato", "person with helmet")
3. sub_goals: 1-3 short action phrases describing what visual tools should do
4. requires_3d: true/false
5. temporal_span_required: true/false
6. estimated_complexity: low/medium/high

Respond ONLY with valid JSON matching this schema:
{
  "reasoning_type": ["detection", "counting"],
  "key_targets": ["red car"],
  "sub_goals": ["Detect and count red cars in the scene"],
  "requires_3d": false,
  "temporal_span_required": false,
  "estimated_complexity": "low"
}"""


class QueryAnalyst:
    """
    Multimodal Query Analyst.
    Parses open-ended visual queries into formal task specifications,
    identifying required visual tools, 3D/temporal requirements, and sub-goals.
    Uses Groq LLM when available for accurate NL understanding.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None) -> None:
        # Auto-select Groq if key is available, else mock
        if llm_client is not None:
            self.llm = llm_client
        else:
            groq_key = os.getenv("GROQ_API_KEY", "").strip()
            if groq_key and not groq_key.startswith("gsk_xxxx"):
                try:
                    groq_model = os.getenv("OMNIVISION_LLM_MODEL", "qwen/qwen3.8-27b")
                    self.llm = LLMFactory.create(
                        provider="groq",
                        model=groq_model,
                        api_key=groq_key,
                        temperature=0.0,
                        max_tokens=512,
                    )
                    logger.info(f"QueryAnalyst: using Groq {groq_model}")
                except Exception as e:
                    logger.warning(f"Groq init failed ({e}), falling back to heuristics")
                    self.llm = LLMFactory.create(provider="mock")
            else:
                self.llm = LLMFactory.create(provider="mock")

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def analyze(self, query: str, media_uri: str, media_type: str = "image") -> QueryAnalysis:
        """
        Analyze user query and return a structured QueryAnalysis specification.
        Tries Groq LLM first; falls back to keyword heuristic.
        """
        clean_query = query.strip()

        # Try LLM-based analysis first
        llm_result = self._llm_analyze(clean_query)
        if llm_result:
            return QueryAnalysis(
                original_query=clean_query,
                modality="video" if media_type == "video" else "image",
                reasoning_type=llm_result.get("reasoning_type", ["detection"]),
                sub_goals=llm_result.get("sub_goals", []),
                key_targets=self._normalise_targets(llm_result.get("key_targets", [])),
                temporal_span_required=llm_result.get("temporal_span_required", media_type == "video"),
                requires_3d=llm_result.get("requires_3d", False),
                ambiguities=[],
                estimated_complexity=llm_result.get("estimated_complexity", "medium"),
            )

        # Heuristic fallback
        return self._heuristic_analyze(clean_query, media_uri, media_type)

    # ------------------------------------------------------------------
    # LLM-based Analysis
    # ------------------------------------------------------------------

    def _llm_analyze(self, query: str) -> Optional[Dict[str, Any]]:
        """Call Groq (or other LLM) to parse query into structured analysis."""
        # Only use real LLM, not mock
        if hasattr(self.llm, "default_content"):  # MockLLMClient check
            return None
        try:
            response = self.llm.generate(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Query: {query}"},
                ],
            )
            raw = response.content.strip()
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
        return None

    # ------------------------------------------------------------------
    # Heuristic Fallback
    # ------------------------------------------------------------------

    def _heuristic_analyze(self, query: str, media_uri: str, media_type: str) -> QueryAnalysis:
        """Keyword-based heuristic query analysis fallback."""
        lower_q = query.lower()

        facets: List[str] = []
        requires_3d = False
        temporal_span = media_type == "video"

        # Counting
        if any(w in lower_q for w in ["how many", "count", "number of", "total"]):
            facets.append("counting")

        # Detection / Localization
        if any(w in lower_q for w in ["find", "detect", "where", "localize", "identify",
                                        "locate", "spot", "is there", "are there", "show",
                                        "standing next to", "holding", "beside"]):
            if "detection" not in facets:
                facets.append("detection")

        # 3D / Depth / Distance / Clearance
        if any(w in lower_q for w in ["distance", "how far", "depth", "closest", "farthest",
                                        "meters", "metres", "3d", "front", "behind",
                                        "fit", "cart", "wide", "clearance", "gap", "cm",
                                        "nearest", "closest to the camera"]):
            facets.append("3d_depth")
            requires_3d = True

        # Spatial relations
        if any(w in lower_q for w in ["left of", "right of", "above", "below", "inside",
                                        "on top of", "next to", "touching", "near",
                                        "beside", "between", "standing next to",
                                        "adjacent"]):
            if "spatial_relation" not in facets:
                facets.append("spatial_relation")

        # Temporal
        if any(w in lower_q for w in ["moving", "speed", "crossed", "track", "entering",
                                        "exiting", "trajectory", "when did", "how long"]):
            facets.append("temporal")
            temporal_span = True

        # OCR / Text / Sign
        if any(w in lower_q for w in ["read", "text", "sign", "written", "letters", "words",
                                        "label", "signage", "placard", "notice"]):
            if "ocr" not in facets:
                facets.append("ocr")

        # Pose / Action / Safety / PPE / Distress
        if any(w in lower_q for w in ["pose", "fall", "fallen", "sitting", "standing", "lying",
                                        "helmet", "hard hat", "vest", "ppe", "wearing", "safety",
                                        "hazard", "slip", "distress", "hurt", "injured",
                                        "collapsed", "accident", "emergency",
                                        "missing ppe", "protective", "equipment"]):
            if "action_pose" not in facets:
                facets.append("action_pose")

        # Attributes / Color
        if any(w in lower_q for w in ["color", "colour", "red", "blue", "green", "yellow",
                                        "black", "white", "material", "texture", "size",
                                        "orange", "purple", "brown"]):
            if "attributes" not in facets:
                facets.append("attributes")

        # Summary / Description
        if any(w in lower_q for w in ["summary", "summarize", "describe", "description",
                                        "explain", "scene", "overview", "what is happening",
                                        "what do you see", "tell me about"]):
            if "summary" not in facets:
                facets.append("summary")

        # VQA catch-all
        if any(w in lower_q for w in ["what", "why", "which", "who", "can you"]):
            if "summary" not in facets:
                facets.append("vqa")

        # Extract key targets
        key_targets = self._extract_targets(lower_q)

        # Ensure detection is included if entities present
        if not facets:
            facets = ["detection", "vqa"]
        elif "detection" not in facets and ("counting" in facets or key_targets):
            facets.insert(0, "detection")

        # Sub-goals
        sub_goals: List[str] = []
        if "detection" in facets or "counting" in facets:
            target_str = ", ".join(key_targets) if key_targets else "relevant objects"
            sub_goals.append(f"Detect and localize {target_str} within the scene")
        if "3d_depth" in facets:
            sub_goals.append("Estimate metric depth and 3D positions")
        if "spatial_relation" in facets:
            sub_goals.append("Evaluate spatial relationships between instances")
        if "temporal" in facets:
            sub_goals.append("Track instances across frames and analyze motion")
        if "ocr" in facets:
            sub_goals.append("Extract and interpret scene text")
        if "action_pose" in facets:
            sub_goals.append("Analyze human posture, actions, and safety equipment")
        if "attributes" in facets:
            sub_goals.append("Extract visual attributes (color, texture, material)")
        if "summary" in facets or "vqa" in facets:
            sub_goals.append("Synthesize grounded visual answer to the query")
        sub_goals.append("Verify evidence and compose final grounded answer")

        # Complexity
        complexity = "low"
        if len(facets) >= 3 or requires_3d or temporal_span:
            complexity = "high" if (len(facets) >= 4 and temporal_span) else "medium"

        return QueryAnalysis(
            original_query=query,
            modality="video" if media_type == "video" else "image",
            reasoning_type=facets,
            sub_goals=sub_goals,
            key_targets=key_targets,
            temporal_span_required=temporal_span,
            requires_3d=requires_3d,
            ambiguities=[],
            estimated_complexity=complexity,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_targets(self, lower_q: str) -> List[str]:
        """Extract object class targets from query text, preserving color/attribute modifiers."""
        targets: List[str] = []

        # 1. Check for color + object phrases first (e.g. "red car", "yellow bus", "blue truck")
        colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "gray", "grey", "silver", "gold", "brown"]
        for c in colors:
            for cls in list(COCO_CLASSES) + list(PLURAL_TO_SINGULAR.keys()):
                if re.search(rf"\b{c}\s+{re.escape(cls)}\b", lower_q):
                    sing = PLURAL_TO_SINGULAR.get(cls, cls)
                    phrase = f"{c} {sing}"
                    if phrase not in targets:
                        targets.insert(0, phrase)

        # 2. Check COCO singulars
        for cls in COCO_CLASSES:
            if re.search(rf"\b{re.escape(cls)}\b", lower_q):
                if cls not in targets and not any(cls in t for t in targets):
                    targets.append(cls)

        # 3. Check plurals -> singular
        for plural, singular in PLURAL_TO_SINGULAR.items():
            if re.search(rf"\b{re.escape(plural)}\b", lower_q):
                if singular not in targets and not any(singular in t for t in targets):
                    targets.append(singular)

        if not targets:
            # Noun extraction heuristic
            stop_words = {
                "what", "where", "which", "how", "many", "is", "are", "the", "this", "that",
                "there", "image", "video", "frame", "count", "number", "total", "scene", "find",
                "detect", "show", "tell", "give", "all", "any", "summarize", "summary",
                "describe", "description", "explain", "provide", "also", "photo", "picture",
                "and", "with", "for", "from", "into", "onto", "over", "under", "about",
                "can", "you", "please", "could", "would", "look", "see", "view", "each",
                "every", "both", "some", "them", "then", "its", "their", "such", "than",
                "have", "has", "was", "were", "been", "being", "will", "would", "should",
                "color", "colour", "size", "type", "kind", "sort", "thing",
            }
            words = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", lower_q) if w not in stop_words]
            targets = words[:3]

        return targets

    def _normalise_targets(self, targets: List[str]) -> List[str]:
        """Normalise target names to canonical singular class names."""
        result = []
        for t in targets:
            t_lower = t.lower().strip()
            words = t_lower.split()
            if len(words) > 1 and words[-1] in PLURAL_TO_SINGULAR:
                canonical = " ".join(words[:-1] + [PLURAL_TO_SINGULAR[words[-1]]])
            elif t_lower in PLURAL_TO_SINGULAR:
                canonical = PLURAL_TO_SINGULAR[t_lower]
            else:
                canonical = t_lower
            if canonical not in result:
                result.append(canonical)
        return result
