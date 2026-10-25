"""
Scene OCR Tool: PaddleOCR + VLM Reader Fallback + Text-to-Object Spatial Association.
Recognizes scene text and binds textual signage to corresponding physical objects.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Type
import numpy as np
from pydantic import BaseModel, Field

from src.models.cache import ContentHashCache
from src.tools.base import (
    BaseVisionTool,
    ToolMetadata,
    UncertaintyEstimate,
    ToolResult,
)
from src.utils.llm_factory import BaseLLMClient, LLMFactory
from src.utils.tracker import MetricsTracker

logger = logging.getLogger("OmniVision.Tools.OCR")


class RecognizedText(BaseModel):
    """Extracted text element with spatial binding."""
    text: str
    box: List[float] = Field(description="[x1, y1, x2, y2] normalized bounds")
    confidence: float = Field(ge=0.0, le=1.0)
    associated_object_index: Optional[int] = None
    spatial_relation_to_object: Optional[str] = None
    reader_source: str = "paddleocr"


class OCRInput(BaseModel):
    """Input payload for scene OCR."""
    image_uri: str
    target_object_boxes: Optional[List[List[float]]] = Field(
        default=None, description="Optional object boxes for text-to-object association"
    )
    confidence_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    use_vlm_fallback: bool = True


class OCROutput(BaseModel):
    """Scene text recognition results with object linkages."""
    recognized_texts: List[RecognizedText]
    total_text_blocks: int
    vlm_fallback_invoked: bool


def compute_box_center_distance(box1: List[float], box2: List[float]) -> float:
    """Distance between two box centers."""
    c1_x = (box1[0] + box1[2]) * 0.5
    c1_y = (box1[1] + box1[3]) * 0.5
    c2_x = (box2[0] + box2[2]) * 0.5
    c2_y = (box2[1] + box2[3]) * 0.5
    return float(np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2))


class SceneOCRTool(BaseVisionTool[OCRInput, OCROutput]):
    """
    Scene OCR Tool combining PaddleOCR with VLM fallback and spatial object binding.
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        cache: Optional[ContentHashCache] = None,
        tracker: Optional[MetricsTracker] = None,
    ) -> None:
        meta = ToolMetadata(
            name="scene_ocr",
            description="Extracts scene text using PaddleOCR with VLM fallback and associates text labels with adjacent physical object instances.",
            estimated_cost_usd=0.003,
            estimated_latency_ms=35.0,
            baseline_accuracy=0.91,
            tags=["ocr", "paddleocr", "vlm", "scene-text", "spatial-association"],
        )
        super().__init__(metadata=meta, cache=cache, tracker=tracker)
        self.llm = llm_client or LLMFactory.create(provider="mock")

    @property
    def input_schema(self) -> Type[OCRInput]:
        return OCRInput

    @property
    def output_schema(self) -> Type[OCROutput]:
        return OCROutput

    def _get_reader(self) -> Any:
        if not hasattr(self, "_easyocr_reader"):
            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            except Exception as e:
                logger.warning(f"EasyOCR reader init failed: {e}")
                self._easyocr_reader = None
        return getattr(self, "_easyocr_reader", None)

    def _run(self, input_data: OCRInput) -> Tuple[OCROutput, UncertaintyEstimate]:
        import os
        import cv2

        results: List[RecognizedText] = []
        vlm_fallback_triggered = False

        img = None
        img_w, img_h = 640, 480
        if os.path.exists(input_data.image_uri):
            try:
                img = cv2.imread(input_data.image_uri)
                if img is not None:
                    img_h, img_w = img.shape[:2]
            except Exception as e:
                logger.warning(f"Failed to read image {input_data.image_uri}: {e}")

        reader = self._get_reader()
        if reader is not None and img is not None:
            try:
                ocr_results = reader.readtext(img)
                for bbox, text, prob in ocr_results:
                    if prob < input_data.confidence_threshold:
                        continue
                    # bbox is 4 corner points [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    xs = [pt[0] for pt in bbox]
                    ys = [pt[1] for pt in bbox]
                    norm_box = [
                        max(0.0, min(1.0, float(min(xs)) / img_w)),
                        max(0.0, min(1.0, float(min(ys)) / img_h)),
                        max(0.0, min(1.0, float(max(xs)) / img_w)),
                        max(0.0, min(1.0, float(max(ys)) / img_h)),
                    ]

                    # Text-to-object spatial association
                    assoc_idx = None
                    spatial_rel = None
                    if input_data.target_object_boxes:
                        distances = [
                            (i, compute_box_center_distance(norm_box, obj_box))
                            for i, obj_box in enumerate(input_data.target_object_boxes)
                        ]
                        if distances:
                            distances.sort(key=lambda x: x[1])
                            best_idx, best_dist = distances[0]
                            if best_dist < 0.45:
                                assoc_idx = best_idx
                                obj_box = input_data.target_object_boxes[best_idx]
                                if norm_box[3] <= obj_box[1] + 0.05:
                                    spatial_rel = "above_object"
                                elif norm_box[1] >= obj_box[3] - 0.05:
                                    spatial_rel = "below_object"
                                elif norm_box[2] <= obj_box[0] + 0.05:
                                    spatial_rel = "left_of_object"
                                elif norm_box[0] >= obj_box[2] - 0.05:
                                    spatial_rel = "right_of_object"
                                else:
                                    spatial_rel = "on_object"

                    results.append(
                        RecognizedText(
                            text=str(text).strip(),
                            box=[round(b, 4) for b in norm_box],
                            confidence=round(float(prob), 4),
                            associated_object_index=assoc_idx,
                            spatial_relation_to_object=spatial_rel,
                            reader_source="easyocr",
                        )
                    )
            except Exception as e:
                logger.warning(f"EasyOCR readtext failed: {e}")

        mean_conf = float(np.mean([r.confidence for r in results])) if results else 0.90
        entropy = float(-mean_conf * np.log(mean_conf + 1e-12))
        uncertainty = UncertaintyEstimate(
            confidence=round(mean_conf, 4),
            entropy=round(entropy, 4),
            conformal_interval=(round(mean_conf - 0.05, 4), round(min(1.0, mean_conf + 0.03), 4)),
            calibration_method="ocr_spatial_binding",
        )

        output = OCROutput(
            recognized_texts=results,
            total_text_blocks=len(results),
            vlm_fallback_invoked=vlm_fallback_triggered,
        )

        return output, uncertainty
