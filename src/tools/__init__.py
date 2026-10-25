"""
OmniVision Tools Package.
Base tool interfaces, observation tracking, and Phase 2 perception stack.
"""

from src.tools.base import (
    BaseVisionTool,
    ToolMetadata,
    ObservationID,
    UncertaintyEstimate,
    ToolResult,
)
from src.tools.detector_ensemble import (
    DetectorEnsembleTool,
    DetectorEnsembleInput,
    DetectorEnsembleOutput,
    Detection,
    weighted_boxes_fusion,
)
from src.tools.open_vocab import (
    OpenVocabDetectorTool,
    OpenVocabInput,
    OpenVocabOutput,
    GroundedInstance,
)
from src.tools.segmentation import (
    GroundedSegmenterTool,
    SegmentationInput,
    SegmentationOutput,
    SegmentedMask,
)
from src.tools.video_tracker import (
    VideoTrackerTool,
    VideoTrackInput,
    VideoTrackOutput,
    Tracklet,
)
from src.tools.classifier import (
    ZeroShotClassifierTool,
    ClassifierInput,
    ClassifierOutput,
)
from src.tools.dense_features import (
    DenseFeatureTool,
    DenseFeatureInput,
    DenseFeatureOutput,
    MatchedPatch,
)
from src.tools.depth import (
    MetricDepthTool,
    DepthInput,
    DepthOutput,
    Instance3D,
    CameraIntrinsics,
)
from src.tools.pose_action import (
    PoseActionTool,
    PoseInput,
    PoseOutput,
    PersonPoseAnalysis,
)
from src.tools.ocr import (
    SceneOCRTool,
    OCRInput,
    OCROutput,
    RecognizedText,
)
from src.tools.vlm import (
    GroundedVLMTool,
    VLMInput,
    VLMOutput,
    GroundedBox,
)
from src.tools.optical_flow import (
    OpticalFlowTool,
    OpticalFlowInput,
    OpticalFlowOutput,
    MotionCluster,
)
from src.tools.attributes import (
    VisualAttributeTool,
    AttributeInput,
    AttributeOutput,
    ExtractedColor,
)
from src.tools.visualizer import (
    VisualizerTool,
    VisualizerInput,
    VisualizerOutput,
    VisualAnnotationBox,
)

__all__ = [
    # Base
    "BaseVisionTool",
    "ToolMetadata",
    "ObservationID",
    "UncertaintyEstimate",
    "ToolResult",
    # 1. Detector Ensemble
    "DetectorEnsembleTool",
    "DetectorEnsembleInput",
    "DetectorEnsembleOutput",
    "Detection",
    "weighted_boxes_fusion",
    # 2. Open-Vocab
    "OpenVocabDetectorTool",
    "OpenVocabInput",
    "OpenVocabOutput",
    "GroundedInstance",
    # 3. Segmentation
    "GroundedSegmenterTool",
    "SegmentationInput",
    "SegmentationOutput",
    "SegmentedMask",
    # 4. Tracking
    "VideoTrackerTool",
    "VideoTrackInput",
    "VideoTrackOutput",
    "Tracklet",
    # 5. Classifier
    "ZeroShotClassifierTool",
    "ClassifierInput",
    "ClassifierOutput",
    # 6. Dense Features
    "DenseFeatureTool",
    "DenseFeatureInput",
    "DenseFeatureOutput",
    "MatchedPatch",
    # 7. Depth
    "MetricDepthTool",
    "DepthInput",
    "DepthOutput",
    "Instance3D",
    "CameraIntrinsics",
    # 8. Pose/Action
    "PoseActionTool",
    "PoseInput",
    "PoseOutput",
    "PersonPoseAnalysis",
    # 9. OCR
    "SceneOCRTool",
    "OCRInput",
    "OCROutput",
    "RecognizedText",
    # 10. VLM
    "GroundedVLMTool",
    "VLMInput",
    "VLMOutput",
    "GroundedBox",
    # 11. Optical Flow
    "OpticalFlowTool",
    "OpticalFlowInput",
    "OpticalFlowOutput",
    "MotionCluster",
    # 12. Attributes
    "VisualAttributeTool",
    "AttributeInput",
    "AttributeOutput",
    "ExtractedColor",
    # 13. Visualizer
    "VisualizerTool",
    "VisualizerInput",
    "VisualizerOutput",
    "VisualAnnotationBox",
]
