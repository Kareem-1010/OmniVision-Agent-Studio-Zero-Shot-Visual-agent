# OmniVision Agent: Running System Interfaces

This document specifies the public API contracts, typed schemas, and component interfaces established across each development phase.

---

## Phase 1: Foundations Interfaces

### 1. Telemetry and Metrics (`src.utils.tracker`)
- **`TokenUsage`** (Pydantic v2 Model)
  - `prompt_tokens: int`
  - `completion_tokens: int`
  - `total_tokens: int`
  - `compute_cost(model_name: str) -> float`: Calculates USD cost based on pricing tables.
- **`InvocationMetric`** (Pydantic v2 Model)
  - `call_id: str`, `component: str`, `model_or_tool: str`, `latency_ms: float`, `tokens: TokenUsage`, `cost_usd: float`, `success: bool`, `error_type: Optional[str]`, `timestamp: float`
- **`MetricsTracker`**
  - `record_call(...) -> InvocationMetric`
  - `get_summary() -> TrackerSummary`
  - `reset() -> None`
- **`get_global_tracker() -> MetricsTracker`**: Access the singleton telemetry instance.

### 2. JSON Repair and Fallback Parser (`src.utils.json_repair`)
- **`repair_json(text: str) -> str`**: Heuristic repair fixing markdown code blocks, python literals, unquoted keys, single quotes, trailing commas, and unclosed brackets.
- **`parse_json_safely(text: str, default: Optional[Any] = None) -> Any`**: Multi-stage parsing with automated repair fallback.

### 3. Prompt Registry with Semantic Versioning (`src.utils.prompt_registry`)
- **`PromptVersion`** (Pydantic v2 Model)
  - `version: str`, `template: str`, `input_variables: List[str]`, `description: str`, `metadata: Dict[str, Any]`
  - `render(**kwargs: Any) -> str`
- **`PromptTemplate`** (Pydantic v2 Model)
  - `name: str`, `active_version: str`, `versions: Dict[str, PromptVersion]`
  - `get_active() -> PromptVersion`
  - `add_version(...) -> PromptVersion`
- **`PromptRegistry`**
  - `register(name: str, version: str, template: str, ...) -> PromptVersion`
  - `get(name: str, version: Optional[str] = None) -> PromptVersion`
  - `render(name: str, version: Optional[str] = None, **kwargs: Any) -> str`
  - `list_prompts() -> Dict[str, List[str]]`
- **`get_prompt_registry() -> PromptRegistry`**: Access singleton prompt registry.

### 4. Unified LLM Factory (`src.utils.llm_factory`)
- **`ToolDefinition`** (Pydantic v2 Model)
  - `name: str`, `description: str`, `parameters: Dict[str, Any]`
- **`FunctionCall`** (Pydantic v2 Model)
  - `call_id: str`, `name: str`, `arguments: Dict[str, Any]`, `raw_arguments: str`
- **`LLMResponse`** (Pydantic v2 Model)
  - `content: str`, `function_calls: List[FunctionCall]`, `structured_data: Optional[Dict[str, Any]]`, `model: str`, `prompt_tokens: int`, `completion_tokens: int`, `total_tokens: int`, `latency_ms: float`, `cost_usd: float`, `finish_reason: str`
- **`BaseLLMClient`** (Abstract Base Class)
  - `generate(messages: List[Dict[str, str]], tools: Optional[List[ToolDefinition]] = None, response_model: Optional[Type[T]] = None, **kwargs: Any) -> LLMResponse`
- **Provider Implementations**:
  - `OpenAIClient`, `AnthropicClient`, `GroqClient`, `VLLMClient`, `MockLLMClient`
- **`LLMFactory`**
  - `create(provider: str = "mock", model: Optional[str] = None, **kwargs: Any) -> BaseLLMClient`

### 5. Content-Addressed Multi-Tier Cache (`src.models.cache`)
- **`hash_artifact(data: Any) -> str`**: Deterministic SHA-256 / BLAKE2b hashing supporting images, tensors, numpy arrays, and dictionaries.
- **`ContentHashCache`**
  - `compute_key(namespace: str, *args: Any, **kwargs: Any) -> str`
  - `get(key: str) -> Optional[Any]` (L1 Memory -> L2 Disk -> L3 Redis lookup)
  - `set(key: str, value: Any, ttl_seconds: Optional[int] = None) -> None`
  - `clear() -> None`
  - `stats() -> Dict[str, Any]`

### 6. GPU Memory Manager and LRU Offloading (`src.models.memory_manager`)
- **`auto_select_device() -> torch.device`**: Selects `cuda:0` -> `mps` -> `cpu`.
- **`GPUMemoryManager`**
  - `register_model(model_id: str, model: torch.nn.Module) -> None`
  - `touch_model(model_id: str, model: Optional[torch.nn.Module] = None) -> None`
  - `get_model(model_id: str) -> Optional[torch.nn.Module]`
  - `ensure_capacity(new_model_id: str) -> None`: Automatic LRU eviction and GPU->CPU offloading.
  - `release_all() -> None`
  - `scoped_allocation() -> ContextManager`: Flushes CUDA cache on exit.
  - `get_status() -> Dict[str, Any]`

### 7. Model Catalog & Lazy Loading (`src.models.registry`)
- **`ModelSpec`** (dataclass)
  - `model_id: str`, `loader: Callable`, `precision: str`, `enable_compile: bool`, `compile_mode: str`, `device: Optional[Union[str, torch.device]]`, `description: str`, `extra_kwargs: Dict[str, Any]`
- **`ModelRegistry`**
  - `register(spec: ModelSpec) -> None`
  - `get(model_id: str, **override_kwargs: Any) -> Any`: Thread-safe lazy loading with precision casting & `torch.compile`.
  - `unload(model_id: str) -> bool`
  - `list_models() -> Dict[str, Dict[str, Any]]`
- **`get_model_registry() -> ModelRegistry`**: Access singleton model registry.

### 8. Typed Vision Tool Base Contract (`src.tools.base`)
- **`ToolMetadata`** (Pydantic v2 Model)
  - `name: str`, `description: str`, `version: str`, `estimated_cost_usd: float`, `estimated_latency_ms: float`, `baseline_accuracy: float`, `tags: List[str]`
- **`ObservationID(str)`**:
  - `ObservationID.create(tool_name: str, payload: Any) -> ObservationID` (`obs_<hash>`)
- **`UncertaintyEstimate`** (Pydantic v2 Model)
  - `confidence: float`, `entropy: float`, `conformal_interval: Tuple[float, float]`, `calibration_method: str`, `is_ambiguous: bool`
- **`ToolResult[TOutput]`** (Pydantic v2 Generic Model)
  - `observation_id: str`, `tool_name: str`, `success: bool`, `data: Optional[TOutput]`, `uncertainty: UncertaintyEstimate`, `latency_ms: float`, `cost_usd: float`, `error_message: Optional[str]`, `provenance: Dict[str, Any]`, `timestamp: float`
- **`BaseVisionTool[TInput, TOutput]`** (Generic Abstract Base Class)
  - `@property input_schema -> Type[TInput]`
  - `@property output_schema -> Type[TOutput]`
  - `_run(input_data: TInput) -> Tuple[TOutput, UncertaintyEstimate]`
  - `execute(input_data: Union[TInput, Dict[str, Any]]) -> ToolResult[TOutput]`
  - `get_tool_definition() -> Dict[str, Any]` (OpenAI tool specification format)

---

## Phase 2: Perception Stack Interfaces

### 1. Detector Ensemble (`src.tools.detector_ensemble`)
- **`DetectorEnsembleInput`** (Pydantic v2 Model)
  - `image_uri: str`, `target_classes: Optional[List[str]]`, `confidence_threshold: float = 0.35`, `iou_threshold: float = 0.55`, `enable_sahi: bool = True`, `slice_height: int = 320`, `slice_width: int = 320`, `overlap_ratio: float = 0.20`, `class_agnostic: bool = False`
- **`DetectorEnsembleOutput`** (Pydantic v2 Model)
  - `detections: List[Detection]`, `total_detections: int`, `detector_weights: Dict[str, float]`, `sahi_tiles_processed: int`
- **`weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights, iou_threshold) -> Tuple[List, List, List]`**
- **`DetectorEnsembleTool(BaseVisionTool[DetectorEnsembleInput, DetectorEnsembleOutput])`**

### 2. Open-Vocabulary Detection (`src.tools.open_vocab`)
- **`OpenVocabInput`** (Pydantic v2 Model)
  - `image_uri: str`, `text_queries: List[str]`, `box_threshold: float = 0.35`, `text_threshold: float = 0.25`, `enable_llm_expansion: bool = True`, `consensus_minimum_votes: int = 2`
- **`OpenVocabOutput`** (Pydantic v2 Model)
  - `instances: List[GroundedInstance]`, `total_found: int`, `expanded_phrases: Dict[str, List[str]]`, `active_detectors: List[str]`
- **`OpenVocabDetectorTool(BaseVisionTool[OpenVocabInput, OpenVocabOutput])`**

### 3. Instance Segmentation & Boundary Snapping (`src.tools.segmentation`)
- **`SegmentationInput`** (Pydantic v2 Model)
  - `image_uri: str`, `boxes: Optional[List[List[float]]]`, `points: Optional[List[List[float]]]`, `auto_mask_mode: bool = False`, `refine_boundaries: bool = True`
- **`SegmentationOutput`** (Pydantic v2 Model)
  - `masks: List[SegmentedMask]`, `total_masks: int`, `refinement_applied: bool`
- **`GroundedSegmenterTool(BaseVisionTool[SegmentationInput, SegmentationOutput])`**

### 4. Video Segmentation & Multi-Object Tracking (`src.tools.video_tracker`)
- **`VideoTrackInput`** (Pydantic v2 Model)
  - `video_id: str`, `num_frames: int = 30`, `target_classes: Optional[List[str]]`, `confidence_threshold: float = 0.40`, `max_lost_frames: int = 10`, `enable_tracklet_stitching: bool = True`
- **`VideoTrackOutput`** (Pydantic v2 Model)
  - `tracks: List[Tracklet]`, `total_unique_tracks: int`, `stitched_tracks_count: int`, `frames_processed: int`
- **`VideoTrackerTool(BaseVisionTool[VideoTrackInput, VideoTrackOutput])`**

### 5. Zero-Shot Classification (`src.tools.classifier`)
- **`ClassifierInput`** (Pydantic v2 Model)
  - `image_uri: str`, `candidate_labels: List[str]`, `crop_box: Optional[List[float]] = None`, `open_set_threshold: float = 0.30`, `temperature: float = 0.07`
- **`ClassifierOutput`** (Pydantic v2 Model)
  - `predicted_label: str`, `confidence: float`, `probabilities: Dict[str, float]`, `is_open_set_rejected: bool`, `entropy: float`
- **`ZeroShotClassifierTool(BaseVisionTool[ClassifierInput, ClassifierOutput])`**

### 6. Dense Feature Matching (`src.tools.dense_features`)
- **`DenseFeatureInput`** (Pydantic v2 Model)
  - `image_uri: str`, `query_box: List[float]`, `target_image_uri: Optional[str] = None`, `similarity_threshold: float = 0.60`, `top_k: int = 5`
- **`DenseFeatureOutput`** (Pydantic v2 Model)
  - `matches: List[MatchedPatch]`, `total_matches: int`, `query_feature_dim: int = 768`, `max_similarity: float`
- **`DenseFeatureTool(BaseVisionTool[DenseFeatureInput, DenseFeatureOutput])`**

### 7. Metric Depth & 3D Spatial Layout (`src.tools.depth`)
- **`DepthInput`** (Pydantic v2 Model)
  - `image_uri: str`, `target_boxes: Optional[List[List[float]]]`, `hfov_degrees: float = 65.0`, `estimate_ground_plane: bool = True`
- **`DepthOutput`** (Pydantic v2 Model)
  - `instances_3d: List[Instance3D]`, `closest_instance: Optional[Instance3D]`, `ground_plane_equation: Optional[List[float]]`, `intrinsics: CameraIntrinsics`, `min_depth_meters: float`, `max_depth_meters: float`
- **`MetricDepthTool(BaseVisionTool[DepthInput, DepthOutput])`**

### 8. Pose and Action Recognition (`src.tools.pose_action`)
- **`PoseInput`** (Pydantic v2 Model)
  - `image_uri: str`, `person_boxes: Optional[List[List[float]]]`, `detect_falls: bool = True`, `check_ppe: bool = True`
- **`PoseOutput`** (Pydantic v2 Model)
  - `persons: List[PersonPoseAnalysis]`, `fall_detected_in_scene: bool`, `total_persons_evaluated: int`
- **`PoseActionTool(BaseVisionTool[PoseInput, PoseOutput])`**

### 9. Scene OCR & Spatial Binding (`src.tools.ocr`)
- **`OCRInput`** (Pydantic v2 Model)
  - `image_uri: str`, `target_object_boxes: Optional[List[List[float]]]`, `confidence_threshold: float = 0.40`, `use_vlm_fallback: bool = True`
- **`OCROutput`** (Pydantic v2 Model)
  - `recognized_texts: List[RecognizedText]`, `total_text_blocks: int`, `vlm_fallback_invoked: bool`
- **`SceneOCRTool(BaseVisionTool[OCRInput, OCROutput])`**

### 10. Grounded VLM Oracle (`src.tools.vlm`)
- **`VLMInput`** (Pydantic v2 Model)
  - `image_or_video_uri: str`, `prompt: str`, `mode: str = 'grounded_vqa'`, `draft_hypothesis: Optional[str] = None`
- **`VLMOutput`** (Pydantic v2 Model)
  - `answer: str`, `grounded_boxes: List[GroundedBox]`, `oracle_verdict: Optional[str]`, `cross_check_rationale: Optional[str]`, `model_name: str`
- **`GroundedVLMTool(BaseVisionTool[VLMInput, VLMOutput])`**

### 11. Optical Flow & Dynamic Motion (`src.tools.optical_flow`)
- **`OpticalFlowInput`** (Pydantic v2 Model)
  - `frame_prev_uri: str`, `frame_curr_uri: str`, `motion_threshold_px: float = 2.5`, `compensate_camera_motion: bool = True`
- **`OpticalFlowOutput`** (Pydantic v2 Model)
  - `moving_clusters: List[MotionCluster]`, `mean_speed_px: float`, `max_speed_px: float`, `camera_pan_detected: bool`
- **`OpticalFlowTool(BaseVisionTool[OpticalFlowInput, OpticalFlowOutput])`**

### 12. Visual Attributes & CIELAB Naming (`src.tools.attributes`)
- **`AttributeInput`** (Pydantic v2 Model)
  - `image_uri: str`, `roi_box: Optional[List[float]] = None`, `known_depth_meters: Optional[float] = None`
- **`AttributeOutput`** (Pydantic v2 Model)
  - `dominant_colors: List[ExtractedColor]`, `primary_color: str`, `material_texture: str`, `relative_size_category: str`, `estimated_metric_size_meters: Optional[Tuple[float, float]]`
- **`VisualAttributeTool(BaseVisionTool[AttributeInput, AttributeOutput])`**

### 13. Visualizer & Evidence Attribution (`src.tools.visualizer`)
- **`VisualizerInput`** (Pydantic v2 Model)
  - `image_uri: str`, `boxes: Optional[List[VisualAnnotationBox]] = None`, `polygons: Optional[List[List[Tuple[float, float]]]] = None`, `tracking_trails: Optional[List[List[Tuple[float, float]]]] = None`, `output_filepath: Optional[str] = None`
- **`VisualizerOutput`** (Pydantic v2 Model)
  - `saved_filepath: str`, `width: int`, `height: int`, `total_elements_rendered: int`
- **`VisualizerTool(BaseVisionTool[VisualizerInput, VisualizerOutput])`**

---

## Phase 3: Geometry and 4D Scene Graph Interfaces

### 1. Spatial Geometry & Relation Engine (`src.geometry.relations`)
- **`SpatialRelation`** (Pydantic v2 Model)
  - `relation_type: str`: 'left_of', 'right_of', 'above', 'below', 'behind', 'in_front_of', 'supports', 'contains', 'inside', 'touching'
  - `source_id: str`, `target_id: str`, `confidence: float`, `metric_distance_meters: float`, `is_symmetric: bool`, `attributes: Dict[str, Any]`
- **`EgocentricDirection`** (Pydantic v2 Model)
  - `azimuth_degrees: float`: -180.0 to 180.0 deg
  - `clock_position: str`: e.g. '12 o'clock', '1 o'clock', '3 o'clock', '9 o'clock'
  - `qualitative_direction: str`: 'straight_ahead', 'front_right', 'front_left', 'to_the_right', 'to_the_left', 'behind_observer'
  - `distance_meters: float`
- **`OcclusionPair`** (Pydantic v2 Model)
  - `occluder_id: str`, `occluded_id: str`, `occlusion_ratio: float`, `depth_separation_meters: float`
- **`LineSegment`** (Pydantic v2 Model)
  - `name: str`, `p1: Tuple[float, float]`, `p2: Tuple[float, float]`
- **`LineCrossingEvent`** (Pydantic v2 Model)
  - `track_id: str`, `line_name: str`, `frame_index: int`, `crossing_direction: str` ('entering' / 'exiting'), `intersection_point: Tuple[float, float]`
- **`GeometryEngine`**
  - `euclidean_distance_3d(p1: List[float], p2: List[float]) -> float`
  - `horizontal_distance(p1: List[float], p2: List[float]) -> float`
  - `egocentric_direction(centroid_3d: List[float]) -> EgocentricDirection`
  - `evaluate_pairwise_relations(inst_a: Dict[str, Any], inst_b: Dict[str, Any]) -> List[SpatialRelation]`
  - `evaluate_occlusions(instances: List[Dict[str, Any]]) -> List[OcclusionPair]`
  - `point_in_polygon(point: Tuple[float, float], polygon_pts: List[Tuple[float, float]]) -> bool`
  - `check_line_crossing(track_id: str, prev_pt: Tuple[float, float], curr_pt: Tuple[float, float], line: LineSegment, frame_idx: int = 0) -> Optional[LineCrossingEvent]`

### 2. SLAM-Lite Multi-View 3D Fusion (`src.geometry.slam_lite`)
- **`CameraPose`** (Pydantic v2 Model)
  - `rotation_matrix: List[List[float]]`: 3x3 SO(3) rotation
  - `translation_vector: List[float]`: 3D translation [X, Y, Z]
  - `frame_index: int`, `timestamp: float`
  - `transform_camera_to_world(pt_cam: List[float]) -> List[float]`: Computes $P_w = R \cdot P_c + t$
  - `transform_world_to_camera(pt_world: List[float]) -> List[float]`: Computes $P_c = R^T \cdot (P_w - t)$
- **`WorldInstance`** (Pydantic v2 Model)
  - `global_instance_id: str`, `label: str`, `world_centroid_3d: List[float]`, `world_dimensions_3d: List[float]`, `confidence: float`, `observation_count: int`, `frames_observed: List[int]`, `embedding: Optional[List[float]]`, `attributes: Dict[str, Any]`
- **`compute_world_distance(p1: List[float], p2: List[float]) -> float`**
- **`SLAMVisualOdometryLite`**
  - `__init__(spatial_merge_threshold_meters: float = 0.65)`
  - `register_camera_pose(pose: CameraPose) -> None`
  - `fuse_frame_observations(pose: CameraPose, detections: List[Dict[str, Any]]) -> List[WorldInstance]`: Running average spatial & dimensional fusion with temporal frame tracking
  - `get_world_instances() -> List[WorldInstance]`
  - `clear() -> None`

### 3. Persistent 4D Scene Graph & Query DSL (`src.memory.scene_graph`)
- **`SceneGraphNode`** (Pydantic v2 Model)
  - `node_id: str`, `label: str`, `track_id: Optional[int]`, `time_span: Tuple[int, int]`, `bbox_2d: Optional[List[float]]`, `centroid_3d: Optional[List[float]]`, `dimensions_3d: Optional[List[float]]`, `attributes: Dict[str, Any]`, `embedding: Optional[List[float]]`, `mask_rle: Optional[str]`
- **`SceneGraphEdge`** (Pydantic v2 Model)
  - `source_id: str`, `target_id: str`, `relation_type: str`, `time_span: Tuple[int, int]`, `metric_distance_meters: Optional[float]`, `confidence: float`, `attributes: Dict[str, Any]`
- **`GraphDiffEvent`** (Pydantic v2 Model)
  - `event_type: str`: 'INSTANCE_APPEARED', 'INSTANCE_DISAPPEARED', 'RELATION_FORMED', 'RELATION_BROKEN', 'ATTRIBUTE_CHANGED'
  - `subject_id: str`, `object_id: Optional[str]`, `frame_index: int`, `details: Dict[str, Any]`
- **`Persistent4DSceneGraph`** (NetworkX MultiDiGraph backed)
  - `add_node(node: SceneGraphNode) -> None`
  - `get_node(node_id: str) -> Optional[SceneGraphNode]`
  - `add_edge(edge: SceneGraphEdge) -> None`
  - `get_edges(source_id: Optional[str] = None, target_id: Optional[str] = None, relation_type: Optional[str] = None) -> List[SceneGraphEdge]`
  - `list_nodes() -> List[SceneGraphNode]`, `list_edges() -> List[SceneGraphEdge]`
  - `get_active_subgraph_at_frame(frame_index: int) -> Persistent4DSceneGraph`: Temporal slice of active nodes and edges at given frame
  - `to_dict() -> Dict[str, Any]`, `to_json(indent: int = 2) -> str`
  - `from_dict(data: Dict[str, Any]) -> Persistent4DSceneGraph`, `from_json(json_str: str) -> Persistent4DSceneGraph`
  - `execute_query(query: str) -> List[Dict[str, Any]]`: Declarative Query DSL interpreter (`FIND ?x WHERE ...`, `FIND ?x, ?y WHERE ?x.holding == ?y`)
- **`compute_graph_diff(graph_t1: Persistent4DSceneGraph, graph_t2: Persistent4DSceneGraph, frame_index: int) -> List[GraphDiffEvent]`**: Frame-to-frame diff detector emitting structural and attribute changes.

---

## Phase 4: Multi-Agent Core (LangGraph) Interfaces

### 1. Agent State & Typed Contracts (`src.agent.state`)
- **`TraceEvent`** (Pydantic v2 Model)
  - `event_id: str`, `timestamp: float`, `node_name: str`, `action: str`, `details: Dict[str, Any]`, `cost_usd: float`, `latency_ms: float`
- **`QueryAnalysis`** (Pydantic v2 Model)
  - `original_query: str`, `modality: Literal['image', 'video']`, `reasoning_type: List[str]`, `sub_goals: List[str]`, `key_targets: List[str]`, `temporal_span_required: bool`, `requires_3d: bool`, `ambiguities: List[str]`, `estimated_complexity: Literal['low', 'medium', 'high']`
- **`PlanStep`** (Pydantic v2 Model)
  - `step_id: str`, `sub_goal: str`, `tool_name: str`, `arguments: Dict[str, Any]`, `dependencies: List[str]`, `status: Literal['pending', 'running', 'completed', 'failed', 'skipped']`, `result_observation_id: Optional[str]`, `cost_estimate_usd: float`, `latency_estimate_ms: float`, `error_message: Optional[str]`
- **`VisionProgramPlan`** (Pydantic v2 Model)
  - `plan_id: str`, `steps: List[PlanStep]`, `strategy_rationale: str`, `estimated_total_cost_usd: float`, `estimated_total_latency_ms: float`
- **`CritiqueVerdict`** (Pydantic v2 Model)
  - `passed: bool`, `score: float`, `should_replan: bool`, `should_active_perceive: bool`, `active_perception_target: Optional[Dict[str, Any]]`, `feedback: str`, `missing_evidence: List[str]`, `identified_contradictions: List[str]`
- **`ClaimAttribution`** (Pydantic v2 Model)
  - `claim_id: str`, `statement: str`, `supported: bool`, `observation_ids: List[str]`, `grounded_box: Optional[List[float]]`, `confidence: float`, `verifier_notes: str`
- **`ClaimsVerificationResult`** (Pydantic v2 Model)
  - `all_verified: bool`, `verified_claims: List[ClaimAttribution]`, `unsupported_claims: List[ClaimAttribution]`, `hallucination_rate: float`, `remediated_answer_text: Optional[str]`
- **`GroundedSynthesis`** (Pydantic v2 Model)
  - `query: str`, `answer: str`, `confidence_interval: Tuple[float, float]`, `calibrated_confidence: float`, `citations: List[str]`, `key_findings: List[str]`, `total_cost_usd: float`, `total_latency_ms: float`, `visual_evidence_summary: Dict[str, Any]`
- **`OmniVisionState`** (Pydantic v2 Model)
  - `query: str`, `media_uri: str`, `media_type: Literal['image', 'video']`, `query_analysis: Optional[QueryAnalysis]`, `plan: Optional[VisionProgramPlan]`, `current_step_index: int`, `observations: Dict[str, Dict[str, Any]]`, `scene_graph: Dict[str, Any]`, `trace_log: List[TraceEvent]`, `critique: Optional[CritiqueVerdict]`, `claims_verification: Optional[ClaimsVerificationResult]`, `final_synthesis: Optional[GroundedSynthesis]`, `iteration_count: int`, `max_iterations: int`, `budget_limit_usd: float`, `accumulated_cost_usd: float`, `accumulated_latency_ms: float`, `status: str`, `error: Optional[str]`

### 2. Query Analyst (`src.agent.query_analyst`)
- **`QueryAnalyst`**
  - `analyze(query: str, media_uri: str, media_type: str = 'image') -> QueryAnalysis`: Decomposes query into required facets (`detection`, `3d_depth`, `spatial_relation`, `temporal`, `ocr`, `action_pose`, `attributes`), target entity classes, sub-goals, and ambiguity warnings.

### 3. Learned Tool Router (`src.agent.tool_router`)
- **`ToolRoutingDecision`** (Pydantic v2 Model)
  - `selected_tool: str`, `rationale: str`, `estimated_cost_usd: float`, `estimated_latency_ms: float`, `expected_accuracy: float`, `fallback_tool: Optional[str]`
- **`ToolRouter`**
  - `route_subgoal(subgoal: str, target_entities: Optional[List[str]] = None, budget_remaining_usd: float = 0.50, prefer_speed: bool = False) -> ToolRoutingDecision`: Evaluates semantic sub-goal and taxonomies to select Pareto-optimal perception tools.

### 4. Vision Program Planner (`src.agent.planner`)
- **`VisionPlanner`**
  - `synthesize_plan(analysis: QueryAnalysis, media_uri: str, budget_limit_usd: float = 0.50, enable_tree_search: bool = True, num_candidates: int = 3) -> VisionProgramPlan`: Generates dependency-resolved plan DAG with candidate branch scoring: $S = 0.5 \cdot \text{Acc} + 0.3 \cdot \text{Cov} - 0.1 \cdot \text{CostPenalty} - 0.1 \cdot \text{LatPenalty}$.
  - `replan_on_feedback(current_plan: VisionProgramPlan, feedback: str, missing_evidence: List[str], media_uri: str) -> VisionProgramPlan`: Dynamically injects recovery steps.

### 5. Active Perception Loop (`src.agent.active_perception`)
- **`ActivePerceptionAction`** (Pydantic v2 Model)
  - `action_type: str`, `target_box: Optional[List[float]]`, `context_margin: float`, `remediation_tool: str`, `arguments: Dict[str, Any]`, `rationale: str`
- **`ActivePerceptionManager`**
  - `evaluate_observations_for_active_perception(observations: Dict[str, Dict[str, Any]], media_uri: str, image_dims: Tuple[int, int] = (640, 480)) -> Optional[ActivePerceptionAction]`: Triggers high-res zoom crops or multi-prompt consensus when confidence falls below calibrated thresholds.

### 6. Critic & Verifier (`src.agent.critic`)
- **`CriticVerifier`**
  - `evaluate(analysis: QueryAnalysis, plan_steps: List[PlanStep], observations: Dict[str, Dict[str, Any]], iteration_count: int = 1, max_iterations: int = 12) -> CritiqueVerdict`: Verifies sub-goal fulfillment, cross-modal consistency, and flags contradictions.

### 7. Hallucination Guard (`src.agent.hallucination_guard`)
- **`HallucinationGuard`**
  - `verify_draft_answer(draft_text: str, observations: Dict[str, Dict[str, Any]], scene_graph_dict: Optional[Dict[str, Any]] = None) -> ClaimsVerificationResult`: Decomposes text into atomic claims, validates each against observation IDs, calculates hallucination rates, and produces remediated citation-attributed text.

### 8. Grounded Synthesizer (`src.agent.synthesizer`)
- **`GroundedSynthesizer`**
  - `synthesize(analysis: QueryAnalysis, observations: Dict[str, Dict[str, Any]], scene_graph_dict: Dict[str, Any], claims_verification: Optional[ClaimsVerificationResult] = None, accumulated_cost_usd: float = 0.0, accumulated_latency_ms: float = 0.0) -> GroundedSynthesis`: Integrates all multi-modal evidence into final response with conformal confidence intervals and citations.

### 9. Multi-Agent Orchestrator (`src.agent.orchestrator`)
- **`ToolDispatcher`**: `execute(tool_name: str, arguments: Dict[str, Any]) -> Any`
- **`OmniVisionAgent`**:
  - `__init__(max_iterations: int = 12, budget_limit_usd: float = 0.50, enable_tree_search: bool = True, enable_active_perception: bool = True)`
  - `run(query: str, media_uri: str, media_type: Literal['image', 'video'] = 'image') -> GroundedSynthesis`: Executes compiled LangGraph multi-agent loop with conditional re-planning, active perception, and hallucination guardrails.

---

## Phase 5: Memory and Long Video Reasoning Interfaces

### 1. Hierarchical Episodic Vector Memory (`src.memory.episodic`)
- **`FrameNode`** (Pydantic v2 Model)
  - `frame_id: str`, `frame_index: int`, `timestamp: float`, `image_uri: str`, `embedding: List[float]`, `detected_entities: List[str]`, `scene_summary: str`, `metadata: Dict[str, Any]`
- **`ClipNode`** (Pydantic v2 Model)
  - `clip_id: str`, `start_frame: int`, `end_frame: int`, `start_time: float`, `end_time: float`, `representative_frame_id: str`, `mean_embedding: List[float]`, `prominent_entities: List[str]`, `activity_summary: str`, `frame_ids: List[str]`
- **`EventNode`** (Pydantic v2 Model)
  - `event_id: str`, `event_type: str`, `start_frame: int`, `end_frame: int`, `start_time: float`, `end_time: float`, `participating_track_ids: List[str]`, `keyframe_id: str`, `description: str`, `confidence: float`, `attributes: Dict[str, Any]`
- **`EpisodeNode`** (Pydantic v2 Model)
  - `episode_id: str`, `title: str`, `start_frame: int`, `end_frame: int`, `start_time: float`, `end_time: float`, `key_event_ids: List[str]`, `overall_summary: str`
- **`cosine_similarity(v1: List[float], v2: List[float]) -> float`**
- **`EpisodicVectorMemory`**
  - `add_frame(frame: FrameNode) -> None`
  - `add_clip(clip: ClipNode) -> None`
  - `add_event(event: EventNode) -> None`
  - `add_episode(episode: EpisodeNode) -> None`
  - `search_frames(query_embedding: Optional[List[float]] = None, entity_filter: Optional[str] = None, time_range: Optional[Tuple[float, float]] = None, top_k: int = 5) -> List[Tuple[FrameNode, float]]`
  - `search_clips(query_embedding: Optional[List[float]] = None, time_range: Optional[Tuple[float, float]] = None, top_k: int = 5) -> List[Tuple[ClipNode, float]]`
  - `search_events(event_type: Optional[str] = None, time_range: Optional[Tuple[float, float]] = None, track_id: Optional[str] = None) -> List[EventNode]`
  - `consolidate_clips_from_frames(frames_per_clip: int = 15, similarity_threshold: float = 0.85) -> List[ClipNode]`
  - `to_dict() -> Dict[str, Any]`, `to_json(indent: int = 2) -> str`
  - `from_dict(data: Dict[str, Any], embedding_dim: int = 512) -> EpisodicVectorMemory`

### 2. Adaptive Content-Aware Keyframe Sampler (`src.video.sampler`)
- **`KeyframeSample`** (Pydantic v2 Model)
  - `frame_index: int`, `timestamp: float`, `image_uri: str`, `salience_score: float`, `selection_reason: str`, `metadata: Dict[str, Any]`
- **`AdaptiveKeyframeSampler`**
  - `__init__(motion_threshold: float = 0.25, semantic_threshold: float = 0.35, min_temporal_gap_frames: int = 5)`
  - `sample_keyframes(frame_uris: List[str], fps: float = 30.0, frame_budget: int = 16, embeddings: Optional[List[List[float]]] = None) -> List[KeyframeSample]`: Selects salience-maximizing keyframes under budget bounds with non-maximum temporal suppression.

### 3. Temporal Event Detection & Boundary Segmentation (`src.video.events`)
- **`SemanticEvent`** (Pydantic v2 Model)
  - `event_id: str`, `event_type: str`, `start_frame: int`, `end_frame: int`, `start_time_sec: float`, `end_time_sec: float`, `peak_frame: int`, `peak_time_sec: Optional[float]`, `primary_track_id: str`, `secondary_track_id: Optional[str]`, `confidence: float`, `description: str`, `evidence_tags: List[str]`, `metadata: Dict[str, Any]`
- **`TemporalEventSegmenter`**
  - `__init__(fps: float = 30.0, min_event_duration_frames: int = 3)`
  - `detect_events_from_tracks_and_diffs(tracks: List[Dict[str, Any]], graph_diffs: Optional[List[GraphDiffEvent]] = None, line_crossings: Optional[List[LineCrossingEvent]] = None) -> List[SemanticEvent]`: Fuses kinematics, boundary tripwires, and 4D scene graph diffs into segmented semantic events.

### 4. Long-Video Temporal QA Engine (`src.video.temporal_qa`)
- **`TemporalQAResponse`** (Pydantic v2 Model)
  - `query: str`, `query_type: str`, `answer: str`, `start_time_sec: Optional[float]`, `end_time_sec: Optional[float]`, `peak_time_sec: Optional[float]`, `target_event_count: Optional[int]`, `cited_event_ids: List[str]`, `cited_frame_ids: List[str]`, `confidence: float`, `timeline: List[Dict[str, Any]]`
- **`TemporalQAEngine`**
  - `__init__(memory: EpisodicVectorMemory, fps: float = 30.0)`
  - `answer_query(query: str, segmented_events: Optional[List[SemanticEvent]] = None) -> TemporalQAResponse`: Resolves localization ('when'), causality ('what happened after'), recurrence ('how many times'), and narrative summary queries with grounded timestamp intervals.

### 5. Streaming Video Ingestion & Sliding Window Buffer (`src.video.streaming`)
- **`StreamAlert`** (Pydantic v2 Model)
  - `alert_id: str`, `alert_type: str`, `severity: Literal['info', 'warning', 'critical']`, `timestamp: float`, `frame_index: int`, `description: str`, `source_track_id: Optional[str]`, `metadata: Dict[str, Any]`
- **`BufferedFrame`** (Pydantic v2 Model)
  - `frame_index: int`, `timestamp: float`, `image_uri: str`, `detections: List[Dict[str, Any]]`, `attributes: Dict[str, Any]`
- **`StreamingVideoProcessor`**
  - `__init__(window_size_frames: int = 60, fps: float = 30.0, alert_callback: Optional[Callable[[StreamAlert], None]] = None)`
  - `ingest_frame(image_uri: str, timestamp: Optional[float] = None, check_ppe_and_falls: bool = True) -> List[StreamAlert]`: FIFO sliding window buffer updates, online 4D graph diffing, and instant safety hazard alert dispatch.
  - `get_window_frames() -> List[BufferedFrame]`
  - `get_alert_history() -> List[StreamAlert]`
  - `clear() -> None`

---

## Phase 6: Training and Self-Improvement Interfaces

### 1. Execution Trace Harvesting & Dataset Builder (`src.training.dataset_builder`)
- **`AgentTrajectory`** (Pydantic v2 Model)
  - `trajectory_id: str`, `query: str`, `media_uri: str`, `query_analysis: Dict[str, Any]`, `plan: Dict[str, Any]`, `trace_events: List[Dict[str, Any]]`, `critique: Dict[str, Any]`, `claims_verification: Dict[str, Any]`, `final_synthesis: Dict[str, Any]`, `hallucination_rate: float`, `total_cost_usd: float`, `total_latency_ms: float`, `reward_score: float`
- **`SFTSample`** (Pydantic v2 Model)
  - `sample_id: str`, `system_prompt: str`, `instruction: str`, `input_context: str`, `output_response: str`, `metadata: Dict[str, Any]`
- **`DPOPreferencePair`** (Pydantic v2 Model)
  - `pair_id: str`, `prompt: str`, `chosen: str`, `rejected: str`, `chosen_reward: float`, `rejected_reward: float`, `reward_margin: float`
- **`DatasetBuilder`**
  - `__init__(reward_pass_threshold: float = 0.75)`
  - `harvest_from_state(state: OmniVisionState, explicit_reward: Optional[float] = None) -> AgentTrajectory`: Extracts state trajectory, computes composite reward $\text{reward} = \text{critique\_score} - 0.4 \times \text{halluc\_rate} - \text{cost\_penalty}$.
  - `build_sft_dataset(trajectories: Optional[List[AgentTrajectory]] = None, min_reward: Optional[float] = None) -> List[SFTSample]`: Synthesizes prompt-plan pairs from high-reward runs.
  - `build_dpo_pairs(trajectories: Optional[List[AgentTrajectory]] = None, min_margin: float = 0.20) -> List[DPOPreferencePair]`: Groups trajectories by query and pairs contrasting high vs low reward executions.
  - `export_to_jsonl(samples: List[Union[SFTSample, DPOPreferencePair]], filepath: str) -> int`: Persists dataset records to disk.

### 2. Planner Self-Improvement & DPO Loss (`src.training.sft_dpo_pipeline`)
- **`compute_dpo_loss(...) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`**
  - Computes Bradley-Terry preference loss: $\mathcal{L}_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$.
  - Returns: `(loss, chosen_rewards, rejected_rewards)`.
- **`TrainingStepMetrics`** (Pydantic v2 Model)
  - `step: int`, `loss: float`, `chosen_reward: float`, `rejected_reward: float`, `reward_margin: float`, `accuracy_or_win_rate: float`
- **`LightweightPlannerPolicy(nn.Module)`**
  - `__init__(vocab_size: int = 64, hidden_dim: int = 128)`
  - `forward(plan_features: torch.Tensor) -> torch.Tensor`
- **`PlannerSelfImprovementTrainer`**
  - `__init__(beta: float = 0.1, learning_rate: float = 1e-4, feature_dim: int = 32)`
  - `train_dpo_step(chosen_features: torch.Tensor, rejected_features: torch.Tensor, step_idx: int = 0) -> TrainingStepMetrics`: One backpropagation step on preference tensors.
  - `run_dpo_training_loop(dpo_pairs: List[DPOPreferencePair], num_epochs: int = 3) -> List[TrainingStepMetrics]`: Optimizes planner policy across epochs.
  - `get_metrics_history() -> List[TrainingStepMetrics]`

### 3. Student Detector Knowledge Distillation Engine (`src.training.distillation`)
- **`DistillationLossOutput`** (Pydantic v2 Model)
  - `total_loss: float`, `hard_label_loss: float`, `soft_teacher_loss: float`, `box_regression_loss: float`
- **`StudentDetectorNetwork(nn.Module)`**
  - `__init__(in_channels: int = 3, num_classes: int = 10, base_dim: int = 32)`: Lightweight depthwise-separable conv backbone + dual prediction heads.
  - `forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`: Returns `(cls_logits [B, num_classes], bbox_coords [B, 4])` bounded in $[0, 1]$.
- **`StudentDetectorDistiller`**
  - `__init__(num_classes: int = 10, temperature: float = 3.0, alpha_hard: float = 0.3, beta_bbox: float = 1.5, learning_rate: float = 5e-4)`
  - `compute_loss(student_logits: torch.Tensor, student_boxes: torch.Tensor, teacher_logits: torch.Tensor, teacher_boxes: torch.Tensor, true_labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, DistillationLossOutput]`: Fuses soft KL-divergence, hard CE, and smooth L1 coordinate loss.
  - `train_step(images: torch.Tensor, teacher_logits: torch.Tensor, teacher_boxes: torch.Tensor, true_labels: Optional[torch.Tensor] = None) -> DistillationLossOutput`: Forward, backward, and optimizer update step.

### 4. SigLIP / CLIP Linear Probing with Temperature Calibration (`src.training.linear_probe`)
- **`ProbeEpochMetrics`** (Pydantic v2 Model)
  - `epoch: int`, `train_loss: float`, `train_accuracy: float`, `val_loss: Optional[float]`, `val_accuracy: Optional[float]`, `expected_calibration_error: Optional[float]`
- **`LinearProbeHead(nn.Module)`**
  - `__init__(embed_dim: int = 768, num_classes: int = 10, init_temp: float = 1.0)`: Linear head with learnable $\log T$ temperature parameter.
  - `forward(embeddings: torch.Tensor) -> torch.Tensor`: Outputs $z / T$ scaled logits.
- **`compute_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float`**
  - Computes Expected Calibration Error across prediction confidence bins: $\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$.
- **`SigLIPLinearProbeTrainer`**
  - `__init__(embed_dim: int = 768, num_classes: int = 10, class_names: Optional[List[str]] = None, learning_rate: float = 1e-3, weight_decay: float = 1e-4)`
  - `fit(train_embeddings: torch.Tensor, train_labels: torch.Tensor, val_embeddings: Optional[torch.Tensor] = None, val_labels: Optional[torch.Tensor] = None, num_epochs: int = 10, batch_size: int = 32) -> List[ProbeEpochMetrics]`: Trains head and calibrates temperature.
  - `predict(embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`: Evaluates class predictions and normalized probability distribution.
  - `evaluate(embeddings: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]`: Evaluates loss, accuracy, and ECE.

---

## Phase 7: Demo Modes (Vertical Domain Presets)

### 1. Preset Core & Registry Architecture (`src.presets.base`)
- **`DomainPresetConfig`** (Pydantic v2 Model)
  - `name: str`, `description: str`, `system_prompt: str`, `enabled_tools: List[str]`, `confidence_threshold: float`, `max_iterations: int`, `budget_limit_usd: float`, `custom_parameters: Dict[str, Any]`
- **`BaseDomainPreset`** (Abstract Base Class)
  - `@property config -> DomainPresetConfig`
  - `get_system_prompt() -> str`
  - `get_toolset() -> List[str]`
  - `analyze(media_uri: str, query: Optional[str] = None, **kwargs: Any) -> Any`
- **`PresetRegistry`**
  - `register(preset: BaseDomainPreset) -> None`
  - `get(domain_name: str) -> BaseDomainPreset`
  - `list_presets() -> List[str]`
- **`get_preset_registry() -> PresetRegistry`**: Access global preset registry.

### 2. Retail Analytics Mode (`src.presets.retail`)
- **`ShelfGapItem`** (Pydantic v2 Model)
  - `shelf_id: str`, `gap_box: List[float]`, `gap_width_ratio: float`, `category_label: str`, `restock_priority: str`
- **`QueueStatus`** (Pydantic v2 Model)
  - `queue_id: str`, `people_count: int`, `queue_head_pos: Optional[List[float]]`, `queue_tail_pos: Optional[List[float]]`, `estimated_wait_time_sec: float`, `congestion_level: str`
- **`RetailKPIReport`** (Pydantic v2 Model)
  - `media_uri: str`, `timestamp: float`, `footfall_in: int`, `footfall_out: int`, `net_store_occupancy: int`, `average_dwell_time_sec: float`, `queues: Dict[str, QueueStatus]`, `shelf_gaps: List[ShelfGapItem]`, `out_of_stock_percentage: float`, `heatmap_resolution: Tuple[int, int]`, `dwell_heatmap_matrix: List[List[float]]`, `summary_text: str`
- **`RetailPreset(BaseDomainPreset)`**
  - `compute_footfall(tracks, tripwire=None) -> Tuple[int, int, Dict[str, float]]`: Tripwire line crossing detection and dwell durations.
  - `monitor_queues(shopper_positions, checkout_zones=None, service_rate_per_min=1.5) -> Dict[str, QueueStatus]`: Queue counts and wait-time estimation ($W = L / \mu$).
  - `detect_shelf_gaps(shelves_data, gap_threshold_ratio=0.15) -> Tuple[List[ShelfGapItem], float]`: Planogram empty slot detection and OOS percentage.
  - `generate_dwell_heatmap(trajectories, grid_size=(50, 50)) -> np.ndarray`: Normalized 2D spatial trajectory density grid.
  - `analyze_retail_scene(media_uri, tracks=None, shelves=None, tripwire=None) -> RetailKPIReport`

### 3. Accessibility Assistant Mode (`src.presets.accessibility`)
- **`EgocentricObject`** (Pydantic v2 Model)
  - `label: str`, `clock_face: str`, `distance_meters: float`, `elevation_level: str`, `description: str`, `is_landmark: bool`
- **`ObstacleAlert`** (Pydantic v2 Model)
  - `hazard_label: str`, `distance_meters: float`, `direction: str`, `urgency: str`, `guidance: str`
- **`AccessibilityNarrationReport`** (Pydantic v2 Model)
  - `media_uri: str`, `user_heading_deg: float`, `clockface_objects: List[EgocentricObject]`, `environmental_texts: List[str]`, `obstacle_alerts: List[ObstacleAlert]`, `speech_transcript: str`, `ssml_markup: str`, `audio_sample_rate_hz: int`, `audio_wav_bytes_length: int`
- **`TTSSynthesizer`**
  - `to_ssml(text: str, rate: str = 'medium', pitch: str = 'default') -> str`: Formats speech into SSML markup.
  - `synthesize_wav_tone(text: str, duration_sec: float = 0.8, base_freq_hz: float = 440.0) -> bytes`: Generates valid 16-bit PCM Mono WAV audio chime notification.
- **`AccessibilityPreset(BaseDomainPreset)`**
  - `calculate_clock_face(bbox: List[float], user_heading_deg: float = 0.0) -> str`: Converts 2D position into 12-hour clock face bearing.
  - `estimate_elevation(bbox: List[float]) -> str`: Ground level, waist level, eye level, overhead.
  - `narrate_scene(media_uri, user_heading_deg=0.0, detections=None, ocr_texts=None) -> AccessibilityNarrationReport`

### 4. Hazard & Safety Mode (`src.presets.hazard_safety`)
- **`PPEAuditItem`** (Pydantic v2 Model)
  - `worker_id: str`, `box: List[float]`, `has_hard_hat: bool`, `has_high_vis_vest: bool`, `has_safety_boots: bool`, `is_fully_compliant: bool`, `missing_gear: List[str]`, `confidence: float`
- **`EnvironmentalHazardItem`** (Pydantic v2 Model)
  - `hazard_id: str`, `hazard_type: str`, `location_box: List[float]`, `estimated_3d_distance_m: float`, `severity: str`, `remediation_action: str`
- **`SafetyAuditReport`** (Pydantic v2 Model)
  - `site_id: str`, `media_uri: str`, `timestamp: float`, `overall_safety_rating: str`, `ppe_compliance_rate: float`, `personnel_count: int`, `compliant_personnel_count: int`, `workers: List[PPEAuditItem]`, `environmental_hazards: List[EnvironmentalHazardItem]`, `max_3d_risk_score: float`, `summary_findings: List[str]`, `pdf_report_path: Optional[str]`
- **`HazardSafetyPreset(BaseDomainPreset)`**
  - `audit_ppe(detected_workers, required_ppe=None) -> Tuple[List[PPEAuditItem], float]`: Per-worker and site-level PPE compliance audit.
  - `compute_3d_proximity_risk(workers, heavy_machinery) -> float`: Metric 3D proximity risk score between personnel and heavy equipment.
  - `generate_pdf_report(site_id, rating, compliance_rate, workers, hazards, risk_score, filepath) -> str`: Generates styled ReportLab PDF audit report with executive tables, badges, and corrective actions.
  - `audit_scene(media_uri, site_id='PLANT-ALPHA-01', workers=None, hazards=None, machinery=None, export_pdf_path=None) -> SafetyAuditReport`

---

## Phase 8: Interfaces, Web Services, and Performance

### 1. Performance, Optimization & Profiling (`src.models.optimization`)
- **`ONNXExporter`**
  - `export_to_onnx(model: nn.Module, sample_input: torch.Tensor, output_path: str, input_names: Optional[List[str]] = None, output_names: Optional[List[str]] = None, dynamic_axes: Optional[Dict[str, Any]] = None, opset_version: int = 17) -> str`: Exports PyTorch module to ONNX, checks graph validity, and verifies with ONNX Runtime CPUExecutionProvider.
- **`QuantizationResult`** (Pydantic v2 Model)
  - `original_size_bytes: int`, `quantized_size_bytes: int`, `compression_ratio: float`, `original_latency_ms: float`, `quantized_latency_ms: float`, `speedup_factor: float`
- **`ModelQuantizer`**
  - `quantize_dynamic_int8(model: nn.Module, qconfig_layers: Optional[Tuple[Type[nn.Module], ...]] = None) -> Tuple[nn.Module, QuantizationResult]`: Post-training dynamic INT8 quantization with compression and latency speedup benchmarking.
- **`DynamicBatcher`**
  - `__init__(batch_fn: Callable[[List[Any]], Any], max_batch_size: int = 16, timeout_ms: float = 8.0)`
  - `enqueue(item: Any) -> Any`: Coalesces asynchronous individual requests into micro-batches dispatched on max size or timeout window.
- **`AsyncToolDispatcher`**
  - `__init__(sync_dispatcher: Any)`
  - `execute_tool_async(tool_name: str, arguments: Dict[str, Any]) -> Any`
  - `execute_parallel(calls: List[Tuple[str, Dict[str, Any]]]) -> List[Any]`: Runs independent perception tools concurrently via `asyncio.gather`.
- **`ComponentLatency`** (Pydantic v2 Model)
  - `component: str`, `sample_count: int`, `mean_ms: float`, `p50_ms: float`, `p90_ms: float`, `p99_ms: float`, `percentage_of_total: float`
- **`ProfilingReport`** (Pydantic v2 Model)
  - `total_calls: int`, `total_duration_ms: float`, `components: List[ComponentLatency]`, `primary_bottleneck: str`, `recommendations: List[str]`
- **`PerformanceProfiler`**
  - `record(component: str, latency_ms: float) -> None`
  - `generate_report() -> ProfilingReport`: Calculates latency percentiles (P50/P90/P99), identifies primary bottleneck, and produces optimization recommendations.

### 2. FastAPI Async REST & WebSocket Service (`src.api.service`)
- **`AgentRunRequest`** (Pydantic v2 Model)
  - `query: str`, `media_uri: str`, `media_type: Literal['image', 'video']`, `preset_name: Optional[str]`, `max_iterations: int`, `budget_limit_usd: float`, `enable_tree_search: bool`, `enable_active_perception: bool`
- **`JobStatusResponse`** (Pydantic v2 Model)
  - `job_id: str`, `status: Literal['queued', 'running', 'completed', 'failed']`, `created_at: float`, `completed_at: Optional[float]`, `execution_time_ms: Optional[float]`, `result: Optional[Dict[str, Any]]`, `error: Optional[str]`, `trace_events_count: int`
- **`SystemHealthResponse`** (Pydantic v2 Model)
  - `status: str`, `version: str`, `uptime_seconds: float`, `active_jobs: int`, `registered_presets: List[str]`, `gpu_status: Dict[str, Any]`
- **`AsyncJobManager`**
  - `create_job(request: AgentRunRequest) -> JobRecord`
  - `get_job(job_id: str) -> Optional[JobRecord]`
  - `broadcast_event(job_id: str, event_data: Dict[str, Any]) -> None`
  - `execute_job_in_background(job_id: str) -> None`
- **`RateLimiter`**: Sliding-window rate limiter enforcing quotas (120 req/min).
- **`verify_api_key`**: Validates `X-API-Key` or `Authorization: Bearer <token>`.
- **REST & Streaming Endpoints**:
  - `POST /v1/agent/run` -> `GroundedSynthesis` (synchronous execution)
  - `POST /v1/agent/submit` -> `JobStatusResponse` (202 Accepted async job)
  - `GET /v1/agent/jobs/{job_id}` -> `JobStatusResponse` (job polling)
  - `GET /v1/presets` -> Dict of domain preset configurations
  - `GET /v1/health` -> `SystemHealthResponse`
  - `WebSocket /v1/ws/traces/{job_id}` -> Streams real-time JSON execution trace events.

### 3. Command Line Interface (`src.api.cli`)
- `omnivision serve --host 127.0.0.1 --port 8000`: Launches FastAPI server via uvicorn.
- `omnivision run -q "..." -m "..." -p "retail"`: Runs visual reasoning query from terminal.
- `omnivision optimize --output-dir artifacts/onnx`: Runs ONNX export and INT8 quantization benchmarks.
- `omnivision ui --port 7860`: Launches interactive Gradio dashboard.

### 4. Gradio Interactive Dashboard (`src.ui.app`)
- **`build_demo() -> gr.Blocks`**:
  - Tab 1: **Agent Studio**: Visual input, query textbox, preset selector, budget/iteration sliders, grounded answer, conformal confidence gauge, claim attribution table.
  - Tab 2: **Plan DAG & Live Trace**: Mermaid DAG rendering and tool execution timings.
  - Tab 3: **Agent vs. Baseline VLM**: Side-by-side comparative evaluation demonstrating hallucination elimination.
  - Tab 4: **3D Spatial Scene**: Metric 3D instance coordinates and depth colormaps.

---

## Phase 9: Evaluation & Scientific Research Rigor

### 1. Evaluation Datasets & Benchmarks (`src.eval.datasets`)
- **`BenchmarkSample`** (Pydantic v2 Model)
  - `query_id: str`
  - `query: str`
  - `media_uri: str`
  - `media_type: Literal['image', 'video']`
  - `task_family: Literal['count', 'spatial_relation', 'grounding_refcoco', 'ocr_scene_text', 'safety_hazard', 'temporal_localization', '3d_metric_distance', 'counterfactual_simulation', 'multi_hop_relational']`
  - `ground_truth_answer: str`
  - `ground_truth_count: Optional[int]`
  - `ground_truth_boxes: Optional[List[List[float]]]`
  - `ground_truth_time_interval: Optional[Tuple[float, float]]`
  - `expected_tools: List[str]`
  - `difficulty: Literal['easy', 'medium', 'hard']`
  - `annotation_notes: str`
- **`BenchmarkDataset`** (Pydantic v2 Model)
  - `dataset_name: str`, `version: str`, `description: str`, `samples: List[BenchmarkSample]`
  - `filter_by_task(task_family: str) -> List[BenchmarkSample]`
  - `to_jsonl(filepath: str) -> int`
- **`Custom300BenchmarkBuilder`**
  - `build_dataset() -> BenchmarkDataset`: Synthesizes 300 rigorous multimodal benchmark queries across 8 reasoning task families.

### 2. Scientific Metrics Engine (`src.eval.metrics`)
- **`EvaluationMetricsSummary`** (Pydantic v2 Model)
  - `sample_count: int`, `overall_accuracy: float`, `counting_mae: float`, `counting_rmse: float`, `grounding_iou_at_05: float`, `grounding_mean_iou: float`, `tracking_mota: float`, `tracking_idf1: float`, `temporal_mean_tiou: float`, `tool_selection_accuracy: float`, `hallucination_rate: float`, `expected_calibration_error: float`, `brier_score: float`, `mean_plan_steps: float`, `mean_latency_ms: float`, `mean_token_cost_usd: float`
- Metric Functions:
  - `compute_vqa_accuracy(predictions: List[str], ground_truths: List[str]) -> float`: Exact match and soft word-overlap accuracy.
  - `compute_counting_mae_rmse(predicted_counts: List[int], ground_truth_counts: List[int]) -> Tuple[float, float]`
  - `calculate_box_iou(box_a: List[float], box_b: List[float]) -> float`
  - `compute_grounding_iou(pred_boxes_list: List[List[float]], gt_boxes_list: List[List[float]], threshold: float = 0.50) -> Tuple[float, float]`
  - `compute_mota_idf1(false_positives: int, false_negatives: int, id_switches: int, ground_truth_objects: int, id_true_positives: int) -> Tuple[float, float]`
  - `compute_temporal_tiou(pred_intervals: List[Tuple[float, float]], gt_intervals: List[Tuple[float, float]]) -> float`
  - `compute_tool_selection_accuracy(pred_tool_sets: List[List[str]], expected_tool_sets: List[List[str]]) -> float`
  - `compute_ece_and_brier(confidences: List[float], binary_outcomes: List[int], n_bins: int = 10) -> Tuple[float, float]`

### 3. Statistical Rigor Engine (`src.eval.statistics`)
- **`SignificanceResult`** (Pydantic v2 Model)
  - `test_name: str`, `statistic: float`, `p_value: float`, `is_significant: bool`, `effect_size: float`, `effect_magnitude: str`, `sample_size: int`, `mean_baseline: float`, `mean_agent: float`, `difference_mean: float`, `ci_95_lower: float`, `ci_95_upper: float`
- Statistical Testing Functions:
  - `compute_bootstrap_ci(data: List[float], n_bootstraps: int = 1000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]`: Empirical non-parametric bootstrap confidence interval.
  - `compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float`: Effect size metric.
  - `classify_effect_magnitude(d: float) -> str`: negligible (<0.2), small (<0.5), medium (<0.8), large (>=0.8).
  - `paired_significance_test(baseline_scores: List[float], agent_scores: List[float], alpha: float = 0.05, alternative: str = 'two-sided') -> SignificanceResult`: Paired Student's t-test / Wilcoxon signed-rank test.
  - `compute_mcnemar_test(baseline_correct: List[bool], agent_correct: List[bool]) -> Dict[str, Union[float, bool, str]]`: Exact binomial or continuity-corrected Chi-Square McNemar test.
  - `adjust_p_values_fdr(p_values: List[float], method: str = 'holm') -> List[float]`: Multi-hypothesis Holm-Bonferroni correction.

### 4. Systematic Ablation Study (`src.eval.ablations`)
- **`AblationConfig`** (Pydantic v2 Model)
  - `config_id: str`, `name: str`, `description: str`, `enable_critic: bool`, `enable_active_perception: bool`, `enable_program_synthesis: bool`, `enable_tree_search: bool`, `enable_learned_router: bool`, `detector_mode: Literal['ensemble', 'single']`, `embedding_backbone: Literal['siglip', 'clip']`, `enable_metric_depth: bool`, `is_monolithic_vlm: bool`
- **`AblationRunResult`** (Pydantic v2 Model)
  - `config: AblationConfig`, `sample_count: int`, `metrics: EvaluationMetricsSummary`, `per_sample_accuracies: List[float]`, `significance_vs_full: Optional[SignificanceResult]`, `delta_accuracy: float`, `delta_latency_ms: float`, `delta_cost_usd: float`
- **`AblationRunner`**
  - `get_canonical_10_configs() -> List[AblationConfig]`: 1. Full OmniVision Agent, 2. Monolithic Single VLM, 3. w/o Critic/Verifier, 4. w/o Active Perception, 5. w/o Program Synthesis, 6. w/o Tree Search, 7. w/o Learned Router, 8. Single Detector vs Ensemble, 9. CLIP vs SigLIP, 10. Metric 3D Depth ON vs OFF.
  - `evaluate_configuration(config: AblationConfig, dataset: BenchmarkDataset, seed: int = 42) -> AblationRunResult`
  - `run_suite(dataset: BenchmarkDataset, configs: Optional[List[AblationConfig]] = None, seed: int = 42) -> List[AblationRunResult]`

### 5. Scientific Reporting, Plots & Failure Taxonomy (`src.eval.reporting`)
- **`generate_latex_table(results: List[Any], caption: str = ..., label: str = ...) -> str`**: Formats LaTeX booktabs table for academic papers.
- **`generate_markdown_table(results: List[Any]) -> str`**: Formats GitHub Markdown comparison table.
- **`generate_calibration_plot(confidences: List[float], binary_outcomes: List[int], filepath: str, n_bins: int = 10, title: str = ...) -> str`**: Renders and saves reliability diagram calibration plot with empirical bins and ECE/Brier annotations.
- **`classify_failure_cases(predictions: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]]) -> List[Dict[str, Any]]`**: 7-category failure taxonomy (`Perception Miss`, `Grounding Misalignment`, `Hallucination`, `Geometry Inaccuracy`, `Temporal Mismatch`, `Reasoning Error`, `Timeout / Budget Exceeded`).
- **`generate_paper_report(benchmark_results: Dict[str, Any], ablation_results: List[Any], failure_analysis: Optional[List[Dict[str, Any]]] = None, filepath: Optional[str] = None) -> str`**: Full research paper report in Markdown format.

---

## Phase 10: Engineering, Hardening, and Delivery

### 1. The 8 Canonical Visual Reasoning Chains (`src.agent.chains`)
- **`ChainExecutionResult`** (Pydantic v2 Model)
  - `chain_id: str`
  - `chain_name: str`
  - `query: str`
  - `media_uri: str`
  - `answer: str`
  - `confidence: float`
  - `steps_executed: List[str]`
  - `tools_invoked: List[str]`
  - `extracted_entities: List[Dict[str, Any]]`
  - `quantitative_data: Dict[str, Any]`
  - `execution_time_ms: float`
  - `is_verified: bool`
- **`VisualReasoningChainRunner`**
  - `run_count_red_cars_chain(image_uri: str, target_color: str = 'red', target_category: str = 'car') -> ChainExecutionResult`
  - `run_safety_hazard_chain(image_uri: str, site_id: str = 'INDUSTRIAL-BAY-04') -> ChainExecutionResult`
  - `run_read_sign_next_to_door_chain(image_uri: str, reference_object: str = 'door', target_object: str = 'sign') -> ChainExecutionResult`
  - `run_unique_people_dwell_chain(video_uri: str) -> ChainExecutionResult`
  - `run_closest_object_distance_chain(image_uri: str) -> ChainExecutionResult`
  - `run_person_umbrella_relation_chain(image_uri: str) -> ChainExecutionResult`
  - `run_vehicle_arrival_and_exit_chain(video_uri: str) -> ChainExecutionResult`
  - `run_cart_shelf_fit_simulation_chain(image_uri: str, cart_dimensions_3d: Optional[Tuple[float, float, float]] = None, shelf_gap_m: Optional[float] = None) -> ChainExecutionResult`
  - `dispatch_chain(chain_identifier: Union[int, str], media_uri: str, **kwargs: Any) -> ChainExecutionResult`

### 2. Geometry Engine Primitives & Spatial Graphs (`src.geometry.engine`, `src.geometry.scene_graph`)
- **`Point2D`** (Pydantic v2 Model): `x: float`, `y: float` (normalized image plane).
- **`Point3D`** (Pydantic v2 Model): `x: float`, `y: float`, `z: float` (metric Euclidean coordinate).
- **`BoundingBox2D`** (Pydantic v2 Model): `x1`, `y1`, `x2`, `y2`, `label`, `score`, with bounds validator ($x_2 \ge x_1, y_2 \ge y_1$).
- **`BoundingBox3D`** (Pydantic v2 Model): `center: Point3D`, `dimensions: Point3D`, `orientation: Point3D`, `volume: float`.
- **`GeometryEngine`**: Extended computational geometry reasoning engine supporting line crossing, spatial distances, and orientation.
- **`PersistentSceneGraph`**: Wrapper for `Persistent4DSceneGraph` with `get_neighbors(node_id, relation_type)`.

### 3. Production Deployment & Security Isolation
- **`Dockerfile`**: Multi-stage production container with CPU/CUDA support, non-root `appuser`, and `/v1/health` healthcheck.
- **`docker-compose.yml`**: Multi-service architecture deploying FastAPI (`api`), Gradio Studio (`ui`), and Redis (`redis`).
- **`Makefile`**: Automation targets for `setup`, `test`, `test-chains`, `eval`, `optimize`, `serve`, `ui`, `docker-build`, `docker-up`, `docker-down`, and `clean`.
- **`README.md`**: Publication-grade technical documentation with Mermaid state machine diagrams, benchmark tables, and CLI guides.

