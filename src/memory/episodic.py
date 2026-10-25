"""
Hierarchical Episodic Vector Memory.
Organizes multi-modal visual memory across four granularities: Frame -> Clip -> Event -> Episode.
Provides vector similarity search, temporal range filtering, entity indexing, and memory consolidation.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger("OmniVision.Memory.Episodic")


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two 1D embedding vectors."""
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-7 or norm_b < 1e-7:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class FrameNode(BaseModel):
    """Atomic frame-level memory node with visual embedding and perceptual tags."""
    frame_id: str
    frame_index: int
    timestamp: float
    image_uri: str
    embedding: List[float] = Field(default_factory=list)
    detected_entities: List[str] = Field(default_factory=list)
    scene_summary: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClipNode(BaseModel):
    """Cohesive short video segment (e.g., 2-5 seconds) aggregating multiple frames."""
    clip_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    representative_frame_id: str
    mean_embedding: List[float] = Field(default_factory=list)
    prominent_entities: List[str] = Field(default_factory=list)
    activity_summary: str = ""
    frame_ids: List[str] = Field(default_factory=list)


class EventNode(BaseModel):
    """Discrete semantic action or interaction spanning one or more clips."""
    event_id: str
    event_type: str  # e.g., 'person_entered', 'object_dropped', 'vehicle_parked'
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    participating_track_ids: List[str] = Field(default_factory=list)
    keyframe_id: str
    description: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    attributes: Dict[str, Any] = Field(default_factory=dict)


class EpisodeNode(BaseModel):
    """High-level narrative episode encapsulating a session or sequence of events."""
    episode_id: str
    title: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    key_event_ids: List[str] = Field(default_factory=list)
    overall_summary: str = ""


class EpisodicVectorMemory:
    """
    Hierarchical Episodic Vector Memory System.
    Stores and indexes video memory across Frame, Clip, Event, and Episode tiers
    with cosine similarity retrieval, temporal slicing, and entity-guided queries.
    """

    def __init__(self, embedding_dim: int = 512) -> None:
        self.embedding_dim = embedding_dim
        self._frames: Dict[str, FrameNode] = {}
        self._clips: Dict[str, ClipNode] = {}
        self._events: Dict[str, EventNode] = {}
        self._episodes: Dict[str, EpisodeNode] = {}
        # Entity Inverted Index: entity_name -> list of frame_ids
        self._entity_index: Dict[str, List[str]] = {}

    def add_frame(self, frame: FrameNode) -> None:
        """Register a frame-level observation node."""
        self._frames[frame.frame_id] = frame
        for ent in frame.detected_entities:
            ent_clean = ent.lower().strip()
            self._entity_index.setdefault(ent_clean, []).append(frame.frame_id)

    def add_clip(self, clip: ClipNode) -> None:
        """Register a consolidated clip node."""
        self._clips[clip.clip_id] = clip

    def add_event(self, event: EventNode) -> None:
        """Register a semantic event node."""
        self._events[event.event_id] = event

    def add_episode(self, episode: EpisodeNode) -> None:
        """Register an episode narrative node."""
        self._episodes[episode.episode_id] = episode

    def search_frames(
        self,
        query_embedding: Optional[List[float]] = None,
        entity_filter: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
        top_k: int = 5,
    ) -> List[Tuple[FrameNode, float]]:
        """
        Search frame memory using vector similarity and/or metadata constraints.
        Returns list of (FrameNode, similarity_score) sorted descending.
        """
        candidates: List[FrameNode] = list(self._frames.values())

        # Filter by entity
        if entity_filter:
            ent_key = entity_filter.lower().strip()
            allowed_ids = set(self._entity_index.get(ent_key, []))
            candidates = [f for f in candidates if f.frame_id in allowed_ids]

        # Filter by temporal range (start_sec, end_sec)
        if time_range:
            t_start, t_end = time_range
            candidates = [f for f in candidates if t_start <= f.timestamp <= t_end]

        if not candidates:
            return []

        results: List[Tuple[FrameNode, float]] = []
        for frame in candidates:
            sim = 1.0
            if query_embedding and frame.embedding:
                sim = cosine_similarity(query_embedding, frame.embedding)
            results.append((frame, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_clips(
        self,
        query_embedding: Optional[List[float]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        top_k: int = 5,
    ) -> List[Tuple[ClipNode, float]]:
        """Search clip nodes by embedding or temporal window."""
        candidates = list(self._clips.values())
        if time_range:
            t_start, t_end = time_range
            candidates = [c for c in candidates if not (c.end_time < t_start or c.start_time > t_end)]

        results: List[Tuple[ClipNode, float]] = []
        for clip in candidates:
            sim = 1.0
            if query_embedding and clip.mean_embedding:
                sim = cosine_similarity(query_embedding, clip.mean_embedding)
            results.append((clip, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_events(
        self,
        event_type: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
        track_id: Optional[str] = None,
    ) -> List[EventNode]:
        """Search structured event nodes matching event type, time, or participant."""
        evs = list(self._events.values())
        if event_type:
            evs = [e for e in evs if event_type.lower() in e.event_type.lower()]
        if time_range:
            t_start, t_end = time_range
            evs = [e for e in evs if not (e.end_time < t_start or e.start_time > t_end)]
        if track_id:
            evs = [e for e in evs if track_id in e.participating_track_ids]
        evs.sort(key=lambda e: e.start_time)
        return evs

    def consolidate_clips_from_frames(
        self,
        frames_per_clip: int = 15,
        similarity_threshold: float = 0.85,
    ) -> List[ClipNode]:
        """
        Consolidate raw sequential frames into canonical ClipNodes.
        Computes average clip embeddings and identifies prominent entities.
        """
        sorted_frames = sorted(self._frames.values(), key=lambda f: f.frame_index)
        if not sorted_frames:
            return []

        new_clips: List[ClipNode] = []
        chunk: List[FrameNode] = []

        for f in sorted_frames:
            chunk.append(f)
            if len(chunk) >= frames_per_clip:
                clip = self._build_clip_from_chunk(chunk)
                self.add_clip(clip)
                new_clips.append(clip)
                chunk = []

        if chunk:
            clip = self._build_clip_from_chunk(chunk)
            self.add_clip(clip)
            new_clips.append(clip)

        return new_clips

    def _build_clip_from_chunk(self, chunk: List[FrameNode]) -> ClipNode:
        """Compute aggregated statistics for a consecutive frame sequence."""
        clip_id = f"clip_{chunk[0].frame_index:05d}_{chunk[-1].frame_index:05d}"
        start_frame = chunk[0].frame_index
        end_frame = chunk[-1].frame_index
        start_time = chunk[0].timestamp
        end_time = chunk[-1].timestamp
        rep_frame = chunk[len(chunk) // 2].frame_id

        # Calculate mean embedding
        valid_embeds = [f.embedding for f in chunk if f.embedding]
        if valid_embeds:
            mean_vec = np.mean(valid_embeds, axis=0).tolist()
        else:
            mean_vec = [0.0] * self.embedding_dim

        # Collect prominent entities
        ent_counts: Dict[str, int] = {}
        for f in chunk:
            for ent in f.detected_entities:
                ent_counts[ent] = ent_counts.get(ent, 0) + 1
        prominent = [ent for ent, cnt in sorted(ent_counts.items(), key=lambda x: x[1], reverse=True)[:5]]

        summary = f"Clip {start_time:.1f}s-{end_time:.1f}s featuring {', '.join(prominent) or 'ambient scene'}."

        return ClipNode(
            clip_id=clip_id,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=round(start_time, 2),
            end_time=round(end_time, 2),
            representative_frame_id=rep_frame,
            mean_embedding=mean_vec,
            prominent_entities=prominent,
            activity_summary=summary,
            frame_ids=[f.frame_id for f in chunk],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory state to dictionary."""
        return {
            "frames": [f.model_dump() for f in self._frames.values()],
            "clips": [c.model_dump() for c in self._clips.values()],
            "events": [e.model_dump() for e in self._events.values()],
            "episodes": [ep.model_dump() for ep in self._episodes.values()],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize memory state to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding_dim: int = 512) -> EpisodicVectorMemory:
        """Instantiate memory system from serialized dictionary."""
        mem = cls(embedding_dim=embedding_dim)
        for fd in data.get("frames", []):
            mem.add_frame(FrameNode.model_validate(fd))
        for cd in data.get("clips", []):
            mem.add_clip(ClipNode.model_validate(cd))
        for ed in data.get("events", []):
            mem.add_event(EventNode.model_validate(ed))
        for ep in data.get("episodes", []):
            mem.add_episode(EpisodeNode.model_validate(ep))
        return mem
