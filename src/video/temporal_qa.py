"""
Long-Video Temporal QA Engine.
Performs hierarchical coarse-to-fine temporal reasoning across long videos,
answering temporal localization ('when'), recurrence ('how many times'),
and causal ordering ('what happened after') questions with grounded timestamp intervals.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from src.memory.episodic import EpisodicVectorMemory
from src.video.events import SemanticEvent

logger = logging.getLogger("OmniVision.Video.TemporalQA")


class TemporalQAResponse(BaseModel):
    """Calibrated answer to a long-video temporal query with time intervals and evidence."""
    query: str
    query_type: str  # 'localization', 'recurrence', 'causal', 'summary'
    answer: str
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None
    peak_time_sec: Optional[float] = None
    target_event_count: Optional[int] = None
    cited_event_ids: List[str] = Field(default_factory=list)
    cited_frame_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)


class TemporalQAEngine:
    """
    Hierarchical Temporal Question Answering Engine for long video.
    Executes coarse-to-fine searches over episodic memory and segmented events.
    """

    def __init__(self, memory: EpisodicVectorMemory, fps: float = 30.0) -> None:
        self.memory = memory
        self.fps = fps

    def answer_query(
        self,
        query: str,
        segmented_events: Optional[List[SemanticEvent]] = None,
    ) -> TemporalQAResponse:
        """
        Answer open-ended temporal question over long video history.
        """
        q_lower = query.lower().strip()
        events = segmented_events or []

        # Classify Temporal Query Type
        if any(w in q_lower for w in ["when", "what time", "at what moment", "timestamp", "which frame"]):
            return self._handle_localization_query(query, q_lower, events)
        elif any(w in q_lower for w in ["how many times", "count", "frequency", "how often", "number of times"]):
            return self._handle_recurrence_query(query, q_lower, events)
        elif any(w in q_lower for w in ["after", "then", "following", "before", "prior to", "cause", "next"]):
            return self._handle_causal_query(query, q_lower, events)
        else:
            return self._handle_summary_query(query, q_lower, events)

    def _handle_localization_query(
        self,
        query: str,
        q_lower: str,
        events: List[SemanticEvent],
    ) -> TemporalQAResponse:
        """Handle 'When did X happen?' by locating the matching event interval."""
        # Find matching events by keyword / entity
        scored_events: List[Tuple[int, float, SemanticEvent]] = []
        tokens = [w for w in re.findall(r"\b\w{3,}\b", q_lower) if w not in ["when", "did", "the", "happen", "occur", "take", "place", "video"]]
        for ev in events:
            ev_text = f"{ev.event_type} {ev.description} {' '.join(ev.evidence_tags)}".lower()
            score = 0
            for tok in tokens:
                if tok in ev_text or (tok == "fall" and "fell" in ev_text):
                    score += 2
                elif any(tok[:4] in w for w in ev_text.split()):
                    score += 1
            if score > 0:
                scored_events.append((score, ev.confidence, ev))

        if scored_events:
            scored_events.sort(key=lambda x: (x[0], x[1]), reverse=True)
            target_ev = scored_events[0][2]
            peak_time = target_ev.peak_time_sec if target_ev.peak_time_sec is not None else round(target_ev.peak_frame / self.fps, 3)
            ans = (
                f"The event was detected between {target_ev.start_time_sec:.1f}s and {target_ev.end_time_sec:.1f}s "
                f"(peak at {peak_time:.1f}s, frame {target_ev.peak_frame}). {target_ev.description}"
            )
            return TemporalQAResponse(
                query=query,
                query_type="localization",
                answer=ans,
                start_time_sec=target_ev.start_time_sec,
                end_time_sec=target_ev.end_time_sec,
                peak_time_sec=peak_time,
                cited_event_ids=[target_ev.event_id],
                confidence=target_ev.confidence,
                timeline=[target_ev.model_dump()],
            )

        # Fallback to clip-level memory scan
        clips = self.memory.search_clips(top_k=3)
        if clips:
            best_clip, sim = clips[0]
            ans = f"Event approximately occurred during clip at {best_clip.start_time:.1f}s - {best_clip.end_time:.1f}s ({best_clip.activity_summary})."
            return TemporalQAResponse(
                query=query,
                query_type="localization",
                answer=ans,
                start_time_sec=best_clip.start_time,
                end_time_sec=best_clip.end_time,
                cited_frame_ids=[best_clip.representative_frame_id],
                confidence=0.75,
            )

        return TemporalQAResponse(
            query=query,
            query_type="localization",
            answer="No matching temporal occurrence could be located in the video memory.",
            confidence=0.30,
        )

    def _handle_recurrence_query(
        self,
        query: str,
        q_lower: str,
        events: List[SemanticEvent],
    ) -> TemporalQAResponse:
        """Handle 'How many times did X happen?' with recurrence counting and interval list."""
        tokens = [w for w in re.findall(r"\b\w{3,}\b", q_lower) if w not in ["how", "many", "times", "did", "the", "occur", "number"]]
        matched_events: List[SemanticEvent] = []

        for ev in events:
            ev_text = f"{ev.event_type} {ev.description} {' '.join(ev.evidence_tags)}".lower()
            if any(tok in ev_text for tok in tokens):
                matched_events.append(ev)

        count = len(matched_events)
        timestamps_str = ", ".join(f"{e.start_time_sec:.1f}s" for e in matched_events) if matched_events else "none"
        ans = f"The target action occurred {count} time(s) across the video (at timestamps: {timestamps_str})."

        return TemporalQAResponse(
            query=query,
            query_type="recurrence",
            answer=ans,
            target_event_count=count,
            cited_event_ids=[e.event_id for e in matched_events],
            confidence=0.88 if count > 0 else 0.70,
            timeline=[e.model_dump() for e in matched_events],
        )

    def _handle_causal_query(
        self,
        query: str,
        q_lower: str,
        events: List[SemanticEvent],
    ) -> TemporalQAResponse:
        """Handle 'What happened after/before X?' by causal sequence ordering."""
        # Find pivot event
        sorted_events = sorted(events, key=lambda e: e.start_time_sec)
        pivot_event: Optional[SemanticEvent] = None
        is_after = "after" in q_lower or "then" in q_lower or "following" in q_lower

        tokens = [w for w in re.findall(r"\b\w{3,}\b", q_lower) if w not in ["what", "happened", "after", "before", "the", "then", "did"]]
        for ev in sorted_events:
            ev_text = f"{ev.event_type} {ev.description}".lower()
            if any(tok in ev_text for tok in tokens):
                pivot_event = ev
                break

        if pivot_event:
            p_time = pivot_event.end_time_sec if is_after else pivot_event.start_time_sec
            if is_after:
                subsequent = [e for e in sorted_events if e.start_time_sec >= p_time]
            else:
                subsequent = [e for e in sorted_events if e.end_time_sec <= p_time]

            if subsequent:
                next_ev = subsequent[0]
                relation_str = "After" if is_after else "Prior to"
                ans = f"{relation_str} '{pivot_event.description}' at {pivot_event.start_time_sec:.1f}s, the next event was '{next_ev.description}' at {next_ev.start_time_sec:.1f}s."
                return TemporalQAResponse(
                    query=query,
                    query_type="causal",
                    answer=ans,
                    start_time_sec=next_ev.start_time_sec,
                    end_time_sec=next_ev.end_time_sec,
                    cited_event_ids=[pivot_event.event_id, next_ev.event_id],
                    confidence=0.90,
                    timeline=[pivot_event.model_dump(), next_ev.model_dump()],
                )

        return TemporalQAResponse(
            query=query,
            query_type="causal",
            answer="Could not determine the causal sequence for the specified reference event.",
            confidence=0.40,
        )

    def _handle_summary_query(
        self,
        query: str,
        q_lower: str,
        events: List[SemanticEvent],
    ) -> TemporalQAResponse:
        """Summarize overall temporal narrative."""
        if not events:
            return TemporalQAResponse(
                query=query,
                query_type="summary",
                answer="Video log contains no distinct segmented events.",
                confidence=0.50,
            )

        top_events = sorted(events, key=lambda e: e.start_time_sec)[:5]
        bullets = [f"- At {e.start_time_sec:.1f}s: {e.description}" for e in top_events]
        ans = f"Video activity summary ({len(events)} total events detected):\n" + "\n".join(bullets)

        return TemporalQAResponse(
            query=query,
            query_type="summary",
            answer=ans,
            start_time_sec=top_events[0].start_time_sec,
            end_time_sec=top_events[-1].end_time_sec,
            cited_event_ids=[e.event_id for e in top_events],
            confidence=0.92,
            timeline=[e.model_dump() for e in top_events],
        )
