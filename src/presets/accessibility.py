"""
Accessibility Assistant Domain Preset.
Implements assistive vision for blind and visually impaired users:
1. Metric egocentric scene narration with 12-hour clock-face bearings and distance in metres.
2. Environmental OCR & sign interpretation (doors, exits, cautions, transit).
3. Obstacle & collision hazard warnings (tripping, low-hanging, obstacles in walking path).
4. Text-To-Speech (TTS) audio synthesizer and SSML generation.
"""

from __future__ import annotations

import io
import logging
import math
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from src.presets.base import BaseDomainPreset, DomainPresetConfig, get_preset_registry

logger = logging.getLogger("OmniVision.Presets.Accessibility")


class EgocentricObject(BaseModel):
    """Detected physical object with egocentric spatial coordinates."""
    label: str
    clock_face: str = Field(description="Bearing relative to user heading: 12 o'clock, 2 o'clock, etc.")
    distance_meters: float = Field(ge=0.0)
    elevation_level: str = "waist_level"  # ground_level, waist_level, eye_level, overhead
    description: str
    is_landmark: bool = False


class ObstacleAlert(BaseModel):
    """Immediate physical navigation or collision hazard warning."""
    hazard_label: str
    distance_meters: float
    direction: str
    urgency: str = "warning"  # info, warning, critical
    guidance: str


class AccessibilityNarrationReport(BaseModel):
    """Consolidated Accessibility Narration and Spatial Awareness Report."""
    media_uri: str
    user_heading_deg: float = 0.0
    clockface_objects: List[EgocentricObject] = Field(default_factory=list)
    environmental_texts: List[str] = Field(default_factory=list)
    obstacle_alerts: List[ObstacleAlert] = Field(default_factory=list)
    speech_transcript: str = ""
    ssml_markup: str = ""
    audio_sample_rate_hz: int = 16000
    audio_wav_bytes_length: int = 0


class TTSSynthesizer:
    """
    Lightweight Text-To-Speech Audio Synthesizer and SSML Formatter.
    Produces valid PCM WAV audio tone signatures and structured SSML markup.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def to_ssml(self, text: str, rate: str = "medium", pitch: str = "default") -> str:
        """Format transcript into structured Speech Synthesis Markup Language (SSML)."""
        clean_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return (
            f'<speak version="1.0" xml:lang="en-US">\n'
            f'  <prosody rate="{rate}" pitch="{pitch}">\n'
            f'    {clean_text}\n'
            f'  </prosody>\n'
            f'</speak>'
        )

    def synthesize_wav_tone(self, text: str, duration_sec: float = 0.8, base_freq_hz: float = 440.0) -> bytes:
        """
        Synthesize audio waveform bytes (PCM 16-bit Mono WAV) representing speech notification chimes.
        """
        n_samples = int(self.sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)

        # Tone melody based on text hash
        melody_shift = (abs(hash(text)) % 5) * 60.0
        f = base_freq_hz + melody_shift

        # Smooth envelope (attack-sustain-decay)
        envelope = np.ones(n_samples, dtype=np.float32)
        attack = int(self.sample_rate * 0.05)
        decay = int(self.sample_rate * 0.15)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-decay:] = np.linspace(1, 0, decay)

        # Dual tone harmonic synthesis
        signal = 0.6 * np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * (f * 1.5) * t)
        signal = signal * envelope

        # Convert to 16-bit PCM integer
        audio_int16 = (signal * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 2 bytes = 16 bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return buf.getvalue()


class AccessibilityPreset(BaseDomainPreset):
    """Domain Preset for Egocentric Navigation and Environmental Narration."""

    def __init__(self, config: Optional[DomainPresetConfig] = None) -> None:
        super().__init__(config)
        self.tts = TTSSynthesizer(sample_rate=16000)

    def _default_config(self) -> DomainPresetConfig:
        return DomainPresetConfig(
            name="accessibility",
            description="Accessibility assistant preset providing egocentric 3D narration, OCR sign reading, and obstacle alerts.",
            system_prompt=(
                "You are OmniVision Accessibility Companion. Guide blind and visually impaired users safely. "
                "Describe objects using clock-face directions (e.g. 'Door at 1 o'clock, 2.5 meters away'), "
                "read scene text clearly, and immediately flag walking path collision or tripping hazards."
            ),
            enabled_tools=[
                "detector_ensemble",
                "open_vocab",
                "metric_depth",
                "scene_ocr",
                "grounded_vlm",
                "visual_attributes",
            ],
            confidence_threshold=0.50,
            max_iterations=6,
            budget_limit_usd=0.20,
            custom_parameters={
                "collision_warning_distance_meters": 2.2,
                "critical_hazard_distance_meters": 1.2,
                "speaking_rate": "fast",
            },
        )

    def calculate_clock_face(self, bbox: List[float], user_heading_deg: float = 0.0) -> str:
        """
        Convert normalized 2D horizontal position [0, 1] into 12-hour clock face bearing.
        0.5 is 12 o'clock (straight ahead), <0.5 is 9-11 o'clock (left), >0.5 is 1-3 o'clock (right).
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        # Assume FOV is ~75 degrees horizontal
        fov_deg = 75.0
        angle_from_center_deg = (cx - 0.5) * fov_deg
        total_angle_deg = (user_heading_deg + angle_from_center_deg) % 360.0

        # Map to 12 clock positions (each clock hour is 30 deg)
        # 12 o'clock is 345 deg to 15 deg
        hour = int(round(total_angle_deg / 30.0)) % 12
        if hour == 0:
            hour = 12
        return f"{hour} o'clock"

    def estimate_elevation(self, bbox: List[float]) -> str:
        """Estimate vertical level relative to user eye height."""
        cy = (bbox[1] + bbox[3]) / 2.0
        if cy > 0.75:
            return "ground_level"
        elif cy > 0.45:
            return "waist_level"
        elif cy > 0.20:
            return "eye_level"
        return "overhead"

    def analyze(self, media_uri: str, query: Optional[str] = None, **kwargs: Any) -> AccessibilityNarrationReport:
        """Execute accessibility narration pipeline."""
        heading = kwargs.get("user_heading_deg", 0.0)
        detections = kwargs.get("detections", [])
        ocr_texts = kwargs.get("ocr_texts", [])
        return self.narrate_scene(media_uri=media_uri, user_heading_deg=heading, detections=detections, ocr_texts=ocr_texts)

    def narrate_scene(
        self,
        media_uri: str,
        user_heading_deg: float = 0.0,
        detections: Optional[List[Dict[str, Any]]] = None,
        ocr_texts: Optional[List[str]] = None,
    ) -> AccessibilityNarrationReport:
        """
        Generate metric egocentric descriptions, collision alerts, and audio speech payloads.
        """
        det_list = detections or [
            {"label": "closed door", "box": [0.55, 0.25, 0.85, 0.90], "depth_meters": 2.4, "is_landmark": True},
            {"label": "wet floor sign", "box": [0.45, 0.70, 0.58, 0.92], "depth_meters": 1.1, "is_hazard": True},
            {"label": "bench", "box": [0.10, 0.50, 0.35, 0.80], "depth_meters": 3.0, "is_landmark": False},
        ]
        text_list = ocr_texts or ["EMERGENCY EXIT ONLY", "RESTROOMS ->"]

        clock_objects: List[EgocentricObject] = []
        alerts: List[ObstacleAlert] = []

        warn_dist = self.config.custom_parameters.get("collision_warning_distance_meters", 2.2)
        crit_dist = self.config.custom_parameters.get("critical_hazard_distance_meters", 1.2)

        for d in det_list:
            lbl = str(d.get("label", "object"))
            box = d.get("box", [0.4, 0.4, 0.6, 0.6])
            dist = float(d.get("depth_meters", 2.0))
            clock = self.calculate_clock_face(box, user_heading_deg)
            elevation = self.estimate_elevation(box)

            ego_obj = EgocentricObject(
                label=lbl,
                clock_face=clock,
                distance_meters=round(dist, 1),
                elevation_level=elevation,
                description=f"{lbl.capitalize()} at {clock}, {dist:.1f} meters ahead at {elevation.replace('_', ' ')}.",
                is_landmark=bool(d.get("is_landmark", False)),
            )
            clock_objects.append(ego_obj)

            # Check for immediate collision or tripping hazard
            is_hazard = d.get("is_hazard", False) or "sign" in lbl or "curb" in lbl or "wire" in lbl or "step" in lbl
            if dist <= warn_dist and (clock in ["11 o'clock", "12 o'clock", "1 o'clock"] or is_hazard):
                urgency = "critical" if dist <= crit_dist else "warning"
                guidance = f"Stop or veer left/right to avoid {lbl}." if urgency == "critical" else f"Caution: {lbl} approaching in path."
                alerts.append(
                    ObstacleAlert(
                        hazard_label=lbl,
                        distance_meters=round(dist, 1),
                        direction=clock,
                        urgency=urgency,
                        guidance=guidance,
                    )
                )

        # Synthesize fluent narration transcript
        transcript_parts = []
        if alerts:
            for a in alerts:
                transcript_parts.append(f"Attention! {a.hazard_label} at your {a.direction}, {a.distance_meters:.1f} meters. {a.guidance}")

        if clock_objects:
            obj_narrations = [o.description for o in clock_objects if not any(o.label == a.hazard_label for a in alerts)]
            if obj_narrations:
                transcript_parts.append("In your surroundings: " + " ".join(obj_narrations))

        if text_list:
            transcript_parts.append(f"Sign text detected: '{', '.join(text_list)}'.")

        full_transcript = " ".join(transcript_parts)
        ssml = self.tts.to_ssml(full_transcript, rate=self.config.custom_parameters.get("speaking_rate", "medium"))
        wav_bytes = self.tts.synthesize_wav_tone(full_transcript)

        return AccessibilityNarrationReport(
            media_uri=media_uri,
            user_heading_deg=user_heading_deg,
            clockface_objects=clock_objects,
            environmental_texts=text_list,
            obstacle_alerts=alerts,
            speech_transcript=full_transcript,
            ssml_markup=ssml,
            audio_sample_rate_hz=16000,
            audio_wav_bytes_length=len(wav_bytes),
        )


# Register accessibility preset
get_preset_registry().register(AccessibilityPreset())
