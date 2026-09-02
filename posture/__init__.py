import os

from posture.hybrid_analyzer import HybridAnalyzer
from posture.mediapipe_analyzer import MediaPipeAnalyzer
from posture.vision_analyzer import VisionAnalyzer

_ANALYZERS = {
    "mediapipe": MediaPipeAnalyzer,
    "openai_vision": VisionAnalyzer,
    "hybrid": HybridAnalyzer,
}


def get_analyzer_name() -> str:
    return os.getenv("POSTURE_ANALYZER", "mediapipe").lower()


def get_analyzer():
    name = get_analyzer_name()
    cls = _ANALYZERS.get(name, MediaPipeAnalyzer)
    return cls()
