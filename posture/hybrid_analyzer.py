from posture.base import PostureAnalyzer
from posture.mediapipe_analyzer import MediaPipeAnalyzer
from posture.vision_analyzer import VisionAnalyzer


class HybridAnalyzer(PostureAnalyzer):
    """Uses MediaPipe for precise numeric scores and GPT-4o Vision for qualitative
    feedback on a handful of keyframes - cheaper than pure vision, richer than pure
    MediaPipe."""

    name = "hybrid"

    def __init__(self):
        self.mediapipe = MediaPipeAnalyzer()
        self.vision = VisionAnalyzer()

    def analyze(self, landmark_summary=None, keyframes=None) -> dict:
        mp_result = self.mediapipe.analyze(landmark_summary, keyframes)

        if not keyframes:
            return mp_result

        vision_result = self.vision.analyze(landmark_summary, keyframes[:4])

        return {
            "posture_score": mp_result.get("posture_score") or vision_result.get("posture_score"),
            "eye_contact_score": mp_result.get("eye_contact_score") or vision_result.get("eye_contact_score"),
            "gesture_score": mp_result.get("gesture_score") or vision_result.get("gesture_score"),
            "feedback": vision_result.get("feedback", mp_result.get("feedback")),
        }
