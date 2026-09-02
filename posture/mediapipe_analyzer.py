from posture.base import PostureAnalyzer

# Expected shape of `landmark_summary`, computed client-side by MediaPipe Pose
# running in the browser (see static/index.html):
#   avg_shoulder_tilt_deg   - average deviation of shoulder line from horizontal
#   avg_head_tilt_deg       - average deviation of head from upright
#   hands_below_waist_ratio - fraction of samples where both hands stayed below waist (0-1)
#   movement_variance       - normalized sway/fidget amount (0-1, higher = more movement)
#   forward_lean_ratio      - fraction of samples leaning noticeably toward the camera (0-1)
#   samples                 - number of pose samples captured


def _score_from_thresholds(value, good, ok, invert=False):
    """Map a raw metric to a 0-100 score using two thresholds."""
    if invert:
        if value <= good:
            return 95
        if value <= ok:
            return 70
        return 45
    if value >= good:
        return 95
    if value >= ok:
        return 70
    return 45


class MediaPipeAnalyzer(PostureAnalyzer):
    name = "mediapipe"

    def analyze(self, landmark_summary, keyframes=None) -> dict:
        if not landmark_summary or not landmark_summary.get("samples"):
            return {
                "posture_score": None,
                "eye_contact_score": None,
                "gesture_score": None,
                "feedback": "No pose data was captured for this session.",
            }

        shoulder_tilt = landmark_summary.get("avg_shoulder_tilt_deg", 0)
        head_tilt = landmark_summary.get("avg_head_tilt_deg", 0)
        hands_below_waist = landmark_summary.get("hands_below_waist_ratio", 0)
        movement_variance = landmark_summary.get("movement_variance", 0)
        forward_lean = landmark_summary.get("forward_lean_ratio", 0)

        posture_score = _score_from_thresholds(shoulder_tilt, good=4, ok=9, invert=True)
        eye_contact_score = _score_from_thresholds(head_tilt, good=6, ok=14, invert=True)
        gesture_score = round(
            100 - (hands_below_waist * 60) - (movement_variance * 25)
        )
        gesture_score = max(20, min(100, gesture_score))

        notes = []
        if shoulder_tilt > 9:
            notes.append("your shoulders were noticeably uneven or slouched")
        if head_tilt > 14:
            notes.append("your head tilted away from the camera often, which reads as broken eye contact")
        if hands_below_waist > 0.7:
            notes.append("your hands stayed down for most of the session instead of gesturing")
        if movement_variance > 0.6:
            notes.append("there was a lot of swaying or fidgeting")
        if forward_lean > 0.5:
            notes.append("you leaned toward the camera for a large part of the session")

        if not notes:
            feedback = "Solid posture overall - upright, steady, and hands were active. Keep this up."
        else:
            feedback = "Posture notes: " + "; ".join(notes) + "."

        return {
            "posture_score": posture_score,
            "eye_contact_score": eye_contact_score,
            "gesture_score": gesture_score,
            "feedback": feedback,
        }
