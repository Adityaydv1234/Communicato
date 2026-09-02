class PostureAnalyzer:
    """Common interface every posture-analysis backend implements."""

    name = "base"

    def analyze(self, landmark_summary: dict | None, keyframes: list[str] | None) -> dict:
        """
        landmark_summary: aggregated MediaPipe metrics computed client-side (or None).
        keyframes: list of base64 JPEG data URLs sampled during the session (or None).

        Returns a dict with: posture_score, eye_contact_score, gesture_score, feedback.
        """
        raise NotImplementedError
