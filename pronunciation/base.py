class PronunciationAnalyzer:
    """Common interface every pronunciation-checking backend implements."""

    name = "base"

    def analyze(self, word: str, wav_bytes: bytes) -> dict:
        """
        word: the target word the user was asked to pronounce.
        wav_bytes: 16-bit PCM WAV audio of their attempt.

        Returns a dict with: accuracy_score, fluency_score, completeness_score,
        overall_score (any may be None if the engine doesn't provide it), correct (bool),
        feedback (str).
        """
        raise NotImplementedError
