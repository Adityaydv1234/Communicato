import difflib
import os
import tempfile

from pronunciation.base import PronunciationAnalyzer


class WhisperFuzzyAnalyzer(PronunciationAnalyzer):
    """Free fallback: transcribes with Whisper and fuzzy-matches against the target word.
    Whisper auto-corrects mispronunciations, so this gives frequent false passes - use it
    only when Azure/GPT-4o audio aren't available."""

    name = "whisper_fuzzy"

    def analyze(self, word: str, wav_bytes: bytes) -> dict:
        import ai  # local import avoids a hard dependency at package import time

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            wav_path = f.name
        try:
            transcript = ai.transcribe_audio(wav_path).strip().lower()
        finally:
            os.unlink(wav_path)

        similarity = difflib.SequenceMatcher(None, transcript, word.lower()).ratio()
        score = round(similarity * 100)
        correct = transcript == word.lower()

        feedback = f'Whisper heard: "{transcript}". '
        feedback += (
            "That matches!"
            if correct
            else f'That\'s close to "{word}" but not exact - note Whisper often auto-corrects '
            f"mispronunciations, so treat this as a rough signal only."
        )

        return {
            "accuracy_score": score,
            "fluency_score": None,
            "completeness_score": None,
            "overall_score": score,
            "correct": correct,
            "feedback": feedback,
        }
