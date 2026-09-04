import os
import tempfile

from pronunciation.base import PronunciationAnalyzer


class AzureAnalyzer(PronunciationAnalyzer):
    """Purpose-built pronunciation scoring: per-phoneme accuracy, fluency, and
    completeness, via Azure Cognitive Services Speech SDK."""

    name = "azure"

    def analyze(self, word: str, wav_bytes: bytes) -> dict:
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as e:
            raise RuntimeError(
                "azure-cognitiveservices-speech is not installed. Run: "
                "pip install azure-cognitiveservices-speech"
            ) from e

        key = os.getenv("AZURE_SPEECH_KEY")
        region = os.getenv("AZURE_SPEECH_REGION")
        if not key or not region:
            raise RuntimeError(
                "AZURE_SPEECH_KEY / AZURE_SPEECH_REGION are not set in .env. "
                "Get a free key at https://portal.azure.com (Speech service, F0 tier)."
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            wav_path = f.name

        try:
            speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
            pron_config = speechsdk.PronunciationAssessmentConfig(
                reference_text=word,
                grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                granularity=speechsdk.PronunciationAssessmentGranularity.Word,
            )
            audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, audio_config=audio_config
            )
            pron_config.apply_to(recognizer)

            result = recognizer.recognize_once()
            if result.reason != speechsdk.ResultReason.RecognizedSpeech:
                return {
                    "accuracy_score": None,
                    "fluency_score": None,
                    "completeness_score": None,
                    "overall_score": None,
                    "correct": False,
                    "feedback": "Could not recognize speech clearly. Try again in a quiet "
                    "room, closer to the mic.",
                }

            pron_result = speechsdk.PronunciationAssessmentResult(result)
            overall = round(pron_result.pronunciation_score)
            correct = overall >= 80

            feedback = (
                f"Accuracy {round(pron_result.accuracy_score)}/100, "
                f"fluency {round(pron_result.fluency_score)}/100, "
                f"completeness {round(pron_result.completeness_score)}/100."
            )
            feedback += (
                " Well pronounced!"
                if correct
                else f" Try breaking '{word}' into syllables and pronouncing each one clearly."
            )

            return {
                "accuracy_score": round(pron_result.accuracy_score),
                "fluency_score": round(pron_result.fluency_score),
                "completeness_score": round(pron_result.completeness_score),
                "overall_score": overall,
                "correct": correct,
                "feedback": feedback,
            }
        finally:
            os.unlink(wav_path)
