import os

from pronunciation.azure_analyzer import AzureAnalyzer
from pronunciation.gpt4o_audio_analyzer import GPT4oAudioAnalyzer
from pronunciation.whisper_fuzzy_analyzer import WhisperFuzzyAnalyzer

_ANALYZERS = {
    "azure": AzureAnalyzer,
    "gpt4o_audio": GPT4oAudioAnalyzer,
    "whisper_fuzzy": WhisperFuzzyAnalyzer,
}


def get_analyzer_name() -> str:
    return os.getenv("PRONUNCIATION_ENGINE", "azure").lower()


def get_analyzer():
    name = get_analyzer_name()
    cls = _ANALYZERS.get(name, AzureAnalyzer)
    return cls()
