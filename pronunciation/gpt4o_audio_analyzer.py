import base64
import json
import os

from openai import OpenAI

from pronunciation.base import PronunciationAnalyzer

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


class GPT4oAudioAnalyzer(PronunciationAnalyzer):
    """Sends the raw audio to GPT-4o's audio model and asks it to judge pronunciation
    qualitatively. Uses the existing OpenAI key - no new account needed."""

    name = "gpt4o_audio"

    def analyze(self, word: str, wav_bytes: bytes) -> dict:
        b64_audio = base64.b64encode(wav_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-audio-mini",
            modalities=["text"],
            messages=[
                {
                    "role": "system",
                    "content": f"You are a pronunciation coach. The user is trying to say the "
                    f"word '{word}'. Listen to the audio and judge whether they pronounced it "
                    f"correctly. Respond with ONLY a JSON object, no other text, in exactly this "
                    f'shape: {{"correct": true or false, "accuracy_score": 0-100, "feedback": '
                    f'"1-2 sentences, specific about what was off if anything, and how to say it '
                    f'correctly"}}',
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": b64_audio, "format": "wav"}},
                    ],
                },
            ],
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
        start, end = raw.find("{"), raw.rfind("}")
        result = json.loads(raw[start : end + 1])
        overall = result.get("accuracy_score")
        return {
            "accuracy_score": overall,
            "fluency_score": None,
            "completeness_score": None,
            "overall_score": overall,
            "correct": result.get("correct", False),
            "feedback": result.get("feedback", ""),
        }
