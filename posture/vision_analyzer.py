import json
import os

from openai import OpenAI

from posture.base import PostureAnalyzer

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

VISION_PROMPT = """You are a presentation coach analyzing frames captured at intervals during
someone's speech. Judge their posture, eye contact with the camera, and hand gestures across
the frames as a whole.

Return a JSON object with exactly these fields:
{
  "posture_score": <0-100 integer>,
  "eye_contact_score": <0-100 integer>,
  "gesture_score": <0-100 integer>,
  "feedback": "<2-4 sentences of specific, actionable feedback on posture, eye contact, and gestures>"
}"""


class VisionAnalyzer(PostureAnalyzer):
    name = "openai_vision"

    def analyze(self, landmark_summary=None, keyframes: list[str] | None = None) -> dict:
        if not keyframes:
            return {
                "posture_score": None,
                "eye_contact_score": None,
                "gesture_score": None,
                "feedback": "No video frames were captured for this session.",
            }

        content = [{"type": "text", "text": VISION_PROMPT}]
        for frame in keyframes[:8]:
            content.append({"type": "image_url", "image_url": {"url": frame}})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return json.loads(response.choices[0].message.content)
