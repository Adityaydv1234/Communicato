import json
import os

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

TTS_VOICE = "nova"


def transcribe_audio(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(model="whisper-1", file=f)
    return result.text.strip()


def conversation_system_prompt(challenge: dict) -> str:
    return f"""You are {challenge['ai_persona']} in the following scenario:
{challenge['scenario']}

Rules:
- Stay fully in character. Never mention that you're an AI.
- Respond in 2-4 sentences. Keep it natural and conversational.
- If the user seems stuck, gently move the conversation forward as your character naturally would.
- Match the user's energy - if they're casual, be casual.
- The conversation should feel like a real interaction, not a test."""


def get_ai_reply(challenge: dict, history: list[dict]) -> str:
    messages = [{"role": "system", "content": conversation_system_prompt(challenge)}]
    for m in history:
        role = "assistant" if m["role"] == "assistant" else "user"
        messages.append({"role": role, "content": m["content"]})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


def synthesize_speech(text: str, out_path: str) -> None:
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=TTS_VOICE,
        input=text,
    ) as response:
        response.stream_to_file(out_path)


EVAL_SCHEMA_INSTRUCTIONS = """Return a JSON object with exactly these fields:
{
  "fluency_score": <0-100 integer>,
  "clarity_score": <0-100 integer>,
  "structure_score": <0-100 integer>,
  "vocabulary_score": <0-100 integer>,
  "confidence_score": <0-100 integer>,
  "overall_score": <0-100 integer>,
  "primary_weakness": "<one of: fluency|clarity|structure|vocabulary|confidence>",
  "weakness_explanation": "<2-3 sentences explaining this specific weakness with examples from the conversation>",
  "exercise": "<a specific exercise the user should try in their next session>"
}"""


def evaluate_session(challenge: dict, history: list[dict]) -> dict:
    transcript = "\n".join(
        f"{'USER' if m['role'] == 'user' else 'AI'}: {m['content']}" for m in history
    )

    system_prompt = f"""You are an expert communication coach analyzing a practice conversation.

The user practiced the following scenario:
{challenge['scenario']}

Analyze the USER's messages only (not the AI's).

{EVAL_SCHEMA_INSTRUCTIONS}

Scoring guidelines:
- Fluency: flow of speech, filler words, hesitation patterns
- Clarity: how well ideas are communicated and understood
- Structure: logical organization of thoughts and responses
- Vocabulary: range and appropriateness of word choices
- Confidence: assertiveness, hedging language, conviction

Identify the ONE weakness that would make the biggest difference if improved.
Be specific - reference actual phrases from the transcript."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return json.loads(response.choices[0].message.content)


def generate_daily_tip(recent_evaluations: list[dict]) -> str:
    """recent_evaluations: list of {primary_weakness, weakness_explanation, exercise, created_at}
    ordered most-recent-first, from the last several sessions."""
    if not recent_evaluations:
        return "Complete a session to get your first personalized tip."

    history_text = "\n".join(
        f"- {e['created_at']}: weakness={e['primary_weakness']} - {e['weakness_explanation']}"
        for e in recent_evaluations
    )

    system_prompt = """You are a communication coach reviewing a student's recent practice history.
Identify the pattern across sessions (a recurring weakness, or a weakness that is improving and what's
emerging next) and give ONE short, specific, encouraging tip for today's practice. 2-3 sentences max.
Do not repeat generic advice - reference what actually happened across their sessions."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": history_text},
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()
