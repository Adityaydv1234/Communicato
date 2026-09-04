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


def describe_image(image_url: str) -> str:
    """One-time GPT-4o Vision call per image, cached by the caller. This description is
    never shown to the user - it's the ground truth used to score how accurately they
    described the image aloud."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in one detailed paragraph: the setting, "
                        "people, objects, actions, and mood. Be specific and thorough - this "
                        "will be used as a reference to judge how well someone else describes "
                        "the same image aloud.",
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


IMAGE_EVAL_SCHEMA_INSTRUCTIONS = """Return a JSON object with exactly these fields:
{
  "fluency_score": <0-100 integer>,
  "clarity_score": <0-100 integer>,
  "structure_score": <0-100 integer>,
  "vocabulary_score": <0-100 integer>,
  "confidence_score": <0-100 integer>,
  "relevance_score": <0-100 integer>,
  "overall_score": <0-100 integer>,
  "primary_weakness": "<one of: fluency|clarity|structure|vocabulary|confidence|relevance>",
  "weakness_explanation": "<2-3 sentences explaining this specific weakness with examples from the transcript>",
  "exercise": "<a specific exercise the user should try in their next session>"
}"""


def evaluate_image_description(challenge: dict, history: list[dict]) -> dict:
    transcript = "\n".join(m["content"] for m in history if m["role"] == "user")

    system_prompt = f"""You are an expert communication coach analyzing someone describing an
image aloud.

Here is what is actually in the image (ground truth, for your reference only - the user never
saw this text, only the image itself):
{challenge['image_description']}

{IMAGE_EVAL_SCHEMA_INSTRUCTIONS}

Scoring guidelines:
- Fluency: flow of speech, filler words, hesitation patterns
- Clarity: how well ideas are communicated and understood
- Structure: logical organization (e.g. foreground to background, general to specific)
- Vocabulary: range and appropriateness of descriptive word choices
- Confidence: assertiveness, hedging language, conviction
- Relevance: how accurately and thoroughly they described what is ACTUALLY in the image, based
  on the ground truth above. Penalize invented details and reward specific, accurate observations.

Identify the ONE weakness that would make the biggest difference if improved.
Be specific - reference actual phrases from the transcript and actual details from the image."""

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


def interview_system_prompt(challenge: dict, resume_text: str) -> str:
    return f"""You are conducting a mock job interview for a {challenge['domain']} role.

Candidate's resume:
{resume_text}

Rules:
- Ask ONE question at a time, based on a natural mix of: (a) the candidate's resume (their
  projects, experience, skills), and (b) general {challenge['domain']} domain knowledge and
  behavioral questions.
- Listen to their answer and ask natural, specific follow-up questions when it makes sense
  (probe deeper, ask for examples), just like a real interviewer would.
- Keep questions concise (1-3 sentences).
- Sound professional but warm - this is practice, not a hostile interview.
- Never break character or mention you are an AI.
- Do not repeat a question you've already asked.
- Vary between technical/role questions and behavioral questions ('tell me about a time...')."""


def get_interview_reply(challenge: dict, resume_text: str, history: list[dict]) -> str:
    messages = [{"role": "system", "content": interview_system_prompt(challenge, resume_text)}]
    for m in history:
        role = "assistant" if m["role"] == "assistant" else "user"
        messages.append({"role": role, "content": m["content"]})
    if not history:
        messages.append(
            {"role": "user", "content": "Please begin the interview with your first question."}
        )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


def generate_interview_report(challenge: dict, resume_text: str, history: list[dict]) -> dict:
    transcript = "\n".join(
        f"{'INTERVIEWER' if m['role'] == 'assistant' else 'CANDIDATE'}: {m['content']}"
        for m in history
    )

    system_prompt = f"""You are an expert interview coach reviewing a completed mock interview
transcript for a {challenge['domain']} role.

Candidate's resume:
{resume_text}

Full interview transcript:
{transcript}

Analyze the CANDIDATE's answers only. Return a JSON object with exactly these fields:
{{
  "overall_score": <0-100 integer>,
  "strengths": ["<specific strength 1>", "<specific strength 2>"],
  "weaknesses": ["<specific weakness 1>", "<specific weakness 2>"],
  "improved_answers": [
    {{"question": "<the interviewer's question>", "your_answer": "<what the candidate actually said, shortened>", "better_answer": "<a stronger example answer>"}}
  ],
  "summary": "<3-4 sentence overall coaching summary - encouraging but honest>"
}}

strengths and weaknesses should each have 2-4 items. improved_answers should cover the 2-3
weakest answers. Be specific - reference actual content from their answers, not generic
feedback."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}],
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    return json.loads(response.choices[0].message.content)


def generate_daily_words() -> list[dict]:
    system_prompt = """Generate 10 moderately advanced English words that are genuinely useful
in everyday conversation and writing (not overly obscure or academic-only). For each word give
its meaning in plain simple English, and one natural example sentence using it.

Return a JSON object: {"words": [{"word": "...", "meaning": "...", "example_sentence": "..."}, ...]}
with exactly 10 items."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}],
        response_format={"type": "json_object"},
        temperature=0.9,
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("words", [])


def evaluate_sentence_drill(word: str, meaning: str, transcript: str) -> dict:
    system_prompt = f"""The user was asked to speak a sentence using the word "{word}"
(meaning: {meaning}) correctly and naturally.

They said: "{transcript}"

Judge whether they used the word correctly (right meaning, right grammatical form) and
naturally (not forced). Return JSON: {{"correct": <boolean>, "feedback": "<1-2 sentences,
specific>"}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return json.loads(response.choices[0].message.content)


def mine_grammar_drills(user_transcripts: list[str]) -> list[dict]:
    if not user_transcripts:
        return []

    combined = "\n---\n".join(user_transcripts[:40])

    system_prompt = """Below are transcripts of things a non-native English speaker said during
speaking practice sessions. Identify up to 4 RECURRING grammar mistake patterns (not one-off
typos) - things like tense errors, article misuse (a/an/the), preposition errors, subject-verb
agreement, plural/singular confusion, etc.

For each pattern found, return:
{
  "mistake_pattern": "<short name, e.g. 'Missing articles (a/an/the)'>",
  "original_example": "<an actual quote from the transcripts showing this mistake>",
  "corrected_example": "<the corrected version of that same sentence>",
  "exercise_prompt": "<a short speaking exercise to practice fixing this specific mistake>"
}

Return JSON: {"drills": [...]}. If there are fewer than 4 clear recurring patterns, return
fewer. If you find none, return {"drills": []}."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("drills", [])


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
