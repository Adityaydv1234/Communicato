import json
import os
import tempfile
import uuid
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import ai
import filler_words
import images
import posture
import pronunciation
import resume as resume_parser
from db import get_conn, init_db

REVIEW_SCHEDULE_DAYS = [3, 7, 14]

AUDIO_DIR = Path(__file__).parent / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

app = FastAPI(title="SpeakUp")
init_db()

app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")


def row_to_dict(row):
    return dict(row) if row else None


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/config")
def get_config():
    return {
        "posture_analyzer": posture.get_analyzer_name(),
        "pronunciation_engine": pronunciation.get_analyzer_name(),
    }


@app.get("/api/challenges/random")
def random_challenge(mode: str = "conversation"):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM challenges WHERE mode = ? ORDER BY RANDOM() LIMIT 1", (mode,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "No challenges available")
    return row_to_dict(row)


@app.get("/api/images/random")
def random_image(query: str | None = None):
    conn = get_conn()
    try:
        img = images.fetch_random_image(query)
    except RuntimeError as e:
        conn.close()
        raise HTTPException(500, str(e))
    except Exception as e:
        conn.close()
        raise HTTPException(502, f"Could not fetch image from Unsplash: {e}")

    existing = conn.execute(
        "SELECT * FROM challenges WHERE image_unsplash_id = ?", (img["unsplash_id"],)
    ).fetchone()
    if existing:
        conn.close()
        result = row_to_dict(existing)
        result.pop("image_description", None)
        return result

    description = ai.describe_image(img["image_url"])
    cur = conn.execute(
        """INSERT INTO challenges
           (title, scenario, category, mode, ai_persona, image_url, image_description, image_unsplash_id)
           VALUES (?, ?, 'image-description', 'image_description', '', ?, ?, ?)""",
        (
            "Describe this image",
            "Look at the image and speak continuously, describing everything you notice - "
            "objects, people, setting, mood, and any story you imagine behind it.",
            img["image_url"],
            description,
            img["unsplash_id"],
        ),
    )
    challenge_id = cur.lastrowid
    conn.commit()
    row = conn.execute("SELECT * FROM challenges WHERE id = ?", (challenge_id,)).fetchone()
    conn.close()
    result = row_to_dict(row)
    result.pop("image_description", None)
    return result


@app.post("/api/sessions")
def create_session(
    challenge_id: int,
    mode: str = "conversation",
    target_duration_seconds: int | None = None,
    camera_enabled: bool = False,
):
    conn = get_conn()
    challenge = conn.execute(
        "SELECT * FROM challenges WHERE id = ?", (challenge_id,)
    ).fetchone()
    if not challenge:
        conn.close()
        raise HTTPException(404, "Challenge not found")

    cur = conn.execute(
        """INSERT INTO sessions (challenge_id, mode, target_duration_seconds, camera_enabled)
           VALUES (?, ?, ?, ?)""",
        (challenge_id, mode, target_duration_seconds, int(camera_enabled)),
    )
    session_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {"session_id": session_id, "challenge": row_to_dict(challenge)}


def get_session_history(conn, session_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at",
        (session_id,),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


# ---------------------------------------------------------------------------
# Conversation mode (turn-based)
# ---------------------------------------------------------------------------


@app.post("/api/sessions/{session_id}/message")
async def post_message(session_id: int, audio: UploadFile):
    conn = get_conn()
    session = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if not session:
        conn.close()
        raise HTTPException(404, "Session not found")
    challenge = conn.execute(
        "SELECT * FROM challenges WHERE id = ?", (session["challenge_id"],)
    ).fetchone()

    user_audio_name = f"{uuid.uuid4()}.webm"
    user_audio_path = AUDIO_DIR / user_audio_name
    contents = await audio.read()
    user_audio_path.write_bytes(contents)

    transcript = ai.transcribe_audio(str(user_audio_path))

    conn.execute(
        "INSERT INTO messages (session_id, role, content, audio_path) VALUES (?, 'user', ?, ?)",
        (session_id, transcript, user_audio_name),
    )
    conn.commit()

    history = get_session_history(conn, session_id)
    ai_text = ai.get_ai_reply(row_to_dict(challenge), history)

    ai_audio_name = f"{uuid.uuid4()}.mp3"
    ai_audio_path = AUDIO_DIR / ai_audio_name
    ai.synthesize_speech(ai_text, str(ai_audio_path))

    conn.execute(
        "INSERT INTO messages (session_id, role, content, audio_path) VALUES (?, 'assistant', ?, ?)",
        (session_id, ai_text, ai_audio_name),
    )
    conn.commit()
    conn.close()

    return {
        "transcript": transcript,
        "ai_text": ai_text,
        "ai_audio_url": f"/audio/{ai_audio_name}",
    }


def save_evaluation(conn, session_id: int, evaluation: dict):
    conn.execute(
        """INSERT INTO evaluations
           (session_id, fluency_score, clarity_score, structure_score, vocabulary_score,
            confidence_score, relevance_score, overall_score, primary_weakness,
            weakness_explanation, exercise, raw_evaluation)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            evaluation.get("fluency_score"),
            evaluation.get("clarity_score"),
            evaluation.get("structure_score"),
            evaluation.get("vocabulary_score"),
            evaluation.get("confidence_score"),
            evaluation.get("relevance_score"),
            evaluation.get("overall_score"),
            evaluation.get("primary_weakness"),
            evaluation.get("weakness_explanation"),
            evaluation.get("exercise"),
            json.dumps(evaluation),
        ),
    )


def save_posture(conn, session_id: int, posture_result: dict, analyzer_name: str):
    conn.execute(
        """INSERT INTO posture_evaluations
           (session_id, analyzer, posture_score, eye_contact_score, gesture_score, feedback, raw_data)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            analyzer_name,
            posture_result.get("posture_score"),
            posture_result.get("eye_contact_score"),
            posture_result.get("gesture_score"),
            posture_result.get("feedback"),
            json.dumps(posture_result),
        ),
    )


@app.post("/api/sessions/{session_id}/end")
def end_session(session_id: int):
    conn = get_conn()
    session = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if not session:
        conn.close()
        raise HTTPException(404, "Session not found")
    challenge = conn.execute(
        "SELECT * FROM challenges WHERE id = ?", (session["challenge_id"],)
    ).fetchone()

    history = get_session_history(conn, session_id)
    if not any(m["role"] == "user" for m in history):
        conn.execute(
            "UPDATE sessions SET status = 'abandoned', ended_at = datetime('now') WHERE id = ?",
            (session_id,),
        )
        conn.commit()
        conn.close()
        raise HTTPException(400, "No user messages to evaluate")

    evaluation = ai.evaluate_session(row_to_dict(challenge), history)
    save_evaluation(conn, session_id, evaluation)
    conn.execute(
        "UPDATE sessions SET status = 'completed', ended_at = datetime('now') WHERE id = ?",
        (session_id,),
    )
    conn.commit()
    conn.close()

    return evaluation


# ---------------------------------------------------------------------------
# Monologue mode (single continuous recording, optional camera/posture)
# ---------------------------------------------------------------------------


class MonologueSubmission(BaseModel):
    landmark_summary: dict | None = None
    keyframes: list[str] | None = None


@app.post("/api/monologue/{session_id}/submit")
async def submit_monologue(session_id: int, audio: UploadFile):
    """Audio is sent as multipart; posture data (landmarks/keyframes) is sent separately
    via /api/monologue/{session_id}/posture right before or after this call."""
    conn = get_conn()
    session = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if not session:
        conn.close()
        raise HTTPException(404, "Session not found")
    challenge = conn.execute(
        "SELECT * FROM challenges WHERE id = ?", (session["challenge_id"],)
    ).fetchone()

    user_audio_name = f"{uuid.uuid4()}.webm"
    user_audio_path = AUDIO_DIR / user_audio_name
    contents = await audio.read()
    user_audio_path.write_bytes(contents)

    transcript = ai.transcribe_audio(str(user_audio_path))
    if not transcript:
        conn.execute(
            "UPDATE sessions SET status = 'abandoned', ended_at = datetime('now') WHERE id = ?",
            (session_id,),
        )
        conn.commit()
        conn.close()
        raise HTTPException(400, "No speech detected in the recording")

    conn.execute(
        "INSERT INTO messages (session_id, role, content, audio_path) VALUES (?, 'user', ?, ?)",
        (session_id, transcript, user_audio_name),
    )

    challenge_dict = row_to_dict(challenge)
    if challenge_dict["mode"] == "image_description":
        evaluation = ai.evaluate_image_description(
            challenge_dict, [{"role": "user", "content": transcript}]
        )
    else:
        evaluation = ai.evaluate_session(
            challenge_dict, [{"role": "user", "content": transcript}]
        )
    save_evaluation(conn, session_id, evaluation)

    conn.execute(
        "UPDATE sessions SET status = 'completed', ended_at = datetime('now') WHERE id = ?",
        (session_id,),
    )
    conn.commit()
    conn.close()

    return {"transcript": transcript, "evaluation": evaluation}


@app.post("/api/monologue/{session_id}/posture")
def submit_posture(session_id: int, submission: MonologueSubmission):
    conn = get_conn()
    session = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if not session:
        conn.close()
        raise HTTPException(404, "Session not found")

    analyzer = posture.get_analyzer()
    result = analyzer.analyze(submission.landmark_summary, submission.keyframes)
    save_posture(conn, session_id, result, analyzer.name)
    conn.commit()
    conn.close()
    return result


# ---------------------------------------------------------------------------
# Progress + history
# ---------------------------------------------------------------------------


@app.get("/api/sessions")
def list_sessions():
    conn = get_conn()
    rows = conn.execute(
        """SELECT s.id, s.started_at, s.status, s.mode, c.title,
                  e.primary_weakness, e.overall_score,
                  p.posture_score
           FROM sessions s
           JOIN challenges c ON c.id = s.challenge_id
           LEFT JOIN evaluations e ON e.session_id = s.id
           LEFT JOIN posture_evaluations p ON p.session_id = s.id
           ORDER BY s.started_at DESC"""
    ).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


@app.get("/api/progress")
def get_progress(group: str = "day"):
    bucket_expr = {
        "day": "date(s.started_at)",
        "week": "date(s.started_at, 'weekday 0', '-6 days')",
        "month": "strftime('%Y-%m-01', s.started_at)",
    }.get(group, "date(s.started_at)")

    conn = get_conn()
    rows = conn.execute(
        f"""SELECT {bucket_expr} AS bucket,
                   AVG(e.overall_score) AS avg_overall,
                   AVG(e.fluency_score) AS avg_fluency,
                   AVG(e.clarity_score) AS avg_clarity,
                   AVG(e.structure_score) AS avg_structure,
                   AVG(e.vocabulary_score) AS avg_vocabulary,
                   AVG(e.confidence_score) AS avg_confidence,
                   AVG(p.posture_score) AS avg_posture,
                   COUNT(*) AS session_count
            FROM sessions s
            JOIN evaluations e ON e.session_id = s.id
            LEFT JOIN posture_evaluations p ON p.session_id = s.id
            GROUP BY bucket
            ORDER BY bucket ASC"""
    ).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


@app.get("/api/daily-tip")
def daily_tip():
    conn = get_conn()
    rows = conn.execute(
        """SELECT primary_weakness, weakness_explanation, created_at
           FROM evaluations
           ORDER BY created_at DESC
           LIMIT 8"""
    ).fetchall()
    conn.close()
    tip = ai.generate_daily_tip([row_to_dict(r) for r in rows])
    return {"tip": tip}


# ---------------------------------------------------------------------------
# English section: daily words + spaced repetition
# ---------------------------------------------------------------------------


def _today_str() -> str:
    return date.today().isoformat()


@app.get("/api/english/daily-words")
def get_daily_words():
    conn = get_conn()
    today = _today_str()
    count = conn.execute(
        "SELECT COUNT(*) AS c FROM daily_words WHERE word_date = ?", (today,)
    ).fetchone()["c"]

    if count == 0:
        words = ai.generate_daily_words()
        for w in words:
            cur = conn.execute(
                "INSERT INTO daily_words (word, meaning, example_sentence, word_date) VALUES (?, ?, ?, ?)",
                (w["word"], w["meaning"], w["example_sentence"], today),
            )
            word_id = cur.lastrowid
            next_review = (date.today() + timedelta(days=REVIEW_SCHEDULE_DAYS[0])).isoformat()
            conn.execute(
                "INSERT INTO word_progress (word_id, review_stage, next_review_date) VALUES (?, 0, ?)",
                (word_id, next_review),
            )
        conn.commit()

    rows = conn.execute(
        """SELECT d.*, p.review_stage, p.next_review_date, p.mastered
           FROM daily_words d
           LEFT JOIN word_progress p ON p.word_id = d.id
           WHERE d.word_date = ?
           ORDER BY d.id""",
        (today,),
    ).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


@app.get("/api/english/reviews/due")
def reviews_due():
    conn = get_conn()
    today = _today_str()
    rows = conn.execute(
        """SELECT d.id AS word_id, d.word, d.meaning, d.example_sentence, p.review_stage
           FROM word_progress p
           JOIN daily_words d ON d.id = p.word_id
           WHERE p.mastered = 0 AND p.next_review_date <= ?
           ORDER BY p.next_review_date ASC""",
        (today,),
    ).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


@app.post("/api/english/reviews/{word_id}/answer")
def answer_review(word_id: int, is_correct: bool):
    conn = get_conn()
    progress = conn.execute(
        "SELECT * FROM word_progress WHERE word_id = ?", (word_id,)
    ).fetchone()
    if not progress:
        conn.close()
        raise HTTPException(404, "Word progress not found")

    stage = progress["review_stage"]
    if is_correct:
        stage += 1
        if stage >= len(REVIEW_SCHEDULE_DAYS):
            conn.execute(
                """UPDATE word_progress SET review_stage = ?, mastered = 1,
                   next_review_date = NULL, updated_at = datetime('now') WHERE word_id = ?""",
                (stage, word_id),
            )
        else:
            next_review = (date.today() + timedelta(days=REVIEW_SCHEDULE_DAYS[stage])).isoformat()
            conn.execute(
                """UPDATE word_progress SET review_stage = ?, next_review_date = ?,
                   updated_at = datetime('now') WHERE word_id = ?""",
                (stage, next_review, word_id),
            )
    else:
        next_review = (date.today() + timedelta(days=1)).isoformat()
        conn.execute(
            "UPDATE word_progress SET next_review_date = ?, updated_at = datetime('now') WHERE word_id = ?",
            (next_review, word_id),
        )
    conn.commit()
    conn.close()
    return {"ok": True}


# ---------------------------------------------------------------------------
# English section: pronunciation practice
# ---------------------------------------------------------------------------


@app.get("/api/english/pronunciation/word")
def pronunciation_word():
    conn = get_conn()
    row = conn.execute(
        "SELECT id, word, meaning FROM daily_words ORDER BY RANDOM() LIMIT 1"
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "No words available yet - visit Daily Words first")
    return row_to_dict(row)


@app.post("/api/english/pronunciation/attempt")
async def pronunciation_attempt(word: str, audio: UploadFile):
    wav_bytes = await audio.read()
    analyzer = pronunciation.get_analyzer()
    try:
        result = analyzer.analyze(word, wav_bytes)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        raise HTTPException(500, f"Pronunciation engine '{analyzer.name}' failed: {e}")

    conn = get_conn()
    conn.execute(
        """INSERT INTO pronunciation_attempts
           (word, engine, accuracy_score, fluency_score, completeness_score, overall_score, correct, feedback)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            word,
            analyzer.name,
            result.get("accuracy_score"),
            result.get("fluency_score"),
            result.get("completeness_score"),
            result.get("overall_score"),
            int(bool(result.get("correct"))),
            result.get("feedback"),
        ),
    )
    conn.commit()
    conn.close()
    return result


# ---------------------------------------------------------------------------
# English section: "use it in a sentence" spoken drill
# ---------------------------------------------------------------------------


@app.get("/api/english/sentence-drill/word")
def sentence_drill_word():
    conn = get_conn()
    row = conn.execute(
        "SELECT id, word, meaning, example_sentence FROM daily_words ORDER BY RANDOM() LIMIT 1"
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "No words available yet - visit Daily Words first")
    return row_to_dict(row)


@app.post("/api/english/sentence-drill/attempt")
async def sentence_drill_attempt(word_id: int, audio: UploadFile):
    conn = get_conn()
    word_row = conn.execute("SELECT * FROM daily_words WHERE id = ?", (word_id,)).fetchone()
    if not word_row:
        conn.close()
        raise HTTPException(404, "Word not found")

    user_audio_name = f"{uuid.uuid4()}.webm"
    user_audio_path = AUDIO_DIR / user_audio_name
    contents = await audio.read()
    user_audio_path.write_bytes(contents)

    transcript = ai.transcribe_audio(str(user_audio_path))
    if not transcript:
        conn.close()
        raise HTTPException(400, "No speech detected")

    result = ai.evaluate_sentence_drill(word_row["word"], word_row["meaning"], transcript)

    conn.execute(
        "INSERT INTO sentence_drills (word_id, transcript, is_correct, feedback) VALUES (?, ?, ?, ?)",
        (word_id, transcript, int(bool(result.get("correct"))), result.get("feedback")),
    )
    conn.commit()
    conn.close()
    return {"transcript": transcript, **result}


# ---------------------------------------------------------------------------
# English section: grammar drills mined from the user's own transcripts
# ---------------------------------------------------------------------------


@app.get("/api/english/grammar-drills")
def grammar_drills():
    conn = get_conn()
    today = _today_str()
    latest = conn.execute(
        "SELECT date(created_at) AS d FROM grammar_drills ORDER BY created_at DESC LIMIT 1"
    ).fetchone()

    if not latest or latest["d"] != today:
        rows = conn.execute(
            "SELECT content FROM messages WHERE role = 'user' ORDER BY created_at DESC LIMIT 40"
        ).fetchall()
        transcripts = [r["content"] for r in rows]
        drills = ai.mine_grammar_drills(transcripts)
        if drills:
            conn.execute("DELETE FROM grammar_drills")
            for d in drills:
                conn.execute(
                    """INSERT INTO grammar_drills
                       (mistake_pattern, original_example, corrected_example, exercise_prompt)
                       VALUES (?, ?, ?, ?)""",
                    (
                        d["mistake_pattern"],
                        d["original_example"],
                        d["corrected_example"],
                        d["exercise_prompt"],
                    ),
                )
            conn.commit()

    result = conn.execute("SELECT * FROM grammar_drills ORDER BY id").fetchall()
    conn.close()
    return [row_to_dict(r) for r in result]


# ---------------------------------------------------------------------------
# English section: filler-word trend
# ---------------------------------------------------------------------------


@app.get("/api/english/filler-words/trend")
def filler_word_trend(group: str = "day"):
    bucket_expr = {
        "day": "date(s.started_at)",
        "week": "date(s.started_at, 'weekday 0', '-6 days')",
        "month": "strftime('%Y-%m-01', s.started_at)",
    }.get(group, "date(s.started_at)")

    conn = get_conn()
    rows = conn.execute(
        f"""SELECT {bucket_expr} AS bucket, m.content
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE m.role = 'user'
            ORDER BY bucket ASC"""
    ).fetchall()
    conn.close()

    buckets: dict[str, dict[str, int]] = {}
    for row in rows:
        b = row["bucket"]
        bucket_data = buckets.setdefault(b, {"fillers": 0, "words": 0})
        bucket_data["fillers"] += filler_words.count_fillers(row["content"])
        bucket_data["words"] += filler_words.count_words(row["content"])

    result = []
    for bucket in sorted(buckets):
        data = buckets[bucket]
        rate = round((data["fillers"] / data["words"]) * 100, 1) if data["words"] else 0
        result.append(
            {
                "bucket": bucket,
                "filler_count": data["fillers"],
                "word_count": data["words"],
                "filler_rate_per_100_words": rate,
            }
        )
    return result


# ---------------------------------------------------------------------------
# Interview mode: resume upload, domain-aware questions, continuous voice loop
# ---------------------------------------------------------------------------


@app.post("/api/interview/resume")
async def upload_resume(file: UploadFile):
    contents = await file.read()
    suffix = "." + file.filename.rsplit(".", 1)[-1] if "." in file.filename else ""

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(contents)
        tmp_path = f.name

    try:
        text = resume_parser.extract_text(tmp_path, file.filename)
    except ValueError as e:
        os.unlink(tmp_path)
        raise HTTPException(400, str(e))
    os.unlink(tmp_path)

    if not text:
        raise HTTPException(400, "Could not extract any text from that file.")

    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO resumes (filename, extracted_text) VALUES (?, ?)",
        (file.filename, text),
    )
    resume_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {"resume_id": resume_id, "filename": file.filename, "preview": text[:400]}


class InterviewStart(BaseModel):
    resume_id: int
    domain: str
    target_duration_seconds: int
    camera_enabled: bool = False


@app.post("/api/interview/start")
def start_interview(payload: InterviewStart):
    conn = get_conn()
    resume_row = conn.execute(
        "SELECT * FROM resumes WHERE id = ?", (payload.resume_id,)
    ).fetchone()
    if not resume_row:
        conn.close()
        raise HTTPException(404, "Resume not found")

    cur = conn.execute(
        """INSERT INTO challenges (title, scenario, category, mode, ai_persona, resume_id, domain)
           VALUES (?, ?, 'interview', 'interview', ?, ?, ?)""",
        (
            f"Mock interview - {payload.domain}",
            f"A mock job interview for a {payload.domain} role, based on the candidate's resume.",
            f"an experienced, professional interviewer for {payload.domain} roles",
            payload.resume_id,
            payload.domain,
        ),
    )
    challenge_id = cur.lastrowid

    cur = conn.execute(
        """INSERT INTO sessions (challenge_id, mode, target_duration_seconds, camera_enabled)
           VALUES (?, 'interview', ?, ?)""",
        (challenge_id, payload.target_duration_seconds, int(payload.camera_enabled)),
    )
    session_id = cur.lastrowid
    conn.commit()

    challenge = conn.execute("SELECT * FROM challenges WHERE id = ?", (challenge_id,)).fetchone()
    question_text = ai.get_interview_reply(row_to_dict(challenge), resume_row["extracted_text"], [])

    ai_audio_name = f"{uuid.uuid4()}.mp3"
    ai.synthesize_speech(question_text, str(AUDIO_DIR / ai_audio_name))

    conn.execute(
        "INSERT INTO messages (session_id, role, content, audio_path) VALUES (?, 'assistant', ?, ?)",
        (session_id, question_text, ai_audio_name),
    )
    conn.commit()
    conn.close()

    return {
        "session_id": session_id,
        "question_text": question_text,
        "question_audio_url": f"/audio/{ai_audio_name}",
    }


@app.post("/api/interview/{session_id}/message")
async def interview_message(session_id: int, audio: UploadFile):
    conn = get_conn()
    session = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not session:
        conn.close()
        raise HTTPException(404, "Session not found")
    challenge = conn.execute(
        "SELECT * FROM challenges WHERE id = ?", (session["challenge_id"],)
    ).fetchone()
    resume_row = conn.execute(
        "SELECT * FROM resumes WHERE id = ?", (challenge["resume_id"],)
    ).fetchone()

    user_audio_name = f"{uuid.uuid4()}.webm"
    user_audio_path = AUDIO_DIR / user_audio_name
    contents = await audio.read()
    user_audio_path.write_bytes(contents)

    transcript = ai.transcribe_audio(str(user_audio_path))
    if not transcript:
        conn.close()
        raise HTTPException(400, "No speech detected")

    conn.execute(
        "INSERT INTO messages (session_id, role, content, audio_path) VALUES (?, 'user', ?, ?)",
        (session_id, transcript, user_audio_name),
    )
    conn.commit()

    history = get_session_history(conn, session_id)
    question_text = ai.get_interview_reply(row_to_dict(challenge), resume_row["extracted_text"], history)

    ai_audio_name = f"{uuid.uuid4()}.mp3"
    ai.synthesize_speech(question_text, str(AUDIO_DIR / ai_audio_name))

    conn.execute(
        "INSERT INTO messages (session_id, role, content, audio_path) VALUES (?, 'assistant', ?, ?)",
        (session_id, question_text, ai_audio_name),
    )
    conn.commit()
    conn.close()

    return {
        "transcript": transcript,
        "ai_text": question_text,
        "ai_audio_url": f"/audio/{ai_audio_name}",
    }


@app.post("/api/interview/{session_id}/end")
def end_interview(session_id: int):
    conn = get_conn()
    session = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not session:
        conn.close()
        raise HTTPException(404, "Session not found")
    challenge = conn.execute(
        "SELECT * FROM challenges WHERE id = ?", (session["challenge_id"],)
    ).fetchone()
    resume_row = conn.execute(
        "SELECT * FROM resumes WHERE id = ?", (challenge["resume_id"],)
    ).fetchone()

    history = get_session_history(conn, session_id)
    if not any(m["role"] == "user" for m in history):
        conn.execute(
            "UPDATE sessions SET status = 'abandoned', ended_at = datetime('now') WHERE id = ?",
            (session_id,),
        )
        conn.commit()
        conn.close()
        raise HTTPException(400, "No answers to evaluate")

    report = ai.generate_interview_report(row_to_dict(challenge), resume_row["extracted_text"], history)

    conn.execute(
        """INSERT INTO interview_reports
           (session_id, overall_score, strengths, weaknesses, improved_answers, summary, raw_report)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            report.get("overall_score"),
            json.dumps(report.get("strengths", [])),
            json.dumps(report.get("weaknesses", [])),
            json.dumps(report.get("improved_answers", [])),
            report.get("summary", ""),
            json.dumps(report),
        ),
    )
    conn.execute(
        "UPDATE sessions SET status = 'completed', ended_at = datetime('now') WHERE id = ?",
        (session_id,),
    )
    conn.commit()
    conn.close()

    return report
