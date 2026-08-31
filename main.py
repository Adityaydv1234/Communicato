import json
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import ai
from db import get_conn, init_db

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


@app.get("/api/challenges/random")
def random_challenge():
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM challenges ORDER BY RANDOM() LIMIT 1"
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "No challenges available")
    return row_to_dict(row)


@app.post("/api/sessions")
def create_session(challenge_id: int):
    conn = get_conn()
    challenge = conn.execute(
        "SELECT * FROM challenges WHERE id = ?", (challenge_id,)
    ).fetchone()
    if not challenge:
        conn.close()
        raise HTTPException(404, "Challenge not found")

    cur = conn.execute(
        "INSERT INTO sessions (challenge_id) VALUES (?)", (challenge_id,)
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

    conn.execute(
        """INSERT INTO evaluations
           (session_id, fluency_score, clarity_score, structure_score, vocabulary_score,
            confidence_score, overall_score, primary_weakness, weakness_explanation, exercise, raw_evaluation)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            evaluation.get("fluency_score"),
            evaluation.get("clarity_score"),
            evaluation.get("structure_score"),
            evaluation.get("vocabulary_score"),
            evaluation.get("confidence_score"),
            evaluation.get("overall_score"),
            evaluation.get("primary_weakness"),
            evaluation.get("weakness_explanation"),
            evaluation.get("exercise"),
            json.dumps(evaluation),
        ),
    )
    conn.execute(
        "UPDATE sessions SET status = 'completed', ended_at = datetime('now') WHERE id = ?",
        (session_id,),
    )
    conn.commit()
    conn.close()

    return evaluation


@app.get("/api/sessions")
def list_sessions():
    conn = get_conn()
    rows = conn.execute(
        """SELECT s.id, s.started_at, s.status, c.title,
                  e.primary_weakness, e.overall_score
           FROM sessions s
           JOIN challenges c ON c.id = s.challenge_id
           LEFT JOIN evaluations e ON e.session_id = s.id
           ORDER BY s.started_at DESC"""
    ).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]
