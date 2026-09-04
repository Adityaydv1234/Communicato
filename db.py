import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "speakup.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _migrate_existing_columns(conn):
    """Add columns introduced after the initial schema, for DBs created before this change."""
    challenge_cols = {row["name"] for row in conn.execute("PRAGMA table_info(challenges)")}
    if "mode" not in challenge_cols:
        conn.execute("ALTER TABLE challenges ADD COLUMN mode TEXT NOT NULL DEFAULT 'conversation'")

    if "image_url" not in challenge_cols:
        conn.execute("ALTER TABLE challenges ADD COLUMN image_url TEXT")
    if "image_description" not in challenge_cols:
        conn.execute("ALTER TABLE challenges ADD COLUMN image_description TEXT")
    if "image_unsplash_id" not in challenge_cols:
        conn.execute("ALTER TABLE challenges ADD COLUMN image_unsplash_id TEXT")
    if "resume_id" not in challenge_cols:
        conn.execute("ALTER TABLE challenges ADD COLUMN resume_id INTEGER REFERENCES resumes(id)")
    if "domain" not in challenge_cols:
        conn.execute("ALTER TABLE challenges ADD COLUMN domain TEXT")

    session_cols = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
    if "mode" not in session_cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN mode TEXT NOT NULL DEFAULT 'conversation'")
    if "target_duration_seconds" not in session_cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN target_duration_seconds INTEGER")
    if "camera_enabled" not in session_cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN camera_enabled INTEGER DEFAULT 0")

    eval_cols = {row["name"] for row in conn.execute("PRAGMA table_info(evaluations)")}
    if "relevance_score" not in eval_cols:
        conn.execute("ALTER TABLE evaluations ADD COLUMN relevance_score INTEGER")

    conn.commit()


def init_db():
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS challenges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            scenario TEXT NOT NULL,
            category TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'conversation',
            ai_persona TEXT,
            target_duration_seconds INTEGER DEFAULT 180
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            challenge_id INTEGER NOT NULL REFERENCES challenges(id),
            mode TEXT NOT NULL DEFAULT 'conversation',
            target_duration_seconds INTEGER,
            camera_enabled INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active',
            started_at TEXT DEFAULT (datetime('now')),
            ended_at TEXT,
            duration_seconds INTEGER
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            audio_path TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER UNIQUE NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            fluency_score INTEGER,
            clarity_score INTEGER,
            structure_score INTEGER,
            vocabulary_score INTEGER,
            confidence_score INTEGER,
            overall_score INTEGER,
            primary_weakness TEXT NOT NULL,
            weakness_explanation TEXT NOT NULL,
            exercise TEXT NOT NULL,
            raw_evaluation TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS posture_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER UNIQUE NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            analyzer TEXT NOT NULL,
            posture_score INTEGER,
            eye_contact_score INTEGER,
            gesture_score INTEGER,
            feedback TEXT NOT NULL,
            raw_data TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            extracted_text TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS interview_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER UNIQUE NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            overall_score INTEGER,
            strengths TEXT NOT NULL,
            weaknesses TEXT NOT NULL,
            improved_answers TEXT NOT NULL,
            summary TEXT NOT NULL,
            raw_report TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS daily_words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            meaning TEXT NOT NULL,
            example_sentence TEXT NOT NULL,
            word_date TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS word_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word_id INTEGER UNIQUE NOT NULL REFERENCES daily_words(id) ON DELETE CASCADE,
            review_stage INTEGER DEFAULT 0,
            next_review_date TEXT,
            mastered INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS pronunciation_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            engine TEXT NOT NULL,
            accuracy_score INTEGER,
            fluency_score INTEGER,
            completeness_score INTEGER,
            overall_score INTEGER,
            correct INTEGER,
            feedback TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sentence_drills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word_id INTEGER NOT NULL REFERENCES daily_words(id) ON DELETE CASCADE,
            transcript TEXT NOT NULL,
            is_correct INTEGER,
            feedback TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS grammar_drills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mistake_pattern TEXT NOT NULL,
            original_example TEXT NOT NULL,
            corrected_example TEXT NOT NULL,
            exercise_prompt TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
        CREATE INDEX IF NOT EXISTS idx_daily_words_date ON daily_words(word_date);
        CREATE INDEX IF NOT EXISTS idx_word_progress_next_review ON word_progress(next_review_date);
        """
    )

    _migrate_existing_columns(conn)

    conversation_count = conn.execute(
        "SELECT COUNT(*) AS c FROM challenges WHERE mode = 'conversation'"
    ).fetchone()["c"]
    if conversation_count == 0:
        conn.executemany(
            "INSERT INTO challenges (title, scenario, category, mode, ai_persona) VALUES (?, ?, ?, 'conversation', ?)",
            [
                (
                    "Networking at a college event",
                    "You meet a senior at a college event whom you've never spoken to. "
                    "Start a conversation and keep it going for a few minutes. Try to learn "
                    "something interesting about them and share something about yourself.",
                    "small-talk",
                    "a friendly senior student at a college networking event, casual and a bit busy",
                ),
                (
                    "Explaining your major to a stranger",
                    "You're on a train and the person next to you asks what you study. "
                    "Explain your major and why you chose it in a way that's clear and interesting, "
                    "not just a list of facts.",
                    "explanation",
                    "a curious stranger sitting next to you on a train, genuinely interested",
                ),
                (
                    "Disagreeing with a group project teammate",
                    "Your teammate wants to take the group project in a direction you think is wrong. "
                    "Explain your concerns and try to find a way forward together.",
                    "conflict",
                    "a group project teammate who is confident in their own idea but willing to listen",
                ),
                (
                    "Telling a story about a challenge you overcame",
                    "A friend asks you to tell them about a time you overcame a difficult challenge. "
                    "Tell the story with a clear beginning, middle, and end.",
                    "storytelling",
                    "a close friend who is genuinely curious and asks good follow-up questions",
                ),
            ],
        )
        conn.commit()

    monologue_count = conn.execute(
        "SELECT COUNT(*) AS c FROM challenges WHERE mode = 'monologue'"
    ).fetchone()["c"]
    if monologue_count == 0:
        conn.executemany(
            "INSERT INTO challenges (title, scenario, category, mode, ai_persona) VALUES (?, ?, ?, 'monologue', '')",
            [
                (
                    "Why I chose my field of study",
                    "Speak continuously about why you chose your current field of study or career "
                    "path, and what excites you about it.",
                    "self-introduction",
                ),
                (
                    "A skill everyone should learn",
                    "Speak continuously about one skill you think everyone should learn, and why "
                    "it matters.",
                    "persuasion",
                ),
                (
                    "Describe your ideal day",
                    "Speak continuously describing what your ideal day would look like, from "
                    "morning to night.",
                    "storytelling",
                ),
                (
                    "A technology that will change the next decade",
                    "Speak continuously about a technology you believe will meaningfully change "
                    "the next ten years, and why.",
                    "explanation",
                ),
                (
                    "Convince someone to visit your hometown",
                    "Speak continuously trying to convince someone to visit your hometown or a "
                    "place you love.",
                    "persuasion",
                ),
            ],
        )
        conn.commit()
    conn.close()
