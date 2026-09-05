# Communicato

Communicato is an AI-driven speech coaching app for learners, students, and job seekers who want to practice spoken communication with real-time, voice-first workflows.

It combines conversational practice, transcription, speech scoring, pronunciation feedback, posture/gesture analysis, vocabulary retention drills, and mock interview practice in one FastAPI-based project.

## Who it is for

- Students improving fluency, confidence, and speaking structure
- English learners building vocabulary and pronunciation
- Job seekers preparing for interviews with resume-aware mock sessions
- Anyone who wants guided communication practice with measurable feedback

## Key benefits

- **Practice by speaking, not typing** with audio-in/audio-out flows
- **Get structured coaching feedback** (fluency, clarity, structure, vocabulary, confidence)
- **Improve delivery** with pronunciation and posture insights
- **Retain vocabulary better** with spaced-repetition review scheduling
- **Prepare for interviews** with domain-aware, resume-based mock interview loops

## Features

- **TTS prompts and replies** using generated audio responses
- **Transcription** of recorded user speech
- **Speech evaluation** for conversations, monologues, and image-description tasks
- **Pronunciation analysis** with pluggable engines (`azure`, `gpt4o_audio`, `whisper_fuzzy`)
- **Posture and gesture analysis** with pluggable analyzers (`mediapipe`, `openai_vision`, `hybrid`)
- **Spaced-repetition vocabulary practice** via daily words and review stages
- **Interview practice** with resume upload, interviewer turns, and end-of-session reports
- **Additional language drills** including sentence drills, grammar drills, and filler-word trends

## Tech stack

- **Backend:** Python, FastAPI, Uvicorn
- **Frontend:** HTML/CSS/JavaScript (served from `static/index.html`)
- **Database:** SQLite (`speakup.db`)
- **AI/Voice integrations:** OpenAI APIs, Azure Cognitive Services Speech, Unsplash API (image mode)

## Getting started

### 1) Clone and enter the repository

```bash
git clone https://github.com/Adityaydv1234/Communicato.git
cd Communicato
```

### 2) Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 3) Configure environment variables

Copy `.env.example` to `.env` and set required keys:

```bash
cp .env.example .env
```

At minimum, configure:

- `OPENAI_API_KEY`

Depending on enabled features, also configure:

- `UNSPLASH_ACCESS_KEY` (image-description practice)
- `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` (if `PRONUNCIATION_ENGINE=azure`)
- `POSTURE_ANALYZER` and `PRONUNCIATION_ENGINE` selection values

### 4) Run the app

Either of the following works:

```bash
python run.py
```

or

```bash
uvicorn main:app --reload
```

Open: `http://127.0.0.1:8000`

## Usage

1. Open the web UI in your browser.
2. Pick a mode: **Conversation**, **Monologue**, **Describe Image**, **English**, or **Interview**.
3. Record responses with microphone input.
4. Review transcription, coaching scores, and generated feedback.
5. Use English drills (daily words, pronunciation, grammar, filler trends) for continued practice.

> Note: Camera-based posture features require browser camera permission.

## Project structure

```text
Communicato/
├── main.py                 # FastAPI app + API routes
├── run.py                  # Local dev entry point
├── ai.py                   # LLM/TTS/transcription/evaluation logic
├── db.py                   # SQLite schema/init/seed data
├── static/
│   └── index.html          # Frontend UI
├── posture/                # Posture analyzer implementations
├── pronunciation/          # Pronunciation analyzer implementations
├── resume.py               # Resume text extraction helpers
├── images.py               # Unsplash image challenge integration
├── filler_words.py         # Filler-word analytics helpers
└── requirements.txt
```

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Make focused changes
4. Open a pull request with a clear description

If you contribute new setup steps, please update this README so others can reproduce them.

## License

No license file is currently present in this repository.
