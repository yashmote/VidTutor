# VidTutor 🎓

A voice-first conversational tutor that lets anyone ask questions about a YouTube video — in any Indian language. Answers are grounded strictly in the video transcript; the bot will never hallucinate beyond it.

## What It Does

- Ask questions about any YouTube video by text or voice
- Answers only from the video transcript — no hallucination
- Speaks back in the same language you speak in
- Auto-detects your language (Hindi, Marathi, Kannada, Tamil, and more)
- Full voice mode — hands-free, auto-loops, no typing needed
- Switch between videos or paste any YouTube URL on the fly
- Conversation history persists across voice and text modes

## Why It Matters

Millions of high-quality English educational videos exist on YouTube. Language is the wall. A student in rural Maharashtra can ask a question in Marathi and hear the answer spoken back — without needing to understand English. VidTutor tears that wall down.

## Stack

| Layer | Tool |
|---|---|
| Backend | FastAPI + Uvicorn |
| Transcript | `youtube-transcript-api` |
| RAG | `sentence-transformers` + FAISS |
| LLM | Sarvam-M (chat completion) |
| Voice Input (STT) | Sarvam Saaras v3 |
| Voice Output (TTS) | Sarvam Bulbul v3 |
| Frontend | Vanilla HTML/CSS/JS |

100% Sarvam stack for all language and voice layers.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt --upgrade
```

### 2. Set API keys
```bash
cp .env.example .env
# Edit .env and add your Sarvam API key
```
Get your Sarvam API key at: https://dashboard.sarvam.ai

### 3. Run
```bash
uvicorn main:app --reload
```
Open http://localhost:8000

## How It Works

1. On startup, fetches the default YouTube transcript and builds a FAISS vector index (cached to disk after first run)
2. User asks a question via text or voice
3. Voice input → Sarvam Saaras v3 transcribes + detects language automatically
4. Top-4 relevant transcript chunks retrieved via cosine similarity (FAISS)
5. Sarvam-M answers using only those chunks — system prompt enforces transcript-only answers
6. Answer displayed in chat + optionally spoken via Sarvam Bulbul v3 in detected language
7. Mic auto-activates for next turn in voice mode — fully hands-free

## Features

### Text Mode
- Type questions, get grounded answers
- Toggle "Speak responses" for TTS output
- Select response language from dropdown
- Voice input button with auto VAD (stops on silence)

### Voice Mode
- Full screen immersive interface
- Auto-detects language from your speech
- Responds in the same language you spoke
- Loops automatically — completely hands-free
- All voice conversations appear in text chat after session

### Video Selection
- 3 preloaded videos (Neural Networks, Transformers, What is Sarvam AI?)
- Paste any YouTube URL to load any video on the fly
- Index cached to disk — instant on second load

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the frontend |
| POST | `/chat` | RAG + LLM endpoint |
| POST | `/stt` | Sarvam speech-to-text |
| POST | `/tts` | Sarvam text-to-speech |
| POST | `/load` | Load a new video by ID |
| GET | `/videos` | List available videos |
| GET | `/status` | Index health check |

## Preloaded Videos

- [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown
- [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown  
- [What is Sarvam AI?](https://www.youtube.com/watch?v=qswEBHoWZMM) — Sarvam

## Requirements

- Python 3.10, 3.11, or 3.12 (recommended: 3.11)
- Sarvam API key