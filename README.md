# VidTutor 🎓
> A voice-first conversational tutor for YouTube videos — in any Indian language.

VidTutor lets you ask questions about any YouTube video by text or voice, and get answers spoken back in your language. Every answer is grounded strictly in the video transcript — no hallucination, no outside knowledge.

---

## What It Does

- Ask questions about any YouTube video — by typing or speaking
- Answers only from the video transcript, never from outside knowledge
- Speaks back in the same language you speak in
- Auto-detects your language (Hindi, Marathi, Kannada, Tamil, and more)
- Full voice mode — hands-free, auto-loops, no typing needed
- Switch between preloaded videos or paste any YouTube URL
- Conversation history persists across voice and text sessions

---

## Why It Matters

India has ~700M smartphone users but only ~10% are comfortable in English. 
Most high-quality educational content like lectures, explainers, tutorials exists only in English. 
This creates a knowledge gap that affects students, farmers, healthcare workers, and small business owners alike.

VidTutor bridges that gap. Load any educational YouTube video, ask questions in your language, 
and get answers spoken back to you, all grounded strictly in the video content.

A Class 10 student in Bhopal can understand 3Blue1Brown's neural network explanation in Hindi.
A farmer in Gujarat can ask about crop disease from an agri-tutorial in Gujarati.
A first-generation college student in Chennai can study from MIT OpenCourseWare in Tamil.

No language barrier. No hallucination. Just learning.

---

## Stack

| Layer | Tool |
|---|---|
| Backend | FastAPI + Uvicorn |
| Transcript | `youtube-transcript-api` |
| RAG | `sentence-transformers` + FAISS |
| LLM | Sarvam-M |
| STT | Sarvam Saaras v3 |
| TTS | Sarvam Bulbul v3 |
| Frontend | Vanilla HTML / CSS / JS |

100% Sarvam stack for all language and voice layers.

---

## Setup

### 1. Requirements
- Python 3.10, 3.11, or 3.12 (recommended: 3.11)
- Sarvam API key — get free at [dashboard.sarvam.ai](https://dashboard.sarvam.ai)

### 2. Install
```bash
pip install -r requirements.txt
```
> ⚠️ This may take 2-3 minutes on first install — `sentence-transformers` and `faiss-cpu` are large packages.
> 
> **Python version matters:** Use Python 3.10, 3.11, or 3.12 only. Python 3.13+ will fail due to missing `faiss-cpu` wheels. If you hit numpy/faiss conflicts, try:
> ```bash
> pip install "numpy<2" --upgrade
> pip install -r requirements.txt --force-reinstall
> ```

### 3. Configure
```bash
# Add your Sarvam API key to .env
```

### 4. Run
```bash
uvicorn main:app --reload
```
> ⏳ First run takes 30-60 seconds — downloads the embedding model (~90MB) and fetches YouTube transcripts.
> Subsequent runs are instant as everything is cached to disk.

Open http://localhost:8000

---

## How It Works

1. On startup, fetches the YouTube transcript and builds a FAISS vector index
2. User asks a question by text or voice
3. Voice input → Sarvam Saaras v3 transcribes + detects language automatically
4. Top-4 relevant transcript chunks retrieved via cosine similarity
5. Sarvam-M answers using only those chunks
6. Answer displayed in chat + spoken via Sarvam Bulbul v3 in detected language
7. Mic auto-activates for next turn in voice mode — fully hands-free

---

## Features

### Text Mode
- Type questions, get grounded answers
- Voice input button with auto VAD (stops on silence)
- Toggle TTS to hear responses spoken aloud
- Select response language from dropdown

### Voice Mode
- Full screen immersive interface
- Auto-detects your language from speech
- Responds and speaks back in same language
- Interrupt button to stop speaking and ask again
- **All voice conversations saved to text chat** — full transcript visible after session ends

### Video Selection
- 3 preloaded videos (Neural Networks, Transformers, What is Sarvam AI?)
- Paste any YouTube URL to load any video instantly
- FAISS index cached to disk — instant on second load

---

## Preloaded Videos

- [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown
- [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown
- [What is Sarvam AI?](https://www.youtube.com/watch?v=qswEBHoWZMM) — Sarvam