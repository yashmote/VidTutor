import os
import re
import base64
import tempfile
import numpy as np
import faiss
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
from sarvamai import SarvamAI

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80
TOP_K         = 4

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

VIDEOS = {
    "aircAruvnKk": "But what is a neural network? — 3Blue1Brown",
    "wjZofJX0v4M": "Transformers, the tech behind LLMs — 3Blue1Brown",
    "qswEBHoWZMM": "What is Sarvam AI? — Sarvam",
}

DEFAULT_VIDEO_ID = "aircAruvnKk"

TTS_LANGUAGE_MAP = {
    "en-IN": "en-IN", "hi-IN": "hi-IN", "bn-IN": "bn-IN",
    "kn-IN": "kn-IN", "ml-IN": "ml-IN", "mr-IN": "mr-IN",
    "ta-IN": "ta-IN", "te-IN": "te-IN", "gu-IN": "gu-IN",
}

SYSTEM_PROMPT = """You are a focused video tutor. Answer ONLY from the transcript provided below.

Rules:
1. Answer ONLY from the transcript. Never use outside knowledge.
2. For greetings like "hello", "hi", "how are you" — reply in ONE short sentence only (e.g. "Hello! Ask me anything about the video."). Nothing more.
3. If the answer is not in the transcript, say: "This wasn't covered in the video."
4. Give complete, natural answers — explain enough to be genuinely useful, but stay focused.
5. Never cut off mid-thought. Always finish your sentence.
6. ALWAYS respond in {language} only.

Transcript:
{context}"""

# ── STARTUP ───────────────────────────────────────────────────────────────────
print("Initialising Sarvam client...")
sarvam_client = SarvamAI(
    api_subscription_key=SARVAM_API_KEY,
    timeout=60.0,
)

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Global state — swappable on /load
current_video_id = DEFAULT_VIDEO_ID
chunks           = []
index            = None

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    result, start = [], 0
    while start < len(text):
        result.append(text[start:start + size])
        start += size - overlap
    return result

def build_index(video_id: str):
    global chunks, index, current_video_id
    print(f"Fetching transcript for {video_id}...")
    fetcher = YouTubeTranscriptApi()
    raw     = fetcher.fetch(video_id)
    text    = " ".join([t.text for t in raw])
    chunks  = chunk_text(text)
    print(f"Building FAISS index over {len(chunks)} chunks...")
    emb  = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    emb  = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    idx  = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    index            = idx
    current_video_id = video_id
    print(f"Index ready for {video_id}.")

# Build default on startup
build_index(DEFAULT_VIDEO_ID)

# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── SCHEMAS ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    language: str = "English"
    lang_code: str = "en-IN"
    history: list = []
    speak: bool = False

class STTRequest(BaseModel):
    audio_b64: str

class TTSRequest(BaseModel):
    text: str
    lang_code: str = "en-IN"

class LoadRequest(BaseModel):
    video_id: str

# ── HELPERS ───────────────────────────────────────────────────────────────────
def retrieve(query: str) -> str:
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    _, ids = index.search(q_emb, TOP_K)
    return "\n\n".join([chunks[i] for i in ids[0]])

def ask_sarvam(question: str, context: str, language: str, history: list) -> str:
    system = SYSTEM_PROMPT.format(context=context, language=language)
    messages = [{"role": "system", "content": system}]
    for msg in history:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": f"{question}\n\n[Respond in {language} only]"})

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {SARVAM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sarvam-m",
                "messages": messages,
                "max_tokens": 1500,
                "temperature": 0.2,
            }
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        # strip all think blocks including incomplete ones
        clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
        clean = re.sub(r'<think>.*$', '', clean, flags=re.DOTALL)  # catch unclosed think tag
        return clean.strip()

def sarvam_tts(text: str, lang_code: str) -> str:
    response = sarvam_client.text_to_speech.convert(
        text=text[:2500],
        target_language_code=TTS_LANGUAGE_MAP.get(lang_code, "en-IN"),
        speaker="shubh",
        model="bulbul:v3",
    )
    return response.audios[0]


def get_video_title(video_id: str) -> str:
    try:
        res = httpx.get(
            f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json",
            timeout=10.0
        )
        return res.json().get("title", f"YouTube Video ({video_id})")
    except:
        return f"YouTube Video ({video_id})"

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/videos")
def get_videos():
    return {"videos": VIDEOS, "current": current_video_id}

@app.post("/load")
def load_video(req: LoadRequest):
    try:
        build_index(req.video_id)
        # use predefined title or fetch from YouTube
        title = VIDEOS.get(req.video_id) or get_video_title(req.video_id)
        return {"status": "ok", "video_id": req.video_id, "title": title}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/chat")
def chat(req: ChatRequest):
    print(f"\n{'='*50}")
    print(f"[QUESTION]   {req.question}")
    print(f"[LANGUAGE]   {req.language} ({req.lang_code})")
    context = retrieve(req.question)
    print(f"[RAG]        Retrieved {len(context)} chars from index")
    answer = ask_sarvam(req.question, context, req.language, req.history)
    print(f"[SARVAM LLM] {answer[:100]}...")
    audio_b64 = None
    if req.speak and SARVAM_API_KEY:
        try:
            audio_b64 = sarvam_tts(answer, req.lang_code)
            print(f"[SARVAM TTS] Audio generated ({len(audio_b64)} chars b64)")
        except Exception as e:
            print(f"[SARVAM TTS] Error: {e}")
    print(f"{'='*50}\n")
    return {"answer": answer, "audio_b64": audio_b64}

@app.post("/stt")
def stt(req: STTRequest):
    if not SARVAM_API_KEY:
        raise HTTPException(400, "SARVAM_API_KEY not set")
    audio_bytes = base64.b64decode(req.audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        response = sarvam_client.speech_to_text.transcribe(
            file=open(tmp_path, "rb"),
            model="saaras:v3",
            mode="transcribe",
        )
        return {
            "transcript": response.transcript,
            "language_code": response.language_code or "en-IN"
        }
    finally:
        os.unlink(tmp_path)

@app.post("/tts")
def tts(req: TTSRequest):
    try:
        audio_b64 = sarvam_tts(req.text, req.lang_code)
        return {"audio_b64": audio_b64}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/status")
def status():
    return {"chunks": len(chunks), "video_id": current_video_id}