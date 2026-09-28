"""Self-hosted voice assistant: Whisper speech-to-text -> local LLM -> Kokoro text-to-speech, served with FastAPI.

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import io
import os
import tempfile
import time
from contextlib import asynccontextmanager
from urllib.parse import quote

import numpy as np
import soundfile as sf
import torch
import whisper
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, Response
from kokoro import KPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = DEVICE == "cuda"  # half precision only pays off (and is only supported by Whisper) on GPU
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
VOICE = os.getenv("VOICE", "af_heart")
SYSTEM = "You are a helpful voice assistant. Answer in one or two short, natural spoken sentences."
M = {}


@asynccontextmanager
async def lifespan(app):
    # Warm loading: every model is loaded onto the GPU once at startup and run once on dummy
    # input, so requests never pay model-load or CUDA kernel-initialization cost.
    t = time.perf_counter()
    M["stt"] = whisper.load_model(WHISPER_MODEL, device=DEVICE)
    M["tok"] = AutoTokenizer.from_pretrained(LLM_MODEL)
    M["llm"] = AutoModelForCausalLM.from_pretrained(LLM_MODEL).to(
        DEVICE, torch.float16 if FP16 else torch.float32).eval()
    M["tts"] = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=DEVICE)
    transcribe(np.zeros(16000, dtype=np.float32))
    reply("Hi")
    speak("Ready.")
    print(f"Models warm on {DEVICE} (fp16={FP16}) in {time.perf_counter() - t:.1f}s", flush=True)
    yield
    M.clear()


app = FastAPI(title="Whisper Voice Assistant", lifespan=lifespan)


def transcribe(audio: np.ndarray) -> str:
    return M["stt"].transcribe(audio, fp16=FP16, language="en")["text"].strip()


@torch.inference_mode()
def reply(text: str) -> str:
    tok = M["tok"]
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    out = M["llm"].generate(**inputs, max_new_tokens=120, do_sample=False)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def speak(text: str) -> bytes:
    audio = torch.cat([a for _, _, a in M["tts"](text, voice=VOICE) if a is not None])
    buf = io.BytesIO()
    sf.write(buf, audio.cpu().numpy(), 24000, format="WAV")  # Kokoro outputs 24 kHz
    return buf.getvalue()


async def load_upload(audio: UploadFile) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio.filename or "")[1]) as f:
        f.write(await audio.read())
        f.flush()
        return whisper.load_audio(f.name)  # ffmpeg: any format (wav/mp3/webm) -> 16 kHz mono float32


def ms(t0: float, t1: float) -> str:
    return f"{(t1 - t0) * 1000:.0f}"


# ponytail: handlers are async and call the GPU inline, so requests run one at a time on the
# single GPU. Move inference to a worker queue with request batching if concurrency matters.
@app.post("/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    pcm = await load_upload(audio)
    t0 = time.perf_counter()
    text = transcribe(pcm)
    return {"text": text, "stt_ms": int(ms(t0, time.perf_counter()))}


@app.post("/talk")
async def talk(audio: UploadFile = File(...)):
    pcm = await load_upload(audio)
    t0 = time.perf_counter()
    text = transcribe(pcm)
    t1 = time.perf_counter()
    answer = reply(text) if text else "Sorry, I didn't catch that."
    t2 = time.perf_counter()
    wav = speak(answer)
    t3 = time.perf_counter()
    return Response(wav, media_type="audio/wav", headers={
        "X-Transcript": quote(text),
        "X-Reply": quote(answer),
        "X-Latency-Ms": f"stt={ms(t0, t1)},llm={ms(t1, t2)},tts={ms(t2, t3)}",
    })


@app.get("/health")
def health():
    return {"device": DEVICE, "fp16": FP16, "whisper": WHISPER_MODEL, "llm": LLM_MODEL, "ready": bool(M)}


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))
