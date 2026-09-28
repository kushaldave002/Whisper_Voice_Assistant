# Whisper Voice Assistant

A self-hosted voice conversation pipeline that runs entirely on your own GPU, as an alternative to hosted
OpenAI audio APIs:

```
microphone ──► Whisper (speech-to-text) ──► Qwen2.5-0.5B-Instruct (reply) ──► Kokoro-82M (text-to-speech) ──► speaker
                PyTorch, CUDA, FP16           PyTorch, CUDA, FP16                PyTorch, CUDA
```

All three models are open source PyTorch models served behind one FastAPI app. No audio or text leaves the machine.

## What's in the repo

| File | Purpose |
|---|---|
| `app.py` | FastAPI server: `POST /talk` (audio in, spoken reply out), `POST /transcribe`, `GET /health`, browser UI at `/` |
| `index.html` | Minimal browser client: record from the mic, send to `/talk`, play the reply |
| `evaluate.py` | Benchmarks Whisper `tiny` / `base` / `small` on LibriSpeech: WER, latency, real-time factor, peak GPU memory, FP16 vs FP32 |
| `results.md` | Output of `evaluate.py` on a GPU |

## GPU inference optimizations

- **FP16 inference.** On CUDA, Whisper decodes in half precision and the LLM weights are cast to `float16`,
  cutting memory traffic and using tensor cores. `evaluate.py` measures FP16 against FP32 on the same model.
- **Warm model loading.** All models are loaded onto the GPU once at startup (FastAPI `lifespan`) and run once
  on dummy input, so no request pays model-load or CUDA kernel-initialization cost. `evaluate.py` reports the
  load time this avoids per request.
- **Accurate GPU timing.** Latency measurements call `torch.cuda.synchronize()` before stopping the clock,
  because CUDA kernels run asynchronously.
- **Model-size trade-off.** The Whisper size is a single env var (`WHISPER_MODEL`), chosen from the WER vs
  latency table below.

## Run it on a GPU (Google Colab)

Runtime → Change runtime type → **T4 GPU**, then:

```python
!git clone https://github.com/kushaldave002/Whisper_Voice_Assistant.git
%cd Whisper_Voice_Assistant
!pip install -q -r requirements.txt
!apt-get -qq install -y espeak-ng   # Kokoro's fallback for words it can't pronounce
```

Benchmark Whisper (writes `results.md`):

```python
!python evaluate.py
```

Start the server and open the voice UI:

```python
!nohup uvicorn app:app --port 8000 > server.log 2>&1 &
!timeout 900 bash -c 'until grep -q "Models warm" server.log; do sleep 5; done'; tail -n 3 server.log  # first run downloads the models
from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(8000)"))   # open this URL, click "Start talking"
```

## Run it locally

Needs Python 3.10–3.12, `ffmpeg`, and ideally an NVIDIA GPU (it falls back to CPU in FP32). Built for Linux +
CUDA; on macOS the espeak-ng library bundled with Kokoro fails to load its data, so use Colab there.

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000, or call the API directly:

```bash
curl -s -F audio=@question.wav http://localhost:8000/talk -D - -o reply.wav
curl -s -F audio=@question.wav http://localhost:8000/transcribe
```

`/talk` returns the spoken reply as `audio/wav`. The transcript, the reply text and per-stage latency
(`stt`, `llm`, `tts` in ms) come back in the `X-Transcript`, `X-Reply` and `X-Latency-Ms` headers.

Configuration (env vars): `WHISPER_MODEL` (default `base`), `LLM_MODEL` (default `Qwen/Qwen2.5-0.5B-Instruct`),
`VOICE` (Kokoro voice, default `af_heart`).

## Results

Whisper on LibriSpeech test clips, measured with `python evaluate.py`. See [`results.md`](results.md) for the
table from the latest GPU run: word error rate, mean latency per clip, real-time factor, peak GPU memory and
load time for each model size in FP16 and FP32.

## Limitations

- Requests are handled one at a time on a single GPU. Serving many users would need a worker queue with
  request batching.
- Each `/talk` call is a single turn; conversation history is not kept between requests.
- English only (`language="en"`, Kokoro American English voice).
