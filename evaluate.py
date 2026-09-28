"""Benchmark Whisper model sizes on GPU: WER, latency, real-time factor, peak GPU memory, FP16 vs FP32.

Run:  python evaluate.py                      # tiny, base, small on all clips -> results.md
      python evaluate.py --models tiny --limit 5
"""
import argparse
import io
import time

import jiwer
import numpy as np
import soundfile as sf
import torch
import whisper
from datasets import Audio, load_dataset
from whisper.normalizers import EnglishTextNormalizer

p = argparse.ArgumentParser()
p.add_argument("--models", default="tiny,base,small")
p.add_argument("--limit", type=int, default=0, help="only use the first N clips (0 = all)")
p.add_argument("--out", default="results.md")
args = p.parse_args()

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"


def sync():
    if cuda:
        torch.cuda.synchronize()  # CUDA is async; without this, timings measure kernel launch, not work


# LibriSpeech (clean) sample clips with reference transcripts, 16 kHz.
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(decode=False))
if args.limit:
    ds = ds.select(range(args.limit))
clips = [(sf.read(io.BytesIO(a["bytes"]) if a["bytes"] else a["path"], dtype="float32")[0], text)
         for a, text in zip(ds["audio"], ds["text"])]
audio_sec = sum(len(x) for x, _ in clips) / 16000
normalize = EnglishTextNormalizer()

rows = []
for name in args.models.split(","):
    whisper.load_model(name, device="cpu")  # make sure weights are downloaded, so load time below excludes it
    sync()
    t = time.perf_counter()
    model = whisper.load_model(name, device=device)
    sync()
    load_s = time.perf_counter() - t
    for fp16 in ([True, False] if cuda else [False]):
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        model.transcribe(clips[0][0], fp16=fp16, language="en")  # warm-up, not timed
        hyps, lat = [], []
        for x, _ in clips:
            sync()
            t = time.perf_counter()
            hyps.append(model.transcribe(x, fp16=fp16, language="en")["text"])
            sync()
            lat.append(time.perf_counter() - t)
        wer = jiwer.wer([normalize(r) for _, r in clips], [normalize(h) for h in hyps])
        mem = torch.cuda.max_memory_allocated() / 2**30 if cuda else float("nan")
        rows.append((name, "FP16" if fp16 else "FP32", wer, np.mean(lat), sum(lat) / audio_sec, mem, load_s))
        print(*rows[-1])
    del model
    if cuda:
        torch.cuda.empty_cache()

hw = torch.cuda.get_device_name() if cuda else "CPU"
out = [f"{len(clips)} LibriSpeech clips ({audio_sec / 60:.1f} min of audio) on {hw}. "
       "RTF = processing time / audio duration (lower is faster).", "",
       "| Model | Precision | WER | Mean latency (s) | RTF | Peak GPU memory (GB) | Load time (s) |",
       "|---|---|---|---|---|---|---|"]
out += [f"| {n} | {pr} | {w:.2%} | {l:.3f} | {r:.3f} | {m:.2f} | {s:.1f} |" for n, pr, w, l, r, m, s in rows]
open(args.out, "w").write("\n".join(out) + "\n")
print("\n".join(out))
