
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import imageio_ffmpeg
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"


def _mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


def build_sliding_windows(
    transcription: dict[str, Any],
    window_sec: float = 12.0,
    stride_sec: float = 6.0,
) -> list[dict[str, Any]]:
    segments = transcription.get("segments") or []
    if not segments:
        return []

    last_end = max(float(s.get("end") or 0) for s in segments)
    windows = []
    start = 0.0
    idx = 1

    while start < last_end:
        end = min(last_end, start + window_sec)
        texts = []

        for seg in segments:
            s = float(seg.get("start") or 0)
            e = float(seg.get("end") or s)
            if e >= start and s <= end:
                text = (seg.get("text") or "").strip()
                if text:
                    texts.append(text)

        joined = " ".join(texts).strip()
        if len(joined) >= 18:
            windows.append({
                "index": idx,
                "start_sec": start,
                "end_sec": end,
                "start_mmss": _mmss(start),
                "end_mmss": _mmss(end),
                "text": joined,
            })
            idx += 1

        start += stride_sec

    return windows


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def rank_semantic_repeat_candidates(
    *,
    api_key: str,
    windows: list[dict[str, Any]],
    min_similarity: float = 0.70,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    if len(windows) < 2:
        return []

    client = OpenAI(api_key=api_key.strip())
    emb = client.embeddings.create(
        model=EMBED_MODEL,
        input=[w["text"] for w in windows],
    )
    vectors = [item.embedding for item in emb.data]

    candidates = []
    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            # Avoid heavily overlapping windows.
            if windows[j]["start_sec"] - windows[i]["start_sec"] < 10.0:
                continue
            # Keep comparisons reasonably local for short-drama duplicate blocks.
            if windows[j]["start_sec"] - windows[i]["start_sec"] > 70.0:
                continue

            sim = _cosine(vectors[i], vectors[j])
            if sim >= min_similarity:
                candidates.append({
                    "candidate_id": f"SEM-{windows[i]['index']:03d}-{windows[j]['index']:03d}",
                    "similarity": round(sim, 3),
                    "earlier": windows[i],
                    "later": windows[j],
                })

    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    return candidates[:top_k]


def extract_repeat_window_frames(
    *,
    video_bytes: bytes,
    original_name: str,
    candidates: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    suffix = Path(original_name).suffix.lower() or ".mp4"
    temp_dir = Path(tempfile.mkdtemp(prefix="short_drama_repeat_"))
    video_path = temp_dir / f"input{suffix}"
    video_path.write_bytes(video_bytes)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    unique_windows = {}
    for c in candidates:
        unique_windows[c["earlier"]["index"]] = c["earlier"]
        unique_windows[c["later"]["index"]] = c["later"]

    result = {}
    for idx, window in unique_windows.items():
        start = float(window["start_sec"])
        end = float(window["end_sec"])
        duration = max(0.5, end - start)
        times = [
            start + duration * 0.20,
            start + duration * 0.50,
            start + duration * 0.80,
        ]
        frames = []
        for k, ts in enumerate(times, 1):
            out_path = temp_dir / f"win_{idx:03d}_{k}.jpg"
            cmd = [
                ffmpeg_exe,
                "-y",
                "-ss", str(ts),
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "2",
                str(out_path),
            ]
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if completed.returncode == 0 and out_path.exists():
                frames.append({
                    "frame_index": k,
                    "timestamp_sec": ts,
                    "timestamp_mmss": _mmss(ts),
                    "path": str(out_path),
                })
        result[idx] = frames

    return result
