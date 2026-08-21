
from __future__ import annotations

import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import imageio_ffmpeg

MAX_FRAMES = 10


def _seconds_to_mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def _compute_sample_timestamps(duration: float, max_frames: int = MAX_FRAMES) -> list[float]:
    if duration <= 0:
        return [1.0]
    if duration < 25:
        count = min(6, max_frames)
    elif duration < 80:
        count = min(8, max_frames)
    else:
        count = max_frames

    start = min(2.0, max(0.0, duration * 0.05))
    end = max(start + 1, duration - 2.0)
    if count == 1:
        return [min(duration / 2, end)]
    step = (end - start) / (count - 1)
    times = [max(0.0, min(duration, start + i * step)) for i in range(count)]
    return sorted(set(round(t, 2) for t in times))


def extract_keyframes(video_bytes: bytes, original_name: str, max_frames: int = MAX_FRAMES) -> dict[str, Any]:
    suffix = Path(original_name).suffix.lower() or ".mp4"
    temp_dir = Path(tempfile.mkdtemp(prefix="short_drama_frames_"))
    video_path = temp_dir / f"input{suffix}"
    video_path.write_bytes(video_bytes)

    frame_count, duration = imageio_ffmpeg.count_frames_and_secs(str(video_path))
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    timestamps = _compute_sample_timestamps(float(duration), max_frames=max_frames)
    frames = []

    for idx, ts in enumerate(timestamps, 1):
        out_path = temp_dir / f"frame_{idx:02d}.jpg"
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
                "index": idx,
                "timestamp_sec": ts,
                "timestamp_mmss": _seconds_to_mmss(ts),
                "path": str(out_path),
            })

    return {
        "duration_seconds": float(duration),
        "frame_count_estimate": int(frame_count),
        "frames": frames,
        "temp_dir": str(temp_dir),
    }


def encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
