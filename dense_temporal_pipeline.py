
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import imageio_ffmpeg


def _seconds_to_mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def _precise_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def build_dense_segments(
    transcription: dict[str, Any],
    max_segments: int = 40,
) -> list[dict[str, Any]]:
    raw = transcription.get("segments") or []
    segments = []

    for idx, seg in enumerate(raw, 1):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start") or 0)
        end = float(seg.get("end") or start)
        if end <= start:
            end = start + 0.4
        segments.append({
            "index": idx,
            "start_sec": start,
            "end_sec": end,
            "start_mmss": _seconds_to_mmss(start),
            "end_mmss": _seconds_to_mmss(end),
            "start_precise": _precise_time(start),
            "end_precise": _precise_time(end),
            "duration_sec": end - start,
            "speaker": seg.get("speaker") or "Unknown",
            "text": text,
        })

    if len(segments) <= max_segments:
        return segments

    # Preserve whole timeline, including the end, through even sampling.
    keep = []
    for i in range(max_segments):
        pos = round(i * (len(segments) - 1) / (max_segments - 1))
        keep.append(segments[pos])

    out, seen = [], set()
    for seg in keep:
        if seg["index"] not in seen:
            out.append(seg)
            seen.add(seg["index"])
    return out


def _relative_times(start: float, end: float) -> list[float]:
    duration = max(0.25, end - start)

    # Four frames across the full utterance.
    rels = [0.08, 0.38, 0.68, 0.94]
    times = [start + duration * r for r in rels]

    # Ensure monotonic, bounded values.
    clean = []
    for t in times:
        t = max(start, min(end, t))
        if not clean or abs(t - clean[-1]) > 0.06:
            clean.append(t)

    return clean


def extract_dense_frames(
    *,
    video_bytes: bytes,
    original_name: str,
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    suffix = Path(original_name).suffix.lower() or ".mp4"
    temp_dir = Path(tempfile.mkdtemp(prefix="short_drama_dense_"))
    video_path = temp_dir / f"input{suffix}"
    video_path.write_bytes(video_bytes)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    out_segments = []

    for seg in segments:
        seg_out = dict(seg)
        seg_out["frames"] = []

        for frame_idx, ts in enumerate(_relative_times(seg["start_sec"], seg["end_sec"]), 1):
            out_path = temp_dir / f"seg_{seg['index']:03d}_{frame_idx}.jpg"
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
                out_seg_frame = {
                    "frame_index": frame_idx,
                    "timestamp_sec": ts,
                    "timestamp_mmss": _seconds_to_mmss(ts),
                    "timestamp_precise": _precise_time(ts),
                    "path": str(out_path),
                }
                seg_out["frames"].append(out_seg_frame)

        out_segments.append(seg_out)

    return out_segments
