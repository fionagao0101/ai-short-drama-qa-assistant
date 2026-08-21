
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from difflib import SequenceMatcher

import imageio_ffmpeg


def _seconds_to_mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def _normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\sáàâãéêíóôõúç]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_aligned_segments(
    transcription: dict[str, Any],
    max_segments: int = 30,
) -> list[dict[str, Any]]:
    """
    Build segment records from ASR output.
    We keep as many ASR segments as possible (up to max_segments) so that later
    subtitle-presence and subtitle-mismatch checks are time-aligned rather than
    based on random global frames.
    """
    raw_segments = transcription.get("segments") or []
    segments = []

    for idx, seg in enumerate(raw_segments, 1):
        start = float(seg.get("start") or 0)
        end = float(seg.get("end") or start)
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        speaker = seg.get("speaker") or "Unknown"
        duration = max(0.1, end - start)

        segments.append({
            "index": idx,
            "start_sec": start,
            "end_sec": end,
            "start_mmss": _seconds_to_mmss(start),
            "end_mmss": _seconds_to_mmss(end),
            "duration_sec": duration,
            "speaker": speaker,
            "text": text,
            "norm_text": _normalize_text(text),
        })

    if len(segments) <= max_segments:
        return segments

    # If too many, keep them evenly sampled but retain more near the end as well.
    keep = []
    for i in range(max_segments):
        pos = round(i * (len(segments) - 1) / (max_segments - 1))
        keep.append(segments[pos])

    dedup = []
    seen = set()
    for seg in keep:
        if seg["index"] not in seen:
            dedup.append(seg)
            seen.add(seg["index"])
    return dedup


def extract_aligned_frames(
    video_bytes: bytes,
    original_name: str,
    aligned_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    For each ASR-aligned segment, extract one representative frame near the mid-point.
    This is the key improvement over Step 5C's random keyframes.
    """
    suffix = Path(original_name).suffix.lower() or ".mp4"
    temp_dir = Path(tempfile.mkdtemp(prefix="short_drama_aligned_"))
    video_path = temp_dir / f"input{suffix}"
    video_path.write_bytes(video_bytes)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    out_segments = []

    for seg in aligned_segments:
        mid = (seg["start_sec"] + seg["end_sec"]) / 2
        out_path = temp_dir / f"seg_{seg['index']:03d}.jpg"

        cmd = [
            ffmpeg_exe,
            "-y",
            "-ss", str(mid),
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

        new_seg = dict(seg)
        if completed.returncode == 0 and out_path.exists():
            new_seg["frame_path"] = str(out_path)
        else:
            new_seg["frame_path"] = None
        out_segments.append(new_seg)

    return out_segments


def detect_duplicate_candidates(
    transcription: dict[str, Any],
    similarity_threshold: float = 0.88,
    min_window_chars: int = 24,
    min_gap_sec: float = 20.0,
) -> list[dict[str, Any]]:
    """
    Detect likely duplicated dialogue windows using ASR transcript similarity.
    We compare 2-segment windows to later 2-segment windows.
    """
    raw_segments = transcription.get("segments") or []
    base = []
    for idx, seg in enumerate(raw_segments, 1):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        base.append({
            "index": idx,
            "start_sec": float(seg.get("start") or 0),
            "end_sec": float(seg.get("end") or 0),
            "start_mmss": _seconds_to_mmss(float(seg.get("start") or 0)),
            "end_mmss": _seconds_to_mmss(float(seg.get("end") or 0)),
            "text": text,
            "norm_text": _normalize_text(text),
        })

    windows = []
    for i in range(len(base) - 1):
        text = (base[i]["norm_text"] + " " + base[i + 1]["norm_text"]).strip()
        if len(text) < min_window_chars:
            continue
        windows.append({
            "first_seg_index": base[i]["index"],
            "last_seg_index": base[i + 1]["index"],
            "start_sec": base[i]["start_sec"],
            "end_sec": base[i + 1]["end_sec"],
            "start_mmss": base[i]["start_mmss"],
            "end_mmss": base[i + 1]["end_mmss"],
            "text": (base[i]["text"] + " " + base[i + 1]["text"]).strip(),
            "norm_text": text,
        })

    candidates = []
    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            if windows[j]["start_sec"] - windows[i]["end_sec"] < min_gap_sec:
                continue
            sim = SequenceMatcher(None, windows[i]["norm_text"], windows[j]["norm_text"]).ratio()
            if sim >= similarity_threshold:
                candidates.append({
                    "candidate_id": f"DUP-{i+1:03d}-{j+1:03d}",
                    "earlier_start_mmss": windows[i]["start_mmss"],
                    "earlier_end_mmss": windows[i]["end_mmss"],
                    "later_start_mmss": windows[j]["start_mmss"],
                    "later_end_mmss": windows[j]["end_mmss"],
                    "earlier_text": windows[i]["text"],
                    "later_text": windows[j]["text"],
                    "similarity": round(sim, 3),
                    "earlier_start_sec": windows[i]["start_sec"],
                    "later_start_sec": windows[j]["start_sec"],
                })

    # Deduplicate near-identical later windows
    final = []
    seen = set()
    for c in candidates:
        key = (c["later_start_mmss"], c["later_end_mmss"], c["earlier_start_mmss"])
        if key not in seen:
            seen.add(key)
            final.append(c)
    return final[:8]


def detect_repeated_word_candidates(
    transcription: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Lightweight local detector for obvious repeated word patterns that ASR preserved,
    e.g. 'quente quente' or 'não não'.
    """
    raw_segments = transcription.get("segments") or []
    results = []
    for idx, seg in enumerate(raw_segments, 1):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        norm = _normalize_text(text)
        tokens = norm.split()
        for i in range(len(tokens) - 1):
            if tokens[i] and tokens[i] == tokens[i + 1]:
                results.append({
                    "segment_index": idx,
                    "start_mmss": _seconds_to_mmss(float(seg.get("start") or 0)),
                    "end_mmss": _seconds_to_mmss(float(seg.get("end") or 0)),
                    "text": text,
                    "repeated_token": tokens[i],
                })
                break
    return results[:10]
