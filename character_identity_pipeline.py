
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import imageio_ffmpeg


def _parse_mmss(value: str) -> float:
    value = (value or "").strip()
    m = re.fullmatch(r"(\d{1,3}):(\d{2})", value)
    if not m:
        raise ValueError(f"Invalid timestamp: {value}. Use mm:ss, e.g. 00:18")
    return int(m.group(1)) * 60 + int(m.group(2))


def parse_video_reference_lines(text: str) -> list[dict[str, Any]]:
    """
    Format:
    Lucas|男主|00:18
    Rafael|男二|00:33
    """
    refs = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            raise ValueError(
                f"Reference line must be Name|Role|mm:ss. Got: {line}"
            )

        name, role, timestamp = parts
        refs.append({
            "name": name,
            "role": role,
            "timestamp_text": timestamp,
            "timestamp_sec": _parse_mmss(timestamp),
            "source": "video_timestamp",
        })
    return refs


def save_uploaded_asset_references(
    uploaded_assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    uploaded_assets items:
    {
      "name": "Lucas",
      "role": "男主",
      "file_name": "...jpg",
      "bytes": b"..."
    }
    """
    if not uploaded_assets:
        return []

    temp_dir = Path(tempfile.mkdtemp(prefix="short_drama_character_assets_"))
    out = []

    for idx, item in enumerate(uploaded_assets, 1):
        suffix = Path(item["file_name"]).suffix.lower() or ".jpg"
        path = temp_dir / f"asset_{idx:02d}{suffix}"
        path.write_bytes(item["bytes"])

        out.append({
            "name": item["name"].strip() or f"Character_{idx}",
            "role": item["role"].strip() or "未指定",
            "path": str(path),
            "source": "uploaded_asset",
            "source_file_name": item["file_name"],
        })

    return out


def extract_video_reference_frames(
    *,
    video_bytes: bytes,
    original_name: str,
    reference_lines: str,
) -> list[dict[str, Any]]:
    refs = parse_video_reference_lines(reference_lines)
    if not refs:
        return []

    suffix = Path(original_name).suffix.lower() or ".mp4"
    temp_dir = Path(tempfile.mkdtemp(prefix="short_drama_character_video_refs_"))
    video_path = temp_dir / f"input{suffix}"
    video_path.write_bytes(video_bytes)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    out = []

    for idx, ref in enumerate(refs, 1):
        out_path = temp_dir / f"video_ref_{idx:02d}.jpg"
        cmd = [
            ffmpeg_exe,
            "-y",
            "-ss", str(ref["timestamp_sec"]),
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
            item = dict(ref)
            item["path"] = str(out_path)
            out.append(item)

    return out


def select_character_check_segments(
    dense_segments: list[dict[str, Any]],
    max_segments: int = 16,
) -> list[dict[str, Any]]:
    """
    Prefer longer dialogue segments and still cover the entire timeline.
    """
    candidates = [
        seg for seg in dense_segments
        if seg.get("frames") and float(seg.get("duration_sec") or 0) >= 0.5
    ]

    if len(candidates) <= max_segments:
        return candidates

    # Blend timeline coverage with longer dialogue segments.
    timeline_positions = set()
    for i in range(max_segments // 2):
        pos = round(i * (len(candidates) - 1) / max(1, (max_segments // 2) - 1))
        timeline_positions.add(pos)

    long_positions = sorted(
        range(len(candidates)),
        key=lambda i: float(candidates[i].get("duration_sec") or 0),
        reverse=True,
    )[:max_segments]

    chosen_positions = list(timeline_positions)
    for p in long_positions:
        if len(chosen_positions) >= max_segments:
            break
        if p not in chosen_positions:
            chosen_positions.append(p)

    chosen_positions = sorted(chosen_positions[:max_segments])
    return [candidates[p] for p in chosen_positions]


def select_deep_review_segments_from_risk(
    dense_segments: list[dict[str, Any]],
    risk_windows: list[dict[str, Any]],
    max_segments: int = 6,
) -> list[dict[str, Any]]:
    """
    Map router risk windows back to dense multi-frame ASR segments.
    Only high-risk windows enter Deep Character Review.
    """
    if not risk_windows:
        return []

    requested_indices = []
    for w in risk_windows:
        idx = int(w.get("segment_index") or 0)
        if idx > 0:
            requested_indices.extend([idx - 1, idx, idx + 1])

    selected = []
    seen = set()

    for seg in dense_segments:
        if seg["index"] in requested_indices and seg["index"] not in seen:
            selected.append(seg)
            seen.add(seg["index"])

    # Prefer exact risk hits, then timeline order.
    selected.sort(key=lambda s: s["index"])
    return selected[:max_segments]
