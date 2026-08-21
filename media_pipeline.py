
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import imageio_ffmpeg
from openai import OpenAI

TRANSCRIBE_MODEL = "gpt-4o-transcribe-diarize"


def _seconds_to_mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def extract_audio_from_video(video_bytes: bytes, original_name: str) -> Path:
    suffix = Path(original_name).suffix.lower() or ".mp4"
    temp_dir = Path(tempfile.mkdtemp(prefix="short_drama_audio_"))
    video_path = temp_dir / f"input{suffix}"
    audio_path = temp_dir / "audio.mp3"
    video_path.write_bytes(video_bytes)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", "64k",
        str(audio_path),
    ]
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0 or not audio_path.exists():
        raise RuntimeError("Audio extraction failed.\n" + completed.stderr[-3000:])
    return audio_path


def transcribe_with_diarization(*, api_key: str, audio_path: Path, language: str = "pt") -> dict[str, Any]:
    client = OpenAI(api_key=api_key.strip())
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=audio_file,
            language=language,
            response_format="diarized_json",
            chunking_strategy="auto",
        )
    if hasattr(transcript, "model_dump"):
        data = transcript.model_dump()
    elif isinstance(transcript, dict):
        data = transcript
    else:
        segments = []
        for seg in getattr(transcript, "segments", []) or []:
            if hasattr(seg, "model_dump"):
                segments.append(seg.model_dump())
            elif isinstance(seg, dict):
                segments.append(seg)
            else:
                segments.append({
                    "start": getattr(seg, "start", 0),
                    "end": getattr(seg, "end", 0),
                    "text": getattr(seg, "text", ""),
                    "speaker": getattr(seg, "speaker", None),
                })
        data = {
            "text": getattr(transcript, "text", ""),
            "duration": getattr(transcript, "duration", None),
            "segments": segments,
        }
    return data


def format_diarized_evidence(transcription: dict[str, Any]) -> str:
    segments = transcription.get("segments") or []
    lines = [
        "【自动 ASR 音频证据】",
        "说明：以下内容仅来自音轨转写，不代表烧录字幕；Speaker A/B/C 只是声纹聚类标签，不等于角色身份。",
        "",
    ]
    if not segments:
        text = (transcription.get("text") or "").strip()
        if text:
            lines.extend([
                "[00:00–00:00]",
                "问题载体：配音/ASR",
                f"配音自动转写：{text}",
            ])
        return "\n".join(lines)

    for seg in segments:
        start = _seconds_to_mmss(float(seg.get("start") or 0))
        end = _seconds_to_mmss(float(seg.get("end") or 0))
        speaker = seg.get("speaker") or "Unknown"
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.extend([
            f"[{start}–{end}]",
            "问题载体：配音/ASR",
            f"说话人：Speaker {speaker}",
            f"配音自动转写：{text}",
            "",
        ])
    return "\n".join(lines).strip()


def process_video_to_evidence(*, api_key: str, video_bytes: bytes, file_name: str) -> dict[str, Any]:
    audio_path = extract_audio_from_video(video_bytes, file_name)
    transcription = transcribe_with_diarization(api_key=api_key, audio_path=audio_path, language="pt")
    evidence = format_diarized_evidence(transcription)
    return {
        "transcription": transcription,
        "evidence": evidence,
        "transcription_model": TRANSCRIBE_MODEL,
        "audio_file_size_bytes": audio_path.stat().st_size,
    }
