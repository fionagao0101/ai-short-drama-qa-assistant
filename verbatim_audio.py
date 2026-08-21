
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from media_pipeline import extract_audio_from_video

VERBATIM_MODEL = "whisper-1"


def _norm(token: str) -> str:
    token = (token or "").lower().strip()
    token = re.sub(r"[^\wáàâãéêíóôõúç]", "", token, flags=re.IGNORECASE)
    return token


def _mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


def run_verbatim_word_transcription(
    *,
    api_key: str,
    video_bytes: bytes,
    file_name: str,
) -> dict[str, Any]:
    audio_path = extract_audio_from_video(video_bytes, file_name)
    client = OpenAI(api_key=api_key.strip())

    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=VERBATIM_MODEL,
            file=f,
            language="pt",
            response_format="verbose_json",
            timestamp_granularities=["word"],
            prompt=(
                "Transcreva literalmente. Preserve repetições, palavras duplicadas, "
                "hesitações e reinícios. Não corrija nem normalize repetições."
            ),
        )

    if hasattr(result, "model_dump"):
        data = result.model_dump()
    elif isinstance(result, dict):
        data = result
    else:
        data = {
            "text": getattr(result, "text", ""),
            "words": [
                w.model_dump() if hasattr(w, "model_dump") else {
                    "word": getattr(w, "word", ""),
                    "start": getattr(w, "start", 0),
                    "end": getattr(w, "end", 0),
                }
                for w in (getattr(result, "words", []) or [])
            ]
        }

    return data


def detect_verbatim_repetitions(
    transcription: dict[str, Any],
    max_gap_sec: float = 0.85,
) -> list[dict[str, Any]]:
    words = transcription.get("words") or []
    out = []

    for i in range(len(words) - 1):
        w1 = words[i]
        w2 = words[i + 1]
        t1 = _norm(w1.get("word", ""))
        t2 = _norm(w2.get("word", ""))

        if not t1 or len(t1) < 3 or t1 != t2:
            continue

        end1 = float(w1.get("end") or w1.get("start") or 0)
        start2 = float(w2.get("start") or 0)
        gap = max(0.0, start2 - end1)

        if gap <= max_gap_sec:
            out.append({
                "type": "adjacent_duplicate_word",
                "word": t1,
                "start_sec": float(w1.get("start") or 0),
                "end_sec": float(w2.get("end") or start2),
                "start_mmss": _mmss(float(w1.get("start") or 0)),
                "end_mmss": _mmss(float(w2.get("end") or start2)),
                "surface": f"{w1.get('word','')} {w2.get('word','')}",
                "gap_sec": round(gap, 3),
            })

    # Also detect immediate repeated 2-word phrase.
    tokens = [_norm(w.get("word", "")) for w in words]
    for i in range(len(tokens) - 3):
        a = tokens[i:i+2]
        b = tokens[i+2:i+4]
        if all(a) and a == b and len(" ".join(a)) >= 6:
            out.append({
                "type": "adjacent_duplicate_phrase",
                "phrase": " ".join(a),
                "start_sec": float(words[i].get("start") or 0),
                "end_sec": float(words[i+3].get("end") or 0),
                "start_mmss": _mmss(float(words[i].get("start") or 0)),
                "end_mmss": _mmss(float(words[i+3].get("end") or 0)),
                "surface": " ".join(w.get("word", "") for w in words[i:i+4]),
                "gap_sec": 0.0,
            })

    # Deduplicate
    seen, final = set(), []
    for item in out:
        key = (
            item["type"],
            item.get("word") or item.get("phrase"),
            item["start_mmss"],
            item["end_mmss"],
        )
        if key not in seen:
            seen.add(key)
            final.append(item)

    return final[:12]


def format_repetition_evidence(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return ""

    lines = [
        "【Verbatim Audio Repetition Candidates】",
        "以下候选来自逐词时间戳转写，专门用于检测 ASR 清洗掉的音频复读。"
    ]

    for c in candidates:
        if c["type"] == "adjacent_duplicate_word":
            lines.append(
                f"- {c['start_mmss']}–{c['end_mmss']} | "
                f"连续重复词：{c['surface']} | gap={c['gap_sec']}s"
            )
        else:
            lines.append(
                f"- {c['start_mmss']}–{c['end_mmss']} | "
                f"连续重复短语：{c['surface']}"
            )

    return "\n".join(lines)
