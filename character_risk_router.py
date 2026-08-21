
from __future__ import annotations

import base64
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import imageio_ffmpeg
from openai import OpenAI

from llm_review import MODEL_DEFAULT


ROLE_SELECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_index": {"type": "integer"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "relation_type": {
                        "type": "string",
                        "enum": [
                            "direct_name_address",
                            "third_person_exclusion",
                            "identity_claim",
                            "relationship_title",
                            "role_sensitive_question",
                            "role_sensitive_statement"
                        ]
                    },
                    "expected_character": {"type": ["string", "null"]},
                    "excluded_characters": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "expected_role": {"type": ["string", "null"]},
                    "excluded_roles": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "scene_role_hypothesis": {"type": ["string", "null"]},
                    "reason": {"type": "string"},
                    "priority": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5
                    }
                },
                "required": [
                    "segment_index",
                    "start_time",
                    "end_time",
                    "relation_type",
                    "expected_character",
                    "excluded_characters",
                    "expected_role",
                    "excluded_roles",
                    "scene_role_hypothesis",
                    "reason",
                    "priority"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": ["candidates"],
    "additionalProperties": False
}


ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "episode_risk_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10
        },
        "trigger_deep_review": {"type": "boolean"},
        "risk_windows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_index": {"type": "integer"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "risk_score": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 5
                    },
                    "risk_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "face_identity_shift",
                                "same_role_collision",
                                "reference_mismatch",
                                "dialogue_role_conflict",
                                "dialogue_addressee_conflict",
                                "active_speaker_conflict",
                                "third_person_identity_conflict",
                                "relationship_logic_conflict",
                                "scene_role_conflict",
                                "dialogue_addressee_impossibility",
                                "uncertain_identity"
                            ]
                        }
                    },
                    "suspected_character": {"type": ["string", "null"]},
                    "conflicting_character": {"type": ["string", "null"]},
                    "reason": {"type": "string"}
                },
                "required": [
                    "segment_index",
                    "start_time",
                    "end_time",
                    "risk_score",
                    "risk_types",
                    "suspected_character",
                    "conflicting_character",
                    "reason"
                ],
                "additionalProperties": False
            }
        },
        "summary": {"type": "string"}
    },
    "required": [
        "episode_risk_score",
        "trigger_deep_review",
        "risk_windows",
        "summary"
    ],
    "additionalProperties": False
}


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


def _raw_segments(transcription: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, seg in enumerate(transcription.get("segments") or [], 1):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start") or 0)
        end = float(seg.get("end") or start)
        out.append({
            "index": idx,
            "start_sec": start,
            "end_sec": end,
            "start_mmss": _mmss(start),
            "end_mmss": _mmss(end),
            "speaker": seg.get("speaker") or "Unknown",
            "text": text,
        })
    return out


def _heuristic_role_candidates(
    segs: list[dict[str, Any]],
    references: list[dict[str, Any]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    """
    Local fallback used only when a semantic-selector API chunk times out.
    It is intentionally recall-oriented: relationship/pronoun/name-sensitive dialogue
    is surfaced for the later visual Character Router.
    """
    names = [str(r.get("name") or "").strip() for r in references if r.get("name")]
    names_lower = [n.lower() for n in names]

    relation_terms = [
        "amigo", "amiga", "marido", "esposa", "namorado", "namorada",
        "noivo", "noiva", "irmão", "irmã", "pai", "mãe", "filho", "filha",
        "chefe", "patrão", "patroa", "doutor", "doutora", "médico", "médica",
        "secretário", "secretária", "assistente", "esposo", "mulher",
    ]
    perspective_terms = [
        " ele ", " ela ", " dele ", " dela ", " seu ", " sua ", " teu ", " tua ",
        " marido ", " esposa ", " chefe ", " amigo ", " amiga ",
    ]

    scored = []
    for seg in segs:
        text = f" {(seg.get('text') or '').lower()} "
        score = 0
        relation_type = "role_sensitive_statement"
        excluded_characters = []
        expected_character = None

        matched_name = None
        for original, lower in zip(names, names_lower):
            if lower and lower in text:
                matched_name = original
                score += 4
                relation_type = "direct_name_address"

        if any(term in text for term in relation_terms):
            score += 3
            if relation_type == "role_sensitive_statement":
                relation_type = "relationship_title"

        if any(term in text for term in perspective_terms):
            score += 2
            if relation_type == "role_sensitive_statement":
                relation_type = "perspective_conflict"

        if "?" in (seg.get("text") or ""):
            score += 1
            if relation_type == "role_sensitive_statement":
                relation_type = "role_sensitive_question"

        # Named third-person / relationship mention is useful as an exclusion candidate.
        if matched_name and any(term in text for term in ["amigo", "amiga", "ele", "ela", "dele", "dela"]):
            excluded_characters = [matched_name]
            relation_type = "third_person_exclusion"
            score += 2

        if score <= 0:
            continue

        priority = max(1, min(5, 1 + score // 2))
        scored.append({
            "segment_index": int(seg["index"]),
            "start_time": seg["start_mmss"],
            "end_time": seg["end_mmss"],
            "relation_type": relation_type,
            "expected_character": expected_character,
            "excluded_characters": excluded_characters,
            "expected_role": None,
            "excluded_roles": [],
            "scene_role_hypothesis": (
                "API 超时后的本地降级候选：该段包含人物关系/视角/称谓信号，"
                "需结合前后台词与人脸 reference 判断当前对话对象身份。"
            ),
            "reason": "角色语义 API 分块超时，使用本地关系词/视角词规则保留该段进入后续视觉路由。",
            "priority": priority,
        })

    scored.sort(key=lambda c: int(c.get("priority") or 0), reverse=True)

    # Ensure timeline coverage if heuristics are sparse.
    if len(scored) < min(3, max_candidates) and segs:
        used = {c["segment_index"] for c in scored}
        for i in range(min(3, len(segs))):
            pos = round(i * (len(segs) - 1) / max(1, min(3, len(segs)) - 1))
            seg = segs[pos]
            if seg["index"] in used:
                continue
            scored.append({
                "segment_index": int(seg["index"]),
                "start_time": seg["start_mmss"],
                "end_time": seg["end_mmss"],
                "relation_type": "scene_role_inference",
                "expected_character": None,
                "excluded_characters": [],
                "expected_role": None,
                "excluded_roles": [],
                "scene_role_hypothesis": "降级模式下的时间线覆盖候选。",
                "reason": "语义选择器超时后的保底时间线候选，交由视觉 Router 做风险预检。",
                "priority": 1,
            })
            used.add(seg["index"])

    return scored[:max_candidates]


def select_role_sensitive_candidates(
    *,
    api_key: str,
    transcription: dict[str, Any],
    references: list[dict[str, Any]],
    model: str = MODEL_DEFAULT,
    max_candidates: int = 8,
    chunk_size: int = 14,
    overlap: int = 4,
    progress_callback=None,
) -> list[dict[str, Any]]:
    """
    Robust semantic selector.

    Instead of sending the full transcript in one heavy request:
    - scan chunks of ~14 ASR turns
    - retain 4-turn overlap for relationship context
    - timeout/failure of one chunk falls back locally
    - merge/dedupe candidates by segment index
    """
    if not references:
        return []

    segs = _raw_segments(transcription)
    if not segs:
        return []

    roster = "\n".join(
        f"- {r['name']} | {r['role']}"
        for r in references
    )

    instructions = f"""
你是“角色语义候选选择器”。只做文本/剧情关系分析，不判断人脸。

已知角色 reference：
{roster}

你现在看到的是整集台词中的一个【连续分块】，块之间有上下文重叠。
请只输出这个分块中最值得进入角色人脸审核的时间点。

【名字不是必需条件。】
要重点找：
- 当前这句话/整段话不可能对某个角色说；
- 当前对面的人应当是某种身份，而不应是另一身份；
- “我/你/他/她/丈夫/妻子/哥哥/老板/医生/朋友”等关系能排除某个角色；
- 连续 3–5 句共同推出 scene role；
- 名字/第三人称提及只是其中一种证据。

relation_type 可使用：
direct_name_address
third_person_exclusion
identity_claim
relationship_title
role_sensitive_question
role_sensitive_statement
dialogue_addressee_impossibility
relationship_logic_conflict
perspective_conflict
scene_role_inference

输出字段：
- expected_character：明确到 reference 名字时才填，否则 null
- excluded_characters：能明确排除 reference 名字时填
- expected_role：能推断“男主/男二/医生/朋友/下属”等时填
- excluded_roles：能排除的角色身份
- scene_role_hypothesis：当前对面的人剧情上应该是什么身份
- reason：必须引用该块内的上下文逻辑
- priority 1–5

误报控制：
- 必须结合上下文，不凭单句脑补。
- 如可能是失忆、伪装、故意装不认识、身份欺骗，priority 降低。
- 只输出身份一旦错绑就会使剧情明显不成立的候选。
"""

    # Create chunks with overlap.
    chunks = []
    step = max(1, chunk_size - overlap)
    start_idx = 0
    while start_idx < len(segs):
        chunk = segs[start_idx:start_idx + chunk_size]
        if chunk:
            chunks.append(chunk)
        if start_idx + chunk_size >= len(segs):
            break
        start_idx += step

    all_candidates = []
    failed_chunks = 0

    # Text-only call: explicit timeout + limited automatic retry.
    client = OpenAI(
        api_key=api_key.strip(),
        timeout=75.0,
        max_retries=1,
    )

    for chunk_no, chunk in enumerate(chunks, start=1):
        transcript_text = "\n".join(
            f"SEG {s['index']} | {s['start_mmss']}–{s['end_mmss']} | {s['text']}"
            for s in chunk
        )

        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=transcript_text,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "role_sensitive_dialogue_candidates_chunk",
                        "schema": ROLE_SELECTOR_SCHEMA,
                        "strict": True,
                    }
                },
            )

            raw = response.output_text
            parsed = json.loads(raw) if raw else {"candidates": []}
            chunk_candidates = parsed.get("candidates", [])
            status = "AI"
        except Exception:
            failed_chunks += 1
            chunk_candidates = _heuristic_role_candidates(
                chunk,
                references,
                max_candidates=max(2, min(4, max_candidates)),
            )
            status = "fallback"

        all_candidates.extend(chunk_candidates)

        if progress_callback:
            progress_callback(
                chunk_no,
                len(chunks),
                status,
                chunk[0]["start_mmss"],
                chunk[-1]["end_mmss"],
            )

    # If every API chunk failed, add full-episode heuristic candidates too.
    if failed_chunks == len(chunks):
        all_candidates.extend(
            _heuristic_role_candidates(
                segs,
                references,
                max_candidates=max_candidates,
            )
        )

    # Deduplicate by segment index, keep highest priority / richer result.
    best_by_seg = {}
    for c in all_candidates:
        idx = int(c.get("segment_index") or 0)
        if idx <= 0:
            continue

        previous = best_by_seg.get(idx)
        if previous is None:
            best_by_seg[idx] = c
            continue

        p_new = int(c.get("priority") or 0)
        p_old = int(previous.get("priority") or 0)
        if p_new > p_old:
            best_by_seg[idx] = c
        elif p_new == p_old:
            # Prefer richer semantic inference over fallback.
            new_richness = len(str(c.get("scene_role_hypothesis") or "")) + len(str(c.get("reason") or ""))
            old_richness = len(str(previous.get("scene_role_hypothesis") or "")) + len(str(previous.get("reason") or ""))
            if new_richness > old_richness:
                best_by_seg[idx] = c

    candidates = list(best_by_seg.values())
    candidates.sort(
        key=lambda c: (
            int(c.get("priority") or 0),
            -int(c.get("segment_index") or 0),
        ),
        reverse=True,
    )
    return candidates[:max_candidates]

def _extract_frames(
    *,
    video_bytes: bytes,
    original_name: str,
    segments: list[dict[str, Any]],
    frames_per_segment: int,
    prefix: str,
) -> list[dict[str, Any]]:
    suffix = Path(original_name).suffix.lower() or ".mp4"
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    video_path = temp_dir / f"input{suffix}"
    video_path.write_bytes(video_bytes)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    rels = [0.22, 0.72] if frames_per_segment == 2 else [0.12, 0.42, 0.72, 0.94]
    out = []

    for seg in segments:
        item = dict(seg)
        item["frames"] = []
        start, end = seg["start_sec"], max(seg["end_sec"], seg["start_sec"] + 0.3)
        duration = end - start

        for k, r in enumerate(rels, 1):
            ts = start + duration * r
            out_path = temp_dir / f"seg_{seg['index']:03d}_{k}.jpg"
            cmd = [
                ffmpeg_exe, "-y",
                "-ss", str(ts),
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "2",
                str(out_path),
            ]
            done = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if done.returncode == 0 and out_path.exists():
                item["frames"].append({
                    "frame_index": k,
                    "timestamp_sec": ts,
                    "timestamp_mmss": _mmss(ts),
                    "path": str(out_path),
                })
        out.append(item)

    return out


def build_dialogue_aware_router_segments(
    *,
    video_bytes: bytes,
    original_name: str,
    transcription: dict[str, Any],
    semantic_candidates: list[dict[str, Any]],
    max_timeline_segments: int = 4,
) -> list[dict[str, Any]]:
    raw = _raw_segments(transcription)
    by_index = {s["index"]: s for s in raw}

    selected_indices = []
    semantic_by_index = {}

    for c in semantic_candidates:
        idx = int(c.get("segment_index") or 0)
        if idx in by_index:
            selected_indices.append(idx)
            semantic_by_index[idx] = c

    # Add sparse timeline coverage so obvious visual identity shifts can still trigger.
    if raw:
        for i in range(max_timeline_segments):
            pos = round(i * (len(raw) - 1) / max(1, max_timeline_segments - 1))
            selected_indices.append(raw[pos]["index"])

    ordered = []
    seen = set()
    for idx in selected_indices:
        if idx in seen or idx not in by_index:
            continue
        seg = dict(by_index[idx])
        seg["semantic_candidate"] = semantic_by_index.get(idx)
        # Context: previous/current/next 2 lines.
        context_lines = []
        for n in range(max(1, idx - 2), min(len(raw), idx + 2) + 1):
            s = by_index.get(n)
            if s:
                context_lines.append(
                    f"SEG {s['index']} {s['start_mmss']}–{s['end_mmss']}: {s['text']}"
                )
        seg["dialogue_context"] = "\n".join(context_lines)
        ordered.append(seg)
        seen.add(idx)

    return _extract_frames(
        video_bytes=video_bytes,
        original_name=original_name,
        segments=ordered,
        frames_per_segment=2,
        prefix="short_drama_char_router_",
    )


def run_character_risk_router(
    *,
    api_key: str,
    episode_id: str,
    source_file_name: str,
    target_locale: str,
    references: list[dict[str, Any]],
    router_segments: list[dict[str, Any]],
    model: str = MODEL_DEFAULT,
) -> dict[str, Any]:
    if not references or not router_segments:
        return {
            "episode_risk_score": 0,
            "trigger_deep_review": False,
            "risk_windows": [],
            "summary": "缺少角色 reference 或可检查片段，本轮未触发角色深审。"
        }

    client = OpenAI(api_key=api_key.strip(), timeout=150.0, max_retries=1)

    instructions = """
你是 Character Risk Router。这里只做“是否值得深审”的风险判断，不做最终 Main Issue 定性。

本版最重要的能力是：
【根据整段剧情/台词关系判断“当前画面这个人应该是谁，或者肯定不应该是谁”】

注意：完全不要求台词出现人物名字。

强风险既可以来自：
1. 名字/第三人称排除；
2. 角色关系词；
3. “我/你/他/她”视角关系；
4. 连续多句对话共同推出的 scene role；
5. 当前这句话如果对某个角色说，会让剧情明显不成立。

例1：
女主面对一个男人说“Você também é amigo do Juliano, não é?”
→ 当前对象通常不是 Juliano。
如果画面却匹配 Juliano reference → 强风险。

例2（无名字）：
女主说“你也是他的朋友，对吗？他怎么没来？”
前文已经明确“他”是男主。
→ 当前对象应是男主的朋友，而不是男主本人。
若画面却匹配男主 reference → 强风险。

例3（无名字）：
“替我转告你老板，这件事没完。”
→ 当前对象剧情身份更像下属/助理，而不是老板本人。
若画面却稳定匹配老板 reference → 强风险。

例4（连续对话推断）：
前几句已经确定当前人在回答关于“新郎在哪里”的问题，
后一句又被称作“他的朋友”。
→ 当前 scene role 应是新郎朋友而非新郎本人。
如果脸却是新郎 reference → 强风险。

上述情况应标：
- dialogue_addressee_conflict / dialogue_addressee_impossibility
- relationship_logic_conflict 或 scene_role_conflict
如同时有清晰 reference 人脸冲突，risk_score 至少 3并触发 Deep Review。

再例如：
“Rafael在哪里？”时，当前被面对面询问的人如果明显就是 Rafael reference，
也应提高风险，除非上下文明确是修辞/误认剧情。

你需要同时看：
1. reference 人脸是谁；
2. 当前 2 张画面里主要对话对象/说话人更像谁；
3. 当前台词及前后上下文；
4. semantic_candidate 给出的 expected / excluded character 线索。

风险类型：
- face_identity_shift
- same_role_collision
- reference_mismatch
- dialogue_role_conflict
- dialogue_addressee_conflict
- active_speaker_conflict
- third_person_identity_conflict
- uncertain_identity

规则：
- Speaker A/B/C 绝不能当角色身份。
- 单纯“脸有点像”不触发深审。
- 但是“明确角色语义排除 + 人脸 reference 冲突”是强风险，即使单纯人脸检测本身不确定，也应该触发 Deep Review。
- dialogue_addressee_conflict / third_person_identity_conflict 若有较清晰人脸支持，risk_score >=3。
- 风险窗口宁可少而准。
- 所有解释中文。
"""

    content = [{
        "type": "input_text",
        "text": f"""【Episode】{episode_id}
【Source】{source_file_name}
【Locale】{target_locale}

先给角色 references，再给角色敏感台词片段和画面。
"""
    }]

    for i, ref in enumerate(references, 1):
        content.append({
            "type": "input_text",
            "text": f"REFERENCE {i}: {ref['name']} | {ref['role']}"
        })
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{_b64(ref['path'])}",
        })

    for seg in router_segments:
        semantic = seg.get("semantic_candidate")
        sem_text = "None"
        if semantic:
            sem_text = json.dumps(semantic, ensure_ascii=False)

        content.append({
            "type": "input_text",
            "text": (
                f"SEG {seg['index']} | {seg['start_mmss']}–{seg['end_mmss']}\n"
                f"ASR: {seg['text']}\n"
                f"Semantic Candidate（包含 expected/excluded role 与 scene role hypothesis）: {sem_text}\n"
                f"Dialogue Context:\n{seg.get('dialogue_context','')}"
            )
        })
        for frame in seg.get("frames", []):
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{_b64(frame['path'])}",
            })

    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "dialogue_aware_character_risk_router",
                    "schema": ROUTER_SCHEMA,
                    "strict": True,
                }
            },
        )

        raw = response.output_text
        if not raw:
            raise RuntimeError("Character Risk Router returned empty response.")

        result = json.loads(raw)

        retained = []
        for w in result.get("risk_windows", []):
            risk = int(w.get("risk_score") or 0)
            types = set(w.get("risk_types") or [])
            semantic_conflict = bool(
                types & {
                    "dialogue_addressee_conflict",
                    "third_person_identity_conflict",
                    "dialogue_role_conflict",
                    "relationship_logic_conflict",
                    "scene_role_conflict",
                    "dialogue_addressee_impossibility",
                }
            )
            if risk >= 3 or (semantic_conflict and risk >= 2):
                retained.append(w)

        result["risk_windows"] = retained
        result["trigger_deep_review"] = bool(retained) or int(result.get("episode_risk_score") or 0) >= 5
        return result

    except Exception:
        # Robust fallback: high-priority dialogue semantics must not silently disappear
        # just because the image router request timed out.
        fallback_windows = []
        for seg in router_segments:
            candidate = seg.get("semantic_candidate") or {}
            priority = int(candidate.get("priority") or 0)
            if priority < 4:
                continue

            fallback_windows.append({
                "segment_index": int(seg["index"]),
                "start_time": seg["start_mmss"],
                "end_time": seg["end_mmss"],
                "risk_score": 3,
                "risk_types": ["uncertain_identity"],
                "suspected_character": candidate.get("expected_character"),
                "conflicting_character": (
                    (candidate.get("excluded_characters") or [None])[0]
                ),
                "reason": (
                    "Character Router API 超时；该片段已被文本语义层判为高优先角色敏感候选，"
                    "为避免漏掉脸混，自动进入 Deep Character Review。"
                ),
            })

        return {
            "episode_risk_score": 5 if fallback_windows else 1,
            "trigger_deep_review": bool(fallback_windows),
            "risk_windows": fallback_windows[:4],
            "summary": (
                "Character Router API 超时，已启用语义候选降级路由。"
                if fallback_windows
                else "Character Router API 超时，且没有高优先语义候选可触发深审。"
            ),
        }


def build_deep_review_segments(
    *,
    video_bytes: bytes,
    original_name: str,
    transcription: dict[str, Any],
    risk_windows: list[dict[str, Any]],
    max_segments: int = 6,
) -> list[dict[str, Any]]:
    raw = _raw_segments(transcription)
    by_index = {s["index"]: s for s in raw}

    indices = []
    for w in risk_windows:
        idx = int(w.get("segment_index") or 0)
        indices.extend([idx - 1, idx, idx + 1])

    selected = []
    seen = set()
    for idx in indices:
        if idx <= 0 or idx in seen or idx not in by_index:
            continue
        seg = dict(by_index[idx])
        matching_risks = [
            w for w in risk_windows
            if int(w.get("segment_index") or 0) == idx
        ]
        seg["router_risk_context"] = matching_risks
        context_lines = []
        for n in range(max(1, idx - 2), min(len(raw), idx + 2) + 1):
            s = by_index.get(n)
            if s:
                context_lines.append(
                    f"SEG {s['index']} {s['start_mmss']}–{s['end_mmss']}: {s['text']}"
                )
        seg["dialogue_context"] = "\n".join(context_lines)
        selected.append(seg)
        seen.add(idx)
        if len(selected) >= max_segments:
            break

    return _extract_frames(
        video_bytes=video_bytes,
        original_name=original_name,
        segments=selected,
        frames_per_segment=4,
        prefix="short_drama_char_deep_",
    )
