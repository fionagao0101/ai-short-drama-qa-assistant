
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from llm_review import MODEL_DEFAULT, MODEL_OUTPUT_SCHEMA, build_final_result


def _encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_instructions(qa_rules: dict[str, Any]) -> str:
    return f"""
你是 AI Short Drama QA Assistant 的 Step 5D：Temporal Alignment 审核模块。

市场：Brazil
语言：pt-BR

输入内容包含：
1. 按 ASR 片段时间对齐的台词
2. 每个片段对应的一张对齐帧（通常截自该片段中部）
3. 若存在，局部重复候选窗口

你的职责：
A. 对每个时间对齐片段，重点检查：
   - 是否出现完整可理解配音，但烧录字幕缺失 / 明显不完整
   - 烧录字幕与配音是否明显不匹配
   - 可见字幕/画面文字中是否有中文/拼音
   - 可见字幕或画面文字中的专有名词是否与 ASR / Known Proper Nouns 明显冲突
B. 对局部重复候选，判断 later 窗口是否与 earlier 窗口构成“异常重复的台词/剧情片段”。
C. precision-first：证据不充分就进 review_hints，不要硬进 issues。
D. Speaker A/B/C 只是声纹聚类标签，不等于角色身份。
E. 若只凭一张帧图仍不能确定完整字幕情况，要下沉到 review_hints。
F. 所有解释使用中文。

重要说明：
- 此阶段比 Step 5C 更偏重时间对齐，所以 main issue 的时间轴应尽量使用对应片段的时间。
- 对重复问题，如果 later 段与 earlier 段语义高度相同且看起来不像正常回忆/ recap，可标为 dialogue_logic 或 visual_logic 的 P1 / P2。
- 你只能使用下列 12 个一级维度：
subtitle_chinese, dubbing_chinese, visual_chinese, cultural_mismatch, bgm_chinese, subtitle_style, proper_noun_inconsistency, proper_noun_localization, dialogue_logic, visual_quality, visual_logic, voice_timbre

业务规则：
{json.dumps(qa_rules, ensure_ascii=False)}
"""


def _coerce(model_result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues = []
    hints = list(model_result.get("review_hints", []))
    for item in model_result.get("issues", []):
        if item.get("evidence_complete") is True:
            issues.append(item)
        else:
            hints.append({
                "dimension": item.get("dimension"),
                "subtype": item.get("subtype"),
                "start_time": item.get("start_time", "00:00"),
                "end_time": item.get("end_time", "00:00"),
                "carrier": item.get("carrier", "multimodal"),
                "suspected_issue": item.get("reason", "模型发现潜在时间对齐问题，但证据不足。"),
                "missing_evidence": ["未通过 Evidence Gate"],
                "why_not_main_issue": "evidence_complete=false，自动下沉到复核提示区。",
                "confidence": item.get("confidence", 0.5),
            })
    return issues, hints


def _run_one_batch(
    *,
    client: OpenAI,
    qa_rules: dict[str, Any],
    episode_id: str,
    source_file_name: str,
    target_locale: str,
    proper_nouns: str,
    segments_batch: list[dict[str, Any]],
    duplicate_candidates: list[dict[str, Any]] | None = None,
    model: str = MODEL_DEFAULT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    content = []
    content.append({
        "type": "input_text",
        "text": f"""【Episode】{episode_id}
【Source File】{source_file_name}
【Target Locale】{target_locale}
【Known Proper Nouns】
{proper_nouns.strip() or "未提供"}

下面是时间对齐片段。每个片段都包含：
- 该段 ASR 文本
- 该段的时间
- 一张与该段对齐的抽帧

请逐段重点检查：漏字幕、字幕与配音错配、字幕/画面中文、专有名词冲突。
"""
    })

    for seg in segments_batch:
        content.append({
            "type": "input_text",
            "text": (
                f"片段 {seg['index']} | {seg['start_mmss']}–{seg['end_mmss']} | "
                f"Speaker {seg['speaker']} | ASR: {seg['text']}"
            )
        })
        if seg.get("frame_path"):
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{_encode_image_base64(seg['frame_path'])}"
            })

    if duplicate_candidates:
        dup_text_lines = ["【局部重复候选】"]
        for c in duplicate_candidates:
            dup_text_lines.append(
                f"- {c['earlier_start_mmss']}–{c['earlier_end_mmss']} 与 "
                f"{c['later_start_mmss']}–{c['later_end_mmss']} 高相似（{c['similarity']}）\n"
                f"  Earlier: {c['earlier_text']}\n"
                f"  Later:   {c['later_text']}"
            )
        content.append({
            "type": "input_text",
            "text": "\n".join(dup_text_lines)
        })

    response = client.responses.create(
        model=model,
        instructions=_build_instructions(qa_rules),
        input=[{"role": "user", "content": content}],
        text={
            "format": {
                "type": "json_schema",
                "name": "temporal_alignment_qa",
                "schema": MODEL_OUTPUT_SCHEMA,
                "strict": True,
            }
        },
    )

    raw = response.output_text
    if not raw:
        raise RuntimeError("Temporal multimodal model returned empty response.")
    parsed = json.loads(raw)
    return _coerce(parsed)


def run_temporal_alignment_review(
    *,
    api_key: str,
    qa_rules: dict[str, Any],
    episode_id: str,
    source_file_name: str,
    target_locale: str,
    proper_nouns: str,
    aligned_segments: list[dict[str, Any]],
    duplicate_candidates: list[dict[str, Any]],
    model: str = MODEL_DEFAULT,
    batch_size: int = 8,
) -> dict[str, Any]:
    client = OpenAI(api_key=api_key.strip())
    all_issues = []
    all_hints = []

    # Put duplicate candidates only in the first batch to control prompt size.
    for i in range(0, len(aligned_segments), batch_size):
        batch = aligned_segments[i:i + batch_size]
        dups = duplicate_candidates if i == 0 else []
        issues, hints = _run_one_batch(
            client=client,
            qa_rules=qa_rules,
            episode_id=episode_id,
            source_file_name=source_file_name,
            target_locale=target_locale,
            proper_nouns=proper_nouns,
            segments_batch=batch,
            duplicate_candidates=dups,
            model=model,
        )
        all_issues.extend(issues)
        all_hints.extend(hints)

    return build_final_result(
        episode_id=episode_id,
        source_file_name=source_file_name,
        target_locale=target_locale,
        issues=all_issues,
        review_hints=all_hints,
        scan_completed=False,
        status_note="当前阶段基于 ASR 对齐片段进行时间对齐视觉扫描。",
    )
