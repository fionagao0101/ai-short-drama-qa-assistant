
from __future__ import annotations

import base64
import json
from typing import Any

from openai import OpenAI

from llm_review import MODEL_DEFAULT, MODEL_OUTPUT_SCHEMA, build_final_result


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _instructions() -> str:
    return """
你是 AI Short Drama QA Assistant 的剧情/动作重复判定模块。

输入是一组“语义相似候选对”：
- earlier 时间窗
- later 时间窗
- 两边各 3 张按时间顺序的画面帧
- 两边对应 ASR 台词

你的任务不是只看台词逐字是否一样，而是判断：
1. 是否出现相同或高度相似的剧情事件 / 动作序列
2. 是否像同一段内容被重复剪入
3. 是否存在明显重复动作，例如同一人物再次泼咖啡、再次递东西、再次拥抱、再次起身等
4. 台词可以不同，但如果剧情动作、镜头结构、角色关系高度重复，也可以构成 visual_logic / dialogue_logic 的重复问题

判定规则：
- 如果语义相似候选较高，且 earlier/later 的多张画面显示相同的“有辨识度动作链/事件链”
  （例如同一人物再次泼咖啡、再次摔杯、再次递出同一道具、同一冲突动作重新发生），
  并且没有回忆/闪回/recap 解释 → 必须 P1 MAIN ISSUE，evidence_complete=true。
- “台词不逐字相同”不能成为降级理由；剧情动作重复本来就可能伴随近义改写。
- 明显是同一动作/事件被重复播放，且无回忆/闪回/recap 解释 → P1 MAIN ISSUE。
- 画面相似但证据不足 → Review Hint。
- 只是同一场景、同一人物，不代表剧情重复。
- 不要因为服装/背景相同就直接报错。
- later 时间段是主要问题时间轴；在 context_evidence 写明 earlier 对照时间段。
- 所有解释使用中文。
"""


def _time_to_seconds(value: str) -> int:
    try:
        parts = str(value or "00:00").split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(float(parts[1]))
    except Exception:
        pass
    return 0


def _hint_matches_candidate(hint: dict[str, Any], candidate: dict[str, Any]) -> bool:
    later = candidate.get("later") or {}
    hs = _time_to_seconds(hint.get("start_time", "00:00"))
    he = _time_to_seconds(hint.get("end_time", hint.get("start_time", "00:00")))
    cs = _time_to_seconds(later.get("start_mmss", "00:00"))
    ce = _time_to_seconds(later.get("end_mmss", later.get("start_mmss", "00:00")))

    # Allow a few seconds of window mismatch because sliding windows are coarse.
    return not (he < cs - 4 or hs > ce + 4)


def _is_explicit_repeat_hint(hint: dict[str, Any]) -> bool:
    text = (
        str(hint.get("suspected_issue") or "") + " " +
        str(hint.get("why_not_main_issue") or "")
    ).lower()

    repeat_keywords = [
        "重复", "再次出现", "重复播放", "剧情重复", "动作重复",
        "同一动作", "同一事件", "重复剪入", "重复片段",
    ]
    uncertainty_keywords = [
        "仅同场景", "只是同一场景", "无法确认人物", "可能是回忆",
        "可能是闪回", "可能是 recap", "证据不足以确认动作重复",
    ]

    return (
        any(k in text for k in repeat_keywords)
        and not any(k in text for k in uncertainty_keywords)
    )


def _coerce(
    parsed: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues = []
    hints = []

    for item in parsed.get("issues", []):
        if item.get("evidence_complete") is True:
            issues.append(item)
        else:
            hints.append({
                "dimension": item.get("dimension"),
                "subtype": item.get("subtype"),
                "start_time": item.get("start_time", "00:00"),
                "end_time": item.get("end_time", "00:00"),
                "carrier": item.get("carrier", "visual"),
                "suspected_issue": item.get("reason", "疑似剧情重复，但证据不足。"),
                "missing_evidence": ["未通过 Evidence Gate"],
                "why_not_main_issue": "evidence_complete=false，自动下沉到复核提示区。",
                "confidence": item.get("confidence", 0.5),
            })

    for hint in parsed.get("review_hints", []):
        confidence = float(hint.get("confidence") or 0)

        matching = [
            c for c in candidates
            if _hint_matches_candidate(hint, c)
        ]
        best_similarity = max(
            [float(c.get("similarity") or 0) for c in matching],
            default=0.0,
        )

        # Restore Main Issue only when BOTH upstream semantic evidence and
        # the visual model's own description strongly support repetition.
        if (
            best_similarity >= 0.88
            and confidence >= 0.82
            and _is_explicit_repeat_hint(hint)
        ):
            match = max(
                matching,
                key=lambda c: float(c.get("similarity") or 0),
            )
            earlier = match.get("earlier") or {}
            later = match.get("later") or {}

            issues.append({
                "dimension": "visual_logic",
                "subtype": "repeated_plot_or_action_sequence",
                "severity": "P1",
                "start_time": later.get("start_mmss", hint.get("start_time", "00:00")),
                "end_time": later.get("end_mmss", hint.get("end_time", "00:00")),
                "carrier": "multimodal",
                "subtitle_text": None,
                "spoken_text": later.get("text"),
                "visual_evidence": hint.get("suspected_issue"),
                "context_evidence": (
                    f"Earlier 对照段：{earlier.get('start_mmss','00:00')}–"
                    f"{earlier.get('end_mmss','00:00')}；"
                    f"语义相似度={best_similarity:.3f}。"
                ),
                "evidence_complete": True,
                "reason": (
                    "语义候选高度相似，且视觉复核已明确识别同一剧情/动作序列重复出现；"
                    "无明确回忆/闪回解释，判断为剧情/动作重复。"
                ),
                "suggested_fix": "删除或替换 later 时间段的重复剧情/动作片段，并复核对应台词/字幕链。",
                "confidence": max(confidence, min(0.97, best_similarity)),
            })
        else:
            hints.append(hint)

    return issues, hints

def run_scene_repeat_review(
    *,
    api_key: str,
    episode_id: str,
    source_file_name: str,
    target_locale: str,
    candidates: list[dict[str, Any]],
    frames_by_window: dict[int, list[dict[str, Any]]],
    model: str = MODEL_DEFAULT,
    pairs_per_batch: int = 2,
) -> dict[str, Any]:
    if not candidates:
        return build_final_result(
            episode_id=episode_id,
            source_file_name=source_file_name,
            target_locale=target_locale,
            issues=[],
            review_hints=[],
            scan_completed=False,
            status_note="未产生语义重复候选。",
        )

    client = OpenAI(api_key=api_key.strip())
    all_issues, all_hints = [], []

    for offset in range(0, len(candidates), pairs_per_batch):
        batch = candidates[offset:offset+pairs_per_batch]
        content = [{
            "type": "input_text",
            "text": (
                f"【Episode】{episode_id}\n"
                f"【Source】{source_file_name}\n"
                "下面逐对比较 earlier/later 剧情窗口。"
            )
        }]

        for c in batch:
            earlier = c["earlier"]
            later = c["later"]
            content.append({
                "type": "input_text",
                "text": (
                    f"CANDIDATE {c['candidate_id']} | semantic_similarity={c['similarity']}\n"
                    f"EARLIER {earlier['start_mmss']}–{earlier['end_mmss']} | {earlier['text']}\n"
                    f"LATER   {later['start_mmss']}–{later['end_mmss']} | {later['text']}"
                )
            })

            content.append({"type": "input_text", "text": "EARLIER frames，按时间顺序："})
            for frame in frames_by_window.get(earlier["index"], []):
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{_b64(frame['path'])}",
                })

            content.append({"type": "input_text", "text": "LATER frames，按时间顺序："})
            for frame in frames_by_window.get(later["index"], []):
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{_b64(frame['path'])}",
                })

        response = client.responses.create(
            model=model,
            instructions=_instructions(),
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "scene_repeat_qa",
                    "schema": MODEL_OUTPUT_SCHEMA,
                    "strict": True,
                }
            },
        )

        raw = response.output_text
        if not raw:
            raise RuntimeError("Scene repeat model returned empty response.")

        parsed = json.loads(raw)
        issues, hints = _coerce(parsed, batch)
        all_issues.extend(issues)
        all_hints.extend(hints)

    return build_final_result(
        episode_id=episode_id,
        source_file_name=source_file_name,
        target_locale=target_locale,
        issues=all_issues,
        review_hints=all_hints,
        scan_completed=False,
        status_note="剧情重复检测使用语义相似候选 + 两段视觉动作序列进行复核。",
    )
