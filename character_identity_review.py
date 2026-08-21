
from __future__ import annotations

import base64
import json
from typing import Any

from openai import OpenAI

from llm_review import MODEL_DEFAULT, MODEL_OUTPUT_SCHEMA, build_final_result


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _instructions(reference_count: int) -> str:
    return f"""
你是 AI Short Drama QA Assistant 的 Character Identity & Speaker Attribution 深审模块。

当前提供了 {reference_count} 个已人工确认身份的角色 reference。

你要判断的是：
【剧情中的“这个位置应该是谁”】与【画面实际使用了谁的脸】是否冲突。

--------------------------------
A. Face Swap / Face Mix
--------------------------------

脸混不要求台词出现名字。

只要通过连续台词和人物关系能稳定推出：
- 当前对话对象应该是角色/身份 A；
- 或当前对话对象明确不可能是角色/身份 B；
而画面人脸却稳定匹配 B 的 reference，
就可以形成 face_swap_or_face_mix 的证据链。

证据链：

Dialogue / Relationship Semantics
+
Scene Role Hypothesis
+
Reference Face Identity
=
Face Swap / Face Mix

可用语义证据包括：

1. 名字或称谓
2. 第三人称排除
   “你也是他的朋友吗？” → 当前对象通常不是“他”
3. 关系角色
   “你是新郎的朋友吗？” → 当前对象通常不是新郎本人
4. 视角冲突
   “我丈夫马上回来。” → 正常面对面对话下，当前对象通常不是丈夫本人
5. 转述关系
   “替我告诉你老板……” → 当前对象通常不是老板本人
6. 连续多句 scene-role inference
   即便任何单句都没有名字，但前后 3–5 句共同确定当前对象是“男二 / 医生 / 下属 / 新郎朋友”等
7. 台词对象不可能性
   如果把这句话理解成是对当前 reference 角色说，剧情关系会明显自相矛盾

当以下两条同时成立：
A. 剧情语义强烈要求当前场景位置不是角色 B；
B. 当前多帧人脸稳定更符合角色 B reference；
则：
- dimension=visual_logic
- subtype=face_swap_or_face_mix
- severity=P1
- evidence_complete=true

reason 必须写成：
“根据前后台词，当前对话对象剧情身份应为 X / 明确不应为 Y；但当前多帧面部稳定更符合 Y reference，因此角色身份与剧情关系冲突，判断为脸混/角色替换。”

--------------------------------
B. Dialogue Role Misattribution
--------------------------------

如果：
- 画面角色本身身份没有错；
- 但是配音/台词内容明显应该属于另一个角色；
则：
- dimension=dialogue_logic
- subtype=dialogue_role_misattribution
- severity=P1

--------------------------------
C. 区分两类错误
--------------------------------

剧情位置/身体应该是 A，但“脸”变成 B
→ face_swap_or_face_mix

画面角色本来就是 B，但“台词”错误安成 A 的
→ dialogue_role_misattribution

无法区分
→ Review Hint

--------------------------------
D. 误报控制
--------------------------------

- Speaker A/B/C 绝不能作为角色身份。
- 单帧像另一个人不够，至少结合多帧。
- 不能因为两个人长得相似就报。
- 必须同时有“角色剧情语义冲突”或连续身份证据。
- 若剧情可能存在假装不认识、失忆、易容、身份欺骗、故意试探，降低置信度并进 Review Hint。
- 所有解释使用中文。
"""

def _coerce(parsed: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues = []
    hints = list(parsed.get("review_hints", []))
    for item in parsed.get("issues", []):
        if item.get("evidence_complete") is True:
            issues.append(item)
        else:
            hints.append({
                "dimension": item.get("dimension"),
                "subtype": item.get("subtype"),
                "start_time": item.get("start_time", "00:00"),
                "end_time": item.get("end_time", "00:00"),
                "carrier": item.get("carrier", "character"),
                "suspected_issue": item.get("reason", "疑似角色身份/台词归属异常，但证据不足。"),
                "missing_evidence": ["未通过 Character Identity Evidence Gate"],
                "why_not_main_issue": "身份或说话人证据不足，自动下沉到复核提示区。",
                "confidence": item.get("confidence", 0.5),
            })
    return issues, hints


def _review_batch(
    *,
    client: OpenAI,
    episode_id: str,
    source_file_name: str,
    target_locale: str,
    references: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    content = [{
        "type": "input_text",
        "text": f"""【Episode】{episode_id}
【Source File】{source_file_name}
【Target Locale】{target_locale}

下面先给出人工确认身份的角色 Reference，随后给出待检查 ASR 片段及对应多帧画面。
"""
    }]

    for idx, ref in enumerate(references, 1):
        source_note = (
            f"资产图 {ref.get('source_file_name','')}"
            if ref.get("source") == "uploaded_asset"
            else f"视频参考时间 {ref.get('timestamp_text','')}"
        )
        content.append({
            "type": "input_text",
            "text": (
                f"REFERENCE {idx}: Name={ref['name']} | Role={ref['role']} | "
                f"Source={source_note}"
            )
        })
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{_b64(ref['path'])}",
        })

    content.append({
        "type": "input_text",
        "text": (
            "下面是待检查片段。请对每一段判断：画面角色最像哪个 reference、"
            "谁最可能在说话、该句语义是否与该角色身份一致。"
        )
    })

    for seg in segments:
        content.append({
            "type": "input_text",
            "text": (
                f"SEG {seg['index']} | {seg['start_mmss']}–{seg['end_mmss']} | "
                f"ASR SpeakerCluster={seg['speaker']} | ASR: {seg['text']}\n"
                f"Router Risk Context:\n{json.dumps(seg.get('router_risk_context', []), ensure_ascii=False)}\n"
                f"Dialogue Context:\n{seg.get('dialogue_context','')}"
            )
        })
        for frame in seg.get("frames", []):
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{_b64(frame['path'])}",
            })

    response = client.responses.create(
        model=model,
        instructions=_instructions(len(references)),
        input=[{"role": "user", "content": content}],
        text={
            "format": {
                "type": "json_schema",
                "name": "character_identity_qa",
                "schema": MODEL_OUTPUT_SCHEMA,
                "strict": True,
            }
        },
    )

    raw = response.output_text
    if not raw:
        raise RuntimeError("Character identity model returned empty response.")

    parsed = json.loads(raw)
    return _coerce(parsed)


def run_character_identity_review(
    *,
    api_key: str,
    episode_id: str,
    source_file_name: str,
    target_locale: str,
    references: list[dict[str, Any]],
    check_segments: list[dict[str, Any]],
    model: str = MODEL_DEFAULT,
    segments_per_batch: int = 2,
    progress_callback=None,
) -> dict[str, Any]:
    if not references:
        return build_final_result(
            episode_id=episode_id,
            source_file_name=source_file_name,
            target_locale=target_locale,
            issues=[],
            review_hints=[],
            scan_completed=False,
            status_note="未提供角色 reference，因此未运行角色身份模块。",
        )

    client = OpenAI(api_key=api_key.strip(), timeout=180.0, max_retries=1)
    all_issues, all_hints = [], []

    total_batches = max(1, (len(check_segments) + segments_per_batch - 1) // segments_per_batch)
    for batch_no, i in enumerate(range(0, len(check_segments), segments_per_batch), start=1):
        batch = check_segments[i:i+segments_per_batch]
        if progress_callback:
            progress_callback(batch_no, total_batches, batch)
        try:
            issues, hints = _review_batch(
                client=client,
                episode_id=episode_id,
                source_file_name=source_file_name,
                target_locale=target_locale,
                references=references,
                segments=batch,
                model=model,
            )
            all_issues.extend(issues)
            all_hints.extend(hints)
        except Exception:
            # One deep-review batch timing out should not kill the episode.
            for seg in batch:
                all_hints.append({
                    "dimension": "visual_logic",
                    "subtype": "character_deep_review_timeout",
                    "start_time": seg.get("start_mmss", "00:00"),
                    "end_time": seg.get("end_mmss", "00:00"),
                    "carrier": "character",
                    "suspected_issue": "该高风险角色时间窗的 Deep Character Review API 超时，建议人工复核。",
                    "missing_evidence": ["深度角色多模态请求超时"],
                    "why_not_main_issue": "未获得足够多模态证据，不能自动定性脸混或台词归属错误。",
                    "confidence": 0.5,
                })

    return build_final_result(
        episode_id=episode_id,
        source_file_name=source_file_name,
        target_locale=target_locale,
        issues=all_issues,
        review_hints=all_hints,
        scan_completed=False,
        status_note=(
            "角色身份模块使用人工标注 reference + 多帧 ASR 片段，"
            "检查脸混、角色替换和台词归属错误。"
        ),
    )
