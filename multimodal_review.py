
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from llm_review import MODEL_DEFAULT, MODEL_OUTPUT_SCHEMA, build_final_result
from visual_pipeline import encode_image_base64


def _build_visual_instructions(qa_rules: dict[str, Any]) -> str:
    return f"""
你是 AI Short Drama QA Assistant 的视觉/多模态审核模块。

市场：Brazil
语言：pt-BR

输入由两部分组成：
1. 全片 ASR 文本证据（含时间戳）
2. 按时间点抽样的关键帧图像

你必须遵守：
- precision-first：只把证据充分、可定位的问题放入 issues。
- 图像只是抽样关键帧，不代表所有帧；不确定时放入 review_hints。
- Speaker A/B/C 只是声纹聚类标签，不得单独作为角色身份证据。
- 如果某个问题依赖连续镜头、口型、完整对话链，但当前只有抽样帧，不得过度自信。
- 若关键帧中能清晰读到烧录字幕、中文/拼音画面文字、明显文化元素、明显连续性错误，可作为正式问题。
- 若能将某帧中可见字幕与附近 ASR 文本明显冲突，可按 dialogue_logic 或 proper noun issue 输出。
- 对视觉类问题，start_time/end_time 应优先使用该关键帧的时间点，必要时给出一个 2-4 秒的短区间。
- 所有解释使用中文。

这是规则引擎：
{json.dumps(qa_rules, ensure_ascii=False)}
"""


def run_multimodal_review(
    *,
    api_key: str,
    episode_id: str,
    source_file_name: str,
    target_locale: str,
    proper_nouns: str,
    asr_evidence: str,
    keyframes: list[dict[str, Any]],
    qa_rules: dict[str, Any],
    model: str = MODEL_DEFAULT,
) -> dict[str, Any]:
    client = OpenAI(api_key=api_key.strip())

    content = []
    content.append({
        "type": "input_text",
        "text": f"""【Episode】{episode_id}
【Source File】{source_file_name}
【Target Locale】{target_locale}
【Known Proper Nouns】
{proper_nouns.strip() or "未提供"}

【ASR Evidence】
{asr_evidence}

下面是按时间点抽样的关键帧。每张图前面会给出对应时间点。
"""
    })

    for frame in keyframes:
        content.append({
            "type": "input_text",
            "text": f"关键帧 {frame['index']}，时间点约 {frame['timestamp_mmss']}。请只基于这张图可见内容判断。"
        })
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{encode_image_base64(frame['path'])}"
        })

    response = client.responses.create(
        model=model,
        instructions=_build_visual_instructions(qa_rules),
        input=[{
            "role": "user",
            "content": content
        }],
        text={
            "format": {
                "type": "json_schema",
                "name": "short_drama_multimodal_qa",
                "schema": MODEL_OUTPUT_SCHEMA,
                "strict": True,
            }
        },
    )

    raw = response.output_text
    if not raw:
        raise RuntimeError("Multimodal model returned empty response.")

    parsed = json.loads(raw)
    issues = []
    review_hints = list(parsed.get("review_hints", []))
    for item in parsed.get("issues", []):
        if item.get("evidence_complete") is True:
            issues.append(item)
        else:
            review_hints.append({
                "dimension": item.get("dimension"),
                "subtype": item.get("subtype"),
                "start_time": item.get("start_time", "00:00"),
                "end_time": item.get("end_time", "00:00"),
                "carrier": item.get("carrier", "multimodal"),
                "suspected_issue": item.get("reason", "视觉模型发现潜在问题，但证据不足。"),
                "missing_evidence": ["未通过 Evidence Gate"],
                "why_not_main_issue": "evidence_complete=false，已自动下沉到复核提示区。",
                "confidence": item.get("confidence", 0.5),
            })

    return build_final_result(
        episode_id=episode_id,
        source_file_name=source_file_name,
        target_locale=target_locale,
        issues=issues,
        review_hints=review_hints,
        scan_completed=False,
        status_note="当前阶段基于关键帧抽样进行视觉扫描。",
    )
