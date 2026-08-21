
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

APP_DIR = Path(__file__).resolve().parent
RULES_PATH = APP_DIR / "rules" / "qa_rules.json"

MODEL_DEFAULT = "gpt-5.6-terra"

DIMENSIONS = [
    "subtitle_chinese",
    "dubbing_chinese",
    "visual_chinese",
    "cultural_mismatch",
    "bgm_chinese",
    "subtitle_style",
    "proper_noun_inconsistency",
    "proper_noun_localization",
    "dialogue_logic",
    "visual_quality",
    "visual_logic",
    "voice_timbre",
]

CARRIERS = [
    "subtitle",
    "dubbing",
    "visual_text",
    "visual",
    "audio",
    "character",
    "multimodal",
]

MODEL_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string", "enum": DIMENSIONS},
                    "subtype": {"type": ["string", "null"]},
                    "severity": {"type": "string", "enum": ["P0", "P1", "P2"]},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "carrier": {"type": "string", "enum": CARRIERS},
                    "subtitle_text": {"type": ["string", "null"]},
                    "spoken_text": {"type": ["string", "null"]},
                    "visual_evidence": {"type": ["string", "null"]},
                    "context_evidence": {"type": ["string", "null"]},
                    "evidence_complete": {"type": "boolean"},
                    "reason": {"type": "string"},
                    "suggested_fix": {"type": ["string", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "dimension", "subtype", "severity", "start_time", "end_time", "carrier",
                    "subtitle_text", "spoken_text", "visual_evidence", "context_evidence",
                    "evidence_complete", "reason", "suggested_fix", "confidence"
                ],
                "additionalProperties": False,
            },
        },
        "review_hints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {"type": ["string", "null"]},
                    "subtype": {"type": ["string", "null"]},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "carrier": {"type": "string"},
                    "suspected_issue": {"type": "string"},
                    "missing_evidence": {"type": "array", "items": {"type": "string"}},
                    "why_not_main_issue": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "dimension", "subtype", "start_time", "end_time", "carrier",
                    "suspected_issue", "missing_evidence", "why_not_main_issue", "confidence"
                ],
                "additionalProperties": False,
            }
        }
    },
    "required": ["issues", "review_hints"],
    "additionalProperties": False,
}


def _load_rules() -> dict[str, Any]:
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_text_instructions() -> str:
    rules = _load_rules()
    return f"""
你是 AI Short Drama QA Assistant 的文本证据审核模块。

市场：Brazil
语言：pt-BR

原则：
1. precision-first：只把证据充分、可定位、确实有修改价值的问题放入 issues。
2. 证据不足但值得人工查看的放入 review_hints。
3. 若未提供烧录字幕，不得脑补字幕内容。
4. 若未提供视觉证据，不得把视觉类问题写进正式 issues。
5. Speaker A/B/C 只是声纹聚类标签，不等于角色身份。
6. 只允许使用以下 12 个一级维度：
{", ".join(DIMENSIONS)}

结构化业务规则如下：
{json.dumps(rules, ensure_ascii=False)}

所有解释使用中文。返回 JSON。
"""


def _build_text_input(
    episode_id: str,
    source_file_name: str,
    target_locale: str,
    manual_evidence: str,
    proper_nouns: str,
) -> str:
    return f"""
【Episode】{episode_id}
【Source File】{source_file_name}
【Target Locale】{target_locale}
【Known Proper Nouns】
{proper_nouns.strip() or "未提供"}

【Evidence】
{manual_evidence.strip()}
"""


def _coerce_hints_from_incomplete_issues(model_result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues = []
    review_hints = list(model_result.get("review_hints", []))
    for item in model_result.get("issues", []):
        if item.get("evidence_complete") is True:
            issues.append(item)
        else:
            review_hints.append({
                "dimension": item.get("dimension"),
                "subtype": item.get("subtype"),
                "start_time": item.get("start_time", "00:00"),
                "end_time": item.get("end_time", "00:00"),
                "carrier": item.get("carrier", "multimodal"),
                "suspected_issue": item.get("reason", "模型发现潜在问题，但证据不足。"),
                "missing_evidence": ["未通过 Evidence Gate"],
                "why_not_main_issue": "evidence_complete=false，已自动下沉到复核提示区。",
                "confidence": item.get("confidence", 0.5),
            })
    return issues, review_hints


def _dedupe_list(items: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for item in items:
        if kind == "issue":
            key = (
                item.get("dimension"), item.get("subtype"), item.get("severity"),
                item.get("start_time"), item.get("end_time"), item.get("reason")
            )
        else:
            key = (
                item.get("dimension"), item.get("subtype"),
                item.get("start_time"), item.get("end_time"), item.get("suspected_issue")
            )
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def build_final_result(
    *,
    episode_id: str,
    source_file_name: str,
    target_locale: str,
    issues: list[dict[str, Any]],
    review_hints: list[dict[str, Any]],
    scan_completed: bool,
    status_note: str | None = None,
) -> dict[str, Any]:
    issues = _dedupe_list(issues, "issue")
    review_hints = _dedupe_list(review_hints, "hint")

    for i, issue in enumerate(issues, 1):
        issue["issue_id"] = f"{episode_id}-I{i:03d}"
    for i, hint in enumerate(review_hints, 1):
        hint["hint_id"] = f"{episode_id}-H{i:03d}"

    p0 = sum(1 for i in issues if i.get("severity") == "P0")
    p1 = sum(1 for i in issues if i.get("severity") == "P1")
    p2 = sum(1 for i in issues if i.get("severity") == "P2")

    dimension_counts = {}
    for issue in issues:
        dim = issue["dimension"]
        dimension_counts[dim] = dimension_counts.get(dim, 0) + 1

    if p0 > 0:
        status = "REJECT"
        applied = "Confirmed P0 exists -> REJECT"
        explanation = "存在至少 1 条证据完整的 P0 阻断级问题，建议拒绝直接通过并优先修复。"
    elif issues or review_hints:
        status = "REVIEW"
        applied = "No P0; confirmed issue or unresolved hint exists -> REVIEW"
        explanation = "当前无 P0，但仍有正式问题或待人工确认项，建议人工复核后决定是否通过。"
    else:
        status = "REVIEW" if not scan_completed else "PASS"
        applied = "No issues found"
        explanation = "本轮未发现明确需修改问题。"
        if not scan_completed:
            explanation = "当前未发现明确问题，但尚未完成完整全片扫描，因此暂不自动判定 PASS。"

    if status_note:
        explanation = explanation + " " + status_note

    high_priority = [
        i.get("suggested_fix") or i.get("reason")
        for i in issues if i.get("severity") in ("P0", "P1")
    ][:5]

    return {
        "episode_id": episode_id,
        "source_file_name": source_file_name,
        "target_locale": target_locale,
        "episode_status": status,
        "scan_completed": scan_completed,
        "issues": issues,
        "review_hints": review_hints,
        "summary": {
            "p0_count": p0,
            "p1_count": p1,
            "p2_count": p2,
            "review_hint_count": len(review_hints),
            "main_issue_count": len(issues),
            "dimension_counts": dimension_counts,
            "high_priority_repair_notes": high_priority,
        },
        "status_logic": {
            "rule_applied": applied,
            "explanation": explanation,
        }
    }


def run_text_review(
    *,
    api_key: str,
    episode_id: str,
    source_file_name: str,
    target_locale: str,
    manual_evidence: str,
    proper_nouns: str = "",
    model: str = MODEL_DEFAULT,
) -> dict[str, Any]:
    client = OpenAI(api_key=api_key.strip())

    response = client.responses.create(
        model=model,
        instructions=_build_text_instructions(),
        input=_build_text_input(
            episode_id=episode_id,
            source_file_name=source_file_name,
            target_locale=target_locale,
            manual_evidence=manual_evidence,
            proper_nouns=proper_nouns,
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "short_drama_text_qa",
                "schema": MODEL_OUTPUT_SCHEMA,
                "strict": True,
            }
        },
    )

    raw = response.output_text
    if not raw:
        raise RuntimeError("Model returned empty response.")
    parsed = json.loads(raw)
    issues, review_hints = _coerce_hints_from_incomplete_issues(parsed)
    return build_final_result(
        episode_id=episode_id,
        source_file_name=source_file_name,
        target_locale=target_locale,
        issues=issues,
        review_hints=review_hints,
        scan_completed=False,
        status_note="当前阶段仅基于音频/文本证据。",
    )


def merge_results(
    *,
    audio_result: dict[str, Any],
    visual_result: dict[str, Any],
    episode_id: str,
    source_file_name: str,
    target_locale: str,
) -> dict[str, Any]:
    issues = list(audio_result.get("issues", [])) + list(visual_result.get("issues", []))
    hints = list(audio_result.get("review_hints", [])) + list(visual_result.get("review_hints", []))
    return build_final_result(
        episode_id=episode_id,
        source_file_name=source_file_name,
        target_locale=target_locale,
        issues=issues,
        review_hints=hints,
        scan_completed=False,
        status_note="当前结果来自音频全片转写 + 关键帧抽样视觉扫描；仍属于 MVP 阶段的非全量视觉扫描。",
    )


def merge_many_results(
    *,
    results: list[dict[str, Any]],
    episode_id: str,
    source_file_name: str,
    target_locale: str,
    status_note: str,
) -> dict[str, Any]:
    issues = []
    hints = []
    for result in results:
        issues.extend(result.get("issues", []))
        hints.extend(result.get("review_hints", []))

    return build_final_result(
        episode_id=episode_id,
        source_file_name=source_file_name,
        target_locale=target_locale,
        issues=issues,
        review_hints=hints,
        scan_completed=False,
        status_note=status_note,
    )
