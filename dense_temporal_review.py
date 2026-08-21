
from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from openai import OpenAI

from llm_review import MODEL_DEFAULT, MODEL_OUTPUT_SCHEMA, build_final_result


TIMING_SUBTYPES = {
    "subtitle_timing_misalignment",
    "subtitle_delay",
    "subtitle_advance",
    "subtitle_timing_error",
}

SEMANTIC_GAP_SUBTYPES = {
    "partial_missing_subtitle_semantic_unit",
    "meaningful_subtitle_omission",
    "partial_missing_subtitle",
}

SEMANTIC_MISMATCH_SUBTYPES = {
    "subtitle_dubbing_semantic_mismatch",
    "subtitle_dubbing_mismatch",
    "subtitle_audio_mismatch",
    "subtitle_asr_mismatch",
    "dialogue_semantic_mismatch",
}


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _instructions(qa_rules: dict[str, Any]) -> str:
    return f"""
你是 AI Short Drama QA Assistant 的 Dense Subtitle Timing 审核模块。

目标市场：Brazil
目标语言：pt-BR

【本版核心：把“正常轻微延迟”和“真实语义漏字幕”分开】

ASR start/end 有切分误差；抽帧是离散采样。因此：
- 不允许仅凭“第一张帧没字幕、后面有字幕”就报错。
- 不允许仅按 0.8s / 1.2s 机械判断。

你必须按以下优先级审核，顺序不能反：

【优先级 1：字幕–配音语义是否一致】
先完全忽略时序，直接比较：
- ASR / 配音实际说了什么
- 当前片段多张帧里烧录字幕实际写了什么

如果字幕与配音是不同语义、不同句子、错误承接上一句/下一句：
→ 这是语义错配，不是 Timing。
→ subtype 必须使用 `subtitle_dubbing_semantic_mismatch`
→ severity=P1
→ 多帧持续支持时 evidence_complete=true
→ Timing Tolerance 绝对不能过滤这类问题。

尤其注意：
- 字幕和 ASR 只有部分词重合，但核心谓语/对象/问答含义不同，也属于语义错配；
- 字幕显示上一句或下一句，而当前 ASR 已经进入另一句，也优先判断为语义错配/字幕脱轨；
- 连续两张以上帧都显示与当前 ASR 不对应的完整语义时，必须主动出表。

【优先级 2：语义一致后，再判断 Timing / 是否缺失】

A. 【正常轻微同步差 — 过滤】
如果：
1. 配音和字幕语义一致；
2. 前面一两张帧暂时没字幕；
3. 后续帧出现的字幕最终覆盖了这整句 ASR 的核心语义，包括开头内容；
4. 没有任何有意义的词组/分句永久缺失；
那么即使字幕晚约 0.5–1.0 秒，也视为正常，不出 Main、不进 Hint。

例：
ASR：Você também é amigo do Juliano, não é?
早期帧：无字幕
稍后帧：出现完整“Você também é amigo do Juliano, não é?”
→ 这是轻微显示偏差，不报。

B. 【真实“前半句漏字幕” — Semantic Gap Exception】
如果：
1. 配音开头已经播放了一个有独立意义的词组/分句；
2. 该开头语义在任何后续字幕帧中都没有被补回；
3. 后续字幕只覆盖后半句，或者从中间内容开始；
那么这是“语义覆盖缺口”，即使时间差小于 1.2 秒，也必须保留。

满足多帧证据时：
- dimension=dialogue_logic
- subtype=partial_missing_subtitle_semantic_unit
- severity=P1
- evidence_complete=true

reason 必须写清“究竟哪一段配音语义没有任何对应字幕”，不能只写“约 1 秒没字幕”。

C. 【纯 Timing Main Issue 门槛】
只有没有语义缺失、只是整体时序漂移时，才应用以下阈值：
- ASR 边界默认允许 ±0.45 秒误差
- 约 0.8 秒内出现完整同义字幕 → 正常
- 0.8–1.2 秒但语义最终完整 → 最多 Hint
- >=1.2 秒且连续多帧支持明显脱轨 → 才可 Main
- 字幕残留/提前导致连续显示错误句 → 可 Main

D. 【字幕–配音语义错配】
如果多张连续帧里的字幕和 ASR 是不同语义，按原规则正常报；
这不属于轻微 Timing Tolerance。

E. 【重要取证规则】
请逐帧判断：
- ASR 开头说了什么
- frame1 / frame2 / frame3 / frame4 字幕分别覆盖了什么
- 后续字幕有没有“补回”开头内容
只有“开头内容始终没有被任何字幕覆盖”才叫真实部分漏字幕。

所有解释使用中文。
Speaker A/B/C 只是声纹聚类标签，不等于角色身份。

一级维度只能使用：
subtitle_chinese, dubbing_chinese, visual_chinese, cultural_mismatch, bgm_chinese,
subtitle_style, proper_noun_inconsistency, proper_noun_localization,
dialogue_logic, visual_quality, visual_logic, voice_timbre

业务规则：
{json.dumps(qa_rules, ensure_ascii=False)}
"""

def _to_hint(item: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "dimension": item.get("dimension"),
        "subtype": item.get("subtype"),
        "start_time": item.get("start_time", "00:00"),
        "end_time": item.get("end_time", "00:00"),
        "carrier": item.get("carrier", "subtitle"),
        "suspected_issue": item.get("reason", "疑似字幕时序异常。"),
        "missing_evidence": [reason],
        "why_not_main_issue": reason,
        "confidence": item.get("confidence", 0.5),
    }


def _coerce(parsed: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues = []
    hints = list(parsed.get("review_hints", []))

    for item in parsed.get("issues", []):
        subtype = (item.get("subtype") or "").strip().lower()
        confidence = float(item.get("confidence") or 0)

        if item.get("evidence_complete") is not True:
            hints.append(_to_hint(item, "未通过 Evidence Gate，自动下沉。"))
            continue

        # Semantic subtitle/dubbing mismatch is independent from Timing Tolerance.
        # Never let a timing false-positive suppression rule remove a true meaning mismatch.
        reason_lower = str(item.get("reason") or "").lower()
        mismatch_language = any(k in reason_lower for k in [
            "语义不一致", "完全不匹配", "字幕与配音不匹配", "字幕和配音不匹配",
            "不同语义", "错配", "显示上一句", "显示下一句", "字幕脱轨"
        ])

        if subtype in SEMANTIC_MISMATCH_SUBTYPES or mismatch_language:
            if confidence >= 0.88:
                item["subtype"] = "subtitle_dubbing_semantic_mismatch"
                item["severity"] = "P1"
                issues.append(item)
            elif confidence >= 0.78:
                hints.append(_to_hint(
                    item,
                    "字幕与配音疑似存在语义错配，但置信度不足 88%，需人工复核。"
                ))
            continue

        # A real semantic subtitle omission is NOT treated like a minor timing offset.
        if subtype in SEMANTIC_GAP_SUBTYPES:
            if confidence >= 0.90:
                issues.append(item)
            elif confidence >= 0.82:
                hints.append(_to_hint(
                    item,
                    "疑似存在有意义分句未被字幕覆盖，但置信度不足 90%，需人工确认。"
                ))
            continue

        # Pure timing-only claims remain precision-first.
        if subtype in TIMING_SUBTYPES:
            if confidence < 0.95:
                continue
            if confidence < 0.98:
                hints.append(_to_hint(
                    item,
                    "纯字幕时序问题属于高误报类别；当前证据不足以进入主表。"
                ))
                continue

        issues.append(item)

    return issues, hints

def _review_batch(
    *,
    client: OpenAI,
    qa_rules: dict[str, Any],
    episode_id: str,
    source_file_name: str,
    target_locale: str,
    proper_nouns: str,
    batch: list[dict[str, Any]],
    model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    content = [{
        "type": "input_text",
        "text": f"""【Episode】{episode_id}
【Source File】{source_file_name}
【Target Locale】{target_locale}
【Known Proper Nouns】
{proper_nouns.strip() or "未提供"}

每个 segment 都包含 ASR 精确起止时间和 3–4 张帧的精确采样时间。
请注意：小数秒时间用于判断，不要只按整数秒理解时序。
"""
    }]

    for seg in batch:
        content.append({
            "type": "input_text",
            "text": (
                f"SEG {seg['index']} | precise={seg.get('start_precise', seg['start_mmss'])}"
                f"–{seg.get('end_precise', seg['end_mmss'])} | "
                f"duration={seg['duration_sec']:.2f}s | "
                f"SpeakerCluster={seg['speaker']} | ASR: {seg['text']}"
            )
        })

        for frame in seg.get("frames", []):
            content.append({
                "type": "input_text",
                "text": (
                    f"SEG {seg['index']} frame {frame['frame_index']} "
                    f"sampled at {frame.get('timestamp_precise', frame['timestamp_mmss'])}. "
                    "只根据该采样点可见字幕判断；单帧缺字幕不是完整时序证据。"
                )
            })
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{_b64(frame['path'])}",
            })

    response = client.responses.create(
        model=model,
        instructions=_instructions(qa_rules),
        input=[{"role": "user", "content": content}],
        text={
            "format": {
                "type": "json_schema",
                "name": "dense_temporal_qa_precision_tuned",
                "schema": MODEL_OUTPUT_SCHEMA,
                "strict": True,
            }
        },
    )

    raw = response.output_text
    if not raw:
        raise RuntimeError("Dense temporal model returned empty response.")

    return _coerce(json.loads(raw))


def run_dense_temporal_review(
    *,
    api_key: str,
    qa_rules: dict[str, Any],
    episode_id: str,
    source_file_name: str,
    target_locale: str,
    proper_nouns: str,
    dense_segments: list[dict[str, Any]],
    model: str = MODEL_DEFAULT,
    batch_size: int = 4,
    max_workers: int = 2,
    progress_callback=None,
) -> dict[str, Any]:
    """
    Performance-tuned version:
    - preserves full dense segment coverage
    - processes up to 2 multimodal batches concurrently
    - slightly increases batch size from 3 -> 4
    - reports completed-batch progress to the UI
    """
    batches = [
        dense_segments[i:i + batch_size]
        for i in range(0, len(dense_segments), batch_size)
    ]

    all_issues, all_hints = [], []
    total_batches = len(batches)

    def _task(batch_no: int, batch: list[dict[str, Any]]):
        # Create a client inside the worker to avoid sharing request state.
        worker_client = OpenAI(api_key=api_key.strip())
        issues, hints = _review_batch(
            client=worker_client,
            qa_rules=qa_rules,
            episode_id=episode_id,
            source_file_name=source_file_name,
            target_locale=target_locale,
            proper_nouns=proper_nouns,
            batch=batch,
            model=model,
        )
        return batch_no, batch, issues, hints

    if total_batches:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_task, batch_no, batch)
                for batch_no, batch in enumerate(batches, start=1)
            ]

            completed = 0
            for future in as_completed(futures):
                batch_no, batch, issues, hints = future.result()
                all_issues.extend(issues)
                all_hints.extend(hints)
                completed += 1

                if progress_callback:
                    progress_callback(
                        completed,
                        total_batches,
                        batch_no,
                        batch,
                    )

    return build_final_result(
        episode_id=episode_id,
        source_file_name=source_file_name,
        target_locale=target_locale,
        issues=all_issues,
        review_hints=all_hints,
        scan_completed=False,
        status_note=(
            "字幕时序层已启用 Timing Tolerance；"
            "Dense Review 保留完整片段覆盖，并使用双路并发批处理加速。"
        ),
    )

