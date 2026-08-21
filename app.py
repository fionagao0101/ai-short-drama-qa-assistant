
import json
import re
from pathlib import Path

import streamlit as st

from llm_review import MODEL_DEFAULT, merge_many_results, run_text_review
from media_pipeline import TRANSCRIBE_MODEL, process_video_to_evidence
from dense_temporal_pipeline import build_dense_segments, extract_dense_frames
from dense_temporal_review import run_dense_temporal_review
from verbatim_audio import (
    VERBATIM_MODEL,
    run_verbatim_word_transcription,
    detect_verbatim_repetitions,
    format_repetition_evidence,
)
from scene_repeat_pipeline import (
    EMBED_MODEL,
    build_sliding_windows,
    rank_semantic_repeat_candidates,
    extract_repeat_window_frames,
)
from scene_repeat_review import run_scene_repeat_review
from character_identity_pipeline import (
    save_uploaded_asset_references,
    extract_video_reference_frames,
    select_character_check_segments,
    select_deep_review_segments_from_risk,
)
from character_risk_router import (
    select_role_sensitive_candidates,
    build_dialogue_aware_router_segments,
    run_character_risk_router,
    build_deep_review_segments,
)
from character_identity_review import run_character_identity_review

APP_DIR = Path(__file__).resolve().parent
RULES_PATH = APP_DIR / "rules" / "qa_rules.json"

with open(RULES_PATH, "r", encoding="utf-8") as f:
    QA_RULES = json.load(f)

st.set_page_config(
    page_title="AI Short Drama QA Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetricValue"] {font-size: 1.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

DIMENSION_ZH = {
    "subtitle_chinese": "字幕含中文/拼音",
    "dubbing_chinese": "配音含中文",
    "visual_chinese": "画面含中文/拼音",
    "cultural_mismatch": "非目标市场文化元素",
    "bgm_chinese": "BGM 含中文歌词",
    "subtitle_style": "字幕样式/位置不一致",
    "proper_noun_inconsistency": "专有名词拼写/读音不一致",
    "proper_noun_localization": "专有名词未本地化",
    "dialogue_logic": "台词逻辑异常",
    "visual_quality": "画质异常",
    "visual_logic": "画面逻辑异常",
    "voice_timbre": "音色异常",
}
STATUS_EMOJI = {"PASS": "✅", "REVIEW": "⚠️", "REJECT": "⛔"}


def normalize_episode_id(raw: str) -> str:
    raw = (raw or "").strip().upper()
    if not raw:
        return "EP001"
    if re.fullmatch(r"EP\d{3,}", raw):
        return raw
    digits = re.sub(r"\D", "", raw)
    if digits:
        return f"EP{int(digits):03d}"
    return "EP001"


def issue_card(issue):
    dim = DIMENSION_ZH.get(issue["dimension"], issue["dimension"])
    st.markdown(f"### {issue['severity']} · {issue['start_time']}–{issue['end_time']} · {dim}")
    st.write(issue["reason"])
    c1, c2 = st.columns(2)
    with c1:
        if issue.get("subtitle_text"):
            st.caption("烧录字幕")
            st.code(issue["subtitle_text"], language=None)
        if issue.get("spoken_text"):
            st.caption("配音/ASR")
            st.code(issue["spoken_text"], language=None)
        if issue.get("visual_evidence"):
            st.caption("画面证据")
            st.code(issue["visual_evidence"], language=None)
    with c2:
        if issue.get("context_evidence"):
            st.caption("上下文证据")
            st.write(issue["context_evidence"])
        if issue.get("suggested_fix"):
            st.caption("建议修改")
            st.write(issue["suggested_fix"])
    st.caption(f"Confidence: {issue.get('confidence', 0):.0%} · Carrier: {issue.get('carrier','-')}")
    st.divider()


def hint_card(hint):
    dim = DIMENSION_ZH.get(hint.get("dimension"), hint.get("dimension") or "待分类")
    st.markdown(f"#### {hint['start_time']}–{hint['end_time']} · {dim}")
    st.write(hint["suspected_issue"])
    if hint.get("missing_evidence"):
        st.caption("缺失证据：" + "；".join(hint["missing_evidence"]))
    st.caption("未进主表原因：" + hint["why_not_main_issue"])
    st.divider()


with st.sidebar:
    st.header("Review Configuration")
    st.selectbox("Target Market", ["Brazil"], index=0)
    target_locale = st.selectbox("Target Locale", ["pt-BR"], index=0)
    st.divider()

    st.subheader("OpenAI API")
    try:
        deployed_api_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        deployed_api_key = ""

    if deployed_api_key:
        api_key = deployed_api_key
        st.success("Using OPENAI_API_KEY from Streamlit secrets.")
        st.caption("Secret is not stored in the repository.")
    else:
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="sk-...",
            help="仅用于当前 Streamlit 会话，不写入项目文件。",
        )
        st.caption("不要把 API Key 提交到 GitHub。")
    st.caption(f"QA model: {MODEL_DEFAULT}")
    st.caption(f"ASR model: {TRANSCRIBE_MODEL}")
    st.caption(f"Verbatim model: {VERBATIM_MODEL}")
    st.caption(f"Semantic model: {EMBED_MODEL}")
    st.divider()
    st.caption("v1.0.3 · Step 5I.2 Final · Regression Locked")

st.title("🎬 AI Short Drama QA Assistant")
st.markdown(
    "**Step 5I.2 Final · Frozen Pilot Baseline** — pt-BR 海外短剧多模态本地化质检："
    "字幕/配音语义与时序、逐词复读、剧情/动作重复、专有名词、角色身份与脸混，"
    "并通过 Evidence Gate 输出 Main Issues 与 Review Hints。"
)

tab_review, tab_asr, tab_dense, tab_repeat, tab_audio, tab_character, tab_rules = st.tabs(
    ["Run Review", "ASR Transcript", "Dense Segments", "Scene Repeat", "Verbatim Audio", "Character Identity", "QA Coverage"]
)

with tab_review:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("1. Upload Episode")
        episode_id_raw = st.text_input("Episode ID", value="EP005")
        video_file = st.file_uploader(
            "Upload MP4",
            type=["mp4", "mov", "m4v"],
            help="执行完整 Step 5I.2 Final：ASR、密集字幕对齐、逐词复读、剧情重复、角色风险路由与深审。",
        )

        if video_file:
            st.video(video_file)
            size_mb = len(video_file.getvalue()) / 1024 / 1024
            st.caption(f"Uploaded video: {video_file.name} · {size_mb:.1f} MB")

        st.subheader("2. Context")
        proper_nouns = st.text_area(
            "Known Proper Nouns",
            value="",
            placeholder="Marina\nLucas\nGabriel\nGrupo Aurora",
        )

        st.subheader("3. Character References")
        st.caption("优先用角色资产图；也可以直接指定视频中哪个时间点是谁。两种方式可同时使用。")

        asset_files = st.file_uploader(
            "Character Asset Images (optional)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help="从飞书资产文档里把人物资产图保存下来后上传。"
        )

        uploaded_asset_specs = []
        if asset_files:
            st.caption("给每张资产图补充人物名和角色：")
            for idx, af in enumerate(asset_files):
                c1, c2, c3 = st.columns([0.25, 0.35, 0.40])
                with c1:
                    st.image(af, width=110)
                with c2:
                    char_name = st.text_input(
                        f"Name #{idx+1}",
                        value=Path(af.name).stem,
                        key=f"char_name_{idx}",
                    )
                with c3:
                    char_role = st.text_input(
                        f"Role #{idx+1}",
                        placeholder="男主 / 男二 / 女主 / 医生...",
                        key=f"char_role_{idx}",
                    )
                uploaded_asset_specs.append({
                    "name": char_name,
                    "role": char_role,
                    "file_name": af.name,
                    "bytes": af.getvalue(),
                })

        video_reference_lines = st.text_area(
            "Video Reference Timestamps (optional fallback)",
            value="",
            height=100,
            placeholder="Lucas|男主|00:18\nRafael|男二|00:33",
            help="如果不想另外贴图，只要人工告诉系统哪个时间点清楚出现的是谁即可。",
        )

        run_clicked = st.button(
            "Run STEP 5I.2 FINAL QA",
            type="primary",
            use_container_width=True,
        )

    with right:
        st.subheader("Step 5H Scope")
        st.metric("Dense Frames / ASR", "3–4")
        st.metric("Verbatim Word Timestamps", "YES")
        st.metric("Semantic + Visual Repeat", "YES")
        st.metric("Character Identity", "YES")
        st.info(
            "角色模块现在默认自动路由：每集先轻量预检；"
            "只有出现高风险窗口时才进入 Deep Character Review。"
        )
        st.warning(
            "这样无需预先知道哪一集会脸混，同时避免每集都跑 10 分钟重型角色审核。"
        )

    if run_clicked:
        episode_id = normalize_episode_id(episode_id_raw)

        if not api_key:
            st.error("请先在左侧粘贴 API Key。")
        elif video_file is None:
            st.error("请先上传视频。")
        else:
            try:
                with st.status("Running Step 5I.2 Final…", expanded=True) as status:
                    video_bytes = video_file.getvalue()

                    st.write("1/11 Full audio ASR")
                    media_result = process_video_to_evidence(
                        api_key=api_key,
                        video_bytes=video_bytes,
                        file_name=video_file.name,
                    )

                    st.write("2/11 Verbatim word-level transcription")
                    verbatim_result = run_verbatim_word_transcription(
                        api_key=api_key,
                        video_bytes=video_bytes,
                        file_name=video_file.name,
                    )
                    verbatim_candidates = detect_verbatim_repetitions(verbatim_result)

                    st.write("3/11 Dense ASR-aligned multi-frame extraction")
                    dense_segments = build_dense_segments(
                        media_result["transcription"],
                        max_segments=40,
                    )
                    dense_segments = extract_dense_frames(
                        video_bytes=video_bytes,
                        original_name=video_file.name,
                        segments=dense_segments,
                    )

                    st.write("4/11 Dense subtitle timing review")
                    dense_progress = st.progress(
                        0,
                        text="Dense Timing: preparing multimodal batches"
                    )

                    def _dense_progress(completed, total, batch_no, batch):
                        pct = int(completed / max(1, total) * 100)
                        time_range = ""
                        if batch:
                            time_range = f" · {batch[0]['start_mmss']}–{batch[-1]['end_mmss']}"
                        dense_progress.progress(
                            pct,
                            text=(
                                f"Dense Timing completed {completed}/{total} batches "
                                f"(latest batch #{batch_no}){time_range}"
                            ),
                        )

                    dense_qa = run_dense_temporal_review(
                        api_key=api_key,
                        qa_rules=QA_RULES,
                        episode_id=episode_id,
                        source_file_name=video_file.name,
                        target_locale=target_locale,
                        proper_nouns=proper_nouns,
                        dense_segments=dense_segments,
                        batch_size=4,
                        max_workers=2,
                        progress_callback=_dense_progress,
                    )
                    dense_progress.empty()

                    st.write("5/11 Semantic repeat candidate ranking")
                    scene_windows = build_sliding_windows(
                        media_result["transcription"],
                        window_sec=12.0,
                        stride_sec=6.0,
                    )
                    repeat_candidates = rank_semantic_repeat_candidates(
                        api_key=api_key,
                        windows=scene_windows,
                        min_similarity=0.70,
                        top_k=8,
                    )

                    st.write("6/11 Visual sequence extraction for repeat candidates")
                    repeat_frames = extract_repeat_window_frames(
                        video_bytes=video_bytes,
                        original_name=video_file.name,
                        candidates=repeat_candidates,
                    )

                    st.write("7/11 Semantic + visual scene repeat review")
                    repeat_qa = run_scene_repeat_review(
                        api_key=api_key,
                        episode_id=episode_id,
                        source_file_name=video_file.name,
                        target_locale=target_locale,
                        candidates=repeat_candidates,
                        frames_by_window=repeat_frames,
                    )

                    st.write("8/11 Preparing character references")
                    uploaded_refs = save_uploaded_asset_references(uploaded_asset_specs)
                    video_refs = extract_video_reference_frames(
                        video_bytes=video_bytes,
                        original_name=video_file.name,
                        reference_lines=video_reference_lines,
                    )
                    character_references = uploaded_refs + video_refs

                    st.write("9/11 Dialogue-semantic character risk router")

                    semantic_progress = st.progress(
                        0,
                        text="Role-semantic scan: preparing transcript chunks"
                    )

                    def _semantic_progress(chunk_no, total, mode, start_t, end_t):
                        pct = int(chunk_no / max(1, total) * 100)
                        semantic_progress.progress(
                            pct,
                            text=(
                                f"Role-semantic chunk {chunk_no}/{total} · "
                                f"{start_t}–{end_t} · {mode}"
                            ),
                        )

                    semantic_role_candidates = select_role_sensitive_candidates(
                        api_key=api_key,
                        transcription=media_result["transcription"],
                        references=character_references,
                        max_candidates=8,
                        chunk_size=14,
                        overlap=4,
                        progress_callback=_semantic_progress,
                    )
                    semantic_progress.empty()

                    router_segments = build_dialogue_aware_router_segments(
                        video_bytes=video_bytes,
                        original_name=video_file.name,
                        transcription=media_result["transcription"],
                        semantic_candidates=semantic_role_candidates,
                        max_timeline_segments=4,
                    )

                    character_router_result = run_character_risk_router(
                        api_key=api_key,
                        episode_id=episode_id,
                        source_file_name=video_file.name,
                        target_locale=target_locale,
                        references=character_references,
                        router_segments=router_segments,
                    )

                    st.caption(
                        f"Role-sensitive candidates: {len(semantic_role_candidates)} · "
                        f"Character Risk Score: {character_router_result.get('episode_risk_score', 0)}/10 · "
                        f"High-risk windows: {len(character_router_result.get('risk_windows', []))}"
                    )

                    st.write("10/11 Risk-triggered deep character review")
                    character_segments = []

                    if character_router_result.get("trigger_deep_review") and character_references:
                        character_segments = build_deep_review_segments(
                            video_bytes=video_bytes,
                            original_name=video_file.name,
                            transcription=media_result["transcription"],
                            risk_windows=character_router_result.get("risk_windows", []),
                            max_segments=6,
                        )

                        char_progress = st.progress(
                            0,
                            text="Deep Character Review: preparing risk windows"
                        )

                        def _char_progress(batch_no, total_batches, batch):
                            pct = int(batch_no / total_batches * 100)
                            time_range = ""
                            if batch:
                                time_range = f" · {batch[0]['start_mmss']}–{batch[-1]['end_mmss']}"
                            char_progress.progress(
                                pct,
                                text=f"Deep Character batch {batch_no}/{total_batches}{time_range}",
                            )

                        character_qa = run_character_identity_review(
                            api_key=api_key,
                            episode_id=episode_id,
                            source_file_name=video_file.name,
                            target_locale=target_locale,
                            references=character_references,
                            check_segments=character_segments,
                            segments_per_batch=2,
                            progress_callback=_char_progress,
                        )
                        char_progress.empty()
                    else:
                        character_qa = {
                            "issues": [],
                            "review_hints": [],
                            "episode_status": "REVIEW",
                            "summary": {
                                "p0_count": 0,
                                "p1_count": 0,
                                "p2_count": 0,
                                "review_hint_count": 0,
                                "main_issue_count": 0,
                                "dimension_counts": {},
                                "high_priority_repair_notes": [],
                            },
                            "status_logic": {
                                "rule_applied": "Character deep review not triggered",
                                "explanation": character_router_result.get(
                                    "summary",
                                    "轻量角色预检未发现足够高的风险。"
                                ),
                            },
                        }
                        st.caption("Character Deep Review skipped: low risk.")

                    st.write("11/11 Audio QA + final merge")
                    audio_evidence = media_result["evidence"]
                    extra = format_repetition_evidence(verbatim_candidates)
                    if extra:
                        audio_evidence += "\n\n" + extra

                    audio_qa = run_text_review(
                        api_key=api_key,
                        episode_id=episode_id,
                        source_file_name=video_file.name,
                        target_locale=target_locale,
                        manual_evidence=audio_evidence,
                        proper_nouns=proper_nouns,
                    )

                    final_result = merge_many_results(
                        results=[audio_qa, dense_qa, repeat_qa, character_qa],
                        episode_id=episode_id,
                        source_file_name=video_file.name,
                        target_locale=target_locale,
                        status_note=(
                            "Step 5F：Step 5E 多模态审核 + 角色 reference 人脸一致性 + 台词归属审核。"
                        ),
                    )

                    status.update(label="Step 5I.2 Final complete", state="complete", expanded=False)

                st.session_state["qa_result"] = final_result
                st.session_state["media_result"] = media_result
                st.session_state["dense_segments"] = dense_segments
                st.session_state["verbatim_result"] = verbatim_result
                st.session_state["verbatim_candidates"] = verbatim_candidates
                st.session_state["scene_windows"] = scene_windows
                st.session_state["repeat_candidates"] = repeat_candidates
                st.session_state["repeat_frames"] = repeat_frames
                st.session_state["character_references"] = character_references
                st.session_state["semantic_role_candidates"] = semantic_role_candidates
                st.session_state["character_router_result"] = character_router_result
                st.session_state["character_segments"] = character_segments
                st.session_state["character_qa"] = character_qa

            except Exception as exc:
                st.error("Step 5I.2 Final 运行失败。把下面报错原样发给我：")
                st.exception(exc)

    result = st.session_state.get("qa_result")
    if result:
        st.divider()
        st.subheader("3. QA Result")
        status_value = result["episode_status"]
        st.markdown(f"## {STATUS_EMOJI.get(status_value,'')} {status_value}")

        summary = result["summary"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("P0", summary["p0_count"])
        m2.metric("P1", summary["p1_count"])
        m3.metric("P2", summary["p2_count"])
        m4.metric("Review Hints", summary["review_hint_count"])
        m5.metric("Main Issues", summary["main_issue_count"])

        issue_tab, hint_tab, json_tab = st.tabs(["Main Issues", "Review Hints", "Structured JSON"])

        with issue_tab:
            if result["issues"]:
                for issue in result["issues"]:
                    issue_card(issue)
            else:
                st.success("当前未发现明确需修改问题。")

        with hint_tab:
            if result["review_hints"]:
                for hint in result["review_hints"]:
                    hint_card(hint)
            else:
                st.success("当前无待人工复核提示。")

        with json_tab:
            st.json(result)
            st.download_button(
                "Download QA JSON",
                data=json.dumps(result, ensure_ascii=False, indent=2),
                file_name=f"{result['episode_id']}_qa_result.json",
                mime="application/json",
            )

with tab_asr:
    media_result = st.session_state.get("media_result")
    if not media_result:
        st.info("请先运行 Step 5I.2 Final。")
    else:
        st.text_area(
            "ASR Transcript",
            value=media_result["evidence"],
            height=520,
        )

with tab_dense:
    dense_segments = st.session_state.get("dense_segments")
    if not dense_segments:
        st.info("请先运行 Step 5I.2 Final。")
    else:
        st.metric("Dense Segments", len(dense_segments))
        for seg in dense_segments:
            st.markdown(
                f"### SEG {seg['index']} · {seg['start_mmss']}–{seg['end_mmss']} · "
                f"Speaker {seg['speaker']}"
            )
            st.caption(seg["text"])
            frames = seg.get("frames", [])
            if frames:
                cols = st.columns(len(frames))
                for col, frame in zip(cols, frames):
                    with col:
                        st.caption(frame["timestamp_mmss"])
                        st.image(frame["path"], use_container_width=True)

with tab_repeat:
    candidates = st.session_state.get("repeat_candidates")
    frames_by_window = st.session_state.get("repeat_frames")
    if candidates is None:
        st.info("请先运行 Step 5I.2 Final。")
    else:
        st.metric("Semantic Repeat Candidates", len(candidates))
        if not candidates:
            st.success("没有达到阈值的语义重复候选。")
        for c in candidates:
            earlier = c["earlier"]
            later = c["later"]
            with st.expander(
                f"{c['candidate_id']} · {earlier['start_mmss']}–{earlier['end_mmss']} "
                f"vs {later['start_mmss']}–{later['end_mmss']} · sim={c['similarity']}"
            ):
                st.caption("Earlier ASR: " + earlier["text"])
                st.caption("Later ASR: " + later["text"])

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Earlier sequence**")
                    for frame in (frames_by_window or {}).get(earlier["index"], []):
                        st.image(frame["path"], use_container_width=True)
                with c2:
                    st.markdown("**Later sequence**")
                    for frame in (frames_by_window or {}).get(later["index"], []):
                        st.image(frame["path"], use_container_width=True)

with tab_audio:
    candidates = st.session_state.get("verbatim_candidates")
    verbatim_result = st.session_state.get("verbatim_result")
    if candidates is None:
        st.info("请先运行 Step 5I.2 Final。")
    else:
        st.metric("Verbatim Repetition Candidates", len(candidates))
        for c in candidates:
            st.markdown(
                f"- **{c['start_mmss']}–{c['end_mmss']}** · "
                f"`{c['surface']}` · {c['type']}"
            )
        if verbatim_result:
            with st.expander("View verbatim full text"):
                st.write(verbatim_result.get("text", ""))


with tab_character:
    refs = st.session_state.get("character_references")
    char_segments = st.session_state.get("character_segments")
    char_qa = st.session_state.get("character_qa")

    if refs is None:
        st.info("请先运行 Step 5I.2 Final。")
    else:
        st.subheader("Character References")
        st.metric("Reference Count", len(refs))

        if not refs:
            st.warning(
                "本轮没有角色 reference，因此 Character Identity 模块没有正式运行。"
                "建议上传资产图，或用 `Name|Role|mm:ss` 指定视频 reference。"
            )
        else:
            cols = st.columns(min(4, max(1, len(refs))))
            for idx, ref in enumerate(refs):
                with cols[idx % len(cols)]:
                    st.image(ref["path"], use_container_width=True)
                    st.markdown(f"**{ref['name']}** · {ref['role']}")
                    st.caption(
                        "Asset" if ref.get("source") == "uploaded_asset"
                        else f"Video ref @ {ref.get('timestamp_text','')}"
                    )

        semantic_candidates = st.session_state.get("semantic_role_candidates") or []
        if semantic_candidates:
            st.divider()
            st.subheader("Role-sensitive Dialogue Candidates")
            for c in semantic_candidates:
                excluded = ", ".join(c.get("excluded_characters") or []) or "-"
                st.markdown(
                    f"- **{c['start_time']}–{c['end_time']}** · "
                    f"{c['relation_type']} · priority={c['priority']}"
                )
                excluded_roles = ", ".join(c.get("excluded_roles") or []) or "-"
                st.caption(
                    f"Expected Character: {c.get('expected_character') or '-'} · "
                    f"Excluded Character: {excluded} · "
                    f"Expected Role: {c.get('expected_role') or '-'} · "
                    f"Excluded Roles: {excluded_roles}"
                )
                if c.get("scene_role_hypothesis"):
                    st.caption("Scene Role Hypothesis: " + c["scene_role_hypothesis"])
                st.caption("Reason: " + c["reason"])

        router_result = st.session_state.get("character_router_result")
        if router_result:
            st.divider()
            st.subheader("Character Risk Router")
            r1, r2, r3 = st.columns(3)
            r1.metric("Episode Risk", f"{router_result.get('episode_risk_score', 0)}/10")
            r2.metric("High-risk Windows", len(router_result.get("risk_windows", [])))
            r3.metric("Deep Review", "TRIGGERED" if router_result.get("trigger_deep_review") else "SKIPPED")

            for rw in router_result.get("risk_windows", []):
                st.markdown(
                    f"- **{rw['start_time']}–{rw['end_time']}** · "
                    f"risk={rw['risk_score']} · {', '.join(rw['risk_types'])}"
                )
                st.caption(rw["reason"])

        if char_qa:
            st.divider()
            st.subheader("Character QA Findings")
            c1, c2 = st.columns(2)
            c1.metric("Character Main Issues", len(char_qa.get("issues", [])))
            c2.metric("Character Review Hints", len(char_qa.get("review_hints", [])))

            if char_qa.get("issues"):
                st.markdown("### Main Issues")
                for issue in char_qa["issues"]:
                    issue_card(issue)

            if char_qa.get("review_hints"):
                st.markdown("### Review Hints")
                for hint in char_qa["review_hints"]:
                    hint_card(hint)

        if char_segments:
            with st.expander("View character-check segments"):
                for seg in char_segments:
                    st.markdown(
                        f"**SEG {seg['index']} · {seg['start_mmss']}–{seg['end_mmss']}** "
                        f"· {seg['text']}"
                    )
                    frames = seg.get("frames", [])
                    if frames:
                        cols = st.columns(len(frames))
                        for col, frame in zip(cols, frames):
                            with col:
                                st.image(frame["path"], use_container_width=True)

with tab_rules:
    st.subheader("12 QA Dimensions")
    for key, value in DIMENSION_ZH.items():
        st.markdown(f"- **{value}**  `{key}`")
