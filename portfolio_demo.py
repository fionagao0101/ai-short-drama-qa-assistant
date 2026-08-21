import json
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "demo" / "sample_result.json").read_text(encoding="utf-8"))

st.set_page_config(
    page_title="AI Short Drama QA Assistant · Portfolio Demo",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 AI Short Drama Localization QA Assistant")
st.caption("Portfolio Demo · Synthetic evidence only · No production video or API key required")
st.markdown(
    "A human-in-the-loop multimodal QA workflow for **pt-BR short-drama localization**, "
    "covering subtitle/audio consistency, timing, repeated dialogue/plot, proper nouns, "
    "and character identity / face-mix risks."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Pilot System Recall", "100%", "5/5 blind GT")
m2.metric("Main Precision", "100%", "4/4 confirmed")
m3.metric("Human Workload", "−90%", "20 min → ~2 min")
m4.metric("Review Speed", "~10×", "pilot estimate")

st.info(
    "Pilot scope: 4 episodes and 5 blind human GT issues. Metrics are MVP pilot validation, "
    "not production-scale accuracy claims."
)

qa_tab, arch_tab, pilot_tab, safety_tab = st.tabs([
    "Synthetic QA Result", "Architecture", "Pilot Evaluation", "Human-in-the-loop"
])

with qa_tab:
    st.subheader("Example Result")
    s = DATA["summary"]
    a, b, c, d = st.columns(4)
    a.metric("P1", s["p1_count"])
    b.metric("Main Issues", s["main_issue_count"])
    c.metric("Review Hints", s["review_hint_count"])
    d.metric("Status", DATA["episode_status"])

    for issue in DATA["issues"]:
        st.markdown(f"### {issue['severity']} · {issue['start_time']}–{issue['end_time']} · {issue['subtype']}")
        st.write(issue["reason"])
        left, right = st.columns(2)
        with left:
            if issue.get("subtitle_text"):
                st.caption("Subtitle evidence")
                st.code(issue["subtitle_text"], language=None)
            if issue.get("spoken_text"):
                st.caption("Audio / ASR evidence")
                st.code(issue["spoken_text"], language=None)
        with right:
            st.caption("Suggested repair")
            st.write(issue["suggested_fix"])
            st.caption(f"Confidence: {issue['confidence']:.0%} · {issue['carrier']}")
        st.divider()

    st.subheader("Review Hint")
    for hint in DATA["review_hints"]:
        st.markdown(f"**{hint['start_time']}–{hint['end_time']} · {hint['dimension']}**")
        st.write(hint["suspected_issue"])
        st.caption("Why not Main: " + hint["why_not_main_issue"])

with arch_tab:
    st.subheader("Multimodal QA Pipeline")
    st.code("""MP4 Episode
├─ Audio extraction → diarized pt-BR ASR
├─ Word-level verbatim layer → repetition candidates
├─ Dense ASR-aligned frames → subtitle timing / semantic mismatch
├─ Transcript embeddings + visual sequence → plot/action repeat
├─ Character references + dialogue-role semantics
│  └─ risk router → suspicious windows → deep character review
└─ Evidence Gate → Main Issues / Review Hints → structured JSON""", language=None)
    st.markdown(
        "**Design choice:** cheap prechecks run broadly; expensive multimodal review is triggered only on "
        "high-risk windows. ASR speaker labels are treated as voice clusters, never as character identity ground truth."
    )

with pilot_tab:
    st.subheader("Frozen Pilot · Step 5I.2 Final")
    st.markdown("""
| Metric | Result |
|---|---:|
| Formal pilot episodes | 4 |
| Blind human GT issues | 5 |
| System-level recall | **100% (5/5)** |
| Main-table recall | **60% (3/5)** |
| P0/P1 critical recall | **100% (5/5)** |
| Main issue precision | **100% (4/4)** |
| AI-only confirmed discoveries | **4** |
| Human workload reduction | **~90%** |
| Human review speed-up | **~10×** |
""")
    st.caption(
        "Two blind GT issues were detected in auxiliary modules but did not reach the final Main Issues surface, "
        "highlighting routing/orchestration as the next product improvement rather than basic detection coverage."
    )

with safety_tab:
    st.subheader("Human-in-the-loop by design")
    st.markdown("""
- Precision-first Evidence Gate: uncertain findings stay in **Review Hints**.
- Model output does **not** autonomously approve or reject production content.
- Character identity requires multiple evidence types; ASR diarization labels are not treated as identities.
- Production assets, videos, and API credentials are intentionally excluded from this public demo.
- The full QA app can be run locally or deployed privately with the operator's own API credentials.
""")
