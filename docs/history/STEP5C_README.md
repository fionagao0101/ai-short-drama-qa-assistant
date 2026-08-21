# Step 5C — Video → ASR + Keyframes → Multimodal QA

## Install/update dependencies

```bash
python3 -m pip install -r requirements.txt
```

## Start

```bash
python3 -m streamlit run app.py
```

## Test flow

1. Paste your OpenAI API key in the sidebar.
2. Upload one short-drama MP4.
3. Add known proper nouns if available.
4. Click `Run FULL MULTIMODAL QA`.
5. Inspect:
   - `ASR Transcript`
   - `Keyframes`
   - `Main Issues / Review Hints`

## What Step 5C adds

Compared with Step 5B, Step 5C adds:
- sampled keyframe extraction from video
- multimodal visual review using frame images + ASR evidence
- ability to catch some burned subtitle / visual text / culture / continuity issues
- merged audio + visual QA result

## Important limitation

This is still an MVP:
- keyframes are sampled, not every frame
- a short-lived issue may still be missed
- speaker diarization labels are not character identity ground truth
- full subtitle timing and lip-sync inspection still require denser frame coverage or frame-level analysis
