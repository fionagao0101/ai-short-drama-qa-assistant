# Step 5D — Temporal Alignment Layer

## Install/update dependencies

```bash
python3 -m pip install -r requirements.txt
```

## Start

```bash
python3 -m streamlit run app.py
```

## What Step 5D changes

Compared with Step 5C:
- no longer relies mainly on random global keyframes
- builds ASR-aligned segments from the transcript
- extracts one representative frame for each aligned segment
- asks the model to judge subtitle presence / mismatch on those aligned segments
- locally detects likely duplicate dialogue windows
- locally flags preserved repeated-word candidates

## Test flow

1. Paste your OpenAI API key in the sidebar.
2. Upload the same video that exposed misses in Step 5C.
3. Click `Run TEMPORAL MULTIMODAL QA`.
4. Check:
   - `QA Result`
   - `Aligned Segments`
   - whether 02:11 missing subtitle, 02:25–02:37 repetition, 02:49–02:52 mismatch get surfaced better.

## Important limitation

This is still an MVP:
- each ASR segment currently uses one representative frame
- if a subtitle only flashes at the very beginning / very end of the segment, it may still be uncertain
- repeated-word issues still depend on ASR preserving the repeated token
- full subtitle timing verification would be further improved by multi-frame-per-segment sampling
