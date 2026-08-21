# Step 5E — Dense Timing + Visual Repeat + Verbatim Audio

## Why Step 5E exists

Step 5D improved temporal alignment but still missed:
- partial missing subtitle
- cleaned-away audio repetitions
- visually repeated plot/action with semantically similar but not identical dialogue

Step 5E adds three independent layers.

## Layer 1 — Dense Subtitle Timing
Each ASR segment gets 3–4 frames across the utterance instead of one midpoint frame.

Target:
- missing subtitle at the beginning of a line
- subtitle delay / advance
- subtitle residue
- subtitle vs ASR mismatch

## Layer 2 — Verbatim Audio
A second word-timestamp transcription uses `whisper-1` with instructions to preserve repetitions.
Local logic detects adjacent repeated words / phrases.

Target:
- `quente quente`
- accidental word repetition
- short phrase duplication

## Layer 3 — Semantic + Visual Scene Repeat
- Build sliding 12s transcript windows with 6s stride
- Rank semantically similar windows with `text-embedding-3-small`
- Extract 3-frame sequences from candidate windows
- Multimodal model judges whether the same plot/action was repeated

Target:
- same action played twice
- same scene block reinserted
- visually repeated plot even when dialogue is not word-for-word identical

## Run

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Regression test

Use the known diagnostic episode and check whether Step 5E surfaces:
- 00:15 repeated `quente`
- 02:11 partial missing subtitle
- 02:11–02:25 vs 02:25–02:37 repeated coffee-spill plot/action
- 02:49–02:52 subtitle/dubbing mismatch

Do not evaluate success by total issue count.
Evaluate against these known Ground Truth items.
