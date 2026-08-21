# Architecture

## End-to-end pipeline

```text
MP4
├─ Audio extraction
│  ├─ diarized pt-BR ASR
│  └─ word-level verbatim transcript
├─ Dense ASR-aligned frame extraction
│  └─ subtitle timing + semantic alignment
├─ Semantic repeat candidate ranking
│  └─ visual sequence comparison
├─ Dialogue-aware character semantics
│  ├─ role-sensitive candidate selection
│  ├─ lightweight Character Risk Router
│  └─ high-risk windows → Deep Character Review
└─ Evidence Gate
   ├─ Main Issues
   ├─ Review Hints
   └─ structured JSON
```

## Key iterations

- **Uniform keyframes → dense temporal alignment:** fixed poor recall on short subtitle timing events.
- **ASR → verbatim word-level layer:** recovered repeated words that normalized transcription could hide.
- **String similarity → semantic + visual repeat:** improved detection of repeated plot actions with paraphrased dialogue.
- **Character deep review → risk-triggered cascade:** reduced unnecessary heavy multimodal calls.
- **Face similarity → dialogue-role semantics + face evidence:** enabled face-mix detection even when no character name appears in dialogue.
- **Timing tolerance → semantic-gap exception:** suppressed benign sub-second offsets without losing true partial missing subtitles.
- **Timeout-safe chunking:** character semantic scanning degrades gracefully instead of failing the entire episode.

## Evidence Gate

The Evidence Gate is the product-level boundary between model perception and production decisions. High-confidence, locatable, actionable findings can enter Main Issues. Uncertain but useful signals stay in Review Hints for human confirmation.
