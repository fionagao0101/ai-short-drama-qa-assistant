# Step 5I.2 Final — Regression Locked

This version is intended to freeze the MVP before formal pilot evaluation.

It preserves the Step 5I.1 stability fixes and restores two previously validated capabilities.

## Regression lock 1 — Subtitle / dubbing semantic mismatch

Timing Tolerance applies ONLY when subtitle and audio mean the same thing.

If multi-frame burned subtitles and ASR have different core meaning:
- subtype: `subtitle_dubbing_semantic_mismatch`
- P1
- timing tolerance cannot suppress it

The dense reviewer now checks semantic mismatch BEFORE timing.

## Regression lock 2 — Plot / action repetition

A review hint is promoted back to P1 only when:
- upstream semantic similarity >= 0.88
- visual-review confidence >= 0.82
- the visual model explicitly describes repeated plot/action/event
- there is no clear flashback / recap explanation

This is designed to recover known cases such as a distinctive coffee-spill action being repeated,
without promoting ordinary same-location / same-character shots.

## Final regression checklist

Before freezing:
1. normal minor timing offset → filtered
2. true partial missing subtitle → detected
3. known subtitle/dubbing semantic mismatch → detected
4. known plot/action repeat → Main Issue
5. known face-mix / wrong character → detected
6. clean character episode → no unnecessary deep-review explosion
7. API timeout → degrades gracefully, does not crash the full episode

If these pass, stop changing architecture and begin formal pilot evaluation.
