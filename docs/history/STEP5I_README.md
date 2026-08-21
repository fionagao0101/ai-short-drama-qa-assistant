# Step 5I — Semantic Timing + Dialogue-aware Character Router

## Fix 1: Timing tolerance without losing true partial-missing-subtitle recall

The system no longer relies on delay duration alone.

It asks:
- Did later subtitles recover the FULL spoken sentence?
  - Yes → minor timing offset, filter.
- Was an independently meaningful beginning clause NEVER represented in any subtitle frame?
  - Yes → `partial_missing_subtitle_semantic_unit`, keep even if delay <1.2s.

Regression goals:
- normal-sync false positive around 00:04–00:06 stays suppressed
- true partial missing subtitle around 02:11 returns

## Fix 2: Dialogue-aware face-mix detection

A new text-only selector scans the full ASR transcript for role-sensitive lines:
- direct name address
- third-person exclusion
- identity claims
- relationship titles
- role-sensitive questions/statements

Example:
`Você também é amigo do Juliano, não é?`

This is a strong third-person exclusion clue:
the visible addressee should normally NOT be Juliano.

If the visible addressee's face matches the Juliano reference:
- Character Router raises `third_person_identity_conflict`
- Deep Review is triggered
- Deep Review may classify `visual_logic / face_swap_or_face_mix`

This makes character QA use both:
`face reference + dialogue relationship semantics + scene context`

rather than face similarity alone.
