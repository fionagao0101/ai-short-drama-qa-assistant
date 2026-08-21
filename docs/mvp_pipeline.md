# MVP Processing Pipeline

## Product Decision

The v0.1 MVP accepts a short-drama episode video as the primary user input, but does not rely on sending the entire raw video directly to a single model.

The pipeline decomposes the episode into structured evidence first:

1. Upload MP4 video.
2. Extract audio track.
3. Generate ASR transcript with timestamps.
4. Sample keyframes around scene changes and dialogue segments.
5. Run subtitle/visual-text inspection on relevant frames.
6. Build or load character/entity context.
7. Send structured evidence + QA rules to the review model.
8. Route findings through Evidence Gates.
9. Output MAIN_ISSUE vs REVIEW_HINT.
10. Render a human-readable Chinese QA report.
11. Preserve structured JSON for later evaluation.

## Why this architecture

- More controllable than one-shot video prompting.
- Easier to debug false positives and false negatives.
- Supports timestamped evidence.
- Makes evaluation possible.
- Can later replace individual modules without changing the whole product.

## MVP Input Modes

### Primary mode
Video upload (`.mp4`).

### Debug mode
Video + optional manually supplied transcript.

### Optional enrichment
Known proper nouns and character assets.

## Episode Status Logic

- `REJECT`: at least one confirmed P0 issue.
- `REVIEW`: no P0, but at least one confirmed P1/P2 issue, or there are review hints requiring human verification.
- `PASS`: no confirmed issues, no unresolved review hints, and required episode scan was completed.

## Human-in-the-loop principle

The model never makes the final business approval decision autonomously. It produces structured evidence and a recommended status. A human reviewer retains final authority.
