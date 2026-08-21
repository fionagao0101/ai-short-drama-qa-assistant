# Step 5G — Subtitle Timing Precision Tuning

This iteration suppresses false positives caused by treating discrete frame sampling and ASR segment boundaries as exact subtitle timing ground truth.

## Changes

- ASR and frame evidence now carries sub-second timestamps.
- ASR segment boundaries are treated with ±0.45s uncertainty.
- Subtitle appearance within ~0.8s of ASR onset is normally ignored when semantic content matches.
- One early frame without subtitle is never enough for a Main Issue.
- A timing Main Issue now requires persistent multi-frame evidence or a materially missing semantic clause.
- Timing-only issues below 95% confidence are filtered.
- Timing-only issues between 95% and 98% are routed to Review Hints.
- Semantic subtitle/dubbing mismatch remains unaffected by these tolerance rules.

## Regression check

Re-run:
1. the known true partial-missing-subtitle case around 02:11 — should still surface;
2. the normal-sync false-positive case around 00:04–00:06 — should disappear from Main Issues or be suppressed.
