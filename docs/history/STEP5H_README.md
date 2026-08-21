# Step 5H — Character Risk Router

## Why this iteration

Step 5G deep-reviewed character identity on every episode, which could take ~10 minutes.
That is not suitable for 70–80 episode batch QA.

Step 5H changes the character layer into a cascade:

```text
Every episode
↓
Always-on Character Risk Router
↓
Low risk → skip Deep Character Review
High risk → deep-review only flagged windows
```

## Lightweight router

Uses:
- character references
- up to 10 episode-wide sampled dialogue segments
- one representative frame per segment
- ASR text as supporting context

Outputs:
- episode risk score (0–10)
- risk windows
- risk types
- whether deep review should trigger

## Deep review

Only runs when:
- a risk window score >= 3, or
- episode risk is high enough

Then only those risk windows (+ adjacent segments) enter multi-frame Character Identity & Speaker Attribution review.

## Expected benefit

For normal episodes:
- character module still runs automatically
- no manual "on/off" decision
- expensive deep review is skipped

For suspicious episodes:
- deep review focuses on a few high-risk windows instead of 16 full segments
- progress is shown batch by batch

## Reference behavior

The MVP still needs character references:
- asset images, or
- video reference timestamps

In a future batch version, these should become a drama-level Character Library shared across all episodes.
