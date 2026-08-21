# Step 5I.1 Stable — Timeout-safe Character Semantics

Fixes the Step 5I Final APITimeoutError in the full-transcript role semantic selector.

## Changes

1. Full transcript semantic scan is chunked
   - ~14 ASR turns per chunk
   - 4-turn overlap preserves relationship context
   - visible progress per chunk

2. Selector timeout is non-fatal
   - 75s explicit timeout
   - 1 automatic retry
   - failed chunks fall back to local relationship/pronoun/name heuristics
   - other chunks continue

3. Character visual router timeout is non-fatal
   - 150s timeout
   - if it times out, priority-4/5 semantic candidates automatically trigger Deep Character Review

4. Deep Character Review timeout is non-fatal
   - 180s per request
   - a failed batch becomes a Review Hint instead of crashing the entire episode

The Step 5I Final logic is preserved:
- face mix does not require names
- scene-role / relationship / perspective inference remains enabled
- semantic timing fix remains enabled
