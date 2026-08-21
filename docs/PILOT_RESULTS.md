# Formal Pilot Results — Step 5I.2 Final

## Protocol

1. Human reviewer watches the episode first without seeing AI output.
2. Record blind Ground Truth: time range, severity, issue type, description, and manual review time.
3. Run the frozen Step 5I.2 Final baseline.
4. Match every blind GT issue to Main Hit / Hint-or-Auxiliary Hit / Miss.
5. Human-confirm AI-only findings and false positives.
6. Record human follow-up time separately from AI runtime.
7. Do not tune the model during T003–T006.

## Results

| Metric | Result |
|---|---:|
| Episodes | 4 |
| Blind GT issues | 5 |
| System-level recall | 100% (5/5) |
| Main-table recall | 60% (3/5) |
| P0/P1 critical recall | 100% (5/5) |
| Main issue precision | 100% (4/4) |
| AI-only confirmed discoveries | 4 |
| Total manual blind review | ~20 min |
| AI-guided human follow-up | ~2 min |
| Human workload reduction | ~90% |
| Human review speed-up | ~10× |
| Average AI runtime | ~2.75 min/episode |

## Interpretation

The strongest result is not “AI replaces human QA.” The useful pattern is:

- Critical issues were surfaced somewhere in the system for all five blind GT cases.
- Four AI Main Issues were all human-confirmed in this pilot.
- Four additional actionable issues were discovered after the blind human pass.
- Two blind GT issues were found in auxiliary evidence layers but were not routed into the final Main Issues surface.

Therefore the next product improvement is **auxiliary evidence → final issue routing**, while preserving the validated underlying detectors.

## Limitations

This is a small MVP pilot: four episodes and five blind GT issues. Results should be described as **pilot validation**, not production-scale recall/precision. More titles, genres, episode lengths, suppliers, and clean negatives are required for generalization claims.
