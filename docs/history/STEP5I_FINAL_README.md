# Step 5I Final — Semantic Role + Face Identity

This is the proposed frozen MVP baseline before formal pilot evaluation.

## Character Identity principle

A face-mix case does NOT require a character name to appear in dialogue.

The system first infers the scene role from the entire dialogue chain:

```text
Dialogue context
+ pronoun perspective
+ relationship terms
+ conversational role
+ neighboring 3–5 turns
↓
Expected / excluded scene role
↓
compare with character reference face
↓
face_swap_or_face_mix
```

Examples without names:

- “你也是他的朋友吗？” + earlier context says “he” is the male lead
  → current addressee is probably not the male lead.

- “替我转告你老板……”
  → current addressee is probably a subordinate/assistant, not the boss.

- “你是新郎的朋友吗？”
  → current addressee is normally not the groom.

- Several consecutive lines establish the interlocutor as a doctor / friend / male second lead,
  even if nobody says the character's name.

If the visible face instead matches an excluded reference identity, Deep Character Review can
classify `visual_logic / face_swap_or_face_mix`.

## False-positive control

A Main Issue still requires:
1. stable multi-frame face evidence against references, AND
2. strong dialogue / relationship / scene-role contradiction.

Possible disguise, amnesia, intentional misrecognition, deception, or role-play should be downgraded.

## Subtitle timing

The Step 5I semantic-gap fix is retained:

- minor delay + later subtitle recovers the full sentence → filter
- meaningful beginning clause is NEVER represented in later subtitle frames → keep as true partial missing subtitle

## Proposed freeze criteria

Before changing architecture again, regression-test:
1. normal-sync timing false positive remains suppressed
2. known partial missing subtitle returns
3. known face-mix case triggers role-sensitive candidate + router + deep review
4. a clean character episode does not trigger unnecessary deep review
