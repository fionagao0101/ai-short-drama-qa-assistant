# Step 5F — Character Identity & Speaker Attribution

## New capability

Step 5F adds a character-reference layer for:

- male lead / male second lead face mixing
- role asset misbinding
- character replacement
- dialogue assigned to the wrong character

## Reference modes

### 1. Character asset images — recommended

Download/save the character asset images from the internal Feishu/Lark document and upload them in the app.

For each image, label:
- character name
- role

Example:
- Lucas — 男主
- Rafael — 男二

### 2. Video reference timestamps — no separate image required

If you know a clean moment where the character is clearly visible, enter:

```text
Lucas|男主|00:18
Rafael|男二|00:33
Bianca|女主|00:47
```

The app extracts those video frames and uses them as reference identities.

You can use both asset images and video timestamps in the same run.

## Why not paste a Feishu private-document link?

Private Feishu/Lark documents require authentication/session access. The local Streamlit MVP does not currently integrate Feishu OpenAPI authentication, and a public portfolio demo should not depend on an internal private link.

Prefer:
- save/download the relevant asset images, or
- use video reference timestamps.

## Evidence Gate

Character identity Main Issue should normally have at least two supporting signals:

1. face identity against a known reference
2. visual active-speaker cue
3. dialogue / role semantics

Speaker A/B/C from ASR is never treated as character identity ground truth.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```
