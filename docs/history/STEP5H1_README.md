# Step 5H.1 — Dense Review Performance Tune

Why Step 4 was slow:
- up to 40 ASR-aligned segments
- 3–4 images per segment
- batch size 3
- roughly up to 14 sequential multimodal requests

Changes:
- keep full dense segment coverage
- batch size 4
- 2 concurrent multimodal batches
- visible batch progress inside Step 4
- no change to Step 5G timing-tolerance logic
- no change to Step 5H Character Risk Router

Expected result:
- Step 4 should feel substantially faster
- UI shows whether it is actively progressing rather than looking frozen

If API rate limits occur, reduce max_workers from 2 to 1.
