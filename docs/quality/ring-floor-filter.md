# Quality loop: `feat/ring-mask` (ring-floor filter, Olympic Boxing)

Started 2026-10-27. Gate: [`docs/gates/ring-floor-filter.md`](../gates/ring-floor-filter.md), committed in
`ddf6399` before any floor was clicked, any margin chosen or any held-out number computed. Olympic Boxing is
research-use only: every number here is an aggregate; no frame, crop or per-person position leaves `results/`.

## Before the build: why the brief's pixel mask was not built (development videos only)
- Two cameras: kam2 (15 chapters of recording 8416) and kam4 (14 chapters of 9681), each at a ring corner at
  about head height. 5-9 people per frame; 69-79% of person-frames are on ids no punch is credited to.
- Most generous boxer-safe region (union of every credited boxer box in both dev videos of a camera): kam4
  0.0% of non-credited person-frames lie mostly outside it; kam2 2.3% and 26.4%. Any region that keeps every
  boxer contains that union, so a pixel mask cannot remove kam4's spectators at all.
- Feet line (fitted on one dev video, applied to the other, 30 px margin): kam4 removes 45-55% of non-credited
  person-frames at 3-4% credited cost; kam2 15-28% at 3-6%.
- Brendan (10-27): feet filter on the existing tables, clicked floor per video; pixel mask recorded as a
  development-only negative; no GPU re-extraction.

## Camera stillness (label-free, all 29 videos, before any floor)
- Between chapters the camera moves (kam2's first chapter vs most later ones: 85-120 px), so floors are per video.
- Within a video, first version compared each frame with the whole-video median and read GH208416 as still
  (0 px) although it pans ~145 px at 62% of the video: a blended median correlates best at zero shift (OpenCV's
  phaseCorrelate agreed, peak response 0.08). Fixed before the gate was committed: the reference is the
  camera's starting position (median of the first 3 samples). Regression test added.

- `prepare` with the fixed reference (10-27): left out GH208416 (143.6 px) and GH188416 (26.3 px), as the
  gate expected; kept GH108416 (15.9), GH079681 (11.2), GH138416 (10.2); every other video <= 2 px. Held-out
  set: 23 videos. Development videos: all still (<= 2 px).

## Iteration 1
- Self-review (before Astra's report): G2 and G3a pooled over every evaluated video while G1 and G3b left out
  videos whose baseline failed; the gate says every metric pools over the compared videos. Fixed: a baseline
  failure is in no arm; a video that fails only when filtered stays in the filter's cost and fails G3c.
  Regression tests added (161 checks).
