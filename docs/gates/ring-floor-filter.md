# Gate: the ring-floor filter (Olympic Boxing) -- written 2026-10-27, before any held-out number

Branch `feat/ring-mask`. Committed BEFORE any ring floor is clicked, before the margin is chosen on the
development videos, and before any held-out video is evaluated; the commit hash of this file is the evidence of
order. Nothing below changes after a held-out number exists. A later change is a new, dated section that says
why, and the original text stays.

## Question
A class suggestion (talk, 2026-10-27): "blur the background of the Olympic videos". Does removing the people who
stand outside the ring improve two-boxer tracking? The problems it targets, all measured on main:
- spectators and the referee become ByteTrack tracks (5-9 people per frame on the development videos; 69-79% of
  person-frames belong to ids no labelled punch is credited to);
- the MILP linker (#11) absorbs spectators and swaps boxers (3 identity errors in about a minute on GH078416, by eye);
- completeness finds many unexplained wrist-speed bursts: 39.4 per exposure-minute on the linked path (RESULTS 7c)
  and 45.4 on credited fragments (7g), re-run on main 10-24, identical.

## What changed from the brief, and why (development videos only, 2026-10-27, before this gate)
The brief's main arm was a pixel MASK before pose extraction (black out everything outside a per-video ring
region, re-extract pose on the GPU laptop). It was checked on the four development videos first, with no GPU time:

| Dev check (scratch, numbers only) | kam4 (14 videos) | kam2 (15 videos) |
|---|---|---|
| Pixel mask, most generous boxer-safe region (union of every credited boxer box in both dev videos of the camera): non-credited person-frames lying mostly outside, an UPPER bound | **0.0%** | 2.3% and 26.4% |
| Feet filter after pose (feet line fitted on one dev video, applied to the other, 30 px margin): non-credited person-frames removed; credited person-frames removed | 45-55%; 3-4% | 15-28%; 3-6% |

Why the mask cannot work here: both cameras stand at a ring corner at about head height, so the crowd behind the
far ropes occupies the same pixels as a boxer standing at those ropes. Any region that keeps every boxer contains
where the credited boxers actually stood, and on kam4 even that subset blacks out nobody. Only depth separates
the two, and in one image the cue to depth is where a person's feet are.

**Decision (Brendan, 2026-10-27):** test the feet filter on the existing tables (no re-extraction, no laptop, no
Syncthing); record the pixel mask as a development-only negative. Consequences, stated now:
- The filter runs AFTER tracking. It cannot change what ByteTrack associated, so an id that already mixes a
  spectator and a boxer stays mixed. It removes whole ids only and never cuts a kept id into pieces.
- The referee stands on the floor and is never removed. "Non-boxer" below includes the referee.
- Blur (the brief's optional second arm) is not run: a blurred person is still detected, and the mask already
  fails on geometry.

## The ring floor (non-circular by construction)
- **Who and how.** Brendan clicks the outline of the ring floor (the canvas inside the ropes, at floor level) once
  per video, in `olympic_ring.py click`. The image is that video's background: the per-pixel median of 24 frames
  sampled evenly through the video, so moving people are gone. The tool shows no label, track, burst or number.
  Where the floor runs off the picture he clicks on the picture's edge. Each click session is timed (`click_s`),
  and the total goes in the PR.
- **Why not from labels.** The gate counts unexplained bursts against the punch labels. A region drawn from the
  labels or credited-punch positions of the scored videos would remove unexplained bursts by construction.
- **Why per video, not per camera.** The camera moves between GoPro chapters: against the dev reference, kam2's
  first chapter sits 85-120 px away from most later ones (label-free phase correlation of background images).
- **The file** (`data/olympic_ring/floors.json`, gitignored) binds each polygon to its video's content
  fingerprint, and the polygon file is a trust boundary: `ring.load_floors` refuses anything malformed.

## Camera stillness (one polygon must fit the whole video)
`motion_px` = the 90th percentile, over the 24 sampled frames, of the phase-correlation shift between the frame
and the camera's starting position (the median of the first 3 samples), in full-size pixels; pixels only, no
labels. Not against the whole-video median: a camera that pans partway through blends both positions into it,
and every frame then correlates best at zero shift (found on GH208416 during the build, 10-27). A video with
`motion_px > 20` is LEFT OUT of both arms, and listed. Disclosed: this threshold was set after seeing the
label-free camera motion of all 29 videos (scratch probe and `prepare`, 10-27), and no outcome metric of any
video. Expected to leave out GH208416 (pans ~145 px at 62% of the video) and GH188416 (~26 px); the rule decides,
not this sentence. A pan shorter than ~10% of a video can pass unseen, and a camera that moves in the first ~12%
of a video is read as moving for the rest of it (it is then left out): stated limits.

## The filter rule
- Feet point of a row = the bottom centre of its box, held inside the frame.
- A track id is DROPPED when fewer than half of its rows have the feet point inside the floor grown by `m` pixels
  (`m` < 0 shrinks it). Exactly half is kept.
- **Choosing `m` (development videos only).** Grid {-30, -15, 0, 15, 30, 60} px. Among the margins that drop at
  most 0.5% of the dev videos' credited punches (their credited person dropped), take the one that removes the
  most non-credited person-frames; ties go to the larger margin. If no margin qualifies, the experiment stops
  there (Future Work). The choice is frozen as `FROZEN_MARGIN_PX` in `olympic_ring.py` by its own commit; the
  held-out run refuses any other margin and refuses to run while it is unset.

## Split (ADR-0013's, fixed before this design)
- DEV (design only; numbers are not evidence): GH078416, GH159681, GH128416, GH069681.
- HELD-OUT (gate data): the other 25, minus any video the stillness rule leaves out. If a floor is missing for a
  held-out video, nothing runs: no partial verdict.
- Disclosed: the held-out videos are part of the all-29 numbers already published (RESULTS 7a-7i). Nothing from
  those numbers enters the floors, the stillness rule or the margin.

## Baseline
The unfiltered pipeline on main, same videos, same code: the tables in `results/ingest/olympic` (YOLO11m-pose +
ByteTrack, extracted on main), then `olympic.attribute` -> `linking.link_video` -> `completeness.find_bursts`
and `find_bursts_credited` with the default rule, exactly as `olympic_attribute.py`, `olympic_link.py` and
`olympic_completeness.py` call them. The filtered arm is the same calls on the same tables minus the dropped ids.
**Self-check before the held-out run:** the evaluator's DEV baseline equals `olympic_completeness.py --only <dev>`
(linked and `--credited`) to the reported precision, and the evaluator asserts in code that its bootstrap rate is
the completeness module's rate. Recorded in the quality log.

A video whose BASELINE fails (linking or completeness raises) is left out of both arms and listed. A video that
fails only when filtered counts against the filter (G3c).

## Metrics, bars and directions (held-out, pooled over the compared videos)

| Id | Metric | Direction | Bar |
|---|---|---|---|
| **G1** | Linked-path unexplained bursts per exposure-minute (as row 7c), filtered / baseline | lower | ratio <= **0.90** AND the 97.5th percentile of a 2,000-resample video bootstrap (seed 20261027) < **1.00** |
| **G2** | Share of the baseline's non-credited person-frames (rows on ids no punch is credited to) that the filter removes | higher | >= **20%** (the mechanism: if it removes few, a G1 change is not its doing) |
| **G3a** | Share of baseline-credited punches whose credited person is dropped ("boxers near the ropes") | lower | <= **1.0%** pooled AND <= **5%** in every video |
| **G3b** | Share of labelled punches credited to anyone (row 7a), after re-attribution | must not fall | filtered >= baseline - **1.0 pp** |
| **G3c** | Videos that fail only when filtered | none | **0** |
| S1 | Linking coverage: credited punches on A or B (row 7b) | higher | filtered >= baseline - 1.0 pp |
| S2 | Absorption proxy: share of A/B person-frames on ids with no credited punch | lower | filtered < baseline |
| S3 | Credited-fragment unexplained bursts per exposure-minute (the 45/min of row 7g, without skips) | lower | filtered <= baseline |
| S4 | Median both-identified share (row 7b) | higher | filtered >= baseline - 2 pp |

**Verdict: PASS iff G1, G2, G3a, G3b and G3c all hold.** S1-S4 are reported with their bars and never change the
verdict. On PASS, the filter may be adopted as an optional step, recorded by an ADR in the vault (Brendan records
the reasons). On FAIL, or if the time-box is missed, it is a Future Work finding (`PAPER-future-work.md`), with no
further iteration.

## Said in advance
- **S3 should barely move, by construction.** The credited-fragment path only reads ids credited with a punch,
  and the filter mostly removes ids no punch is credited to. The 45/min figure is therefore NOT where an effect
  can show; the linked path (G1), where spectators are absorbed into A and B, is.
- **Identity errors are not measurable on Olympic** (no label pins boxer identity over time; STATE 2026-10-24,
  option C). S2 is a proxy. An eye check of the filtered GH078416 overlay (a DEV video) may be reported as an
  anecdote, never as a gate number.
- A PASS says the filter cuts unexplained bursts on the linked path of these two venues' cameras without losing
  boxers. It says nothing about other camera heights, and nothing about whether the remaining bursts are missed
  labels.

## Time-box
- Fri 2026-10-30: this gate committed, floors clicked, margin chosen on dev and frozen. (The brief's "pose
  re-extracted on the development videos" no longer applies: there is no re-extraction.) If that slips, stop.
- Tue 2026-11-03: the held-out verdict, in the PR, `docs/RESULTS.md` and `docs/quality/ring-floor-filter.md`.
- Fri 2026-11-06: results freeze. Nothing after Nov 3 except write-up.

## Order of evidence
1. This file's commit. 2. The commit that sets `FROZEN_MARGIN_PX` (with the dev grid in the quality log).
3. The single held-out run. All three hashes go in the PR.
