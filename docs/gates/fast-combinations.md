# Gate: finding both punches of a fast combination -- written 2026-10-28, before any tuning or held-out number

Branch `feat/fast-combinations`. Committed BEFORE any arm is tuned, before any tuning video is chosen, and
before any held-out video is scored with either arm; the commit hash of this file is the evidence of order.
Nothing below changes after a held-out number exists. A later change is a new, dated section that says why,
and the original text stays.

## Question
The finder (`punchfind.detect`, rule `results/punch_finder_v6.json`: variant relative, threshold 3.62 body
lengths/s, minimum gap 0.4 s, pooled across hands) under-finds jabs. Measured on main:
- end to end on V5+V9 (RESULTS row 5b, #20): jab recall 0.32 (84 of 264), against 0.76 for crosses; on V9 alone
  0 of 27 crosses and 4 of 31 jabs;
- on Brendan's two labelled clips, 40% and 32% of punches follow the previous one within 0.4 s, but only 11% and
  0% follow one by the same hand (#22). The pooled rule allows one detection per 0.4 s whichever hand threw it,
  so a fast jab-cross is one detection;
- per-hand peaks (`detect(..., per_hand=True)`, #22) recover recall and lose precision: own footage recall
  +0.11 / +0.13, precision -0.18 / -0.13; V6 second-half F1 0.489 per hand against 0.571 pooled. The extra
  detections were the OTHER wrist's retraction.

## The two arms (exact definitions)
Both use the pooled rule's signal unchanged: variant `relative` (the classifier's window was measured on it, and
`classify_punches.py` refuses another), the track bridged over dropouts up to `BRIDGE_S`, per-hand validity from
joint confidence.

**Arm 1, per hand + extension gate** (`mode: per_hand_extension`). Each hand's own peaks, minimum gap 0.4 s
within a hand (the V6 rule's gap, fixed). A peak of hand h at frame i is a CANDIDATE only if hand h's wrist is
moving away from its shoulder there:
- d = the shoulder-to-wrist distance in body lengths, from the same smoothed, hole-filled positions the signal
  is computed from (`SMOOTH_S` moving average, holes up to `BRIDGE_S` interpolated, inside one run of
  consecutive valid frames);
- k = max(1, round(W x fps / 2)) frames, W the extension window in seconds;
- eligible iff frames i-k and i+k both lie in the same valid run as i, and d[i+k] - d[i-k] > 0.

A frame whose window leaves the run (missing joints, a pose gap, a run edge) is NOT eligible: missing data never
passes. Ineligible frames are removed BEFORE the minimum-gap suppression (`detect`'s existing `eligible`
semantics), so a retraction cannot suppress the extension peak next to it and then be discarded itself.

**Arm 2, pooled with a cross-hand gap** (`mode: cross_hand_gap`). Peaks of the faster hand's speed, as the
pooled rule, but the minimum gap depends on the hands: 0.4 s between two peaks of the same hand (fixed), G s
between peaks of different hands. Suppression is the pooled rule's: taller peaks first, a shorter one removed
if it is within the gap that applies to the pair. With G = 0.4 s this is the pooled rule exactly (a test).

## Data (no circularity)
- **Tuning videos:** every BoxingVI source video NOT held out by the split on main when tuning runs. Tuning
  waits for the athlete-disjoint split (#38, O55) to merge; the tuning script refuses to run without it
  (`data.PERSON_OF_VIDEO` absent) and refuses any held-out video. Expected from #38: V1, V4, V6, V7, V8, V10.
  Pose is YOLO11-pose on the downloaded source (pinned weights, GPU laptop), at the source's native frame rate.
- **Each video must pass the alignment gate** before it is used (`poseswitch.align`: best offset 0, sharpness
  >= 1.5). A video that fails is excluded and listed, never silently dropped. V4 was checked on 2026-10-28 while
  writing this (offset 0, sharpness 2.24, every 10th of 560 clips): usable. Nothing was scored.
- **V6 labels only hooks and uppercuts** (its straight punches are unlabelled; checked by eye 2026-09-27), so an
  arm that finds more straight punches would be charged false positives there. V6 therefore counts towards the
  per-type recall checks (hooks, uppercuts) and NOT towards the F1 objective.
- **Held-out videos:** `data.HELD_OUT_VIDEOS` on main after #38 merges (expected V2, V3, V5, V9). V2's and V3's
  video links are swapped in the release (ISSUES-LOG O61): V2's labels are scored on the pose of the video
  linked as V3, and V3's on the video linked as V2; each pairing must pass the alignment gate.
- **Brendan's own clips** (IMG_4406, IMG_4395) are a secondary check, reported, with no role in tuning, in
  choosing the arm, or in the verdict.

## Tuned on training only
| Parameter | Grid (fixed now) |
|---|---|
| Threshold, both arms (body lengths/s) | 2.0 to 6.0 in steps of 0.1 (41 values; the V6 rule's 3.62 lies inside) |
| Arm 1 extension window W (s) | 0.05, 0.10, 0.15, 0.20, 0.30 |
| Arm 2 cross-hand gap G (s) | 0.10, 0.12, 0.15, 0.20, 0.25, 0.30 |

Fixed, not tuned: variant relative, within-hand gap 0.4 s, matching tolerance 0.1 s, nearest matching
(`punchfind.match_nearest`, as RESULTS row 5a), detections on unusable labelled rows ignored (as row 5a).

**Choosing within an arm.** Score every grid point on the tuning videos, the whole of each video. Admissible:
jab recall above the pooled baseline's on the same videos, and no type group's recall more than 0.05 below the
baseline's (groups: jab, cross, hooks = lead + rear, uppercuts = lead + rear; V6 included for hooks and
uppercuts). Among admissible points, the highest F1 pooled over the F1 videos (all tuning videos but V6). Ties:
the higher threshold, then the smaller W or G. If no point is admissible, the highest F1 is taken and recorded
as inadmissible on training.

**Choosing the arm (ONE, before any held-out run).** The arm whose chosen point has the higher training F1;
an exact tie goes to arm 2 (fewer moving parts). Recorded in `results/fast_combinations_tune.json` with the
whole grid. The held-out script refuses to run without that file and reads the arm from it; it never takes an
arm as an argument.

**Control, reported only:** the pooled rule with its threshold re-tuned on the same grid and videos. If an arm
beats the baseline only because the threshold moved, the control shows it. It is not a candidate.

## Held-out metrics (one run)
On the held-out videos, the chosen arm against the pooled baseline (`results/punch_finder_v6.json`, unchanged),
same videos, same matching:
1. **F1** of each, and the paired difference arm - baseline, with a 95% percentile interval from a block
   bootstrap: 30 s blocks, resampled with replacement within each video, 2000 draws, seed 0.
2. **Recall** overall and per type group (jab, cross, hooks, uppercuts; all six types also listed), each with
   the same paired interval on the difference.
3. **Precision**, a LOWER bound: labels are incomplete. The blinded label audit on V5+V9 (RESULTS row 5d) found
   19 of 50 far detections were real punches; inside the labelled stretch 19 of 37, outside it 0 of 13.
4. **Found AND named right** (end to end through `classify_punches.py` with the pipeline checkpoint) on V5+V9
   only: the pipeline checkpoint is trained on the old split (V2/V5/V9 held out, V3 trained on; #38 keeps it
   there), so its readings of V2 and V3 are not held-out readings. Paired difference with the same bootstrap.

**Before anything is scored**, the baseline on V5+V9 must reproduce RESULTS rows 5a-5c exactly (precision
0.633, recall 0.585, jab 84 of 264, found and named right 352 of 742). If it does not, the run stops: the
scoring has drifted, and no arm number is written.

## Pass (all three, on held-out)
1. F1(arm) - F1(baseline) > 0, with the 95% interval excluding zero.
2. Jab recall(arm) > jab recall(baseline).
3. For cross, hooks and uppercuts: recall(arm) >= recall(baseline) - 0.05.

Reported, not part of the verdict: found-and-named-right difference (a drop with an interval excluding zero is
stated as a cost of adoption), precision, the control, own footage.

**A failure is a documented negative** (`docs/RESULTS.md` and the vault's `PAPER-future-work.md`), not a reason
to tune further. No second held-out run with other settings.

## Adoption
A pass does not change the default by itself. If Brendan adopts the arm as the pipeline's finder, an ADR goes in
the vault's `Decisions/`, and every RESULTS row downstream of the finder is re-run and listed in the PR.

## Deadlines
Gate committed and code tested by Sat Oct 31 · held-out verdict by Wed Nov 4 · freeze Fri Nov 6.
