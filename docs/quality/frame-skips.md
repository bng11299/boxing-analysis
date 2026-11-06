## `feat/frame-skips` -- 2026-10-09

Branch from `feat/completeness-credited` (b912027, PR #12 untouched).

**The finding (Brendan; confirmed numerically on GH069681).** The Olympic GoPro files have perfectly regular packet timestamps (every
interval 20 ms; `probe` reports no VFR), but the camera drops CAPTURED frames: what is on screen jumps forward in one frame while the
timestamps stay regular, and the wrist-speed rule reads the jump as a fast punch. Of the four unexplained bursts of GH069681 Brendan
classified, the two he called "not a punch" (frames 1287 and 7018, tracks 64 and 1203) sit on such a jump; the two real unlabelled
punches (1965, 13293) do not. Measured as the mean absolute grey difference between consecutive frames (image decimated 4x), divided by the
video's median: 3.46x at frame 1287 and 5.36x at 7019 (a reviewer's earlier inspection read 3.39x and 5.43x), nothing comparable at 1965 or
13293. No exact duplicate frame exists (the minimum change is 0.07 grey levels against a median of 1.18), so it is a skip, not a repeat.

**The rule (`src/boxingvi/skips.py`).** Frame f is a SKIP when `change[f] > 3.0 x` the median of `change` over [f-10, f+10] EXCLUDING f.
The neighbours' median is the local normal: a pan or a crowd surge raises it and is not flagged, an isolated jump is. Frame 0 (NaN) is
never a skip; NaN never enters a median; there is no absolute floor (none was observed to be needed; one would be a tuned constant). On
GH069681 it finds 147 skips (one every 1.9 s) where the first inspection found 145; the small difference is the grey conversion and the
decimation.

**Declaration.** The rule and `SKIP_RATIO = 3.0` come from Brendan's observation and a reviewer's inspection of ONE development video
(GH069681). They were fixed by reading that observation, not tuned to any completeness, precision or score, and no other video was used to
choose them. That they transfer to the other 28 videos is a claim the numbers below test, not one the code makes. (Brendan's four classified rows are the only ground truth about skips used in this section; the rest of the sample he is still
classifying was not used for anything.)

**What was built.**
1. `video.frame_change(path, info=None, step=4)`: one float per frame, NaN for frame 0, decoded once through `iter_frames`; no pixel
   leaves the function (an aggregate per frame only). 2. `skips.find_skips`, `skips.Skips` (change array bound to the source file name,
   the frame count and the video's content fingerprint; `<tracks>/<id>.framechange.npz`, saved atomically, strict loader, no pickle).
3. `olympic_framechange.py --data --tracks [--only] [--overwrite]`. 4. `completeness.cut_at_skips` inside `_scan_segments`, so
   `find_bursts` and `find_bursts_credited` take `skips=`: every segment is cut between f-1 and f for each skip f inside it (also a skip on
   a frame the segment is missing: the hole the bridge would have interpolated across), BEFORE bridging, smoothing or differencing;
   `skips=None` is the old output exactly (every existing check unchanged). 5. `olympic_completeness.py --skips`
   (both modes), `--skips-measure` (loads the same files, reports the same numbers, does NOT cut: the "before"), `--skipcheck SAMPLE.csv
   --skipcheck-out OUT.csv`. New default output names (`..._skips.json`, `..._skips_sample.csv`; `..._skipsmeasure...`); a missing,
   malformed or mismatched framechange file stops the whole run with exit 2 and writes nothing.

**Deviations from the brief (and why).** (a) `--skips-measure` and `--skipcheck` are added: "before and after" needs the near-skip counts
of the b912027 behaviour, and the existing JSONs do not contain them; the measure run reproduces every pre-existing number (below), so it
is also a regression check on the real data. (b) The framechange file stores the fingerprint of the VIDEO FILE it decoded (always
present), and the run checks it against the video and, when the table has one, against the table's `video_fp`; the brief said "the table's
video_fp if present", which is the weaker half of that. (c) A synthetic "dropped frame" in the tests is a dropped STRETCH (the pattern
jumps 6 px where it moves 1 px a frame): a one-frame drop of a 1 px/frame pattern is a 2x change and, correctly, not a skip; the real
camera's jumps are 3-5x. (d) `still scene -> no skip` is tested on a flat picture: a static GRATING re-encoded by H.264 at crf 18 is not
bit-still (changes of 0.001-0.006, median 0.0013) and the rule, having no floor, flags its frame 1 (0.0059). Real footage is orders of
magnitude above that (smallest per-video median change 1.04), so it did not occur there, but it is a property of the rule (residual risk 4).

**Tests (all suites):** `test_frame_skips.py` 155 (new), `test_completeness_credited.py` 95, `test_linking.py` 266, `test_tracks.py` 79,
`test_olympic.py` 33, `test_hands.py` 38, `test_labels.py` 50, `test_ingest.py` 124, `test_env.py` 36, `test_pipeline.py` 260; every
pre-existing count and result is unchanged. Negative tests at the trust boundaries: a skip list that is unsorted, repeated, at frame 0, past
the video, float or 2-D; a change array that is integer, 2-D, negative, infinite, NaN anywhere but frame 0; a framechange file with a wrong
source, length or fingerprint, a missing field, a pickled array, garbage, a truncated zip, a bad ratio / window / step; both CLIs'
refusals leave no output behind.

**Mutation check (change applied in a scratch copy; `test_frame_skips.py` must fail)**
| # | Mutation | Result | First failing checks |
|---|---|---|---|
| M1 | skips ignored in `_scan_segments` | caught, 8 fail | the jump is cut out; two segments; `find_bursts` cuts; `--skips` leaves one unexplained |
| M2 | the window includes f itself | caught, 1 | 4.6 between ten 1.0s and ten 2.0s is a skip |
| M3 | ratio test replaced by a fixed absolute threshold | caught, 6 | 30.1 over 10 is a skip and 25 is not; a pan is not flagged; a ramp; a still stretch |
| M4a | `Skips.verify` checks nothing | caught, 11 | wrong name / length / fingerprint all accepted |
| M4b | the CLI does not compare the file's fingerprint to the video | caught, 5 | a file measured on another file (the later failures are the first's leftover output) |
| M5 | `>=` instead of `>` | caught, 3 | exactly 3.0x is not a skip; zeros in a still stretch |
| M6 | a skip cuts only at a row, not at a hole | caught, 1 | a skip on a missing frame cuts there |
| M7 | `--skips-measure` also cuts | caught, 3 | same bursts as without skips; the before numbers |
| M8 | skipcheck ignores the hand | **missed at first** (151 checks), caught after adding 9 rows, 1 | other hand is false |
| M9 | base rate counts all video frames, not exposure | caught, 4 | 10 of 150 exposure frames |
| M10 | NaN not ignored in the medians | caught, 26 | 2.99x / 3.01x boundary and 25 more |
| M11 | the near-skip window is one-sided | caught, 7 | counts, base rate, no wrap at the start |
| M12 | skipcheck ignores the track id | **missed at first**, caught after the same 9 rows, 1 | other track is false |

**Real data.** `olympic_framechange.py` on all 29 videos: 312,774 frames (104.3 video minutes), **3,367 skips = 32.3 per video minute**
pooled (one every 1.9 s), per video 22.1 (GH089681) to 50.4 (GH128416) per minute, median 32.8; the smallest per-video median change is
1.04 grey levels. Wall time: 5,560 s summed over videos (1.5 h of CPU-bound decoding at ~10-20 ms a frame, measured with up to four of
these running at once beside another job on the same machine; GH069681 alone, 13,800 frames, took 147 s). Two videos were measured twice
by two of the parallel processes (a race on the output name; Iteration 3 corrects the claim that this was harmless: both writers happened to produce the same content and neither failed, but the shared temporary name was a bug, now fixed).

| Quantity (`--credited`) | min 1, b912027 | min 1, `--skips` | min 2, b912027 | min 2, `--skips` |
|---|---|---|---|---|
| bursts | 10,077 | 9,685 | 6,227 | 6,029 |
| unexplained | 7,318 | 6,903 | 4,342 | 4,117 |
| explained | 2,759 | 2,782 | 1,885 | 1,912 |
| exposure minutes (summed over fragments) | 161.36 | 161.21 | 96.05 | 95.98 |
| unexplained per exposure minute | 45.35 | 42.82 | 45.21 | 42.90 |
| bursts per exposure minute | 62.45 | 60.08 | 64.83 | 62.82 |
| own labels per exposure minute | 14.92 | 14.93 | 18.13 | 18.14 |
| segments (short) | 2,758 (260) | 6,689 (403) | 1,380 (116) | 3,667 (191) |
| `possible_duplicates` | 177 | 145 | 66 | 56 |
| unexplained within 2 frames of a skip | 1,113 / 7,318 = 15.2% | 52 / 6,903 = 0.8% | 624 / 4,342 = 14.4% | 35 / 4,117 = 0.9% |
| explained within 2 frames of a skip | 209 / 2,759 = 7.6% | 32 / 2,782 = 1.2% | 137 / 1,885 = 7.3% | 26 / 1,912 = 1.4% |
| base rate (exposure frames within 2 of a skip) | 4.8% (23,032 / 484,084) | 4.7% (22,631 / 483,614) | 4.6% (13,163 / 288,154) | 4.5% (12,962 / 287,935) |
| unexplained share / base rate | 3.20x | 0.16x | 3.15x | 0.19x |
| explained share / base rate | 1.59x | 0.25x | 1.59x | 0.30x |

The "b912027" columns are from `--skips-measure` and equal the pre-existing `olympic_completeness_credited.json` / `_min2.json` in
every burst, label, exposure and rate (compared programmatically): no burst was lost to a re-implementation. The pooled near-skip
excess before the fix is real and general (15% of unexplained bursts against a 5% base rate, in 29 videos, not only GH069681's 14%),
and within 2 frames of a skip it is gone after.

**Three things the table says that the brief did not expect.** (1) The excess did not just vanish, it reversed: 0.16x. (2) The
unexplained rate fell by only 5.6% (45.35 to 42.82 per exposure minute), although 1,061 unexplained bursts were removed near skips,
because the cut ADDS bursts a little further out. Bursts per 1,000 exposure frames by distance to the nearest skip (min 1, unexplained /
explained), before then after: 0-2 frames 48.3 / 9.1 then 2.3 / 1.4; 3-5 frames 12.9 / 4.0 then 22.8 / 7.1; 6-10 frames 11.4 / 4.4 then
16.7 / 6.1; 11 or more frames 13.7 / 5.8 then 14.1 / 5.9. So the far field is unchanged and a ring of extra bursts now sits 3-10 frames
from each skip (1.6x and 1.2x the far-field rate). The likely cause, not isolated: the cut pieces are about 1.9 s long, the smoothing is
edge-normalised (less smoothing at a piece's end, so a noisier speed), the body scale is re-estimated per piece, and `find_peaks` never
reports a piece's first or last sample. (3) Explained went UP (+23 at min 1, +27 at min 2): the one-to-one matching ran on a changed burst set.
The four 2-frame-window numbers (0.8% / 1.2%) therefore overstate what the cut achieved: a real fix would also have to stop the edge ring
(for example by giving a cut piece the scale of the fragment and ignoring peaks inside the smoothing half-window of a cut), which this
branch did not do because it would be a second, unevaluated tuning.

**Brendan's four GH069681 rows (min 2).** 1287 / track 64: **gone** (no burst within 10 frames of it remains; skip at that frame).
7018 / track 1203: **gone** (skip one frame away). 1965 / track 253: **still an unexplained burst**. 13293 / track 2268: **still an
unexplained burst**. Exactly the expected outcome.

**The sample** (`results/olympic_completeness_credited_min2_sample_skipcheck.csv`, the 40 existing rows with
`still_unexplained_with_skips`; the original CSV was only read, its hash is unchanged): 27 true, 13 false. Eight of the 13 are within
2 frames of a skip (distance 0, 0, 1, 1, 1, 1, 1, 2 frames) and all of those vanished, as intended. Five are not: they are 5, 6, 9, 13
and 174 frames from the nearest skip. Four of them lost their burst outright (no burst of that track within 10 frames remains); three
of the four had peaks of only 3.67 to 4.48 in the old run against the threshold 3.62, which is consistent with the per-piece scale effect
above, not proven. The fifth (GH178416 frame 14485) still has its burst, three frames earlier (14482), so the strict 2-frame
test calls it false. The 27 rows kept had skips 4 to 444 frames away. This is also why `still_unexplained_with_skips = false` does not
mean "was caused by a skip": for 5 of the 13 it means "changed".

**Residual risks.** (1) A real punch exactly across a skip is now split in two (a punch at 150 with a skip 2 frames away came out as
one or two bursts within 140..160 in the synthetic test; on the real data it is the edge ring above); the cut cannot tell a real
jump from a skip-made one when a real punch coincides with a dropped frame. (2) Skips at a scene cut or a replay edit would be cut as
well (harmless: a cut only removes differencing across it) and counted. (3) The rule is GoPro-specific in the sense that it was seen
on this camera's drops; another encoder's frame-doubling would not trace as a spike. (4) No absolute floor: on a stretch where the
picture hardly changes at all (a frozen frame, a black leader) codec noise can be flagged; none was seen in the 29 videos, and the
consequence of a false skip is a cut, not a burst. (5) The `skipcheck` window is exactly 2 frames and the track, hand and video: a
burst that the cut shifted by 3 frames counts as gone. (6) The skip signal is one aggregate per frame on a 4x-decimated grey image,
so a drop during uniform motion of a small part of the picture (a dropped frame that only moved the boxers a little) is under the
rule's ratio and is not found. (7) The fingerprint is still the first and last 4 MiB, not every byte.


### Iteration 2 -- masking replaces cutting (2026-10-09)

**Why the cut was replaced.** Iteration 1 cut every segment at each skip. The table above shows what that cost: a ring of extra
unexplained bursts 3-10 frames from each skip (1.61x and 1.18x the far-field rate at 3-5 and 6-10 frames), and 5 of the 13 sample rows
that dropped out were not near a skip. A cut makes new segment edges: edge-normalised smoothing, a body scale re-estimated on a piece
about 1.9 s long, and `find_peaks` never reporting a piece's first or last sample. Masking keeps the segments, the body scale and the
smoothing exactly as they are without skips and only removes what a jump can have contaminated. `cut_at_skips` is deleted (nothing uses
it) together with its tests, which tested only that function; every other Iteration-1 check stands, and the counts that changed are
the ones that asserted cut-specific numbers (2 segments, 400 exposure frames), replaced by the masked ones.

**The mask (`completeness.skip_mask_half`, `skip_mask`, `_scan_segments`).** After `wrist_signal`, per hand:
1. *Signal reach, derived.* `wrist_signal` smooths with a centred moving average of `punchfind.smooth_window(fps, smooth_s)` frames
   (5 at 50 fps and 0.1 s; half-width h = 2; one definition, now shared by the signal and the mask) and then takes `np.gradient`, which at
   frame j reads the smoothed values at j-1 and j+1. A step between f-1 and f enters the smoothed value at i when i in [f-h, f+h-1]; the
   gradient at j is contaminated when j in [f-h-1, f+h] = [f-3, f+2]. The mask is symmetric, h + 1 = 3 either side of f
   (`skip_mask_half = smooth_window // 2 + 1`, computed from `smooth_s` and the frame rate, never written as 3). The signal there is set to 0.
2. *Wider where interpolation spreads the jump.* If f lies in a table hole that `punchfind.bridge` interpolated across, the jump
   becomes a ramp over the whole hole and the mask covers the hole plus the reach beyond each end; if the zone meets one hand's
   interpolated hole (a low-confidence stretch of at most the bridge length), that hand's mask grows over it too.
3. *Exposure, one frame wider (a finding, not in the brief).* The frames within `skip_mask_half + 1` of a skip leave the exposure and
   cannot be a burst; the extra frame keeps its true signal. With the mask at the derived reach and the same exposure, the first
   unmasked frames carried 1.56x (min 1) and 1.63x (min 2) the far-field unexplained rate, and widening the mask to 4 and to 5 frames only moved
   the excess to the new edge (1.91x, then 1.30x at 6-10), so it was the mask's own edge, not a contamination reach: `detect` calls
   frame j a peak by comparing it with j-1 and j+1, so next to a zeroed frame the flank of a peak that is hidden under the mask is
   "higher than a zero" and the punch came back as a burst beside the mask. Keeping the extra frame's true signal and refusing it as a
   peak removes that (and takes the frame out of numerator and denominator alike). Zeroing the signal itself still matters: a giant
   skip-made peak would otherwise suppress a real punch within the 0.4 s minimum gap (test: a punch 12 frames after a jump).
   The burst-in-exposure assertion still holds (a burst outside the exposure is a RuntimeError). The design with the signal reach only
   is reported beside the final one below.

**Gate (unexplained bursts per 1,000 exposure frames by distance to the nearest skip, each of 3-5, 6-10, 11-25 within +-25% of the
>25 rate; 0-2 must be empty).** Final design: **PASS at min 1 (1.14x, 1.04x, 0.99x) and at min 2 (1.17x, 1.04x, 1.02x); 0-2 is empty.**
The signal-reach-only mask FAILS the 3-5 bin (1.56x, 1.63x), as does the cut (1.61x, 1.66x). With no skips the 0-2 bin is 3.36x and 3.32x. The
3-5 bin of the final design holds a single offset (5 frames) and ~150 bursts at min 1, 86 at min 2: a Poisson standard error of about
8% and 11% of the bin's rate, so 1.14x and 1.17x are 1-2 standard errors above 1.0, not "a few percent".
**Declared (reviewer, review):** the gate was used to CHOOSE between designs (cut, signal-reach mask, mask plus one exposure frame) on the
same 29 videos it then reports, so the PASS is not independent evidence. The final design is physically motivated (only the frame
`detect` can falsely call a peak is added, nothing was widened to fit), but its confirmation is Brendan's classification of the sample
rows, not this gate.

**The bin tables, all three designs side by side**

Unexplained bursts per 1,000 exposure frames by distance to the nearest skip, `--credited` min 1 (burst count / exposure frames; x = ratio to the >25 rate)

| Distance (frames) | no skips | cut (Iteration 1) | mask, signal reach only | mask, final |
|---|---|---|---|---|
| 0-2 | 48.3 (1113/23,032) x3.36 | 2.3 (52/22,631) x0.16 | empty (masked) | empty (masked) |
| 3-5 | 12.9 (358/27,783) x0.90 | 22.8 (633/27,733) x1.61 | 22.4 (410/18,296) x1.56 | 16.4 (149/9,091) x1.14 |
| 6-10 | 11.4 (519/45,600) x0.79 | 16.7 (762/45,590) x1.18 | 15.4 (701/45,519) x1.07 | 14.9 (674/45,372) x1.04 |
| 11-25 | 12.2 (1352/110,943) x0.85 | 13.8 (1528/110,934) x0.97 | 14.3 (1584/110,942) x1.00 | 14.2 (1578/110,927) x0.99 |
| >25 | 14.4 (3976/276,726) x1.00 | 14.2 (3928/276,726) x1.00 | 14.3 (3965/276,726) x1.00 | 14.3 (3965/276,726) x1.00 |

Explained bursts per 1,000 exposure frames by distance to the nearest skip, `--credited` min 1 (burst count / exposure frames; x = ratio to the >25 rate)

| Distance (frames) | no skips | cut (Iteration 1) | mask, signal reach only | mask, final |
|---|---|---|---|---|
| 0-2 | 9.1 (209/23,032) x1.51 | 1.4 (32/22,631) x0.24 | empty (masked) | empty (masked) |
| 3-5 | 4.0 (112/27,783) x0.67 | 7.1 (197/27,733) x1.19 | 7.3 (133/18,296) x1.22 | 4.5 (41/9,091) x0.76 |
| 6-10 | 4.4 (199/45,600) x0.73 | 6.1 (278/45,590) x1.03 | 6.2 (282/45,519) x1.04 | 6.0 (273/45,372) x1.01 |
| 11-25 | 5.2 (578/110,943) x0.87 | 5.7 (630/110,934) x0.96 | 5.7 (636/110,942) x0.96 | 5.8 (638/110,927) x0.96 |
| >25 | 6.0 (1661/276,726) x1.00 | 5.9 (1645/276,726) x1.00 | 6.0 (1651/276,726) x1.00 | 6.0 (1651/276,726) x1.00 |

Unexplained bursts per 1,000 exposure frames by distance to the nearest skip, `--credited` min 2 (burst count / exposure frames; x = ratio to the >25 rate)

| Distance (frames) | no skips | cut (Iteration 1) | mask, signal reach only | mask, final |
|---|---|---|---|---|
| 0-2 | 47.4 (624/13,163) x3.32 | 2.7 (35/12,962) x0.19 | empty (masked) | empty (masked) |
| 3-5 | 13.7 (216/15,815) x0.96 | 23.4 (370/15,797) x1.66 | 23.2 (242/10,411) x1.63 | 16.7 (86/5,163) x1.17 |
| 6-10 | 11.2 (291/25,936) x0.79 | 16.7 (432/25,936) x1.18 | 15.3 (396/25,895) x1.08 | 14.8 (382/25,816) x1.04 |
| 11-25 | 12.4 (790/63,683) x0.87 | 13.9 (886/63,683) x0.99 | 14.5 (926/63,682) x1.02 | 14.5 (923/63,679) x1.02 |
| >25 | 14.3 (2421/169,557) x1.00 | 14.1 (2394/169,557) x1.00 | 14.2 (2412/169,557) x1.00 | 14.2 (2412/169,557) x1.00 |

Explained bursts per 1,000 exposure frames by distance to the nearest skip, `--credited` min 2 (burst count / exposure frames; x = ratio to the >25 rate)

| Distance (frames) | no skips | cut (Iteration 1) | mask, signal reach only | mask, final |
|---|---|---|---|---|
| 0-2 | 10.4 (137/13,163) x1.49 | 2.0 (26/12,962) x0.29 | empty (masked) | empty (masked) |
| 3-5 | 4.5 (71/15,815) x0.64 | 7.7 (122/15,797) x1.12 | 7.2 (75/10,411) x1.04 | 4.6 (24/5,163) x0.67 |
| 6-10 | 4.9 (127/25,936) x0.70 | 7.0 (182/25,936) x1.01 | 7.1 (185/25,895) x1.03 | 6.9 (177/25,816) x0.99 |
| 11-25 | 5.8 (367/63,683) x0.83 | 6.4 (408/63,683) x0.93 | 6.4 (409/63,682) x0.93 | 6.4 (408/63,679) x0.92 |
| >25 | 7.0 (1183/169,557) x1.00 | 6.9 (1174/169,557) x1.00 | 6.9 (1177/169,557) x1.00 | 6.9 (1177/169,557) x1.00 |

("Mask, signal reach only" = zero over the derived reach and remove only those frames from the exposure; "mask, final" = the design above.)
The unexplained 0-2 bin of "no skips" and "cut" is the skip-made excess of Iteration 1; the explained rows are not gated. The
explained 3-5 cell of the final design is low (0.76x and 0.67x) on 41 and 24 bursts.

**Before / after (`--credited`; "no skips" = b912027, reproduced by `--skips-measure`).**
| Quantity | min 1 no skips | min 1 cut | min 1 mask (final) | min 2 no skips | min 2 cut | min 2 mask (final) |
|---|---|---|---|---|---|---|
| bursts | 10,077 | 9,685 | 8,969 | 6,227 | 6,029 | 5,589 |
| unexplained | 7,318 | 6,903 | 6,366 | 4,342 | 4,117 | 3,803 |
| explained | 2,759 | 2,782 | 2,603 | 1,885 | 1,912 | 1,786 |
| exposure minutes | 161.36 | 161.21 | 147.37 | 96.05 | 95.98 | 88.07 |
| **unexplained per exposure minute** | **45.35** | **42.82** | **43.20** | **45.21** | **42.90** | **43.18** |
| bursts per exposure minute | 62.45 | 60.08 | 60.86 | 64.83 | 62.82 | 63.46 |
| own labels per exposure minute | 14.92 | 14.93 | 15.63 | 18.13 | 18.14 | 18.95 |
| own labels outside the exposure | 16 | 16 | 120 | 3 | 3 | 75 |
| `possible_duplicates` | 177 | 145 | 123 | 66 | 56 | 52 |
| segments (unchanged by a mask) | 2,758 | 6,689 | 2,758 | 1,380 | 3,667 | 1,380 |

The unexplained rate falls 4.7% (min 1) and 4.5% (min 2) against no skips. The bursts within 2 frames of a skip were 15% of the
unexplained ones, and the rate falls by less than that because peaks that a skip-made peak used to hide through the 0.4 s minimum gap
(the no-skip bins 3-25 sit at 0.79-0.90 of the far field) are visible again, and because the mask removes exposure as well as bursts.

**What the mask removes.** 13.99 exposure minutes of 161.36 (8.7%) at min 1 and 7.98 of 96.05 (8.3%) at min 2. A real punch whose
span lies wholly in the removed frames is invisible: 104 of the 2,407 own-labelled punches at min 1 (4.3%) and 72 of 1,741 at min 2
(4.1%) are now "outside the exposure" (16 and 3 before), which is what raises "own labels per exposure minute". The completeness
question is asked only of the frames that are left; it says nothing about the removed 8-9%.

**Brendan's four GH069681 rows (min 2), mask run.** 1287 / track 64: **gone** (skip at that frame; the frame is out of the exposure).
7018 / track 1203: **gone** (skip 1 frame away; out of the exposure). 1965 / track 253: **still an unexplained burst** (38 frames
from a skip). 13293 / track 2268: **still an unexplained burst** (14 frames from a skip). As expected.

**The sample, mask run** (`results/olympic_completeness_credited_min2_sample_skipcheck.csv`, overwritten with the mask run; the
original sample CSV is untouched, its hash unchanged): **29 true, 11 false** (the cut run: 27 and 13). All 29 true rows are 5 or more
frames from a skip and in the exposure. Ten of the 11 false rows are inside the removed frames of a skip, at 0, 0, 1, 1, 1, 1, 1, 2, 4
and 4 frames, so their frame is out of the exposure and no burst can exist there. The eleventh (GH199681 frame 9937, 5 frames from a
skip, in the exposure) lost its burst for a different, legitimate reason: the skip at 9913 used to put a giant peak at 9912 that
suppressed everything within the 0.4 s minimum gap, and with it hidden a real 11.66 peak appears at 9922, 15 frames before 9937, and
now suppresses the weaker 6.28 one. Of the five rows that dropped out of the cut run without being near a skip, four are back (true).

**Tests.** `test_frame_skips.py` 189 (155 in Iteration 1), the other nine suites unchanged: `test_completeness_credited.py` 95,
`test_linking.py` 266, `test_pipeline.py` 260, `test_ingest.py` 124, `test_tracks.py` 79, `test_labels.py` 50, `test_hands.py` 38,
`test_env.py` 36, `test_olympic.py` 33. New: the reach follows `smooth_s` and the frame rate (0.1 s/50 fps -> 3; 0.2 s -> 6; 25 fps -> 2;
100 fps -> 6; a grid of rates and windows equals `smooth_window // 2 + 1`); `skip_mask` frames for a plain skip, both ends of the track, a
bridged hole, one hand's hole; a one-frame jump at a skip gives no burst and 391 of 400 exposure frames; a real punch peaking 5 frames from
a skip is found with the same frame and peak speed, one at 4, 3 or 1 frame is invisible, out of the exposure and does not come back
beside the removed range, on both sides; `_scan_segments` with `smooth_s` 0.1 / 0.2 / 0.02 removes 9 / 15 / 5 frames and the 0.2 s
signal has a lower peak than the 0.1 s one; a punch 12 frames after a jump is suppressed by the jump's peak without skips and found with
them; `skips=None`, an empty list and a skip past the track equal the old output on every scene.

**Mutation check (scratch copy; `test_frame_skips.py` must fail)**
| # | Mutation | Result |
|---|---|---|
| A | mask applied to the signal but not to the exposure | caught, 12 fail |
| B | `skip_mask_half` fixed at 0 | caught, 10 |
| C | mask one-sided (frames from f on only) | caught, 24 |
| D | mask one-sided (frames up to f only) | caught, 23 |
| E | skips ignored in `_scan_segments` | caught, 17 |
| F | `skip_mask_half` fixed at 3 (ignores `smooth_s` and fps) | caught, 4 |
| G | mask ignores a bridged hole | caught, 3 |
| H | mask ignores one hand's hole | caught, 1 |
| I | `smooth_s` not passed on to `wrist_signal` | caught, 1 (it passed the first 178 checks; the `smooth_s` signal-peak check was added) |
| J | signal not zeroed, exposure only | **missed** by the first 187 checks, caught by 1 after the punch-12-frames-after-a-jump check was added |
| K | the peak test's extra frame dropped (exposure = signal mask) | caught, 10 |
| L | peaks beside the mask kept (no eligibility filter) | caught by the in-code assertion: RuntimeError "a burst ... outside the exposure" |

An earlier version of C (changing only the first line of the mask) survived because it is an equivalent mutant: the bridged-hole line
`min(lo, p+1-half)` re-derives the left reach whenever f lies inside the track. The mutation was moved to the final slice, where it bites.
The Iteration-1 mutations on code this iteration did not change (M2, M3, M4a, M4b, M5, M7-M12) were re-run against the 189 checks: all caught (1, 6, 11, 5, 3, 3, 1, 5, 26, 6, 1 failures).

**Residual risks.** (1) The mask removes 8-9% of the exposure, and real punches inside it are invisible (about 4% of the labelled ones);
the completeness question is not asked there. (2) The reach is derived for a single-frame step; a real drop of several frames at once
is still one step in the position, but a pose estimator that is unsettled for a few frames after the jump (a tracker re-acquiring) is
not covered, and would show as a rate excess in the 3-10 bins; the final design shows none (1.14x, 1.04x). (3) The extra exposure frame
is justified by measurement on this data (the edge excess at 3 -> 4 -> 5 frames); the same rule on another detector would need the same
check. (4) A false skip (a real, sudden body movement that the image-change rule flags) now hides a real punch instead of cutting its
segment; the rule's 3x ratio is one observation's, so the number of real punches lost to false skips is unmeasured. (5) Peaks
suppressed by the 0.4 s minimum gap in the no-skip run reappear (as for GH199681 9937 above), so some "dropped" rows are
reshuffles, not skip artefacts. (6) The Iteration-1 residual risks 2-7 stand.


### Iteration 3 -- Astra 6 review of `feat/frame-skips` (2026-10-09; verdict FIX FIRST: 3 blocking, 3 smaller)

| # | Finding | Fix | Evidence |
|---|---|---|---|
| B1 blocking | `detect()` applied the 0.4 s minimum gap BEFORE the eligibility filter: an ineligible peak just outside the zeroed zone (a flank at 204 for a skip at 200) could suppress a real, eligible one within 20 frames and then be discarded itself, losing a burst while its exposure stayed (understating the rate). The burst-in-exposure assertion cannot see a missing burst | `punchfind.detect(..., eligible=None)`: ineligible frames keep their true signal as neighbours in the peak test but are not candidates; only the eligible local maxima compete in the minimum-gap suppression (the same scipy suppression, run on their heights). `_scan_segments` passes `~gone` and no longer filters afterwards. `eligible=None` is the original line, untouched | Regression: skip at 200, a strong punch hidden under the mask whose return lobe peaks at the ineligible 204, a weaker real punch at 212: without skips bursts at 62 and 204 only (212 suppressed), with skips 62 and 215 (the weak punch's burst), in the exposure. The all-eligible path equals the plain call on the scene and on 200 random signals with plateaus and ties; marking only 204 ineligible removes the burst there and nothing else |
| B2 blocking | A framechange file bound to the right video but measured with pixel step 8 passed (only ratio and window were compared) and the report said step 4 | `load_skips` requires ratio, half window AND pixel step to equal this code's, naming all three in the refusal | step 8 and half window 5 refused (exit 2, nothing written), beside the existing ratio 2.0 |
| B3 blocking | A skipcheck sample row naming frame 999999 of a processed 1,500-frame video was reported as an ordinary "false" | `skipcheck_rows` takes each processed video's length and requires `0 <= frame < length`, and that the video was processed; the run stops before any output | frames 400, 999999 and 10^12 of a 400-frame video refused; 0 and 399 accepted; the CLI with a row at 999999 stops with no output file |
| N1 | `Skips.save` used one shared `<name>.part`: concurrent writers could truncate each other's temporary file or fail at the rename; the "harmless" claim in Iteration 1 was unjustified | a `NamedTemporaryFile` of its own in the destination directory, then `os.replace`, retried a few times on the `PermissionError` Windows raises when two writers replace one destination in the same instant | eight threads saving one path 15 times each: no failure, the file is one writer's complete content, nothing left behind. The shared name fails this test (a PermissionError). Correction: in the Iteration-1 run two videos were measured by two processes at once and neither failed, but that was luck, not a property |
| N2 | The distance-bin computation behind the gate table was not in the repo | `completeness.distance_bins` / `bin_rates` / `pool_bins`; `--skips` and `--skips-measure` write `skip_bins` per video and pooled into the json and print the per-bin line with the gate verdict | exact bin assignment and exposure counts on a planted video, nearest of two skips, the gate at exactly +-25% and each way of failing; both CLI modes end to end. `--skips-measure` reproduces the "no skips" columns of every earlier table to the frame (checked: same counts as the script that made them) |
| N3 | The pooling test used identical rows and could not tell pooling from averaging per-video shares | two UNEQUAL videos | pooled 5 of 33 (15.2%) and base rate 30 of 1,650 (1.8%); the mean of the per-video shares would be 38.3% and 4.0% |

**`skips=None` is bit-for-bit unchanged: how it was verified.** (1) `detect` with `eligible=None` is the original `find_peaks(..., height,
distance)` line; the new branch is only taken when an eligibility array is passed, and `_scan_segments` passes one only when skips
exist. (2) Every pre-existing check passes unchanged (counts below). (3) On the real data, `olympic_completeness.py --credited` and
`--credited --min-punches 2` WITHOUT skips were re-run to scratch paths and compared with the existing
`olympic_completeness_credited.json` / `_min2.json`: byte-identical except the one line that records the output file name; the min-1
sample csv is byte-identical to the existing one. (4) `--skips-measure` (bursts and exposure from the unmasked path) reproduces the
earlier "no skips" bin counts exactly.

**Real data after the fixes (`--credited`, mask run).** B1 gives back bursts the old order had lost, so more bursts, a slightly
higher rate and a mildly higher near-skip rate than Iteration 2 reported:
| Quantity | min 1, Iteration 2 | min 1, Iteration 3 | min 2, Iteration 2 | min 2, Iteration 3 |
|---|---|---|---|---|
| bursts | 8,969 | 9,157 | 5,589 | 5,700 |
| unexplained | 6,366 | 6,502 | 3,803 | 3,881 |
| explained | 2,603 | 2,655 | 1,786 | 1,819 |
| exposure minutes | 147.37 | 147.37 | 88.07 | 88.07 |
| **unexplained per exposure minute** | **43.20** | **44.12** | **43.18** | **44.07** |
| (no skips, b912027) | 45.35 | 45.35 | 45.21 | 45.21 |
| bursts per exposure minute | 60.86 | 62.14 | 63.46 | 64.72 |
| `possible_duplicates` | 123 | 127 | 52 | 54 |

The unexplained rate now falls 2.7% (min 1) and 2.5% (min 2) from the no-skip rate. The exposure is unchanged (B1 changes which peaks
survive, not which frames count).

**The bin tables, re-run from the repo**

Unexplained bursts per 1,000 exposure frames by distance to the nearest skip, `--credited` min 1 (count / exposure frames; x = ratio to the >25 rate)

| Distance (frames) | no skips (`--skips-measure`) | mask, Iteration 2 | mask, Iteration 3 (B1 fixed) |
|---|---|---|---|
| 0-2 | 48.3 (1113/23,032) x3.36 | empty (masked) | empty (masked) |
| 3-5 | 12.9 (358/27,783) x0.90 | 16.4 (149/9,091) x1.14 | 16.6 (151/9,091) x1.16 |
| 6-10 | 11.4 (519/45,600) x0.79 | 14.9 (674/45,372) x1.04 | 16.2 (736/45,372) x1.13 |
| 11-25 | 12.2 (1352/110,943) x0.85 | 14.2 (1578/110,927) x0.99 | 14.9 (1657/110,927) x1.04 |
| >25 | 14.4 (3976/276,726) x1.00 | 14.3 (3965/276,726) x1.00 | 14.3 (3958/276,726) x1.00 |

Explained bursts per 1,000 exposure frames by distance to the nearest skip, `--credited` min 1 (count / exposure frames; x = ratio to the >25 rate)

| Distance (frames) | no skips (`--skips-measure`) | mask, Iteration 2 | mask, Iteration 3 (B1 fixed) |
|---|---|---|---|
| 0-2 | 9.1 (209/23,032) x1.51 | empty (masked) | empty (masked) |
| 3-5 | 4.0 (112/27,783) x0.67 | 4.5 (41/9,091) x0.76 | 4.5 (41/9,091) x0.76 |
| 6-10 | 4.4 (199/45,600) x0.73 | 6.0 (273/45,372) x1.01 | 6.7 (306/45,372) x1.13 |
| 11-25 | 5.2 (578/110,943) x0.87 | 5.8 (638/110,927) x0.96 | 5.9 (657/110,927) x0.99 |
| >25 | 6.0 (1661/276,726) x1.00 | 6.0 (1651/276,726) x1.00 | 6.0 (1651/276,726) x1.00 |

Unexplained bursts per 1,000 exposure frames by distance to the nearest skip, `--credited` min 2 (count / exposure frames; x = ratio to the >25 rate)

| Distance (frames) | no skips (`--skips-measure`) | mask, Iteration 2 | mask, Iteration 3 (B1 fixed) |
|---|---|---|---|
| 0-2 | 47.4 (624/13,163) x3.32 | empty (masked) | empty (masked) |
| 3-5 | 13.7 (216/15,815) x0.96 | 16.7 (86/5,163) x1.17 | 16.9 (87/5,163) x1.19 |
| 6-10 | 11.2 (291/25,936) x0.79 | 14.8 (382/25,816) x1.04 | 16.3 (422/25,816) x1.15 |
| 11-25 | 12.4 (790/63,683) x0.87 | 14.5 (923/63,679) x1.02 | 15.2 (966/63,679) x1.07 |
| >25 | 14.3 (2421/169,557) x1.00 | 14.2 (2412/169,557) x1.00 | 14.2 (2406/169,557) x1.00 |

Explained bursts per 1,000 exposure frames by distance to the nearest skip, `--credited` min 2 (count / exposure frames; x = ratio to the >25 rate)

| Distance (frames) | no skips (`--skips-measure`) | mask, Iteration 2 | mask, Iteration 3 (B1 fixed) |
|---|---|---|---|
| 0-2 | 10.4 (137/13,163) x1.49 | empty (masked) | empty (masked) |
| 3-5 | 4.5 (71/15,815) x0.64 | 4.6 (24/5,163) x0.67 | 4.6 (24/5,163) x0.67 |
| 6-10 | 4.9 (127/25,936) x0.70 | 6.9 (177/25,816) x0.99 | 7.5 (194/25,816) x1.08 |
| 11-25 | 5.8 (367/63,683) x0.83 | 6.4 (408/63,679) x0.92 | 6.7 (425/63,679) x0.96 |
| >25 | 7.0 (1183/169,557) x1.00 | 6.9 (1177/169,557) x1.00 | 6.9 (1176/169,557) x1.00 |


**The gate, re-run from the repo (`skip_bins` in the json; the gate chose the design, so a PASS is a check on it, not independent
evidence).** Unexplained, ratio to the >25 rate in 3-5 / 6-10 / 11-25: min 1 **1.16 / 1.13 / 1.04 PASS** (Iteration 2: 1.14 / 1.04 /
0.99), min 2 **1.19 / 1.15 / 1.07 PASS** (1.17 / 1.04 / 1.02); 0-2 empty. All within +-25%, but the 6-10 bin moved up by 9-11 points: part
of what Iteration 2 reported as agreement with the far field was the B1 loss. The explained bursts at 6-10 are now 1.13x (min 1) and
1.08x (min 2) as well. The 3-5 bin holds one offset (5 frames) and about 150 (min 1) and 87 (min 2) bursts.

**Brendan's four GH069681 rows (min 2).** Unchanged: 1287 / 64 gone, 7018 / 1203 gone, 1965 / 253 still an unexplained burst, 13293 /
2268 still an unexplained burst. **Skipcheck:** 29 true, 11 false, row for row the same as in Iteration 2 (the ten false rows inside
the removed frames at 0-4 frames, and GH199681 9937 which is the minimum-gap reshuffle explained there).

**Tests.** `test_frame_skips.py` 222 (189 before); `test_completeness_credited.py` 95, `test_linking.py` 266, `test_pipeline.py` 260,
`test_ingest.py` 124, `test_tracks.py` 79, `test_labels.py` 50, `test_hands.py` 38, `test_env.py` 36, `test_olympic.py` 33: all
unchanged and passing.

**Mutation check (scratch copy; `test_frame_skips.py` must fail)**
| # | Mutation | Result |
|---|---|---|
| B1 | filter after suppression restored (ineligible peaks suppress, then are discarded) | caught, 1 (the 212 regression) |
| B1b | `detect` does not pre-filter its candidates | caught: the in-code exposure assertion raises (a burst outside the exposure) |
| B2 | the decimation step (and the other constants) not compared | caught, 6 |
| B2b | only ratio and half window compared | caught, 6 (the step 8 file passes) |
| B3 | sample frame not checked against the video length | caught, 4 |
| B3b | a sample row of an unprocessed video accepted | caught, 2 |
| N1 | shared `.part` name restored | caught, 1 (a PermissionError among the eight writers) |
| N2a | bin edges exclusive | caught, 5 |
| N2b | distance to the previous skip only | caught, 5 |
| N2c | gate ignores the 0-2 bin | caught, 1 |
| N3 | pooled shares replaced by the mean of per-video shares | caught, 2 |

**Residual risks added.** (1) The distance-bin gate and the design were fixed on the same data (declared before); a held-out video
set would be the independent test. (2) `Skips.save`'s retry covers the Windows rename race, not two writers with different content
deciding which one wins: the last replace wins, and the file is always one writer's complete content. (3) A plateau that straddles
the edge of the removed range is reported at its midpoint; if that frame is ineligible the plateau yields no burst.

