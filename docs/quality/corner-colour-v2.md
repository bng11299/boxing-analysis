## `feat/corner-colour`, version 2 -- 2026-10-08 (rule FROZEN in this commit, before any test video was read)

Version 1 failed its self-check (29.1% opposite, 80 of 275 decided pairs). Version 2 redesigns the colour test and
re-asks the same question with a clean split. **Protocol, fixed by video before any v2 design:** DEVELOPMENT videos
GH078416 and GH159681 (already inspected after v1 failed) plus GH128416 (kam2) and GH069681 (kam4), drawn with
`random.Random(20261008)` from the rest of each camera; every other video (13 kam2, 12 kam4) is TEST. No colour
statistic, decision count or self-check number of a test video was computed, printed or read until this rule and every
constant were committed. The v1 colours files of the test videos sat in `results/` and were not opened. Only the four
dev videos were extracted before this commit (`olympic_colour.py` defaults to `--split dev` for that reason).
The gate: on the 25 test videos pooled, co-present (>= 0.5 s) credited pairs both decided red/blue, opposite share
>= 0.90 AND >= 100 decided pairs; per camera, < 20 decided test pairs means "colour not established" there.

### The rule (`appearance.py`; reasons beside each constant)
| | v1 | v2 | why (dev data) |
|---|---|---|---|
| region | torso + head disc + 2 glove discs, pooled | torso only | the head disc includes background (red and blue corner pads); gloves are mostly arm; no combination beat torso-only on the self-check |
| red | hue <= 12 or >= 340, sat >= 0.45, value >= 0.20 | hue [340, 25), sat >= 0.70, value >= 0.20 | skin is hue 15-40 with saturation up to ~0.6 (read as red by v1); red kit is hue 10-20 (kam2) / 350-5 (kam4) at saturation 0.7-1.0; non-red fragments have 0.000-0.009 of the torso there, red-kit fragments 0.3-0.7 |
| blue | hue 200-255, sat >= 0.45, value >= 0.20 | hue [200, 290), sat >= 0.50, value >= 0.08 | the blue boxer's vest is DARK navy (value 0.08-0.2, hue 220-240, sat ~0.9); v1's value floor threw it away. Credited fragments called blue on dev: 27 with floor 0.20, 54 with 0.08 |
| decision | median share >= 10% and >= 3x the other | per row the kit index (B - R) / (B + R) over kit pixels; a kit row has R + B >= max(20 px, 5% of the torso); per fragment the median index over kit rows: >= 0.5 blue, <= -0.5 red (one colour at least 3x the other); needs >= 3 rows, >= 3 kit rows, kit rows >= half of the rows | relative, so a navy vest that is 7% of the torso still counts; a fixed share floor does not |
| none | both medians < 2%, >= 5 rows | median kit share < 2%, >= 5 rows | unchanged in spirit |
| unsure | everything else | everything else | |

Kept from v1: sampling every 5th frame plus each fragment's first and last row with a confident torso, the torso shrunk
20%, min confidence 0.3, min region 30 px, median over rows, the binding of the colours file to its table (format is
now `boxingvi-colours-2`, strict loader). No per-video adaptive threshold: the dev kit index is bimodal at about -1 and
+1, and the decisions are flat for thresholds 0.2-0.5. The colours file stores `index`, `kit`, `rows`, `kit_rows` per
track id.

### Development numbers (4 dev videos; NOT evidence, they were looked at as much as wanted)
Stability on dev (production code, 192 credited fragments): red 105 / blue 53 / none 19 / unsure 15. One-at-a-time
changes of the constants moved red 100-111, blue 47-62, none 10-31, unsure 9-37 (red saturation floor 0.6-0.8, red hue
ceiling 15-35, blue saturation floor 0.4-0.6, blue hue band 190-300, ROW_KIT_MIN 0.02-0.12, index threshold 0.2-0.8,
kit-row fraction 0.3-0.7); only the blue value floor at 0.20 (the v1 value) was a cliff (blue 27, none 44).
Dev self-check at the frozen rule: 146 decided co-present credited pairs of 221, **opposite 73 (50.0%)**:
GH078416 4 of 6, GH128416 6 of 9, GH069681 41 of 60 (68.3%), GH159681 22 of 71 (31.0%); kam2 15 decided pairs, kam4
131. Dev wall time 197 s for the 4 videos (38,137 frames).
Region and fusion experiments on dev (decided pairs, opposite share): torso 148, 50%; head 107, 64%; gloves 108, 44%;
torso+head pooled 122, 53%; torso+head+gloves 121, 50%; head-then-torso 135, 64%; torso and head must agree 80, 66%.
None is near 90%, and the differences are within what about 100 pairs from 4 videos can resolve, so the simplest
rule was frozen rather than a fitted fusion.
Credited fragments called `none` on dev: 19 of 192. For 13 of them (all 10 on kam4, 3 of 9 on kam2) a decided red AND
a decided blue fragment were also on screen with them for >= 25 frames, which is what a third person (a referee)
looks like; they carry 1-4 credited punches each. The other 6 are on kam2, where blue is rarely decided, so that check
cannot say.

### What dev says about the premise (found while designing; relevant to reading the test result)
In GH159681 from about 90 s on, nearly all credited fragments have a red torso (credited red 56, blue 15 in the
video), and 49 co-present credited pairs have the same colour with both fragments' median kit share >= 0.25. Looking
at positions only: two boxer-sized boxes (height 400-600 px) 100-250 px apart, on screen together for up to 16 s,
both with about 70% of the torso red and red gloves (e.g. ids 2315 and 2329 at 236-252 s); of the 49 pairs, in 11 one
head region is decided blue and the other red, and in 30 more one head is undecided with a blue-leaning index (the
blue share of those discs is 0.12-0.29). So in this bout the torso (and the gloves) do not separate the boxers. No
pixel was viewed; this is counts. If the test videos behave like this, the opposite share is capped below 90% by the
footage and not by the colour rule; the gate is reported as it comes out.

### Tests (synthetic only; `tests/test_appearance.py`, 171 checks, was 118 for v1)
Replaced: every v1 threshold-edge assertion (saturation 0.45, value 0.20, hue 12 / 255), the share-based `decide`
boundaries (MIN_KIT 0.10, RATIO 3, NONE_MAX on both shares), the all-regions row-share checks, the v1 `Colours` keys
(`red` / `blue`), the CLI test's single `passes` flag. Added: every v2 edge (red hue 340 / 25, saturation 0.70, value
0.20; blue hue 200 / 290, saturation 0.50, value 0.08), the skin band (skin, hot skin, pale pink are not red; the v1
floor shown to read hot skin as red), the navy vest (and the v1 value floor shown to lose it), black, torso-only (red
headgear and gloves on a white vest read none; the same figure reads red with all four regions), `use=` in
`region_masks`, kit-row boundaries (19 / 20 px, 2% / 5%), split vests with the index counted from the torso mask (80%
blue is blue; half and half, and blue 2.3x red, are unsure), patches of 9 / 45 / 80 px (speck / weak / kit), median
robustness, kit rows 3 of 7 vs 4 of 7, row thresholds, 25 loader / saver refusals (index range, kit_rows > rows, index without
kit rows, v1 file and v1 keys, ...), the dev/test split constants, `--split` default dev (a test video is not read),
a dev-only run does not evaluate the gate (passes null), the 100-pair requirement (a perfect 1-of-1 pair fails it),
per-camera "not established", and the gate failing when both boxers are called red (exit 2).

### Mutations (27, each run on a copy of the code with only `tests/test_appearance.py`)
All 27 are caught (failing checks): skin band included (red saturation 0.45 + hue ceiling 45) 6; absolute share floor
(a kit row needs 10% of the torso) 3; blue value floor back to 0.20 14; head and gloves read too 13 (plus a failed
assertion); median -> mean for the index 2, for the kit share 1; index margin removed 6; kit rows need not be half
2; MIN_ROWS 1: 3; NONE_ROWS 1: 2; NONE_MAX 0.5: 3; kit-row pixel floor removed 1; torso not shrunk 12; red hue wrap
broken 10; blue hue band 150-330 2; blue saturation floor removed 2; loader skips the implied-decision check 1; loader
accepts kit_rows > rows 1; co-presence needs no time 1; undecided counted as decided 1; gate 0.9 -> 0 2; no minimum
pairs 2; no camera minimum 2; default split all 1; split inverted 2; dev list changed 1; gate evaluated on dev numbers
1. (A first run left "blue value floor" uncaught because the edit had hit the constant's name in the docstring; the
harness now edits code lines only and the corrected run is the one reported.)

All nine suites pass at this commit: pipeline 260, ingest 124, env 36, labels 50, hands 38, olympic 33, tracks 79,
linking 266, appearance 171.

### TEST RESULT (run after the freezing commit e1697c1; 25 test videos, nothing tuned afterwards)
Extraction on the 25 test videos in four parallel shards (633-753 s each; about 12.5 min), then the pooled check from
the colours files. **The gate FAILS.**
| | decided co-present credited pairs | opposite | same | opposite share |
|---|---|---|---|---|
| **test, pooled (gate: >= 90% and >= 100 pairs)** | **582** (of 1,071 co-present) | **304** | 278 | **52.2% -> FAIL** |
| kam2 (13 videos) | 196 | 82 | 114 | 41.8% (established: >= 20 pairs) |
| kam4 (12 videos) | 386 | 222 | 164 | 57.5% (established) |
| dev, for reference (not evidence) | 146 | 73 | 73 | 50.0% |
The test result is the dev result (50%) to within the noise of ~600 pairs: the chance level, not a near miss. The
count requirement (>= 100 pairs) is met, so the failure is the opposite share alone. Videos with at least 20 decided
pairs range from 0% (GH178416: 60 pairs, no blue fragment among the credited ones; GH098416 is the same with 14 pairs)
and 19% (GH079681, 42 pairs) to 81% (GH088416, 26 pairs); none reaches 90%.
Credited fragments on test (1,053): red 494 / blue 318 / none 175 / unsure 66. The 278 same-colour pairs are 209 red-red
and 69 blue-blue, and 191 of them (69%) have a median kit share >= 0.25 in BOTH fragments, i.e. both are confidently
kit-coloured, not borderline. Credited `none` is 16.6% of credited fragments; for 107 of the 175 a decided red AND a
decided blue fragment were also on screen with it for >= 25 frames (a third person, as on dev), the other 68 cannot be
checked that way (blue is rarely decided on kam2).

### Reading it
The colour rule does what it was designed to do at the level of single fragments (dev: bimodal index, stable
under every one-at-a-time change of the constants, `none` mostly a third person), and the two decision counts (red 494
vs blue 318 credited) show the excess of red that dev showed. What it cannot do is separate the two boxers of a bout
in the pairs that matter: 278 of the 582 decided pairs are two people with the same torso colour (red-red 209), and
for 191 of those both are confidently kit-coloured. Two hypotheses, which the counts cannot tell apart (the footage was
not viewed, by rule): both boxers of those bouts really wear the same vest colour, or some credited punches belong to
a person of the same corner as the boxer (a second, a spectator at the ropes) rather than to the opponent. Either
way the premise "two boxers on screen together have opposite torso colours" does not hold for this dataset well
enough to constrain a linker, and a rule tuned on 4 videos cannot fix a premise.

### Status and what was not done
- The gate failed, so, as briefed: the frozen rule and this result are committed, NOTHING was tuned on test videos, and
  the linking integration (`link(..., colours=)`, `olympic_link.py --colours`, the colour overlay, the
  `--reacquire` runs, the GH078416 timelines, the before / after table, completeness, the mutation checks for the
  linking) was NOT re-applied. `linking-integration-NOT-COMMITTED.patch` stays in the scratchpad.
- Colour is "established" in the brief's sense for both cameras (>= 20 decided pairs each), but established means
  measured, and what was measured is chance level, so no camera's colour is usable as a constraint.
- Next option, Brendan's call: learned re-ID (BoT-SORT with appearance embeddings) on the GPU laptop. Cheaper
  questions first, if wanted: (a) whether both boxers really share a vest colour in the same-colour bouts (look at ONE
  such pair of frames by eye; this session's rules forbid viewing footage, so it needs you), since the whole premise
  rests on that; (b) the head region alone gave 64% on dev (107 pairs), better than the torso but far from 90%, and
  was not tried on test because the rule was frozen.
- Files not committed (gitignored `results/`): `olympic_colour_check_v2.dev.json`, `olympic_colour_check_v2.test.json`,
  and the 29 `<id>.colours.json` (v2 format, replacing v1's).

