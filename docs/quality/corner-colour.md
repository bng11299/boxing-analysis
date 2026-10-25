## `feat/corner-colour` -- 2026-10-08

Identity by corner colour. Olympic boxers wear their corner's colour (red or blue), so a per-fragment kit colour
could say WHICH boxer a fragment is (the pose-only linker swapped the boxers once on GH078416) and that a
well-seen fragment with neither colour is not a boxer. **Status: the colour test FAILED its own non-circular
self-check on the real data (pooled opposite share 29.1%, gate 90%), so the work stopped there, as the brief
required. `appearance.py`, `olympic_colour.py` and their tests are committed. The linking integration
(`link(..., colours=)`, `olympic_link.py --colours`, the colour overlay, `olympic_completeness.py --colours`) was built
and tested, then WITHDRAWN from the branch: it is not committed and no linking run with colours was made.**

- **Acceptance:** (1) from painted synthetic frames the decisions are exact (red vest, blue vest, white shirt,
  skin, pale pink on skin, both colours); (2) the real-data gate asserts a fact that can fail: two credited
  fragments on screen together are two people, so they must get OPPOSITE colours; the labels never say red or
  blue, so this cannot pass by construction; (3) only numbers are kept, no pixel of the footage is written.
- **Threat sketch:** asset: the per-fragment colour decision that would constrain which fragments may form a
  boxer identity. Entry points: the video (untrusted pixels, decoded by `video.iter_frames`), the tracks `.npz`,
  the `.colours.json` file (an input to the linker), a GoPro id from the command line. Top threats: (a) a colour
  test that confidently labels the wrong thing (skin, a spectator's shirt) and then silently decides identities
  (this is what happened: see below); (b) a colours file from another table or another recording applied to
  this one. Mitigations: the self-check gate before any use; `Colours` is bound to the table (source name,
  n_frames, fingerprint when present, the exact set of track ids) and refused otherwise; the loader validates
  strictly (no NaN, known keys, shares in [0, 1], a decision must be the one its own medians imply).

### Design (`src/boxingvi/appearance.py`, `olympic_colour.py`)
- Rows read: every 5th frame (10 Hz at 50 fps) plus each fragment's first and last row with a confident torso.
- Regions per row (joints >= 0.3): torso quadrilateral (shoulders + hips) shrunk 20% to its centroid; head disc
  at the mean of the confident face keypoints; a glove disc at each confident wrist; discs 0.35 torso lengths
  (the box height x 0.3 when the four torso joints are not all confident). A region under 30 pixels is skipped;
  the regions are pooled as a union (a pixel counts once); pixels outside the frame are ignored.
- Per pixel in HSV (numpy): red = hue <= 12 or >= 340 degrees, blue = 200..255, both with saturation >= 0.45 and
  value >= 0.20. Per fragment: the MEDIAN over its rows of the red and the blue share; red / blue when the median
  is >= 0.10, >= 3x the other and >= 3 rows; none when >= 5 rows and both medians < 0.02; else unsure.
- Constants and reasons are beside each constant in `appearance.py`; all were fixed from colour physics before
  any colour result on the footage was looked at, and NONE was changed afterwards (the diagnostic below was read
  AFTER the gate failed and changed nothing). Honest caveat: a one-video trial run (GH078416) was read before
  the full run; it led to no change either.
- `olympic_colour.py` decodes each video once, writes `<tracks>/<id>.colours.json` (decision, medians, rows per
  track id; `allow_nan=False`), refuses an unbound tracks table (name, fingerprint), then runs the self-check and
  writes `results/olympic_colour_check.json`; exit 2 when the pooled opposite share is below 0.9, 1 when a video
  failed. `--check-only` recomputes the check from existing colour files.

### The self-check, real data (29 videos, all results gitignored)
| | |
|---|---|
| fragments coloured | 15,029 (red 3,606, blue 546, none 4,522, unsure 6,355) |
| credited fragments | 1,245 (red 436, blue 68, none 212, unsure 529) |
| co-present (>= 0.5 s) credited pairs | 1,292, of which both decided red/blue: 275 |
| **opposite / same** | **80 (29.1%) / 195 (70.9%)**, gate >= 90%: **FAIL** |
| by camera | kam2: 10 decided pairs, 3 opposite; kam4: 265 decided, 77 opposite |
| wall time | one video (3,186 frames) 24 s alone; all 29 in four parallel shards 23 min (shards 975 / 1,010 / 1,299 / 1,336 s) |

Two boxers on screen together are called the SAME colour 71% of the time and the test calls far more fragments
red than blue (3,606 vs 546; 436 vs 68 among credited ones). Credited fragments are mostly unsure (529) or
none (212): the kit does not reach MIN_KIT = 0.10 of the pooled region for most boxers.

**Post-hoc diagnostic (numbers only, read after the gate failed, NOT used to change anything).** For the credited
rows of GH078416 (kam2) and GH159681 (kam4): the share of region pixels by saturation band and 30 degree hue bin.
In both videos the hue 0-30 bin at saturation >= 0.45 holds 47% / 33% of the torso pixels, 34% / 36% of the
HEAD disc and 62% / 65% of the glove discs. The head disc is face skin plus headgear (kit-coloured headgear would fall in the same bin, so this is suggestive, not proof), so under this footage's colour grading
skin is highly saturated and warm-hued: the saturation floor of 0.45 does not separate skin from red kit, and the
red band (hue <= 12) takes part of the skin. Genuine blue (hue 210-240, saturation >= 0.45) is present but small
(torso 4% / 2%, gloves 1% / 9%), under MIN_KIT. In GH159681 a second, separate mass sits at hue 330-360 with
saturation >= 0.45 (torso 25%, gloves 23%): red kit with a magenta-ish hue that the 340 degree edge only partly
catches and that skin (hue 0-30) does not share. So the constants as pre-registered fail on this footage for a
specific, visible reason; a redesign (a hue band that excludes skin, per-video kit calibration from the
pooled hue distribution, or reading the vest only) is a decision for the author, and would have to be
pre-registered the same way and re-run through the same gate. None of that was done here.

### Tests (synthetic only; `tests/test_appearance.py`, 118 checks)
HSV against `colorsys` on 3,000 random pixels; every threshold edge (hue 340 / 12 / 200 / 255, saturation 0.45,
value 0.20); the torso shrink (10..90 of a 0..100 square holds exactly 6,400 pixels); disc areas; clipping at all four
frame edges and a fully off-frame figure; a wild 10-million-pixel keypoint; regions under 30 pixels skipped;
painted red / blue / white-shirt / skin / pale-pink-on-skin figures through the whole pipeline; median robustness
(2 of 7 rows full of blue stays red; the mean would be unsure; 4 of 7 flips it); rows thresholds 2 / 3 / 4 / 5;
which rows are read; `Colours` save/load with 17 refusals (malformed JSON, NaN, Infinity, unknown decision, a
decision its medians do not imply, shares above 1, negative / boolean rows, extra keys, bad ids, oversize file) and
the binding (source, n_frames, fingerprint, ids); the self-check (opposite -> 1.0, same -> 0.0, 0.5 s exactly is a
pair and 24 frames is not, none / unsure are not decided pairs, no credited punch gives None); the CLI end to end
on a painted clip, including the gate FAILING (both boxers called red -> opposite 0.0, exit 2), a renamed table,
another recording's fingerprint, path-like ids, a missing colours file.

### Mutations
| Guard reverted | Result | First failing checks |
|---|---|---|
| median -> mean (`appearance.fragment_colours`) | caught, 2 in `test_appearance` | median robustness; 4 of 7 rows blue |
| saturation floor removed (`SAT_MIN = 0`) | caught, 10 in `test_appearance` | white shirt reads red 25%; skin; the 0.451 / 0.447 edge |
| colour constraint removed (red / blue allowed in both corners) | caught, 13 (withdrawn linking tests) | swapped colours; all-red leaves blue empty |
| `none` allowed in a corner | caught, 6 (withdrawn linking tests) | stepping stone; excluded count |
| capacity shared across corners removed | caught: the suite aborts in the solver's own consistency check ("the solution leaves one fragment twice") | one boxer, every fragment unsure |

The last three were run on the withdrawn linking code before it was taken out of the branch; they are recorded because
they show the tests had teeth, not because that code ships.

### What was built and withdrawn (not committed)
`linking.link(..., colours=)` duplicated the layered path model per corner (chain 0 red, chain 1 blue; one chain per
corner; capacity shared; red only in red, blue only in blue, unsure in either, none in neither), `olympic_link.py
--colours`, the colour overlay (red / blue skeletons, yellow border), `olympic_completeness.py --colours`; with 335
checks in `test_linking.py` (266 existing unchanged + 69 new, including a 500-instance brute force of the corner model
and the 00:24 crossing: no swap with colours). They passed; they are not on this branch because building on a colour
test that fails its gate was ruled out. The `solve_chains` change kept variable and row order identical when
`corners is None`, but that equivalence was only exercised by the existing 800-instance brute force, not by a
byte comparison.

### Residual risks
Skin saturation under this grading (the failure above); spectators, coaches or judges in a red or blue top are
red / blue like a boxer; coloured lighting moves every hue; a clinch fills a box with the other boxer's kit (the
median absorbs a few rows, not a long clinch); the colours were checked by the self-check, not by eye (and no
Olympic frame was viewed); fragments of a few frames have 2 rows and are always unsure; pairs between the same two
boxers are not independent, so the 275 decided pairs are far fewer than 275 independent observations.

