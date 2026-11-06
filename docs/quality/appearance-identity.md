## `feat/appearance-identity` -- 2026-10-22

Can appearance keep two boxers apart where pose failed? Gate: `docs/gates/appearance-identity.md` (committed
`c1b8b9c` before the design). Base: `origin/main` (has #13) + `feat/corner-colour` merged for its harness;
`feat/identity-continuity` not merged (it cost coverage, 65% -> 38%).

### Evidence that chose the route (2026-10-22, before the gate)
- Brendan's eye check of the 3 kit clips: in all 3 "both read red" pairs, one person was the blue boxer. A torso-region
  overlay of the DEV pair (GH159681 1057 / 1102, rendered locally, not viewed by the reviewer) showed why: BOTH wear red
  vests; 1057 has blue gloves and headgear. The torso pixels of 1057 are red (mean RGB about 85 / 33 / 33), so v2
  read the vest correctly; the vest is just not the corner colour in that bout.
- Same overlay: id 1102 jumps to the referee at frame 4880 and to the other boxer at 5039. ByteTrack fragments are
  not always one person.
- The two people of each pair are distinct (box IoU median 0.00-0.05, shoulders 1.7-1.9 torso lengths apart).
- No cross-camera twin was found from label timing (best r 0.16-0.31 per kam2 video, never clearly above the next).

### Design on DEV only (4 videos, 221 co-present credited pairs; numbers are not evidence)
Per sampled row (10 Hz), coarse HSV histograms (15-degree hue bins; saturation edges 0.3 / 0.5 / 0.7; value edges
0.08 / 0.2 / 0.4) of five candidate regions were extracted once (counts only, scratch, not committed) and scored
with v2's decision constants. Decided pairs opposite:

| region (whole fragment) | v2 bands (red sat >= 0.7) | red sat >= 0.5, blue val >= 0.2 |
|---|---|---|
| v2 head disc (face-centred) | 85 / 123 (69%) | 9 / 150 (6%) |
| cap around the face, face core excluded | 86 / 118 (73%) | 42 / 148 (28%) |
| crown above the face | 89 / 124 (72%) | 44 / 149 (30%) |
| glove past the wrist, 0.30 / 0.55 forearm | 82 / 141, 86 / 145 (58-59%) | 27-34 / 146-154 |
| crown or cap + glove pooled | 74-79 / 112-122 (65-67%) | 16-20% |

The red floor at 0.5 reads skin as red (as v2 found). The decisive diagnostic: read on the frames a pair SHARES,
the crown gives 98 / 108 (90.7%; kam4 97 / 106), crown + glove 98 / 122 (80%). On kam2 the blue kit sits at
saturation < 0.3 (blue-hue share 0.10 / 0.03 / 0.00 / 0.00 by saturation bin on GH078416's crown), so neither rule
decides there (2 of 50 co-present pairs). 16 variants were looked at on dev in all; the simplest that worked was
frozen: crown, v2 bands as binned, window reading. No per-camera band was fitted (15 co-present kam2 pairs on dev
cannot support one).

Production code on dev: 98 / 108 decided opposite (Wilson [0.838, 0.949]; 4-video bootstrap [0.50, 0.92], 48 of
10,000 draws dropped), 48.9% of co-present credited pairs decided; credited fragments red 91 / blue 54 / none 28 /
unsure 19. kam2 1 / 2, kam4 97 / 106. Wall 330 s for 38,137 frames.

### Iteration 1 -- review gate (Astra, read-only, effort medium): FIX FIRST, all fixed
| # | finding | fix | test / mutant |
|---|---|---|---|
| 1 | blocker: any unknown id counted as held-out; a dataset with 1 of 23 videos could PASS | frozen 23-video `HELD_OUT` manifest; missing or failed = not evaluated (exit 1); unknown ids never read; duplicate ids refused | 4 checks; mutants "unknown ids held-out", "gate on an incomplete set", "duplicate ids allowed" caught |
| 2 | counts bound to metadata only; a regenerated table could reuse them | sha256 of the table content + sampling step stored and checked | regenerated-table check; mutant caught |
| 3 | int64 overflow let impossible counts load | each bound checked alone (red, blue <= n <= 4K frame) before the sum | overflow file refused; mutant caught |
| 4 | compressed-size cap does not bound allocation | zip directory read first: exact member names, declared total <= 1 GiB, before `np.load` | np.load replaced by a trap in the test; mutant caught |
| 5 | fragment counts were per pair side | `credited_by_decision` counts each credited fragment once | star scene (1 red, 3 blue partners); mutant caught |
| open | empty bootstrap draws | dropped and counted; > 1% fails G1 (pre-registered in the gate's dated note) | gate edge 100 / 101 of 10,000; mutant caught |

Tests: `tests/test_headgear.py` 137 checks (synthetic only, mostly negative); 17 of 17 hand-made mutants caught.

### Held-out result (frozen `9d18ca6`, run once, 2,209 s): G1 FAILS
277 / 333 decided co-present credited pairs opposite = **83.2%** (Wilson [0.788, 0.868]; video bootstrap
[0.756, 0.880], 0 of 10,000 draws dropped) on all 23 manifest videos, none missing or failed. Bars: share 0.90 no,
100 pairs yes, lower bound 0.80 no. kam4 269 / 307 (87.6%, Wilson [0.835, 0.908]); kam2 8 / 26 (30.8%: when kam2
decides, it mostly reads both people red). Without the 3 chapter twins: 199 / 241 (82.6%). Per kam4 video:
72.7%-100% (GH189681 8 / 11 lowest; GH129681 45 / 54, GH169681 35 / 43, GH109681 17 / 21 the largest misses).
G2 not run. Baseline linker on the same 23 videos, for context only (the existing 10-07 outputs, not re-run):
1,165 / 1,788 credited punches on A or B (65.2%; 82.2% with `--reacquire`), median both-identified 57%.

What it says: the corner colour IS on the headgear (kam4 at 87.6% against v2's chance), but at this resolution and
on these tracks it is not reliable enough to constrain linking, and on kam2 it is not readable at all. Not
iterated, as the gate requires.

### Iteration 2 -- review gate (Astra, read-only, effort medium): 3 of 5 fixes confirmed, 2 partly; both completed
Astra confirmed the recorded FAIL is trustworthy: both bars fail independently, and the held-out run computed fresh
(no cached counts were loaded), so neither partial fix could reach the verdict.
| # | finding | fix | test / mutant |
|---|---|---|---|
| 2b | the content digest rounded keypoints and confidences to float32 (0.299999999 and 0.300000001 hash alike, either side of MIN_CONF) | hash as float64 (lossless for float32 and float64 tables) | digest test both ways; mutant caught |
| 4b | a tiny member's NPY header could still declare 10^10 elements | every member's NPY header read first: dtype int or text only, declared bytes must equal the bytes held | zip-bomb member refused with np.load trapped; mutant caught |

Tests 140 checks; 19 of 19 mutants caught. Fresh clone of `9d18ca6` from GitHub: test_headgear 137, test_appearance
171, test_linking 266, test_tracks 79, test_olympic 29, test_pipeline 314, all pass, clean tree.

### Security gate
The trust boundary is the counts file (an input): size, zip directory, NPY headers, dtypes, shapes, bounds and the
table's content digest are checked before use, no pickles. Video ids pass `_check_video_id`; the counts path cannot
leave its folder; ffmpeg runs with an argument list (`video.py`). No secret, no data, no result file and no pixel in
the diff; Olympic-derived numbers stay in the gitignored `results/`. The eye-check overlay was written to the
gitignored `results/eyecheck/` and not viewed by the reviewer.

### Residual risks
kam2 colour is unreadable at these saturations; ids hop between people (window reading limits, not removes, this);
the co-present pairs assume two people on screen together are different people (box overlap was near zero in the
3 checked pairs); bout-level boxer recurrence across chapters (robustness row barely moves).

