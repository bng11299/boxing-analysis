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
- **Astra review 1** (Codex, read-only, effort medium, on `ddf6399`): **FIX FIRST**, 5 blocking + 2 should-fix.
  Astra could not start Python in its sandbox ("Access is denied"), so its findings are source-derived.

  | # | Finding | Severity | Resolution |
  |---|---|---|---|
  | 1 | Baseline-failed videos still counted in G2/G3a and the dev margin choice | BLOCKING | already fixed in `6abd33f` (self-review); tests |
  | 2 | Stillness file: duplicate keys load (last wins), `samples` != 24 and `max_px < motion_px` accepted | BLOCKING | duplicate-key hook, `samples == 24`, `0 <= motion <= max`, top-level keys exact; tests |
  | 3 | Videos left out for camera motion before fingerprints are checked: a stale entry silently drops a video | BLOCKING | fingerprints of stillness (all videos) checked first; floors for the kept ones; test |
  | 4 | `[A, B, B, ...]` reads 0 px, contradicting the gate's "moves in the first ~12% -> left out" | BLOCKING | behaviour is the stated short-segment limit; gate wording corrected in a dated amendment; tests pin both cases |
  | 5 | Every filtered arm failing -> `KeyError` in `verdict`, no report | BLOCKING | unmeasured bars FAIL; tests |
  | 6 | Zero exposure or zero baseline unexplained -> `ZeroDivisionError` | SHOULD-FIX | rates via `_share` (None), no ratio -> G1 fails; test |
  | 7 | JSON integer 10**400 -> `OverflowError`, not `ValueError` | SHOULD-FIX | caught in `_number`; floor and stillness tests |

  175 checks pass after the fixes.
- Added in iteration 1 on LESSONS 2026-10-25: a sweep of every floor field (11) and stillness field (9) x 8
  wrong-type values; all 160 combinations behave (`cde3291`, 177 checks). Full suite: 21 of 21 Python suites green.

## Iteration 2
- **Astra review 2** (on `ea42d7b`): **FIX FIRST**, 6 findings, all on failure paths; Astra confirmed fixes
  1, 3, 4, 5 and 7 of review 1 correct. Source-derived again (Python blocked in its sandbox).

  | # | Finding | Resolution |
  |---|---|---|
  | 1 | Zero baseline -> infinite bootstrap bound -> strict JSON write fails, no report | unbounded bound written as null; G1 fails on null; test serialises it |
  | 2 | Every video left out -> `rows[0]` IndexError | empty summary; every gate bar FAILS (G3c too: unmeasured); dev and CLI tests |
  | 3 | Every baseline failed -> G3a passes on `None or 0.0` | G2 and G3a need a measured share; test |
  | 4 | Two tasks with one video id, or one recording under two ids, pass the "all 25" count and pool twice | distinct ids and distinct fingerprints required before anything else; two CLI tests |
  | 5 | Stillness accepts 16384x16384 and `samples: 24.0` | `MAX_PIXELS` check; samples must be the integer 24; tests |
  | 6 | `+ringfloor` note can push `model` past the loader's 4096-character cap: written tables refused by the CLIs | `without_ids` refuses before writing; test |

  187 checks pass after the fixes.

## Iteration 3
- **Astra review 3** (on `c092a51`): **FIX FIRST**, 4 blocking + 3 should-fix; source-derived again.

  | # | Finding | Resolution |
  |---|---|---|
  | 1 | A dev run missing dev tables still chooses a margin, from a subset | dev needs all four tables (amendment 2026-10-28); test |
  | 2 | Held-out accepts any `--rule`: G1 and S3 could measure another experiment | `REGISTERED_RULE` pinned to the V6 rule; held-out refuses others; reports record the rule; tests |
  | 3 | A 1e-200 px edge passes validation and makes distances NaN | edges under 1 px refused; zero-length edges handled in `signed_distance`; tests |
  | 4 | G3c passes when every baseline failed (nothing compared) | G3c needs at least one compared video; test |
  | 5 | A `None` rate crashes the dev printout | formatted through `_fmt`; covered by the zero-rate tests |
  | 6 | A link planted at `docs/x.json` pointing outside passes the folder guard | the guard resolves the folder, not the name (the writer replaces links); test (symlinks need privilege here, so it checks the name path) |
  | 7 | The duplicate-recording test could pass without its guard | duplicate checks run first over the whole dataset; tests assert each reason; synthetic dev set is now four distinct recordings |

  193 checks pass after the fixes.
## Iteration 4
- Fresh clone at `ac91dfb`: all 21 Python suites green (3,313 checks; `test_ring` 193).
- **Astra review 4** (on `ac91dfb`): **FIX FIRST**, 4 blocking + 1 should-fix; source-derived.

  | # | Finding | Resolution |
  |---|---|---|
  | 1 | G3b pooled only videos that linked in both arms: a video whose credited people were all removed (linker fails) left G3b's denominator | G3b pools every video whose baseline ran (attribution precedes linking); test: the lost credits halve the share and G3b fails |
  | 2 | Iteration 3's guard regressed folders: `--out-dir results/..` reaches the repo root; a junction under results/ into docs/ passes | both the folder-resolved name and the fully resolved path are checked; `.`/`..` names refused; tests (links simulated: no symlink privilege here) |
  | 3 | Held-out checks a count of 25, not membership: a substitute recording can stand in for a missing video | `HELD_OUT_VIDEOS`, ADR-0013's 25 ids, required exactly; test with a substitute |
  | 4 | A textureless starting reference reads any motion as 0 px | a reference or frame under 2 grey levels of texture is unmeasurable and counts as the largest move; tests |
  | 5 | The no-symlink fallback test passed with or without the fix | replaced by a simulated link through `Path.resolve`, which the old check fails |

  201 checks pass after the fixes.
## Iteration 5
- **Astra review 5** (on `ac7ecdc`): **did not complete.** Codex hit its usage limit (retry after 21:26) after one
  note: when every filtered arm fails, G3b was never computed (reported unmeasured) although attribution counts
  exist. Fixed: G3b's credited shares are computed before any early return, over every video whose baseline ran;
  tests for both cases (credits kept -> G3b measured and passing; credits lost -> 0% and failing). 204 checks.
- A complete review of the final commit is still owed; it is not counted as passed.- Fresh clone at `e787657`: all 21 Python suites green (3,324 checks; `test_ring` 204). The machine scripts were not
  touched by this branch and were not re-run.

## Security gate (secure-software-assurance defaults), at `e787657`
- Trust boundaries: the floor file and the stillness file (size cap, strict UTF-8 JSON, duplicate keys, NaN/Infinity
  and 10**400 refused, exact key sets, a sweep of every field x 8 wrong-type values); video files through the
  existing `probe` caps; tables through the existing strict `PoseTable.load`; labels through the strict CVAT loader.
- Every subprocess is an argument list with a timeout (ffmpeg seeks and the PNG encoder); no shell strings.
- Outputs: only under the gitignored data/ or results/ (or outside the repo); the guard checks the resolved path and
  the name's folder; writes replace any link planted at the name. `git check-ignore` confirms every derived path
  (backgrounds, stillness, floors, evaluation reports, self-check) is ignored; nothing derived is tracked.
- Inputs bound to their video by content fingerprint before anything is left out; duplicate ids and recordings refused.
- No secrets, no network, no pickles, no participant footage touched.