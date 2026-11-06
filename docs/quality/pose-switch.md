## `feat/pose-switch` -- 2026-10-12

How much the classifier loses reading YOLO11-pose instead of AlphaPose, on the held-out
punches themselves. Stacked on `feat/classify-detected`. Acceptance: (1) the saved classifier
reproduces 0.8145 / 0.7207 on the corpus clips before anything is scored; (2) each video's
pose is the source at its native rate and covers every span; (3) spans align with YOLO's
frame numbering, or the video is excluded and named; (4) A - Y with a paired interval;
(5) suites green in a clean clone.

Threat sketch. Assets: the honesty of one number the poster will carry. Entry points: pose
npz from the laptop, corpus skeletons and annotation sheets, classifier manifest. Trust
boundary: files synced between machines. Top threats: scoring skeletons against the wrong
frames (a silent, plausible-looking number); a different person in frame scored as the boxer.

### Iteration 1 (2026-10-09 to 10-11)
- **Test:** 16 checks; 6 guards reverted, 5 caught (one after a slow-motion fixture with
  two-model jitter), 1 equivalent (no shared joint is already NaN).
- **Smoke test** with AlphaPose clips standing in for YOLO at their own frames: gate 1 0.8145,
  offset 0 on all three videos, A == Y on 974 clips. Plumbing only.

### Iteration 2 -- the real run (2026-10-12)
- **GPU extraction** (laptop, RTX 3060): V2 32,611 frames, V5 15,519, V9 22,541 at native
  rates; person found in 99.9%, 98.2%, 97.5%. Files arrived byte-identical in size.
- **Gate 3 stopped the first run on V2** (best offset 7, sharpness 1.01), as built. Probing:
  V5 and V9 sit exactly at their annotated frames (0.009 frame heights). V2 does not, at any
  shift in -12,000..12,000 or frame-rate ratio (best 0.035), and an exhaustive search of the
  video finds 0 of 12 clips (V9 control: 12 of 12). Same place, same size, different moment:
  V2's skeletons are not from the video linked as V2 (vault ISSUES-LOG O61). Gate 3 now
  excludes such a video by name and scores the rest; none aligned is still a hard stop.
- **Real-world facts, V5 + V9, 742 punches** (`results/pose_switch.json`):

  | reading | accuracy | macro F1 | hand |
  |---|---|---|---|
  | A AlphaPose | 0.896 | 0.819 | 0.987 |
  | Y_all YOLO, every detected joint | 0.889 | 0.801 | 0.978 |
  | Y YOLO, joints under 0.3 blanked | 0.348 | 0.291 | 0.535 |

  A - Y_all **+0.007 [-0.009, +0.024]** (macro F1 +0.018 [-0.018, +0.054]); per video +0.007
  both. A - Y +0.549 [+0.511, +0.582]. The pose switch costs nothing measurable; the
  confidence rule cost 55 points (fixed in `classify`, that branch's iteration 6).
- **Cut readings, every joint kept:** labelled span 0.889; old fixed window 0.806; per-punch
  extent 0.774; one window per video at its median extent 0.823. Y_ext - Y_fix -0.032
  [-0.057, -0.009]; Y_ext - Y_long -0.050 [-0.073, -0.027]; Y_long - Y_fix +0.018 [-0.005,
  +0.039]. Extent overlap with the label by type: jab 0.49, cross 0.58, lead uppercut 0.51,
  lead hook 0.65, rear hook 0.63, rear uppercut 0.67. 0 clips unseen by YOLO, 0 other-person.
- **Residual, accepted:** two of three held-out videos; coaches' YouTube footage, not phone
  footage; V5 and V9 may share athletes with training videos (O55); the script's gates are
  exercised by the smoke test and the real run, not by unit tests.

### Iteration 3 -- the per-clip cut, through the pipeline's own function (2026-10-13)
- `Y_clip` now calls `classify.clip_cut` / `clip_window` (it was a hand-rolled control, `Y_long`).
- **V5 + V9, 742 punches, every joint kept:** labelled span 0.889, per clip **0.826**, old fixed
  window 0.806, per punch 0.774. Y_all - Y_clip +0.063 [+0.034, +0.094]; Y_clip - Y_fix +0.020
  [-0.001, +0.043]; Y_ext - Y_clip **-0.053 [-0.075, -0.030]**; Y_ext - Y_fix -0.032 [-0.057,
  -0.009]. Clip lengths 0.360 s (V5) and 0.468 s (V9); no punch left out of either median.
- The earlier control gave 0.823 with the same lengths to within a frame.
- **Limit added to the results file:** the per-clip cut was chosen looking at these numbers, so
  they are no longer a clean test of it.

### Iteration 4 -- the two Codex findings (2026-10-21)
Codex (2026-10-20), both confirmed by reading: (1) every reading used `seen and seen_all`, so
A - Y_all, the pose-switch cost under the pipeline's every-joint rule, was computed only on
punches the obsolete 0.3 rule also sees; (2) when alignment passes but no punch has usable
wrist speed, the cut readings are `{"n": 0}` and the print raised `KeyError: 'accuracy'`
before the results file was written.
- **Acceptance:** (a) A vs Y_all read on `all_seen`, legacy Y only on a named subset
  `legacy_common`, both n's in the file; (b) a run with sharp alignment and nothing cuttable
  exits 0, writes results and says so; (c) reverting either fix fails a test; (d) gate 1 still
  0.8145 / 0.7207 on the real run; (e) suites green in a fresh clone.
- **Build:** scoring moved into a pure `aggregate(rows, predict_fn, scored)`; printing into
  `summary_lines(report)`, where every reading may be empty and prints `n/a` (the per-video
  line had the same crash waiting for a video with no seen punch). Results layout:
  `all_seen` {A, Y_all, A_minus_Y_all}; `legacy_common` {A, Y, Y_all, A_minus_Y,
  A_minus_Y_all}; `same_person` and `by_video` each hold both; `n_legacy_blanked`; the cut
  readings (`window`) now start from `all_seen` and carry their own `n`. Window keys unchanged.
- **Test:** 13 new checks in `tests/test_poseswitch.py` (29 total). Eight on `aggregate` with a
  fake classifier: 6 punches both rules see plus 3 only the every-joint rule sees, where the old
  filter reads A - Y_all as 0 on 6 and the fix as 1/3 on 9. Five on the CLI, end to end: a
  synthetic corpus, a random-init ST-GCN++ and a V5 pose of a rigid body on a whole-pixel random
  walk (alignment sharp at offset 0) whose wrists are never detected (no wrist speed, no cut);
  4 of its 20 spans have every joint at 0.2. The HEAD script on that fixture: `KeyError:
  'accuracy'`, no file. **Mutants, 5 of 5 caught:** old filter for all_seen (7 checks fail);
  legacy Y scored on all_seen (6); unguarded accuracy print (1); window guard removed (1);
  window drawn from the common subset (1).
- **Real run, V5 + V9 (same inputs: the pre-fix code reproduced the recorded file exactly):**
  gate 1 0.8145 / 0.7207. `all_seen` 742, `legacy_common` 742, `n_legacy_blanked` 0, 0 missed,
  0 other-person. **The headline does not change:** A 0.896 (F1 0.819), Y_all 0.889 (F1 0.801),
  A - Y_all **+0.007 [-0.009, +0.024]** on 742; per video +0.007 on 594 (V5) and on 148 (V9);
  legacy Y 0.348 on its 742. 27 result blocks compared against the pre-fix run: identical to
  the bit. Why nothing moved: the old rule drops a punch only if fewer than 2 of its frames keep
  ANY joint at 0.3 or above; it blanks joints, not frames, and on these videos no punch lost
  whole frames. The finding was a real flaw with no effect on this data; it would on darker
  footage (IMG_4395) or V2.
- **Downstream:** #20's gate 4 reads `window.Y_all` / `window.Y_clip` (n and accuracy): values
  unchanged (0.889 / 0.826 on 742), but its selection copies the old `seen and seen_all` filter
  and must drop the `seen` half to stay the same selection as this file.
- **Review:** self-review restored macro F1 in the printed pooled lines (still in the JSON).
  **Astra (Codex), one read-only review of 46ba71c..5d86b9c, effort medium: SHIP.** Both
  findings FIXED with line evidence; the new tests judged not vacuous; no new defect. One
  non-blocking note, the #20 gate 4 selection above (its `eval_pipeline_v5v9.py:565`, checked by
  hand). It could not run the tests (no Python on its PATH), so its verdict rests on reading.
- **Security:** no new input path; `aggregate` / `summary_lines` are pure over rows the gated
  loaders built; the test runs the CLI by argument list in `mkdtemp`. No findings.
- **Gate reproduce:** fresh clone of 5d86b9c: pipeline 260, classify 202, hands 38, ingest 109,
  labels 50, env 36, poseswitch 29, machine scripts 38, all passing; the same counts as the
  worktree.

