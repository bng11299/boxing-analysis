## `feat/e2e-v5v9` -- 2026-10-19

The solo pipeline end to end on real continuous footage with all six punch types: `find_punches.py`
(V6 rule) then `classify_punches.py` (per-clip cut, `--stance orthodox`) on YOLO11-pose of V5 and V9,
scored against every annotated span. Stacked on `feat/pose-switch`; a new script,
`eval_pipeline_v5v9.py`, no existing one changed. Acceptance: (1) it runs the two entry points and stops,
nothing scored, unless the classifier hash is pose_switch.json's, poses are at native fps, spans align at
offset 0, its own perfect-finder readings reproduce pose_switch.json (Y_all 0.889, Y_clip 0.826 on 742)
exactly, and each classified CSV matches its sidecar (hash, clip, stance, classifier, rule); (2) finder
recall by type, precision, count ratio, shift floor, block-bootstrap intervals; type accuracy, macro F1,
hand on matched detections, paired against Y_clip and Y_all on the same punches, commonest-type floor;
(3) a blinded label-completeness audit with controls; (4) negative tests at every file boundary, each
guard reverted fails a test; (5) suites green in a clean clone.

Threat sketch. Assets: the honesty of the first end-to-end number. Entry points: pose npz, annotation
xlsx, classified CSV + sidecar, pose_switch.json, audit judgments CSV, the mp4 (ffmpeg, argv list).
Trust boundary: files on disk between tools. Top threats: a stale or other-clip classified CSV scored;
judgments made on one audit sample applied to another.

### Iteration 1
- **Finding before any gate: the matching, not the pipeline.** First run: matched-detection accuracy
  0.576 against 0.841 for a perfect finder on the same punches, cross recall 0.29. Diagnosis: 147 of 356
  V5 matches sat off their span's wrist-speed peak, 115 exactly on ANOTHER labelled punch's. In a fast
  jab-cross the finder (0.4 s minimum gap) reports one peak, usually the cross; `punchfind.match` takes
  spans by end and hands the jab (which ends first) the cross's detection. On-peak matches scored 0.880.
  Added `match_nearest`: a maximum matching (same count, so recall and precision are unchanged; asserted)
  with the least total distance to span middles. Both attributions are reported. `classify_punches.py
  --labels` and `eval_classify_v6.py` still attribute with `punchfind.match` (not changed here).
- **Audit:** 50 far detections (> 0.3 s from every label) and 10 matched controls, shuffled, judged blind
  from frame strips. Controls 9 of 10 punches, 1 unclear, 0 missed.
- **Test:** 66 checks; 24 mutants, one survived (per-type hits credited to the detection's block: no
  fixture straddled a block edge) and got one.
- **Review: FIX FIRST.** B1 judgments bound only to a key file any run could rewrite: stale judgments
  could be counted against a new sample. Also: key beside the strips (blinding leak); strips never checked
  against the pose's video; `keep` not eval_pose_switch.py's predicate (held only because counts matched);
  missing reference a bare KeyError; tests that compared code with itself (hit count already asserted
  inside; no optimality check; ignored detections untested); NaN time accepted; sidecar rule unchecked;
  stale outputs not removed; BOM-saved judgments refused.
- **Security:** same B1 found independently. Subprocess via `video.iter_frames` (argv list); npz without
  pickle; JSON refuses NaN; CSV bounded; outputs only under `--out-dir`.

### Iteration 2
- **Fix:** every judgments row carries the sample digest (`id,sample_digest,judgment,note`), template
  prefilled; a key that differs from this run's is never silently replaced (re-render, which deletes the
  old strips); key moved out of the judge's folder; the video probed against the pose (size, fps, constant
  rate); eval_pose_switch.py's `seen` predicate; matched spans without a reference counted; finite time;
  rule in the sidecar checked; this run's per-video outputs deleted before the pipeline writes them; BOM
  tolerated; row cap; brute-force optimality over 150 small cases; an ignored detection out of the counts.
- **Test:** 76 checks; 32 mutants, all caught (the row cap needed a message check: without it the id-set
  check still refuses, after reading every row). Stale judgments refused on the real run.
- **Real-world facts** (`results/e2e_v5v9/pipeline_v5v9.json`, orthodox, gate 4 reproduced 742 exactly):

  | finder (V6 rule) | V5 | V9 | pooled |
  |---|---|---|---|
  | precision (lower bound) | 0.860 [0.794, 0.920] | 0.287 [0.159, 0.418] | 0.633 [0.552, 0.706] |
  | recall | 0.599 [0.546, 0.652] | 0.527 [0.336, 0.709] | 0.585 [0.530, 0.641] |
  | count ratio | 0.70 | 1.84 | 0.92 |
  | F1 / shift-floor F1 | 0.706 / 0.366 | 0.371 / 0.096 | |

  Recall by type, pooled: cross 0.76 (127/167), jab **0.32** (84/264), lead hook 0.66, lead uppercut 0.48,
  rear hook 0.86, rear uppercut 0.83. V9 straight punches: cross 0/27, jab 4/31.

  | type, 434 matched detections | accuracy | macro F1 | hand |
  |---|---|---|---|
  | pipeline | **0.811** [0.754, 0.868] | 0.772 | 0.940 |
  | Y_clip, perfect finder, same punches | 0.834 | 0.792 | 0.956 |
  | Y_all, labelled spans, same punches | 0.853 | 0.794 | 0.979 |

  Pipeline - Y_clip -0.023 (clip [-0.060, +0.012], block [-0.061, +0.017]); pipeline - Y_all -0.041
  (clip [-0.085, +0.005], block [-0.117, +0.047]). Commonest-type floor (cross) 0.293. V5 0.860 on 356, V9
  0.590 on 78 (Y_clip 0.679, Y_all 0.821; the pipeline's V9 window is 0.56 s against 0.47 s from the
  labelled punches). End to end: 352 of 742 labelled punches found and named right (0.474).
- **Label completeness:** far detections 19 of 50 real punches [0.26, 0.52]; V5 11 of 25 (8 unclear), V9
  8 of 25. Post hoc: inside the stretch of video that carries labels 19 of 37, outside it 0 of 13 (V9's
  labels cover 144-634 s of 940 s; the rest is talk, squat jumps, push-ups). Precision if the audited
  share holds: V5 0.91 [0.89, 0.93], V9 0.51 [0.41, 0.65].
- **Stance:** the finder's hand read as orthodox lead/rear matches the label's side on 95.2% (V5) and
  94.9% (V9) of matches: consistent with orthodox, or with a southpaw in mirrored footage. Unverified.
- **Residual, accepted:** the per-clip cut was chosen on these videos; V2 excluded (O61); coaches'
  YouTube footage, not phone footage; one judge (a reviewer) from strips; main's key-on-disk and re-render
  paths are exercised by the real run, not by unit tests.

