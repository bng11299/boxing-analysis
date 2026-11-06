## `feat/per-hand-punch-finder` -- 2026-10-06

The wrist-speed signal judged per hand, with short dropouts bridged, and `find_punches.py`
(one clip -> `results/punches/<clip>.punches.csv`, scored against imported labels).

- **Acceptance:** (1) on a track without confidence (V6) the signal is bit-for-bit the old one
  and every saved V6 number stands; (2) a hidden far wrist no longer erases the near hand;
  (3) dropouts up to 0.05 s are bridged, longer gaps never differenced across; (4) scoring
  asserts real recall against labels, which are re-validated before use.
- **Threat sketch:** asset: the punch counts users will read. Entry points: pose files, label
  files. Top threats: interpolation inventing or inflating a punch; a stale or hand-edited label
  file scored as ground truth.

### Iteration 1
- **Real-world facts:** `run_punch_finder.py` re-run on V6 with the new code: **0 fields differ**
  from the saved 09-27 JSON (recall 0.741, F1 0.571, same CI). A frozen copy of the old
  `wrist_signal` equals the new one bitwise on no-confidence and all-confident tracks, random
  gappy ones included, at 23.976 and 59.946 fps.
- **Test:** 28; each of 8 guards reverted fails >= 1 test. The body-scale test was strengthened
  after a mutation passed: a median hid a minority of bad frames, so the bad frames are now the
  majority.
- **Review: SHIP.** Interpolation cannot inflate a peak (chord speed <= mean real speed);
  no off-by-one. Findings: coverage printed on two denominators; labels trusted unchecked
  (KeyError, truncation, no bounds); the scored CLI test asserted only exit code (shape, not
  substance); chance floor meaningless on clips < 20 s; a bridge-conf test passed a
  copy-left-neighbour mutation; peaks on interpolated frames not flagged.
- **Security:** pose and label files are loaded without pickle; labels re-validated (type,
  bounds, known labels, NaN/Infinity refused); output paths from the local clip name only.

### Iteration 2
- **Fix:** all of the above. One run definition (`_hand_runs`/`_runs`) shared by the signal and
  its coverage, tested equal to where the signal is non-zero. Coverage on one denominator (all
  clip frames). CSV flags peaks on interpolated frames. Score JSON written; the CLI test asserts
  recall and precision 3/3 on planted punches. Floor refused under 20 s.
- **Real-world facts, own footage (unscored until labels exist):** IMG_4395 signal coverage
  71.4% (both-wrists frame rule) -> per hand L 81.6% R 73.3% -> + bridging L 82.9% R 75.6%;
  IMG_4406 L 100% R 97.7%. 41 and 42 punches found with the V6-tuned rule.
- **Test:** hands 38, labels 50, ingest 109, env 36, pipeline 260, scripts 38.
- **Residual, accepted:** (a) identity switches: if `select_person` jumps to another person for a
  few frames, their confident joints read as a punch; low risk solo, real in a gym, handled with
  two-boxer tracking in Phase 2. (b) A bridged id-gap next to a filled hand hole can chain to ~7
  synthetic frames in 8; such peaks are flagged in the CSV, not capped. (c) The rule's threshold
  is V6's (AlphaPose, 24 fps, hooks and uppercuts only): untuned for own footage until labels exist.

