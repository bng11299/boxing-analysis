## `feat/nearest-attribution` -- 2026-10-20

`feat/e2e-v5v9` found that `punchfind.match` (greedy, spans by end) hands a fast jab-cross's one
detection to the jab, and added `match_nearest` to its own script. This branch moves it into
`punchfind` and makes every score that reads a matched detection's TYPE or OVERLAP attribute with it:
`classify_punches.py --labels` ("matched detections"), `eval_classify_v6.py` readings C, E, G
(`finder_pairs`) and `run_punch_extent.py`'s gate (`matched_inside`, overlap at matched detections; not
named in the task, switched because it reads the same pairs as reading G and would otherwise disagree
with it). `punchfind.match` stays for counts. Stacked on `feat/e2e-v5v9`.
Acceptance: (1) one implementation, in punchfind; eval_pipeline_v5v9.py keeps no copy; (2) each of the
three consumers attributes a jab-cross detection to the cross, by a test through its own code path,
each failing when that consumer is reverted to `punchfind.match`; (3) counts unchanged (asserted inside
every call); (4) V6 re-run, C/E/G and G - C reported against the greedy numbers; the extent gate re-run
and its verdict reported; (5) suites green in a clean clone.

Threat sketch. Assets: the honesty of the V6 and own-footage type numbers. Entry points: unchanged
(labels JSON, V6 annotation xlsx, punches CSV); no new file, network or subprocess path. Trust
boundary: files between tools. Top threats: a solver answer that is not a maximum matching silently
lowering recall (guarded: count checked against `punchfind.match` on every call); an O(spans x
detections) cost matrix on very long labelled footage (measured, accepted below).

### Iteration 1
- **Baseline first.** Greedy runs of `eval_classify_v6.py` and `run_punch_extent.py` from this branch's
  base reproduced the stored `results/classify_v6.json` and `results/punch_extent.json` exactly.
- **Build:** `punchfind.match_nearest` (moved; now also refuses tol < 0 itself); the three consumers;
  provenance `"matching": "nearest"` in `classify_score.json`, `"finder_attribution": "nearest"` in
  classify_v6.json and (dropped in iteration 2) punch_extent.json.
- **Test:** the matcher's unit tests (jab-cross, count first, 200 random equal-count cases, 150-case
  brute-force optimality, empty inputs) moved from test_e2e.py to test_hands.py, plus overlapping
  hooks as V6 has them, sort order, tol < 0 with and without detections, and a solver stub that drops a
  hit (must raise). test_classify.py: the IMG_7777 fixture relabelled as a jab ending 3 frames before
  the right hand's 14 s peak and a cross around it, through `classify_punches.main --labels`, with a
  control that `punchfind.match` gives it to the jab; `finder_pairs` and `matched_inside` on a hook
  pair at 24 fps. test_e2e.py pins `split_detections`' default and the absence of a local copy.
  hands 50, classify 211, e2e 72, pipeline 260, labels 50, ingest 109, poseswitch 16, env 36: all pass.
- **Guards reverted, one at a time (9):** match_nearest returning the greedy pairs (fails 4 in hands, 3
  in classify, 1 in e2e); each consumer back to `punchfind.match` (classify_punches, finder_pairs,
  matched_inside, split_detections: each fails its test); tol < 0 check removed (fails); count-first
  cost removed (the internal assertion raises); the max-matching assertion removed (the solver stub
  test fails; it failed nothing before that test was added); the provenance field dropped (fails).
- **Review (code-reviewer): FIX FIRST.** B1: `classify.GATED_CUT` unchanged although the gate now
  measures other pairs ("change it whenever ... how the gate measures it ... changes"), and `read_extent`
  ignored the new `finder_attribution` field, so the 13 Oct greedy-gated `results/punch_extent.json`
  still loaded everywhere as certified. Also: pair order changed (span-end -> span index) under
  `paired_gain`'s positional bootstrap; docstrings said nearest-per-detection where the code minimises
  total distance among maximum matchings; "same count" where the matched SPANS can differ; a test
  comment implying unchanged random draws; two missing blank lines. Done well: one matcher, pinned; a
  greedy-fails control on every jab-cross fixture; count checked on every call. Stress (reviewer):
  4000 x 4500 over 200k frames and 300 mutually overlapping spans near frame 1e9, counts equal.
- **Security:** no new entry point, file path, subprocess or network use; inputs are frame integers and
  spans already validated by `load_labels` / `load_v6_spans`. Distances are multiples of 0.5 and the
  infeasible cost stays far below 2^52, so the count-first ordering is exact. One availability note:
  the cost matrix is dense, spans x detections: 35 MB on V6 (685 x 1085), 680 MB at 5000 x 8000
  (a 3.5 h labelled video at V6's density). Accepted (residual below).

### Iteration 2
- **Fix:** `GATED_CUT` now names the attribution ("..., each attributed by nearest middle
  (punchfind.match_nearest)"), so the greedy-gated rule file is refused, with a negative test using the
  old string (reverted: that test fails); the redundant `finder_attribution` field dropped from
  punch_extent.json; docstrings and comments corrected; the order effect measured and commented.
- **Test:** hands 50, classify 212, e2e 72, pipeline 260, labels 50, ingest 109, poseswitch 16, env 36.
  Against the base in a clean clone: hands 38 -> 50 (6 moved in, 6 new), e2e 76 -> 72 (6 out, 2 new),
  classify 202 -> 212 (10 new). 10 guards reverted, each fails a test.
- **Real-world facts** (V6, AlphaPose, seed-0 classifier; files in the session scratchpad, not results/):
  - Before switching, the greedy runs reproduced the stored classify_v6.json and punch_extent.json exactly.
  - The stored `results/punch_extent.json` is now refused by `eval_classify_v6.py` ("gated another cut").
    Re-gated: frac / lead / tail / variant identical to the stored file, verdict YES; eval_classify_v6
    with it reproduces the nearest run exactly.
  - Attribution: 28 of 493 greedy pairs change (V5: 115 of 356). The count is unchanged; 18 spans
    swap out (lead uppercut 9, lead hook 7, rear uppercut 2: the first punch of a combination) and 18
    swap in (rear uppercut 9, rear hook 7, lead uppercut 2); 10 more keep a match at another detection.

  | accuracy, greedy -> nearest | all V6 (493) | second half (301) |
  |---|---|---|
  | C old window at finder | 0.594 -> 0.613 | 0.598 -> 0.605 |
  | E per punch at finder | 0.663 -> 0.688 | 0.684 -> 0.694 |
  | G per clip at finder (the pipeline) | 0.663 -> 0.684 | 0.678 -> 0.684 |
  | G - C | +0.069 [+0.043, +0.099] -> **+0.071 [+0.045, +0.101]** | +0.080 [+0.043, +0.120] -> **+0.080 [+0.043, +0.120]** |
  | E - G | +0.000 -> +0.004 [-0.016, +0.024] | +0.007 -> +0.010 [-0.020, +0.040] |
  | F - G (perfect finder minus finder) | +0.020 [-0.016, +0.057] -> +0.004 [-0.028, +0.037] | +0.013 -> +0.003 [-0.043, +0.047] |

  G hand right 0.832 -> 0.858 (all), 0.864 -> 0.877 (second half). The matched set's commonest-type
  floor moves 0.369 -> 0.355 (all) because the spans differ. A, B, D, F do not use the finder: unchanged.
  - Extent gate, second half at finder detections (301): per-clip window IoU 0.613 -> 0.628, old window
    0.512 -> 0.525, per-clip minus old **+0.101 [+0.097, +0.106] -> +0.103 [+0.099, +0.108]**, verdict
    YES both ways. Greedy pairs re-sorted by span index give +0.101 [+0.097, +0.106] again: the shift is
    the attribution, not the bootstrap order.
- **Residual, accepted:** dense O(spans x detections) matrix (above); every V6 number here was already
  not a clean test of the per-clip cut (ADR-0014); after merge every checkout must re-run
  `run_punch_extent.py` once (5 s) before `classify_punches.py` / the evals will load the rule.

