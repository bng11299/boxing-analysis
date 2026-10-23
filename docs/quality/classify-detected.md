## `feat/classify-detected` -- 2026-10-09

The classify step: a window of continuous pose around each punch the finder found, mapped
into BoxingVI's clip space, named by the saved ST-GCN++. Acceptance criteria: (1)
`train_classifier.py` saves a checkpoint + manifest only if it reproduces the reported
0.8145 / 0.7207 held-out; (2) `classify_punches.py` writes one row per finder punch, typed
or `unclassified` with a reason, refusing a missing stance or another clip's CSV; (3) the
loader refuses a hash mismatch, a path out of its folder, a non-state-dict and a
code-carrying pickle whose code never runs; (4) `eval_classify_v6.py` measures the window's
cost on V6 against the commonest-type floor, with paired intervals; (5) all suites green in
a clean clone.

Threat sketch. Assets: classifier integrity; correctness of per-punch outputs. Entry points:
manifest JSON, checkpoint pickle, punches CSV, pose npz, labels JSON. Trust boundary: files
on disk between tools. Top threats: a swapped checkpoint executing code via pickle; a stale
CSV from another clip classified against the wrong pose.

Design facts, measured on the training clips only (3,778 of the 3,781): a clip is 9 real
frames (0.375 s) at 24 fps, its wrist-speed peak 3 frames in, torso 205 px of 1080, hips at
(0.48, 0.53) of the frame. No classifier had ever been saved: `run_baseline.py` scored and
discarded it, so `run_stgcnpp` gained `save_to`.

### Iteration 1
- **Test:** 87 checks; 7 guards reverted, 2 survived and got sharper tests: a `..` checkpoint
  name (refused later by `is_file`, so the name guard was untested) and `weights_only=False`
  (the malicious pickle was refused afterwards anyway); the pickle test now asserts its code
  never ran (a marker file).
- **Review: FIX FIRST.** B1: the training lead-in was `(argmax + 0.5) / fps`, but
  `np.gradient` is centred, so the peak is AT the frame where `detect` reports it: every
  window began half a training frame early, and at 24 fps `np.round`'s ties-to-even
  duplicated or dropped frames by the peak's parity. B2: the sampling test computed its
  expectation with the code's own formula, so it could not catch B1 (shape, not
  substance). Also: CSV check cannot tell two clips of the same rate and length apart;
  finder variant unchecked against the window; geometry unbounded from an unhashed manifest;
  `write_manifest` extras could replace pins; unbounded reads; V6 reading A cut from the
  bridged track; empty readings crash the eval; matched-but-unclassifiable dropped
  uncounted; `isdigit` accepts '²'.
- **Security:** pickle only under `weights_only=True` on hashed bytes; manifest field-by-field;
  checkpoint name a bare file name; CSV re-validated. Findings folded into the review list.
- **A training run already under way was stopped**: it would have written B1's geometry
  into the manifest.

### Iteration 2
- **Fix:** lead-in is `argmax / fps`; nearest frame with ties upward. Expected frames now
  worked by hand (60 fps: offsets -7 -5 -2 0 2 5 7 10 12), plus a round trip (corpus clips
  cut from a continuous 24 fps track; `cut_window` at the finder's peak returns exactly
  each clip's 9 frames) and a half-frame lead-in at both peak parities. Geometry capped at
  25 frames; manifest extras cannot replace pins; reads bounded; CSV must be named for the
  pose's clip; rule variant must equal the window's; ASCII digits; unclassifiable matches
  counted; eval A from the unbridged track; reference must match `t_out` and the
  pretrained hash. Declined: a per-window torso (it would shrink as the body turns into a
  hook); stated as a limit in `classify.py` instead.
- **Test:** 100 checks; B1 reverted fails 2, ties-to-even 1, every new guard >= 1. One
  equivalent mutant: removing the manifest size check still refuses an oversized manifest,
  because the bounded read truncates it to invalid JSON.
- **Gate reproduce:** clean clone of `6fa5540`: classify 100, pipeline 260, hands 38,
  labels 50, ingest 109 -- the working tree's numbers.

### Iteration 3 -- real data (2026-10-09)
- **Reproduction gate held twice.** First retrain: 30 epochs, because results/stgcnpp.json
  records `epochs: 30` (the other arms) and no `stgcnpp_epochs`, while its log shows ST-GCN++
  ran 20. It scored 0.7951 / 0.7141 and was **refused, nothing saved**. Fix: the epochs come
  from the reference, which must record them (the 09-22 seed-0 sweep run: 20 epochs, same
  confusion matrix). Second retrain: **0.8145 / 0.7207, exactly the reported run**; saved,
  sha256 `32c1982d91a9...`, geometry lead-in 0.125 s, 9 frames at 24 fps.
- **V6 window cost (results/classify_v6.json), 685 hooks and uppercuts:** A corpus-style span
  clips 0.712 accuracy; B the pipeline's window at the span's wrist-speed peak 0.504; C the
  window at the finder's own matched detections 0.594 on 493. A - B **+0.209 [+0.169,
  +0.247]**; B - C -0.010 [-0.032, +0.014]: finder timing costs nothing measurable, the
  window costs 21 points. All three beat the commonest-type floor (0.378).
- **Ablation (scratch, not committed):** span frames scaled and centred 0.708 (canonicalisation
  is harmless); fixed 0.375 s window raw 0.450, scaled 0.504; fixed 0.5 / 0.625 / 0.75 s raw
  0.523 / 0.536 / 0.543. V6 spans are a median 16 frames (0.67 s) against the training median
  9 (0.375 s, mostly jabs and crosses): a fixed window cuts hooks and uppercuts short, and no
  fixed length recovers the span-cut accuracy. **Open design decision:** estimate each punch's
  start and end (not just its peak), or classify several windows and keep the most confident.
- **Own footage, provisional (stance unconfirmed, outputs in scratch only):** IMG_4395 42
  punches, IMG_4406 41; typed 64% / 46% lead uppercuts, 5% jabs. Type-vs-wrist hand agreement
  54.8% and 56.1% against chance 48.2% and 50.9%: barely above chance. The step runs end to
  end; its types on own footage are not credible until the window is fixed and the pose
  switch (feat/pose-switch) is measured.

### Iteration 4 -- the punch extent (2026-10-11, vault ADR-0014)
Brendan chose to estimate each punch's start and end from the wrist-speed curve. Acceptance:
(1) the extent beats the fixed window on V6's untouched second half, with an interval, in
overlap with the labelled span and in classifier accuracy; (2) a rule that did not is refused;
(3) every number says which half of V6 it is from; (4) own footage re-run; (5) suites green.
- **Why V6 lost 21 points:** not hooks being longer than jabs. In the training clips every
  type runs 0.3-0.4 s; V6's labelled punches run 0.67 s. It is slow instructional footage.
- **Tried, dropped:** walking out from the peak while speed holds a fraction of it. Typical
  error a frame, tails 8-42 frames against labels of 11-23. First-half accuracy 0.538.
- **Built:** width of the speed peak at half height (sub-frame), times ratios from the training
  clips whose peak is whole: 1.33 widths before, 1.83 after (1,582 clips kept, 2,196 cut by
  the clip's edge). `predict_proba` had to learn windows of different lengths (a real bug:
  `np.stack` on the first run).
- **Test:** 149 checks; 10 mutants, one survived (truncating every window to 9 frames: the test
  compared the function with itself, and a randomly initialised classifier barely reacts to
  its input, 7e-6). Replaced with a stand-in model whose answer is a known function of every
  frame.
- **Review: FIX FIRST.** B1 an extent over 25 training frames lost its tail, or the peak itself,
  to a head-first trim. B2 the width walk treated a wrist dropping below confidence as the end
  of the punch. B3 `every_labelled_punch` now read the labelled frames but was still described
  as "a perfect finder" (0.754 vs 0.601 on V6). B4 two new tests were vacuous (a bound any
  fixed window passes; a ratio recomputed with the code's own call). Also: the censoring is a
  selection that leans the ratios long; "nothing is fitted to V6" was false as written; the
  gate could barely fail, so add a control.

### Iteration 5 -- review fixes and the control (2026-10-11)
- **Fix:** cap around the peak in the rule's proportion, flagged (14 of 685 V6 extents); the
  walk stays inside the hand's own signal, censored at its edge, flagged in the CSV note and the
  sidecar; two perfect-finder readings under their own names; a CSV row that is not that hand's
  speed peak in this pose is refused; a 24.000 fps span read whole against 23.976. Wording
  corrected: V6 chose the half-height level and the rule's form.
- **Test:** 172 checks; 9 new mutants, one survived (a wrong-hand CSV row raised for another
  reason) and got its own fixture. Tests now plant a slow punch (read over twice the frames),
  scale the ratios with frames planted before the peak (7/5), and check IoU 1 and 1/2.
- **Real-world facts, V6 second half (406 punches), `results/classify_v6.json`:**

  | reading | accuracy |
  |---|---|
  | A labelled spans | 0.754 |
  | B fixed 0.375 s window at the span's peak | 0.500 |
  | D extent at the span's peak | 0.608 |
  | F control: one fixed 0.46 s window (the median extent) | 0.631 |
  | C fixed window at the finder's detections (301) | 0.598 |
  | E extent at the finder's detections (301) | 0.684 |

  D - B **+0.108 [+0.071, +0.148]**; E - C +0.086 [+0.047, +0.126]; A - D +0.145 [+0.099,
  +0.192]; **D - F -0.022 [-0.054, +0.010]**. Overlap with the label 0.647 vs 0.505 (+0.142
  [+0.131, +0.153]).
- **What the control says:** the gain is the longer window, not following each punch. The
  clips kept by the censoring are a median 0.434 s against 0.375 s for all measurable clips, so
  the ratios lean long, which is what V6 rewards. Per-punch tempo has not earned its place on
  V6; whether it does on mixed-tempo footage is what `feat/pose-switch` measures (`Y_long`).
- **Own footage (orthodox):** median punch length 0.42 and 0.40 s, the old window's length,
  so nothing moved: IMG_4395 26 of 41 lead uppercuts, IMG_4406 17 of 41; type-vs-wrist hand
  agreement 53.7% and 48.8% against chance 47.6% and 50.8%. 3 and 0 extents cut short by a
  wrist dropout, 1 and 1 capped. The window was V6's problem, not this footage's.
- **Residual, accepted:** the IoU intervals resample punches independently (blocks would be
  wider); the same punch reads 5-12% wider at 24 fps than at 60; `read_extent` is not bound to
  a classifier hash (the ratios do not depend on its weights).

### Iteration 6 -- the confidence rule (2026-10-12)
Found by `feat/pose-switch`, not by this branch's own gates.
- **Real-world facts, V5 and V9, 742 held-out punches, the same frames read two ways**
  (`results/pose_switch.json`): AlphaPose 0.896; YOLO11-pose with every detected joint
  **0.889** (-0.007 [-0.009, +0.024]); YOLO with joints under confidence 0.3 blanked, as
  `classify._window` did, **0.348**. The rule blanked about 5% of the joints there (mostly a
  hidden ear or eye), 7% and 20% on own footage.
- **Fix:** `CLASSIFY_MIN_CONF = 0`: the classifier is shown every joint the pose model detected;
  a joint with confidence 0 is still blanked. The finder keeps 0.3. 175 checks; both guards
  reverted fail (keeping undetected joints: 5; back to 0.3: 2).
- **Own footage, same finder and classifier:** IMG_4395 26 of 41 lead uppercuts -> 13 jabs, 12
  crosses, 9 lead hooks, 6 rear uppercuts, 2 lead uppercuts; hand agreement 53.7% -> 71.4%
  (chance 49.7%). IMG_4406 -> 16 jabs, 8 crosses, 7 rear uppercuts, 5 lead hooks, 5 lead
  uppercuts; 48.8% -> 82.9% (chance 50.3%). Still unmeasured against labels.
- **Why three iterations of gates missed it:** every synthetic fixture had confident joints;
  V6 carries no joint confidence, so its evaluation could not see the rule; and the one check
  on own footage, hand agreement at chance, was read as the window's fault (iteration 3) and
  then as the pose model's (iteration 5). The rule itself was never a decision: it was the
  finder's default, reused. What caught it was a reading specified before the run for another
  reason (`Y_all`, "AlphaPose drops few").
- **V6 control paired with the old window:** F - B +0.131 [+0.094, +0.167].

### Iteration 7 -- one window per clip (2026-10-13, ADR-0014 amended by Brendan)
Acceptance: (1) every punch in a clip is read over one length, the median of the clip's own
punch extents; (2) the rule file's gate measures that cut as the pipeline runs it, and a file
that gated another cut is refused; (3) the evaluations name which numbers the choice was made
on; (4) own footage re-run; (5) suites green in a clean clone.
- **Build:** `classify.clip_cut` / `clip_window`; `classify_punches.py` uses them and writes the
  cut to the sidecar; `eval_classify_v6.py` readings F (median over labelled punches) and G
  (median over every finder detection, windows at its matched detections: the pipeline as it
  runs); `run_punch_extent.py` gates on G's form. 188 checks; 7 mutants, all caught.
- **Review: FIX FIRST** (code sound). B1 a rule file from before this change carried the same
  `beats_fixed_window` flag for the per-punch cut and passed. B2 the gate measured the cut with
  a perfect finder (0.46 s), not as the pipeline runs (0.44 s from 1,084 detections). B3 the
  docstrings overstated: "measured and lost" is true on V5/V9 and a tie on V6; 0.823 came from a
  precursor estimator; the censoring rule is unmeasured; and the choice was made on the test
  videos without saying so. B4 no negative tests for an empty CSV, the --labels refusal, an even
  count, a tie, the fallback's length.
- **Fix:** the rule file names the cut it gated (`gated_cut`), `read_extent` requires it; the
  gate is `clip_cut` over all finder detections, windows at matched detections; wording
  corrected and both evaluations' limits now say the per-clip cut was chosen on V5, V9 and V6's
  second half; lengths in exact seconds (peak width x ratios), not rounded frame spans; whole
  punches must be a strict majority to leave the censored ones out; under 5 punches the clip has
  no tempo to read and gets the training clips' median window (`source: "training"`,
  unmeasured); --labels with no punches refused before anything is written; a labelled punch
  with no wrist speed is not scored "at its peak". 202 checks; 7 more mutants, one survivor
  (the no-speed guard) given its own fixture.
- **Real-world facts, V6 second half** (`results/classify_v6.json`): labelled span 0.754; per
  clip 0.621 (F), per punch 0.608 (D), old window 0.500 (B). F - B **+0.121 [+0.081, +0.163]**;
  D - F -0.012 [-0.044, +0.020]. As the pipeline runs: G 0.678, E per punch 0.684, C old window
  0.598. G - C **+0.080 [+0.043, +0.120]**; E - G +0.007 [-0.023, +0.037]. Clip length 0.458 s
  from the 685 labelled punches, 0.440 s from the finder's 1,084 (1 left out).
- **Gate** (`results/punch_extent.json`): overlap with the labels at the finder's 301 matched
  second-half detections, per clip 0.613 against old window 0.512, +0.101 [+0.097, +0.106]. YES.
- **Own footage (orthodox, unlabelled):** clip length 0.41 s and 0.39 s (3 and 0 punches left
  out of the median). IMG_4395: 16 jabs, 10 crosses, 6 lead hooks, 6 rear uppercuts, 4 lead
  uppercuts; hand agreement 61.9% (per punch 71.4%; chance 49.4%). IMG_4406: 17 jabs, 7
  crosses, 7 rear uppercuts, 6 lead hooks, 4 lead uppercuts; 80.5% (per punch 82.9%; chance
  50.4%). The consistency check moved slightly AGAINST the change on both clips (4 and 1
  punches). The benchmarks, with labels, favour it. Labels on these clips decide.
- **Residual, accepted:** V5, V9 and V6's second half are no longer clean tests of this cut;
  the censoring rule and the too-few fallback are unmeasured; intervals resample punches
  independently.

### Iteration 8 -- Codex's two findings (2026-10-21, `e2459d5`)
Acceptance: (1) the under-5 fallback reads exactly `cut_window`'s frames, checked by the frames
read, not the stored length; (2) a repeated (frame, hand) CSV row is refused; (3) decide and
record whether threshold and local-peak checks belong here; (4) no row the finder emits is
refused; (5) suites green here and in a clean clone; one Astra review.
- **Finding 1, and why the obvious fix was wrong.** `clip_cut` stored `geom.duration_s` (9/24 s,
  a sample count) as `length_s`, which `clip_window` treats as first-to-last: 10 frames at 24 fps
  where `cut_window` reads 9. Correcting the length to (9 - 1)/24 s fixes 24 fps but, read through
  `clip_window` + `cut_span`, reads **8** frames at 25, 29.97 and 59.94 fps: the start snaps to a
  frame and `cut_span` reads only whole training steps. The old one-step-too-long length had
  masked that (9 frames at phone rates, 10 only at multiples of 24). So the fallback, which IS
  the old fixed window, is now read by `cut_window`'s own frames through the new
  `classify.read_clip_window`; a clip's own cut goes through `clip_window` + `cut_span` exactly
  as gated (tested equal). `classify_punches.py` and `eval_classify_v6.py` call it, and the
  recorded span is now the frames read (it had been 2 frames wider at 59.94 fps). A manifest
  whose lead-in would put the peak after the window's last frame is refused.
- **Finding 2, and the decision.** Duplicates refused in `read_punches`. Threshold and local-peak
  checks: **added here**. They are facts about this branch's own finder (`punchfind.detect`:
  `find_peaks` on the faster hand's speed, `height` = threshold), they close the same hole
  (rows the finder cannot emit setting the clip's tempo), and leaving them to #22 would leave
  #14's pipeline open on its own. Here they read the faster-hand curve the finder peaks on; #22's
  version reads each hand's curve because it adds a per-hand finder. Necessary, not sufficient:
  a peak the minimum gap suppressed, or a CSV with rows deleted, still passes.
- **Tests:** 202 -> 221 checks. Frames compared by index coded into the nose: 47-55 at a 24 fps
  peak at 50; every peak tried at 23.976, 24, 25, 29.97, 30, 59.94 and 60 fps. Negative: five
  copies, two copies, one frame up and one down the slope, threshold one ulp above the slowest
  punch (control at exactly its speed passes: `>=`, as `find_peaks`), unknown cut source, late
  lead-in. Two existing expectations encoded the bug (0.375 s length, 22-frame CLI span) and
  were corrected to 8/24 s and 20 frames. **Mutants: 9, all caught**, including "length fixed but
  read through `cut_span`" (fails the 7-rate check).
- **Real-world facts, own footage at 59.946 fps** (outputs to scratch; finder re-run with this
  branch): IMG_4406 and IMG_4395, 41 and 42 finder rows, all accepted by the new checks;
  classified CSVs **byte-identical** to `49473f9`. Their first 4 rows alone (the fallback): 9
  frames per punch before and after, the length-only fix 8; 7 of 8 types unchanged, one
  low-confidence punch moved (rear hook 0.53 -> rear uppercut 0.36).
- **Gates:** test green (221/260/38/109/50/36 + 38 PowerShell), the same counts in a clean clone
  at `e2459d5`. Review (self): one overstated comment fixed. Security: changed files only; the
  CSV boundary and manifest checks are stricter and fail closed; the duplicate set is bounded by
  MAX_ROWS; errors name file, line and frame only.
- **Astra (Codex, read-only, effort medium): FIX FIRST, both items in stacked branches, none in
  this diff.** (a) `eval_pose_switch.py:166` on #16 reads through `clip_window` + `cut_span`:
  route it through `read_clip_window` in #16's fix pass (its V5/V9 cuts have far more than 5
  punches, so no reported number moves). (b) Merging into #22: keep #22's per-hand check, this
  branch's duplicate guard, and `read_clip_window` at both read sites. No false refusal found
  (plateaus, segment edges, threshold equality, same signal code in both tools). Astra could
  not run Python in its sandbox ("Access is denied"); the runtime evidence above is this
  session's.
- **Residual, accepted:** `run_punch_extent.py` still takes fallback spans from `clip_window`
  (description only; V6 never falls back). At exactly 60 fps `cut_window`'s x.5 ties fall either
  way by float noise (pre-existing). The fallback remains unmeasured on labelled footage.

