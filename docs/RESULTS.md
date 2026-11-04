# Reported numbers, reproduced on merged main

Every number the project reports, re-run from one codebase: `origin/main` at `664a511` (PRs #3–#25
merged), on 2026-10-22/23, CPU only (Ryzen 5 9600X, Windows 11, Python 3.13.15, torch 2.13.0+cpu,
scikit-learn 1.9.0, numpy 2.2.6, ffmpeg 9.0.1). Each number was first measured on its own branch;
this file says whether it survives the merge.

- **Inputs** are the gitignored ones every machine holds: `data/` (BoxingVI, the V6 video, Olympic
  Boxing, the `data/own/` labels), `weights/` (classifier checkpoint and manifest, NTU-120 backbone)
  and the existing pose files under `results/ingest/`. No pose was re-extracted and nothing here
  needed the GPU, so no row is "pending GPU".
- **Outputs** go to `results/repro/` (gitignored). Only aggregate numbers appear in this file: no data,
  clips, pose files, weights, or per-item Olympic rows. Nothing Olympic-derived leaves `results/`.
- **Status:** *match* = equal at the recorded precision. For deterministic runs the whole results
  file was also compared with the earlier one where it still exists, and differences are named.
  *changed* = the value differs; the cause is given and the new value is the one to report. No code
  was changed to reach an old number.
- **Fixed randomness:** every bootstrap and every audit or completeness sample is seeded in code
  (`--seed`, default 0; `n_boot` 2000); training uses `--seed`. The same command gives the same file.
- **Three files feed later rows** and are passed explicitly every time: the finder rule
  (`results/repro/punch_finder_v6.json`, row 3), the extent rule (`results/repro/punch_extent.json`,
  row 0, which must be re-made after #23 changed `classify.GATED_CUT`), and
  `results/repro/pose_switch.json` (row 2, read by gate 4 of row 5).
- **Code changes made at merge time** that could move a number (vault STATE.md, checked against main's
  code): #16/#20 read the under-5-punch fallback through `classify.read_clip_window`; #20's gate 4 uses
  `reference_keep` (no old 0.3-rule filter); #22's `detect(eligible=)` drops ineligible peaks before the
  minimum gap; point labels are matched with `POINT_TOL_S`; #23's nearest-middle attribution now scores
  every matched detection's type. Only the last one moved a reported number (row 6a).
- **Athlete-disjoint split (O55, 2026-10-27).** V2 and V3 show the same instructor, so the old
  hold-out V2/V5/V9 was not athlete-disjoint. Rows 1f–1i are the headline now: held out V2/V3/V5/V9,
  run on `fix/o55-athlete-disjoint-split` (`b65a043`). Rows 1a, 1d and 1e stay as the record of the
  old split. Note 13 explains what changed and which other rows it touches.

## The numbers

Commands are in the "Commands" section (C0–C15), written as run from a clean clone's root.

| # | Claim | Value now | Cmd | Output | Recorded | Source | Status |
|---|---|---|---|---|---|---|---|
| 0 | Extent rule gate (ratios from the checkpoint's training side, which includes V1 = V6's instructor; note 13): per-clip window minus old window, IoU, V6 second half at finder detections | +0.103 [+0.099, +0.108], verdict YES; ratios 1.33 / 1.83 peak-widths | C1 | `results/repro/punch_extent.json` | +0.103 [+0.099, +0.108], YES (was +0.101 before #23) | #23 | match |
| 1a | *Old split, superseded by 1g.* ST-GCN++ pretrained, seed 0, held-out V2/V5/V9 (976 clips; not athlete-disjoint, O55): accuracy / macro F1 | 0.8145 / 0.7207 | C2 at `8b25ebf` | `results/repro/sweep/seed0.json` | 0.8145 / 0.7207 | #2, `results/sweep-2026-09-22/seed0.json` | match (every result field identical; note 7); **superseded** (note 13) |
| 1b | Saved checkpoint re-evaluated on the same 976 clips (its own split, `split_pipeline`; V2 not athlete-disjoint) | 0.8145 / 0.7207; manifest's pinned sha256 equals the `.pt` file's (`32c1982d…`) | C4 (gate 1) | printed by C4 | 0.8145491803 / 0.7207303869 (manifest `held_out`) | #14 | match |
| 1c | Retrain seed 0 through `train_classifier.py`: reproduces and saves | 0.8145 / 0.7207; saved checkpoint byte-identical (sha256 `32c1982d…`), manifest identical | C3 | `results/repro/weights/stgcnpp-boxingvi-seed0.{pt,json}` | 0.8145 / 0.7207; sha256 `32c1982d…` | #14 | match |
| 1d | *Old split, superseded by 1f.* ST-GCN++ pretrained, mean of seeds 0–2, held-out V2/V5/V9 | 0.8118 / 0.7082 (seeds 0.8145, 0.8094, 0.8115 / 0.7207, 0.6956, 0.7084) | C2 ×3 at `8b25ebf` | `results/repro/sweep/seed{0,1,2}.json` | 0.812 / 0.708 | `results/sweep-2026-09-22/` | match (all three files identical in results; note 7); **superseded** (note 13) |
| 1e | *Old split, superseded by 1h.* Random forest; floors: commonest type, static pose (accuracy / macro F1), seed 0, held-out V2/V5/V9 | RF 0.653 / 0.510; commonest 0.205 / 0.057; static pose 0.491 / 0.215 | C2 at `8b25ebf` | `results/repro/sweep/seed0.json` | RF 0.653 / 0.510; commonest 0.205; static pose 0.491 | #2, README | match; **superseded** (note 13) |
| 1f | **Headline, athlete-disjoint (O55).** ST-GCN++ pretrained, mean of seeds 0–2, held-out V2/V3/V5/V9 (1,787 clips; trained on 2,970 from V1/V4/V7/V8/V10): accuracy / macro F1 | **0.7722 / 0.6331** (seeds 0.7795, 0.7851, 0.7521 / 0.6344, 0.6525, 0.6123) | C2 ×3 | `results/sweep-athlete-disjoint-2026-10-27/seed{0,1,2}.json` | new, 2026-10-27 | `fix/o55-athlete-disjoint-split` | **new: replaces 1d** (note 13) |
| 1g | Same, seed 0 | 0.7795 / 0.6344; per-class F1 cross 0.906, jab 0.853, lead hook 0.634, lead uppercut 0.541, rear hook 0.210, rear uppercut 0.663 | C2 | `…/seed0.json` | new | same | **new: replaces 1a** |
| 1h | Same split: random forest; floors (commonest type, static pose), accuracy / macro F1 | seed 0: RF 0.681 / 0.500; commonest 0.310 / 0.079; static pose 0.501 / 0.271. Mean of seeds 0–2: RF 0.674 / 0.494; static pose 0.488 / 0.264 (commonest is the same every seed) | C2 ×3 | same | new | same | **new: replaces 1e** |
| 1i | Same split: interpretability gap, ST-GCN++ pretrained − random forest, macro F1 (95% CI), seeds 0 / 1 / 2 | +0.135 [+0.104, +0.164]; +0.159 [+0.128, +0.189]; +0.124 [+0.095, +0.155]: every interval excludes zero | C2 ×3 | same | new | same | **new** |
| 2 | Pose switch on V5+V9, 742 punches: A, Y_all, A−Y_all | A 0.896, Y_all 0.889, +0.007 [−0.009, +0.024]; Y_clip 0.826 | C4 | `results/repro/pose_switch.json` | same | #16 | match |
| 2v | Pose switch per video (A / Y_all on n) | V5 0.906 / 0.899 on 594; V9 0.858 / 0.851 on 148 | C4 | same | same | #16 | match |
| 3 | Wrist-speed finder, V6 test half (406 punches): recall, F1, chance floor | 0.741; 0.571 [0.472, 0.654]; 0.353 | C0 | `results/repro/punch_finder_v6.json` | 0.741; 0.571 [0.472, 0.654]; 0.353 | #4 | match (see note 1) |
| 4 | Classify at finder detections, V6 (same person as training video V1: not athlete-disjoint; note 13): G−C accuracy (all; second half) | +0.071 [+0.045, +0.101] on 493; +0.080 [+0.043, +0.120] on 301 | C5 | `results/repro/classify_v6.json` | same | #23 | match |
| 4r | Readings C / E / G (all V6; second half) | 0.613 / 0.688 / 0.684; 0.605 / 0.694 / 0.684; F−G +0.004, +0.003 | C5 | same | same | #23 | match |
| 5a | End to end V5+V9, finder: precision (lower bound), recall, count ratio | 0.633 [0.552, 0.706]; 0.585 [0.530, 0.641]; 0.92 | C6 | `results/repro/e2e_v5v9/pipeline_v5v9.json` | same | #20 | match |
| 5b | End to end: type accuracy on matched detections; jab recall | 0.811 [0.754, 0.868] on 434; 0.32 (84/264) | C6 | same | same | #20 | match |
| 5c | End to end: found and named right | 352 of 742 (0.474) | C6 | same | 352 of 742 | #20 | match |
| 5d | Blinded label audit: far detections real; controls; inside / outside labelled stretch | 19 of 50 [0.26, 0.52]; 9 of 10; 19 of 37; 0 of 13 | C7 | `results/repro/e2e_v5v9_audit/pipeline_v5v9.json` | same | #20 | match (see note 2) |
| 6a | Own footage IMG_4406, ±0.2 s: found; real; type on matched (floor) | 35/47; 35/41; **0.857 of 35 (floor 0.314)** | C8 | `results/repro/own_tol0.2/IMG_4406.*score.json` | 35/47; 35/41; 0.886 of 35 (floor 0.343) | #22 | **changed: #23 attribution** (note 3) |
| 6b | Own footage IMG_4395, ±0.2 s | 35/40; 35/42; 0.600 of 35 (floor 0.371) | C8 | `results/repro/own_tol0.2/IMG_4395.*score.json` | same | #22 | match |
| 6c | Own footage, ±0.1 s (IMG_4406; IMG_4395) | 28/47, 28/41, 0.821 of 28; 27/40, 27/42, 0.667 of 27 | C9 | `results/repro/own_tol0.1/` | same | #22 | match |
| 6d | Framing marks flagged as punches, ±0.2 s (bounds) | 1 to 1 of 5; 1 to 1 of 2 | C8 | `results/repro/own_tol0.2/` | same | #22 | match |
| 6e | Own footage IMG_5835 (a full 3:04 round), ±0.2 s: found; real; type on matched (floor) | 117/149; 117/162; **0.744 of 117 (floor 0.248)** | C8 | `results/repro/own_tol0.2/IMG_5835.*score.json` | new, 2026-10-28 | `feat/img5835-own-scores` | **new** (note 11) |
| 6f | Own footage IMG_5835, ±0.1 s | 92/149; 92/162; 0.793 of 92 (floor 0.196) | C9 | `results/repro/own_tol0.1/` | new, 2026-10-28 | `feat/img5835-own-scores` | **new** (note 11) |
| 6g | Own footage pooled over IMG_4406, IMG_4395 and IMG_5835 — **ONE boxer**: found; real; type on matched (sums of the per-clip counts in 6a–6c, 6e, 6f) | ±0.2 s: 187/236 (0.792); 187/245 (0.763); 138/187 (0.738). ±0.1 s: 147/236 (0.623); 147/245 (0.600); 114/147 (0.776) | C8, C9 | the per-clip files above | new, 2026-10-28 | `feat/img5835-own-scores` | **new: one boxer, not a population** (note 11) |
| 6h | Framing marks flagged as punches (bounds): IMG_5835; pooled over the three clips | ±0.2 s: 3 to 3 of 8; 5 of 15. ±0.1 s: 1 to 1 of 8; 2 of 15 | C8, C9 | same | new, 2026-10-28 | `feat/img5835-own-scores` | **new** (note 11) |
| 7a | Olympic: punches credited to a thrower; convention test | 2,423 of 4,605 (52.6%); nearest wrist is the labelled hand in 77.6% of 1,794 (629 ties) | C10 | `results/repro/olympic_attribution.json` | 2,363 of 4,490 (53%); 78% of 1,756 (607 ties) | #10 (quality-loop, 10-07 run) | **changed: input** (note 4) |
| 7b | Olympic linking: credited punches on boxer A or B; median both-identified | 65.4% (50.7% with half held out); 57% | C11 | `results/repro/olympic_linking.json` | same | #11 | match |
| 7c | Linked completeness: unexplained bursts; per exposure-minute vs own labels | 5,766 of 7,984; 39.4 vs 10.8 | C12 | `results/repro/olympic_completeness.json` | same | #11 | match |
| 7d | Credited-fragment completeness, min 1; min 2 (unexplained; possible twins) | 7,318 of 10,077, 177; 4,342 of 6,227, 66 | C12 | `results/repro/olympic_completeness_credited{,_min2}.json` | same | #12 | match |
| 7e | Camera frame skips | 3,367 across 29 videos (32.3 per minute) | C13 | `results/repro/olympic_tracks/*.framechange.npz` | 3,367, about 32 per minute | #13 | match (files identical) |
| 7f | Skip gate, unexplained per 1,000 exposure frames vs far field, bands 3–5 / 6–10 / 11–25 (min 1; min 2); 0–2 band unmasked | 1.16 / 1.13 / 1.04; 1.19 / 1.15 / 1.07; 0–2 band unmasked 3.36× (min 1; 3.32× at min 2) | C14 | `results/repro/olympic_completeness_credited{,_min2}_skips{,measure}.json` | same (#13 gives the 0–2 band for min 1 only) | #13 | match |
| 7g | Unexplained per exposure-minute, unmasked → masked (min 1; min 2) | 45.352 → 44.120; 45.205 → 44.066 | C14 | same | 45.35 → 44.12; 45.21 → 44.07 | #13 | match |
| 7h | Mask cost: exposure removed; own-labelled punches hidden (min 1; min 2) | 8.7%, 104 of 2,407 (4.3%); 8.3%, 72 of 1,741 (4.1%) | C14 | same | body: 8.5%, 4.3%; log: 8.7% / 8.3%, 4.3% / 4.1% | #13, quality-loop | match (note 5) |
| 7i | Sample rows still unexplained with skips (min-2 sample) | 29 of 40 (11 drop out) | C14 | `results/repro/olympic_completeness_credited_min2_sample_skipcheck.csv` | 29 of 40, 11 | #13 | match |
| 8 | Face blur, own footage: frames with no face found | not re-run | C15 | — | 89 of 2,353 (3.8%); 843 of 2,659 (31.7%) | #21 | **not re-run: needs Brendan's OK** (note 6) |
| 9a | Per-boxer path, Olympic, linker (#11) as the identity input (ungated): identity files written and read back through every check; finder punches identical to linked completeness | 29 of 29 videos both; 7,984 punches on A + B (= row 7c), 10,077 on credited fragments (= row 7d) | C16 | `results/repro/olympic_boxers.json` | new, 2026-10-25 | `feat/per-boxer-report` | **new** (note 9) |
| 9b | Punches the linker identity leaves unassigned (on credited fragments in neither chain) | 3,772 of 11,756 found (32.1%) | C16 | same | new | `feat/per-boxer-report` | **new** (note 9) |
| 9c | Labels with no owner among boxers A and B | 3,020 of 4,605 (65.6%): 2,182 credited to nobody, 838 to a fragment in neither chain; owned 1,585 of 2,423 credited (65.4%, = row 7b) | C16 | same | new | `feat/per-boxer-report` | **new** (note 9) |
| 9d | Punches found per labelled punch: per boxer against its owned labels (conditional on the linker's identity); whole video; credited fragments | per boxer 5.04 (left 4.34, right 6.23; per boxer-video relative error median +4.32, IQR +3.31 to +5.53, 58 boxer-videos); whole video 2.55; credited fragments 4.16 | C16 | same | new | `feat/per-boxer-report` | **new: NOT a count error** (note 9) |
| 10a | Ring-floor feet filter, DEVELOPMENT videos only (4): credited punches whose credited person is dropped; non-boxer person-frames removed, margins −30 to +60 px (selection bar: at most 0.5% dropped) | best, +60 px: 2.0% (8 of 392; worst video 3.4%) and 28.2%; at 0 px 4.6% and 75.4%. **No margin qualifies: stopped before any held-out run** | C17 | `results/olympic_ring/dev_grid.json` | new, 2026-10-27 | `feat/ring-mask` | **new: stopped** (note 10) |
| 10b | Same, +60 px: linked unexplained per exposure-minute baseline → filtered (ratio, 95% video bootstrap); link coverage; A/B person-frames on uncredited ids | 36.837 → 35.494 (0.96 [0.92, 1.13]); 69.1% → 62.4%; 24.3% → 26.5% | C17 | same | new, 2026-10-27 | `feat/ring-mask` | **new: design numbers, not evidence** (note 10) |
| 12a | Punch combinations (gap 1.0 s), own footage IMG_4406 + IMG_4395, pooled: named labelled combinations recovered exactly (every punch found, matched and named right) vs the shuffled-type floor's 97.5th percentile (gate bar 1, strictly greater) | 2 of 19 (0.105) vs 0.105 (floor mean 0.020; 2,000 permutations, seed 20261028): **bar 1 FAILS** | C19 | `results/combinations_gate.json` | new, 2026-10-28 | `feat/combinations` | **new: gate FAIL** (note 12) |
| 12b | Headline statements, found vs labels: favourite combination (bar 2); punch that starts combinations (bar 3) | IMG_4406 jab-jab within {cross-lead hook, jab-cross, jab-jab}, starter jab = jab; IMG_4395 jab-jab vs jab-cross, starter jab = jab: **bar 2 FAILS**, bar 3 passes | C19 | same | new, 2026-10-28 | `feat/combinations` | **new** (note 12) |
| 12c | Why: labelled combinations with a gap under the finder's 0.4 s minimum; structure recovered (punches only, no types); combinations and punches in them, found vs labelled | IMG_4406 11 of 12, 1 of 12, 9 / 35 of 41 vs 12 / 40 of 47; IMG_4395 6 of 7, 4 of 7, 9 / 38 of 42 vs 7 / 37 of 40 | C19 | same | new, 2026-10-28 | `feat/combinations` | **new** (note 12) |

**Suites:** all 18 green in a fresh `git clone` of main at `664a511` (no data present): analyse_clip
113, blur 177, classify 297, completeness_credited 95, e2e 74, env 36, frame_skips 222, hands 99,
ingest 183, labels 50, linking 266, olympic 29, pipeline 314, poseswitch 30, render_punches 66,
report 289, tracks 79, machine scripts 38.

## Linker identity, measured by hand (option A, 2026-10-28)

Gate `docs/gates/linker-join-check.md` (committed `ec52160` before any code or check; bar ADR-0018). Brendan
answered same / different / can't tell for 100 seeded, gap-stratified joins of the #11 linker (default parameters,
as the per-boxer report uses it) on the 23 held-out Olympic videos; the window showed only the two fragments.
Supersedes "identity error not measurable" (rows 9a-9d, note 9) and the option C record for the linker. Same-session
hand labels, one annotator; not reproducible from a clone (the answers are Olympic-derived and stay local).

| # | Claim | Value now | Cmd | Output | Recorded | Source | Status |
|---|---|---|---|---|---|---|---|
| 11a | Linker joins that connect two DIFFERENT people (D / decided, Wilson 95%), held-out, n = 100 | **0.653 [0.555, 0.740]** (64 D, 34 S, 2 U; 98 decided); p\* 0.0088 -> **measured, unreliable** | C18 | `results/linker_join_check/result.json` | new, 2026-10-28 | `feat/linker-join-check` | **new** |
| 11b | Same, by gap: short (<= median) / long | 0.510 [0.375, 0.644] (25 of 49) / 0.796 [0.664, 0.885] (39 of 49) | C18 | same | new | `feat/linker-join-check` | **new: descriptive** |
| 11c | Implied share of credited punches on the wrong person (two-person model, independent joins; an estimate) | 0.457 [0.454, 0.459] (saturates at 0.5) | C18 | same | new | `feat/linker-join-check` | **new: estimate** |
| 11d | Wrong joins per chain-minute (video-cluster bootstrap, 10,000 draws) | 5.78 [4.82, 6.77]; frame 1,030 joins | C18 | same | new | `feat/linker-join-check` | **new: descriptive** |

## Notes

1. **Finder F1 interval.** The lower bound is 0.4722. A brief listed 0.473, which was the value
   before #4's bootstrap fix (`docs/quality-loop.md`, "F1 CI lower bound 0.473 → 0.472"); #4's
   body says 0.472. The whole results file is identical to the earlier one.
2. **Audit re-score.** The judgments (`audit/judgments.csv`) are bound row by row to the digest of
   the sample they were judged on, and the run refuses to score unless the `audit_key.json` in its
   out-dir has the same digest. C7 copies the judged run's key into the new out-dir; it passes only
   because this run drew the identical sample. Without `--judgments`, C6's file is identical to the
   earlier `pipeline_v5v9.json` except for the audit block and one description string.
3. **IMG_4406 type accuracy, 0.886 → 0.857.** #22 measured it on its branch, where
   `classify_punches.py --labels` paired detections with labels by `punchfind.match` (greedy, spans
   taken by end). #23, merged first, switched that pairing to `match_nearest` (least total distance
   to each label's middle). The finder output is unchanged (same 35 matches, same 41 detections), but
   one fast combination's detection now goes to the other label of the pair. That changes both the
   accuracy (30 of 35, was 31) and the commonest-type floor (11 of 35, was 12). Checked by swapping
   only the attribution on the same CSV: greedy gives 0.886 (floor 0.343) exactly; nearest gives
   0.857 with either the widened-span middle or the clicked frame. Report 0.857. IMG_4395 and the
   ±0.1 s readings do not move.
4. **Olympic attribution, 2,363 → 2,423 credited.** #10's figures come from the first real run
   (10-07), when GH088416's tracks file held only its first 3,000 of 12,524 frames
   (`docs/quality-loop.md`, "28 of 29 tables complete"). That table was completed before #11, and
   #11's numbers already use 2,423 of 4,605 (4,605 is every labelled punch). Same code, more input.
   By target, too-close-to-call shares are within 0.003 of #10's (head 0.412, body 0.608, block 0.638,
   miss 0.464).
5. **Mask cost.** #13's body says the mask removes "8.5% of exposure"; its own log records 8.7%
   (min 1) and 8.3% (min 2), which is what this run gives. The 4.3% hidden is the min-1 figure.
6. **Face blur not re-run.** It needs no pose model (the default deface policy runs CenterFace
   on CPU), but `blur_faces.py` writes only under `~/comps-participant-data/` unless
   `--allow-outside-participant-root` is passed, and never into a git checkout. Writing Brendan's own
   footage into the participant folder would mix the two, and this session's permission check
   refused the flag. To re-run, Brendan runs C15 himself (or approves it); it needs `deface==1.5.0`
   in the venv. The two counts are in each run's `.blur.json` (`frames_without_face`, `frames`).
7. **Training is deterministic here.** All three sweeps (seeds 0, 1, 2) match the 09-22 files in every
   result field: all eight arms' accuracy, macro F1, per-class F1, confusion matrices and intervals.
   They differ only in `train_seconds` and two metadata fields added since (`environment`,
   `pretrained_from.spec`). Same CPU, same library versions, torch's default thread count; another
   CPU or torch build may differ in the last digits.
8. **Not compared:** wall-clock times (the linking solver's "at most 0.09 s" reads 0.28 s here, on a
   CPU shared with training). The Olympic attribution and linking files bind tracks by file name
   only (`video_bound: false`), as the tables predate fingerprints.

9. **Rows 9a–9d (per-boxer report, 2026-10-25) measure the plumbing and the coverage, not identity.** The
   Olympic labels name the target and the thrower's own hand only; nothing pins which boxer threw a punch
   (`docs/quality/olympic-swap-rate.md`). So no ground-truth identity exists, no perfect-identity run can be
   built, and the share of per-boxer count error that is identity error cannot be measured; an identity swap
   moves punches and their owned labels together and is invisible to 9d. Punch types are not scored either:
   the labels are not the six BoxingVI types and the stances are unknown, so 9a–9d run the finder only. 9d is
   not the finder's count error: two thirds of the labels have no owner among A and B (9c), and the finder's
   own over-count on V6 is 1.6 (row 3). Run on 2026-10-25 against the main checkout's
   `results/punch_finder_v6.json` (the rule rows 7c/7d were made with: both reproduce exactly), not
   `results/repro/`; the identity files went to the gitignored `results/olympic_boxers/`.

10. **Rows 10a–10b (ring-floor filter, 2026-10-27) are a stop on the development videos, not a held-out
   result.** The gate (`docs/gates/ring-floor-filter.md`, committed before any floor or number) drops whole
   track ids whose box bottom ("feet") lies outside a hand-clicked ring floor in most of their rows, and
   chooses the margin on ADR-0013's four development videos: the largest removal among margins that drop at
   most 0.5% of credited punches, else stop. None qualified (2.0% at the best), so `FROZEN_MARGIN_PX` was
   never set and the held-out run refuses to start; nothing was tuned further. The lost credits are four
   tracks whose box bottoms sit 70–150 px beyond the FAR edge of the floor (a boxer at the far ropes with
   hidden feet, or a punch credited to a spectator behind; the labels cannot say which): the same depth
   ambiguity that ruled out a pixel mask. The linked burst rate barely moves even when 50–80% of non-boxer
   rows go (+0 to +30 px), and linking coverage falls. The evaluator's dev baseline equals
   `olympic_completeness.py` on both paths (linked 36.837, credited 45.465 per exposure-minute). Floors:
   27 videos clicked in 329 s; GH188416 and GH208416 left out (the camera moves). Floors and reports stay
   in the gitignored `data/` and `results/`.
12. **Rows 12a–12c (punch combinations, 2026-10-28) are a pre-registered FAIL.** Gate
   `docs/gates/combinations.md`, committed (`909486e`) before any combination code or number; rule and gap (1.0 s,
   Brendan's choice over 0.75 s, vault ADR-0019) fixed, nothing tuned. Detections regenerated with C8's commands
   reproduce rows 6a/6b exactly (35/47 and 35/41; 35/40 and 35/42; type 0.857 and 0.600 of 35). Bar 1 fails by
   the strict inequality: the observed recall equals the floor's 97.5th percentile. The finder's 0.4 s minimum gap
   is the main cause: 17 of the 19 labelled combinations contain a gap under 0.4 s, which no found combination can
   reproduce exactly by construction. At about one punch a second of shadowboxing the 1.0 s gap also chains most
   punches into long runs (found: 35 of 41 and 38 of 42 punches in combinations, longest 9 and 14). The reports
   keep the combination block and state this failure in it; the result applies only to this classifier, finder
   rule and gap (a finder change such as PR #40 voids it).

11. **Rows 6e–6h (IMG_5835, 2026-10-28): the third own clip, and the first scored with every choice fixed
   before its labels existed.** A full shadowboxing round (184.5 s, 11,056 frames at **59.937 fps**, not the
   59.946 of the first two clips, so the Label Studio config's `frameRate` was set to 59.937 for this
   project; the importer's two frame checks passed, offset +1). Brendan labelled it on 10-27/28 in 56.3 min
   of stopwatch time, two sittings: 157 marks, 149 punches (143 single-frame clicks, 6 start-to-end spans)
   and 8 "unclear" marks meaning FRAMING, scored as not a punch (`--unclear not-a-punch`). The finder rule,
   the extent rule, the per-clip window and both point tolerances were all set before these labels were
   made (the 0.2 s tolerance was chosen on IMG_4406/4395 after seeing their scores,
   `docs/quality/own-footage-scoring.md`), so 6e/6f are a cleaner test of them than 6a–6c. **All three clips
   are the same boxer (Brendan, orthodox)**: 6g is a pooled count over one
   person, not an estimate for boxers in general. Pooled rows are sums of the per-clip counts (type correct
   = accuracy × n per clip: 30 + 21 + 87 at ±0.2 s, 23 + 18 + 73 at ±0.1 s); no pooled floor is given,
   since a commonest-type floor does not add across clips (per clip: 0.314, 0.371, 0.248 at ±0.2 s). Run
   on `origin/main` at `42e1c04`, outputs in the gitignored `results/repro5835/` with the rule files
   byte-identical to `results/repro/`'s (sha256 `60d92c2d…` finder, `2a44b74d…` extent); the same run
   re-scored IMG_4406 and IMG_4395 and reproduced rows 6a–6d exactly at both tolerances.

13. **Why rows 1a/1d/1e changed (O55, 2026-10-27).** Brendan watched all ten videos. V2 (held out)
    and V3 (training) show the same instructor, and so do V1 and V6. Every other video is a
    different person. He chose the new hold-out before any number on it existed: the old set plus
    V3, with nothing else chosen. `data.PERSON_OF_VIDEO` records the people, and `split_by_video`
    now refuses any hold-out that puts one person on both sides. The old rows are kept as the record
    and reproduce with C2 at `8b25ebf`, the last commit before the change.
    - **The drop is not the size of the leak.** 0.812 → 0.772 accuracy (−0.040) and 0.708 → 0.633
      macro F1 (−0.075) mix three effects that this run does not separate: V2's instructor no longer
      trains; training lost V3's 811 clips (3,781 → 2,970, −21%); and the test set changed (976 →
      1,787 clips, 20.5% → 37.6% of the corpus, with V3 making up 45% of it).
    - **The floors moved because the test mix moved.** Cross is 31% of the new test set, so
      commonest-type accuracy goes 0.205 → 0.310. Compare a model with its own split's floor, never
      across splits. The random forest went up (0.653 → 0.681 accuracy, seed 0) while ST-GCN++ went
      down, so the interpretability gap narrows from +0.211 to +0.135 macro F1 (seed 0). It still
      excludes zero on all three seeds (row 1i).
    - **The test set is 37.6% of the clips, not about 20%** like the old split and the paper's.
      That is the price of holding out whole people without choosing anything new. Its intervals
      are narrower for it, and its model is trained on less.
    - **Rows unaffected:** 2, 2v and 5a–5d score V5 and V9, each a person seen in no other video,
      with the checkpoint, which stays on its own split (`split_pipeline`). So they were already
      athlete-disjoint. 1b and 1c still reproduce, but their 976 clips include V2, whose instructor
      the checkpoint trained on (V3). The earlier `--cv 3` runs (`results/cv-2026-09-22/`) are
      unaffected: their video folds [V1], [V2, V3, V8, V10] and [V4, V5, V7, V9] are exactly the
      person folds `--cv` now builds.
    - **Rows now known not to be athlete-disjoint:** 4 and 4r classify V6 with a checkpoint trained
      on V1, the same person. Row 0's extent ratios come from that training side. Row 3 (the finder)
      uses no trained model. Fixing these would mean retraining the checkpoint, which was declined
      for now (it would move every pipeline row).
    - **Run:** `b65a043`, the three seeds concurrently on the CPU above (6 torch threads each), about
      3 h wall-clock. **Re-run 2026-11-02** at `3f7d04c` (this branch after merging main) the same
      way: all three seeds reproduced every result field exactly (accuracies, F1s, intervals, gaps,
      split videos). Only `train_seconds` differed. The environment was the same (Python 3.13.15,
      torch 2.13.0+cpu, 6 threads). So the run is deterministic on this split too (as note 7 showed
      for the old one), and main's changes do not move rows 1f-1i.

## Commands

From the repo root, PowerShell, with the venv's `python`.

```
# C0  finder rule (row 3)
python run_punch_finder.py --data data --video data/rgb/V6.mp4 --out results/repro/punch_finder_v6.json
# C1  extent rule (row 0; needed by every classify step after #23)
python run_punch_extent.py --data data --classifier weights/classifier/stgcnpp-boxingvi-seed0.json --rule results/repro/punch_finder_v6.json --out results/repro/punch_extent.json
# C2  full baseline sweep, one seed (rows 1f-1i; at 8b25ebf, rows 1a, 1d, 1e); SEED = 0, 1, 2
python run_baseline.py --data data --stgcnpp both --pretrained weights/stgcnpp_ntu120_xsub_hrnet_j.pth --stgcnpp-epochs 20 --seed SEED --out results/repro/sweep/seedSEED.json
# C3  retrain the pipeline's classifier; saves only if it equals the reference (row 1c)
python train_classifier.py --data data --pretrained weights/stgcnpp_ntu120_xsub_hrnet_j.pth --out results/repro/weights/stgcnpp-boxingvi-seed0.pt --reference results/sweep-2026-09-22/seed0.json --rule results/repro/punch_finder_v6.json
# C4  pose switch; gate 1 re-scores the saved checkpoint (rows 1b, 2, 2v)
python eval_pose_switch.py --data data --poses results/ingest --classifier weights/classifier/stgcnpp-boxingvi-seed0.json --extent results/repro/punch_extent.json --out results/repro/pose_switch.json
# C5  classify at finder detections on V6 (rows 4, 4r)
python eval_classify_v6.py --data data --classifier weights/classifier/stgcnpp-boxingvi-seed0.json --rule results/repro/punch_finder_v6.json --extent results/repro/punch_extent.json --out results/repro/classify_v6.json
# C6  end to end V5 + V9 (rows 5a-5c)
python eval_pipeline_v5v9.py --data data --poses results/ingest --classifier weights/classifier/stgcnpp-boxingvi-seed0.json --stance orthodox --rule results/repro/punch_finder_v6.json --extent results/repro/punch_extent.json --reference results/repro/pose_switch.json --out-dir results/repro/e2e_v5v9
# C7  same, scoring the existing blind judgments (row 5d; see note 2)
New-Item -ItemType Directory -Force results/repro/e2e_v5v9_audit
Copy-Item results/e2e_v5v9/audit_key.json results/repro/e2e_v5v9_audit/audit_key.json
python eval_pipeline_v5v9.py ...C6's arguments... --out-dir results/repro/e2e_v5v9_audit --judgments results/e2e_v5v9/audit/judgments.csv
# C8  own footage, as shipped (+-0.2 s points); CLIP = IMG_4406, IMG_4395 (rows 6a, 6b, 6d), IMG_5835 (rows 6e, 6g, 6h)
python find_punches.py --pose results/ingest/CLIP.pose.npz --labels data/own/CLIP.labels.json --unclear not-a-punch --rule results/repro/punch_finder_v6.json --out-dir results/repro/own_tol0.2
python classify_punches.py --pose results/ingest/CLIP.pose.npz --punches results/repro/own_tol0.2/CLIP.punches.csv --classifier weights/classifier/stgcnpp-boxingvi-seed0.json --stance orthodox --labels data/own/CLIP.labels.json --unclear not-a-punch --rule results/repro/punch_finder_v6.json --extent results/repro/punch_extent.json --out-dir results/repro/own_tol0.2
# C9  own footage at +-0.1 s: the same two main()s with punchfind.POINT_TOL_S = 0.1 (no CLI flag exists; this is how #22 scored it)
python -c "import sys; sys.path[:0]=['src','.']; from boxingvi import punchfind; punchfind.POINT_TOL_S=0.1; import find_punches; sys.exit(find_punches.main(sys.argv[1:]))" ...C8's find arguments... --out-dir results/repro/own_tol0.1
python -c "import sys; sys.path[:0]=['src','.']; from boxingvi import punchfind; punchfind.POINT_TOL_S=0.1; import classify_punches; sys.exit(classify_punches.main(sys.argv[1:]))" ...C8's classify arguments, punches from own_tol0.1... --out-dir results/repro/own_tol0.1
# C10 Olympic crediting (row 7a); T = a folder holding the 29 <id>.tracks.npz (here a copy, results/repro/olympic_tracks)
python olympic_attribute.py --data data/olympic --tracks T --out results/repro/olympic_attribution.json
# C11 linking (row 7b)
python olympic_link.py --data data/olympic --tracks T --out results/repro/olympic_linking.json
# C12 completeness: linked, credited min 1, credited min 2 (rows 7c, 7d)
python olympic_completeness.py --data data/olympic --tracks T --rule results/repro/punch_finder_v6.json --out results/repro/olympic_completeness.json --sample-out results/repro/olympic_completeness_sample.csv
python olympic_completeness.py ...same... --credited --out results/repro/olympic_completeness_credited.json --sample-out results/repro/olympic_completeness_credited_sample.csv
python olympic_completeness.py ...same... --credited --min-punches 2 --out results/repro/olympic_completeness_credited_min2.json --sample-out results/repro/olympic_completeness_credited_min2_sample.csv
# C13 camera frame skips (row 7e): decodes each video once, writes T/<id>.framechange.npz (~1 h on CPU)
python olympic_framechange.py --data data/olympic --tracks T
# C14 masked and measured completeness, sample check (rows 7f-7i); N = 1, 2 (file suffix _min2 for 2)
python olympic_completeness.py --data data/olympic --tracks T --rule results/repro/punch_finder_v6.json --credited --min-punches N --skips --out results/repro/olympic_completeness_credited[_min2]_skips.json --sample-out results/repro/olympic_completeness_credited[_min2]_skips_sample.csv
python olympic_completeness.py ...same, with --skips-measure in place of --skips (they cannot be combined)... --out results/repro/olympic_completeness_credited[_min2]_skipsmeasure.json --sample-out results/repro/olympic_completeness_credited[_min2]_skipsmeasure_sample.csv
python olympic_completeness.py ...same, --skips... --skipcheck results/repro/olympic_completeness_credited_min2_sample.csv --skipcheck-out results/repro/olympic_completeness_credited_min2_sample_skipcheck.csv
# C16 per-boxer path on Olympic Boxing, linker identity (rows 9a-9d)
python olympic_boxers.py --data data/olympic --tracks T --rule results/repro/punch_finder_v6.json --out results/repro/olympic_boxers.json --identity-dir results/repro/olympic_boxers
# C17 ring-floor filter, development videos (rows 10a-10b): backgrounds + stillness, Brendan's clicks, the margin grid
python olympic_ring.py prepare --data data/olympic
python olympic_ring.py click --floors data/olympic_ring/floors.json
python olympic_ring.py evaluate --split dev --data data/olympic --tracks T --floors data/olympic_ring/floors.json --rule results/repro/punch_finder_v6.json --out results/olympic_ring/dev_grid.json
# C18 hand-checked linker joins (rows 11a-11d): draw once, Brendan answers in the window, score once (answers stay local)
python linker_join_check.py sample --data data/olympic --tracks T
python linker_join_check.py check --data data/olympic --tracks T
python linker_join_check.py score
# C19 punch combinations gate (rows 12a-12c): C8's find + classify into a fresh folder, then score
python score_combinations.py --clip IMG_4406 --clip IMG_4395 --punches-dir results/repro/own_tol0.2 --labels-dir data/own --pose-dir results/ingest --classifier weights/classifier/stgcnpp-boxingvi-seed0.json --out results/repro/combinations_gate.json
# C15 face blur, own footage only (row 8; not run here, note 6); CLIP = IMG_4406, IMG_4395
python blur_faces.py --video data/own/CLIP.mov --out <a folder outside any checkout>/own-CLIP.blurred.mp4 --allow-outside-participant-root
```

## How to reproduce from a clean clone

1. `git clone` the repo, check out `main`, and install with `.\scripts\setup-env.ps1` (CPU torch).
   For row 8 only: `pip install --no-deps deface==1.5.0` into the same venv.
2. Put the gitignored inputs in place, as on any working machine (`docs/MACHINE-SYNC.md`):
   `data/Skeleton_data/`, `data/Annotation_files/`, `data/rgb/V6.mp4`, `data/olympic/`,
   `data/own/IMG_4406.*`, `IMG_4395.*` and `IMG_5835.*`; `weights/stgcnpp_ntu120_xsub_hrnet_j.pth` and
   `weights/classifier/stgcnpp-boxingvi-seed0.{pt,json}` (the manifest pins the checkpoint by sha256,
   `32c1982d…`); `results/ingest/{V2,V5,V9,IMG_4406,IMG_4395,IMG_5835}.pose.npz` and the 29
   `results/ingest/olympic/<id>.tracks.npz` (pose comes from `run_ingest.py` on the GPU machine);
   for C3 and C7, `results/sweep-2026-09-22/seed0.json` and `results/e2e_v5v9/` from the run they check.
3. Run the suites: every `python tests/test_*.py`, and
   `powershell -ExecutionPolicy Bypass -File tests\test_machine_scripts.ps1`. All 18 must pass.
4. Run C0, C1 and C4 first, in that order: they make the files the other steps read. Then any order.
   C2 takes about 70 min per seed and C3 about 35 min on this CPU when run alone; C13 about an hour.
   Everything else takes seconds to five minutes.
5. Compare each printed value with "Value now".

## Pre-registered negatives (identity and background, 2026-10-22 to 10-28)

N1 and N2 each had a gate committed before any rule code or number. N3's gate was committed with its
code, before any floor, margin or feet-filter number (after a pixel-mask check on the development videos,
recorded in the gate). Each result is reported as it came out: nothing was tuned on a held-out video
afterwards, and none is iterated further. N4 had no gate, because no existing label can score it; it was later measured with new hand-checked labels (row 11a). All four are Future Work, not adopted methods. They sit here, apart from
the numbers table, so later re-runs of that table do not touch them. The sparring-video results (N2) are
aggregate counts only, from the three consented 10-22 videos: no frame, file name or person is recorded.

| # | Question | Gate (pre-registered) | Numbers | Source | Status |
|---|---|---|---|---|---|
| N1 | Headgear corner colour: do two co-present credited punchers read opposite corners? (Olympic Boxing) | G1: opposite ≥ 0.90 of decided pairs, ≥ 100 decided pairs, 95% video-cluster bootstrap lower bound ≥ 0.80 (more than 1% empty bootstrap draws also fails), on the 23 held-out videos (`docs/gates/appearance-identity.md`, `c1b8b9c`; rule frozen `9d18ca6`, run once). G2 (linking) only if G1 passes | **277 of 333 decided pairs opposite, 83.2%**; Wilson [0.788, 0.868]; bootstrap [0.756, 0.880]. Share ✗, pairs ✓, lower bound ✗ (0 of 10,000 draws empty). kam4 87.6% (269/307), kam2 30.8% (8/26). Decided 35.8% of 931 co-present credited pairs. G2 not run | #29 (merged) | **negative (gate failed)** (note N1) |
| N2 | Glove colour: who is who, moment by moment, in the sparring videos | Bars 1–3 on EACH of two TEST videos separately: (1) accuracy ≥ 0.90, Wilson lower bound ≥ 0.80, ≥ 30 decided moments; (2) unsure ≤ 0.30; (3) ≤ 1 swap per 3-min round and correct share above both baselines (ByteTrack alone; the #11 linker). Bar 4 (fails safe on a same-colour video) not testable: no such video exists (`docs/gates/glove-identity.md` on `feat/glove-identity`, `0e95deb`, addendum `60a3e29`; frozen `cc48822`, each TEST scored once) | TEST (v2): verdict "separable", **28 of 44 decided, 28 of 28 right**, 0 swaps, unsure **0.36**: bar 1 ✗ (28 < 30 decided), bar 2 ✗, bar 3 ✓ (0.64 vs best baseline 0.30). TEST2 (v1): "not separable", **0 of 50** decided: bars 1–3 ✗ (unsure 1.00; 0 vs 0.57). DEV (v3, design data): "not separable", 0 of 59 | #31 (draft; gate result posted on the PR 2026-10-28) | **negative (gate failed)** (note N2) |
| N3 | Ring-floor filter: does dropping everyone outside the ring help two-boxer tracking? (Olympic Boxing) | Choose the margin on the 4 development videos: the largest removal among margins dropping ≤ 0.5% of credited punches, else stop before any held-out run (`docs/gates/ring-floor-filter.md`, `ddf6399`) | Best margin (+60 px) drops 2.0% of credited punches (8 of 392); no margin qualifies. **Stopped on the development videos; no held-out run**, by the gate's own rule. Pixel mask dropped earlier on dev: on kam4 even the most generous boxer-safe region blacks out 0% of bystanders. Summary: rows 10a–10b; full grid: `docs/quality/ring-floor-filter.md` | #36 (merged) | **negative (gate failed: stopped on dev)** (note 10, note N3) |
| N4 | Olympic ID-swap rate of the #11 linker | None committed: no existing label can score it (new identity labels could) | **Not measurable.** The 29 annotation files (4,609 punch tracks) record target × the thrower's own hand only; every `attributes` list is empty and nothing names a corner or a boxer. #29's co-present pairs stay true under an A/B swap, so they cannot detect one. Decision (Brendan, 2026-10-24): option C, no held-out swap rate reported (`docs/quality/olympic-swap-rate.md`). **Superseded by row 11a:** new hand-checked identity labels (C18, 2026-10-28) measured it: 0.653 [0.555, 0.740] of linker joins connect different people, so the linker is unreliable | #32 (merged); #39 | **not measurable from existing labels; measured in row 11a (0.653, unreliable)** |

N1. **Why headgear was tried, and where it fails.** An eye check of a development pair showed the torso
   rule (ADR-0013) reading the vests correctly, but both boxers wore red vests; the corner colour is on the
   headgear and gloves. The headgear read agrees far more often on one
   camera group (kam4, 87.6%, still under 0.90) than on the other (kam2, 30.8% on only 26 decided pairs: its
   blue kit sits below 0.3 saturation, and when kam2 decides it mostly reads both people red). On the Olympic
   development videos, gloves scored only about 58–60% opposite in #29's design checks, so the evidence
   was against gloves before the sparring test (N2) began.

N2. **Why gloves failed, and what the one success shows.** On two of three videos the rule never found
   a glove-colour split. On DEV its two clusters were grey/white against black, with every hue bin near
   zero. About 51% of the pixels it read were skin, so the read disc is probably not landing on the
   glove. Where it did split (TEST), it never named a
   boxer wrong (28 of 28) and beat both baselines (ByteTrack 0 of 44, linker 13 of 43), but it spoke on
   too few moments to pass. Even a pass would have been reported only as "bars 1–3 passed on two held-out
   videos; bar 4 not testable": two videos of the same pair of boxers, in the same gym and light, are not
   evidence that the method generalises. Ground-truth labels: Brendan, 10.5 min in all, under a protocol
   that does not show the labeller any method's output.

N3. **Why masking stopped.** The cameras sit at a ring corner at head height, so a boxer at the far ropes
   and the crowd behind them share pixels; only depth separates them, and in one image the cue to depth is
   where a person's feet are. All 8 credits lost at +60 px are on tracks whose box bottoms sit 70–150 px
   beyond the far edge of the floor (a boxer with hidden feet, or a punch credited to a spectator; the labels
   cannot say which). The linked burst rate barely moved (36.837 → 35.494 per exposure-minute at +60 px,
   ratio 0.96 [0.92, 1.13]; it rose at 0 and −30 px), and link coverage fell (69.1% → 62.4%). Details in
   note 10.
