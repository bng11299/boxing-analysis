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

## The numbers

Commands are in the "Commands" section (C0–C15), written as run from a clean clone's root.

| # | Claim | Value now | Cmd | Output | Recorded | Source | Status |
|---|---|---|---|---|---|---|---|
| 0 | Extent rule gate: per-clip window minus old window, IoU, V6 second half at finder detections | +0.103 [+0.099, +0.108], verdict YES; ratios 1.33 / 1.83 peak-widths | C1 | `results/repro/punch_extent.json` | +0.103 [+0.099, +0.108], YES (was +0.101 before #23) | #23 | match |
| 1a | ST-GCN++ pretrained, seed 0, held-out V2/V5/V9 (976 clips): accuracy / macro F1 | 0.8145 / 0.7207 | C2 | `results/repro/sweep/seed0.json` | 0.8145 / 0.7207 | #2, `results/sweep-2026-09-22/seed0.json` | match (every result field identical; note 7) |
| 1b | Saved checkpoint re-evaluated on the same 976 clips | 0.8145 / 0.7207; manifest's pinned sha256 equals the `.pt` file's (`32c1982d…`) | C4 (gate 1) | printed by C4 | 0.8145491803 / 0.7207303869 (manifest `held_out`) | #14 | match |
| 1c | Retrain seed 0 through `train_classifier.py`: reproduces and saves | 0.8145 / 0.7207; saved checkpoint byte-identical (sha256 `32c1982d…`), manifest identical | C3 | `results/repro/weights/stgcnpp-boxingvi-seed0.{pt,json}` | 0.8145 / 0.7207; sha256 `32c1982d…` | #14 | match |
| 1d | ST-GCN++ pretrained, mean of seeds 0–2 | 0.8118 / 0.7082 (seeds 0.8145, 0.8094, 0.8115 / 0.7207, 0.6956, 0.7084) | C2 ×3 | `results/repro/sweep/seed{0,1,2}.json` | 0.812 / 0.708 | `results/sweep-2026-09-22/` | match (all three files identical in results; note 7) |
| 1e | Random forest; floors: commonest type, static pose (accuracy / macro F1), seed 0 | RF 0.653 / 0.510; commonest 0.205 / 0.057; static pose 0.491 / 0.215 | C2 | `results/repro/sweep/seed0.json` | RF 0.653 / 0.510; commonest 0.205; static pose 0.491 | #2, README | match |
| 2 | Pose switch on V5+V9, 742 punches: A, Y_all, A−Y_all | A 0.896, Y_all 0.889, +0.007 [−0.009, +0.024]; Y_clip 0.826 | C4 | `results/repro/pose_switch.json` | same | #16 | match |
| 2v | Pose switch per video (A / Y_all on n) | V5 0.906 / 0.899 on 594; V9 0.858 / 0.851 on 148 | C4 | same | same | #16 | match |
| 3 | Wrist-speed finder, V6 test half (406 punches): recall, F1, chance floor | 0.741; 0.571 [0.472, 0.654]; 0.353 | C0 | `results/repro/punch_finder_v6.json` | 0.741; 0.571 [0.472, 0.654]; 0.353 | #4 | match (see note 1) |
| 4 | Classify at finder detections, V6: G−C accuracy (all; second half) | +0.071 [+0.045, +0.101] on 493; +0.080 [+0.043, +0.120] on 301 | C5 | `results/repro/classify_v6.json` | same | #23 | match |
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

**Suites:** all 18 green in a fresh `git clone` of main at `664a511` (no data present): analyse_clip
113, blur 177, classify 297, completeness_credited 95, e2e 74, env 36, frame_skips 222, hands 99,
ingest 183, labels 50, linking 266, olympic 29, pipeline 314, poseswitch 30, render_punches 66,
report 289, tracks 79, machine scripts 38.

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

## Commands

From the repo root, PowerShell, with the venv's `python`.

```
# C0  finder rule (row 3)
python run_punch_finder.py --data data --video data/rgb/V6.mp4 --out results/repro/punch_finder_v6.json
# C1  extent rule (row 0; needed by every classify step after #23)
python run_punch_extent.py --data data --classifier weights/classifier/stgcnpp-boxingvi-seed0.json --rule results/repro/punch_finder_v6.json --out results/repro/punch_extent.json
# C2  full baseline sweep, one seed (rows 1a, 1d, 1e); SEED = 0, 1, 2
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
