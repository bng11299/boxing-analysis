# Quality loop: `fix/o55-athlete-disjoint-split` (2026-10-27)

Task: make the held-out split and `--cv` athlete-disjoint after Brendan's skim (vault ISSUES-LOG
O55) found the same instructor in V2 (held out) and V3 (training). His choices, made before any
new number existed: hold out V2/V3/V5/V9 (the old set plus V3), group `--cv` folds by person, and
keep the pipeline checkpoint on its old V2/V5/V9 split.

## Acceptance criteria

1. Who is in each video is recorded in exactly one place (`data.PERSON_OF_VIDEO`) and covers
   exactly the ten videos. *Check:* test "the person map covers exactly the ten videos".
2. `split_by_video` never returns a split with a person on both sides, and refuses exactly the
   hold-outs that would. *Check:* the exhaustive test over all 510 hold-outs of the nine corpus videos.
3. An unknown video id, held out or in the corpus, is refused. *Check:* the `person_of` and
   `split_by_video` negative tests.
4. `run_baseline.py` on real data holds out V2/V3/V5/V9, and `--cv` groups by person. *Check:* the
   `main()` wiring tests (loader stubbed).
5. Nothing scores the pipeline checkpoint on V3 (which it trained on), and its reproduction still
   works. *Check:* `train_classifier`, extent rule, pose-switch gate 1 and the report fallback use
   `split_pipeline`; `reference_metrics` refuses a reference from another split.

Threat sketch (no network, auth or DB; the input is local data files): **assets**: the
integrity of reported numbers. **Entry points**: video ids read from the corpus, and `held_out`
arguments. **Trust boundary**: the loader and the split functions. **Top threats**: (1) a silent
leak, where an athlete straddles train and test and inflates the headline (this is O55 itself);
(2) a consumer scoring a checkpoint on clips it trained on, after a constant changes under it.

## Iteration 1

- **Build:** `PERSON_OF_VIDEO`, `person_of`, `person_key`, an athlete-disjoint `split_by_video`,
  `split_pipeline`, and `HELD_OUT_VIDEOS` set to V2/V3/V5/V9. `run_baseline` uses `person_key`. The
  pipeline consumers moved to `split_pipeline`. Commit `b65a043`.
- **Test gate:** green. Pipeline 314 → 348, all 20 Python suites and the machine-script suite pass.
  Mutation check in a throwaway copy, with three mutants (person check disabled; old hold-out
  restored; `run_baseline` back to the video key): all three killed (4, import-time and 1 failures).
- **Review gate** (code-reviewer agent): verdict FIX FIRST. One blocking finding: RESULTS.md and
  README still certified 0.8145 / 0.812 as reproduced by C2, which now runs the new split. Also
  non-blocking: stale "same channel / may not be athlete-disjoint" wording in the V6, pose-switch
  and end-to-end limits; `learning_curve.py` not recording its split; `reference_metrics` not
  checking the reference's held-out videos; a bare string `held_out` failing with a misleading
  message; a mutable person map; `try/except` checks that ran zero checks if nothing was raised; a
  held-out pre-check test that the corpus check could also satisfy; leftover "video-disjoint"
  wording.
- **Security gate** (changed files only): no injection surface, no secrets, no paths or writes
  added. Untrusted ids are validated at the boundary (`person_of` fails closed on unknown, empty or
  non-string ids). Errors name only video ids from the corpus. No findings above informational.

## Iteration 2

- **Fixes:** every review finding above. The V6 limits now say "same person as training video V1
  (O55): not athlete-disjoint". The pose-switch and end-to-end limits say V5/V9 are clean and V2 is
  not. `learning_curve.py` writes `val_videos`. `reference_metrics` refuses a reference whose
  `val_videos` is not the checkpoint's split (test fixture now records it, as the real reference
  does, plus two negative tests). A bare string hold-out is refused. The map is a
  `MappingProxyType`. Every `try/except` check has an `else` failure. The held-out pre-check test
  uses a corpus without V11. Project rules, README and docstrings are updated. RESULTS.md: see below.
