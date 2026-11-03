# Quality loop: `feat/fast-combinations` (2026-10-28)

Gate first: `docs/gates/fast-combinations.md`, committed `246dcd5` before any code, any tuning video choice
or any held-out number. No arm has been tuned or scored yet: tuning waits for the O55 split (#38) and for
pose on the training videos (GPU laptop).

## Acceptance criteria
1. Arm 1 never counts a retraction, nor a frame whose window meets a missing joint, a pose gap or a run edge.
2. Arm 2 with equal gaps is the pooled rule exactly; a jab-cross the pooled rule merges is split by both arms.
3. A rule file with no `finder` block loads exactly as before; a malformed block is refused; the pooled
   rule's held-out score is never credited to an arm (`report._rule_of`).
4. Tuning refuses before the O55 split, never reads a held-out video, and refuses once the held-out run
   has started; the held-out run refuses a second run (whatever `--out` is), a rule file tuning did not
   write, and a baseline that does not reproduce RESULTS 5a-5c, checked before the arm is touched.
5. The scorer reproduces RESULTS row 5a on the real V5+V9 pose.

Threat sketch. Assets: the integrity of a pre-registered result. Entry points: rule, tune and pose files
on disk; command-line paths. Trust boundary: the loaders (files sit on disk between tools). Top threats:
(a) a stale or edited rule/tune file silently changes which arm or baseline is scored; (b) a held-out
number reaches tuning, or a second held-out run is made after seeing the first.

## Iteration 1
- **Test gate:** `tests/test_fast_combinations.py` 76/76, `tests/test_fastcombo_eval.py` 61/61; affected
  suites unchanged and green: hands 99 (includes the bitwise check of `wrist_signal` against its
  pre-refactor version, which guards the `_smoothed_runs` refactor), report 289, classify 297, e2e 74,
  analyse_clip 113. CPU-light: only these suites, one at a time (a seed sweep may run on this desktop).
- **Real-world check:** `fastcombo.counts` with the pooled rule on the real V5+V9 pose gives 686
  detections, 742 labels, 434 hits, jabs 84 of 264: RESULTS rows 5a-5b exactly. No arm was run.
- **Mutation:** 15 deliberate bugs, 14 caught. Survivor M5: `detect_rule` running arm 1 WITHOUT the
  extension gate passed, because the only `detect_rule` arm-1 test (the jab-cross) gives the same answer
  either way. Fixed with a `detect_rule` case where the gate decides (a jab that returns faster than it
  went out; the retraction-only clip).
- **Review (Astra/Codex, medium): FIX FIRST**, 3 blocking, all accepted:
  1. The held-out script checked found-and-named-right (RESULTS 5c) only after scoring the arm and running
     its end-to-end evaluation. Now every baseline check (finder 5a-5b, end to end 5c) runs first; the arm
     is never scored when one fails (main-level test with stubs asserts no arm detection, no arm
     end-to-end run, no marker, no output).
  2. The one-run and no-retune guards checked paths the user can change (`--out`, `--heldout`). Now a
     fixed marker, `results/fast_combinations_heldout.started` under the repo, is created atomically
     (`open("x")`) before the arm is scored; both scripts refuse while it exists, and a crashed run also
     leaves it. `--heldout` removed.
  3. Tuning accepted a pooled baseline with any threshold, which moves the admissibility reference. Now
     pinned to the registered 3.6238817996457016 (`fastcombo.BASELINE_THRESHOLD`); 3.62, 6.0 and 2.0 refused.
  - Astra's UNCERTAIN, one-frame holes made eligible after bridging: by design; the gate requires holes up
    to `BRIDGE_S` to be interpolated, as the signal does.
  - Astra could not launch the tests in its sandbox ("Access is denied"); it reviewed statically.
- **Mutation of the fixes:** M5 and one mutant per fix (arm scored before the 5c check, marker ignored,
  threshold unpinned, marker not atomic): 5 of 5 caught. **Total 19 of 19.**
- **Security (changed files only):**
  - Rule `finder` block → `parse_finder` checks type, finiteness, range and the exact key set; a bad
    block is refused naming the file. Validated by 15 negative cases.
  - `git` is called with an argument list (`subprocess.run([...])`, no shell); no secrets read or
    logged; no network.
  - The tune file is ours but sits on disk: the held-out script binds the rule file to it by sha256 and
    re-checks the arm, the rule, the gate commit, the held-out list and the baseline's sha256.
  - Pre-existing, not changed here: `find_punches.load_rule` accepts NaN/Infinity for the three base
    numbers (noted open in #22); the tuning baseline check now pins all three, so it cannot reach tuning.
  - Nothing above informational.
- **Housekeeping:** a `git add -A` swept the untracked local Codex config (`.codex/`, `AGENTS.md`) into
  an unpushed commit; removed before anything left this machine.

## Iteration 2
- **Review (Astra, round 2): FIX FIRST**, 1 blocking, 1 non-blocking; round-1 fixes confirmed.
  1. (blocking) The held-out run validated the arm's rule file, then later passed the FILE NAME to the
     end-to-end stage, which re-read it: a retune finishing in between could make the finder numbers and
     the end-to-end numbers describe different arms. Now the held-out run reads each rule file once,
     checks its sha256 and writes a frozen byte copy; every later stage reads only the copy, and the
     classified rows are checked against the in-memory rule. Tuning re-checks the started marker just
     before it writes anything.
  2. (non-blocking) A NaN baseline threshold escaped the pin (`abs(NaN - x) > 1e-9` is False). Now refused.
- **Test gap found while fixing:** neither script's `main()` success path had ever run in a test. Added
  stubbed whole-path runs: tuning on synthetic cross-then-shorter-jab combinations (492 grid points written,
  an unaligned video excluded and listed, V6 kept out of F1, both arms find more jabs than the pooled rule)
  and the held-out run to a written verdict, with the order baseline-reproduced -> marker -> arm asserted.
- **Tests:** fast_combinations 78/78, fastcombo_eval 91/91. **Mutation:** 5 more (the arm's end-to-end
  stage reads the original file; no late marker check; NaN accepted; freeze skips its sha256 check;
  freeze returns the original path): 5 of 5 caught.

## Iteration 3
- **Fresh clone at `a849e21`:** 14 suites, all exit 0 (fast_combinations 78, fastcombo_eval 91, hands 99,
  report 289, classify 297, e2e 74, analyse_clip 113, pipeline 314, boxers 167, completeness_credited 95,
  frame_skips 222, linking 266, render_punches 66, ring 219).
- **Review (Astra, round 3): FIX FIRST**, 2 blocking:
  1. Two concurrent held-out invocations shared one frozen-copy folder, so the second could overwrite the
     first's copy before the first claimed the marker. Now each invocation writes into its own folder
     (`tempfile.mkdtemp` under `--e2e-dir`), recorded as `run_dir` in the result.
  2. **The tests touched real experiment state**: one section restored the real marker path, then created
     and deleted the marker there, and one call wrote a frozen copy under the checkout's `results/`.
     Confirmed: a `results/fast_combinations_e2e/frozen/baseline_rule.json` (the test's own fixture) was
     in the worktree; removed. Now the marker points at a temp folder from the first line of the test
     file to the last, the tests run from a temp working directory, every call passes temp outputs, and
     a final check asserts the checkout's `results/` is unchanged and no real marker exists.
- **Tests:** fastcombo_eval 93/93. **Mutation:** shared run folder: caught. **Total 25 of 25.**
- **Fresh clone at `614b947` (final):** fast_combinations 78, fastcombo_eval 93, hands 99, report 289,
  classify 297, e2e 74, analyse_clip 113, pipeline 314; the same numbers as the working tree; no
  `results/` folder created in the clone.
- **Cap reached (3 iterations).** Round 3's two fixes are verified by tests, a mutant and the fresh clone,
  but have NOT had a fourth Astra review. Astra could never run the tests itself ("Access is denied" in its
  sandbox, all three rounds) and could not independently confirm scipy tie-equivalence at runtime; the
  300-signal randomized test (ties and plateaus included) covers it here.

## 2026-11-03: real pose on the training videos (V1, V6, V7, V8, V10 from the laptop)
Pose and alignment were checked before any tuning; nothing was scored.

| Video | fps (native) | pose length / video | detected | alignment: best offset, sharpness | labelled spans |
|---|---|---|---|---|---|
| V1 | 24.000 (24.000) | 1940.1 / 1940 s | 88.1% | 0, **1.00: FAILS** | 1,865 |
| V4 | 23.976 (23.976) | 1474.9 / 1475 s | 99.3% | 0, 2.19 | 560 |
| V6 | 23.976 (23.976) | 1948.0 / 1948 s | 99.7% | 0, 2.12 | 685 (= `punch_finder_v6.json`) |
| V7 | 29.970 (29.970) | 936.3 / 936 s | 99.8% | 0, 2.64 | 195 |
| V8 | 29.970 (29.970) | 803.5 / 804 s | 96.7% | 0, 3.07 | 199 |
| V10 | 25.000 (25.000) | 1076.7 / 1077 s | 99.8% | 0, 2.00 | 151 |

- **V1 is excluded by the gate's rule.** Its shipped skeletons are ~0.2 frame heights from the YOLO person at
  every offset in +-12 frames (aligned videos: ~0.008), across all five fifths of the video. Not a small
  shift: like O61, the skeletons are not from the video the link gives, or the download is another cut (the
  link carries `&t=709s`). No offset was searched for; using V1 with a found offset would change the gate.
- **Bug found on the real data, fixed (`5473b70`):** V6 ships one continuous skeleton, not clips, so
  `held_out_clips` refused it. `v6_ground_truth` builds a <= 25-frame alignment window per span from the
  continuous skeleton, so V6 takes the same gate. The stubbed tuning test could not catch this: it stubs
  `load_video`. Tests: fastcombo_eval 98/98 (5 new V6 checks), fast_combinations 78/78; 2 mutants on the new
  branch caught (**27 of 27 in total**). Astra (one review of this commit): **SHIP**, no findings.

## Residuals (accepted)
- Results files are gitignored (`results/`), so the tune JSON lives on disk only: copy its chosen point
  and the arm into the PR when tuning runs, before the held-out run.
- The started marker lives under the checkout that runs the evaluation; a run from a second checkout
  would not see it. The evaluation is run from the main checkout only (where the data is).
- Precision is a lower bound (RESULTS 5d); V6 counts for recall checks only.
