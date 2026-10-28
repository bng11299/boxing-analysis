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

## Residuals (accepted)
- Results files are gitignored (`results/`), so the tune JSON lives on disk only: copy its chosen point
  and the arm into the PR when tuning runs, before the held-out run.
- The started marker lives under the checkout that runs the evaluation; a run from a second checkout
  would not see it. The evaluation is run from the main checkout only (where the data is).
- Precision is a lower bound (RESULTS 5d); V6 counts for recall checks only.
