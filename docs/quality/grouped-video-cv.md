## `main..feat/grouped-video-cv` — 2026-09-22

**Why.** Closes residual risk #7 above. Git history shows the ST-GCN++ settings
were committed (`82318d2`, 20:04) before the only ST-GCN++ run (finished 21:41),
and the operator recalls no tuning on validation. But every design decision so
far was still checked against V2/V5/V9, so a single held-out split cannot serve
as the test set. `--cv K`: leave-videos-out folds, assigned from clip counts only.

**Acceptance criteria.** (1) No video on both sides of any fold, checked where the
models are fitted. (2) Every clip scored exactly once; pooled labels and
predictions aligned. (3) The fold assignment cannot read labels or scores.
(4) The single-split path is unchanged: same results JSON, same keys, same order.
(5) The checkpoint's sha256 record survives into the CV output.

**Threat sketch.** No new trust boundary. The new input is an integer CLI argument.
The threat is to the *integrity of reported numbers*: a leak across folds,
misaligned pooling, a CV file overwriting the single-split headline, or folds
picked to flatter a result.

### Iteration 1

- **Test:** 145 → 161. Four planted mutations each caught (misaligned pooled
  labels, missing-class guard removed, no-group guard removed, fold balancing
  inverted). The single-split synthetic JSON matched the pre-change output exactly.
- **Review (code-reviewer subagent, adversarial):** SHIP-WITH-FIXES. Blocking:
  (a) the default `--out` would let a CV run overwrite `results/baseline.json`;
  (b) CV output dropped `pretrained_from` (the sha256), `top_features`, `params`
  and `t_out`; (c) no test caught `run_arms(clips, ...)`, which trains on the
  whole corpus. Suggestions: three folds are too few for a sign check, so report
  per-video gaps; zero gaps were counted as disagreement; per-fold macro F1
  penalises classes a fold lacks; `--cv --include-v6` would put our V6
  segmentation into the headline; too many folds gave a traceback; the console
  did not show the clustering caveat; single-split key order was not under test.
  No leakage found in scaling, features, class weights, floors or seeding.
- **A real run was stopped** 15 min in, because its file would have lacked (b).

### Iteration 2

- **Changed:** all three blocking findings and suggestions 1–7 fixed. A CV run
  defaults to `results/baseline_cv<K>.json`. Fold records keep provenance and
  importances, and constant fields are lifted onto pooled rows. `run_arms` is
  spied on for leaks. Per-video gaps come with an exact sign test (ties counted
  as ties), on macro F1 over present classes and on accuracy. `--cv` with
  `--include-v6` is refused, and too many folds is a CLI error. The caveat is
  printed with the intervals. A test pins the single-split key order.
- **Test:** 179 passing. Mutations: the reviewer's whole-corpus leak fails 2
  tests; dropping `pretrained_from` fails the suite (KeyError, exit 1); counting
  ties as disagreement fails 1. The single-split JSON is still identical.
- **Security:** no new file, network or deserialisation path. Each fold still
  goes through `load_pretrained`, so the sha256 is re-verified per fold.

### Iteration 3 — 2026-09-28

- **Why:** iteration 2's fixes were never re-reviewed, and no gate reproduction
  was recorded.
- **Clean clone:** cloned from GitHub at `7add784`, the suite gave 179/0, which
  matches iteration 2's count.
- **Review** (fresh code-reviewer subagent, on `main..7add784`): FIX FIRST.
  **No leakage found.** It checked the scaler, features, class weights, floors,
  epochs, the pretrained head and seeds. It checked the sign test against a
  brute-force binomial, and the alignment of pooled predictions. But:
  - (B1) the test for blocking finding (a) could not fail: it checked a parser
    default, not what `main` writes;
  - (S2) the present-class macro-F1 test gave the same answer under plain macro
    F1;
  - (S3) the sign test treats videos as independent, but with K=3, videos in a
    fold share one model;
  - (S4) "single split unchanged" was pinned on key order only;
  - (S5) an explicit `--cv 3 --out results/baseline.json` still overwrote the
    headline;
  - nits: constant fields were never checked across folds, a missing-class fold
    ended in a traceback, `p=0.000` formatting, and "every clip scored once"
    was not asserted.
- **Changed:**
  - `main` is tested from an empty working directory with no `--out`.
  - The macro-F1 value is asserted directly.
  - `CV_INTERVAL_NOTE` states that the sign test is optimistic, and that one
    fold per group makes it less so but not exact.
  - Model order, per-row fields and the split bookkeeping are pinned **to main's
    own output**, generated in a worktree of main.
  - `--cv` refuses an `--out` resolving to the headline file.
  - Constant fields are checked across folds, or `run_cv` raises.
  - A fold missing a class is a CLI error.
  - The console prints both sign tests, with p formatted `.2g`.
  - `run_cv` asserts that every clip, group and prediction is scored exactly
    once.
- **Test:** 185/0.
  - **Mutations,** each caught: the CV default going back to the headline (2
    tests); plain macro F1 (1); the headline refusal removed (2); the
    constant-field check removed (1).
  - **Main vs branch:** main's full synthetic single-split JSON equals this
    branch's value for value (minus `environment` and `train_seconds`).
- **Re-review:** APPROVE. It tried to slip past the headline guard with case
  changes, backslashes and a trailing dot, and every attempt was refused. Its
  wording nits on the sign-test note were applied.
- **Security:** no new surface. The only new file behaviour is a refusal. The
  test's `chdir` is confined to a temp directory and restored in `finally`.

### Residual risk

- The pooled clip-level interval is too narrow (within-video correlation,
  between-fold variance). The per-video sign test is the check, but it is
  itself optimistic at K=3: videos sharing a fold share a model, so there are
  fewer than 9 independent units. Its floor of p = 0.0039 is a best case.
- The headline guard, like the default path, is relative to the working
  directory, so running from outside the repo root can still reach the
  headline file by an absolute path. Run from the repo root.
- Seed 2's CV run never finished: the log stops mid-fold 3 on 2026-09-22 and
  there is no JSON. The PR body's "seeds 1–2 are running" is out of date.
- The V1 fold trains without 39% of the corpus, so absolute CV scores are not
  comparable to the single split. Compare gaps, not levels.

