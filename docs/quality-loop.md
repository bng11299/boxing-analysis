# Quality loop log

One entry per iteration: gates run, findings, what changed.

## `main..feat/stgcnpp-pretrained` — 2026-09-22

**Acceptance criteria.** (1) `tests/test_pipeline.py` passes at the branch head,
with negative tests at the checkpoint boundary. (2) The sha256 is computed on the
exact bytes passed to `torch.load`, before loading. (3) Every `torch.load` uses
`weights_only=True`, with no unsafe fallback. (4) No `verify=False` anywhere.
(5) `weights/` is gitignored, and no commit on any ref contains weights.
(6) A clean clone of the head passes the same suite with the same count.

**Threat sketch.** Assets: this machine, the integrity of reported numbers, and
licence compliance for the NTU-120 weights (non-commercial; the repo is meant to
go public). Entry points: the checkpoint file, corpus files, CLI args. Trust
boundary: `load_pretrained`, and the corpus loader. Top threats: a swapped or
malicious pickle (code execution, or silently wrong weights); weights committed
into a repo that later goes public.

### Iteration 1

- **Test:** 133 → 142 passing at `18f01be`. New: scratch − RF interval present
  and correctly oriented; `--seed` parsed; random forest and ST-GCN++
  deterministic under a fixed seed; a spec with no sha256 refused at the CLI;
  `torch.load` handed the hashed in-memory bytes (a spy). The spy test **fails
  against the previous loader**, which hashed on one open and unpickled on another.
- **Security:** that check/load race was the one real finding, fixed in the same
  commit (read once, digest the bytes, `torch.load(BytesIO(bytes),
  weights_only=True)`). `weights_only=True` is on the only `torch.load` call;
  corpus `np.load` uses `allow_pickle=False`. The repo has no network fetch code,
  so no `verify=False` is possible; `grep` finds none. `weights/` is ignored
  (`.gitignore:17`); `git log --all -- weights/`, every object path on every ref,
  and every blob over 1 MB show nothing.
- **Review (code-reviewer subagent, adversarial):** SHIP, no blocking findings.
  No leakage, no eval-mode or optimiser error; the PYSKL port was checked line by
  line against the upstream source. Non-blocking: deny-by-default was enforced at
  the CLI only (#1); three negative tests could pass without exercising their
  check (#2, #3, #5); a misleading docstring (#10); missing environment metadata
  (#9); only one of five interval pairs had its direction asserted (#11).

### Iteration 2

- **Changed:** `load_pretrained` refuses a spec without a sha256 unless the caller
  passes `allow_unpinned=True` (tests of the later checks do); the not-tensors
  test now uses a full key set; the CLI-refusal test asserts the reason; the
  size-ceiling test observes that no read happens; all five interval directions
  are asserted; library versions and the thread count are written to the results
  JSON; a dead `copy.copy` is removed.
- **Test:** 145 passing. Mutation checks: deleting the deny-by-default guard fails
  2 tests; deleting the tensor-type check fails 1.
- **Numbers:** none of these changes touches training, prediction or the bootstrap.
  The loader returns the same tensors (`copy.copy` was a no-op). The seed sweep ran
  from `18f01be`, and its reproduction result stands for this head.

### Residual risk, accepted and recorded

- Review #7: validation doubles as the test set. If the ST-GCN++ epochs, `t_out`
  or learning rates were chosen by looking at validation scores, the headline is
  optimistic in a way the seed sweep cannot reveal. Asked in the vault
  (ISSUES-LOG O44).
- Review #6: the port is verified by names, shapes and a line-by-line read. There
  is no numeric fixture against PYSKL's own output.
- Review #8: fine-tuning input differs from pretraining (1 person, 50 frames, a
  0/1 score channel, against 2 persons, 100 frames and detector scores). This
  works against the pretrained arm, not for it. It belongs in the method section.

## `main..feat/grouped-video-cv` — 2026-09-22

**Why.** Closes residual risk #7 above. Git history shows the ST-GCN++ settings
were committed (`14d61b4`, 20:04) before the only ST-GCN++ run (finished 21:41),
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

### Residual risk

- The pooled clip-level interval is too narrow (within-video correlation,
  between-fold variance). The per-video sign test is the check. With 9 videos,
  its floor is p = 0.0039.
- The V1 fold trains without 39% of the corpus, so absolute CV scores are not
  comparable to the single split. Compare gaps, not levels.
