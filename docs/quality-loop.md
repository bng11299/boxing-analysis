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

### Iteration 3 — 2026-09-28

- **Why:** iteration 2's fixes were never re-reviewed, and no gate reproduction
  was recorded.
- **Clean clone:** cloned from GitHub at `bc00dbf`, the suite gave 179/0, which
  matches iteration 2's count.
- **Review** (fresh code-reviewer subagent, on `main..bc00dbf`): FIX FIRST.
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

## `feat/v6-alignment-punch-finder` (`1947b42` + fixes) — 2026-09-28

**Why.** The V6 alignment gate and the wrist-speed punch finder were committed
without running this loop. The headline (recall 0.741, F1 0.571) rests on the gate,
so the gate has to be as trustworthy as the number.

**Acceptance criteria.**
1. The suite passes, both in the working tree and in a clean clone, with the same count.
2. The V6 run passes its gate and reproduces the stored results.
3. The gate fails closed: misaligned labels or video exit non-zero and write nothing.
4. Tuning is blind to the test half.
5. The loaders refuse malformed input.

**Threat sketch.**
- *Assets:* the integrity of the reported numbers.
- *Entry points:* the CLI options, and third-party `.npy`/`.xlsx` files and video.
- *Trust boundary:* the loaders and the ffmpeg subprocess.
- *Top threats:* a crafted `.npy`; argument injection into ffmpeg. The biggest
  *realistic* risk is a gate that passes wrong data.

### Iteration 1

- **Test:** 223/0 passed. The real rerun was bit-identical to the stored JSON.
- **Security:**
  - No blocking findings.
  - `allow_pickle=False` is set, and a test refuses a pickled array.
  - ffmpeg runs from an argument list, on an absolute path, with no shell.
  - No secrets anywhere.
  - Informational, older than this branch: `.xlsx` is parsed without
    `defusedxml`. Python 3.13's expat blocks entity expansion.
- **Review** (code-reviewer subagent): FIX FIRST.
  - **B1:** the gate checked a peak's position, never its strength. Measured on
    real V6, **27 of 388 deliberately wrong skeleton offsets passed the video
    check**, and so did labels shifted 4 frames.
  - **B2:** without `--video`, the video check was skipped silently and results
    were still written.
  - **B3:** the block bootstrap could report precision above 1. Its hits were
    credited to the span's block, but detections to their own block.
  - **B4:** no test covered `main()` failing closed.
  - **Should-fix:** `cut_point` broke on a span ending before the window
    (negative slice); detections could be matched across the cut.
  - **Nits:** the "48 frames late" comment was wrong (it is 157 or 205 frames);
    one function could divide by zero; negative span starts were accepted.
  - **Done well:** tuning is blind to the test half, nothing is differenced
    across pose gaps, `match` is optimal, and the loaders fail closed.
- **Changed:**
  - **Label check:** the scan covers every integer shift in ±3 s. It needs a
    ratio of at least 1.25, a 1.25× margin over shifts a second or more away,
    and a peak prominence of at least 1.5.
  - **Video check:** it scans ±48 lags. It needs the peak at lag 0, r(0) ≥ 0.30,
    and r(0) at least 3× every lag beyond ±6. The margin skips nearby lags
    because smooth motion correlates with itself there, aligned or not; lag 0
    must still win.
  - **`--video` is required** unless `--no-video-check` is given. A skipped
    check is stamped in the output and may not write the default file.
  - The bootstrap credits each hit twice: to its detection's block for
    precision, to its span's block for recall.
  - `cut_point` skips spans outside the window and keeps a guard of `tol` frames.
  - `main` exits 2 if the labels run outside the pose's frame range.
  - The synthetic fixture's punch times became irregular (seeded). A strictly
    periodic fixture lines up with itself at one period inside a ±3 s scan. No
    assertion was removed or loosened.
- **Test:** 247/0 passed.
  - **Real V6:** the true alignment passes, and **0 of 400 wrong offsets pass**
    (±1–200 frames).
  - **Mutations**, each caught: video check back to position-only (3 tests);
    loose label check (1); shared bootstrap credit (1, precision CI reached
    1.079); video check optional again (3). The first try at the guard mutation
    survived, because the fixture's middle cut happened to clear the guard.
    Replaced with a fixture where the unguarded cut is 1 frame from a span;
    it is now caught (2 tests).

### Iteration 2

- **Re-review** (same agent): APPROVE. It tried to break each of B1–B4 and all
  held. Should-fix items:
  - labels +3 frames late (tol + 1) passed, because the scan's peak sits a frame
    off-centre on aligned data;
  - `--out` given as an absolute path got around the default-path rule;
  - the prominence check never ran in any test (median 0 gives `inf`), and
    `inf` would have been written as non-standard JSON;
  - the labels-past-the-pose check looked at the last span by start, not the
    latest end.
- **Changed:**
  - The label limit is now tol − 1.
  - Resolved paths are compared.
  - Output goes through `json_safe` and is written with `allow_nan=False`.
  - The labels check uses the maximum end.
  - New tests: ±3-frame labels; a swaying body with weak punches, which fails
    on prominence alone while the same sway with clear punches passes; the
    absolute-path bypass; an early span ending past the pose; `json_safe`.
- **Test:** 254/0 passed.
  - **Mutations:** tol limit caught; path compare caught; prominence removed
    caught; last-span-by-start caught (by a crash in the scan; see residual).
  - **Real V6 rerun:** gate passed; numbers unchanged apart from the bootstrap
    fix (F1 CI lower bound 0.473 → 0.472); strict-JSON parse OK.
- **Security:** no new file, network, subprocess or deserialisation path. The
  new CLI flags only narrow what may be written.

### Residual risk

- The thresholds were set knowing aligned V6's values. They are pass/fail
  margins, not tuned parameters, and each sits several times away from
  misaligned data. Revisit them for 60 fps sparring.
- A span covering the whole labelled range makes `label_shift_scan` raise. The
  run still fails closed (non-zero exit, nothing written), but with a traceback
  rather than a gate message.
- The chance floor's shifted detections can land in pose gaps, and the uniform
  floor cannot. The effect is small on V6 (51 of 44k frames).
- Everything here is one athlete, hooks and uppercuts only. See the output's caveats.

## `feat/yolo-ingest-overlay` — 2026-10-02

**Acceptance criteria.** (1) A video becomes a `PoseSequence` (one row per frame, gaps
kept) and a skeleton overlay, with COCO-17 order so it converts to a `data.Track`.
(2) The video is untrusted: size capped before allocation, rotation honoured, no
shell, garbage and empty files refused. (3) The YOLO checkpoint is a pickle:
refused without a pinned sha256. (4) No dependency added to the base install.

**Threat sketch.** Assets: this machine and participant footage. Entry points: the
video file, the checkpoint, the pose `.npz`. Top threats: a hostile container or
checkpoint reaching a parser/unpickler; participant video leaking via an output
directory that gets committed (`results/` and `*.mp4` are gitignored).

### Iteration 1

- **Test:** `tests/test_ingest.py`, 55 passing; `test_pipeline.py` unchanged at 260.
- **Review (self, adversarial):** found by the tests -- `select_person` accepted a box
  with inverted corners (positive area); fixed by testing the sides. Two of my own
  tests could not fail (collinear joints made a bone check vacuous; a continuity case
  the design rightly loses) -- rewritten so the precondition is asserted.
- **Real-fact check:** BoxingVI V6/V1 probe to 23.976/24.0 fps at 1920x1080, matching
  `NATIVE_FPS`; 300 frames decoded each.
- **NOT verified:** the YOLO11-pose path itself (`load_yolo`, `yolo_predictor`) has not
  run -- no ultralytics or CUDA on this machine. It is tested only through an injected
  predictor. Needs a first run on the GPU machine with real footage.

### Iteration 2 -- first real clip (2026-10-03, RTX 3060 Laptop, YOLO11m-pose)

- **Real-world fact:** a 44 s, 1080x1920, 59.946 fps phone clip: person flagged in 95.5%
  of frames, but only 84.5% had >= 8 joints at conf >= 0.3. `detected` overstated usable
  data -- exactly the shape-vs-substance failure the gates exist for. Added
  `PoseSequence.usable()`; the overlay border and the run summary now follow it.
- **Test:** 61 ingest tests, 260 pipeline, all passing.
- **Still unverified:** left/right colour convention against a known-handed stance;
  bystander handling (no second person in the clip); fast-punch blur.
- **Weights:** `yolo11m-pose.pt` sha256 `29b17eaf...6eb2` is trust-on-first-use; GitHub
  publishes no asset digest, so it is NOT yet checked against an independent source.

### Iteration 3 -- review fixes in `iter_frames` (2026-10-03)

- **Bug 1, stream mismatch:** `probe` measured `v:0` but ffmpeg decoded its own pick.
  A file with a larger second stream marked default gave 250 chopped "frames" from a
  10-frame file, silently. Fixed with `-map 0:v:0`.
- **Bug 2, exit code never read:** ffmpeg is still alive at stdout EOF (20/20 runs),
  so the old `poll() is None` test always killed it and the failure check was dead
  code. Now it kills only when we stopped early, waits at EOF, and raises on a
  non-zero exit or a partial trailing frame.
- **Test:** 69 ingest (8 new), 260 pipeline, all passing in `.venv`. Against the old
  `video.py` the 4 tests aimed at the two bugs FAIL, so they can fail.
- **Real-world fact:** a 4 s stream-copied V6 excerpt decodes to EOF as 96 frames,
  equal to ffprobe's `-count_frames`; V6 with `max_frames=300` gives 300, no error.
- **Still open (from the review):** the `usable` gate does not require wrists, and
  `to_track` drops confidence; VFR detection misses a 60-nominal / 59.946-average
  clip; `render_overlay` does not refuse a pose longer than the video.

### Iteration 4 -- the three open review items (2026-10-03)

- **Acceptance:** (1) a detected frame with >= 8 solid joints but an unsure wrist or
  shoulder is not usable, and `to_track()` drops it by default and carries confidence;
  (2) a 60 fps clip with one dropped frame (average 59.9, inside the old 0.5 fps
  tolerance) is VFR, ms-rounding jitter is not, unmeasurable timing counts as VFR;
  (3) `render_overlay` refuses a pose longer than the video, or at another fps, and
  draws found joints under the red border.
- **Threat sketch:** assets: own footage, pose files. Entry points: video container,
  ffprobe stdout, `.npz`. Boundaries: untrusted file -> ffprobe/ffmpeg subprocess; pose
  file -> loader. Top threats: a container with an extreme packet count exhausting the
  timing scan; a pose file smuggling NaN/out-of-range confidence that becomes punches.
- **Build:** `usable()` requires `REQUIRED_JOINTS` (both shoulders, both wrists -- what
  `wrist_signal` differences); `to_track(usable_only=True)` by default; `Track.conf`
  (optional, validated; V6 has none); `PoseSequence` validates conf in [0, 1] on load.
  VFR is measured from packet pts (`video.frame_timing` -> `Timing`), not metadata.
  Overlay: longer-pose and fps checks inside the writer (a refusal deletes the file).
- **Test:** 97 ingest, 260 pipeline. Reverting each of 8 fixes makes >= 1 test fail.
- **Review (code-reviewer): FIX FIRST.** Blocking: a 30 -> 24 fps switch (a phone in low
  light) was not flagged -- every interval is only 0.25x off the median. Non-blocking:
  edit-list discard packets inflated `n_frames` (600 counted, 510 decoded); `to_track`
  ignored the run's `--min-conf`; two `max_frames` cases untested; `probe` ran 3x per run.
- **Security:** S1 (low) the packet scan was unbounded -> cap. S2 argument lists and an
  absolute path, no shell, no option injection. S3 conf validated at both loaders. S4
  every timing failure fails closed (VFR) and prints UNMEASURABLE.

### Iteration 5 -- drift, discard packets, rate agreement (2026-10-03)

- **Fix:** cumulative drift against the clip's MEAN rate (line through first and last
  pts), `MAX_DRIFT` 1 interval. Not `i * median`: V6's median is 42 ms against a true
  41.7 ms, which drifts ~325 frames. Rejected the review's "1/median vs average > 1%":
  ms rounding at 29.97 fps is already 1.1% off. Packets flagged `D` dropped. Packet cap
  `MAX_TIMING_PACKETS`. `to_track(min_conf=)`. `info` passed through, one probe per run.
  Then, from the second review: measured mean rate must match `avg_frame_rate` to 0.1%
  (every consumer times frames with the latter).
- **Test:** 109 ingest (69 -> 97 in iteration 4, -> 109 here), 260 pipeline.
  New controls cut both ways: the 30 -> 24 clip has 0 irregular intervals; the cut clip
  has more packets than it decodes; constant skew moves the median but not the verdict.
  A fixture bug was caught by its own control: without a 1/600 time base in the filter
  AND the encoder, 1/24 s rounded to 1/15. Reverting each of 13 fixes fails >= 1 test.
- **Real-world fact:** on all 10 BoxingVI sources, measured frame count == container
  `nb_frames` (V6 46,705, as O54 recorded on 09-27), drift <= 0.047 intervals, mean rate
  within 5e-7 of `avg_frame_rate`, none flagged. The reviewer's repro clips: 30->24,
  24->30, 30->60 flagged (drift 14.8 / 6.6 / 27.9); the edit-list cut counts 510 = decoded.
- **Review (second pass): SHIP**, no blocking findings.
- **Residual risk, accepted:** (a) the packet cap bounds parsing, not ffprobe's buffered
  output -- memory until the 300 s timeout on a crafted file with tens of millions of
  packets; own footage only. (b) A slow-motion export retimed onto a regular timeline is
  undetectable from timestamps. (c) `MAX_DRIFT` is one interval regardless of clip length:
  a long phone clip with a slowly wandering rate may be flagged although its speed error
  is < 1%. The flag is a warning, not a refusal. **The original phone clip (nominal 60,
  average 59.946) is on the GPU machine and has NOT been re-probed** -- run it, or the
  reshoot, and record `max_drift`. (d) Requiring both wrists drops frames where only the
  far wrist is hidden (three-quarter view); per-hand masking in the punch finder, reading
  `Track.conf`, is the recovery, deferred to the learned-finder build.

## `chore/two-machine-setup` -- 2026-10-04

Tooling to bring the GPU laptop to the same state as SLOTMACHINE: `scripts/setup-env.ps1`
(+ `scripts/check_env.py`), read-only `scripts/inventory-machine.ps1`, one-shot
`scripts/bootstrap.ps1`, and `docs/MACHINE-SYNC.md` (git / rebuilt venv / Syncthing).

- **Acceptance:** (1) `setup-env` builds a venv whose torch is the build the machine
  needs, and nothing later can swap it; (2) `check_env` fails on a wrong build, a missing
  or unloadable package, no ffmpeg, or a GPU that cannot compute; (3) the inventory writes
  nothing, reads no cloud-only file, leaks no credential or profile path; (4) `-Recreate`
  can delete only `<repo>\.venv`, never through a junction.
- **Threat sketch:** assets: the laptop's existing state, the user's tokens in git remotes.
  Entry points: package indexes, git config, file attributes. Top threats: an install
  silently replacing the CUDA build; the inventory mutating repos or leaking a token into
  pasted output.

### Iteration 1
- **Test:** 27 py / 26 ps. Real-world: `check_env --torch cpu` OK here; `cu130` and `--pose`
  fail as they should; `setup-env -NoPose` exit 0 with nothing downloaded. Inventory: 7 repos
  + vault index/HEAD unchanged, 0 token-like strings, 0 profile paths, and its sha256 of the
  ST-GCN++ checkpoint equals the pin `c6e07421...`.
- **Review: FIX FIRST.** Blocking: ultralytics needs torchvision; left to PyPI, pip takes
  0.29.1, which requires torch >= 2.14, and replaces the CUDA torch with PyPI's CPU-only
  2.14.1 -- the default laptop run could never succeed. Non-blocking: OneDrive cloud-only
  weights would be downloaded to hash; a repo with `core.fsmonitor` starts a daemon on
  `git status`; query-string tokens unredacted; nvidia-smi stderr aborts under 'Stop' in
  PS 5.1; three delete-guard tests vacuous (no pyvenv.cfg, so they threw for another reason).
- **Security:** S3 (medium) junction under `.venv` -> refuse reparse points. S4 cloud-only
  hashing. S1 torch via `--index-url` (no dependency confusion). Residual: pins without hashes.

### Iteration 2
- **Fix:** `torchvision==0.28.0` pinned (PyPI metadata: requires `torch==2.13.0`; `+cu130`
  and `+cpu` wheels on the PyTorch index). torch + torchvision installed as one command,
  one build; everything else under a constraints file pinning both. `check_env`: torchvision
  build-checked; real `torchvision.ops.nms` on CUDA. The non-blocking items above.
- **Real-world fact:** `pip --dry-run --report` in this venv: without constraints, the pose
  install brings torch 2.14.1 + torchvision 0.29.1; with them, torch is kept, torchvision
  0.28.0, ultralytics 8.4.171, exit 0. Confirmed git creates `.git\fsmonitor--daemon` on
  `status` without the override and nothing with it, then tested exactly that.
- **Test:** 35 py / 36 ps; each of 9 fixes reverted fails >= 1 test.
- **Review (second pass): SHIP.** Fixed after it: `startswith("import torch")` also matched
  torchvision and skipped the CUDA check; the cloud-only test now goes through the hashing
  function the inventory calls. 36 py / 38 ps / 109 ingest / 260 pipeline.
- **Residual, accepted:** a fresh venv resolves numpy 2.5.2 with torch, then downgrades to
  the 2.2.6 pin (churn, not a failure); unpinned transitive packages may differ between
  machines; `-NoPose` after a full run stops checking pose packages (documented).
  Not run here: the CUDA path end to end -- that is the laptop's first bootstrap.

### 2026-10-06 -- first run on the laptop (MSI, RTX 3060 Laptop, driver 617.14)
- **The CUDA path, end to end:** `bootstrap.ps1` on a fresh pull ended `ENVIRONMENT OK`, so
  `check_env --torch cu130 --pose` passed: torch 2.13.0+cu130, ultralytics 8.4.171, numpy
  2.2.6, a CUDA tensor op and `torchvision.ops.nms` on the GPU. Closes the residual
  "not run here: the CUDA path".
- **YOLO11m-pose weights no longer trust-on-first-use:** the laptop's sha256 `29b17eaf...6eb2`
  equals GitHub's server-computed digest for the asset in Ultralytics release `v8.4.0`
  (42,459,307 bytes). Pinned in `docs/MACHINE-SYNC.md` section 4.
- **Fixed:** the bootstrap summary printed the full profile path (the username); it is now
  shortened to `~` like the inventory. Checked: 0 occurrences in the summary.

## `feat/own-footage-labels` -- 2026-10-06

Label Studio config (`labelstudio/punch-labels.xml`) and importer (`src/boxingvi/lsimport.py`,
`import_labels.py`) for the ground truth every punch-finding number on own footage is scored
against. Label the OVERLAY mp4: H.264 CFR (browsers rarely play the iPhone's HEVC) with the
true frame number burned in.

- **Acceptance:** (1) the config passes Label Studio 1.23.2's own validator; (2) a correct export
  imports with every punch on the right video frame; (3) a wrong frame rate, misplaced frame
  checks, a duplicate, an unknown label, an out-of-range frame, a second annotation or hostile
  JSON is refused with the reason named; (4) existing labels are never silently replaced.
- **Threat sketch:** asset: the ground-truth labels. Entry point: the export JSON (untrusted).
  Top threats: a silent frame shift (LS defaults to 24 fps; numbering base undocumented) that
  corrupts every score with nothing visible on screen; a duplicated punch quietly lowering recall.

### Iteration 1
- **Real-world facts:** config VALID under LS 1.23.2's `validate_label_config` (from its own venv),
  parsed as TimelineLabels with 8 labels; controls: a broken `toName` is rejected, a misspelt tag
  is NOT (so the parsed type, not the validator, is the evidence the tag is right).
- **Test:** 34; each of 10 guards reverted fails >= 1 test.
- **Review: SHIP**, with findings from the installed LS source: frames are 1-based
  (`TimelineRegionEditor` `min={1}`, `VideoCanvas` `ceil(t*fps)`), so the offset is exactly +1 and
  "within +-1" would let checks marked one frame early shift every punch; regions are not checked
  for `from_name`/`to_name`; drafts give a misleading "found 0"; `lead_time` is cumulative
  wall-clock including idle; an `unclear` duplicate escapes the per-hand overlap rule; `Infinity`,
  `NaN` and deep nesting fail without a clean error; VFR video and overwrites not refused.
- **Security:** export size capped before parsing; nothing in the export names a file to open;
  output path from the local video name only; `allow_nan=False` on write.

### Iteration 2
- **Fix:** all of the above. Offset must be exactly +1; tag names checked; drafts named; lead time
  reported as an upper bound; unclear-on-unclear overlap and unclear IoU >= 0.5 with any punch
  refused; `parse_constant` rejects NaN/Infinity; RecursionError becomes a clean refusal; VFR
  video refused; `--force` required to replace labels; the export's sha256 recorded with them.
- **Test:** 50; each of 8 new guards reverted fails >= 1 test. All suites: labels 50, ingest 109,
  env 36, pipeline 260, scripts 38.
- **Not yet covered:** every fixture is hand-built. The first real export is the real-world test,
  and the importer enforces it: it refuses unless both frame checks give exactly +1.

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

## `feat/olympic-loader` -- 2026-10-06

Strict loader for the Olympic Boxing release (`src/boxingvi/olympic.py`) and an audit
(`olympic_audit.py`, writes `results/olympic_audit.json`).

- **Acceptance:** every one of the 29 annotation files parses under strict rules; the release
  reproduces the paper's countable facts or the mismatch is reported; CVAT track semantics
  (visible until the next `outside` keyframe) are honoured.
- **Threat sketch:** asset: training labels whose "no punch" frames must really be no punch.
  Entry point: third-party CVAT JSON. Top threats: a mis-read track end shortening every
  punch; an under-labelled video teaching "no punch" on real punches.

### Iteration 1
- **Real-world facts:** 29 videos, **312,774 frames: exactly the paper's total**; 4,605 events
  + 4 empty tracks = 4,609 tracks. The paper: a licensed referee "looked at 312,774 frames" and
  judged each punch or not, misses and blocks included.
- **Test:** 23 incl. real-data; each of 6 guards reverted fails a test.
- **Review: SHIP**, findings: an event must end at the frame before its `outside` keyframe
  (965 of 4,605 ran 2-12 frames past the last visible box); one track reappears after a 1-frame
  hide (kam4_gh069681) and a longer reappearance would have merged silently; box bounds too
  loose (none of 17,649 real boxes crosses the edge); the audit called the release "matching
  the paper" while the paper's 11,345 punch frames matches no derivable count; minutes
  hard-coded at 50 fps; kam2/kam4 may film the same punches, so splits must hold out bouts.

### Iteration 2
- **Fix:** all of the above. Reappearance merged only after <= 2 hidden frames, else refused.
- **Real-world facts:** punch frames to the outside keyframe **21,564 (paper: 11,345) -- reported
  as a mismatch, not fitted**; 1 reappearing track, 0 open-ended; strict bounds accept every box.
  Flags (each needs a viewed spot-check before any exclusion): kam2_gh208416 and kam2_gh218416
  at 21 and 18 punches/min against a median of 42.7; kam4_gh099681 95% single-keyframe labels.
  Cameras cover 47.1 vs 57.2 min, so their density gap (38.5 vs 48.9/min) is not like-for-like.
- **Test:** 33; all suites green.
- **Open:** completeness can only be tested with pose -- fast wrist bursts far from any label --
  which needs both boxers tracked (YOLO + ByteTrack), i.e. Phase 2's tracking brought forward.

## `feat/two-boxer-tracking` -- 2026-10-06

Every tracked person per frame (`pose_yolo.PoseTable`, `extract_tracks`, `yolo_tracker` with
ultralytics' ByteTrack, `config/bytetrack_50fps.yaml`), `run_ingest.py --track`, a tracks
overlay with ids, and crediting each labelled Olympic punch to a person (`olympic.attribute`,
`olympic_attribute.py`).

- **Acceptance:** (1) nobody is chosen at extraction (the ring has a referee); (2) a punch is
  credited without using the label's hand, so the left/right convention is TESTED; (3) blocks
  and clinches come out ambiguous, not settled arbitrarily; (4) ids on the overlay are legible.
- **Threat sketch:** asset: per-boxer punch sets that later train and score the finder. Entry
  points: tracker output, CVAT boxes. Top threats: a statistic that cannot fail certifying a
  convention; an attribution rule that quietly drops one kind of punch from the training set.

### Iteration 1
- **Test:** 28 (fake tracker: two boxers + referee, ids -1 and duplicates); 8 guards bite.
- **Review: FIX FIRST.** Blocking: the thrower was chosen BY the labelled hand's wrist, so under
  a mirrored convention events dropped out and "labelled hand nearer" stayed high whatever was
  true -- a gate that could not fail. Also: first-box-only distances; 10% boxer rule shape-only;
  ByteTrack buffer 0.6 s at 50 fps and conf 0.25 starving its second pass; overlay ids clipped
  at the edge, border only when nobody tracked (never, with a referee present).

### Iteration 2
- **Fix:** nearest wrist of ANY person, either hand, against the per-frame interpolated glove box;
  credit only with a margin (else ambiguous); convention = share of credited punches whose
  nearest wrist is the labelled hand. Tracker yaml (buffer 100 frames), `--track-conf 0.1`.
  Overlay: border under 2 people; ids clamped whole and below the frame counter.
- **Test:** a mirrored pose model gives 0% agreement vs 100% unmirrored; restoring the old
  choice fails it. The edge-id test was wrong twice (the id still fit; then a pixel COUNT that
  could not separate the cases) and is now a position check.
- **Review: SHIP**, with two biases: a one-frame defender touch vetoed credit (minimum over
  frames), which would under-represent blocks and head shots in any credited subset; and a
  left-sorts-first tie in the thrower's own wrists inflated agreement for left-labelled punches.

### Iteration 3
- **Fix:** rank and margin on the MEDIAN distance over the event's frames (minimum kept only for
  "reached the glove"); punches whose thrower's two wrists are equally near give no hand evidence
  and are counted apart; ambiguous share reported by target.
- **Test:** 34; each of 5 attribution rules reverted fails a test. All suites green.
- **Not yet run on real footage:** the laptop smoke run (one Olympic video, `--track
  --max-frames 3000`) is the first real-world fact: ids, boxer co-presence, credited share,
  the convention share, and ambiguity by target.

### 2026-10-07 -- first real-world run (29 Olympic videos, RTX 3060 laptop, YOLO11m-pose + ByteTrack)
- **Coverage:** 28 of 29 tables complete; GH088416 holds only its first 3,000 of 12,524 frames
  (the smoke run's file: the full run's resume check tested that a file existed, not that it was
  complete). 303,250 of 312,774 frames tracked.
- **Crowd:** a median of 8 people per frame and 541 track ids per video (116-916): spectators,
  judges and coaches are in view, and boxers fragment across many ids. The two most-credited ids
  take 7-33% of a video's credited punches and are almost never on screen together, so boxer
  identity CANNOT be read from ids yet: tracklet linking is Phase 2 work, now measured.
- **Attribution (4,490 punches in tracked frames):** credited 2,363 (53%), too close to call ~46%,
  no confident wrist near the glove < 1% -- the pose stage reaches almost every labelled punch.
- **Left/right convention, tested:** the nearest wrist is on the labelled hand in **78% of 1,756**
  credited punches with clear hand evidence (607 ties left out). A mirrored convention would sit
  near 0%; chance is 50%. COCO's left is the boxer's left. The 22% disagreement is the size of
  wrist confusion at contact, worth knowing before per-hand training.
- **Ambiguity by target:** head 41%, miss 47%, body 61%, block 64%. The credited subset therefore
  under-represents body shots and blocks; any training set drawn from it must be reweighted or
  say so. kam4_gh099681 (95% single-keyframe labels) is 82% ambiguous: its labels give the
  median rule a single frame, at contact.

## `feat/boxer-linking` -- 2026-10-07

Link each Olympic Boxing video's ByteTrack ids into two boxer identities (`src/boxingvi/linking.py`,
`olympic_link.py`, a linked overlay), then the completeness test per boxer (`completeness.py`,
`olympic_completeness.py`). ADR-0012: link first, test completeness second.

- **Acceptance:** (1) from planted scenes (two boxers each in 4 fragments with gaps, a moving referee,
  static spectators outside the ring) exactly the two boxers' fragment sets come back; (2) a chain's
  fragments are pairwise time-disjoint and the two chains share no fragment; (3) every parameter is
  fixed, with its reason beside it, before any real-data number is looked at; (4) the real-data gate
  asserts facts that can fail: how many credited punches sit inside the two identities (also with half
  the punches held out of the linking) and whether the identities are on screen together; (5) the
  completeness test judges nothing: it counts bursts away from labels and hands a seeded sample to a human.
- **Threat sketch:** asset: per-boxer minutes and burst counts that decide whether the referee's labels
  are complete. Entry points: tracks `.npz` (loaded without pickle), CVAT JSON (already strict), a GoPro id
  from the command line that becomes part of an output file name. Top threats: (a) an identity swap or a
  referee absorbed into a boxer's track silently moves the unexplained-burst rate (a referee waving is a
  burst); (b) a circular gate: punches drive the linking, so "most punches are inside the identities"
  certifies itself. Mitigations: punch evidence is required for any identity; held-out share and
  co-presence reported beside the in-sample share; the overlay exists to see swaps by eye.

### Iteration 1 -- build, test, real data (as specified)
- **Parameters** (value and reason beside each constant in `linking.py`): max gap 2 s (ByteTrack's own
  buffer); speed limit 5 torso lengths/s + 1 of slack; fragments under 0.2 s ignored; ring = 1st-99th
  percentile of credited glove-box centres widened by 2 torsos; a punch is worth 10 points, a second in
  the ring 0.5, a link 1 (+ up to 3 jump, 1 gap, 2 scale).
- **Found by the first synthetic tests, fixed before any real run:** with a free second chain, the
  cheapest use of it was to cut the ONLY boxer in two to save a link (`CHAIN_POINTS` = 8: more than the
  dearest plausible link, less than one punch); a chain of ring-time alone (a lone referee) could win the
  second slot (`MIN_CHAIN_PUNCHES` = 1); `--overlay ""` was falsy and silently skipped the id check.
- **Real-world facts, 29 videos, as specified (`results/olympic_linking.json`):** both boxers identified in
  50.5% of frames (frames-weighted; median video 57%), boxer A 76.6%, B 65.4%, either 91.5%.
  **65.4% of the 2,423 credited punches are on A or B (901 / 684); 50.7% with half the punches held out of the
  linking.** Co-presence (both identified, over the shorter identity's time) median 78%; one video
  (GH208416) 0%: its identity B is two fragments and A starts at frame 4557. Fragments per identity: median
  27 / 19 of a median 541 ids. Longest stretch with a boxer unidentified: median 4.2 s (A) and 13.9 s (B),
  worst 91 s and 156 s. Diagnosis (`credited punches by fate`): of 2,423, 1,585 inside, 803 in
  eligible fragments that overlap a chain in time (the chain was cut elsewhere), 12 unreachable, 23 outside
  the ring or too short. A chain is one path, so a stretch longer than 2 s with the boxer untracked ends it.

### Iteration 2 -- one change made AFTER seeing the real-data numbers (declared)
- **Change:** re-acquisition links (`reacquire_s`, off by default; `--reacquire` = 60 s): across gaps
  longer than 2 s, no speed limit (after 2 s a boxer can be anywhere in the ring), a flat price (4 points
  + 0.05/s). Reason: the diagnosis above, not a number to hit. It was NOT tuned: one value (a round's rest)
  was chosen and run once; no other parameter changed after seeing results.
- **Result (`results/olympic_linking.reacquire.json`):** credited punches on A or B 81.7% (59.1% held
  out), but both-identified falls to 44.5% (median 43%) and co-presence to 69%: the chain hops to
  punch-bearing fragments and skips punch-less time. It buys punches, not minutes, and the A/B labels are then only
  reliable between hops (nothing recognises a person). Hence off by default; both are reported.
- **Held out is the honest figure.** The in-sample share is high partly because punches choose the chains.
  Half held out, 51% (59% with re-acquisition) of the other half's credited punches are on the two identities
  -- a lower bound, because a fragment whose only punch is held out has no evidence left.

### Iteration 3 -- completeness (`results/olympic_completeness*.json`, sample `*.csv`)
- Rule: `results/punch_finder_v6.json` "chosen" (relative speed > 3.62 body lengths/s, 0.4 s apart) on each
  boxer's own rows, dropouts <= 0.05 s bridged, tolerance 0.1 s to ANY labelled punch of the video.
- **As specified:** 8,135 bursts on the two identified boxers (148 identified boxer-minutes: A 79.8, B 68.2);
  **5,506 are more than 0.1 s from any of the 4,605 labelled punches = 37.2 per identified boxer-minute**
  (A 38.1, B 36.1), against 31.1 labelled punches per boxer-minute; 2,705 of the 4,605 labelled punches have a
  burst within 0.1 s. With re-acquisition: 8,694 / 5,647 / 39.8 per boxer-minute, 3,035 labelled punches seen.
- **Superseded by Iteration 4** (the matching, the rates and the "upper bound" wording below were wrong; the numbers
  there are the pre-fix ones).
- **Not a finding about the labels.** On V6's tuning half the same rule had precision 0.46, i.e. over half its
  detections were not labelled punches; footwork, guard changes and slips move a wrist fast. "Unexplained" is a candidate for a
  missed label (NOT an upper bound: see Iteration 4) until a human classifies the sample (40 bursts, seed 20261007, stratified over videos):
  `results/olympic_completeness_sample.csv`, column `user_label` empty. Nothing was viewed or judged here.

### Iteration 4 -- Astra 6 review fixes (2026-10-07; verdict was FIX FIRST: 5 blocking, 4 not)

> **Partly superseded by Iteration 5** (the second review). Three things below were wrong in one respect each and are
> marked where they occur: the denominator of the completeness rates (F4, F5: identified minutes, not the frames the
> rule ran on), the greedy matching (F2) and the "like for like" comparison in the numbers. The rest stands.

Order of work: each finding got a failing test first (or a mutation that shows the old behaviour
fails it), then the fix. Nothing was re-tracked and no existing `.npz` was rewritten.

| # | Finding | Fix | Evidence |
|---|---|---|---|
| F1 blocking | A punch-free chain (a referee) took a slot, was dropped AFTER solving, and left boxer B empty (A 60 s / 10 punches, B 10 s / 1, referee 60 s / 0) | `linking.solve_chains`: an exact integer program (`scipy.optimize.milp`, zero gap, 300 s limit, anything but "optimal" is a `RuntimeError`) over a two-state layered path model (state 0 = no punch yet, state 1 = holds one; only state 1 may end). networkx (never pinned) is gone. Costs are rounded to 1/1000 point so the solver's absolute gap cannot hide a difference. | Astra's three tracks return A + B = 129 (was A + empty); 800 random instances (7 parameter sets) match an exhaustive enumeration of every pair of vertex-disjoint chains exactly, and the punch rule changes the optimum in at least 20 of them (asserted); negative: a time-limited or failed solve raises |
| F2 blocking | A burst was "explained" by ANY label within 0.1 s, including one thrown by the other boxer, and one burst near two labels counted both | `completeness.find_bursts`: one-to-one, nearest first (**greedy; replaced by an optimal assignment in Iteration 5, R2**), a boxer's OWN labels first (a label is owned by the boxer whose identity holds its credited track id in the event's frames), then the remaining bursts of both boxers against the UNATTRIBUTED labels, each label once. `match` is own / unattributed / none; the matched label's hand is recorded (`hand_agrees`, information only, 62% of matched bursts) and not required. The count is now called "unexplained = candidates for missed labels", not an upper bound; `events_with_burst` is gone. | both boxers burst together, only A labelled: B's burst is unexplained; one burst between two labels matches one; one unattributed label near two bursts matches one; greedy nearest-first unit tests |
| F3 blocking | Joined fragments read a pose jump at a handover as wrist speed (~10 body lengths/s at 50 fps, threshold 3.62) | Rows are split into segments at an id change or a missing frame; `bridge`, `wrist_signal`, `detect` run per segment; segments under the smoothing window (5 frames) are skipped and counted (`short_segments`), segments with no confident body are `bodyless_segments`. | two stationary fragments a torso apart: the joined track reads a burst (asserted in the test as the fault), per segment there is none; a real punch inside one fragment is still found |
| F4 blocking | All labels of the video were divided by identified boxer-minutes (**the matched fix still divided by identified minutes: superseded by Iteration 5, R1**) | `own_labels_per_identified_boxer_min` (labels owned by an identified boxer / his identified minutes) and `labels_per_video_boxer_min` (all labels / 2 x video minutes); unattributed labels reported as a count and share; durations and counts aggregated unrounded, rounded only when the json is written; the CLI line compares unexplained with OWN labels per identified boxer-minute | Astra's example (60 labels, 0.1 min each): 6 / 0.2 = 30, not 300; two videos of 0.0004 min each still divide by 0.0016 |
| F5 | "Signal minutes" averaged the two hands | (`signal_min` is now `exposure_min`, an explicit mask: Iteration 5) `signal_min` = elapsed minutes with EITHER hand valid (new `punchfind.signal_frames`, the one definition shared with `signal_coverage`), plus per-hand minutes and per-hand burst / unexplained counts | left wrist invalid, right valid throughout: signal minutes = elapsed, not half |
| F6 | Stratified quota: ties took the first 40 video names whatever the seed, small pools were over-sampled | `completeness.random_sample`: seeded simple random sample without replacement over ALL unexplained bursts, inclusion probability n/N for every burst; the csv has `inclusion_prob` (`user_label` still empty), the json the population N | 2,000 seeds over 12 bursts: every inclusion rate within 3 standard errors of 1/3; 60 videos x 3 bursts, sample 40: every video appears across seeds |
| F7 | The held-out test would pass with leakage (every fragment had many labels) | `olympic_link.analyse` returns `ring_heldout`; new scene where B's last fragment is supported only by held-out (odd) labels | held-out share 14/18 = 77.8%, and it WOULD read 100% if all labels leaked in (asserted); changing the held-out events' glove boxes moves the full ring and not `ring_heldout`; mutation: pass all attrs, 4 checks fail |
| F8 blocking | Size, fps and length do not identify a video; a renamed table was accepted by `load_video` and the overlays | `pose_yolo.verify_binding`: the table's `source` must equal the video's file name; `PoseTable` / `PoseSequence` gain `video_fp` = `video.fingerprint(path)` (sha256 of size as 8 bytes big-endian + first 4 MiB + last 4 MiB), written by `run_ingest`, round-tripped by save/load (old files load as ""), verified when present. Empty = accepted on the name alone and reported `video_bound: false`. Used by `olympic_link.load_video`, `olympic_attribute.py`, `render_tracks_overlay`, `render_linked_overlay`, and (content only) `render_overlay`. | same size, rate and length, other name: refused; same name, other recording, with a fingerprint: refused; legacy table: accepted, reported unbound; the fingerprint does not see an edit in the middle of a file (asserted, documented) |
| F9 | `PoseTable` allowed id -1, and with empty identities (`membership == -1`) both boxers selected every -1 row | `boxer_rows`: `(want >= 0) & (want == track_id)`; `PoseTable` refuses negative ids on construction and on load | empty linking against a table holding id -1: no rows, no bursts, no minutes |

**Existing assertions replaced (not weakened), and fixtures changed**
- `ring time valued 10x: a chain with no credited punch is emptied` asserted `identities[1] == ()` and the
  first identity equal to boxer A. That encoded the post-filter. With the rule inside the optimisation and
  ring time worth 5 points a second, the exact optimum may use a referee fragment as the tail of a punching
  chain (documented residual risk 2), so the assertion is now "no identity is punch-free and no
  referee/spectator-only chain is reported".
- The completeness tests that called `_near_label`, `events_with_burst`, `stratified_sample` and the
  csv header were rewritten for `span_gap`, `labels`, `random_sample` and the new header; the 5-frame
  tolerance boundary (5 frames inside, 6 not) is kept as `span_gap`.
- Fixtures: the planted scene's table source is now `GH000001.mp4` (the video's real name; it was
  `.MP4`), and the tables built to be "another video" (frame rate, length) now carry the right name, so those
  tests still fail for their own reason and not for the new name check.

**Decisions that differ from or sharpen the brief**
- Segments break at every id change. As first built (the brief's wording) they also broke on ANY missing
  frame inside one id, which dropped the 0.05 s pose bridge the rule was tuned with and skipped 549 of 3,542
  segments as too short. **Changed in review (a reviewer, after a second reviewer's report):** inside ONE id a hole of up to
  `BRIDGE_S` (2 frames at 50 fps) is the same person and is bridged by `punchfind.bridge` as before; only a
  longer hole ends the segment. Now 262 of 2,945 segments are under 5 frames (default linking). Tests: a 2-frame
  hole is bridged (its frames count as signal), a 3-frame hole splits, an id change splits even with no hole;
  mutations "strict split" (1 fail) and "no id split" (6 fail) are caught.
- A segment is "too short" under 5 frames (0.1 s at 50 fps = the smoothing window); this value was set
  before the real run.
- Greedy nearest-first matching, ties by frame, boxer, label index: deterministic, not a maximum matching.
  **Superseded by Iteration 5, R2** (Astra's counterexample: bursts at 100 and 120, owned labels [99,120] and [95,95]).
- `Linking` gained `solve_s` (not part of equality); `olympic_link.py` reports `link_solve_s`,
  `video_bound`, and pooled `videos_content_bound`.
- The held-out linking is solved too; its time is not recorded, only the full-label solve.

**Mutation check (each guard reverted in a scratch copy; the named suites must fail)**
| Guard reverted | Result | First failing checks |
|---|---|---|
| F1 punch rule moved back outside the optimisation (all fragments treated as punching, dead chains dropped after) | caught, 5 fail | Astra's three tracks (114 instead of 129); brute force 746/800 exact; lone punch-free fragment |
| F2a ownership ignored (any label explains any boxer) | caught, 4 | burst at 300 explained by the unattributed label; label counts; coincident bursts |
| F2b not one-to-one | caught, 6 | label used up; label already taken; unattributed label matched once |
| F3 segments ignore id changes (joined) | caught, 4 | per-segment: no burst at a handover; real punch still found; short-segment count |
| F4 rate uses all labels | caught, 2 | 6 / 0.2 = 30 not 300 |
| F5 signal minutes averaged over hands | caught, 1 | left invalid, right valid: elapsed minutes |
| F6 sample = first n bursts | caught, 4 | different seed differs; inclusion rate within 3 SE; every video can appear |
| F7 held-out linking sees all labels | caught, 4 | held-out share below 100%; would-be 100%; ring_heldout |
| F8a source not compared with the video name | caught, 4 (+1 test_tracks) | renamed table refused (load_video, CLI, tracks overlay) |
| F8b fingerprint not verified | caught, 3 (+1 test_tracks, +1 test_ingest) | same name other content refused |
| F8c overlays skip `verify_binding` | caught, 4 | renamed / wrong-fingerprint table refused by both overlays |
| F9a `(want >= 0)` removed | caught, 3 | empty linking selects no rows; no bursts, no minutes |
| F9b negative ids accepted | caught, 2 (+2 test_tracks) | construction and load refused |

**Numbers, before -> after (29 Olympic videos, default parameters, `results/*.json`, gitignored)**
| Quantity | Before | After |
|---|---|---|
| credited punches on A or B, all / half held out | 65.4% / 50.7% | 65.4% / 50.7% (901 + 684 of 2,423) |
| both identified: median video / frame-weighted | 57.2% / 50.5% | 57.2% / 50.5% |
| co-presence (median) | 78% | 77.9% |
| with `--reacquire`: credited on A or B, all / held out | 81.7% / 59.1% | 81.7% / 59.1% |
| `--reacquire`: both identified, median / frame-weighted | 43.5% / 44.5% | 43.5% / 44.5% |
| videos whose fragments in A / B changed | | 0 of 29 (and 0 of 29 with `--reacquire`) |
| MILP solve time per video (full-label linking) | | default: max 0.09 s, total 1.3 s; `--reacquire`: max 2.6 s, total 30 s; none near 60 s |
| bursts / unexplained | 8,135 / 5,506 | 7,984 / 5,772 (own 973, unattributed 1,239, none 5,772) |
| unexplained per identified boxer-minute (**denominator superseded: Iteration 5 divides by exposure**) | 37.2 | 39.0 (148.1 identified boxer-minutes) |
| own labels per identified boxer-minute | (31.1 = ALL labels over the same minutes, wrong) | 10.7 |
| labels per video boxer-minute (all labels / 2 x 104.3 min) | | 22.1 |
| unattributed labels | | 3,020 of 4,605 (65.6%); 57.0% with `--reacquire` |
| labels matched to a burst | 2,705 "within 0.1 s" (not one-to-one) | 2,212 one-to-one (973 own, 1,239 unattributed) |
| `--reacquire`: unexplained / bursts, per identified boxer-minute, own labels | 5,647 / 8,694, 39.8 | 6,185 / 8,657, 43.6, 14.0 |

- The linking is unchanged on this data: the punch-free-chain displacement of F1 is real (the unit test shows
  it) but did not occur in any of the 29 videos. The completeness numbers move because of F2-F4: the
  unexplained count rises (a label that sat next to two bursts, or thrown by the other boxer, no longer
  excuses them) and the like-for-like labelled rate falls to 10.7 per identified boxer-minute, because two
  thirds of the labels cannot be credited to an identified boxer (the identities cover 65.4% of credited
  punches and only 53% of labels are credited at all).
- **Still not a finding about the labels.** Unexplained bursts are candidates; nothing about missed labels follows until
  the sample is classified. (This bullet used to compare 53.9 bursts per identified boxer-minute with 22.1 labels per
  video boxer-minute as "like for like" and read the 2.4 times ratio against the rule's V6 precision. Those are two
  different observation populations and the inference needs recall and transfer assumptions nobody has tested: withdrawn
  in Iteration 5, R1.) The sample (seed 20261007, 40 of 5,772, inclusion probability 0.0069 each) is for you to
  classify; nothing was viewed or judged here.
  The GH078416 linked overlay was re-rendered (`results/ingest/olympic/GH078416.linked.overlay.mp4`) and NOT opened.
- All 29 tables are unbound by content (`video_bound: false`, `videos_content_bound: 0`): they predate the
  fingerprint and were accepted on the file name, which matched for all 29. Re-running `run_ingest.py --track`
  would bind them; that was not done.

- **Tests:** `test_linking.py` 224 (was 130), `test_tracks.py` 55 (34), `test_ingest.py` 114 (109);
  unchanged and green: olympic 33, hands 38, labels 50, pipeline 260, env 36.
- **Security pass:** a stored fingerprint is validated as 64 lowercase hex on construction and load (a loaded
  file is untrusted); the name and fingerprint checks run before the video is decoded or any output is
  written; the fingerprint reads two 4 MiB windows with `open(..., "rb")` and no subprocess; negative track
  ids and malformed fingerprints fail closed; the solver cannot return a non-optimal answer silently (status,
  fractionality, chain structure and the objective are all re-checked from the extracted chains); no new
  dependency (networkx removed from the code, scipy was already pinned).
- **Residual risks:** (1) the fingerprint identifies a file, not every byte (an edit in the middle of a
  recording is invisible to it; asserted and documented); (2) the 29 existing tables are bound by name only;
  (3) nearest-first matching is greedy, not a maximum matching (**fixed in Iteration 5**); (4) a referee fragment can still be absorbed
  as a tail or connector of a punching chain; (5) the identity swap after a crossing is unchanged;
  (6) 262 short segments and 6 bodyless ones are skipped, so some real bursts are unseen.

### Iteration 5 -- Astra 6 second review (2026-10-07; verdict FIX FIRST: 1 blocking, 5 not)

Reviewed `eed8ea0..35da53d`. 35da53d (a reviewer, after the round-2 fixes) had already changed `boxer_segments` to bridge holes of
up to `BRIDGE_S` inside ONE id and to split only at an id change or a longer hole; that stays.

| # | Finding | Fix | Evidence |
|---|---|---|---|
| R1 blocking | `identified_min` counted original rows, but detection runs on BRIDGED segments (a peak can land on an interpolated frame) and skips short / bodyless segments that still counted as identified time; the report called bursts per identified boxer-minute against labels per video boxer-minute "like for like" | `completeness.find_bursts` builds a per-boxer **exposure** mask over the video frames: every frame of every processed segment (not short, not bodyless) where the bridged track has a valid signal for at least one hand (`signal_frames(track).any(axis=1)`), bridged frames INCLUDED (policy stated in the docstring). `exposure_min` = mask.sum() / fps / 60. Every burst frame is asserted to lie in its boxer's exposure (`RuntimeError` otherwise). Own labels are "in exposure" when their span touches the owner's exposure (`own_labels_in_exposure` / `_outside_exposure`); unattributed labels when it touches either boxer's. Rates per boxer and pooled (raw counts and raw frame counts summed, divided last): `unexplained_per_exposure_min`, `bursts_per_exposure_min`, `own_labels_in_exposure_per_exposure_min`. `identified_min` stays as a description and nothing divides by it; `labels_per_video_boxer_min` stays and is labelled a different population. The like-for-like sentence and the precision-based reading are removed from the CLI print and from Iteration 4. | integrated through `find_bursts`: (a) the peak at frame 102, which has no row in the table, is in the numerator and the mask (398 identified rows, 400 exposure frames); (b) a 3-row skipped segment adds 3 identified frames and 0 exposure frames; (c) an owned label wholly inside the skipped segment is `own_labels_outside_exposure`; a patched `signal_frames` that empties the mask makes the run raise; no key anywhere is named `*per_identified*` |
| R2 | Greedy nearest-first matching strands a burst (tolerance 5, bursts at 100 and 120, owned labels [99,120] and [95,95]: the broad label goes to 100, 120 is left unexplained) and favours A on ties | `completeness._match` is an optimal assignment (`scipy.optimize.linear_sum_assignment`), **optimal within each phase, not necessarily the global maximum (Iteration 6, D1)**: cost = span gap for feasible pairs (gap <= tolerance), a larger-than-any-feasible-assignment M otherwise, infeasible pairs dropped afterwards, so it maximises the number of matches and then minimises the total gap. Two phases kept (own labels per boxer, then both boxers' leftovers against unattributed labels; a label once). Exact ties: bursts sorted by (frame, key) and labels by (first, last, index), and two integer tie-break terms scaled below one frame of gap, so the burst that sorts first by (frame, boxer) takes a contested label. | the counterexample, a unit test and the same case through `find_bursts` (303 and 337, a broad and a narrow owned label); 400 random small instances equal a brute-force enumeration of (most matches, least total gap); simultaneous bursts of both boxers for one label: A wins the exact tie whatever the list order (exact ties go to the burst that sorts first by (frame, boxer)) |
| R3 | `PoseTable.load` converted (`astype`, `int`, `float`) before checking: a stored id -0.5 became id 0, a frame 1.9 became frame 1 | `pose_yolo._whole_array` / `_scalar` / `_float_array`: dtype kind, finiteness, whole values, range, 1-D shape, scalar shape, `n_frames`/`width`/`height` whole and >= 1, `fps` finite and > 0, a missing field is a `ValueError` naming it. `PoseSequence.load` has the same checks. | fractional negative id, fractional frame, NaN id, whole-but-negative frame, string ids, 2-D frame array, NaN / zero fps, fractional n_frames and width, zero height, vector fps, missing field; whole-valued FLOAT arrays still load unchanged |
| R4 | Labels enough for a ring but no eligible identity: the rate was `None` and `:.1f` raised after the files were written | `olympic_completeness.main` computes the summary and both files' contents first, then writes, then prints; absent rates print `n/a` and are `null` in the json | every person re-identified every 5 frames (under the 0.2 s fragment minimum): 33 labels, no identity, exit 0, rates null, `n/a` printed, sample file header only |
| R5 | Equal MILP optima have no canonical selection rule | **Claim narrowed, no secondary optimisation** (as instructed): `linking.py` now says the answer is repeatable for the same input and solver version and that among exactly equal optima the choice is the solver's; the A / B sort names the chains after the selection | docstring only |
| R6 | Held-out pooling rebuilt counts from rounded shares | `analyse` keeps `credited_heldout_inside` (integer); `olympic_link.pooled_heldout_share` pools integer counts and divides once | 1 of 5 and 4 of 6: rebuilt from rounded shares 0.4546, the integer ratio 5 / 11 = 0.4545 |

**Existing assertions replaced or extended (nothing weakened)**
- `signal_min` -> `exposure_min` in four assertions: the same quantity (the frames with a valid signal in processed segments) under its new
  name, now an explicit mask. The values asserted are unchanged (200 / 400 frames).
- The label-count dict equality gains `unattributed_in_exposure: 1`.
- The Iteration 4 rate tests (`own_labels_per_identified_boxer_min`, the 60-labels / 0.2 min = 300 example, the 0.0004-minute rounding
  case, and the end-to-end checks on `ca["...per_identified_boxer_min"]`) encoded the identified-minute denominator that R1 shows to
  be the wrong observation. They are replaced by exposure-based equivalents with the same intent (matched numerator and denominator;
  aggregation on unrounded numbers; a real fact from the planted run); the 6 / 0.2 = 30 and 4 / 0.0016 = 2500 arithmetic is kept.
- The test titled "greedy nearest-first" is renamed; its assertion (the burst on the label takes it, the other takes the farther one)
  is unchanged and still holds under the optimal assignment.

**Mutation check (each change reverted or broken in the working tree, `git checkout` after; the named suite must fail)**
| Mutation | Suite | Result | First failing checks |
|---|---|---|---|
| R1 rates divide by `identified_min` | test_linking | caught, 4 fail | 6 / 0.2 = 30; 2 / 0.2 = 10; per boxer 1 / 0.1; own-labels rate 1 over 200 frames not 2 over 203 |
| R1 burst-in-exposure assertion removed | test_linking | caught, 1 | "a burst outside the exposure ... is a RuntimeError" |
| R1 bridged frames excluded from exposure | test_linking | caught (the run raises the new RuntimeError at frame 102, the suite aborts) + 1 fail | the 2-frame-hole segment test |
| R1 skipped segment counted in exposure | test_linking | caught, 4 | (b), (c), the video-row rate, the short-segment test |
| R2 greedy matching restored | test_linking | caught, 4 | counterexample; labels used up; 400-instance brute force; in the scene |
| R3 `PoseTable.load` coerces before validating | test_tracks | caught, 13 | -0.5 id, 1.9 frame, NaN id, negative frame, string ids, 2-D frame, fractional / zero scalars |
| R3 `PoseSequence.load` coerces | test_ingest | caught, 3 | NaN fps, fractional width, zero height |
| R4 `n/a` formatting removed (`None:.1f`) | test_linking | caught (the run crashes on the new test) | zero-identity video |
| R6 pooled from rounded shares | test_linking | caught, 2 | 5 / 11 = 0.4545 |

**Numbers, before (35da53d) -> after (29 Olympic videos, default parameters; `results/*.json`, gitignored)**
| Quantity | Before | After |
|---|---|---|
| bursts / unexplained | 7,984 / 5,772 | 7,984 / 5,766 (the optimal assignment explains 6 more) |
| exposure boxer-minutes vs identified boxer-minutes | (identified 148.06; signal 146.40) | exposure 146.40, identified 148.06 |
| unexplained per exposure-minute | 39.0 per identified | 39.4 |
| bursts per exposure-minute | 53.9 per identified | 54.5 |
| own labels in exposure, per exposure-minute | 10.7 (1,585 own labels / identified) | 10.8 (1,580 in exposure) |
| own labels outside exposure | | 5 of 1,585 |
| unattributed labels in exposure | | 2,749 of 3,020 |
| labels matched (own / unattributed) | 973 / 1,239 | 973 / 1,245 |
| sample N (population), inclusion probability | 5,772 | 5,766, 0.00694 |
| `--reacquire`: bursts / unexplained | 8,657 / 6,185 | 8,657 / 6,178 |
| `--reacquire`: exposure vs identified boxer-minutes | | 140.78 vs 141.79 |
| `--reacquire`: unexplained / bursts / own labels in exposure per exposure-minute | 43.6 / 61.1 / 14.0 per identified | 43.9 / 61.5 / 14.0 |
| `--reacquire`: own labels outside exposure; unattributed in exposure | | 7 of 1,980; 2,440 of 2,625 |
| `--reacquire`: labels matched (own / unattributed) | 1,212 / 1,260 | 1,212 / 1,267; sample N 6,178 |
| held-out credited share, pooled | 50.7% | 601 of 1,185 = 50.72% (credited on A or B, all: 1,585 of 2,423 = 65.4%) |
| MILP solve time | max 0.09 s, total 1.3 s | max 0.09 s, total 1.22 s |

- **The correction is one of definition, not of size.** On this data the exposure is 98.9% of the identified time (146.40 of 148.06
  minutes; 99.2% with `--reacquire`), no burst fell outside it in any of the 29 videos (the new assertion held in both runs), and
  the rates move by about 1%. What changed is that the numerator and the denominator are now guaranteed to describe the same frames,
  and that the unsupported comparison is gone. Nothing new is learned about the labels: unexplained bursts are candidates for missed
  labels, and the sample (40 of 5,766, for YOU to classify; nothing was viewed or judged) is what will say how many are punches.
- 2,749 of the 3,020 unattributed labels lie in time when a linked boxer was in the exposure: they have no owner among the selected boxer
  identities (ambiguous, unreached, or credited to a track outside A and B, a referee for instance), not that nobody was being observed.
  That is a statement about attribution, not about completeness.
- The linking is unchanged on this data: 0 of 29 videos changed which fragments are in A / B (R5 and R6 do not touch the optimisation);
  all 29 tables are still unbound by content (`videos_content_bound: 0`). The GH078416 linked overlay was not re-rendered or opened this round.

- **Tests:** `test_linking.py` 256 (was 224), `test_tracks.py` 71 (55), `test_ingest.py` 117 (114); unchanged and green: olympic 33,
  hands 38, labels 50, pipeline 260, env 36.
- **Security pass:** a stored pose file is validated before conversion (fractions, NaN, negative, wrong dtype, wrong shape, missing field
  all fail closed); the burst-in-exposure assertion turns a silent numerator / denominator mismatch into an error; the CLI computes
  everything before it writes, so a late formatting failure cannot leave half a result; no new dependency (`scipy.optimize` was already
  used by the MILP).
- **Residual risks:** (1) exposure is a property of the rule's signal, not of the footage: a frame can be in the exposure while the
  boxer is hidden behind the referee but the pose is confident; (2) the matching is optimal within each phase only (Iteration 6) and has ties beyond its tie-break (two different
  sets of bursts with the same position sum); there the solver decides, deterministically for the same input; (3) among exactly equal
  MILP optima the identity pick is the solver's (R5, claim narrowed and not engineered away); (4) the rule is V6's (AlphaPose, 24 fps)
  and untuned for 50 fps sparring, so "unexplained" says nothing about missed labels until the sample is classified; (5) the 29 tables
  are bound by file name only.

### Iteration 6 -- Astra 6 third review (2026-10-07; verdict FIX FIRST: 2 blocking, 2 documentation)

| # | Finding | Fix | Evidence |
|---|---|---|---|
| B1 blocking | `PoseSequence` accepted a `(F, 17)` bool array as the per-frame `detected` mask (`len()` gave F, the dtype passed, `xy[detected]` is valid indexing); both loaders `str()`'d `source` / `model` / `video_fp`, so a number or a vector became plausible text; "every stored field is validated before conversion" was untrue | `detected` must be 1-D (constructor, and on load also bool dtype via `_bool_vector`); `_text` requires a 0-d unicode array before `str()`; every stored field is read through the one `_field` missing-field helper in both loaders; the load docstrings now list exactly what is checked and what is left to the constructor | `(F, 17)` mask refused on load and, with finite coordinates, at construction; integer `detected` refused; numeric `source`, numeric `model`, vector `model`, vector `source`, numeric `video_fp` refused; missing `source` / `model` / `detected` are `ValueError`s naming the field; proper strings load unchanged |
| B2 blocking | No held-out observations reported as 0%: every video unable to estimate a held-out ring gave `0 / max(0, 1)`, printed "0.0% held out" | `pooled_heldout_share` returns `None` on a zero denominator (json `null`); the per-video share is `None` when there is no held-out punch to evaluate; the pooled print uses the absent-safe `_pct` | 16-label video (ring for the full linking, 8 per half: none for the held-out one): exit 0, pooled and per-video share null, counts exactly 0, CLI prints `n/a held out` and `n/a with half held out`, never `0.0%`; empty and all-zero pools are `None`; 0 of 4 is `0.0` (a real zero); a held-out half with a ring but no credited punch is `None` |
| D1 | "Optimal assignment" omitted that the two phases are optimised separately | Wording in `completeness.py` (module and `_match` docstrings), the json `matching` field, the project rules and the Iteration 5 row: optimal within each phase (own labels first, then unattributed), not necessarily the global maximum of explained bursts; exact ties go to the burst that sorts first by (frame, boxer), so to A on a simultaneous tie | the two-phase counterexample is a test of the CURRENT behaviour: owned label [302,337] takes burst 303, unattributed label [298,298] cannot reach 337, which stays `none`, although narrow -> 303 and broad -> 337 would explain both |
| D2 | "Every rate divides by exposure" excluded the whole-video density; "the attribution could not tell whose punch it was" did not describe labels credited to a track outside A and B | the project rules, `find_bursts` and the Iteration 5 text now say the burst rates and the exposure-qualified own-label rate divide by exposure and labels per video boxer-minute is a whole-video density; unattributed means no owner among the selected boxer identities (ambiguous, unreached, or credited to a track outside A and B, a referee for instance) | wording only |

**Mutation check (change broken in the working tree, restored with `git checkout`; the named suite must fail)**
| Mutation | Suite | Result | First failing checks |
|---|---|---|---|
| B1 `detected` shape check removed (load and constructor) | test_ingest | caught, 2 fail | `(F, 17)` mask on load; the same at construction |
| B1 string-field check removed | test_tracks | caught, 5 | numeric source, numeric model, vector model, vector source, numeric video_fp |
| B1 string-field check removed | test_ingest | caught, 2 | numeric source, vector model |
| B2 zero denominator -> 0 | test_linking | caught, 4 | null pooled share; per-video null and `n/a` print; empty / all-zero pool; held-out half with no credited punch |

The first version of the construction-time `(F, 17)` test passed under the mutation (the mask selected NaN coordinates and another
ValueError refused it for the wrong reason); it now uses finite coordinates and asserts the shape rule's message.

**Real data:** `olympic_link.py` on the 29 videos is unchanged: 601 of 1,185 held-out credited punches inside A or B = 0.5072
(50.7%), 1,585 of 2,423 = 65.4% overall; solver max 0.09 s, total 1.24 s. No pooled or per-video share was null there.
**Tests:** `test_linking.py` 263 (was 256), `test_tracks.py` 79 (71), `test_ingest.py` 124 (117); olympic 33, hands 38, labels 50,
pipeline 260, env 36 unchanged.
**Residual risks:** (1) the two phases are a policy (ownership first), so the explained count can be lower than a global assignment
would give; (2) ~~`credited_in_either_share` divides by `max(credited_punches, 1)`~~ **closed in review (a reviewer):** with no
credited punch the share is None (json null, printed n/a) per video and pooled; a measured 0.0 (credited punches, none on A or B) is
still 0.0; 3 tests, mutation `max(nc, 1)` restored -> 1 fail; (3) the two output files of `olympic_completeness.py` are
written one after the other, not as one atomic transaction.

### Mutation check, iterations 1-3 (each guard reverted in a copy; `tests/test_linking.py` must fail)
| Guard reverted | Result | First failing test |
|---|---|---|
| gap limit (reach unbounded) | caught, 4 fail | gap limit inclusive; 3 s hole never bridged |
| speed limit (always plausible) | caught, 6 | 139 px plausible / 141 px not |
| forward-only: both `lo` and `gap < 1` removed | caught, 2 | overlap by 5 frames; same-frame start |
| forward-only: either one alone | SURVIVED (redundant by design: each guards the other) | -- |
| node capacity 2 (overlapping chains) | caught, 7 | one boxer: second identity must be empty |
| chain start fee 0 | caught, 3 | same test (the only boxer gets cut in two) |
| punches worth nothing | caught (crash + fails) | exact boxer sets recovered |
| a chain needs no punch | caught, 1 | ring time valued 10x: no identity without a punch |
| ring prior: eligibility | caught, 3 | punches wrongly credited to a spectator |
| ring prior: share as reward weight | caught, 2 | reward = 10/punch + 0.5/s x ring share |
| credited ids must be in the table | caught, 1 | attributions from another video refused |
| re-acquisition: ordinary gap un-gated | caught, 5 | speed-limit tests |
| overlay: table vs video size/rate | caught, 2 | refuses another frame rate / size |
| overlay: linking vs table | caught, 2 | other source / other length |
| overlay: table longer than video | caught, 1 | longer recording refused |
| completeness: boxer rows by membership | caught (crash) | frame ids not increasing |
| completeness: tolerance inclusive | caught, 1 | 5 frames explains, 6 do not |
| completeness: explained flag | caught, 4 | planted bursts |
| CLI: `--overlay` id check | caught, 5 | `../GH000001` etc. refused |
| CLI: refuse incomplete tracks | caught, 1 | a `--max-frames` table refused |

- **Test:** `tests/test_linking.py` 130 (new); all other suites unchanged and green (tracks 34, olympic 33,
  hands 38, labels 50, ingest 109, env 36, pipeline 260, machine scripts 38).
- **Security pass:** tracks loaded by `PoseTable.load` (`allow_pickle=False`); the only subprocesses are in
  `video.py` (argument lists); the GoPro id is `[A-Za-z0-9]{1,32}` before it becomes a file name, the overlay
  is written only inside `--tracks`, refuses to overwrite the input, and refuses a table or linking from
  another video; stale attributions (ids not in the table), negative counts, NaN parameters and an incomplete
  tracks file are refused; JSON written with `allow_nan=False`; a video that fails is reported and the exit code is 1.
- **Residual risks:** (1) no appearance model: after two people cross with ids lost, the nearer continuation
  wins and the boxers swap (asserted in the tests as the actual behaviour); (2) a referee fragment beside a
  boxer's hole can be absorbed (punch evidence is the only separator), and a punch credited to the referee
  makes his fragment worth 10 points; (3) coverage is deliberately conservative (ring time is worth 0.5/s), so
  holes of seconds remain; (4) a burst on a bridged or fragment-handover frame is not flagged; (5) the rule is
  V6's (AlphaPose, 24 fps, hooks/uppercuts) and untuned for 50 fps sparring; (6) the overlay for GH078416
  (`results/ingest/olympic/GH078416.linked.overlay.mp4`) has NOT been viewed by anyone yet: identity swaps in
  the real data are unmeasured.

## `feat/completeness-credited` -- 2026-10-09

The completeness test on CREDITED fragments only, no linking (`completeness.find_bursts_credited`,
`olympic_completeness.py --credited [--min-punches N]`). Branch from `feat/boxer-linking` (1d67459).

- **Decision (ADR-0012 Option 2, Brendan, 2026-10-09):** two-boxer identity is parked until W9. Pose-only linking swaps boxers and
  absorbs spectators, and two corner-colour attempts failed their non-circular gate (torso colour does not separate the boxers in
  this footage). The completeness question for Phase 1 is answered by testing only the fragments that are credited with a labelled
  punch. No identity is needed: a ByteTrack id never changes person by design, so each credited fragment is one person for its whole
  length. An id switch inside a fragment is the tracker's residual error and is NOT detected here.
- **Design.** Population: every track id credited (`olympic.attribute`) with at least `min_punches` (default 1) labelled punches.
  Per fragment: its rows are split into segments as `boxer_segments` does (the id never changes, so only a hole longer than the bridge
  splits it); bridge, wrist signal and detection, the short-segment and bodyless rules, the exposure mask (processed segments, bridged
  frames included, at least one hand valid) and the assertion that every burst lies in its fragment's exposure are the SAME code as the
  linked boxers' (`completeness._scan_segments`, extracted from `find_bursts`; the 266 existing linking tests are unchanged and green).
  Ownership: a label is OWNED by the fragment it is credited to. Phase 1 matches each fragment's bursts to its own labels, phase 2
  matches the remaining bursts of all fragments to the UNATTRIBUTED labels (credited to no one), each label once; both phases use the
  existing `_match` (optimal within the phase, ties to the burst that sorts first by (frame, key), here the lower track id). Two
  co-present fragments are two people and are kept apart: a burst near a label owned by another fragment is not explained by it.
  Rates, the whole-video density and the seeded simple random sample are the existing ones; the CSV adds `track_id`, the fragment's
  box centre and height (pixels) at the burst frame (interpolated across a bridged hole) and `time_mss` (m:ss.s, rounded to a tenth
  first) so the person can be found by eye. Outputs `results/olympic_completeness_credited.json` and `..._credited_sample.csv`
  (the linking-based files are never overwritten). `--reacquire` with `--credited` is refused; so is `--min-punches` without
  `--credited`.
- **Decision beyond the brief:** a label credited to a track id that is NOT in the population (fewer than `min_punches` credited
  punches) belongs to a person this test does not follow. It is neither owned nor unattributed: it explains nothing and is counted as
  `credited_to_excluded`. Treating it as unattributed would let a person we do not follow excuse the bursts of those we do (mutation 8).
- **The stated limits (in the module docstring, the json `bias`, the CLI print and here; WORDING CORRECTED in Iteration 2, which
  replaced "can only UNDERCOUNT").** This sees only credited fragments. Missed punches outside them (boxer time in a fragment with no
  credited punch, punches too close to call) are never seen, so coverage can only lose; but fragment-level detections can DUPLICATE
  one physical punch when two overlapping ids follow the same person. So neither a bout-level upper nor a lower bound follows
  without de-duplication. A clean result is weak evidence of completeness; a dirty one shows labels are missing inside the credited
  fragments, though the count may include duplicates. A credited fragment can be a non-boxer wrongly credited (a referee, a
  coach, a spectator); the fragments with exactly one credited punch are the least certain and are counted.
- **Tests:** new `tests/test_completeness_credited.py`, 75 checks on planted scenes (two co-present credited fragments with a
  simultaneous burst, an uncredited referee, a one-punch fragment; `min_punches` 1/2/3 and bad values; an ambiguous label explains one
  burst once across fragments and the tie goes to the lower id; own labels first; a bridged-frame peak counted in numerator and
  exposure with its interpolated box; a 3-row segment adds no exposure and its label is outside it; a 2-frame hole stays one
  segment and a 3-frame hole ends it; the exposure RuntimeError; a punch credited to an id missing from the table refused (Iteration 2 widens this to any attribution that differs from the table's own); box_at / mss including
  59.96 s = 1:00.0; the CLI end to end on a planted dataset: exact `track_id`, box centre and height, `time_s`, `time_mss` for a burst
  on a bridged frame and one after the first minute, `inclusion_prob` = n/N = 0.666667, reproducible sample, `--credited
  --reacquire` refused with nothing written, `--min-punches 0 / -3` and `--min-punches` alone refused, default outputs leave sentinel
  linking files untouched, no credited punch runs to the end with null rates).

**Mutation check (change applied in a scratch copy; the named suite must fail)**
| # | Mutation | Suite | Result | First failing checks |
|---|---|---|---|---|
| 1 | ownership ignored (phase 1 skipped, every label unattributed) | credited | caught, 9 fail | P's own match; Q's burst not explained by P's label; S |
| 2 | uncredited fragments included in the population | credited | caught, 9 | population is {1, 2, 4}; the referee's burst not counted; min_punches=2 |
| 3 | one-to-one removed (every burst takes its nearest label) | credited / linking | caught, 2 / 12 | exactly one burst explained by the unattributed label; label used once / twelve existing matching tests |
| 4 | exposure from all rows instead of processed segments | credited | caught, 4 | bridged peak: 400 exposure frames; 3-row tail adds none; label outside exposure |
| 5 | burst box from the fragment's first row | credited | caught, 2 | exact CSV values; bridged-frame box |
| 6 | box not interpolated on a bridged frame | credited | caught, 3 | box_at; exact CSV values; bridged-frame box |
| 7 | m:ss.s truncates instead of rounding first | credited | caught, 1 | 59.96 s is 1:00.0 |
| 8 | label credited to an excluded fragment treated as unattributed | credited | caught, 2 | credited_to_excluded; still unexplained |
| 9 | burst-in-exposure assertion removed from the shared scan | credited / linking | caught, 1 / 1 | the RuntimeError test (both suites) |
| 10 | CLI: `--credited --reacquire` accepted | credited | caught, 1 | refused with exit 2 |
| 11 | CLI: credited default names are the linking-based ones | credited | caught, 2 | defaults leave the sentinels untouched; the linking run writes its own names |
| 12 | CLI: `inclusion_prob` = 1 instead of n/N | credited | caught, 1 | 0.666667 |
| 13 | CLI: csv writes the frame in the track_id column | credited | caught, 1 | exact values |
| 14 | `min_punches` ignored | credited | caught, 3 | min_punches=2 / =3 |

**Numbers (29 Olympic videos, rule `punch_finder_v6.json`, 0.1 s tolerance, `results/*.json`, gitignored; nothing viewed or judged)**
| Quantity | Credited, min 1 | Credited, min 2 | Linking (default, for comparison) |
|---|---|---|---|
| fragments in the population (with exactly 1 credited punch) | 1,245 (679) | 566 (0) | two identities per video |
| exposure minutes (summed over fragments) | 161.4 (91.1 with at least one fragment in exposure) | 96.1 | 146.4 |
| bursts / unexplained | 10,077 / 7,318 | 6,227 / 4,342 | 7,984 / 5,766 |
| unexplained per exposure minute | 45.4 | 45.2 | 39.4 |
| bursts per exposure minute | 62.5 | 64.8 | 54.5 |
| own labels in exposure per exposure minute | 14.9 (2,407 in exposure, 16 outside) | 18.1 (1,741, 3) | 10.8 (1,580) |
| labels owned / unattributed / credited to an excluded fragment | 2,423 / 2,182 / 0 | 1,744 / 2,182 / 679 | 1,585 / 3,020 / n.a. |
| labels matched (own + unattributed) | 1,473 + 1,286 = 2,759 | 1,053 + 832 = 1,885 | 973 + 1,245 = 2,218 |
| sample N (population), inclusion probability | 7,318, 0.00547 | 4,342, 0.00921 | 5,766, 0.00694 |
| segments / short / bodyless | 2,758 / 260 / 1 | 1,380 / 116 / 1 | |
| labels per whole-video boxer-minute (a different population) | 22.1 | 22.1 | 22.1 |

- The credited population holds more of the labels (2,423 owned of 4,605, against 1,585 on the two linked identities) because it needs
  no identity, and the unattributed share falls from 65.6% to 47.4% (the labels attribution itself could not call). The rates are NOT
  a finding about the labels: the populations differ (credited ids include referees, coaches and spectators; the exposure is summed
  over fragments, so two people on screen together count twice, and one person under two overlapping ids would too), the rule is V6's,
  and unexplained bursts are candidates until the sample is classified. Samples (40 each, seed 20261007) are for YOU to classify;
  nothing was viewed or judged.
- 3,850 of the 10,077 bursts (38%) lie in the 679 one-punch fragments, 2,904 of them unexplained. That is where a wrongly credited
  non-boxer would hide, which is why `--min-punches 2` was run beside the default.

**Tests (all suites):** `test_completeness_credited.py` 75 (new), `test_linking.py` 266 (unchanged), `test_tracks.py` 79,
`test_olympic.py` 33, `test_hands.py` 38, `test_labels.py` 50, `test_ingest.py` 124, `test_env.py` 36, `test_pipeline.py` 260.
**Security pass:** the only new inputs are two integers (`--min-punches`, validated whole and >= 1 at the CLI and in the function,
bool refused) and a track id taken from the attributions (refused when it is not in the table); no new path is built from user input,
no dependency, the CLI computes everything before writing, JSON written with `allow_nan=False`, a burst outside its exposure still
raises.

**Residual risks:** (1) the limits above: credited fragments only, so a clean result is weak evidence, and duplicate ids can inflate a dirty one; (2) a credited fragment can be a
non-boxer, and one-punch fragments carry 38% of the bursts; (3) an id switch inside a fragment is undetected; (4) the same person under
two overlapping ids is counted twice (Iteration 2 counts the likely cases, `possible_duplicates`, and does not subtract them), so the pooled exposure is fragment-minutes, not video time; (5) the rule is V6's (AlphaPose,
24 fps) and untuned for 50 fps sparring; (6) the 29 tables are bound by file name only; (7) matching is optimal within each phase,
ties go to the lower track id.

### Iteration 2 -- Astra 6 review of `feat/completeness-credited` (2026-10-09; verdict FIX FIRST: 1 blocking, 2 not)

| # | Finding | Fix | Evidence |
|---|---|---|---|
| B1 blocking | Attributions were not bound to the table that produced them: with the same id set but the people swapped (a re-tracked video), attributions of table A passed with table B went through every check and silently changed the population, matches and rates | `completeness.check_attrs_bound`, called by BOTH `find_bursts_credited` and `find_bursts`: recompute `olympic.attribute(events, table)` (default parameters) and require, event by event, the same credited track id, wrist, distance, runner-up distance, ambiguity flag and hand-clear flag (`ValueError` naming the event). What is checked is exactly that, not "ids exist in the table": the earlier missing-id test is kept and checked first. The CLI passes the attributions `load_video` made (computed once per video there) and the test recomputes them once more; nothing else changed | same ids, ownership swapped (ids 1 / 2 credited 1,1,1,2,2,2 in table A and 2,2,2,1,1,1 in table B): refused by both functions, accepted with the table's own attributions; credit removed, ambiguity flipped and wrist changed after the fact are refused. Attributions made with non-default `attribute` parameters would be refused too by construction (not separately tested; pass the defaults) |
| N1 wording | "can only UNDERCOUNT" overstated the inference: two overlapping ids on one person each report the same unlabelled punch | Wording replaced in the module docstring, `BIAS_CREDITED` (json `bias`, CLI print), the CLI docstring, the project rules and above: restricting to credited fragments loses missed punches outside them (coverage can only lose), while fragment detections can duplicate a punch, so neither a bout-level upper nor a lower bound follows without de-duplication. New number `possible_duplicates` (per fragment, per video, pooled; `possible_duplicate_pairs` beside it): an unexplained burst with an unexplained burst of ANOTHER credited fragment within the matching tolerance (0.1 s) whose boxes overlap at IoU >= `DUP_IOU` = 0.3 at the earlier of the two frames. 0.3 is a generous screening level (one person under two ids gives IoU far above 0.5, two fighters in a clinch overlap partly); it comes from the review brief and was not tuned on the Olympic numbers. Reported, never subtracted | two ids on one person (4 px apart, IoU 0.87): 2 bursts, 1 pair; two separate people bursting together: 0; 20 px shift (IoU 0.47) counts, 35 px (0.23) does not; a twin 5 frames later counts, 6 does not; an explained twin does not count |
| N2 tests | Nothing protected pooling or the summed exposure of overlapping fragments (a mean of per-video rates, or a union exposure, passed the one-fragment fixture) | tests only | videos of 2 unexplained / 1.0 min (2.0 per min) and 18 / 3.0 min (6.0 per min) pool to 5.0, not the mean 4.0; own labels 3 / 1.0 min and 3 / 3.0 min pool to 1.5, not 2.0; two fragments on screen for the same 400 frames keep 800 exposure frames, not the union's 400, and the rates divide by 800 |

The CSV was NOT changed (no `possible_duplicate` column); a sampled row's twin can be found by its frame and box.

**Existing assertions replaced (nothing weakened):** the two checks that asserted the words "UNDERCOUNT" / "decisive" in the json bias and
the CLI print now assert the corrected statement ("coverage can only lose", "DUPLICATE", "neither a bout-level upper nor a lower bound",
"weak evidence", no "UNDERCOUNT", "possible twin"), which is a stricter set of facts about the same text.

**Mutation check (change applied in a scratch copy; the named suite must fail)**
| # | Mutation | Suite | Result | First failing checks |
|---|---|---|---|---|
| 15 | B1: `find_bursts_credited` does not verify the attributions | credited | caught, 3 fail | swapped ownership refused; credit removed; ambiguity flipped |
| 16 | B1: `find_bursts` does not verify them | credited / linking | caught, 2 / 0 | `find_bursts` refuses swapped ownership; wrist changed (the linking suite never swaps tables) |
| 17 | B1: verification compares only whether a track is credited, not which | credited | caught, 4 | swapped ownership refused by both functions |
| 18 | N1: duplicate counter without the IoU check | credited | caught, 2 | two separate people are not duplicates; the threshold bites on both sides |
| 19 | N1: duplicate counter ignores the time window | credited | caught, 1 | 5 frames counts, 6 does not |
| 20 | N1: duplicate counter counts explained twins | credited | caught, 1 | an explained twin does not count |
| 21 | N2: pooled rate is the mean of per-video rates | credited | caught, 3 | 5.0 not 4.0; 1.5 not 2.0; the no-credited-punch run |
| 22 | N2: a video's exposure is the longest fragment, not the sum | credited | caught, 2 | 800 frames, not 400; rates over 800 |

**Real data, `--credited` and `--credited --min-punches 2`, re-run:** every pre-existing number is unchanged (the same fragments, bursts,
unexplained, exposure, labels, rates, and the two samples are byte-identical); only the new keys appear.
| Quantity | Credited, min 1 | Credited, min 2 |
|---|---|---|
| unexplained bursts / `possible_duplicates` (share) | 7,318 / 177 (2.4%) | 4,342 / 66 (1.5%) |
| `possible_duplicate_pairs` | 89 | 33 |
| videos with no possible duplicate | 6 of 29 | 16 of 29 |
| most in one video | 22 (GH089681), 16 (GH109681, GH069681) | 10 (the same three videos) |

These are SCREENING counts, not a bound: 177 of 7,318 unexplained bursts (2.4%, 89 candidate pairs) at min 1 and 66 of 4,342
(1.5%, 33 pairs) at min 2 are flagged as possible duplicates. Duplication outside the screen is unmeasured: one physical punch can give
peaks more than 0.1 s apart on two ids, boxes overlapping at IoU below 0.3, or a twin that was matched to a label, and none of those is
flagged. (Corrected after Astra 6's re-review: an earlier wording called 89 / 7,318 an upper bound on the inflation and concluded that
duplication could not explain the rate; the counter cannot establish either.) Whether the unexplained bursts are punches is for the sample to say.
**Tests (all suites):** `test_completeness_credited.py` 95 (was 75), `test_linking.py` 266, `test_tracks.py` 79, `test_olympic.py` 33,
`test_hands.py` 38, `test_labels.py` 50, `test_ingest.py` 124, `test_env.py` 36, `test_pipeline.py` 260.
**Residual risks added:** (1) `possible_duplicates` is a screen: boxes that overlap at IoU >= 0.3 and bursts within 0.1 s, not a proof of
one person, and a twin that was matched to a label is not counted; (2) the verification recomputes attribution with default parameters,
so attributions made with other parameters are refused rather than trusted.


## `feat/frame-skips` -- 2026-10-09

Branch from `feat/completeness-credited` (b912027, PR #12 untouched).

**The finding (Brendan; confirmed numerically on GH069681).** The Olympic GoPro files have perfectly regular packet timestamps (every
interval 20 ms; `probe` reports no VFR), but the camera drops CAPTURED frames: what is on screen jumps forward in one frame while the
timestamps stay regular, and the wrist-speed rule reads the jump as a fast punch. Of the four unexplained bursts of GH069681 Brendan
classified, the two he called "not a punch" (frames 1287 and 7018, tracks 64 and 1203) sit on such a jump; the two real unlabelled
punches (1965, 13293) do not. Measured as the mean absolute grey difference between consecutive frames (image decimated 4x), divided by the
video's median: 3.46x at frame 1287 and 5.36x at 7019 (a reviewer's earlier inspection read 3.39x and 5.43x), nothing comparable at 1965 or
13293. No exact duplicate frame exists (the minimum change is 0.07 grey levels against a median of 1.18), so it is a skip, not a repeat.

**The rule (`src/boxingvi/skips.py`).** Frame f is a SKIP when `change[f] > 3.0 x` the median of `change` over [f-10, f+10] EXCLUDING f.
The neighbours' median is the local normal: a pan or a crowd surge raises it and is not flagged, an isolated jump is. Frame 0 (NaN) is
never a skip; NaN never enters a median; there is no absolute floor (none was observed to be needed; one would be a tuned constant). On
GH069681 it finds 147 skips (one every 1.9 s) where the first inspection found 145; the small difference is the grey conversion and the
decimation.

**Declaration.** The rule and `SKIP_RATIO = 3.0` come from Brendan's observation and a reviewer's inspection of ONE development video
(GH069681). They were fixed by reading that observation, not tuned to any completeness, precision or score, and no other video was used to
choose them. That they transfer to the other 28 videos is a claim the numbers below test, not one the code makes. (Brendan's four classified rows are the only ground truth about skips used in this section; the rest of the sample he is still
classifying was not used for anything.)

**What was built.**
1. `video.frame_change(path, info=None, step=4)`: one float per frame, NaN for frame 0, decoded once through `iter_frames`; no pixel
   leaves the function (an aggregate per frame only). 2. `skips.find_skips`, `skips.Skips` (change array bound to the source file name,
   the frame count and the video's content fingerprint; `<tracks>/<id>.framechange.npz`, saved atomically, strict loader, no pickle).
3. `olympic_framechange.py --data --tracks [--only] [--overwrite]`. 4. `completeness.cut_at_skips` inside `_scan_segments`, so
   `find_bursts` and `find_bursts_credited` take `skips=`: every segment is cut between f-1 and f for each skip f inside it (also a skip on
   a frame the segment is missing: the hole the bridge would have interpolated across), BEFORE bridging, smoothing or differencing;
   `skips=None` is the old output exactly (every existing check unchanged). 5. `olympic_completeness.py --skips`
   (both modes), `--skips-measure` (loads the same files, reports the same numbers, does NOT cut: the "before"), `--skipcheck SAMPLE.csv
   --skipcheck-out OUT.csv`. New default output names (`..._skips.json`, `..._skips_sample.csv`; `..._skipsmeasure...`); a missing,
   malformed or mismatched framechange file stops the whole run with exit 2 and writes nothing.

**Deviations from the brief (and why).** (a) `--skips-measure` and `--skipcheck` are added: "before and after" needs the near-skip counts
of the b912027 behaviour, and the existing JSONs do not contain them; the measure run reproduces every pre-existing number (below), so it
is also a regression check on the real data. (b) The framechange file stores the fingerprint of the VIDEO FILE it decoded (always
present), and the run checks it against the video and, when the table has one, against the table's `video_fp`; the brief said "the table's
video_fp if present", which is the weaker half of that. (c) A synthetic "dropped frame" in the tests is a dropped STRETCH (the pattern
jumps 6 px where it moves 1 px a frame): a one-frame drop of a 1 px/frame pattern is a 2x change and, correctly, not a skip; the real
camera's jumps are 3-5x. (d) `still scene -> no skip` is tested on a flat picture: a static GRATING re-encoded by H.264 at crf 18 is not
bit-still (changes of 0.001-0.006, median 0.0013) and the rule, having no floor, flags its frame 1 (0.0059). Real footage is orders of
magnitude above that (smallest per-video median change 1.04), so it did not occur there, but it is a property of the rule (residual risk 4).

**Tests (all suites):** `test_frame_skips.py` 155 (new), `test_completeness_credited.py` 95, `test_linking.py` 266, `test_tracks.py` 79,
`test_olympic.py` 33, `test_hands.py` 38, `test_labels.py` 50, `test_ingest.py` 124, `test_env.py` 36, `test_pipeline.py` 260; every
pre-existing count and result is unchanged. Negative tests at the trust boundaries: a skip list that is unsorted, repeated, at frame 0, past
the video, float or 2-D; a change array that is integer, 2-D, negative, infinite, NaN anywhere but frame 0; a framechange file with a wrong
source, length or fingerprint, a missing field, a pickled array, garbage, a truncated zip, a bad ratio / window / step; both CLIs'
refusals leave no output behind.

**Mutation check (change applied in a scratch copy; `test_frame_skips.py` must fail)**
| # | Mutation | Result | First failing checks |
|---|---|---|---|
| M1 | skips ignored in `_scan_segments` | caught, 8 fail | the jump is cut out; two segments; `find_bursts` cuts; `--skips` leaves one unexplained |
| M2 | the window includes f itself | caught, 1 | 4.6 between ten 1.0s and ten 2.0s is a skip |
| M3 | ratio test replaced by a fixed absolute threshold | caught, 6 | 30.1 over 10 is a skip and 25 is not; a pan is not flagged; a ramp; a still stretch |
| M4a | `Skips.verify` checks nothing | caught, 11 | wrong name / length / fingerprint all accepted |
| M4b | the CLI does not compare the file's fingerprint to the video | caught, 5 | a file measured on another file (the later failures are the first's leftover output) |
| M5 | `>=` instead of `>` | caught, 3 | exactly 3.0x is not a skip; zeros in a still stretch |
| M6 | a skip cuts only at a row, not at a hole | caught, 1 | a skip on a missing frame cuts there |
| M7 | `--skips-measure` also cuts | caught, 3 | same bursts as without skips; the before numbers |
| M8 | skipcheck ignores the hand | **missed at first** (151 checks), caught after adding 9 rows, 1 | other hand is false |
| M9 | base rate counts all video frames, not exposure | caught, 4 | 10 of 150 exposure frames |
| M10 | NaN not ignored in the medians | caught, 26 | 2.99x / 3.01x boundary and 25 more |
| M11 | the near-skip window is one-sided | caught, 7 | counts, base rate, no wrap at the start |
| M12 | skipcheck ignores the track id | **missed at first**, caught after the same 9 rows, 1 | other track is false |

**Real data.** `olympic_framechange.py` on all 29 videos: 312,774 frames (104.3 video minutes), **3,367 skips = 32.3 per video minute**
pooled (one every 1.9 s), per video 22.1 (GH089681) to 50.4 (GH128416) per minute, median 32.8; the smallest per-video median change is
1.04 grey levels. Wall time: 5,560 s summed over videos (1.5 h of CPU-bound decoding at ~10-20 ms a frame, measured with up to four of
these running at once beside another job on the same machine; GH069681 alone, 13,800 frames, took 147 s). Two videos were measured twice
by two of the parallel processes (a race on the output name; Iteration 3 corrects the claim that this was harmless: both writers happened to produce the same content and neither failed, but the shared temporary name was a bug, now fixed).

| Quantity (`--credited`) | min 1, b912027 | min 1, `--skips` | min 2, b912027 | min 2, `--skips` |
|---|---|---|---|---|
| bursts | 10,077 | 9,685 | 6,227 | 6,029 |
| unexplained | 7,318 | 6,903 | 4,342 | 4,117 |
| explained | 2,759 | 2,782 | 1,885 | 1,912 |
| exposure minutes (summed over fragments) | 161.36 | 161.21 | 96.05 | 95.98 |
| unexplained per exposure minute | 45.35 | 42.82 | 45.21 | 42.90 |
| bursts per exposure minute | 62.45 | 60.08 | 64.83 | 62.82 |
| own labels per exposure minute | 14.92 | 14.93 | 18.13 | 18.14 |
| segments (short) | 2,758 (260) | 6,689 (403) | 1,380 (116) | 3,667 (191) |
| `possible_duplicates` | 177 | 145 | 66 | 56 |
| unexplained within 2 frames of a skip | 1,113 / 7,318 = 15.2% | 52 / 6,903 = 0.8% | 624 / 4,342 = 14.4% | 35 / 4,117 = 0.9% |
| explained within 2 frames of a skip | 209 / 2,759 = 7.6% | 32 / 2,782 = 1.2% | 137 / 1,885 = 7.3% | 26 / 1,912 = 1.4% |
| base rate (exposure frames within 2 of a skip) | 4.8% (23,032 / 484,084) | 4.7% (22,631 / 483,614) | 4.6% (13,163 / 288,154) | 4.5% (12,962 / 287,935) |
| unexplained share / base rate | 3.20x | 0.16x | 3.15x | 0.19x |
| explained share / base rate | 1.59x | 0.25x | 1.59x | 0.30x |

The "b912027" columns are from `--skips-measure` and equal the pre-existing `olympic_completeness_credited.json` / `_min2.json` in
every burst, label, exposure and rate (compared programmatically): no burst was lost to a re-implementation. The pooled near-skip
excess before the fix is real and general (15% of unexplained bursts against a 5% base rate, in 29 videos, not only GH069681's 14%),
and within 2 frames of a skip it is gone after.

**Three things the table says that the brief did not expect.** (1) The excess did not just vanish, it reversed: 0.16x. (2) The
unexplained rate fell by only 5.6% (45.35 to 42.82 per exposure minute), although 1,061 unexplained bursts were removed near skips,
because the cut ADDS bursts a little further out. Bursts per 1,000 exposure frames by distance to the nearest skip (min 1, unexplained /
explained), before then after: 0-2 frames 48.3 / 9.1 then 2.3 / 1.4; 3-5 frames 12.9 / 4.0 then 22.8 / 7.1; 6-10 frames 11.4 / 4.4 then
16.7 / 6.1; 11 or more frames 13.7 / 5.8 then 14.1 / 5.9. So the far field is unchanged and a ring of extra bursts now sits 3-10 frames
from each skip (1.6x and 1.2x the far-field rate). The likely cause, not isolated: the cut pieces are about 1.9 s long, the smoothing is
edge-normalised (less smoothing at a piece's end, so a noisier speed), the body scale is re-estimated per piece, and `find_peaks` never
reports a piece's first or last sample. (3) Explained went UP (+23 at min 1, +27 at min 2): the one-to-one matching ran on a changed burst set.
The four 2-frame-window numbers (0.8% / 1.2%) therefore overstate what the cut achieved: a real fix would also have to stop the edge ring
(for example by giving a cut piece the scale of the fragment and ignoring peaks inside the smoothing half-window of a cut), which this
branch did not do because it would be a second, unevaluated tuning.

**Brendan's four GH069681 rows (min 2).** 1287 / track 64: **gone** (no burst within 10 frames of it remains; skip at that frame).
7018 / track 1203: **gone** (skip one frame away). 1965 / track 253: **still an unexplained burst**. 13293 / track 2268: **still an
unexplained burst**. Exactly the expected outcome.

**The sample** (`results/olympic_completeness_credited_min2_sample_skipcheck.csv`, the 40 existing rows with
`still_unexplained_with_skips`; the original CSV was only read, its hash is unchanged): 27 true, 13 false. Eight of the 13 are within
2 frames of a skip (distance 0, 0, 1, 1, 1, 1, 1, 2 frames) and all of those vanished, as intended. Five are not: they are 5, 6, 9, 13
and 174 frames from the nearest skip. Four of them lost their burst outright (no burst of that track within 10 frames remains); three
of the four had peaks of only 3.67 to 4.48 in the old run against the threshold 3.62, which is consistent with the per-piece scale effect
above, not proven. The fifth (GH178416 frame 14485) still has its burst, three frames earlier (14482), so the strict 2-frame
test calls it false. The 27 rows kept had skips 4 to 444 frames away. This is also why `still_unexplained_with_skips = false` does not
mean "was caused by a skip": for 5 of the 13 it means "changed".

**Residual risks.** (1) A real punch exactly across a skip is now split in two (a punch at 150 with a skip 2 frames away came out as
one or two bursts within 140..160 in the synthetic test; on the real data it is the edge ring above); the cut cannot tell a real
jump from a skip-made one when a real punch coincides with a dropped frame. (2) Skips at a scene cut or a replay edit would be cut as
well (harmless: a cut only removes differencing across it) and counted. (3) The rule is GoPro-specific in the sense that it was seen
on this camera's drops; another encoder's frame-doubling would not trace as a spike. (4) No absolute floor: on a stretch where the
picture hardly changes at all (a frozen frame, a black leader) codec noise can be flagged; none was seen in the 29 videos, and the
consequence of a false skip is a cut, not a burst. (5) The `skipcheck` window is exactly 2 frames and the track, hand and video: a
burst that the cut shifted by 3 frames counts as gone. (6) The skip signal is one aggregate per frame on a 4x-decimated grey image,
so a drop during uniform motion of a small part of the picture (a dropped frame that only moved the boxers a little) is under the
rule's ratio and is not found. (7) The fingerprint is still the first and last 4 MiB, not every byte.


### Iteration 2 -- masking replaces cutting (2026-10-09)

**Why the cut was replaced.** Iteration 1 cut every segment at each skip. The table above shows what that cost: a ring of extra
unexplained bursts 3-10 frames from each skip (1.61x and 1.18x the far-field rate at 3-5 and 6-10 frames), and 5 of the 13 sample rows
that dropped out were not near a skip. A cut makes new segment edges: edge-normalised smoothing, a body scale re-estimated on a piece
about 1.9 s long, and `find_peaks` never reporting a piece's first or last sample. Masking keeps the segments, the body scale and the
smoothing exactly as they are without skips and only removes what a jump can have contaminated. `cut_at_skips` is deleted (nothing uses
it) together with its tests, which tested only that function; every other Iteration-1 check stands, and the counts that changed are
the ones that asserted cut-specific numbers (2 segments, 400 exposure frames), replaced by the masked ones.

**The mask (`completeness.skip_mask_half`, `skip_mask`, `_scan_segments`).** After `wrist_signal`, per hand:
1. *Signal reach, derived.* `wrist_signal` smooths with a centred moving average of `punchfind.smooth_window(fps, smooth_s)` frames
   (5 at 50 fps and 0.1 s; half-width h = 2; one definition, now shared by the signal and the mask) and then takes `np.gradient`, which at
   frame j reads the smoothed values at j-1 and j+1. A step between f-1 and f enters the smoothed value at i when i in [f-h, f+h-1]; the
   gradient at j is contaminated when j in [f-h-1, f+h] = [f-3, f+2]. The mask is symmetric, h + 1 = 3 either side of f
   (`skip_mask_half = smooth_window // 2 + 1`, computed from `smooth_s` and the frame rate, never written as 3). The signal there is set to 0.
2. *Wider where interpolation spreads the jump.* If f lies in a table hole that `punchfind.bridge` interpolated across, the jump
   becomes a ramp over the whole hole and the mask covers the hole plus the reach beyond each end; if the zone meets one hand's
   interpolated hole (a low-confidence stretch of at most the bridge length), that hand's mask grows over it too.
3. *Exposure, one frame wider (a finding, not in the brief).* The frames within `skip_mask_half + 1` of a skip leave the exposure and
   cannot be a burst; the extra frame keeps its true signal. With the mask at the derived reach and the same exposure, the first
   unmasked frames carried 1.56x (min 1) and 1.63x (min 2) the far-field unexplained rate, and widening the mask to 4 and to 5 frames only moved
   the excess to the new edge (1.91x, then 1.30x at 6-10), so it was the mask's own edge, not a contamination reach: `detect` calls
   frame j a peak by comparing it with j-1 and j+1, so next to a zeroed frame the flank of a peak that is hidden under the mask is
   "higher than a zero" and the punch came back as a burst beside the mask. Keeping the extra frame's true signal and refusing it as a
   peak removes that (and takes the frame out of numerator and denominator alike). Zeroing the signal itself still matters: a giant
   skip-made peak would otherwise suppress a real punch within the 0.4 s minimum gap (test: a punch 12 frames after a jump).
   The burst-in-exposure assertion still holds (a burst outside the exposure is a RuntimeError). The design with the signal reach only
   is reported beside the final one below.

**Gate (unexplained bursts per 1,000 exposure frames by distance to the nearest skip, each of 3-5, 6-10, 11-25 within +-25% of the
>25 rate; 0-2 must be empty).** Final design: **PASS at min 1 (1.14x, 1.04x, 0.99x) and at min 2 (1.17x, 1.04x, 1.02x); 0-2 is empty.**
The signal-reach-only mask FAILS the 3-5 bin (1.56x, 1.63x), as does the cut (1.61x, 1.66x). With no skips the 0-2 bin is 3.36x and 3.32x. The
3-5 bin of the final design holds a single offset (5 frames) and ~150 bursts at min 1, 86 at min 2: a Poisson standard error of about
8% and 11% of the bin's rate, so 1.14x and 1.17x are 1-2 standard errors above 1.0, not "a few percent".
**Declared (reviewer, review):** the gate was used to CHOOSE between designs (cut, signal-reach mask, mask plus one exposure frame) on the
same 29 videos it then reports, so the PASS is not independent evidence. The final design is physically motivated (only the frame
`detect` can falsely call a peak is added, nothing was widened to fit), but its confirmation is Brendan's classification of the sample
rows, not this gate.

**The bin tables, all three designs side by side**

Unexplained bursts per 1,000 exposure frames by distance to the nearest skip, `--credited` min 1 (burst count / exposure frames; x = ratio to the >25 rate)

| Distance (frames) | no skips | cut (Iteration 1) | mask, signal reach only | mask, final |
|---|---|---|---|---|
| 0-2 | 48.3 (1113/23,032) x3.36 | 2.3 (52/22,631) x0.16 | empty (masked) | empty (masked) |
| 3-5 | 12.9 (358/27,783) x0.90 | 22.8 (633/27,733) x1.61 | 22.4 (410/18,296) x1.56 | 16.4 (149/9,091) x1.14 |
| 6-10 | 11.4 (519/45,600) x0.79 | 16.7 (762/45,590) x1.18 | 15.4 (701/45,519) x1.07 | 14.9 (674/45,372) x1.04 |
| 11-25 | 12.2 (1352/110,943) x0.85 | 13.8 (1528/110,934) x0.97 | 14.3 (1584/110,942) x1.00 | 14.2 (1578/110,927) x0.99 |
| >25 | 14.4 (3976/276,726) x1.00 | 14.2 (3928/276,726) x1.00 | 14.3 (3965/276,726) x1.00 | 14.3 (3965/276,726) x1.00 |

Explained bursts per 1,000 exposure frames by distance to the nearest skip, `--credited` min 1 (burst count / exposure frames; x = ratio to the >25 rate)

| Distance (frames) | no skips | cut (Iteration 1) | mask, signal reach only | mask, final |
|---|---|---|---|---|
| 0-2 | 9.1 (209/23,032) x1.51 | 1.4 (32/22,631) x0.24 | empty (masked) | empty (masked) |
| 3-5 | 4.0 (112/27,783) x0.67 | 7.1 (197/27,733) x1.19 | 7.3 (133/18,296) x1.22 | 4.5 (41/9,091) x0.76 |
| 6-10 | 4.4 (199/45,600) x0.73 | 6.1 (278/45,590) x1.03 | 6.2 (282/45,519) x1.04 | 6.0 (273/45,372) x1.01 |
| 11-25 | 5.2 (578/110,943) x0.87 | 5.7 (630/110,934) x0.96 | 5.7 (636/110,942) x0.96 | 5.8 (638/110,927) x0.96 |
| >25 | 6.0 (1661/276,726) x1.00 | 5.9 (1645/276,726) x1.00 | 6.0 (1651/276,726) x1.00 | 6.0 (1651/276,726) x1.00 |

Unexplained bursts per 1,000 exposure frames by distance to the nearest skip, `--credited` min 2 (burst count / exposure frames; x = ratio to the >25 rate)

| Distance (frames) | no skips | cut (Iteration 1) | mask, signal reach only | mask, final |
|---|---|---|---|---|
| 0-2 | 47.4 (624/13,163) x3.32 | 2.7 (35/12,962) x0.19 | empty (masked) | empty (masked) |
| 3-5 | 13.7 (216/15,815) x0.96 | 23.4 (370/15,797) x1.66 | 23.2 (242/10,411) x1.63 | 16.7 (86/5,163) x1.17 |
| 6-10 | 11.2 (291/25,936) x0.79 | 16.7 (432/25,936) x1.18 | 15.3 (396/25,895) x1.08 | 14.8 (382/25,816) x1.04 |
| 11-25 | 12.4 (790/63,683) x0.87 | 13.9 (886/63,683) x0.99 | 14.5 (926/63,682) x1.02 | 14.5 (923/63,679) x1.02 |
| >25 | 14.3 (2421/169,557) x1.00 | 14.1 (2394/169,557) x1.00 | 14.2 (2412/169,557) x1.00 | 14.2 (2412/169,557) x1.00 |

Explained bursts per 1,000 exposure frames by distance to the nearest skip, `--credited` min 2 (burst count / exposure frames; x = ratio to the >25 rate)

| Distance (frames) | no skips | cut (Iteration 1) | mask, signal reach only | mask, final |
|---|---|---|---|---|
| 0-2 | 10.4 (137/13,163) x1.49 | 2.0 (26/12,962) x0.29 | empty (masked) | empty (masked) |
| 3-5 | 4.5 (71/15,815) x0.64 | 7.7 (122/15,797) x1.12 | 7.2 (75/10,411) x1.04 | 4.6 (24/5,163) x0.67 |
| 6-10 | 4.9 (127/25,936) x0.70 | 7.0 (182/25,936) x1.01 | 7.1 (185/25,895) x1.03 | 6.9 (177/25,816) x0.99 |
| 11-25 | 5.8 (367/63,683) x0.83 | 6.4 (408/63,683) x0.93 | 6.4 (409/63,682) x0.93 | 6.4 (408/63,679) x0.92 |
| >25 | 7.0 (1183/169,557) x1.00 | 6.9 (1174/169,557) x1.00 | 6.9 (1177/169,557) x1.00 | 6.9 (1177/169,557) x1.00 |

("Mask, signal reach only" = zero over the derived reach and remove only those frames from the exposure; "mask, final" = the design above.)
The unexplained 0-2 bin of "no skips" and "cut" is the skip-made excess of Iteration 1; the explained rows are not gated. The
explained 3-5 cell of the final design is low (0.76x and 0.67x) on 41 and 24 bursts.

**Before / after (`--credited`; "no skips" = b912027, reproduced by `--skips-measure`).**
| Quantity | min 1 no skips | min 1 cut | min 1 mask (final) | min 2 no skips | min 2 cut | min 2 mask (final) |
|---|---|---|---|---|---|---|
| bursts | 10,077 | 9,685 | 8,969 | 6,227 | 6,029 | 5,589 |
| unexplained | 7,318 | 6,903 | 6,366 | 4,342 | 4,117 | 3,803 |
| explained | 2,759 | 2,782 | 2,603 | 1,885 | 1,912 | 1,786 |
| exposure minutes | 161.36 | 161.21 | 147.37 | 96.05 | 95.98 | 88.07 |
| **unexplained per exposure minute** | **45.35** | **42.82** | **43.20** | **45.21** | **42.90** | **43.18** |
| bursts per exposure minute | 62.45 | 60.08 | 60.86 | 64.83 | 62.82 | 63.46 |
| own labels per exposure minute | 14.92 | 14.93 | 15.63 | 18.13 | 18.14 | 18.95 |
| own labels outside the exposure | 16 | 16 | 120 | 3 | 3 | 75 |
| `possible_duplicates` | 177 | 145 | 123 | 66 | 56 | 52 |
| segments (unchanged by a mask) | 2,758 | 6,689 | 2,758 | 1,380 | 3,667 | 1,380 |

The unexplained rate falls 4.7% (min 1) and 4.5% (min 2) against no skips. The bursts within 2 frames of a skip were 15% of the
unexplained ones, and the rate falls by less than that because peaks that a skip-made peak used to hide through the 0.4 s minimum gap
(the no-skip bins 3-25 sit at 0.79-0.90 of the far field) are visible again, and because the mask removes exposure as well as bursts.

**What the mask removes.** 13.99 exposure minutes of 161.36 (8.7%) at min 1 and 7.98 of 96.05 (8.3%) at min 2. A real punch whose
span lies wholly in the removed frames is invisible: 104 of the 2,407 own-labelled punches at min 1 (4.3%) and 72 of 1,741 at min 2
(4.1%) are now "outside the exposure" (16 and 3 before), which is what raises "own labels per exposure minute". The completeness
question is asked only of the frames that are left; it says nothing about the removed 8-9%.

**Brendan's four GH069681 rows (min 2), mask run.** 1287 / track 64: **gone** (skip at that frame; the frame is out of the exposure).
7018 / track 1203: **gone** (skip 1 frame away; out of the exposure). 1965 / track 253: **still an unexplained burst** (38 frames
from a skip). 13293 / track 2268: **still an unexplained burst** (14 frames from a skip). As expected.

**The sample, mask run** (`results/olympic_completeness_credited_min2_sample_skipcheck.csv`, overwritten with the mask run; the
original sample CSV is untouched, its hash unchanged): **29 true, 11 false** (the cut run: 27 and 13). All 29 true rows are 5 or more
frames from a skip and in the exposure. Ten of the 11 false rows are inside the removed frames of a skip, at 0, 0, 1, 1, 1, 1, 1, 2, 4
and 4 frames, so their frame is out of the exposure and no burst can exist there. The eleventh (GH199681 frame 9937, 5 frames from a
skip, in the exposure) lost its burst for a different, legitimate reason: the skip at 9913 used to put a giant peak at 9912 that
suppressed everything within the 0.4 s minimum gap, and with it hidden a real 11.66 peak appears at 9922, 15 frames before 9937, and
now suppresses the weaker 6.28 one. Of the five rows that dropped out of the cut run without being near a skip, four are back (true).

**Tests.** `test_frame_skips.py` 189 (155 in Iteration 1), the other nine suites unchanged: `test_completeness_credited.py` 95,
`test_linking.py` 266, `test_pipeline.py` 260, `test_ingest.py` 124, `test_tracks.py` 79, `test_labels.py` 50, `test_hands.py` 38,
`test_env.py` 36, `test_olympic.py` 33. New: the reach follows `smooth_s` and the frame rate (0.1 s/50 fps -> 3; 0.2 s -> 6; 25 fps -> 2;
100 fps -> 6; a grid of rates and windows equals `smooth_window // 2 + 1`); `skip_mask` frames for a plain skip, both ends of the track, a
bridged hole, one hand's hole; a one-frame jump at a skip gives no burst and 391 of 400 exposure frames; a real punch peaking 5 frames from
a skip is found with the same frame and peak speed, one at 4, 3 or 1 frame is invisible, out of the exposure and does not come back
beside the removed range, on both sides; `_scan_segments` with `smooth_s` 0.1 / 0.2 / 0.02 removes 9 / 15 / 5 frames and the 0.2 s
signal has a lower peak than the 0.1 s one; a punch 12 frames after a jump is suppressed by the jump's peak without skips and found with
them; `skips=None`, an empty list and a skip past the track equal the old output on every scene.

**Mutation check (scratch copy; `test_frame_skips.py` must fail)**
| # | Mutation | Result |
|---|---|---|
| A | mask applied to the signal but not to the exposure | caught, 12 fail |
| B | `skip_mask_half` fixed at 0 | caught, 10 |
| C | mask one-sided (frames from f on only) | caught, 24 |
| D | mask one-sided (frames up to f only) | caught, 23 |
| E | skips ignored in `_scan_segments` | caught, 17 |
| F | `skip_mask_half` fixed at 3 (ignores `smooth_s` and fps) | caught, 4 |
| G | mask ignores a bridged hole | caught, 3 |
| H | mask ignores one hand's hole | caught, 1 |
| I | `smooth_s` not passed on to `wrist_signal` | caught, 1 (it passed the first 178 checks; the `smooth_s` signal-peak check was added) |
| J | signal not zeroed, exposure only | **missed** by the first 187 checks, caught by 1 after the punch-12-frames-after-a-jump check was added |
| K | the peak test's extra frame dropped (exposure = signal mask) | caught, 10 |
| L | peaks beside the mask kept (no eligibility filter) | caught by the in-code assertion: RuntimeError "a burst ... outside the exposure" |

An earlier version of C (changing only the first line of the mask) survived because it is an equivalent mutant: the bridged-hole line
`min(lo, p+1-half)` re-derives the left reach whenever f lies inside the track. The mutation was moved to the final slice, where it bites.
The Iteration-1 mutations on code this iteration did not change (M2, M3, M4a, M4b, M5, M7-M12) were re-run against the 189 checks: all caught (1, 6, 11, 5, 3, 3, 1, 5, 26, 6, 1 failures).

**Residual risks.** (1) The mask removes 8-9% of the exposure, and real punches inside it are invisible (about 4% of the labelled ones);
the completeness question is not asked there. (2) The reach is derived for a single-frame step; a real drop of several frames at once
is still one step in the position, but a pose estimator that is unsettled for a few frames after the jump (a tracker re-acquiring) is
not covered, and would show as a rate excess in the 3-10 bins; the final design shows none (1.14x, 1.04x). (3) The extra exposure frame
is justified by measurement on this data (the edge excess at 3 -> 4 -> 5 frames); the same rule on another detector would need the same
check. (4) A false skip (a real, sudden body movement that the image-change rule flags) now hides a real punch instead of cutting its
segment; the rule's 3x ratio is one observation's, so the number of real punches lost to false skips is unmeasured. (5) Peaks
suppressed by the 0.4 s minimum gap in the no-skip run reappear (as for GH199681 9937 above), so some "dropped" rows are
reshuffles, not skip artefacts. (6) The Iteration-1 residual risks 2-7 stand.


### Iteration 3 -- Astra 6 review of `feat/frame-skips` (2026-10-09; verdict FIX FIRST: 3 blocking, 3 smaller)

| # | Finding | Fix | Evidence |
|---|---|---|---|
| B1 blocking | `detect()` applied the 0.4 s minimum gap BEFORE the eligibility filter: an ineligible peak just outside the zeroed zone (a flank at 204 for a skip at 200) could suppress a real, eligible one within 20 frames and then be discarded itself, losing a burst while its exposure stayed (understating the rate). The burst-in-exposure assertion cannot see a missing burst | `punchfind.detect(..., eligible=None)`: ineligible frames keep their true signal as neighbours in the peak test but are not candidates; only the eligible local maxima compete in the minimum-gap suppression (the same scipy suppression, run on their heights). `_scan_segments` passes `~gone` and no longer filters afterwards. `eligible=None` is the original line, untouched | Regression: skip at 200, a strong punch hidden under the mask whose return lobe peaks at the ineligible 204, a weaker real punch at 212: without skips bursts at 62 and 204 only (212 suppressed), with skips 62 and 215 (the weak punch's burst), in the exposure. The all-eligible path equals the plain call on the scene and on 200 random signals with plateaus and ties; marking only 204 ineligible removes the burst there and nothing else |
| B2 blocking | A framechange file bound to the right video but measured with pixel step 8 passed (only ratio and window were compared) and the report said step 4 | `load_skips` requires ratio, half window AND pixel step to equal this code's, naming all three in the refusal | step 8 and half window 5 refused (exit 2, nothing written), beside the existing ratio 2.0 |
| B3 blocking | A skipcheck sample row naming frame 999999 of a processed 1,500-frame video was reported as an ordinary "false" | `skipcheck_rows` takes each processed video's length and requires `0 <= frame < length`, and that the video was processed; the run stops before any output | frames 400, 999999 and 10^12 of a 400-frame video refused; 0 and 399 accepted; the CLI with a row at 999999 stops with no output file |
| N1 | `Skips.save` used one shared `<name>.part`: concurrent writers could truncate each other's temporary file or fail at the rename; the "harmless" claim in Iteration 1 was unjustified | a `NamedTemporaryFile` of its own in the destination directory, then `os.replace`, retried a few times on the `PermissionError` Windows raises when two writers replace one destination in the same instant | eight threads saving one path 15 times each: no failure, the file is one writer's complete content, nothing left behind. The shared name fails this test (a PermissionError). Correction: in the Iteration-1 run two videos were measured by two processes at once and neither failed, but that was luck, not a property |
| N2 | The distance-bin computation behind the gate table was not in the repo | `completeness.distance_bins` / `bin_rates` / `pool_bins`; `--skips` and `--skips-measure` write `skip_bins` per video and pooled into the json and print the per-bin line with the gate verdict | exact bin assignment and exposure counts on a planted video, nearest of two skips, the gate at exactly +-25% and each way of failing; both CLI modes end to end. `--skips-measure` reproduces the "no skips" columns of every earlier table to the frame (checked: same counts as the script that made them) |
| N3 | The pooling test used identical rows and could not tell pooling from averaging per-video shares | two UNEQUAL videos | pooled 5 of 33 (15.2%) and base rate 30 of 1,650 (1.8%); the mean of the per-video shares would be 38.3% and 4.0% |

**`skips=None` is bit-for-bit unchanged: how it was verified.** (1) `detect` with `eligible=None` is the original `find_peaks(..., height,
distance)` line; the new branch is only taken when an eligibility array is passed, and `_scan_segments` passes one only when skips
exist. (2) Every pre-existing check passes unchanged (counts below). (3) On the real data, `olympic_completeness.py --credited` and
`--credited --min-punches 2` WITHOUT skips were re-run to scratch paths and compared with the existing
`olympic_completeness_credited.json` / `_min2.json`: byte-identical except the one line that records the output file name; the min-1
sample csv is byte-identical to the existing one. (4) `--skips-measure` (bursts and exposure from the unmasked path) reproduces the
earlier "no skips" bin counts exactly.

**Real data after the fixes (`--credited`, mask run).** B1 gives back bursts the old order had lost, so more bursts, a slightly
higher rate and a mildly higher near-skip rate than Iteration 2 reported:
| Quantity | min 1, Iteration 2 | min 1, Iteration 3 | min 2, Iteration 2 | min 2, Iteration 3 |
|---|---|---|---|---|
| bursts | 8,969 | 9,157 | 5,589 | 5,700 |
| unexplained | 6,366 | 6,502 | 3,803 | 3,881 |
| explained | 2,603 | 2,655 | 1,786 | 1,819 |
| exposure minutes | 147.37 | 147.37 | 88.07 | 88.07 |
| **unexplained per exposure minute** | **43.20** | **44.12** | **43.18** | **44.07** |
| (no skips, b912027) | 45.35 | 45.35 | 45.21 | 45.21 |
| bursts per exposure minute | 60.86 | 62.14 | 63.46 | 64.72 |
| `possible_duplicates` | 123 | 127 | 52 | 54 |

The unexplained rate now falls 2.7% (min 1) and 2.5% (min 2) from the no-skip rate. The exposure is unchanged (B1 changes which peaks
survive, not which frames count).

**The bin tables, re-run from the repo**

Unexplained bursts per 1,000 exposure frames by distance to the nearest skip, `--credited` min 1 (count / exposure frames; x = ratio to the >25 rate)

| Distance (frames) | no skips (`--skips-measure`) | mask, Iteration 2 | mask, Iteration 3 (B1 fixed) |
|---|---|---|---|
| 0-2 | 48.3 (1113/23,032) x3.36 | empty (masked) | empty (masked) |
| 3-5 | 12.9 (358/27,783) x0.90 | 16.4 (149/9,091) x1.14 | 16.6 (151/9,091) x1.16 |
| 6-10 | 11.4 (519/45,600) x0.79 | 14.9 (674/45,372) x1.04 | 16.2 (736/45,372) x1.13 |
| 11-25 | 12.2 (1352/110,943) x0.85 | 14.2 (1578/110,927) x0.99 | 14.9 (1657/110,927) x1.04 |
| >25 | 14.4 (3976/276,726) x1.00 | 14.3 (3965/276,726) x1.00 | 14.3 (3958/276,726) x1.00 |

Explained bursts per 1,000 exposure frames by distance to the nearest skip, `--credited` min 1 (count / exposure frames; x = ratio to the >25 rate)

| Distance (frames) | no skips (`--skips-measure`) | mask, Iteration 2 | mask, Iteration 3 (B1 fixed) |
|---|---|---|---|
| 0-2 | 9.1 (209/23,032) x1.51 | empty (masked) | empty (masked) |
| 3-5 | 4.0 (112/27,783) x0.67 | 4.5 (41/9,091) x0.76 | 4.5 (41/9,091) x0.76 |
| 6-10 | 4.4 (199/45,600) x0.73 | 6.0 (273/45,372) x1.01 | 6.7 (306/45,372) x1.13 |
| 11-25 | 5.2 (578/110,943) x0.87 | 5.8 (638/110,927) x0.96 | 5.9 (657/110,927) x0.99 |
| >25 | 6.0 (1661/276,726) x1.00 | 6.0 (1651/276,726) x1.00 | 6.0 (1651/276,726) x1.00 |

Unexplained bursts per 1,000 exposure frames by distance to the nearest skip, `--credited` min 2 (count / exposure frames; x = ratio to the >25 rate)

| Distance (frames) | no skips (`--skips-measure`) | mask, Iteration 2 | mask, Iteration 3 (B1 fixed) |
|---|---|---|---|
| 0-2 | 47.4 (624/13,163) x3.32 | empty (masked) | empty (masked) |
| 3-5 | 13.7 (216/15,815) x0.96 | 16.7 (86/5,163) x1.17 | 16.9 (87/5,163) x1.19 |
| 6-10 | 11.2 (291/25,936) x0.79 | 14.8 (382/25,816) x1.04 | 16.3 (422/25,816) x1.15 |
| 11-25 | 12.4 (790/63,683) x0.87 | 14.5 (923/63,679) x1.02 | 15.2 (966/63,679) x1.07 |
| >25 | 14.3 (2421/169,557) x1.00 | 14.2 (2412/169,557) x1.00 | 14.2 (2406/169,557) x1.00 |

Explained bursts per 1,000 exposure frames by distance to the nearest skip, `--credited` min 2 (count / exposure frames; x = ratio to the >25 rate)

| Distance (frames) | no skips (`--skips-measure`) | mask, Iteration 2 | mask, Iteration 3 (B1 fixed) |
|---|---|---|---|
| 0-2 | 10.4 (137/13,163) x1.49 | empty (masked) | empty (masked) |
| 3-5 | 4.5 (71/15,815) x0.64 | 4.6 (24/5,163) x0.67 | 4.6 (24/5,163) x0.67 |
| 6-10 | 4.9 (127/25,936) x0.70 | 6.9 (177/25,816) x0.99 | 7.5 (194/25,816) x1.08 |
| 11-25 | 5.8 (367/63,683) x0.83 | 6.4 (408/63,679) x0.92 | 6.7 (425/63,679) x0.96 |
| >25 | 7.0 (1183/169,557) x1.00 | 6.9 (1177/169,557) x1.00 | 6.9 (1176/169,557) x1.00 |


**The gate, re-run from the repo (`skip_bins` in the json; the gate chose the design, so a PASS is a check on it, not independent
evidence).** Unexplained, ratio to the >25 rate in 3-5 / 6-10 / 11-25: min 1 **1.16 / 1.13 / 1.04 PASS** (Iteration 2: 1.14 / 1.04 /
0.99), min 2 **1.19 / 1.15 / 1.07 PASS** (1.17 / 1.04 / 1.02); 0-2 empty. All within +-25%, but the 6-10 bin moved up by 9-11 points: part
of what Iteration 2 reported as agreement with the far field was the B1 loss. The explained bursts at 6-10 are now 1.13x (min 1) and
1.08x (min 2) as well. The 3-5 bin holds one offset (5 frames) and about 150 (min 1) and 87 (min 2) bursts.

**Brendan's four GH069681 rows (min 2).** Unchanged: 1287 / 64 gone, 7018 / 1203 gone, 1965 / 253 still an unexplained burst, 13293 /
2268 still an unexplained burst. **Skipcheck:** 29 true, 11 false, row for row the same as in Iteration 2 (the ten false rows inside
the removed frames at 0-4 frames, and GH199681 9937 which is the minimum-gap reshuffle explained there).

**Tests.** `test_frame_skips.py` 222 (189 before); `test_completeness_credited.py` 95, `test_linking.py` 266, `test_pipeline.py` 260,
`test_ingest.py` 124, `test_tracks.py` 79, `test_labels.py` 50, `test_hands.py` 38, `test_env.py` 36, `test_olympic.py` 33: all
unchanged and passing.

**Mutation check (scratch copy; `test_frame_skips.py` must fail)**
| # | Mutation | Result |
|---|---|---|
| B1 | filter after suppression restored (ineligible peaks suppress, then are discarded) | caught, 1 (the 212 regression) |
| B1b | `detect` does not pre-filter its candidates | caught: the in-code exposure assertion raises (a burst outside the exposure) |
| B2 | the decimation step (and the other constants) not compared | caught, 6 |
| B2b | only ratio and half window compared | caught, 6 (the step 8 file passes) |
| B3 | sample frame not checked against the video length | caught, 4 |
| B3b | a sample row of an unprocessed video accepted | caught, 2 |
| N1 | shared `.part` name restored | caught, 1 (a PermissionError among the eight writers) |
| N2a | bin edges exclusive | caught, 5 |
| N2b | distance to the previous skip only | caught, 5 |
| N2c | gate ignores the 0-2 bin | caught, 1 |
| N3 | pooled shares replaced by the mean of per-video shares | caught, 2 |

**Residual risks added.** (1) The distance-bin gate and the design were fixed on the same data (declared before); a held-out video
set would be the independent test. (2) `Skips.save`'s retry covers the Windows rename race, not two writers with different content
deciding which one wins: the last replace wins, and the file is always one writer's complete content. (3) A plateau that straddles
the edge of the removed range is reported at its midpoint; if that frame is ineligible the plateau yields no burst.

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

## `feat/output-report` -- 2026-10-09

The report a boxer reads: `<clip>.report.xlsx` (Punches, Rounds, Reliability, About) and
`<clip>.summary.txt`, from the classified punches. Stacked on `feat/classify-detected`.
Acceptance criteria: (1) the summary is fixed templates and holds exactly the numbers they
inserted; (2) every reliability statement comes from a measured number, linked to the
model that named the punches by hashes, and the summary says when nothing was measured on
your footage; (3) rounding never flips a claim; (4) inputs are validated and the workbook
holds no formulas; (5) all suites green in a clean clone.

Threat sketch. Assets: the honesty of what the user reads. Entry points: classified CSV and
its sidecar, manifest, reference run, finder rule, score JSON, `--rounds`. Trust boundary:
files on disk between tools. Top threats: a number-backed but misleading sentence; a
reliability table borrowed from a different model than the one that named the punches.

Real-world fact checked before building: the reported run's held-out confusion matrix
(results/stgcnpp.json) gives cross +28%, lead uppercut -41%, hand (type-implied) 98.3% --
the figures the project plan states.

### Iteration 1
- **Test:** 65 checks; 8 guards reverted, all caught. The first run caught my own arithmetic
  in a hand-built matrix (a row summing to 50, not 100): the code was right.
- **Review: FIX FIRST.** B1: the finder's 74% recall was stated but not its V6 count ratio
  (648 flagged for 406 labelled, 1.6x), so the totals read as an under-count. B2: "71% from
  your lead hand" came from the finder's wrist while "right 98%" was the classifier's
  type-implied hand -- a figure attached to a reading it never measured. B3: rounding flipped
  claims ("1 punch, 0 a minute"; "0% rear hooks (1)"; "dropped by 8%, from 10 to 10").
  B4: the per-class-F1 match linked two JSON files, never the CSV to the model, and could
  not fail in practice. Also: hardcoded "V2/V5/V9"; "too little of you was in view" for
  every unclassified reason; "between rounds" for punches before the first; floats like
  `1_0` accepted; fractional confusion cells truncated; the provenance check ran on one
  summary only.
- **Security:** bounded JSON reads, NaN refused, formula guard at write time; folded in above.

### Iteration 2
- **Fix:** finder sentence states recall and the count ratio, and that the error on your
  footage is unknown. Lead share by punch TYPE over named punches; type/wrist disagreements
  counted; the 98% tied to the type-implied hand. Rates under 10 keep a decimal; 0%/100%
  only when exact; shares by largest remainder; a rate change only beyond 2 SE of a Poisson
  count, worded from the printed rates. `classify_punches.py` writes
  `<clip>.classified.json` (CSV sha256, classifier sha256, stance, rule, pose frames); the
  report checks CSV -> sidecar -> manifest (classify's own schema) -> per-class F1
  recomputed from the matrix, and an own-footage score must name the same classifier.
  Inputs: decimal regex, integer cells, known notes only, one stance per file.
- **Test:** 110 checks; the provenance check runs on every summary variant and caught a
  real slip on its first run ("benchmark video V6" put an unrecorded digit in the text).
  9 new guards reverted, all caught.

### Iteration 3 -- the extent's columns, caveats, and real reports (2026-10-11 to 10-12)
- **Fix / build:** the report reads `punch_start_frame` / `punch_end_frame` and shows each
  punch's length; a named punch may carry only the two caveats `classify_punches.py` writes
  (wrist dropout beside the peak; longer than the classifier can be shown), an unnamed one only
  a known reason; the summary counts punches whose start and end are unsure. A contract test
  ties the report's caveat list and CSV header to the writer's.
- **Test:** 122 checks. Same numbers in a clean clone.
- **Real reports, after `classify`'s confidence-rule fix:** IMG_4395, 42 punches, 57 a minute:
  31% jabs, 29% crosses, 21% lead hooks, 14% rear uppercuts, 5% lead uppercuts; type and wrist
  disagree on the hand for 12. IMG_4406, 41 punches, 63 a minute: 39% jabs, 20% crosses, 17%
  rear uppercuts, 12% lead hooks, 12% lead uppercuts; disagree for 7. Before the fix the same
  reports led with 63% and 41% lead uppercuts: the report was faithful to a broken input, and
  nothing in it said so beyond the hand-disagreement count (19 and 21 of 41).
- **Residual, open:** the summary does not yet lead with a warning when type-and-wrist hand
  agreement is near chance; it would have flagged the broken input on its own.

### Iteration 4 -- the per-clip cut in the report (2026-10-13)
- Per-punch caveat notes are gone with the per-clip window (a punch's window no longer depends
  on its own peak width): a named punch carries no note, and the "start and end are unsure"
  sentence is removed. The About sheet states how the clip was cut, from the sidecar, validated
  field by field: the clip's median punch length and how many punches set it, or that there
  were too few punches and the training clips' median was used.
- **Test:** 130 checks. Same numbers in a clean clone.
- **Real reports:** IMG_4395 read over 0.41 s (39 punches; 3 left out), 42 punches: 38% jabs,
  24% crosses, 14% lead hooks, 14% rear uppercuts, 10% lead uppercuts. IMG_4406 read over
  0.39 s (41 punches), 41 punches: 41% jabs, 17% crosses, 17% rear uppercuts, 15% lead hooks,
  10% lead uppercuts. Unmeasured against labels.

### Iteration 5 -- the summary leads with a warning (2026-10-19)
Closes iteration 3's open residual. Acceptance: (1) the summary's first sentence is a fixed-template
warning whenever type-hand vs wrist agreement is not above the chance agreement for those proportions
(`classify.hand_agreement`) by a one-sided exact binomial test at 0.05, and that test is the only rule;
(2) every number in it goes through `_Writer`, and the provenance check passes on every variant; (3) the
Reliability sheet carries the agreement, chance and p; (4) the old broken inputs warn, and today's two
clips are reported as they fall; (5) suites green in a clean clone.
- **Build:** `stats.binomial_greater_p` (exact tail, summed in log space so n up to the row cap does not
  overflow); `report.hand_check` in the facts; the warning template; a "this clip's check" block under
  the per-type rows of the Reliability sheet; the rule stated on the About sheet. With no named punch
  there is no punch type to warn about, so no warning.
- **Test:** 186 checks. Hand-worked tails (1/16, 1/32, 56/1024, 11/1024, 0.104, 0.51), an exact
  Fraction oracle sharing no code, 3,000 random cases against scipy (worst 3e-13, scratch); 8 of 10
  at chance 1/2 warns (p 0.055), 9 of 10 does not (0.011); edges: no named punch, all one side agreeing
  (100% vs 100%, p 1) and disagreeing (0 of 9, chance 0), one constant reading, one punch, southpaw.
  13 mutants (a tuned threshold, a tuned margin, chance fixed at 1/2, the warning last, P(X > k),
  two-sided, numbers not recorded...), all caught.
- **Review: FIX FIRST.** B1 the About sheet said the warning fires when agreement is "no more often
  than chance"; IMG_4395 agrees 62% against 49% and warns, so About contradicted the clip in front of
  it. Fixed ("not significantly more often") with a test. Also taken: p under 0.1 printed to three
  decimals (0.055 had read "p = 0.05, not under 0.05"); the sheet keeps three significant figures, so
  6.83e-05 is not shown as 0. 188 checks; 14 mutants, all caught.
- **Security:** no new input; the binomial validates k, n and p (negative tests); its loop is bounded
  by MAX_ROWS; no cell is a formula. No findings.
- **Real-world facts.** Old 0.3 confidence rule, as recorded: IMG_4395 22 of 41 against 0.482, p 0.293;
  IMG_4406 20 of 41 against 0.509, p 0.666. **Both warn.** (0.482 is not reachable with 22 of 41
  agreeing, nearest 0.483; it is iteration 3's figure for 23 of 42, p 0.243, which also warns.) Today
  (`results/punches`, 2026-10-13): IMG_4395 26 of 42 (61.9%) against 49.4%, p **0.071: still warns**;
  IMG_4406 33 of 41 (80.5%) against 50.4%, p 0.00007: no warning. Both reports regenerated; after the
  warning, each summary is identical to the one before.
- **Residual, accepted:** chance is plugged in from the same clip's proportions. Conditioning on both
  margins, the exact test is one-sided Fisher; the binomial is mostly the more cautious of the two
  but can pass a balanced table Fisher would not (n = 42, (16, 9, 6, 11): binomial 0.0500, Fisher
  0.065). Neither real clip changes (Fisher 0.088 and 0.00005). When one reading is constant the check
  cannot tell good from bad and always warns; over hundreds of punches a modest excess over chance
  passes. Both follow from the test being the only rule.

### Iteration 6 -- a floor, both tests, and when the check cannot confirm anything (2026-10-20, vault ADR-0015)
Brendan, on iteration 5's caveats: add an agreement floor ("start 70%, want to work up to 80"); show both tests
("use both tests to output both results and see"); for a clip concentrated on one hand, say so ("there appears
to be an extreme concentration ... if this is not expected, checking the footage is recommended") instead of
"may be wrong"; say the check sees the hand only. Acceptance: (1) agreement under `HAND_FLOOR` (0.70) warns
whatever the tests say; (2) the binomial and Fisher one-sided exact p are both stated, and the clip passes only
if both are under 0.05; (3) when full agreement on the clip's type split could not pass both tests, the summary
leads with a caution (concentrated, or too few punches); (4) the passing summary states both p-values and that
a jab read as a lead hook passes the check; (5) suites green in a clean clone.
- **Build:** `stats.fisher_greater_p` (scipy's `fisher_exact`, one-sided); `hand_check` verdicts in order:
  under the floor -> warn; both tests pass -> pass; full agreement could pass -> warn; otherwise concentrated,
  or too_few (under 7 punches at 0.05). The Reliability row carries both p, the floor and a one-cell result.
- **Test:** 222 checks. Fisher worked by hand (1/10, 26/252, 4/286, 4/56, 3/66) and against an integer
  hypergeometric oracle; the two tests disagreeing in both directions (7 of 8: binomial 9/256 passes, Fisher
  4/56 fails; 11 of 12: binomial 0.054 fails, Fisher 3/66 passes); 68 of 100 failing on the floor alone; IMG_4406
  at an 80% floor passing by one punch. 26 mutants, 2 survived: the float floor (`k < HAND_FLOOR * n`) and a
  binomial-only "could pass". The second got a 5-punch fixture; the first I judged equivalent.
- **Review: FIX FIRST.** B1 scipy returns Fisher's exact 1/20 as 0.04999999999999999: on (12, 2, 0, 2) the
  summary said "p = 0.050 (Fisher); only the Fisher test is under 0.05". B2 "the commonest jabs (6)" picked a
  tie by alphabet. Also: the float floor is not equivalent in general (0.68 at 51 of 75); the "70 of 100"
  test could not see floats; stance went unmentioned where it matters; "That is under..." had no clear subject.
- **Fix:** every comparison with alpha goes through `_under_alpha` (a p within a billionth of alpha is not
  under it); a p under alpha that would print as 0.050 prints "just under 0.05"; every tied type is named
  ("most often jabs and lead hooks (6 each)"); agreement no higher than chance adds that a wrong stance swaps
  lead and rear; "The 62% agreement is under the 70% ..."; floor patches in tests restored in `finally`. 234
  checks; 30 mutants, all caught.
- **Security:** no new input; Fisher validates its cells (negative tests); no cell is a formula. No findings.
- **Real-world facts.** IMG_4395: 26 of 42 (62%) against 49.4%; binomial 0.071, Fisher 0.088; under 70%:
  warns on all three. IMG_4406: 33 of 41 (80.5%) against 50.4%; binomial 0.00007, Fisher 0.00005; passes, and
  still passes at 80% by one punch (32 of 41 would warn). Old 0.3-rule inputs: all under 70%, all warn. Both
  reports regenerated.
- **Residual, accepted:** the floor is chosen, not measured. The check is blind within a hand (jab vs lead
  hook); a within-hand cue needs its own test against labels.

### Iteration 7 -- the four Codex findings (2026-10-21; feat/output-report's iteration 5, merged here)
Codex (10-20) found four ways the report could state a reliability that does not describe
the punches in front of the user; each was confirmed by reading before fixing.
Acceptance criteria: (1) a CSV classified at another fps is refused, at the sidecar AND at
every row's time, by a CLI test that changes only the fps; (2) an own-footage score is
refused unless it names the classified CSV's sha256, which `classify_punches.py --labels`
now writes; (3) the finder's held-out score is refused unless the sidecar's rule (variant,
threshold, min_gap_s) is `--rule`'s; (4) a reference matrix that is not, cell for cell, the
matrix the classifier's manifest pins is refused (transpose included), and
`train_classifier.py` pins it; (5) all suites green in the working tree and a clean clone.
Threat sketch unchanged (files on disk between tools; reliability borrowed from another run).

- **Fix:** (1) `report.check_pose` compares the sidecar's `pose_frames` and `fps` with the
  pose; `read_classified(..., fps=)` (required, keyword-only) refuses a frame past the clip
  and a time more than 2 ms from frame / fps, as `classify_punches.read_punches` does. (2)
  The score records `csv_sha256`; `own_reliability` takes the sidecar's (already verified)
  hash and refuses a mismatch, or a score with no hash. (3) `finder_reliability(rule, used)`
  compares the three settings exactly. (4) The manifest's `held_out.confusion` is the saved
  model's matrix (`run_stgcnpp` saves after scoring); the report requires equality and says
  when the reference is the transpose. Per-class F1 is symmetric under transpose, so the old
  F1 check could not see it.
- **Test:** report 130 -> 185, classify 202 -> 209; other suites unchanged (env 36, hands 38,
  ingest 109, labels 50, pipeline 260). New refusal tests name the phrase the error must
  carry, so none passes on another refusal. A real fact is pinned: the reported held-out
  matrix (V2/V5/V9, 976 punches) gives cross +28% and lead uppercut -41%; its transpose has
  the same F1 and would have said lead uppercuts run HIGH by 69% -- refused. The manifest's
  matrix is checked against one counted by hand from the saved model's own predictions.
  20 guards reverted one at a time in a scratch copy: all 20 caught.
- **Review (self):** one bug in my own fix -- `math.isfinite` on a 400-digit JSON integer
  raises OverflowError, not ValueError (sidecar fps, rule threshold), and the same class in
  `_count` (int64 overflow on a confusion cell, pre-existing). Fixed: exact int/float
  comparison, bounded counts (<= 10^9); tests added. Message for a pre-10-21 score said
  "classified again since" when the truth is "never recorded"; split.
- **Review (Astra / Codex, read-only, effort medium, on `d3975ef`): SHIP.** All four findings
  CLOSED, each cited to file:line; the stacked-branch list confirmed for #17, #18 and #19
  (#21/#24/#25 were outside its brief; `git diff --stat` shows they do not touch these files).
  One nit, confirmed by reading: `read_classified(..., fps=-(10**400))` overflowed, because the
  integer guard bounded only the top. Fixed (`0 < fps <= 10_000` for integers); test and a
  20th reverted guard added.
- **Security (changed files):** no queries, shells or new paths; every new input goes through
  the bounded JSON reader; errors truncate file-supplied numbers to 20 characters. The
  overflow above was the one finding (low: it failed closed, with a traceback).
- **Real data, read-only:** on IMG_4406 and IMG_4395 the fps, row-time (41 and 42 rows) and
  rule checks accept today's outputs. Two refusals, both intended: today's manifest has no
  matrix (re-run `train_classifier.py`), and today's score files have no CSV hash (re-run
  `classify_punches.py --labels`). Until both are re-run the real reports refuse.
- **Stacked branches:** #17 (edits report.py / report_punches.py / test_report.py: conflicts,
  fixtures need the matrix, rule and fps), #18 (`render_punches` calls `read_classified`
  positionally with `data`; needs `fps=`), #19 (`analyse_clip.preflight` calls
  `finder_reliability(a.rule)` and needs the rule; its test manifest lacks the matrix;
  `check_classified` can use `check_pose`). #21 and #24 (and #25) do not touch these files.
  #22 also edits the `--labels` block of `classify_punches.py`: merge by hand.

### Iteration 8 -- feat/output-report's four fixes merged in (2026-10-21)
Acceptance: (1) the merge resolves with both branches' work kept; (2) every suite green, the
report count equal to this branch's 234 plus the 55 checks the fixes added; (3) the hand check's
published real figures unchanged when the real CSVs go through the stricter reader.
- **Merge:** one conflict, this log (both branches appended an iteration 5); kept both, the incoming
  entry renumbered 7. `report.py`, `report_punches.py` and `test_report.py` merged without conflict.
  The hand check adds no input of its own: it reads rows `read_classified` has already validated
  (now fps-checked), so no code change was needed.
- **Test:** report 289 = 234 + 55; classify 209, env 36, hands 38, ingest 109, labels 50, pipeline 260.
- **Review:** the warning's code is untouched by the merge (`git diff origin/feat/output-report`
  shows only the hand check and its templates). No findings.
- **Security:** no new input. No findings.
- **Real-world facts, read-only, through the merged reader:** IMG_4395 26 of 42 (62%) against 49.4%,
  binomial 0.071, Fisher 0.088, warns; IMG_4406 33 of 41 (80.5%) against 50.4%, binomial 0.00007,
  Fisher 0.00005, passes -- iteration 6's numbers exactly. The full reports cannot be regenerated
  until `train_classifier.py` and `classify_punches.py --labels` are re-run (iteration 7).
- **Review (Astra / Codex, read-only, effort medium, one review of the #17/#18/#19 merges): SHIP all.** Merge OK on each branch, no adaptation defects; it checked that the shared checks are identical to feat/output-report's on all three (confirmed here: no changed line). Its sandbox could not run Python, so its verdict is from reading plus the results above; fresh clones of 2e2059f, f2412e8 and a2f90be reproduce every count.

## `fix/pose-load-cap` -- 2026-10-20

`PoseSequence.load`, which every tool that reads a `.pose.npz` shares, called `np.load`, and
`np.load` allocates whatever shape a `.npy` header claims. Found in the 2026-10-19 review of
`render_punches.py` (PR #18), which converts the MemoryError at its own load site only; the
cap belongs in the loader. Stacked on `feat/output-report`.

Acceptance criteria: (1) a ~2 KB file whose headers claim 10^12 frames is a ValueError
naming the cap, with a traced peak under 5 MB; (2) a missing key, mismatched shapes and an
object dtype are each a ValueError naming that check, raised before any array is allocated;
(3) more than `MAX_POSE_FRAMES` (2,000,000, ~9 h at 60 fps, `video.MAX_TIMING_PACKETS`'s
bound) is refused and exactly the cap is not; (4) the seven real pose files load identical
to `np.load`, and save/load round-trips exactly; (5) all suites green in a clean clone.

Threat sketch. Assets: the memory of the machine running the tools (a training run may share
it), and the integrity of what every downstream tool reads. Entry points: `--pose` of
`find_punches`, `classify_punches`, `report_punches` (and `render_punches` on PR #18).
Trust boundary: files on disk between tools, which move between machines
(`docs/MACHINE-SYNC.md`). Top threats: a small crafted file forcing a huge allocation; a
pickled member executing code (already off, kept off by construction).

Real-world fact checked before building: the bomb reproduced on the old loader. A 1.8 KB
file claiming 10^12 frames attempted a 124 TiB allocation (MemoryError); one claiming 3M
frames allocated 408 MB, then failed on missing data. All seven real files
(`results/ingest/`) share one layout: xy/conf `<f4`, detected `|b1`, fps `<f8`, width/height
`<i8`, short `<U` text; the longest is V4 at 35,363 frames. So strict dtype rules refuse
nothing real.

### Iteration 1
- **Build:** the loader opens the file once and reads it as a zip. Every member's header is
  read with `read_magic` and `read_array_header_1_0`/`_2_0` and checked before anything is
  allocated: all eight keys present, dtype kind (xy/conf float, detected bool, fps numeric,
  width/height integer, source/model unicode up to 4,096 characters), shapes agreeing with
  `detected`'s (an int frame count, not `(True,)`), and the frame cap. Only then is data
  read, in 16 MB chunks, into one buffer of exactly the validated size, from the same open
  file. `allow_pickle` is never involved: object dtypes are refused from the header.
- **Test:** 147 checks in `test_ingest.py` (38 new). 9 guards reverted one at a time, and all
  9 were caught. The truncation guard was caught as a hang: without it the read loop never
  ends on short data. My own first test had a wrong needle: the old pickled-xy fixture does
  hold `xy`, so the loader rightly names the other seven keys.
- **Security (own pass, changed files):** two more ways past the header check, both measured.
  (a) zipfile decompresses bz2 and lzma members with no output limit, so reading the 8-byte
  magic of a 2.1 KB bz2 member expanded 584 MB. (b) Format 2.0 gives a header a 4-byte length,
  and numpy reads the whole claimed length before checking its own 10,000-byte limit. A
  263 KB deflated member reached 584 MB this way, about 1000:1. `np.load` had both. Fixed:
  only stored and deflate are accepted (what `np.savez` and `np.savez_compressed` write), and
  the header parser reads through a 10,012-byte cap. All three vectors now peak at 0.1 MB or
  less. Error messages name the file's basename only.
- **Review (code-reviewer): FIX FIRST.** B1: a header that `literal_eval` rejects makes numpy
  re-tokenize it, and `tokenize.TokenError` (from an unterminated header, or 300 nested
  parentheses) escaped the ValueError contract. Non-blocking: a 1.3 KB file whose headers
  claim exactly the cap at float64 passes the cap and commits 544 MB before finding the data
  missing; one `refuses` needle was loose; `NotImplementedError` is already a
  `RuntimeError`. The reviewer also fuzzed 40,000 random byte mutations of a stored and a
  deflated real save, and nothing escaped other than ValueError/OSError.

### Iteration 2
- **Fix:** header parse errors from tokenize become ValueError. Before `np.empty`, the bytes
  a header promises are compared with what its zip entry could hold: the stored size, or
  1032x the compressed size for deflate (deflate's ceiling), never more than the declared
  uncompressed size, and the compressed size never taken as more than the file's real size.
  The 544 MB case is now refused at 0.1 MB. The needle is tightened; the docstring says a
  file that cannot be opened is OSError and anything inside it ValueError.
- **Test:** 159 checks (50 new). The two new guards were reverted and both caught; 11 of 11
  overall. The reviewer's 33 probe files all either load (the legitimate ones) or are
  refused as ValueError. Re-fuzzed with 2x20,000 mutations on the final source: 0 escapes.
  Seven real files identical to `np.load` (values and dtypes). Other suites unchanged:
  pipeline 260, classify 202, report 130, labels 50, env 36, hands 38, machine scripts 38.
- **Review (own re-read of the final diff):** no new findings.
- **Reproduce:** a clean clone at d435520 (no `results/`, no `data/`) gives the same numbers
  in every suite, and the ingest check list matches the working tree line for line.
- **Residual, accepted:** a legitimate file at the cap still costs up to ~410 MB (float32)
  or ~820 MB (float64) to load; that is the cap's stated price. A zero-frame pose loads,
  as before. When this and PR #18 both land, `render_punches`' own MemoryError conversion
  becomes redundant but harmless.

## `fix/corpus-load-cap` -- 2026-10-20

`data.py` read the BoxingVI corpus with `np.load` at four sites (`SKELETON_READERS['.npy']`,
`_load_presegmented`, `_load_v6`, `load_v6_track`). The corpus is third-party data and this
repo's project rules name the loader as the trust boundary. Follow-up from `fix/pose-load-cap`
(PR #24), on whose branch this is stacked, reusing its header-first approach.

Acceptance criteria: (1) a ~128-byte corpus `.npy` whose header claims 10^12 or 2M frames is
a ValueError naming the size check, with a traced peak under 5 MB, through every entry point;
(2) wrong dtype, wrong shape and object dtype are each a ValueError naming that check, raised
before any array is allocated, and an object array is never unpickled; (3) real-shaped arrays
(and all ten real files) load identical to `np.load`; (4) `pose_yolo` behaves as before
(`test_ingest` 159); (5) all suites green in a clean clone.

Threat sketch. Assets: the memory of the machine (training runs share it) and the integrity
of every clip the classifier and punch finder read. Entry points: `load_corpus`,
`load_v6_track` (`run_baseline`, `run_punch_finder`, `run_punch_extent`,
`eval_classify_v6`, `train_classifier`), `SKELETON_READERS`. Trust boundary: `data/`,
downloaded from a third party's Drive folder. Top threats: a small crafted file forcing a
huge allocation; a pickled array executing code (line 157 relied on numpy's
`allow_pickle=False` default rather than saying so).

Real-world facts checked before building: the bomb reproduced on the old loaders. A 128-byte
`V6.npy` claiming 2M frames allocated 816 MB, then raised an ordinary-looking ValueError
("cannot reshape array of size 0"); 10^12 frames was a MemoryError. New beyond PR #24: on a
plain file, a 112-byte `.npy` whose format-2.0 header claims a 4 GB header length made
`np.load` allocate 4.3 GB (Python's buffered `read(n)` reserves n bytes first), and 300
nested parentheses escape `np.load` as `tokenize.TokenError`. All ten real files share two
layouts, read from their headers: nine `(N, 25, 17, 2)` and V6 `(46497, 1, 17, 3)`, all
`<f8`, format 1.0. So the strict rules refuse nothing real.

### Iteration 1
- **Build:** `load_corpus_npy(path, layouts)` reads the header through the byte cap, then
  checks the dtype (plain float of 8 bytes or less; object, structured, subarray, integer,
  bool, complex and text refused), integer dims matching a named layout (`clips`
  `(N, 25, 17, 2 or 3)` or `v6` `(F, 1, 17, 3)`), and that the promised bytes fit in the file
  after the header (`fstat` on the open handle, exact Python-int product). Only then is one
  buffer of that size filled. The four call sites use it; their own post-load shape checks
  are now redundant and removed. The capped header reader and the chunked data reader move
  from `pose_yolo` to `npyread.py`, shared by both (`data` cannot import `pose_yolo`:
  `pose_yolo` imports `data`). `pose_yolo`'s messages are byte-identical.
- **Test:** `test_pipeline` 260 -> 311 (51 new): bombs through all four entry points, a
  one-byte-short file, the 4 GB header-length claim, an unparseable header, bad magic; five
  wrong dtypes plus structured; nine wrong shapes including V6's layout given to the clip
  loader and vice versa, and a `(True, ...)` dim; object arrays through all four entry points
  with a pickle probe that records if it is ever unpickled (the fixture is checked to fire
  under `np.load(allow_pickle=True)`); six real-shaped layouts (release `<f8`, float32
  3-channel, Fortran order, big-endian, zero clips, V6) identical to `np.load`; a peak no
  more than one array's size; trailing bytes ignored. Every bomb and order check asserts
  both the message and the traced peak, because the old loader also raised ValueError, just
  after allocating. Can the tests fail? Against the base commit's `data.py`, 37 of the 51
  fail (peaks of 680 MB, 816 MB and 4.3 GB). Six guards reverted one at a time (dtype,
  layout, int dims, size, header cap, TokenError conversion), and all six were caught.
  Other suites unchanged: ingest 159, classify 202, report 130, labels 50, env 36, hands 38,
  machine scripts 38; `run_baseline.py --synthetic` exits 0.
- **Real data:** all ten files load identical to `np.load` (values and dtype).
  `load_corpus(include_v6=True)` gives 5,442 clips with the same SHA-256 over every clip's
  source, label and keypoints as the base branch, and `load_v6_track` has the same digest.
- **Security (own pass, changed files):** no `allow_pickle` path exists; an allocation is
  bounded by the file's own size, taken by `fstat` on the handle that is read, so growth
  after the check is irrelevant and shrinkage is caught by the truncation guard. Errors name
  the basename, integer dims and the dtype string only.
- **Review (code-reviewer): FIX FIRST.** B1: numpy's header parser lets more than
  `TokenError` out of a malformed header, so the "anything inside it, ValueError" contract
  still broke: `'descr': ()` raised IndexError, a 4,000-term `1+1+...+1` shape raised
  RecursionError, and 9,000 minus signs raised MemoryError (a parser-stack overflow on a
  9 KB header, with nothing large allocated). `pose_yolo` shared the gap. B2: "allocates the
  array once" could not fail: on a 27 KB array its 0.5 MB slack passed `np.load(...)` copied
  twice. Non-blocking: dtype-check duplication between the two loaders; float16 accepted
  without saying so; big-endian returned as the file has it. The reviewer also fuzzed 20,000
  header-heavy byte mutations of three valid files: 19,561 ValueError, 438 loaded, 1
  UnicodeDecodeError (a ValueError), nothing else escaped, and no peak tied to a claimed size.

### Iteration 2
- **Fix:** `npyread.read_header` turns every non-ValueError from the capped header parse into
  ValueError("unparseable .npy header"). That is safe because the header is at most 10,012
  bytes, so these are malformed files and not the machine failing. B2, done properly, then
  failed on real behaviour: the reader peaked at 2x the array, because each chunk came back
  as a `bytes` object (up to 16 MB) and was then copied in. `read_data` now uses `readinto`
  straight into the array's buffer, so a plain file is read with no second copy. The
  docstring states that float16 is accepted and that byte order is left to the consumers,
  which all convert. Not taken: a shared dtype helper and moving the size bound into
  `read_data`. The two loaders bound sizes differently (zip ratio against file size) and
  check dtypes differently (per-key kinds, text length), so one helper would hide that.
- **Test:** `test_pipeline` 314 (54 new): the four malformed-header cases; the
  allocate-once check on a 6.8 MB array at 1.25x, with no slack (measured 6.8 MB now, 13.6 MB
  with the old reader). `test_ingest` 162 (3 new): the same three non-TokenError headers
  through `PoseSequence.load`. Can the tests fail? Against the base commit's `data.py`, 40 of
  the 54 fail. Eight guards were reverted one at a time (dtype, layout, int dims, size,
  header cap, error conversion narrowed back to TokenError, error conversion off, the old
  copying reader), and all eight were caught. Other suites unchanged: classify 202,
  report 130, labels 50, env 36, hands 38, machine scripts 38; `run_baseline.py --synthetic`
  exits 0.
- **Real data, again:** the ten corpus files are identical to `np.load`, and the 5,442
  clips and the V6 track give the same digests as the base branch. Because `readinto` also
  serves the pose path, the seven real pose files (`results/ingest/`) were rechecked: they
  are byte-identical to `np.load`. My first comparison said "DIFFERENT" for six of them,
  but that was NaN != NaN in undetected joints, and the base branch gives the same result.
- **Security (own re-pass):** the broadened `except` sits inside the header cap only;
  `read_data`'s truncation guard is unchanged in substance (`readinto` returning 0 or None).
- **Residual, accepted:** a genuine N-byte corpus file costs N bytes to load, as with
  `np.load`; there is no absolute cap because the release's largest file is 19 MB and a
  plain `.npy` cannot amplify. A FIFO planted in `Skeleton_data` would block a read on a
  POSIX machine, but planting one needs local write access to `data/`.
- **Reproduce:** a clean clone at afaf819 (no `data/`, no `results/`, none tracked) gives the
  same numbers in every suite (pipeline 314, ingest 162, classify 202, report 130, labels 50,
  env 36, hands 38, machine scripts 38; `run_baseline.py --synthetic` exits 0). The pipeline
  and ingest check lists match the working tree's line for line.

## `feat/pose-switch` -- 2026-10-12

How much the classifier loses reading YOLO11-pose instead of AlphaPose, on the held-out
punches themselves. Stacked on `feat/classify-detected`. Acceptance: (1) the saved classifier
reproduces 0.8145 / 0.7207 on the corpus clips before anything is scored; (2) each video's
pose is the source at its native rate and covers every span; (3) spans align with YOLO's
frame numbering, or the video is excluded and named; (4) A - Y with a paired interval;
(5) suites green in a clean clone.

Threat sketch. Assets: the honesty of one number the poster will carry. Entry points: pose
npz from the laptop, corpus skeletons and annotation sheets, classifier manifest. Trust
boundary: files synced between machines. Top threats: scoring skeletons against the wrong
frames (a silent, plausible-looking number); a different person in frame scored as the boxer.

### Iteration 1 (2026-10-09 to 10-11)
- **Test:** 16 checks; 6 guards reverted, 5 caught (one after a slow-motion fixture with
  two-model jitter), 1 equivalent (no shared joint is already NaN).
- **Smoke test** with AlphaPose clips standing in for YOLO at their own frames: gate 1 0.8145,
  offset 0 on all three videos, A == Y on 974 clips. Plumbing only.

### Iteration 2 -- the real run (2026-10-12)
- **GPU extraction** (laptop, RTX 3060): V2 32,611 frames, V5 15,519, V9 22,541 at native
  rates; person found in 99.9%, 98.2%, 97.5%. Files arrived byte-identical in size.
- **Gate 3 stopped the first run on V2** (best offset 7, sharpness 1.01), as built. Probing:
  V5 and V9 sit exactly at their annotated frames (0.009 frame heights). V2 does not, at any
  shift in -12,000..12,000 or frame-rate ratio (best 0.035), and an exhaustive search of the
  video finds 0 of 12 clips (V9 control: 12 of 12). Same place, same size, different moment:
  V2's skeletons are not from the video linked as V2 (vault ISSUES-LOG O61). Gate 3 now
  excludes such a video by name and scores the rest; none aligned is still a hard stop.
- **Real-world facts, V5 + V9, 742 punches** (`results/pose_switch.json`):

  | reading | accuracy | macro F1 | hand |
  |---|---|---|---|
  | A AlphaPose | 0.896 | 0.819 | 0.987 |
  | Y_all YOLO, every detected joint | 0.889 | 0.801 | 0.978 |
  | Y YOLO, joints under 0.3 blanked | 0.348 | 0.291 | 0.535 |

  A - Y_all **+0.007 [-0.009, +0.024]** (macro F1 +0.018 [-0.018, +0.054]); per video +0.007
  both. A - Y +0.549 [+0.511, +0.582]. The pose switch costs nothing measurable; the
  confidence rule cost 55 points (fixed in `classify`, that branch's iteration 6).
- **Cut readings, every joint kept:** labelled span 0.889; old fixed window 0.806; per-punch
  extent 0.774; one window per video at its median extent 0.823. Y_ext - Y_fix -0.032
  [-0.057, -0.009]; Y_ext - Y_long -0.050 [-0.073, -0.027]; Y_long - Y_fix +0.018 [-0.005,
  +0.039]. Extent overlap with the label by type: jab 0.49, cross 0.58, lead uppercut 0.51,
  lead hook 0.65, rear hook 0.63, rear uppercut 0.67. 0 clips unseen by YOLO, 0 other-person.
- **Residual, accepted:** two of three held-out videos; coaches' YouTube footage, not phone
  footage; V5 and V9 may share athletes with training videos (O55); the script's gates are
  exercised by the smoke test and the real run, not by unit tests.

### Iteration 3 -- the per-clip cut, through the pipeline's own function (2026-10-13)
- `Y_clip` now calls `classify.clip_cut` / `clip_window` (it was a hand-rolled control, `Y_long`).
- **V5 + V9, 742 punches, every joint kept:** labelled span 0.889, per clip **0.826**, old fixed
  window 0.806, per punch 0.774. Y_all - Y_clip +0.063 [+0.034, +0.094]; Y_clip - Y_fix +0.020
  [-0.001, +0.043]; Y_ext - Y_clip **-0.053 [-0.075, -0.030]**; Y_ext - Y_fix -0.032 [-0.057,
  -0.009]. Clip lengths 0.360 s (V5) and 0.468 s (V9); no punch left out of either median.
- The earlier control gave 0.823 with the same lengths to within a frame.
- **Limit added to the results file:** the per-clip cut was chosen looking at these numbers, so
  they are no longer a clean test of it.

### Iteration 4 -- the two Codex findings (2026-10-21)
Codex (2026-10-20), both confirmed by reading: (1) every reading used `seen and seen_all`, so
A - Y_all, the pose-switch cost under the pipeline's every-joint rule, was computed only on
punches the obsolete 0.3 rule also sees; (2) when alignment passes but no punch has usable
wrist speed, the cut readings are `{"n": 0}` and the print raised `KeyError: 'accuracy'`
before the results file was written.
- **Acceptance:** (a) A vs Y_all read on `all_seen`, legacy Y only on a named subset
  `legacy_common`, both n's in the file; (b) a run with sharp alignment and nothing cuttable
  exits 0, writes results and says so; (c) reverting either fix fails a test; (d) gate 1 still
  0.8145 / 0.7207 on the real run; (e) suites green in a fresh clone.
- **Build:** scoring moved into a pure `aggregate(rows, predict_fn, scored)`; printing into
  `summary_lines(report)`, where every reading may be empty and prints `n/a` (the per-video
  line had the same crash waiting for a video with no seen punch). Results layout:
  `all_seen` {A, Y_all, A_minus_Y_all}; `legacy_common` {A, Y, Y_all, A_minus_Y,
  A_minus_Y_all}; `same_person` and `by_video` each hold both; `n_legacy_blanked`; the cut
  readings (`window`) now start from `all_seen` and carry their own `n`. Window keys unchanged.
- **Test:** 13 new checks in `tests/test_poseswitch.py` (29 total). Eight on `aggregate` with a
  fake classifier: 6 punches both rules see plus 3 only the every-joint rule sees, where the old
  filter reads A - Y_all as 0 on 6 and the fix as 1/3 on 9. Five on the CLI, end to end: a
  synthetic corpus, a random-init ST-GCN++ and a V5 pose of a rigid body on a whole-pixel random
  walk (alignment sharp at offset 0) whose wrists are never detected (no wrist speed, no cut);
  4 of its 20 spans have every joint at 0.2. The HEAD script on that fixture: `KeyError:
  'accuracy'`, no file. **Mutants, 5 of 5 caught:** old filter for all_seen (7 checks fail);
  legacy Y scored on all_seen (6); unguarded accuracy print (1); window guard removed (1);
  window drawn from the common subset (1).
- **Real run, V5 + V9 (same inputs: the pre-fix code reproduced the recorded file exactly):**
  gate 1 0.8145 / 0.7207. `all_seen` 742, `legacy_common` 742, `n_legacy_blanked` 0, 0 missed,
  0 other-person. **The headline does not change:** A 0.896 (F1 0.819), Y_all 0.889 (F1 0.801),
  A - Y_all **+0.007 [-0.009, +0.024]** on 742; per video +0.007 on 594 (V5) and on 148 (V9);
  legacy Y 0.348 on its 742. 27 result blocks compared against the pre-fix run: identical to
  the bit. Why nothing moved: the old rule drops a punch only if fewer than 2 of its frames keep
  ANY joint at 0.3 or above; it blanks joints, not frames, and on these videos no punch lost
  whole frames. The finding was a real flaw with no effect on this data; it would on darker
  footage (IMG_4395) or V2.
- **Downstream:** #20's gate 4 reads `window.Y_all` / `window.Y_clip` (n and accuracy): values
  unchanged (0.889 / 0.826 on 742), but its selection copies the old `seen and seen_all` filter
  and must drop the `seen` half to stay the same selection as this file.
- **Review:** self-review restored macro F1 in the printed pooled lines (still in the JSON).
  **Astra (Codex), one read-only review of 46ba71c..5d86b9c, effort medium: SHIP.** Both
  findings FIXED with line evidence; the new tests judged not vacuous; no new defect. One
  non-blocking note, the #20 gate 4 selection above (its `eval_pipeline_v5v9.py:565`, checked by
  hand). It could not run the tests (no Python on its PATH), so its verdict rests on reading.
- **Security:** no new input path; `aggregate` / `summary_lines` are pure over rows the gated
  loaders built; the test runs the CLI by argument list in `mkdtemp`. No findings.
- **Gate reproduce:** fresh clone of 5d86b9c: pipeline 260, classify 202, hands 38, ingest 109,
  labels 50, env 36, poseswitch 29, machine scripts 38, all passing; the same counts as the
  worktree.

## `feat/e2e-v5v9` -- 2026-10-19

The solo pipeline end to end on real continuous footage with all six punch types: `find_punches.py`
(V6 rule) then `classify_punches.py` (per-clip cut, `--stance orthodox`) on YOLO11-pose of V5 and V9,
scored against every annotated span. Stacked on `feat/pose-switch`; a new script,
`eval_pipeline_v5v9.py`, no existing one changed. Acceptance: (1) it runs the two entry points and stops,
nothing scored, unless the classifier hash is pose_switch.json's, poses are at native fps, spans align at
offset 0, its own perfect-finder readings reproduce pose_switch.json (Y_all 0.889, Y_clip 0.826 on 742)
exactly, and each classified CSV matches its sidecar (hash, clip, stance, classifier, rule); (2) finder
recall by type, precision, count ratio, shift floor, block-bootstrap intervals; type accuracy, macro F1,
hand on matched detections, paired against Y_clip and Y_all on the same punches, commonest-type floor;
(3) a blinded label-completeness audit with controls; (4) negative tests at every file boundary, each
guard reverted fails a test; (5) suites green in a clean clone.

Threat sketch. Assets: the honesty of the first end-to-end number. Entry points: pose npz, annotation
xlsx, classified CSV + sidecar, pose_switch.json, audit judgments CSV, the mp4 (ffmpeg, argv list).
Trust boundary: files on disk between tools. Top threats: a stale or other-clip classified CSV scored;
judgments made on one audit sample applied to another.

### Iteration 1
- **Finding before any gate: the matching, not the pipeline.** First run: matched-detection accuracy
  0.576 against 0.841 for a perfect finder on the same punches, cross recall 0.29. Diagnosis: 147 of 356
  V5 matches sat off their span's wrist-speed peak, 115 exactly on ANOTHER labelled punch's. In a fast
  jab-cross the finder (0.4 s minimum gap) reports one peak, usually the cross; `punchfind.match` takes
  spans by end and hands the jab (which ends first) the cross's detection. On-peak matches scored 0.880.
  Added `match_nearest`: a maximum matching (same count, so recall and precision are unchanged; asserted)
  with the least total distance to span middles. Both attributions are reported. `classify_punches.py
  --labels` and `eval_classify_v6.py` still attribute with `punchfind.match` (not changed here).
- **Audit:** 50 far detections (> 0.3 s from every label) and 10 matched controls, shuffled, judged blind
  from frame strips. Controls 9 of 10 punches, 1 unclear, 0 missed.
- **Test:** 66 checks; 24 mutants, one survived (per-type hits credited to the detection's block: no
  fixture straddled a block edge) and got one.
- **Review: FIX FIRST.** B1 judgments bound only to a key file any run could rewrite: stale judgments
  could be counted against a new sample. Also: key beside the strips (blinding leak); strips never checked
  against the pose's video; `keep` not eval_pose_switch.py's predicate (held only because counts matched);
  missing reference a bare KeyError; tests that compared code with itself (hit count already asserted
  inside; no optimality check; ignored detections untested); NaN time accepted; sidecar rule unchecked;
  stale outputs not removed; BOM-saved judgments refused.
- **Security:** same B1 found independently. Subprocess via `video.iter_frames` (argv list); npz without
  pickle; JSON refuses NaN; CSV bounded; outputs only under `--out-dir`.

### Iteration 2
- **Fix:** every judgments row carries the sample digest (`id,sample_digest,judgment,note`), template
  prefilled; a key that differs from this run's is never silently replaced (re-render, which deletes the
  old strips); key moved out of the judge's folder; the video probed against the pose (size, fps, constant
  rate); eval_pose_switch.py's `seen` predicate; matched spans without a reference counted; finite time;
  rule in the sidecar checked; this run's per-video outputs deleted before the pipeline writes them; BOM
  tolerated; row cap; brute-force optimality over 150 small cases; an ignored detection out of the counts.
- **Test:** 76 checks; 32 mutants, all caught (the row cap needed a message check: without it the id-set
  check still refuses, after reading every row). Stale judgments refused on the real run.
- **Real-world facts** (`results/e2e_v5v9/pipeline_v5v9.json`, orthodox, gate 4 reproduced 742 exactly):

  | finder (V6 rule) | V5 | V9 | pooled |
  |---|---|---|---|
  | precision (lower bound) | 0.860 [0.794, 0.920] | 0.287 [0.159, 0.418] | 0.633 [0.552, 0.706] |
  | recall | 0.599 [0.546, 0.652] | 0.527 [0.336, 0.709] | 0.585 [0.530, 0.641] |
  | count ratio | 0.70 | 1.84 | 0.92 |
  | F1 / shift-floor F1 | 0.706 / 0.366 | 0.371 / 0.096 | |

  Recall by type, pooled: cross 0.76 (127/167), jab **0.32** (84/264), lead hook 0.66, lead uppercut 0.48,
  rear hook 0.86, rear uppercut 0.83. V9 straight punches: cross 0/27, jab 4/31.

  | type, 434 matched detections | accuracy | macro F1 | hand |
  |---|---|---|---|
  | pipeline | **0.811** [0.754, 0.868] | 0.772 | 0.940 |
  | Y_clip, perfect finder, same punches | 0.834 | 0.792 | 0.956 |
  | Y_all, labelled spans, same punches | 0.853 | 0.794 | 0.979 |

  Pipeline - Y_clip -0.023 (clip [-0.060, +0.012], block [-0.061, +0.017]); pipeline - Y_all -0.041
  (clip [-0.085, +0.005], block [-0.117, +0.047]). Commonest-type floor (cross) 0.293. V5 0.860 on 356, V9
  0.590 on 78 (Y_clip 0.679, Y_all 0.821; the pipeline's V9 window is 0.56 s against 0.47 s from the
  labelled punches). End to end: 352 of 742 labelled punches found and named right (0.474).
- **Label completeness:** far detections 19 of 50 real punches [0.26, 0.52]; V5 11 of 25 (8 unclear), V9
  8 of 25. Post hoc: inside the stretch of video that carries labels 19 of 37, outside it 0 of 13 (V9's
  labels cover 144-634 s of 940 s; the rest is talk, squat jumps, push-ups). Precision if the audited
  share holds: V5 0.91 [0.89, 0.93], V9 0.51 [0.41, 0.65].
- **Stance:** the finder's hand read as orthodox lead/rear matches the label's side on 95.2% (V5) and
  94.9% (V9) of matches: consistent with orthodox, or with a southpaw in mirrored footage. Unverified.
- **Residual, accepted:** the per-clip cut was chosen on these videos; V2 excluded (O61); coaches'
  YouTube footage, not phone footage; one judge (a reviewer) from strips; main's key-on-disk and re-render
  paths are exercised by the real run, not by unit tests.

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

## `feat/own-footage-scoring` -- 2026-10-19

Brendan's labels on IMG_4395 and IMG_4406 are single-frame clicks (94 of 94), not start-to-end
spans, and he used "unclear" for FRAMING (hands out to measure or block), not for a punch nobody
could name. Acceptance: (1) a point label matches within half a punch; (2) "unclear" can be
scored as a non-punch, with what the finder and classifier did with those marks reported; (3) a
per-hand finder option, measured against the pooled rule on own footage AND V6; (4) suites green.

### Iteration 1
- **Build:** `punchfind.POINT_TOL_S = 0.2` and `for_matching` (a single-frame label is widened so
  it matches within 0.2 s; spans keep 0.1 s); `--unclear {punch,not-a-punch}` on find_punches and
  classify_punches (framing marks leave the punch set; detections on them are false positives and
  are reported with the type the classifier gave them); `detect(..., per_hand=True)` and
  `find_punches --per-hand` (each hand's own peaks, the 0.4 s gap within a hand; threshold and gap
  unchanged). classify_punches now accepts a row that is its hand's speed peak above the rule's
  threshold, rather than requiring that hand to be the faster one.
- **The 0.2 s point tolerance was chosen after seeing the scores at 0.1 s and 0.2 s**; both are
  below (iteration 2). A click marks one instant in a ~0.4 s punch, and the finder reports the speed
  peak.
- **Test:** hands 51, classify 211; 8 mutants, one survivor (a click read as a one-frame span)
  given a fixture: a click on a still frame within 0.2 s of a punch's peak.
- **Real-world facts, own footage (orthodox; framing as not-a-punch; point rule):**

  | | IMG_4406 pooled | IMG_4406 per hand | IMG_4395 pooled | IMG_4395 per hand |
  |---|---|---|---|---|
  | punches found (recall) | 35/47 0.745 | 40/47 0.851 | 35/40 0.875 | 40/40 1.000 |
  | detections real (precision) | 35/41 0.854 | 40/59 0.678 | 35/42 0.833 | 40/57 0.702 |
  | type right on matched | 0.886 (35) | 0.825 (40) | 0.600 (35) | 0.700 (40) |
  | framing marks flagged | 1 of 5 (cross) | 1 of 5 | 1 of 2 (jab) | 1 of 2 |

  Why the finder misses: 40% and 32% of these punches follow the previous one within 0.4 s, but
  only 11% and 0% follow one by the same hand. The per-hand rule finds them (recall +0.11, +0.13)
  and pays in precision (-0.18, -0.13): the other wrist's retraction is counted too. V6 says the
  same: per hand recall 0.761 vs 0.741, precision 0.361 vs 0.465, F1 0.489 vs 0.571 (second half).
  **Pooled stays the default.** A cross-hand duplicate rule would be a new parameter; it must be
  tuned on data that is not these clips.
- **Framing:** 2 of 7 framing marks drew a detection, named cross and jab, as Brendan expected; most
  false positives come from elsewhere (6 and 7 on the pooled finder, 1 of each on framing). (This
  count was only a lower bound when written; see iteration 2. Here it is exact.)

### Iteration 2 (review: FIX FIRST, 3 blocking)
- **B1, framing count was a lower bound shown as a count.** Punches are matched first and may take
  the one detection a framing mark also had. Both scripts now report `flagged_as_punches_at_least`
  (marks on detections no punch took) and `_at_most` (marks with any detection in reach), and print
  "a to b" when they differ. On Brendan's clips they agree everywhere (pooled 1/1 and 1/1, per hand
  1/1 and 1/1), so "2 of 7" stands as exact.
- **B2, `labelled_length_s_median` counted clicks as zero-length punches** and was NaN (a crash in
  `json.dumps(allow_nan=False)`) when every label was framing. Now the median of start-to-end
  labels only, None when there are none; `point_labels` counts punch labels only, in both scripts.
- **B3, the 0.1 s scores were claimed and absent.** Re-scored (framing as not-a-punch, pooled):

  | | IMG_4406 +-0.1 s | IMG_4406 +-0.2 s | IMG_4395 +-0.1 s | IMG_4395 +-0.2 s |
  |---|---|---|---|---|
  | punches found (recall) | 28/47 0.596 | 35/47 0.745 | 27/40 0.675 | 35/40 0.875 |
  | detections real (precision) | 28/41 0.683 | 35/41 0.854 | 27/42 0.643 | 35/42 0.833 |
  | type right on matched (floor) | 0.821 of 28 (0.321) | 0.886 of 35 (0.343) | 0.667 of 27 (0.444) | 0.600 of 35 (0.371) |
  | framing marks flagged | 1 of 5 | 1 of 5 | 0 of 2 | 1 of 2 |

  The +-0.2 s column reproduces iteration 1 exactly. The wider tolerance is generous by design: it
  credits a detection anywhere in the punch a click sits in, so it is an upper reading of recall
  for point labels; the 0.1 s column is the strict one. Type accuracy barely moves (0.82 to 0.89,
  0.67 to 0.60): the extra matches are not easier punches.
- **Also from the review:** a negative test for the provenance check's local-maximum clause (a row
  above the threshold on the rising slope is refused); a control that `min_gap_s=0` sees more peaks
  in a fast combination than 0.4 s does.
- **Test:** hands 53, classify 218; 6 new mutants, 6 killed (one survivor on the first run, the
  classifier's upper bound, given a fixture where the bounds differ).
- **Deferred (not blocking):** recording `per_hand` in the punches CSV / classified sidecar (a CSV
  format change that the output branch reads); a hand-specific local peak for the point perfect-
  finder reading.

### Iteration 3 (review of 9d10d86 by the code-reviewer agent and by Codex: 3 blocking) -- 2026-10-21
**Iteration 2 got two things wrong, and this corrects the record:**
- **"They agree everywhere, so '2 of 7' stands as exact" did not follow.** The "lower bound" was the
  framing count under ONE maximum punch matching, the greedy's. Which detections a maximum matching
  takes is a tie-break, so it was not a bound. Counterexample (60 fps, tol 6, point_tol 12):
  detections at 88 and 112, a punch click at 100 (reach 88-112), a framing click at 120 (reach
  108-132). The greedy gives the punch 88 and reported 1 to 1. A punch matching that takes 112 is
  just as large and leaves the mark nothing, so the true minimum is 0.
- **B2 read as fixing the all-framing crash, but find_punches.py still crashed.** (Iteration 2's
  "in both scripts" qualified `point_labels`; `labelled_length_s_median` exists in classify only.)
  find_punches.py crashed when every label was a framing mark: `score()` returns `count_ratio = NaN` with no
  punch spans, and `json.dumps(allow_nan=False)` raised after printing precision and recall 0.000.
  No score.json was written.
- **"6 of 6 mutants killed" missed one.** In every fixture `at_most == marks`, so the mutant
  `at_most = len(framing)` survived.

**The fixes:**
- **`punchfind.framing_bounds`, shared by both scripts.**
  - *Lower bound:* framing marks matched maximally against only the detections in NO punch span's
    reach ([start - tol, end + tol] of the widened span). No punch matching can take those, so the
    bound holds for every matching, maximum or not.
  - *Upper bound:* framing marks against all detections, unchanged.
  - The lower bound's detections are reported as `lower_bound_frames` (find_punches) and
    `lower_bound_named_as` (classify); these keys are new in iteration 4.
- **`match`'s docstring now says the greedy is optimal in COUNT only.** Which detections it leaves
  over is a tie-break, not a fact about the footage.
- **find_punches with no punch spans** writes recall, F1 and count ratio as null, skips the chance
  floor, and prints "no punches labelled". `score()` is unchanged; the V6 evaluation depends on it.
- **The `per_hand` record (iteration 2's deferred item).** find_punches writes `<clip>.punches.json`
  beside the CSV: clip, CSV sha256, rule, per_hand. classify records `finder_per_hand` (provenance)
  and `finder.per_hand` (score) only when the sidecar matches the CSV's bytes and the clip, and
  otherwise null. The CSV header is unchanged; a test asserts it, because feat/output-report reads it.

**Tests:** hands 72, classify 228. New cases:
- the 88/112/100/120 counterexample, at the helper and through both scripts (a jab-cross per-hand
  fixture, with controls that the greedy takes the jab and each detection is within reach of the
  intended click);
- a framing mark with no detection in reach (`at_most < marks`), in both suites;
- reach boundaries: 112 vs 113 for a point label, 106 vs 107 for a drag;
- the framing-only run in find_punches, asserting the null scores, no chance floor on a 20 s clip,
  and the printed line;
- the sidecar's negative cases.

**Mutants (round 1, on 4e45764; each applied to a copy of the tree; both suites run):**

| # | mutant | killed |
|---|---|---|
| M1 | lower bound counted against all detections (lower = upper) | yes (hands 6, classify 2 failures) |
| M2 | lower bound = what the greedy left (iteration 2's code) | yes (3, 1) |
| M3 | upper bound = number of marks (the survivor from iteration 2) | yes (3, 1) |
| M4 | upper bound = lower bound | yes (5, 2) |
| M5 | punch reach without match's tol | yes (6, 1) |
| M6 | punch reach on unwidened point labels | yes (5, 1) |
| M7 | lower-bound set taken from the upper-bound matching | yes (3, 2) |
| M8 | find_punches: framing-only score left with NaN | yes (hands crashes: NaN not JSON compliant) |
| M9 | find_punches: chance floor not skipped with no punches | yes (hands 1) |
| M10 | find_punches: F1 left at 0 with no punches | yes (hands 1) |
| M11 | find_punches: framing bounds ignore the punches | yes (hands 2) |
| M12 | classify: framing bounds ignore the punches | yes (classify 2) |
| M13 | classify: sidecar hash not checked | yes (classify 1) |
| M14 | classify: sidecar clip not checked | yes (classify 1) |
| M15 | classify: sidecar per_hand type not checked | yes (classify 1) |
| M16 | find_punches: sidecar always says pooled | yes (hands 1, classify 3) |

**Re-measured on Brendan's clips** (orthodox; framing as not-a-punch), in-process, on iteration 4's
final code. Every finder and type number reproduces iteration 2 exactly:

| | IMG_4406 0.2 s | IMG_4395 0.2 s | IMG_4406 0.1 s | IMG_4395 0.1 s | IMG_4406 per hand 0.2 s | IMG_4395 per hand 0.2 s |
|---|---|---|---|---|---|---|
| punches found | 35/47 | 35/40 | 28/47 | 27/40 | 40/47 | 40/40 |
| detections real | 35/41 | 35/42 | 28/41 | 27/42 | 40/59 | 40/57 |
| type right, matched | 0.886 of 35 | 0.600 of 35 | 0.821 of 28 | 0.667 of 27 | 0.825 of 40 | 0.700 of 40 |
| framing flagged (sound bounds) | 1 to 1 of 5 | 1 to 1 of 2 | 1 to 1 of 5 | 0 to 0 of 2 | 1 to 1 of 5 | 1 to 1 of 2 |
| lower-bound detection named | cross | jab | cross | -- | cross | jab |
| per_hand recorded | false | false | false | false | true | true |

- **The framing counts did not change, and now they are shown, not assumed.** On these clips, the
  old greedy-leftover count also equals the new lower bound in every column (recomputed side by
  side). The two framing detections (frames 1557 and 1436) are out of every punch's reach. So
  "2 of 7 at 0.2 s" (1 of 7 at 0.1 s) is exact. Iteration 2's conclusion was right; its argument
  was not.
- **The upper bound is below the mark count in every column** (1 of 5, 1 of 2): on real data it is
  not the trivial "all marks".

**Review: FIX FIRST** (code-reviewer agent, 1 blocking). See iteration 4.

### Iteration 4 (review of 4e45764: FIX FIRST, 1 blocking) -- 2026-10-21
- **What the reviewer checked and found correct:**
  - **Soundness, by brute force.** 6,000 random instances, enumerating every punch matching. The
    instances covered 0-7 detections with duplicates, points and drags, overlapping spans, spans
    clipped at frame 0, tol in {0, 2, 6} and point_tol in {0, 3, 6, 12}. No violation of
    `at_least <= min` or `max <= at_most`.
  - **The bounds hold for non-maximum matchings too.** The docstring now says so.
  - **Loose but sound.** The lower bound is loose in about 10% of random cases and the upper in
    about 14%. Exact min/max over punch matchings is not done (optional).
  - **Both scripts pass the same spans.**
  - **Nothing on any local branch reads the renamed keys.**
- **B1, the sidecar reader crashed on deeply nested JSON.** `"[" * 30000 + "]" * 30000` is 60 KB,
  under the 64,000-byte cap, and makes `json.loads` raise `RecursionError`. That is not a
  ValueError, so classify died on a valid CSV. Now caught, as `lsimport.load_export` already does,
  with a negative test and controls (the input is under the cap; json itself raises).
- **Also fixed:**
  - **The lower-bound set overclaimed.** It is ONE maximum framing matching's detections, not every
    detection that is certainly on a mark (`[118, 122]` around a mark at 120 lists one). Keys
    renamed to `lower_bound_frames` and `lower_bound_named_as`; docstrings and the printed line now
    say "the detections the lower bound counted".
  - **Precision 0/0.** With framing-only labels AND no detections, precision is null, not 0.0.
    Tested on a still clip.
  - **Mixed provenance.** The sidecar's rule must equal this run's `--rule`, or `per_hand` is
    unknown. Tested with a threshold changed by 0.5.
  - **A vacuous test.** The "holds NaN" half of a sidecar test could not fail: the bool, clip and
    rule checks reject that file anyway. Dropped, not reworded.
- **Not done (noted):**
  - feat/analyse-clip's `OUTPUTS` does not publish `punches.json`, so a standalone classify of a
    published out-dir records `per_hand` as null. Decide it at that merge.
  - Exact framing min/max over punch matchings.
- **Tests:** hands 73, classify 232, pipeline 260, labels 50, ingest 109, env 36.
- **Mutants (round 2, on the final code, all 19 re-run; M8 and M10 re-targeted at the edited lines):**
  **19 of 19 killed.** M1-M16 again, with the same suites failing as in round 1, plus:

  | # | mutant | killed |
  |---|---|---|
  | M17 | classify: RecursionError from a nested sidecar not caught | yes (classify crashes: RecursionError) |
  | M18 | classify: sidecar rule not checked | yes (classify 1) |
  | M19 | find_punches: precision 0, not null, with no detections and no punches | yes (hands 1) |
- **Final review (c60fa9b): SHIP**, no blocking findings. Its re-runs of M5, M17, M18 and M19
  reproduced the results above. Left open (non-blocking):
  - find_punches still reports precision 0.0 via `score()` when punches are labelled but nothing was
    detected (0/0, as in the framing-only case). Override it in the report, not in `score()`.
  - `find_punches.load_rule` accepts NaN/Infinity (no `parse_constant`); this predates the range.
  - an always-true `sx is not None` conjunct in one classify check.
  - this entry's misquote of iteration 2, corrected above.
- **Merge state.** PR #22 is CONFLICTING with its base after #14's fixes (e2459d5, 9f33f3d). A trial
  merge in a throwaway clone conflicts in `classify_punches.py` and this log. Not merged here: no
  gate needed it. At the stack step, keep #22's per-hand peak check, #14's duplicate-row guard
  (threshold and local-peak checks), and `classify.read_clip_window` at both read sites.

### Iteration 5 -- merging #14's fixes in (2026-10-21, base `9f33f3d`)
Brendan asked for the conflict to be resolved here, keeping #22's per-hand peak check, #14's
duplicate-row guard, and `classify.read_clip_window` at both read sites. Merged (not rebased, so
no force-push).
- **The peak check: the two sides disagreed in substance, not only in text.**
  - **#14** refuses a row unless its hand is the FASTER one, and judges the threshold and the
    local peak on the faster hand's speed. That is right for the pooled finder.
  - **#22** judges both on the row's OWN hand, because the per-hand finder reports each hand's
    peaks and the other hand may be faster.
  - **Kept #22's**, plus #14's "that hand has speed" clause, which #22's lacked; it matters only
    for a rule whose threshold is 0.
  - **Cost:** #14's accepted rows are a strict subset of the merge's. For a pooled list, three
    kinds of row the pooled finder cannot emit now pass where #14 refused them:
    - a slower hand's own peak;
    - a faster hand's peak that is not a peak of the two hands' max curve (left [0, 5, 3] and
      right [6, 1, 0]: the max curve [6, 5, 3] has no peak at frame 1);
    - the second hand on an exact speed tie.

    #14's comment already called the check "necessary, not sufficient". (Correction, from the
    review: the reason first given here for not using the per_hand sidecar, "recorded, not
    trusted", argues only against LOOSENING by it. Applying #14's pooled check when the sidecar
    says pooled would fail safe. It was not done, because Brendan asked for #22's check; it is an
    open option.) Done in iteration 6, at Brendan's request.
- **The read sites:** `read_clip_window` at the pipeline's read and at the perfect-finder reading,
  as #14 had them. The IoU line keeps #22's exclusion of point labels.
- **#14's duplicate-row guard** (`read_punches`) merged without conflict. Per-hand rows at one
  frame for both hands are different (frame, hand) keys, so they are not duplicates.
- **The log:** #14's iteration 8 now ends its own section, before this one.
- **Mutants on the resolution** (copies of the tree, test_classify):

  | # | mutant | first run | after the new tests |
  |---|---|---|---|
  | X1 | drop the "has speed" clause | survived: `clip_cut`'s `peak_width` refused the row later, so the outcome matched | killed: the test now pins that the PEAK CHECK refuses it (a still frame at threshold 0) |
  | X2 | drop the threshold clause | killed | killed |
  | X3, X4 | allow a faster previous / next frame | killed | killed |
  | X5 | judge on the faster hand's speed (#14's signal) | killed | killed |
  | X6 | restore #14's faster-hand clause | **survived**: no fixture had a per-hand row whose hand is the slower one | killed: both hands punch together, the right at half reach; the right's rows (left faster) must be accepted |
  | X7 | pipeline read back to `cut_span(clip_window)` | **survived** | killed |
  | X8 | perfect-finder read back to `cut_span(clip_window)` | **survived** | killed |
  | X9 | IoU includes point labels | killed | killed |
  | X10 | allow duplicate (frame, hand) rows | killed | killed |
  | X11 | local peak judged on the faster hand's curve, threshold on its own (found by the review) | **survived**: in the slower-hand fixture both hands peak on one frame | killed: the right hand's punches 2 frames later, so at its peak the left is faster AND on its slope; accepted, while a row on the right's own rising slope is still refused |

  - **Why X7 and X8 survived:** #14's frame-rate tests call `read_clip_window` directly, so no
    script-level test pinned either read site. A spy on `classify.read_clip_window` now runs
    classify on a clip with fewer than 5 punches (the training cut, where the two reads differ).
    It asserts that every finder row goes through it, and that each named labelled punch does too
    (3 + 2 calls).
- **Tests:** hands 73, classify 262 (232 + 19 from #14 + 11 new), pipeline 260, labels 50, ingest
  109, env 36.
- **Review of the merge (82ff682, code-reviewer agent): SHIP**, no blocking findings. It diffed the
  merge against both parents: nothing was lost except the deliberate lines. It found the X11
  survivor and the incomplete cost statement above, both fixed in the follow-up commit. Left open:
  - applying #14's pooled check when the sidecar says pooled (above; done in iteration 6);
  - for clips under 5 punches, the printed window length is `cut.length_s` (8/24 s, "0.33 s") where
    the docstring says 0.375 s, and the provenance writes `"per": "clip"` beside
    `source: "training"`. Both predate the merge.
- **Re-measured on the merged code:** every number in iteration 3's table is unchanged, including
  the framing bounds and per_hand. Both clips have far more than 5 punches, so they use the clip's
  own cut, which `read_clip_window` reads exactly as before. #14's change is to the under-5
  fallback only.

### Iteration 6 -- the pooled check, when the sidecar says pooled (2026-10-21)
Brendan's call, on iteration 5's open option. When the verified sidecar says `per_hand: false`,
classify_punches also applies #14's pooled check. Otherwise it keeps the own-hand check.
- **The checks are now two functions,** so each case can be tested on three-frame arrays:
  - `own_hand_peak`: every row, any finder. That hand has speed, reaches the threshold, and no
    neighbouring frame of that hand is faster.
  - `pooled_peak`: only when `finder_per_hand` is False. The hand is the faster one (`np.argmax`,
    so the left on an exact tie, as `punchfind.detect` names it), and the threshold and the local
    peak are judged on the two hands' max.
- **Why it is safe to let the sidecar choose:** the sidecar only TIGHTENS. True and None
  (missing, stale, forged, another clip or rule) both get the own-hand check, which every finder
  passes, so a bad sidecar can never loosen the check. A forged `false` could at worst refuse the
  CSV. The sha256 catches a stale sidecar but authenticates nothing: anyone holding the CSV can
  compute it, so the safety is that `false` only tightens. `finder_per_hand`'s docstring says so;
  it no longer says "never trusted for anything else".
- **Tests (classify 262 to 274):**
  - Unit cases, each passing the own check and refused by the pooled one:
    - the review's left [0, 5, 3] / right [6, 1, 0], a faster hand's peak that is not a peak of
      the max;
    - a slower hand's own peak;
    - an exact tie, refused for the right hand;
    - plus the threshold, the strict neighbourhood, and the first and last rows.
  - Through the script, on IMG_5151 found by the POOLED finder (left only), plus a right-hand row
    at one of the right's own peaks, the sidecar re-hashed to the edited CSV:
    - refused with a verified pooled sidecar, by the pooled check's own message;
    - accepted with a per-hand sidecar;
    - accepted with no sidecar.
  - A control that the pooled finder's own list, with its sidecar, passes.
- **Mutants: 20 of 20 killed.** X1-X11 were retargeted at `own_hand_peak` and re-run:

  | # | mutant | killed by |
  |---|---|---|
  | P1 | pooled check never applied | the pooled-sidecar refusal |
  | P2 | pooled check applied to every list | 4: per-hand rows where the other hand is faster |
  | P3 | an unknown sidecar treated as pooled | the no-sidecar acceptance |
  | P4 | faster-hand clause dropped | the slower-hand unit case (3) |
  | P5 | a tie accepted for either hand | the tie case |
  | P6 | local peak on the max dropped | the [0, 5, 3] / [6, 1, 0] case (3) |
  | P7, P8 | a faster previous / next frame of the max allowed | the [0, 5, 3] case; the neighbourhood case |
  | P9 | threshold on the max dropped | the unit threshold case only. In `main` it is redundant: the own check runs first, and the argmax hand's speed IS the max |

- **Real data:** the stricter check accepted every row of the pooled finder's real output. That is
  41 rows on IMG_4406 and 42 on IMG_4395, at both point tolerances, each with a verified
  `per_hand: false` sidecar, so the check ran. Every number in iteration 3's table is unchanged.
- **Review of b9bebaf (code-reviewer agent): FIX FIRST, 1 blocking, a test gap; the code was
  correct.**
  - **Nothing tested that a row on a plateau is accepted.** `find_peaks` reports the middle of a
    flat top, whose neighbour is EQUAL. So `>` to `>=` in any of the four neighbour clauses would
    refuse genuine finder output, the worst failure this check can have, and all 274 still passed.
  - **The fix:** a 3-long and a 2-long plateau, with the row taken from scipy's own `find_peaks`
    (what `detect` calls), must pass both checks.
  - **Mutants Q1-Q4 (`>` to `>=` in each neighbour clause): 4 of 4 killed**, each by the plateau
    cases.
- **What the reviewer verified by construction:** no false refusal of genuine output.
  - A fuzz of `detect` on 20,000 random tracks with gaps, quantised speeds (so plateaus and ties),
    thresholds 0-3 and gaps 0-0.2 s: of 36,434 pooled and 79,956 per-hand rows, 0 refused by
    either check.
  - 150 synthetic poses through `find_punches.main` and `classify_punches.main`, at 24, 30 and
    60 fps, pooled with verified sidecars and per hand: 0 of 1,796 rows refused.
  - **Why none can be:** `find_peaks` never reports the first or last sample of the segment it is
    given, so the neighbour comparison never crosses a frame-id gap.
- **Also from the review:**
  - The hash wording no longer reads as authentication: it detects staleness only.
  - **The edges test.** Renamed, and the first rename overclaimed. It said "no wrap-around", but no
    case could catch one: the re-review's `i > 0` to `i >= 0` mutants (W1 in `own_hand_peak`, W2 in
    `pooled_peak`) passed all 276.
    - **The fix:** row 0 of `[6, 3, 9]` must pass both checks. Under W1 and W2 it is compared with
      `[-1]`, the 9, and refused. W1 and W2 are killed.
    - **The last row:** it cannot wrap. Without its guard, `i + 1` raises IndexError. Its case
      (`[3, 6]`, row 1) now runs for both checks.
    - **No genuine output is affected:** `find_peaks` never reports row 0.
  - **Noted, not changed:** both checks accept a row on a track's or segment's first or last row,
    which the finder never emits. Tightening that is possible but out of scope.
- **Re-review of 88c8ea3: FIX FIRST**, for the overclaiming test name above; Q1-Q4 confirmed killed.
- **Re-review of 89fc9dc: SHIP**, no blocking findings.
  - W1 and W2 killed. Removing either last-row guard (`i + 1 < len(...)`) crashes the suite with
    IndexError on the new `[3, 6]` row-1 cases.
  - The test names and this entry now claim only what the tests show.
  - Optional, not done: record a crash as a named FAIL instead of a traceback, and give W1 and W2
    one check each.
- **Tests:** hands 73, classify 277, pipeline 260, labels 50, ingest 109, env 36.

## `feat/punch-overlay` -- 2026-10-19

The plan's third output, the demo-page asset: the footage with the skeleton AND each punch
drawn on it (`render_punches.py`, one function `render_punch_overlay(...)` for
`analyse_clip.py`). Stacked on `feat/output-report`. Acceptance criteria: (1) H.264, constant
frame rate, no audio, one frame per pose frame; (2) label pixels change on exactly the frames
inside a punch's span; (3) a span running past the first or last frame labels those frames;
(4) a CSV changed since its sidecar, a sidecar for another clip / pose length / rate, a
missing sidecar, a peak outside the pose, mismatched clip names, a pose that does not name
its video, and an output onto an input are all refused, as ValueError (the caller's
"refused"); (5) the burned-in frame number survives; (6) IMG_4406 rendered and six frames
looked at.

Threat sketch. Assets: participant privacy (never rendered) and the honesty of the labels on
a public demo. Entry points: the classified CSV, its sidecar, the pose .npz, the video. Trust
boundary: files on disk between tools. Top threats: a stale or edited CSV putting wrong labels
on the demo (sha256 against the sidecar, hashed and parsed from one read); a video/pose
mismatch drawing a skeleton or label on the wrong frames (render_overlay's own size, rate and
length checks, reused through an `annotate` hook rather than copied).

Drawing reuses overlay.py: a 5x7 capital font (`draw_text`, refusing any character it has no
glyph for), `draw_ring` built on `draw_line`, and the hook. No Pillow (not in the venv), no
OpenCV. The font was rendered and read before any test was written.

### Iteration 1
- **Test:** 48 checks. Two caught at once: the label cap's "+N MORE" had no glyph (`+`; the
  refusal worked, the glyph was added) and a fixture put both wrists inside one ring.
  A frame-number check comparing pixels across lossy encodes failed on encoder noise
  (~5 levels on unlabelled frames too); replaced by reading the number back bit by bit
  (`read_number`), with a negative control. Mutation: spans made end-exclusive -> 3 fail.
  Removing the clamp was NOT caught -- it was dead (start <= i <= end already holds past
  either edge), so it was removed.
- **Review (code-reviewer): FIX FIRST.** B1: a sidecar `fps` of 10^400 escaped as
  OverflowError. B2: hash-matched crafted inputs escaped as RecursionError (nested JSON),
  csv.Error (a 200k-char field), KeyError (an .npz missing arrays), FileNotFoundError --
  `analyse_clip.py` catches only ValueError. B3: an empty `pose.source` skipped the
  pose-to-video check. Suggestions: hash and parse one read; three tests checked shape not
  substance (overlap, cap, text_size); dead variables; duplicated frame-number constants;
  glyph failures found mid-encode; hard link onto an input.
- **Real run:** IMG_4406, 2353 frames in 41 s; h264, one stream, 0 irregular intervals,
  same 47058/785 fps as the source. Frames 42/50 (jab), 158/166 (lead hook) labelled with the
  lead wrist ringed; 100/130 bare.

### Iteration 2
- **Fix:** all of the above, the tests made exact (the label block compared pixel-for-pixel
  against the expected lines drawn on black), one negative test per escaping type.
- **Test:** 62 checks. Mutation: dropping the conversion -> 6 fail; restoring the empty-source
  bypass -> 2 fail.
- **Re-review:** B1, B3 CLOSED. B2 OPEN: a 300-byte .npz claiming a 10^12-frame array raised
  MemoryError; probe's TimeoutExpired and a decode RuntimeError were outside the conversion.
  And the conversion wrapped all of `_checked_inputs`, so a KeyError from a real bug would
  have read as "refused".

### Iteration 3
- **Fix:** conversion moved to the load sites only (`_load(path, read)`, naming the file):
  the pose load, the CSV read, the sidecar, the CSV parse, and `probe` (now inside the checks,
  so a hanging ffprobe is a refusal). MemoryError and SubprocessError added.
- **Decided, not fixed:** ffmpeg failing part way through a decode stays a RuntimeError,
  partial file removed, and is documented as such: it cannot be told apart from the encoder
  failing (a full disk), which is not a refusal. In `analyse_clip.py` the same video has
  already been decoded once for the pose, so a video that fails here failed there first.
- **Test:** 64 checks; mutation: removing MemoryError and SubprocessError -> 2 fail. Every
  suite: render_punches 64, ingest 109, report 130, classify 202, hands 38, labels 50,
  pipeline 260, env 36, machine scripts 38; all green.
- **Residual, open:** `PoseSequence.load` (shared by every tool) has no size cap read from the
  .npy headers, so a large-but-allocatable crafted pose spends RAM before any check. Converted
  to a refusal here only when allocation fails outright. A loader-level cap belongs in
  `pose_yolo.py`, for every consumer.
- **Residual, accepted:** a refusal found while decoding removes whatever was at `out`
  (VideoWriter's existing behaviour, shared with run_ingest).

### Iteration 4 -- feat/output-report's four fixes merged in (2026-10-21)
Acceptance: (1) the merge keeps this branch's `data` argument and the base's required `fps=`;
(2) the overlay refuses exactly the sidecars the report refuses, and a hash-matched CSV whose
times belong to another frame rate; (3) every suite green.
- **Merge:** conflicts in `report.py` (`read_classified`'s signature: now
  `(path, duration_s, data=None, *, fps)`) and this log (output-report's iteration 5 placed in
  its own section). `render_punches.py` now calls `report.check_pose` instead of its own
  frames/fps check (that one had a 1e-6 tolerance and its own overflow handling; the library's
  is exact and overflow-safe, and accepts both real clips' sidecars, checked read-only on #15),
  and passes `fps=pose.fps` to the reader.
- **Test:** render_punches 64 -> 66 (a hash-matched CSV with times at frame / 30 is refused
  with "is not frame"; bytes and fps together read the same rows as the path); report 185,
  classify 209, env 36, hands 38, ingest 109, labels 50, pipeline 260. Every earlier sidecar
  test (another length, another fps, no fps, a 400-digit fps, a float length) still refuses.
  Mutants: the `check_pose` call removed, the row-time check off, `check_pose` ignoring fps --
  3 of 3 caught.
- **Review / security:** no new input; one duplicated check removed in favour of the shared one.
  No findings.
- **Review (Astra / Codex, read-only, effort medium, one review of the #17/#18/#19 merges): SHIP all.** Merge OK on each branch, no adaptation defects; it checked that the shared checks are identical to feat/output-report's on all three (confirmed here: no changed line). Its sandbox could not run Python, so its verdict is from reading plus the results above; fresh clones of 2e2059f, f2412e8 and a2f90be reproduce every count.

## `feat/face-blur` -- 2026-10-19

A privacy control: protocol FA26-22-NG as amended (vault ISSUES-LOG O57, O58, O63)
says the second annotator sees face-blurred clips only and is never told names.
`blur_faces.py` + `src/boxingvi/faceblur.py`.

**Acceptance criteria.** (1) Every planted face in synthetic frames is covered by the
blur AND its texture is destroyed in the pixels, in every mode, including after H.264
encoding. (2) Every failure path (predictor raised, malformed output, nobody found, a
person missing against neighbouring frames, overlapping people with a missing or
ambiguous face, a face outside its box) blurs the whole frame. Under random confidence
loss on every joint, random body sizes and cropped boxes, no planted face survives.
(3) The output has one H.264 video stream, no audio, and none of the input's planted
metadata, or the participant name in its file name, anywhere in its bytes; the report
names no file. (4) Output is refused, with a reason code and before a frame is
decoded: outside `~/comps-participant-data/` (unless overridden, recorded in the
report), and inside a git checkout, the vault, a Syncthing folder (config or marker),
a cloud-synced folder or a network path, override or not. (5) No unblurred frame is
ever handed to the encoder (a spy on the writer); a failure leaves no file behind.
(6) On own footage only, no face is recognisable in sampled frames that include the
fastest motion; the frames are saved for Brendan to confirm.

**Threat sketch.** Assets: participants' identities (faces; names in file names and
metadata; GPS and creation time in phone video). Entry points: the input video
(untrusted container), the predictor's output (a model can be wrong in any way), the
output path, the YOLO checkpoint (a pickle; pinned sha256, as in ingest). Trust
boundaries: predictor output into `plan_frame` (validated, and every doubt goes to
more blur); the output path into `check_output_location`. Top threats: (1) a face
that the detector misses or misplaces reaches the annotator unblurred, most likely in
fast motion or a clinch; (2) blurred footage, or the report, lands somewhere it syncs
or is committed, or carries a name or a location.

### Iteration 1

- **Build:** mosaic (block means; cannot be undone the way a Gaussian can be partly
  deconvolved). 3x3 cells over a face patch, 16 cells along the long side of a frame
  blurred whole. Two passes: predictions only are kept from the first. Frames reach
  the encoder through a pipe, and the writer's `private=True` strips the muxer's own
  tags. `YOLO_OFFLINE=1` is set before ultralytics is imported: it sends usage events
  to Google Analytics when online (`ultralytics/utils/events.py`).
- **Test:** the random-degradation test found its first hole at once. When the only
  confident face point is one ear, the patch centre sits on the side of the head, and
  PAD 0.6 left a 2-px sliver uncovered. PAD is now 0.8, with a named regression test.
  73 checks. All other suites green.
- **Review (code-reviewer subagent): FIX FIRST.** B1: a head-and-shoulders box with
  one face point got a patch covering 31% of the head (the box-height floor assumed a
  whole body). B2: in a clinch both skeletons can claim one head, leaving the other
  bare. B3: counting people lets a false detection stand in for a missed boxer. B4:
  the output check was a denylist. B5: the network-path test could not fail (an
  earlier rule fired first), and no refusal test checked which rule fired.
- **Security (own pass):** the report carried the input's file name, and participant
  clips may be named after the participant. It now carries the input's sha256 instead.

### Iteration 2

- **Fixes:** B1, box size when the torso is not measurable. B2, overlapping people
  whose faces are close together, or inside each other's box, blur the whole frame.
  B3, one-to-one matching of boxes by position against every frame within 0.25 s.
  B4, deny by default under the participant root. B5, reason codes asserted in every
  refusal test. Also: input hashed before and after (the plans must fit the pixels);
  a re-check of the output location before the final rename; overflow-safe sizes;
  `missing_conf=0.0` (a model with no keypoint confidences blurs more, not less);
  `\\?\` and network-drive handling; cloud names matched exactly where they are
  generic ("boxing" is not "Box").
- **Test:** the random test, extended to cropped boxes and body sizes, found 20
  scenes where the box-only patch (top third) missed the chin of a cut-off box. A box
  that is not body-shaped (height < 2 x width) is now blurred whole. This departs
  from the brief's "top third" toward more blur, and is flagged in the PR. 103
  checks. 15,000 extra random scenes on three other seeds: 0 faces left. All other
  suites green.

### Iteration 3

- **Review of iteration 2: FIX FIRST.** B1-B5 and every suggestion taken were verified
  closed. Two new holes appeared in the new rules. N1: the distinct-head test ran only
  for boxes that intersect, but a face may sit up to 25% outside its own box, so a
  skeleton reached a neighbour's head across a gap, leaving the neighbour's real head
  0% covered. N2: hips the model imagines just under the shoulders in a crop "measured"
  a torso and switched off the box-size floor (31% of the head, B1 again).
- **Fixes:** the distinct-head test runs on every pair with faces. The box floor stays
  unless the skeleton reaches the legs (`_whole_body`). A legs-only box no longer
  vouches for a person in neighbour matching (top edges must agree). An
  `InvalidPrediction` sentinel. `report_extra` cannot overwrite core fields. A failed
  report move removes the video and any stale report.
- **Test:** 108 checks. The random test now imagines hips and legs inside crops.
  15,000 extra scenes: 0 faces left.
- **Real data (iteration-2 code, own footage IMG_4395, 2659 frames at 60 fps):** 25.9%
  of frames blurred whole: 577 a person missing against the neighbouring frames, 52 nobody
  found, 61 overlaps. YOLO found 2-4 "people" in 96 frames of a solo clip. Over-blur,
  not leakage; it is the cost of DET_CONF 0.10 and the neighbour rule.

### Iteration 4 -- past the cap, NOT independently re-reviewed

- **Review of iteration 3: FIX FIRST.** N1 and N2 closed. Two siblings: P1, one person
  with a face and a neighbour without, the face sitting on the neighbour's head across a
  gap (owner's head 0% covered); P2, a knee imagined BELOW a mid-frame crop satisfied
  `_whole_body` (31%).
- **Fixes (each only adds blur):** a lone face within the slack of another person's box
  blurs the whole frame. `_whole_body` requires the leg joint inside the box. Both are
  named tests; the random test also imagines legs below crops. 110 checks, 15,000 extra
  scenes with 0 faces left, every other suite green.
- **Not taken:** P3, a tall crop with three layers of wrong confident joints reaches 80%
  head coverage. Closing it (a patch floor of half the box width) enlarges every patch
  on a wide guard.

### Blocker report (iteration cap reached)

**Gate that keeps failing:** review. Each round closed what it found, and the next round
found a sibling of the same shape: a rule that trusts keypoint geometry, defeated by a
second piece of confidently wrong geometry (N1 -> P1, N2 -> P2).

**Root cause hypothesis:** the patch is placed and sized from the keypoints, and each
new guard enumerates one more way they can be wrong. The space of wrong-but-confident
skeletons is open-ended, so a guard list never closes it. Tests and random scenes show
no remaining hole, but they only sample the failure shapes we already thought of.

**Options:**
1. **Box-based blur is the floor; keypoints only ADD blur.** Every person always gets
   the box-only patch (top third, or the whole box if not body-shaped), and the
   keypoint patch is unioned on top. Closes the whole class, P3 included, because no
   keypoint can shrink what the box alone would blur. Cost: the top third of every
   boxer (head, shoulders, guard) is always mosaicked, which is where feints show.
   That is a real cost to the feint labels this clip exists for.
2. **Keep the guard list; make the human check part of the protocol.** Ship as is,
   with a per-clip spot check of sampled frames (fastest motion, every `box` and
   `ambiguous_faces` frame) by Brendan before a clip goes to the annotator. Cheapest.
   It relies on a person catching what the code missed.
3. **Another review round with a raised cap** (`--max-iterations 5`), specifically on
   P3 and on sampled frames from the first real sparring clip. That is real data with
   two boxers, which no test here has.

### Real check -- own footage only, final code (2026-10-19)

Run with `--allow-outside-participant-root` (own footage, scratchpad output), YOLO11m-pose
on CPU, weights pinned by sha256, `YOLO_OFFLINE=1`.

| Clip | Frames | Face patch | Box patch | Whole frame | People per frame |
|---|---|---|---|---|---|
| IMG_4395 | 2659 | 1828 | 117 | 714 (26.9%): 603 missing vs neighbours, 52 nobody, 33 overlap, 26 ambiguous | 0:52, 1:2511, 2:85, 3:10, 4:1 |
| IMG_4406 | 2353 | 1877 | 90 | 386 (16.4%): 376 missing vs neighbours, 10 overlap | 1:2265, 2:88 |

Twelve frames were extracted from the blurred output into `results/face-blur-check/`
(gitignored): per clip, the three fastest-motion frames (measured on the original in
memory), one box-patch frame and one whole-frame blur, plus the two fastest
face-patch frames of IMG_4395. Looked at one by one: **no face is recognisable in any
of them.** Patch frames show a 3x3 mosaic covering the head with a wide margin
(shoulders and guard included). Whole-frame blurs leave a close-up face as at most a
few flat 120-px cells. Pending Brendan's own look.

Observed costs, not leaks: the solo clips have 2-4 "people" in 96 and 88 frames (false
detections at DET_CONF 0.10, e.g. furniture at the frame edge). Each one blurs about
+/-15 frames whole through the neighbour rule. Patches are large (about 3 head widths),
so the guard is mosaicked whenever the face is.

### Iteration 5 -- option 1, the box is the floor (Brendan's choice, 2026-10-20)

- **Change:** every person's box patch is always blurred; the face patch is added on
  top, so keypoints only add blur. All whole-frame rules unchanged.
- **Test:** the floor at a third missed P3 (96%). The floor became max(1/3 of the box,
  1 box width), then 1.5 box widths after review H1 (a box-wide head at exactly 2:1,
  no face points: 96%). New tests: every person's box patch is blurred in every
  non-full plan (500 noisy-skeleton scenes); face points confidently on the chest
  still leave the head covered; the random test also scatters face points confidently
  anywhere in the box. 114 checks; 15,000 extra scenes with 0 faces left; every other
  suite green; a clean clone of `b7a6e88` gives the same counts.
- **Review: SHIP.** All eight earlier leak probes (B1-B3, N1, N2, P1-P3) closed. The
  reviewer found no guard rule dominated, so all are kept. Residuals, now in the
  operator notes: a person never detected, and a box that consistently leaves out the
  head with no confident face points.
- **Security:** the change is blur geometry only, with no new input, path or I/O
  handling. The iteration-1-4 findings stand.
- **Real check (own footage, final code):** modes and whole-frame counts are as before
  (26.9% and 16.4%); the floor changes regions, not frames. The 12 frames in
  `results/face-blur-check/` were re-extracted and looked at again: **no face
  recognisable.** **Cost seen:** a boxer's box in guard is ~0.4x as wide as tall, so
  1.5 box widths is ~60% of the box. The floor mosaics head to hips (IMG_4406 frame
  209), or to the knees (frame 2204): the guard, the torso and most of the arms. That
  is far more than the "top third" option 1 described, driven by P3 and H1, which are
  synthetic close-up crops. Open decision for Brendan.

### Iteration 6 -- option (c): the 1.5-width floor only at the frame edge (Brendan, 2026-10-20)

- **Change:** a body-shaped box gets the top third (+pad); it gets 1.5 box widths only
  when it touches the frame edge, where real close-up crops come from. A box that is not
  body-shaped still gets all of it.
- **Review: FIX FIRST, then SHIP.** The first version used a fixed 2 px edge band, which
  is finer than YOLO places an edge (one model pixel is about 3 frame pixels at 1080p).
  A crop 3 px short of the edge lost the extension, leaving 70% of the head covered. The
  band is now 1% of the frame's long side (6.4 model pixels at any resolution). New
  tests: crops 3 px short at 1080p and 6 px short at 4K, in all four orientations
  (69.6% each with the old band, 100% now), plus a mid-frame whole boxer whose floor
  must stop at the top third. The reviewer would move to 2% unless a histogram of
  real box-to-edge gaps argues for 1%; left as a tuning decision. Black bars baked
  into a clip are an operator rule ("feed the camera's own file"), not a code check.
- **Accepted residual (Brendan's decision):** a mid-frame body-shaped box whose head
  reaches below the top third + pad, with no face patch.
- **Test:** 121 checks; 15,000 extra scenes, 0 faces left; every other suite green; a
  clean clone of `284cbc5` gives the same counts.
- **Real check (own footage, final code, `edge_frac` 0.01 in both reports):** mode and
  whole-frame counts are unchanged (IMG_4395: 26.9%; IMG_4406: 16.4%). All 12 frames
  were re-extracted and looked at: **no face recognisable.** The floor on a mid-frame
  boxer now stops at about the chest: in IMG_4406 frame 2204 the torso, arms and a
  hand are visible again. Close-up and edge-touching boxes (IMG_4395) are still
  mosaicked wholly or to 1.5 widths.

### Iteration 7 -- tighter blur for feint labels (Brendan, 2026-10-20)

- **Why:** Brendan found no face recognisable, but he "genuinely can't see" the boxer. Measured over
  every frame (16x16 blocks that were textured in the original and flat in the output):
  44-53% of the textured picture was mosaicked, and even typical frames hid the head,
  guard, shoulders and chest.
- **Change:**
  - Face patch: with two or more confident face points, the head is measured from the
    points (`head_width`: max over visible pairs of distance x frontal ratio), and the
    patch is ~1.8 head widths (was 3-4). One point, or clumped points, keep the
    body-sized patch.
  - Floor: the top fifth of a whole body (was a third), or down to the LOWER confident
    shoulder; a third + 10% with no confident shoulder, and always for boxes 4:1 or
    taller.
- **Review:** FIX FIRST twice, then SHIP.
  - R1: with arms raised the box top is the gloves, so the head is below the top fifth.
    With face points on a glove the head was 0% covered; the shoulder anchor fixes it.
  - Then the skeleton fitted up a raised arm (face on the glove, shoulders on the
    elbows) is ONE failure (0%). It is closed by the box shape: 4:1 or taller always
    gets the deep floor. Guard (2.6-3.4:1) and punch boxes keep the shallow floor.
  - A mean of the shoulders gave 47% with one shoulder high; the lower shoulder is now
    used.
  - The factors were checked against a 3-D head model (frontal 1.09-1.41x the true
    width), not against real heads.
- **Test:** 134 checks; 15,000 extra scenes with 0 faces left; every other suite green; a
  clean clone of `446d42d` gives the same counts.
- **Real check, own footage, det-conf 0.10 vs 0.25:**

  | Clip | Whole-frame blur | Nobody found | Textured area mosaicked (mean / median) |
  |---|---|---|---|
  | IMG_4395 @0.10 | 26.9% | 52 | 53% / 40% |
  | IMG_4395 @0.25 | 15.3% | 119 | 46% / 35% |
  | IMG_4406 @0.10 | 16.4% | 0 | 44% / 33% |
  | IMG_4406 @0.25 | 3.8% | 0 | 38% / 28% |

  0.25 removes most of the false "people" in these solo clips (2-4 people in 96 and 88
  frames, down to 35 and 2), and with them most whole-frame blurs. Its cost is misses
  (IMG_4395: nobody found in 119 frames instead of 52); those frames are blurred whole,
  so they are safe here. On two-boxer footage a sustained miss is not caught by the
  neighbour rule; pass --expect-persons 2. Recall on sparring is unmeasured.
- **What the tighter blur bought:** in face-patch frames the median mosaicked area moved
  little (27-31%), because the box floor now dominates. It must be box-wide (the head
  can be anywhere across the box when no keypoint is trusted) and reach the shoulders,
  so gloves held in guard at face height stay mosaicked. Below the shoulders the arms,
  hands and torso are visible again (IMG_4406 frame 970: arm, hand and watch). Typical
  frames for both settings are in `results/face-blur-typical/`.

### Iteration 8 -- hybrid with deface's face detector (Brendan, 2026-10-21/22)

- **Why:** Brendan found the blur "terrible" for labelling and asked for a free tool.
  deface (CenterFace, MIT, local) on own footage obscured ~8% of the textured picture
  (ours 39-47%) and left the guard visible, but alone it found no face in 3.8% (IMG_4406)
  and 28.5% (IMG_4395) of frames where YOLO saw a person. A contact sheet of those frames
  showed real faces unblurred: mid-punch, in profile, behind a glove.
- **Change:** a person drops the box floor only when the CenterFace box and YOLO's
  keypoints agree. That requires a frontal face (nose and both eyes confident, all inside
  the box), a box at least half the head wide (head measured from eye-to-eye distance), a
  box centre on that person, and no confident wrist inside the box. The agreed person gets
  the box grown 1.3x plus a 1.2-head-width square on the nose and eyes; a box 4:1 or
  taller keeps its deep floor anyway. Everyone else keeps the floor + face patch.
  Unclaimed detections are blurred; malformed output or a failing detector means no
  detections. The model is pinned by sha256; deface 1.5.0 is installed `--no-deps`.
- **Test:** 155 checks. A 1000-scene random hybrid test found 79 leaks before the keypoint
  square was unioned in (one-sided face points "agreeing" with a shifted box); now 0, and
  15,000 more scenes on other seeds give 0.
- **Review:** FIX FIRST, then SHIP with one residual for Brendan.
  - D1: arms up, face points and box both on the glove: 0%.
  - D2: guard, glove decoy: 30%.
  - Partial three-quarter boxes.
  - Closed by the arms-up floor under agreement, a wrist veto and frontal-only agreement.
  - **Residual:** a glove at the cheek fools both models while YOLO misses that wrist.
- **Real data:** only 6 of 200 sampled persons agreed. The cause was the max-over-pairs
  head estimate running ~2x the true head on real keypoints, so every CenterFace box
  failed the size check. With the head measured from the eyes: 97 of 200.

  | Clip @det-conf | Face blur only (agreed) | Fallback | Whole frame | Textured area obscured (median) |
  |---|---|---|---|---|
  | IMG_4406 @0.25 | 48% | 48% | 3.8% | 17% (deface 7%, previous blur 28%) |
  | IMG_4395 @0.25 | 28% | 66% | 15.3% | 30% (deface 8%, previous 36%) |
  | IMG_4406 @0.05 | 27% | 28% | 44.5% | -- |
  | IMG_4395 @0.05 | 23% | 45% | 32.3% | -- |

  0.05 (Brendan's ask) multiplies false "people" (up to 6 in a frame) and with them
  neighbour-rule whole-frame blurs; 0.25 is the better setting. At every frame where deface
  leaked a face (18 sampled per clip), the hybrid covers it. Agreed frames look like
  deface: face mosaicked, arms, hands and guard visible. Three-quarter and profile faces
  (the bladed stance) keep the fallback.

### Iteration 9 -- pure deface as the default, leak recorded as an accepted residual (Brendan, 2026-10-22)

- **Decision (Brendan, 2026-10-22):** "keep deface. deface only doesn't work amazingly when
  the face is occluded, so it's fine." Asked again after side-by-sides of the middle ground
  (deface plus the body-based blur only where deface finds no face) against pure deface:
  "Use pure deface with the leak rate recorded as an accepted residual in the PR and
  quality-loop log."
- **The measured residual, stated plainly:**
  - **Detector misses:** on own shadowboxing footage, deface (CenterFace, threshold 0.2)
    found no face in 89 of 2353 frames of IMG_4406 (3.8%) and 843 of 2659 of IMG_4395
    (31.7%; 724 of the 2540 frames with a person, 28.5%).
  - **Visible faces among the misses:** in contact sheets of 18 sampled no-face frames per
    clip, about 6 per clip showed the face substantially visible and unblurred (looking
    down mid-combination, profile while punching, half behind a glove); the rest were
    occluded, turned away or out of frame.
  - **Estimated leak rate (eyeballed sample, not a measurement of every frame):** roughly
    1% (IMG_4406) and 9% (IMG_4395) of frames with a person show an unblurred, visible
    face.
  - **Unmeasured:** sparring footage, with more profile, motion blur, two boxers and
    clinches.
  - **Disagreement on record:** the reviewer's assessment differs from the stated reason. Not
    all misses are occluded faces; some are visible faces in profile or mid-punch.
- **Protocol note (stated once, per the provenance/disclosure rule's spirit):** amendment
  O57 promises the second annotator face-blurred clips only. A visible face reaching the
  annotator is a deviation from the approved procedure; whether that needs reporting to
  the HSRRC is Brendan's call with Hector.
- **Change:**
  - `--face-policy {deface (default), agree, trust}`. deface means `plan_faces_only`: every
    CenterFace box grown 1.3x and mosaicked, and a frame with no detection is left
    untouched.
  - A detector error or malformed output still blurs the whole frame. No YOLO is needed.
  - The report counts `frames_without_face` and lists `no_face_ranges`; the CLI prints them
    on every run and warns above 30%.
  - **Review F1:** a detector that (nearly) never fires is broken, not the accepted miss
    rate. Over 50% faceless frames is refused before anything is written, unless
    `--allow-no-faces` (recorded). `--face-thresh` above 0.5 is refused under deface, and
    the threshold is recorded in the report.
  - `MODE_CODES` replaces first letters, because "faces" and "full" collided.
  - agree/trust (the earlier fail-closed paths, and the middle ground) remain as options.
- **Test:** 177 checks. New for the deface policy:
  - faces mosaicked, nothing else touched;
  - a no-face frame passes through unchanged, counted (the residual, tested so it stays
    visible);
  - errors and malformed output blur the whole frame;
  - a never-firing, a one-hit and a raise-once detector are all refused; 25% faceless
    passes;
  - the threshold cap; no YOLO needed.
- **Review:** FIX FIRST twice (F1, then its one-hit near-miss), then the remaining change
  was the fraction-based refusal, applied. agree/trust are unchanged from the approved
  round.
- **Real check (own footage, default policy):** IMG_4406 3.8% no-face, IMG_4395 31.7% (under
  the 50% refusal, over the 30% warning). At deface's miss frames the output is
  pixel-for-pixel the same residual as the deface CLI (sheet compared). These runs started
  on the code before the F1 refusal, which changes only refusals and report fields, not
  pixels.

## `feat/corner-colour` -- 2026-10-08

Identity by corner colour. Olympic boxers wear their corner's colour (red or blue), so a per-fragment kit colour
could say WHICH boxer a fragment is (the pose-only linker swapped the boxers once on GH078416) and that a
well-seen fragment with neither colour is not a boxer. **Status: the colour test FAILED its own non-circular
self-check on the real data (pooled opposite share 29.1%, gate 90%), so the work stopped there, as the brief
required. `appearance.py`, `olympic_colour.py` and their tests are committed. The linking integration
(`link(..., colours=)`, `olympic_link.py --colours`, the colour overlay, `olympic_completeness.py --colours`) was built
and tested, then WITHDRAWN from the branch: it is not committed and no linking run with colours was made.**

- **Acceptance:** (1) from painted synthetic frames the decisions are exact (red vest, blue vest, white shirt,
  skin, pale pink on skin, both colours); (2) the real-data gate asserts a fact that can fail: two credited
  fragments on screen together are two people, so they must get OPPOSITE colours; the labels never say red or
  blue, so this cannot pass by construction; (3) only numbers are kept, no pixel of the footage is written.
- **Threat sketch:** asset: the per-fragment colour decision that would constrain which fragments may form a
  boxer identity. Entry points: the video (untrusted pixels, decoded by `video.iter_frames`), the tracks `.npz`,
  the `.colours.json` file (an input to the linker), a GoPro id from the command line. Top threats: (a) a colour
  test that confidently labels the wrong thing (skin, a spectator's shirt) and then silently decides identities
  (this is what happened: see below); (b) a colours file from another table or another recording applied to
  this one. Mitigations: the self-check gate before any use; `Colours` is bound to the table (source name,
  n_frames, fingerprint when present, the exact set of track ids) and refused otherwise; the loader validates
  strictly (no NaN, known keys, shares in [0, 1], a decision must be the one its own medians imply).

### Design (`src/boxingvi/appearance.py`, `olympic_colour.py`)
- Rows read: every 5th frame (10 Hz at 50 fps) plus each fragment's first and last row with a confident torso.
- Regions per row (joints >= 0.3): torso quadrilateral (shoulders + hips) shrunk 20% to its centroid; head disc
  at the mean of the confident face keypoints; a glove disc at each confident wrist; discs 0.35 torso lengths
  (the box height x 0.3 when the four torso joints are not all confident). A region under 30 pixels is skipped;
  the regions are pooled as a union (a pixel counts once); pixels outside the frame are ignored.
- Per pixel in HSV (numpy): red = hue <= 12 or >= 340 degrees, blue = 200..255, both with saturation >= 0.45 and
  value >= 0.20. Per fragment: the MEDIAN over its rows of the red and the blue share; red / blue when the median
  is >= 0.10, >= 3x the other and >= 3 rows; none when >= 5 rows and both medians < 0.02; else unsure.
- Constants and reasons are beside each constant in `appearance.py`; all were fixed from colour physics before
  any colour result on the footage was looked at, and NONE was changed afterwards (the diagnostic below was read
  AFTER the gate failed and changed nothing). Honest caveat: a one-video trial run (GH078416) was read before
  the full run; it led to no change either.
- `olympic_colour.py` decodes each video once, writes `<tracks>/<id>.colours.json` (decision, medians, rows per
  track id; `allow_nan=False`), refuses an unbound tracks table (name, fingerprint), then runs the self-check and
  writes `results/olympic_colour_check.json`; exit 2 when the pooled opposite share is below 0.9, 1 when a video
  failed. `--check-only` recomputes the check from existing colour files.

### The self-check, real data (29 videos, all results gitignored)
| | |
|---|---|
| fragments coloured | 15,029 (red 3,606, blue 546, none 4,522, unsure 6,355) |
| credited fragments | 1,245 (red 436, blue 68, none 212, unsure 529) |
| co-present (>= 0.5 s) credited pairs | 1,292, of which both decided red/blue: 275 |
| **opposite / same** | **80 (29.1%) / 195 (70.9%)**, gate >= 90%: **FAIL** |
| by camera | kam2: 10 decided pairs, 3 opposite; kam4: 265 decided, 77 opposite |
| wall time | one video (3,186 frames) 24 s alone; all 29 in four parallel shards 23 min (shards 975 / 1,010 / 1,299 / 1,336 s) |

Two boxers on screen together are called the SAME colour 71% of the time and the test calls far more fragments
red than blue (3,606 vs 546; 436 vs 68 among credited ones). Credited fragments are mostly unsure (529) or
none (212): the kit does not reach MIN_KIT = 0.10 of the pooled region for most boxers.

**Post-hoc diagnostic (numbers only, read after the gate failed, NOT used to change anything).** For the credited
rows of GH078416 (kam2) and GH159681 (kam4): the share of region pixels by saturation band and 30 degree hue bin.
In both videos the hue 0-30 bin at saturation >= 0.45 holds 47% / 33% of the torso pixels, 34% / 36% of the
HEAD disc and 62% / 65% of the glove discs. The head disc is face skin plus headgear (kit-coloured headgear would fall in the same bin, so this is suggestive, not proof), so under this footage's colour grading
skin is highly saturated and warm-hued: the saturation floor of 0.45 does not separate skin from red kit, and the
red band (hue <= 12) takes part of the skin. Genuine blue (hue 210-240, saturation >= 0.45) is present but small
(torso 4% / 2%, gloves 1% / 9%), under MIN_KIT. In GH159681 a second, separate mass sits at hue 330-360 with
saturation >= 0.45 (torso 25%, gloves 23%): red kit with a magenta-ish hue that the 340 degree edge only partly
catches and that skin (hue 0-30) does not share. So the constants as pre-registered fail on this footage for a
specific, visible reason; a redesign (a hue band that excludes skin, per-video kit calibration from the
pooled hue distribution, or reading the vest only) is a decision for the author, and would have to be
pre-registered the same way and re-run through the same gate. None of that was done here.

### Tests (synthetic only; `tests/test_appearance.py`, 118 checks)
HSV against `colorsys` on 3,000 random pixels; every threshold edge (hue 340 / 12 / 200 / 255, saturation 0.45,
value 0.20); the torso shrink (10..90 of a 0..100 square holds exactly 6,400 pixels); disc areas; clipping at all four
frame edges and a fully off-frame figure; a wild 10-million-pixel keypoint; regions under 30 pixels skipped;
painted red / blue / white-shirt / skin / pale-pink-on-skin figures through the whole pipeline; median robustness
(2 of 7 rows full of blue stays red; the mean would be unsure; 4 of 7 flips it); rows thresholds 2 / 3 / 4 / 5;
which rows are read; `Colours` save/load with 17 refusals (malformed JSON, NaN, Infinity, unknown decision, a
decision its medians do not imply, shares above 1, negative / boolean rows, extra keys, bad ids, oversize file) and
the binding (source, n_frames, fingerprint, ids); the self-check (opposite -> 1.0, same -> 0.0, 0.5 s exactly is a
pair and 24 frames is not, none / unsure are not decided pairs, no credited punch gives None); the CLI end to end
on a painted clip, including the gate FAILING (both boxers called red -> opposite 0.0, exit 2), a renamed table,
another recording's fingerprint, path-like ids, a missing colours file.

### Mutations
| Guard reverted | Result | First failing checks |
|---|---|---|
| median -> mean (`appearance.fragment_colours`) | caught, 2 in `test_appearance` | median robustness; 4 of 7 rows blue |
| saturation floor removed (`SAT_MIN = 0`) | caught, 10 in `test_appearance` | white shirt reads red 25%; skin; the 0.451 / 0.447 edge |
| colour constraint removed (red / blue allowed in both corners) | caught, 13 (withdrawn linking tests) | swapped colours; all-red leaves blue empty |
| `none` allowed in a corner | caught, 6 (withdrawn linking tests) | stepping stone; excluded count |
| capacity shared across corners removed | caught: the suite aborts in the solver's own consistency check ("the solution leaves one fragment twice") | one boxer, every fragment unsure |

The last three were run on the withdrawn linking code before it was taken out of the branch; they are recorded because
they show the tests had teeth, not because that code ships.

### What was built and withdrawn (not committed)
`linking.link(..., colours=)` duplicated the layered path model per corner (chain 0 red, chain 1 blue; one chain per
corner; capacity shared; red only in red, blue only in blue, unsure in either, none in neither), `olympic_link.py
--colours`, the colour overlay (red / blue skeletons, yellow border), `olympic_completeness.py --colours`; with 335
checks in `test_linking.py` (266 existing unchanged + 69 new, including a 500-instance brute force of the corner model
and the 00:24 crossing: no swap with colours). They passed; they are not on this branch because building on a colour
test that fails its gate was ruled out. The `solve_chains` change kept variable and row order identical when
`corners is None`, but that equivalence was only exercised by the existing 800-instance brute force, not by a
byte comparison.

### Residual risks
Skin saturation under this grading (the failure above); spectators, coaches or judges in a red or blue top are
red / blue like a boxer; coloured lighting moves every hue; a clinch fills a box with the other boxer's kit (the
median absorbs a few rows, not a long clinch); the colours were checked by the self-check, not by eye (and no
Olympic frame was viewed); fragments of a few frames have 2 rows and are always unsure; pairs between the same two
boxers are not independent, so the 275 decided pairs are far fewer than 275 independent observations.

## `feat/corner-colour`, version 2 -- 2026-10-08 (rule FROZEN in this commit, before any test video was read)

Version 1 failed its self-check (29.1% opposite, 80 of 275 decided pairs). Version 2 redesigns the colour test and
re-asks the same question with a clean split. **Protocol, fixed by video before any v2 design:** DEVELOPMENT videos
GH078416 and GH159681 (already inspected after v1 failed) plus GH128416 (kam2) and GH069681 (kam4), drawn with
`random.Random(20261008)` from the rest of each camera; every other video (13 kam2, 12 kam4) is TEST. No colour
statistic, decision count or self-check number of a test video was computed, printed or read until this rule and every
constant were committed. The v1 colours files of the test videos sat in `results/` and were not opened. Only the four
dev videos were extracted before this commit (`olympic_colour.py` defaults to `--split dev` for that reason).
The gate: on the 25 test videos pooled, co-present (>= 0.5 s) credited pairs both decided red/blue, opposite share
>= 0.90 AND >= 100 decided pairs; per camera, < 20 decided test pairs means "colour not established" there.

### The rule (`appearance.py`; reasons beside each constant)
| | v1 | v2 | why (dev data) |
|---|---|---|---|
| region | torso + head disc + 2 glove discs, pooled | torso only | the head disc includes background (red and blue corner pads); gloves are mostly arm; no combination beat torso-only on the self-check |
| red | hue <= 12 or >= 340, sat >= 0.45, value >= 0.20 | hue [340, 25), sat >= 0.70, value >= 0.20 | skin is hue 15-40 with saturation up to ~0.6 (read as red by v1); red kit is hue 10-20 (kam2) / 350-5 (kam4) at saturation 0.7-1.0; non-red fragments have 0.000-0.009 of the torso there, red-kit fragments 0.3-0.7 |
| blue | hue 200-255, sat >= 0.45, value >= 0.20 | hue [200, 290), sat >= 0.50, value >= 0.08 | the blue boxer's vest is DARK navy (value 0.08-0.2, hue 220-240, sat ~0.9); v1's value floor threw it away. Credited fragments called blue on dev: 27 with floor 0.20, 54 with 0.08 |
| decision | median share >= 10% and >= 3x the other | per row the kit index (B - R) / (B + R) over kit pixels; a kit row has R + B >= max(20 px, 5% of the torso); per fragment the median index over kit rows: >= 0.5 blue, <= -0.5 red (one colour at least 3x the other); needs >= 3 rows, >= 3 kit rows, kit rows >= half of the rows | relative, so a navy vest that is 7% of the torso still counts; a fixed share floor does not |
| none | both medians < 2%, >= 5 rows | median kit share < 2%, >= 5 rows | unchanged in spirit |
| unsure | everything else | everything else | |

Kept from v1: sampling every 5th frame plus each fragment's first and last row with a confident torso, the torso shrunk
20%, min confidence 0.3, min region 30 px, median over rows, the binding of the colours file to its table (format is
now `boxingvi-colours-2`, strict loader). No per-video adaptive threshold: the dev kit index is bimodal at about -1 and
+1, and the decisions are flat for thresholds 0.2-0.5. The colours file stores `index`, `kit`, `rows`, `kit_rows` per
track id.

### Development numbers (4 dev videos; NOT evidence, they were looked at as much as wanted)
Stability on dev (production code, 192 credited fragments): red 105 / blue 53 / none 19 / unsure 15. One-at-a-time
changes of the constants moved red 100-111, blue 47-62, none 10-31, unsure 9-37 (red saturation floor 0.6-0.8, red hue
ceiling 15-35, blue saturation floor 0.4-0.6, blue hue band 190-300, ROW_KIT_MIN 0.02-0.12, index threshold 0.2-0.8,
kit-row fraction 0.3-0.7); only the blue value floor at 0.20 (the v1 value) was a cliff (blue 27, none 44).
Dev self-check at the frozen rule: 146 decided co-present credited pairs of 221, **opposite 73 (50.0%)**:
GH078416 4 of 6, GH128416 6 of 9, GH069681 41 of 60 (68.3%), GH159681 22 of 71 (31.0%); kam2 15 decided pairs, kam4
131. Dev wall time 197 s for the 4 videos (38,137 frames).
Region and fusion experiments on dev (decided pairs, opposite share): torso 148, 50%; head 107, 64%; gloves 108, 44%;
torso+head pooled 122, 53%; torso+head+gloves 121, 50%; head-then-torso 135, 64%; torso and head must agree 80, 66%.
None is near 90%, and the differences are within what about 100 pairs from 4 videos can resolve, so the simplest
rule was frozen rather than a fitted fusion.
Credited fragments called `none` on dev: 19 of 192. For 13 of them (all 10 on kam4, 3 of 9 on kam2) a decided red AND
a decided blue fragment were also on screen with them for >= 25 frames, which is what a third person (a referee)
looks like; they carry 1-4 credited punches each. The other 6 are on kam2, where blue is rarely decided, so that check
cannot say.

### What dev says about the premise (found while designing; relevant to reading the test result)
In GH159681 from about 90 s on, nearly all credited fragments have a red torso (credited red 56, blue 15 in the
video), and 49 co-present credited pairs have the same colour with both fragments' median kit share >= 0.25. Looking
at positions only: two boxer-sized boxes (height 400-600 px) 100-250 px apart, on screen together for up to 16 s,
both with about 70% of the torso red and red gloves (e.g. ids 2315 and 2329 at 236-252 s); of the 49 pairs, in 11 one
head region is decided blue and the other red, and in 30 more one head is undecided with a blue-leaning index (the
blue share of those discs is 0.12-0.29). So in this bout the torso (and the gloves) do not separate the boxers. No
pixel was viewed; this is counts. If the test videos behave like this, the opposite share is capped below 90% by the
footage and not by the colour rule; the gate is reported as it comes out.

### Tests (synthetic only; `tests/test_appearance.py`, 171 checks, was 118 for v1)
Replaced: every v1 threshold-edge assertion (saturation 0.45, value 0.20, hue 12 / 255), the share-based `decide`
boundaries (MIN_KIT 0.10, RATIO 3, NONE_MAX on both shares), the all-regions row-share checks, the v1 `Colours` keys
(`red` / `blue`), the CLI test's single `passes` flag. Added: every v2 edge (red hue 340 / 25, saturation 0.70, value
0.20; blue hue 200 / 290, saturation 0.50, value 0.08), the skin band (skin, hot skin, pale pink are not red; the v1
floor shown to read hot skin as red), the navy vest (and the v1 value floor shown to lose it), black, torso-only (red
headgear and gloves on a white vest read none; the same figure reads red with all four regions), `use=` in
`region_masks`, kit-row boundaries (19 / 20 px, 2% / 5%), split vests with the index counted from the torso mask (80%
blue is blue; half and half, and blue 2.3x red, are unsure), patches of 9 / 45 / 80 px (speck / weak / kit), median
robustness, kit rows 3 of 7 vs 4 of 7, row thresholds, 25 loader / saver refusals (index range, kit_rows > rows, index without
kit rows, v1 file and v1 keys, ...), the dev/test split constants, `--split` default dev (a test video is not read),
a dev-only run does not evaluate the gate (passes null), the 100-pair requirement (a perfect 1-of-1 pair fails it),
per-camera "not established", and the gate failing when both boxers are called red (exit 2).

### Mutations (27, each run on a copy of the code with only `tests/test_appearance.py`)
All 27 are caught (failing checks): skin band included (red saturation 0.45 + hue ceiling 45) 6; absolute share floor
(a kit row needs 10% of the torso) 3; blue value floor back to 0.20 14; head and gloves read too 13 (plus a failed
assertion); median -> mean for the index 2, for the kit share 1; index margin removed 6; kit rows need not be half
2; MIN_ROWS 1: 3; NONE_ROWS 1: 2; NONE_MAX 0.5: 3; kit-row pixel floor removed 1; torso not shrunk 12; red hue wrap
broken 10; blue hue band 150-330 2; blue saturation floor removed 2; loader skips the implied-decision check 1; loader
accepts kit_rows > rows 1; co-presence needs no time 1; undecided counted as decided 1; gate 0.9 -> 0 2; no minimum
pairs 2; no camera minimum 2; default split all 1; split inverted 2; dev list changed 1; gate evaluated on dev numbers
1. (A first run left "blue value floor" uncaught because the edit had hit the constant's name in the docstring; the
harness now edits code lines only and the corrected run is the one reported.)

All nine suites pass at this commit: pipeline 260, ingest 124, env 36, labels 50, hands 38, olympic 33, tracks 79,
linking 266, appearance 171.

### TEST RESULT (run after the freezing commit 0ea8472; 25 test videos, nothing tuned afterwards)
Extraction on the 25 test videos in four parallel shards (633-753 s each; about 12.5 min), then the pooled check from
the colours files. **The gate FAILS.**
| | decided co-present credited pairs | opposite | same | opposite share |
|---|---|---|---|---|
| **test, pooled (gate: >= 90% and >= 100 pairs)** | **582** (of 1,071 co-present) | **304** | 278 | **52.2% -> FAIL** |
| kam2 (13 videos) | 196 | 82 | 114 | 41.8% (established: >= 20 pairs) |
| kam4 (12 videos) | 386 | 222 | 164 | 57.5% (established) |
| dev, for reference (not evidence) | 146 | 73 | 73 | 50.0% |
The test result is the dev result (50%) to within the noise of ~600 pairs: the chance level, not a near miss. The
count requirement (>= 100 pairs) is met, so the failure is the opposite share alone. Videos with at least 20 decided
pairs range from 0% (GH178416: 60 pairs, no blue fragment among the credited ones; GH098416 is the same with 14 pairs)
and 19% (GH079681, 42 pairs) to 81% (GH088416, 26 pairs); none reaches 90%.
Credited fragments on test (1,053): red 494 / blue 318 / none 175 / unsure 66. The 278 same-colour pairs are 209 red-red
and 69 blue-blue, and 191 of them (69%) have a median kit share >= 0.25 in BOTH fragments, i.e. both are confidently
kit-coloured, not borderline. Credited `none` is 16.6% of credited fragments; for 107 of the 175 a decided red AND a
decided blue fragment were also on screen with it for >= 25 frames (a third person, as on dev), the other 68 cannot be
checked that way (blue is rarely decided on kam2).

### Reading it
The colour rule does what it was designed to do at the level of single fragments (dev: bimodal index, stable
under every one-at-a-time change of the constants, `none` mostly a third person), and the two decision counts (red 494
vs blue 318 credited) show the excess of red that dev showed. What it cannot do is separate the two boxers of a bout
in the pairs that matter: 278 of the 582 decided pairs are two people with the same torso colour (red-red 209), and
for 191 of those both are confidently kit-coloured. Two hypotheses, which the counts cannot tell apart (the footage was
not viewed, by rule): both boxers of those bouts really wear the same vest colour, or some credited punches belong to
a person of the same corner as the boxer (a second, a spectator at the ropes) rather than to the opponent. Either
way the premise "two boxers on screen together have opposite torso colours" does not hold for this dataset well
enough to constrain a linker, and a rule tuned on 4 videos cannot fix a premise.

### Status and what was not done
- The gate failed, so, as briefed: the frozen rule and this result are committed, NOTHING was tuned on test videos, and
  the linking integration (`link(..., colours=)`, `olympic_link.py --colours`, the colour overlay, the
  `--reacquire` runs, the GH078416 timelines, the before / after table, completeness, the mutation checks for the
  linking) was NOT re-applied. `linking-integration-NOT-COMMITTED.patch` stays in the scratchpad.
- Colour is "established" in the brief's sense for both cameras (>= 20 decided pairs each), but established means
  measured, and what was measured is chance level, so no camera's colour is usable as a constraint.
- Next option, Brendan's call: learned re-ID (BoT-SORT with appearance embeddings) on the GPU laptop. Cheaper
  questions first, if wanted: (a) whether both boxers really share a vest colour in the same-colour bouts (look at ONE
  such pair of frames by eye; this session's rules forbid viewing footage, so it needs you), since the whole premise
  rests on that; (b) the head region alone gave 64% on dev (107 pairs), better than the torso but far from 90%, and
  was not tried on test because the rule was frozen.
- Files not committed (gitignored `results/`): `olympic_colour_check_v2.dev.json`, `olympic_colour_check_v2.test.json`,
  and the 29 `<id>.colours.json` (v2 format, replacing v1's).
