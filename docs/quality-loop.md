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
both with about 70% of the torso red and red gloves, and in about a third of those pairs one of the two with a partly
blue head region (0.12-0.29 of the disc). So in this bout the torso (and the gloves) do not separate the boxers. No
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
