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
