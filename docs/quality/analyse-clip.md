## `feat/analyse-clip` -- 2026-10-19

`analyse_clip.py`: the Phase 1 deliverable as one command. Video in; pose, punches CSV,
classified CSV + sidecar, report .xlsx, summary .txt and overlay out, all as `<clip>.*` under
one `--out-dir`. Stacked on `feat/output-report`.

**Acceptance criteria.** (1) The four stages run through their `main()`s (no subprocess), the
overlay through one function `render_overlay` (the seam for `feat/punch-overlay`). (2)
`--stance` is required, never defaulted. (3) A failing stage stops the chain with a message
naming it; no stage reads a file from an earlier run; classify's sidecar hashes are checked
before the report runs; `--out-dir` is never left holding two runs mixed. (4) `--skip-ingest`
reuses a pose only if its source name, frame count, fps, size and model match the video. (5)
On IMG_4406 the outputs equal those made stage by stage (same classified CSV hash).

**Threat sketch.** Assets: participant footage and its pose; the honesty of the report.
Entry points: the video, a reused pose file, CLI arguments that become stage argv. Trust
boundary: files on disk between runs and between stages. Top threats: a stale or mismatched
intermediate (an old punches CSV, another clip's pose) reported as this clip's; a clip name
or path that a stage reads as an option, or that scatters or deletes files outside the
intended names.

**Design.** Stages write into a fresh `mkdtemp` folder inside `--out-dir` (so footage-derived
files never land in the system temp), and each stage's outputs must exist before the next
runs. Publishing is renames only: the old set is moved aside, the new set moved in, and any
failure (a file open in Excel, Ctrl+C) undoes every move made so far. A pose the model just
extracted and checked is kept when a later stage fails. Every path given to a stage is
absolute and option values use `--opt=value`; the clip name may not contain a dot or start
with `-`.

### Iteration 1
- **Test:** `tests/test_analyse_clip.py` 83 checks, synthetic video and an injected predictor
  (no GPU, no ultralytics). Stage-by-stage equivalence is asserted inside the suite (byte-equal
  CSVs, sidecar and summary; equal workbook cells: openpyxl stamps write times, so .xlsx bytes
  differ run to run).
- **Mutations:** 18 guards reverted, 16 caught. fps and frame size survived: those poses were
  still refused, but later, by the overlay, so the tests could not tell which gate refused.
- **Review (code-reviewer): FIX FIRST.** B1: publish deleted the old set before moving the new
  one in, so a report open in Excel lost both runs (and a freshly extracted pose). B2: the
  frame-count gate was proven on synthetic video only. Non-blocking: a publish failure was
  told apart by string matching; raw tracebacks from `mkdir`/`mkdtemp`, `probe` timeouts and
  the pose hash; a leading `-` in the clip name; `--weights-sha256` unvalidated; the model is
  matched by file name only; untested branches (full-run failure, the sidecar's pose length,
  overlay not run after report's refusal); a no-op edit in one test.
- **Real-world fact for B2:** on both participant clips the pose length equals the decoder's
  count and the container's (IMG_4395 2659, IMG_4406 2353), neither VFR.

### Iteration 2
- **Fix:** all of the above. Publish is renames with undo; `PublishFailed` carries what
  `--out-dir` holds; the fresh pose is kept; clean messages for every escaped error;
  argv hardening; skip tests require the refusal to come from the ingest stage.
- **Test:** 93. Mutations: 25 guards reverted, all caught (fps and size now included).
- **Re-review: SHIP.** Remaining, fixed after it: a failed tidy-up could still lose the kept
  pose; Ctrl+C during publish left the old set in a folder about to be deleted; a directory
  named like an output would have been moved and deleted; the Windows-only lock test skipped
  silently. 96 checks; the 3 new guards reverted, all caught.
- **Security:** no new subprocess (ffprobe via `probe`, argument list). Stage argv: absolute
  paths, `=`-form option values, each validated (`--device`, `--weights-sha256`, `--rounds`,
  `--stance` from a fixed list). Deletion touches only `<out-dir>/<clip>.<fixed suffix>` (moved
  aside, never unlinked) and the `mkdtemp` folder it created; a folder with an output's name
  is refused. Errors name paths on the local machine only. No secrets.
- **Real check (IMG_4406, `--skip-ingest --stance orthodox`, 45 s on CPU):** punches.csv,
  classified.csv (`d82a679e...`), classified.json and summary.txt byte-identical to the stored
  stage-by-stage outputs; workbook cells equal; overlay 2353 frames at 59.946 fps. A second run
  publishing over the first gave the same hashes, the overlay's included.
- **All suites:** analyse 96, pipeline 260, ingest 109, classify 202, report 130, hands 38,
  labels 50, env 36.

- **Gate reproduce:** clean clone of `7710557`: analyse 96, pipeline 260, ingest 109, classify
  202, report 130, hands 38, labels 50, env 36 -- the working tree's numbers.

### Residual risk, accepted
- `--skip-ingest` matches the pose model by its weights' file name; the pose file records no
  hash. A same-named, different checkpoint passes.
- The fresh-ingest path (`load_yolo` + the injected-predictor seam) is exercised here only
  with an injected predictor; the real YOLO run is the GPU laptop's.
- The punch-labelled overlay (`feat/punch-overlay`, PR #18) is not swapped in: two lines in
  `render_overlay` once it is merged into the base.

### Iteration 3 -- feat/output-report's four fixes merged in (2026-10-21)
Acceptance: (1) the preflight refuses a manifest without the pinned held-out matrix, and a
transposed reference, before the pose model runs; (2) the sidecar check refuses a sidecar that
differs only in fps, the way the report does; (3) a sidecar recording another finder rule stops
the chain at the report stage with out-dir untouched; (4) every suite green.
- **Merge:** one conflict, this log (output-report's iteration 5 placed in its own section). Code:
  `preflight` passes the rule it loads to `finder_reliability(a.rule, rule)` (the finder runs with
  it; the report stage compares it again with the sidecar's); `check_classified` calls
  `report.check_pose` (exact, overflow-safe) under its own "was classified on another pose"
  prefix. Its old check compared fps exactly too, so this unifies the code, not the behaviour.
- **Test:** analyse_clip 96 -> 107 (fps-only sidecar, a 400-digit fps, another finder rule,
  a pre-10-21 manifest, a transposed reference); report 185, classify 209, env 36, hands 38,
  ingest 109, labels 50, pipeline 260. Mutants: preflight without the matrix check,
  `check_pose` skipped, fps ignored, rule not compared -- 4 of 4 caught.
- **Review / security:** no new input; the duplicate frames/fps check replaced by the shared one.
  No findings.
- **Real-world fact, read-only:** the preflight on the real model files refuses today's manifest
  ("re-run train_classifier.py") before the pose model would run: the one-command run stops at
  its first step until the classifier is retrained.
- **Review (Astra / Codex, read-only, effort medium, one review of the #17/#18/#19 merges): SHIP all.** Merge OK on each branch, no adaptation defects; it checked that the shared checks are identical to feat/output-report's on all three (confirmed here: no changed line). Its sandbox could not run Python, so its verdict is from reading plus the results above; fresh clones of 2e2059f, f2412e8 and a2f90be reproduce every count.
