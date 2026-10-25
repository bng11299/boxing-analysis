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

