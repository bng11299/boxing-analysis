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
- **Review (Astra / Codex, read-only, effort medium, one review of the #17/#18/#19 merges): SHIP all.** Merge OK on each branch, no adaptation defects; it checked that the shared checks are identical to feat/output-report's on all three (confirmed here: no changed line). Its sandbox could not run Python, so its verdict is from reading plus the results above; fresh clones of 28a2a86, 0b492c0 and 549dab2 reproduce every count.

