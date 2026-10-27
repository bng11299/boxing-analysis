## label-moments -- 2026-10-27: a labelling window for the glove-identity gate CSVs

Branch `feat/label-moments` from `main` (`8b25ebf`). Independent of #31: it edits the CSVs #31's `template`
writes and #31's `parse_labels` reads, and imports nothing from that branch.

### Setup
**Task.** Brendan fills three labels CSVs (193 rows; L / R / ? every 5 s, `# A:`, `# minutes:`) by
watching the BLURRED sparring videos. Make that fast and hard to get wrong, without ever showing the glove
method's output.

**Acceptance criteria.**
1. `tests/test_label_moments.py` passes on synthetic videos and CSVs only, with named negative tests for: an
   unblurred `PXL_*` path, a video outside `blurred/` (path, `..`, junction), a name in `# A:`, rows past the
   end, an interrupted session resuming with nothing lost.
2. The frame shown as "the moment" is stored frame round(t x fps), proved on H.264 with B-frames at 30 and
   29.97 fps by frames that spell their own number.
3. Only `# A:`, `# minutes:` and the a_side values change; every other byte survives (LF, CRLF, BOM).
4. A file this tool fills passes #31's own `glovescore.parse_labels` (real fact, checked against the branch).
5. `--check` passes on the three real pairs and refuses a mismatched pair, without a window and without
   changing the CSVs; nothing opens the method's `.npz` / `.log`.

**Threat sketch.** Assets: the independence and correctness of Brendan's labels (they are the gate's ground
truth); participant footage and anything derived from it; participant names. Entry points: the CSV and the
video paths on the command line, the CSV's contents, the video file, the `# A:` prompt, key presses.
Trust boundaries: every file read from disk; the keyboard (auto-repeat). Top threats: (1) a wrong label
written silently -- the wrong video, a moment that was not shown, a held key, a truncated or reordered file
-- which corrupts the gate without any error; (2) participant data or a name leaking -- an unblurred
original shown, a name typed into `# A:`, labels written outside `~/comps-participant-data/`.

### Iteration 1 -- build, test gate, two reviews
- Built `label_moments.py` (CLI, Tkinter window: the frame at t in a red border, then t-1..t+1 s on a loop),
  `src/boxingvi/labelmoments.py` (checks, CSV editing, session, clip decoding by ffmpeg), the test script.
  Tkinter, not OpenCV: base Python has no cv2, so the tool adds no dependency.
- Test gate: 121 passed, 1 skipped (symlink creation needs Developer Mode; a junction test covers the same
  escape). `tests/test_pipeline.py` 314 passed (unchanged).
- Real facts: (a) #31's `write_template` -> this tool -> #31's `parse_labels`: accepted, A, minutes and all
  70 values as written; a half-filled file is refused by the gate. (b) `--check` on v3/v2/v1: ok, 70/61/62
  empty rows; v3's CSV against v2's video refused (fingerprints differ); CSVs unchanged (10-26 timestamps).
  Late-moment decode ~0.5 s per clip on the real 1080p files.
- Review, Astra (read-only, code only, medium): **FIX FIRST**, 7 blocking + 1 should-fix. Code-reviewer
  subagent: **FIX FIRST**, 2 blocking + 5 should-fix + nits. Merged:
  1. a held key answered several rows (dwell only delayed auto-repeat) -- both;
  2. a failed decode still accepted L/R, old picture on screen -- both;
  3. `00,`, `05,` and non-ASCII digits loaded and were rewritten as other bytes -- both;
  4. a truncated CSV passed and `--verify` said COMPLETE; the gate refuses it -- both;
  5. `Gray Brown` passed the name filter: colour words are surnames too -- Astra (reviewer as a nit);
  6. the video could be swapped between check and decode -- Astra blocking, reviewer accept-and-note;
  7. a failed save on quit closed the window and printed "saved" -- Astra;
  8. `mkstemp` outside the guarded save -- Astra; a short decode would move the red frame; probe skipped
     `video.probe`'s VFR and rotation handling; pending decodes never evicted, `.result()` could freeze
     the window -- reviewer.
- Own security pass: subprocess argument lists throughout, no shell; ffmpeg/ffprobe could follow a
  non-file protocol in a crafted input (fix: `-protocol_whitelist file`); an unexpected `OSError` would print
  a traceback with paths (fix: caught, generic message).

### Iteration 2 -- fixes, test gate, re-review
- Fixes (`9bea291`): an answer needs the previous answer's key released (a release followed by a press
  within 60 ms is auto-repeat, which covers Windows press-only and X11 release+press repeat); a failed decode
  clears the picture and accepts only `?`; canonical ASCII times only, other `splitlines` breaks refused;
  rows must equal `glovescore.label_times` exactly, from the decoded frame count of `video.probe`'s timing
  scan, and VFR video is refused; `# A:` is generated from glove colours chosen off a fixed list (free text
  dropped); `VideoGuard` holds the video open (Windows then refuses to replace or rename it) and re-checks
  identity before each decode; a failed save on quit keeps the window open; `mkstemp` inside the guard; a
  short decode is refused; pending decodes evicted, decoding polled with `after()`; `-protocol_whitelist file`.
- Test gate: 152 passed, 0 failed, 1 skipped. New negative tests for each finding (held key and auto-repeat,
  failed decode, short decode, `00,` / `05,` / U+0665 / U+2028 / form feed, truncated CSV, VFR video, a
  video swapped after its check, a closed guard, `mkstemp` failure, failed quit) and a 29.97 fps
  frame-accuracy test.
- Real facts re-run: #31 compatibility as before; `--check` on the three real pairs ok, so all three have
  regular frame timing and their measured frame counts give exactly 70/61/62 grid rows, as #31 wrote.
- Re-review (code-reviewer, same context, on `9bea291`): **SHIP**; all seven of its findings closed, no new
  blocking issue; it re-ran the suite (152 / 0 / 1). Two nits, both taken (iteration 3).

### Iteration 3 -- the reviewer's nits, a rendered look, the exit gate
- Fixes: only the release of the key that gave the last answer re-arms (letting go of Shift while still
  holding `/` answered the next row as `?`); any decode error, not only a refusal, shows the cannot-show
  state instead of leaving "loading" on screen.
- Rendered once on a synthetic 1080p clip and screenshotted (not participant footage): red border on the
  held moment (ffmpeg testsrc timestamp 00:00:00.000 = frame 0), grey while playing, all text legible. One
  change from reading it: "left" now says "left of the picture" (the gate's left is image left, not the
  boxer's own left).
- Test gate: 157 passed, 0 failed, 1 skipped (new: Shift-release, a non-refusal decode error).
- Review budget: one Astra review by instruction; the re-review was the code-reviewer subagent.

### Residual risks, accepted
- Auto-repeat detection assumes a real release and the next press are more than 60 ms apart.
- `video.probe` (repo code, unchanged) runs ffprobe without `-protocol_whitelist`; inputs reaching it have
  already passed the path checks.
- Someone with write access to `~/comps-participant-data/` could edit the CSV between sessions; the tool
  re-validates it on every start, and the gate parses it again.
- Labels the operator gives are trusted as given: the tool cannot tell a careless answer from a careful one.
