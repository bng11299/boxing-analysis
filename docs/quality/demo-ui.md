# `claude/demo-ui-e10778` — 2026-11-12

Task: a local Streamlit demo for `analyse_clip.py` (Gallery of finished clips; Analyse-a-clip upload tab), with the logic in `src/boxingvi/demo.py`, importable without streamlit.

Acceptance criteria (each checked by a command or an inspection):
1. `streamlit run` serves on 127.0.0.1 only; the Gallery shows both finished clips with a playable web-preview video, the verbatim summary, a chart, a table and working downloads (checked in the browser pane).
2. A clip is listed only with all seven `analyse_clip.OUTPUTS`; uploads are stored under generated names inside the uploads folder; nothing under a participant folder is read, written or deleted (`tests/test_demo.py`).
3. Both subprocesses (analyse_clip, ffmpeg) get argument lists, never a shell; stance is passed exactly and has no default (`tests/test_demo.py`).
4. `tests/test_demo.py`, `tests/test_pipeline.py` and `tests/test_analyse_clip.py` pass, in the working tree and in a clean clone.

Threat sketch:
- **Assets:** the user's own footage and skeletons in `results/`; the participant folder (must never be touched); the laptop's filesystem.
- **Entry points:** an uploaded video (bytes, file name); the `--results-dir` / `--uploads-dir` flags; the rounds text; the files in the results folder (summary, CSV, overlay).
- **Trust boundaries:** browser -> Streamlit server (loopback only); upload bytes -> disk -> ffprobe/ffmpeg/pose model (parsers of untrusted video); app -> `analyse_clip.py` subprocess.
- **Top threats:**
  - a path built from user text escaping its folder (or reaching a participant folder) on read, write or delete; the defence is that no user text is ever a path (generated names, fixed output suffixes), plus resolve-then-compare containment;
  - the server being reachable beyond the laptop; the defence is `server.address = "127.0.0.1"`, asserted by a test.

### Iteration 1
- **Built:** `demo.py`, `demo_app.py`, `.streamlit/config.toml`, `requirements-demo.txt` (`streamlit==1.66.1`; installing it changed no existing pin in the venv), tests, `docs/commands/demo.md`.
- **Test gate:** `test_demo.py` 218/0; `test_pipeline.py` 351/0; `test_analyse_clip.py` 113/0.
- **Browser (built-in pane, `http://127.0.0.1:8501`):** both clips listed; each overlay's web preview plays (406x720; IMG_5835 transcoded in about 90 s to 9.6 MB); the summary on the page is IMG_5835's text; chart header "162 in all"; both downloads return the right bytes (xlsx `PK`, 9,881 B; CSV 2,968 B). Upload path with a 2 KB dummy file named `../../My Friend; rm -rf.MP4`: stored as `upload_<time>_<hex>.mp4`, argv accepted by `analyse_clip`'s parser, run failed at its ffprobe check (exit 2) with the log shown, delete removed the file. A bad rounds string disabled the run button and showed the error.
- **Real fact, by hand:** the summariser on IMG_5835's `classified.csv` gives 162 punches: 41 jab, 39 lead hook, 32 cross, 25 rear uppercut, 20 lead uppercut, 5 rear hook, equal to its `summary.txt`.
- **Mutants:** 19/19 killed (containment, participant refusal, completeness, extension, stance/rounds/device validation, shell=True, delete-name check, verbatim summary, count off-by-one, lexical containment, upload keeps user name, size cap, cache, bind address, telemetry, ffmpeg flags, `--stance` form).
- **Review gate (code-reviewer subagent):** SHIP, no blocking findings; 16 suggestions.

### Iteration 2 (review findings fixed)
- **Fixed, each with a test that fails when the fix is reverted (mutants below):**
  - S1 `save_upload` deleted an existing file on a name collision (the exclusive create sat inside the cleanup `try`): the create is now outside it; pinned-name collision test.
  - S2 a UNC loopback share (`\\localhost\C$\...`) walked past the participant check: network paths are refused (`faceblur._is_network`).
  - S4 a page rerun killed only the top-level python, orphaning ffmpeg/ffprobe: the child gets its own process group and the tree is killed (`taskkill /T /F`); pipe closed.
  - S5 both views ran on every rerun, so a cold transcode stalled the Analyse view, and two sessions started two ffmpegs: one view runs per rerun (`st.segmented_control`, styled like tabs) and a per-file lock makes one preview.
  - S6 the 368 MB download handle was opened and read on every rerun: downloads take a callable, read on click (`on_click="ignore"`).
  - S7 child output encoding: `PYTHONUTF8=1`. S10 the out dir must be exactly `<uploads>/out`. S11 a preview is stamped with its overlay's mtime and stale on any difference (an older restored overlay too). S13 device checked against the real list. S8/S9 dead code removed; delete counts only what is gone; out/ as a link to an outside folder is not followed (tested with a junction).
  - S3 the chart counts every row of `classified.csv`, the summary only punches inside the rounds: the page says so. The fixture now uses the pipeline's `unclassified` and one test compares against the real `IMG_4406.classified.csv` counts (17/7/7/6/4) when that gitignored file is present (`BOXING_DEMO_REAL_RESULTS`).
- **Not changed, on purpose:** the 720p scale (`scale=-2:720`, as specified; a portrait clip is 406x720); the preview's `-pix_fmt`; a hardlink into a participant folder is not detected (own-footage rule, stated in `docs/commands/demo.md`).
- **Test gate:** `test_demo.py` 237/0 with the real results folder, 236/0 and 3 loud SKIPs without it; mutants: the 9 new ones killed; the 1 survivor (the outer freshness check) is equivalent, the check inside the lock does the same job.
- **Gate reproduce:** fresh `git clone` of the branch at the commit under test, same suite, same counts (`test_demo.py` 236/0 + 3 skips there).
- **Unverified on this machine:** two behaviours need a file symlink, which Windows will not create without privilege (an output that is a link out of the folder is excluded; a link in `out/` is removed as a link). Directory junctions, which need no privilege, are tested for the same properties. The real pose-model run through the Analyse tab was not done (no weights in the worktree; wiring checked up to the ffprobe stage, and with a monkeypatched subprocess).
