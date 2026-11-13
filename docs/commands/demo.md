# Commands: Local demo app

A one-page Streamlit app over `analyse_clip.py`, for showing the tool on your own laptop (poster session, advisor, friends). Local only: it binds to `127.0.0.1`, sends no telemetry and loads nothing from the internet.

Install (once, into the existing pose `.venv`; `requirements-demo.txt` is separate from `requirements.txt` on purpose):

```
.\.venv\Scripts\python -m pip install -r requirements-demo.txt
```

Run, **from the repo root** (Streamlit reads `.streamlit/config.toml` from the folder it is started in), then open <http://127.0.0.1:8501>:

```
.\.venv\Scripts\python -m streamlit run demo_app.py -- --results-dir results/analyse
```

Flags after `--`: `--results-dir` (finished clips; default `results/analyse`) and `--uploads-dir` (uploads; default `results/demo_uploads`, outputs in its `out/`). Both sit under `results/`, which is gitignored.

## The two tabs

- **Gallery** (the default; instant, safe live). Lists every clip in the results folder that has **all seven** outputs of `analyse_clip.OUTPUTS` (`pose.npz`, `punches.csv`, `classified.csv`, `classified.json`, `report.xlsx`, `summary.txt`, `overlay.mp4`); a clip missing any is left out. For the chosen clip: the overlay video, `summary.txt` word for word (its caveats are the honest-claims part of the project, so it is never shortened), a punch-type bar chart and a table from `classified.csv`, and downloads of `report.xlsx` and `classified.csv`.
- **Analyse a clip.** Upload an `.mp4`, `.mov` or `.m4v` (cap: `server.maxUploadSize` in `.streamlit/config.toml`, 1 GB), choose the stance (no default: the tool never guesses one), a device (`cpu`, plus `cuda:N` if torch sees one), optionally rounds (`0-180,240-420`), tick the own-footage confirmation, and run. It starts `analyse_clip.py` as a subprocess (argument list, no shell), streams its log into an expander, then shows the same result as the Gallery. **CPU pose extraction takes minutes per minute of video**: do not do this live. **Delete this upload and its outputs** removes the video, all outputs, the preview and any staging folder; "Earlier uploads on this laptop" lists leftovers from other sessions with the same button.

## Before a live demo

- The overlays are huge (`IMG_5835.overlay.mp4` is 1080x1920 at ~60 fps, 368 MB), so the first view of a clip makes `<clip>.overlay.web.mp4` next to it with ffmpeg (720p, CRF 28, no audio; about 1.5 minutes for IMG_5835, ~9 MB result). It is cached and remade only if the overlay is newer. **Open every clip once beforehand** so the preview exists. Without ffmpeg on `PATH` the page says so and offers the original overlay as a download instead of crashing.
- Run from the repo root of the checkout that has `results/`, `weights/` and `data/` (a fresh worktree has none of them); the pose weights are only needed for the Analyse tab.

## Rules the app enforces

- An upload is never stored under its own name: it becomes `upload_<YYYYMMDD_HHMMSS>_<6 hex>` (which satisfies `analyse_clip._CLIP_NAME`) inside the uploads folder; every path is canonicalized and checked to stay inside its folder before it is read, written or deleted.
- Nothing under a participant folder (`boxers.PARTICIPANT_ROOTS`, `~/comps-participant-data`; see [privacy](privacy.md)) is listed, read, written or deleted, and `--results-dir` / `--uploads-dir` pointed there are refused.
- Own footage only; participant footage never goes through the demo.

Code: `src/boxingvi/demo.py` (everything that decides what is trusted; importable without streamlit) and `demo_app.py` (widgets only). Tests: `python tests/test_demo.py`
