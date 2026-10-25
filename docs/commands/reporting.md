# Commands: One-clip chain and demo overlays

Moved verbatim from the project rules' `## Commands` on 2026-10-22. Add new recipes here, not to the project rules.

One clip, every Phase 1 output: `python analyse_clip.py --video data/own/<clip>.mov --stance orthodox --device cuda:0 [--skip-ingest] [--rounds 0-180,240-420]` (chains ingest -> find -> classify -> report -> overlay through their `main()`s into `results/analyse/<clip>.*`; stages run in a staging folder and publish only if all pass; `--skip-ingest` reuses `<out-dir>/<clip>.pose.npz` only if it matches the video) · Chain tests: `python tests/test_analyse_clip.py` ·

Punches drawn on the footage (demo asset): `python render_punches.py --video data/own/<clip>.mov --pose results/ingest/<clip>.pose.npz --classified results/punches/<clip>.classified.csv` -> `results/ingest/<clip>.punches.overlay.mp4` (refuses a CSV changed since its `.classified.json`; own footage only, never participants) · Tests: `python tests/test_render_punches.py`
