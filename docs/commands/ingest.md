# Commands: Pose ingest, punch finding and own-footage labels

Moved verbatim from the project rules' `## Commands` on 2026-10-22. Add new recipes here, not to the project rules.

Punch finder on V6: `python cli/run_punch_finder.py --data data --video data/rgb/V6.mp4` ·

Video -> pose + overlay: `python cli/run_ingest.py --video clip.mp4 --weights weights/yolo11m-pose.pt --weights-sha256 <hash> --device cuda:0` (the pinned hash and its provenance are in `docs/MACHINE-SYNC.md` §4; `--hash-weights` prints a file's hash) · Ingest tests: `python tests/test_ingest.py`

Pose on the laptop, from the desktop (`docs/MACHINE-SYNC.md` §5): laptop once `.\scripts\install-pose-watcher.ps1` (runs `pose_watcher.py` at logon: `data/own` -> `results/ingest/<clip>.pose.npz`, or `<clip>.pose.failed`); desktop per clip `python cli/wait_for_pose.py --video data/own/<clip>.mov --stance orthodox [--rounds ...]` (waits, checks, then `analyse_clip.py --skip-ingest`) · Tests: `python tests/test_pose_watcher.py`

Punch labels: config `labelstudio/punch-labels.xml` (label the overlay mp4; frame checks at burned-in 100 and 2000), import `python cli/import_labels.py --export <LS JSON export> --video results/ingest/<clip>.overlay.mp4` · Label tests: `python tests/test_labels.py`

Punches in a clip: `python cli/find_punches.py --pose results/ingest/<clip>.pose.npz [--labels data/own/<clip>.labels.json]` (per-hand wrist-speed rule, V6-tuned) · Per-hand tests: `python tests/test_hands.py`
