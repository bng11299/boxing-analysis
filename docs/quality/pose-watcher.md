## pose-watcher -- 2026-10-26: pose on the laptop, started from the desktop

Branch `feat/pose-watcher` from `origin/main` (`8b25ebf`).

### Setup
**Task.** The desktop has no pose step (`setup-env.ps1 -NoPose`), so a new own-footage clip needed someone at
the laptop. Syncthing already shares `data/` and `results/ingest/` between the two machines. Closing the loop:
`pose_watcher.py` on the laptop (a logon scheduled task, `scripts/install-pose-watcher.ps1`) makes the pose;
`wait_for_pose.py` on the desktop waits for it and runs `analyse_clip.py --skip-ingest`. Setup steps are in
`docs/MACHINE-SYNC.md` section 5.

**Acceptance criteria.**
1. A clip in `data/own` gets `results/ingest/X.pose.npz` only when it has finished arriving: no Syncthing temp
   file, and the same size and mtime on two looks at least `--settle` apart by the watcher's clock. The pose is
   checked against the video and published in one rename, never partially.
2. A failed pose writes `X.pose.failed`, which is not retried until deleted. Nothing a clip or the ingest folder
   does stops the watcher.
3. `wait_for_pose.py` refuses a pose of another recording, a `.failed`, a timeout, participant footage and a bad
   clip name, and runs nothing in those cases. Otherwise it hands the pose to `analyse_clip --skip-ingest`, which
   publishes the full output set.
4. The scheduled task runs the repo venv's python with quoted absolute paths, at limited privilege, and refuses
   to register when the venv, the script or the weights are missing.
5. Every existing suite passes unchanged.

**Threat sketch.** Assets: correctness of the pose a report rests on; own footage (participant footage must never
take this path). Entry points: files Syncthing writes into `data/own` and `results/ingest`, CLI arguments, the
scheduled task. Trust boundary: every synced file (written by the other machine, possibly mid-transfer). Top
threats: (1) a partial or wrong-recording pose read as this clip's; (2) participant footage picked up and
derived files written into a synced folder.

### Iteration 1 -- build, test gate, review gate (code-reviewer, read-only, commit `83e3212`): FIX FIRST
- Built `pose_watcher.py`, `wait_for_pose.py` and `scripts/install-pose-watcher.ps1`, plus `tests/test_pose_watcher.py`
  and seven tests in `tests/test_machine_scripts.ps1`.
- Own finding while testing: Syncthing copies the source's mtime, so an mtime-age settle passes a clip the moment
  it lands. Readiness now needs two agreeing looks, with no reading on the first look; a test was added for that.
- Review, blocking: (R1) a clip deleted between the readiness check and pose, or a failed `mkdtemp`, raised out of
  `poll` and killed the watcher; (R2) staging and ingest on different volumes, or an unwritable ingest folder,
  made the `.failed` write raise inside the except block: the same crash, and the task restarts it into it 10
  times; (R3) neither path had a test.
- Non-blocking, taken: settle timed from mtime is close to a no-op; unbounded `seen`; `wait()` stat race; stale
  `.failed` message; conflict copies untested; the lock-free check always passed; the lock was tested in one
  process only.
- Not taken: a second manual watcher exiting 0 instead of 1. The task has `MultipleInstances IgnoreNew`, and a
  manual run should report that it did nothing.

### Iteration 2 -- fixes (`33049c9`), test gate
- `run_one` does everything inside its try; a vanished clip logs and is left for later (no `.failed`); the
  `.failed` write has its own guard; `poll` catches per clip as a last line; `main` refuses staging and ingest on
  different volumes (`same_device`); settle is timed by the watcher's clock from first sight of the current
  state; `--once` waits `max(interval, settle)` between its two looks; state for removed clips is pruned;
  `wait_for_pose` treats a stat race as not ready.
- Tests: `test_pose_watcher` 67 passed (53 before the fixes), including the vanish (two ways), EXDEV on every
  rename (survives, publishes nothing, retried later), different volumes refused, conflict copies skipped, lock
  contention from a second process. Unchanged and green: ingest 185, analyse_clip 113, pipeline 314, env 36,
  machine scripts 45.
- Real-world fact the gate asserts (not shape): the end-to-end run goes from the watcher's pose through
  `wait_for_pose` to `analyse_clip --skip-ingest` without running the pose model on the desktop (predictor call
  count unchanged), and the summary reports the clip's 0:20 of footage.
