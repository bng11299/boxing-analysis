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

### 2026-10-06 -- first run on the laptop (MSI, RTX 3060 Laptop, driver 617.14)
- **The CUDA path, end to end:** `bootstrap.ps1` on a fresh pull ended `ENVIRONMENT OK`, so
  `check_env --torch cu130 --pose` passed: torch 2.13.0+cu130, ultralytics 8.4.171, numpy
  2.2.6, a CUDA tensor op and `torchvision.ops.nms` on the GPU. Closes the residual
  "not run here: the CUDA path".
- **YOLO11m-pose weights no longer trust-on-first-use:** the laptop's sha256 `29b17eaf...6eb2`
  equals GitHub's server-computed digest for the asset in Ultralytics release `v8.4.0`
  (42,459,307 bytes). Pinned in `docs/MACHINE-SYNC.md` section 4.
- **Fixed:** the bootstrap summary printed the full profile path (the username); it is now
  shortened to `~` like the inventory. Checked: 0 occurrences in the summary.

