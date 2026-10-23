# Two machines, one project

SLOTMACHINE (desktop: Ryzen 5 9600X, RX 6700 XT, no CUDA) and the GPU laptop
(RTX 3060 Laptop, 6 GB). Three kinds of state, three mechanisms. None of them
is "copy the folder".

| What | How it moves | Never |
|---|---|---|
| Code, docs, tests | **git** — one clone per machine at `~\repos\boxing-comps`; pull before work, push after | Syncthing on the repo root (it races `.git` and corrupts it) |
| Python environment | **rebuilt on each machine** by `scripts\setup-env.ps1` | copied: torch is `+cpu` on one and `+cu130` on the other, and a venv stores absolute paths |
| Data, video, weights, pose output | **Syncthing**, peer-to-peer, three folders below | git, GitHub, or any cloud drive |

## 0. One shot: steps 1 and 2 together

`scripts\bootstrap.ps1` runs the inventory (saved to
`~\boxing-comps-inventory-<machine>-<time>.txt`) and then `setup-env.ps1`, and
prints a summary. It moves and deletes nothing. On a machine that does not have
the repo yet, the command in the PR description gets it first.

## 1. Survey the laptop first (changes nothing)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\inventory-machine.ps1
```

Paste the output into the assistant session. It lists every clone, environment,
video and weights file (weights with sha256), and the GPU, ffmpeg and BitLocker
state. Moves and deletions are decided from it, before anything is touched.

## 2. One clone, one environment

```powershell
cd ~\repos\boxing-comps
.\scripts\setup-env.ps1 -CheckOnly    # is the .venv here good? (exit 1 if not)
.\scripts\setup-env.ps1               # build or repair it; auto-picks CUDA when nvidia-smi sees a GPU
.\scripts\setup-env.ps1 -NoPose       # on SLOTMACHINE: CPU torch, no ultralytics
```

It ends with `ENVIRONMENT OK` or a list of what is wrong. Run it the same way
each time on a machine: `-NoPose` after a full run leaves the pose packages in
place but stops checking them, so a broken torchvision would go unreported. Never run a bare
`pip install -r requirements.txt` on the laptop: that file pins
`torch==2.13.0+cpu` and would replace the CUDA build.

## 3. Syncthing: three folders, nothing else

Install Syncthing on both machines (syncthing.net), pair the two devices, then
share exactly these, Send & Receive, with **Trash Can file versioning** on each
(a deletion on one machine otherwise deletes on the other):

| Folder | Holds |
|---|---|
| `~\repos\boxing-comps\data` | BoxingVI, Olympic Boxing (when cleared), `data\own\` (your footage) |
| `~\repos\boxing-comps\weights` | checkpoints — verify by hash on arrival (step 4) |
| `~\repos\boxing-comps\results\ingest` | pose `.npz` and overlays coming back from the laptop |

All three are gitignored, so nothing Syncthing writes can reach a commit.

### Start Syncthing from Windows, never from an assistant terminal

Learned 2026-10-07. A Syncthing started from inside the assistant desktop app (its
terminal pane, or an assistant tool call) is closed when the app restarts: Windows
logged "syncthing.exe stopped interacting with Windows and was closed" at the
same minute the app restarted (10-06 12:53, 10-07 12:06). On the laptop the first
install also kept its identity somewhere the app's environment pointed, so the
Syncthing restarted from Windows came up as a NEW device (B66PIM4..., config
created 10-07 13:24) with no pairing, and the 29 Olympic tracks files could not
come back. Fix: start it with Win+R -> `shell:startup` -> double-click
Syncthing (or just log in), check its parent is `explorer` running since login,
and re-pair a device whose ID changed (a new device has no history, so nothing
it holds can propagate a deletion).

## 4. Weights are checked, not trusted

Every checkpoint is a pickle and is loaded only against a pinned sha256. After a
sync, re-run the inventory on the receiving machine and compare the hashes it
prints with the pins. ST-GCN++: `c6e07421...` (pinned in code). YOLO11m-pose:

    yolo11m-pose.pt  sha256 29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2

Independently checked 2026-10-06: equal to the server-computed digest GitHub
publishes for the asset in the Ultralytics `v8.4.0` release
(`api.github.com/repos/ultralytics/assets/releases`, 42,459,307 bytes). That says
the file is byte-identical to what Ultralytics published, not that it is benign;
the loader still refuses it unless this hash is passed and matches.

## Participant footage: not in these folders yet

The approved protocol (FA26-22-NG, item 15) keeps data "password-protected and
encrypted on the researcher's computer", with no cloud copies. **Hector confirmed
2026-10-06 that the GPU laptop counts as the researcher's computer**, so
participant footage may live on both machines. What item 15 still requires:

- **BitLocker on, on both machines, before any participant footage is stored.**
  The inventory reports it from an admin shell; on 2026-10-06 neither machine had
  been checked. Until then, participant footage lives in
  `~\comps-participant-data\`, outside every synced folder.
- When it syncs, turn off relaying and global discovery and pair by LAN address,
  so the traffic never transits a third-party relay (item 10: footage is not sent
  to any third party).

Your own footage and the public datasets have none of these constraints.
