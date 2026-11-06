"""Offline smoke run: the whole chain, video in -> every output out, with the network switched off.

  python scripts/offline_smoke.py --allow-synthetic [--out-dir DIR] [--keep]

Builds a SYNTHETIC fixture (a generated grey video, a stick figure with planted punches, random classifier
weights, hand-written rule files), then runs `analyse_clip.py` solo and `analyse_clip.py --two-boxer` on it with
every socket call refused, and checks the outputs. A pose predictor is injected in place of YOLO, so neither the
pose weights nor ultralytics are needed. Needs only what a fresh clone has plus requirements.txt and ffmpeg.

No number this writes says anything about boxing: --allow-synthetic is required, and every output folder gets a
SYNTHETIC.txt stamp. Exit 0 = every check passed; 1 = a check failed; 2 = refused (no --allow-synthetic).
tests/test_fresh_clone.py runs this inside a fresh `git clone` of the repo.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SOLO_OUTPUTS = ("pose.npz", "punches.csv", "classified.csv", "classified.json", "report.xlsx", "summary.txt",
                "overlay.mp4")


class NetworkUsed(RuntimeError):
    pass


def block_network() -> None:
    """Every way out of the process through the socket module raises. Called before the pipeline is imported."""
    def refuse(*a, **k):
        raise NetworkUsed("the offline chain tried to use the network")
    socket.socket.connect = refuse            # type: ignore[method-assign]
    socket.socket.connect_ex = refuse         # type: ignore[method-assign]
    socket.socket.sendto = refuse             # type: ignore[method-assign]
    socket.create_connection = refuse         # type: ignore[assignment]
    socket.getaddrinfo = refuse               # type: ignore[assignment]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-synthetic", action="store_true", help="required: the outputs are synthetic")
    ap.add_argument("--out-dir", type=Path, help="where to build (default: a new temporary folder)")
    ap.add_argument("--keep", action="store_true", help="keep the temporary folder")
    a = ap.parse_args(argv)
    if not a.allow_synthetic:
        print("offline_smoke: refused without --allow-synthetic (its outputs are synthetic, never results)",
              file=sys.stderr)
        return 2
    if shutil.which("ffmpeg") is None:
        print("offline_smoke: ffmpeg is not on PATH", file=sys.stderr)
        return 1
    os.environ.pop("GEMINI_API_KEY", None)   # the default summary path must not depend on one
    block_network()

    import numpy as np
    import torch

    import analyse_clip
    import make_identity
    from boxingvi import boxers, classify, video
    from boxingvi.data import CLASSES, N_JOINTS
    from boxingvi.pose_yolo import PoseTable
    from boxingvi.stgcnpp import STGCNPP

    tmp = a.out_dir or Path(tempfile.mkdtemp(prefix="boxingvi-smoke-"))
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "SYNTHETIC.txt").write_text("Synthetic fixture from scripts/offline_smoke.py: not a result.\n",
                                       encoding="utf-8")
    fails: "list[str]" = []

    def check(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"  {detail}"))
        if not cond:
            fails.append(name)

    def ffmpeg_video(path: Path, w: int, h: int, fps: int, n: int) -> Path:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=gray:s={w}x{h}:r={fps}",
                        "-frames:v", str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
        return path

    fps, n = 60, 60 * 20
    pose = np.array([[0, -150], [-8, -158], [8, -158], [-16, -154], [16, -154], [-50, -100], [50, -100],
                     [-60, -50], [60, -50], [-55, -10], [55, -10], [-30, 0], [30, 0], [-32, 90], [32, 90],
                     [-34, 180], [34, 180]], dtype=np.float64)

    def body(cx, left, right):
        t = np.arange(n) / fps
        xy = np.broadcast_to(pose * 0.25 + (cx, 60), (n, N_JOINTS, 2)).copy()
        for hand, times in ((0, left), (1, right)):
            for p in times:
                ph = np.clip((t - p) / 0.25, 0, 1)
                xy[:, 9 + hand, 0] += (-1 if hand == 0 else 1) * 30 * np.sin(np.pi * ph)
        return xy.astype(np.float32).astype(np.float64)

    # Model files: random weights and hand-written rules. The plumbing is under test, not the punch types.
    torch.manual_seed(0)
    ck = tmp / "clf.pt"
    torch.save(STGCNPP().state_dict(), ck)
    C = np.array([[90, 5, 0, 0, 5, 0], [2, 98, 0, 0, 0, 0], [0, 0, 80, 20, 0, 0],
                  [0, 10, 40, 50, 0, 0], [30, 0, 0, 0, 20, 0], [10, 0, 0, 0, 0, 90]])
    f1 = {c: 2 * int(C[i, i]) / int(C[i].sum() + C[:, i].sum()) for i, c in enumerate(CLASSES)}
    manifest = tmp / "clf.json"
    classify.write_manifest(manifest, ck, classify.WindowGeometry(0.375, 0.125, 24.0, 200.0, 0.5, 0.55,
                                                                  "relative", 100),
                            50, {"held_out": {"per_class_f1": f1, "confusion": C.tolist()}})
    ref = tmp / "ref.json"
    ref.write_text(json.dumps({"val_videos": ["V2", "V5", "V9"], "results": [
        {"model": "st-gcn++ pretrained", "confusion": C.tolist(), "per_class_f1": f1}]}), encoding="utf-8")
    rule = tmp / "rule.json"
    rule.write_text(json.dumps({"video": "V6", "chosen": {"variant": "relative", "threshold": 1.0, "min_gap_s": 0.4,
                    "test": {"recall": 0.741, "precision": 0.465, "count_ratio": 1.596, "n_labelled": 406}}}),
                    encoding="utf-8")
    extent = tmp / "extent.json"
    extent.write_text(json.dumps({"frac": 0.5, "lead_widths": 1.3, "tail_widths": 1.8, "variant": "relative",
                                  "beats_fixed_window": True, "gated_cut": classify.GATED_CUT}), encoding="utf-8")
    fake_weights = tmp / "fake-pose.pt"   # never loaded: the predictor below stands in for YOLO
    fake_weights.write_bytes(b"not a checkpoint")
    files = ["--classifier", str(manifest), "--reference", str(ref), "--rule", str(rule), "--extent", str(extent)]

    def run(argv, predict=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = analyse_clip.main(argv, predict=predict) if predict else analyse_clip.main(argv)
            except SystemExit as e:
                rc = e.code
        return rc, out.getvalue(), err.getvalue()

    print("solo: video -> pose -> punches -> classified -> xlsx + summary + overlay")
    solo_xy = body(80, (2, 5, 8, 11, 14), (3.5, 9.5, 16))
    k = {"i": 0}

    def predict(frame):
        kxy = solo_xy[k["i"] % n]
        k["i"] += 1
        box = np.array([[kxy[:, 0].min(), kxy[:, 1].min(), kxy[:, 0].max(), kxy[:, 1].max()]])
        return box, kxy[None], np.full((1, N_JOINTS), 0.9)

    vid = ffmpeg_video(tmp / "SMOKE01.mp4", 160, 120, fps, n)
    out = tmp / "solo"
    rc, so, se = run(["--video", str(vid), "--stance", "orthodox", "--out-dir", str(out),
                      "--weights", str(fake_weights), *files], predict)
    check("solo: exit 0 with the network off", rc == 0, se[-1500:])
    check("solo: every output written", all((out / f"SMOKE01.{s}").is_file() for s in SOLO_OUTPUTS),
          str(sorted(p.name for p in out.iterdir())) if out.is_dir() else "no out dir")
    if rc == 0:
        rows = list(csv.DictReader((out / "SMOKE01.classified.csv").open(encoding="utf-8")))
        # a fact, not a shape: the fixture plants 8 punches, 5 with the left hand
        check("solo: the 8 planted punches were found, 5 left-handed",
              len(rows) == 8 and sum(r["hand"] == "left" for r in rows) == 5, f"{len(rows)} rows")
        summary = (out / "SMOKE01.summary.txt").read_text(encoding="utf-8")
        check("solo: the summary counts the 8 punches", "8" in summary and summary.strip() != "")
        check("solo: the summary is the template (no language model ran)", "language model" not in summary)
        k["i"] = 0
        out_g = tmp / "solo-gemini-no-key"
        rc, so, se = run(["--video", str(vid), "--stance", "orthodox", "--out-dir", str(out_g), "--no-overlay",
                          "--weights", str(fake_weights), "--narrate", "gemini", *files], predict)
        check("solo --narrate gemini with no key: exit 0, network untouched", rc == 0, se[-1500:])
        check("solo --narrate gemini with no key: the template summary, byte for byte",
              rc == 0 and (out_g / "SMOKE01.summary.txt").read_text(encoding="utf-8") == summary)
        check("solo --narrate gemini with no key: says why it kept the template", "GEMINI_API_KEY" in so)

        # participant footage with a key set: never sent (the source video is checked, not just the staged files)
        from boxingvi import gemini
        proot = tmp / "participant-root"
        proot.mkdir()
        pvid = shutil.copy(vid, proot / "SMOKE01.mp4")
        sent = []
        old_post, old_roots = gemini._post, boxers.PARTICIPANT_ROOTS
        gemini._post = lambda *x, **y: sent.append(1) or b""
        boxers.PARTICIPANT_ROOTS = (proot,)
        os.environ["GEMINI_API_KEY"] = "AIzaSMOKE_not_a_real_key_0123456789"
        try:
            k["i"] = 0
            rc, so, se = run(["--video", str(pvid), "--stance", "orthodox", "--out-dir", str(proot / "out"),
                              "--no-overlay", "--weights", str(fake_weights), "--narrate", "gemini", *files], predict)
        finally:
            gemini._post, boxers.PARTICIPANT_ROOTS = old_post, old_roots
            os.environ.pop("GEMINI_API_KEY", None)
        check("participant video + --narrate gemini + a key: nothing sent, template kept",
              rc == 0 and sent == [] and "never leaves" in so, (so + se)[-1500:])

    print("two boxers: tracks + identity -> per-boxer punches, reports and summaries")
    w2 = 320
    xa, xb = body(80, (2, 5, 8, 11, 14), (3.5, 9.5, 16)), body(240, (3, 12), (4, 13, 17.5))
    vid2 = ffmpeg_video(tmp / "BOUT01.mp4", w2, 120, fps, n)
    fr = np.r_[np.arange(n), np.arange(n)]
    tid = np.r_[np.full(n, 1), np.full(n, 2)].astype(np.int64)
    xy = np.concatenate([xa, xb])
    bx = np.stack([xy[:, :, 0].min(1), xy[:, :, 1].min(1), xy[:, :, 0].max(1), xy[:, :, 1].max(1)], axis=1)
    table = PoseTable(frame=fr.astype(np.int64), track_id=tid, box=bx, xy=xy,
                      conf=np.full((2 * n, N_JOINTS), np.float64(np.float32(0.9))), n_frames=n, fps=float(fps),
                      width=w2, height=120, source="BOUT01.mp4", model="fake", video_fp=video.fingerprint(vid2))
    tracks = tmp / "BOUT01.tracks.npz"
    table.save(tracks)
    table = PoseTable.load(tracks)
    ident = tmp / "BOUT01.identity.csv"
    with contextlib.redirect_stdout(io.StringIO()):
        rc = make_identity.main(["--tracks", str(tracks), "--a", "1", "--b", "2"])
    check("make_identity.py writes the identity from the chosen track ids", rc == 0 and ident.is_file()
          and boxers.read_identity(ident, table, tracks).method == "by-hand")
    out2 = tmp / "two"
    rc, so, se = run(["--two-boxer", "--video", str(vid2), "--tracks", str(tracks), "--identity", str(ident),
                      "--stance-a", "orthodox", "--stance-b", "southpaw", "--out-dir", str(out2), *files])
    check("two-boxer: exit 0 with the network off", rc == 0, se[-1500:])
    check("two-boxer: every output written",
          all((out2 / f"BOUT01.{s}").is_file() for s, _ in analyse_clip.TWO_BOXER_OUTPUTS),
          str(sorted(p.name for p in out2.iterdir())) if out2.is_dir() else "no out dir")
    if rc == 0:
        rows = list(csv.DictReader((out2 / "BOUT01.boxers.csv").open(encoding="utf-8")))
        per = {b: sum(r.get("boxer") == b for r in rows) for b in "AB"}
        check("two-boxer: A's 8 planted punches and B's 5 were found", per == {"A": 8, "B": 5}, str(per))

    print(f"{'OK' if not fails else 'FAILED'}: {len(fails)} failed check(s); fixture in {tmp}")
    if a.out_dir is None and not a.keep and not fails:
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
