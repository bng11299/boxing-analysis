"""The accuracy lost when the classifier reads YOLO11-pose instead of AlphaPose.

Step 1, on the NVIDIA laptop (pose only; nothing here needs a GPU):

    foreach ($v in 'V2','V5','V9') {
      .venv\\Scripts\\python.exe run_ingest.py --video data\\rgb\\$v.mp4 --weights weights\\yolo11m-pose.pt `
        --weights-sha256 29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2 --device cuda:0 --no-overlay
    }

Step 2, anywhere, once results/ingest/V2.pose.npz, V5 and V9 have synced back:

    python eval_pose_switch.py --data data --poses results/ingest \\
        --classifier weights/classifier/stgcnpp-boxingvi-seed0.json

Gates, in order; each one that fails stops the run with nothing scored:
1. the saved classifier, reading the corpus's own held-out clips, reproduces the
   held-out accuracy and macro F1 its manifest records -- so the model and the
   input path are the reported ones, and any drop below is the pose switch;
2. each video's YOLO pose is the downloaded source: its frame rate is the
   video's native one (O54) and it covers every annotated span;
3. each video's spans align with YOLO's frame numbering (poseswitch.align).

Readings of the same held-out punches by the same classifier:
  A      AlphaPose clips, as shipped
  Y      YOLO11-pose over the same frames, joints below confidence 0.3 dropped
         (the pipeline's rule)
  Y_all  YOLO11-pose keeping every detected joint (AlphaPose drops few)
  Y_win  YOLO11-pose through the pipeline's own window (classify.cut_window) at the
         wrist-speed peak inside the span: the pipeline with perfect punch timing.
         Y - Y_win is the cost of the fixed-length window, on all six punch types
         (on V6, hooks and uppercuts only, it was 21 points: results/classify_v6.json)
Reported on every clip YOLO saw, and separately on clips where YOLO followed the
same person as AlphaPose; clips YOLO missed are counted. Paired bootstrap
intervals on A - Y.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from boxingvi import classify, poseswitch, punchfind  # noqa: E402
from boxingvi.data import CLASSES, NATIVE_FPS, load_corpus, split_by_video, to_arrays  # noqa: E402
from boxingvi.pose_yolo import PoseSequence  # noqa: E402
from boxingvi.stats import bootstrap_gap_ci  # noqa: E402
from classify_punches import span_peak  # noqa: E402

DEFAULT_OUT = Path("results/pose_switch.json")
TOL = 1e-9


def scores(y, p) -> dict:
    from sklearn.metrics import f1_score

    y, p = np.asarray(y), np.asarray(p)
    if not len(y):
        return {"n": 0}
    side = np.array([classify.class_side(c) for c in CLASSES])
    return {"n": int(len(y)), "accuracy": float((y == p).mean()),
            "macro_f1": float(f1_score(y, p, labels=range(len(CLASSES)), average="macro", zero_division=0)),
            "per_class_f1": dict(zip(CLASSES, f1_score(y, p, labels=range(len(CLASSES)), average=None,
                                                       zero_division=0).round(4).tolist())),
            "hand_accuracy": float((side[y] == side[p]).mean())}


def predict(model, man, clips: "list[np.ndarray]") -> np.ndarray:
    return np.argmax(classify.predict_proba(model, clips, man["t_out"]), axis=1) if clips else np.zeros(0, int)


def paired(y, a, b) -> dict:
    if len(y) < 2:
        return {"n": len(y)}
    return {"n": len(y), "accuracy": bootstrap_gap_ci(y, a, b, metric="accuracy"),
            "macro_f1": bootstrap_gap_ci(y, a, b, metric="macro_f1")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--poses", type=Path, required=True, help="folder with <video>.pose.npz from run_ingest.py")
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--videos", default="V2,V5,V9")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    videos = a.videos.split(",")
    model, man = classify.load_classifier(a.classifier)

    # Gate 1: the reported model and input path, on the corpus's own held-out clips.
    held = (man.get("held_out") or {})
    _, val = split_by_video(load_corpus(a.data))
    if sorted({c.source for c in val}) != sorted(videos):
        raise SystemExit(f"the corpus holds out {sorted({c.source for c in val})}, not {sorted(videos)}")
    xv, yv = to_arrays(val)
    gate = scores(yv, predict(model, man, list(xv)))
    print(f"gate 1: corpus held-out clips, accuracy {gate['accuracy']:.4f} macro F1 {gate['macro_f1']:.4f} "
          f"(manifest {held.get('accuracy')} / {held.get('macro_f1')})")
    if not (isinstance(held.get("accuracy"), float) and abs(gate["accuracy"] - held["accuracy"]) < TOL
            and abs(gate["macro_f1"] - held["macro_f1"]) < TOL):
        raise SystemExit("GATE 1 FAILED: this classifier does not reproduce its own held-out numbers; nothing scored")

    rows, report = [], {"classifier_sha256": man["sha256"], "gate_corpus_held_out": gate, "videos": {}}
    for v in videos:
        pose = PoseSequence.load(a.poses / f"{v}.pose.npz")
        clips, mismatched = poseswitch.held_out_clips(a.data, v)
        if abs(pose.fps - NATIVE_FPS[v]) > 1e-3:
            raise SystemExit(f"GATE 2 FAILED: {v} pose is {pose.fps:.3f} fps, the source is {NATIVE_FPS[v]:.3f}")
        if max(c.end for c in clips) >= len(pose):
            raise SystemExit(f"GATE 2 FAILED: {v} pose has {len(pose)} frames; spans run to {max(c.end for c in clips)}")
        al = poseswitch.align(pose, clips)
        info = {"clips": len(clips), "span_mismatched": mismatched, "frames": len(pose),
                "detected_share": float(pose.detected.mean()), "alignment": al}
        report["videos"][v] = info
        print(f"{v}: {len(clips)} clips ({mismatched} spans unusable), best offset {al['best']}, "
              f"sharpness {al['ratio'] if al['ratio'] is None else round(al['ratio'], 2)}")
        if not al["sharp"]:
            raise SystemExit(f"GATE 3 FAILED: {v} spans do not align with the video's frames; nothing scored")
        o = al["best"]
        geom = man["geometry"]
        track = punchfind.bridge(pose.to_track(usable_only=False), int(round(punchfind.BRIDGE_S * pose.fps)))
        torso = punchfind.body_scale(track)
        sig = punchfind.wrist_signal(track, geom.variant)
        for c in clips:
            win = classify.cut_window(track, span_peak(track, sig, c.start + o, c.end + o), geom, torso)
            y = poseswitch.yolo_clip(pose, c.start + o, c.end + o)
            y_all = poseswitch.yolo_clip(pose, c.start + o, c.end + o, None)
            rows.append({"video": v, "label": c.label, "A": c.alphapose, "Y": y, "Y_all": y_all, "Y_win": win.keypoints,
                         "seen": int((y[..., 2] > 0).any(axis=1).sum()) >= 2,
                         "seen_all": int((y_all[..., 2] > 0).any(axis=1).sum()) >= 2,
                         "same": poseswitch.same_person(c, y_all)})

    seen = [r for r in rows if r["seen"] and r["seen_all"]]
    y = np.array([r["label"] for r in seen])
    pa = predict(model, man, [r["A"] for r in seen])
    py = predict(model, man, [r["Y"] for r in seen])
    pya = predict(model, man, [r["Y_all"] for r in seen])
    same = np.array([r["same"] is True for r in seen])
    wseen = [r for r in seen if r["Y_win"] is not None]
    yw = np.array([r["label"] for r in wseen])
    pyw_y = predict(model, man, [r["Y"] for r in wseen])
    pyw = predict(model, man, [r["Y_win"] for r in wseen])
    report.update({
        "n_clips": len(rows), "n_yolo_missed": len(rows) - len(seen),
        "n_same_person": int(same.sum()), "n_other_person": int(sum(r["same"] is False for r in seen)),
        "all_seen": {"A": scores(y, pa), "Y": scores(y, py), "Y_all": scores(y, pya),
                     "A_minus_Y": paired(y, pa, py), "A_minus_Y_all": paired(y, pa, pya)},
        "window": {"Y": scores(yw, pyw_y), "Y_win": scores(yw, pyw), "Y_minus_Y_win": paired(yw, pyw_y, pyw),
                   "n_unwindowable": len(seen) - len(wseen)},
        "same_person": {"A": scores(y[same], pa[same]), "Y": scores(y[same], py[same]),
                        "A_minus_Y": paired(y[same], pa[same], py[same])},
        "limits": ["held-out videos are coaches' YouTube footage at 24-30 fps, not phone footage",
                   "the video-disjoint split may not be athlete-disjoint (O55)",
                   "YOLO keeps one person per frame; other-person clips are counted, and scored in all_seen"]})
    for name, block in (("every clip YOLO saw", report["all_seen"]), ("same person only", report["same_person"])):
        print(f"{name}: " + ", ".join(f"{k} acc {block[k]['accuracy']:.3f} F1 {block[k]['macro_f1']:.3f}"
                                      for k in ("A", "Y", "Y_all") if k in block and block[k].get("n")))
        g = block["A_minus_Y"].get("accuracy")
        if g:
            print(f"  A - Y accuracy {g['point']:+.3f} [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}] on {block['A_minus_Y']['n']}")
    g = report["window"]["Y_minus_Y_win"].get("accuracy")
    if g:
        print(f"window cost (Y - Y_win) accuracy {g['point']:+.3f} [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}] "
              f"on {report['window']['Y_minus_Y_win']['n']}")
    print(f"{report['n_yolo_missed']} clips YOLO did not see; {report['n_other_person']} where it followed someone else")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, allow_nan=False, default=float), encoding="utf-8")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
