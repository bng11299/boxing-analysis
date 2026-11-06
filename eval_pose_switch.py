"""The accuracy lost when the classifier reads YOLO11-pose instead of AlphaPose.

Step 1, on the NVIDIA laptop (pose only; nothing here needs a GPU):

    foreach ($v in 'V2','V5','V9') {
      .venv\\Scripts\\python.exe run_ingest.py --video data\\rgb\\$v.mp4 --weights weights\\yolo11m-pose.pt `
        --weights-sha256 29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2 --device cuda:0 --no-overlay
    }

Step 2, anywhere, once results/ingest/V2.pose.npz, V5 and V9 have synced back:

    python eval_pose_switch.py --data data --poses results/ingest \\
        --classifier weights/classifier/stgcnpp-boxingvi-seed0.json --extent results/punch_extent.json

Gates, in order. Gates 1 and 2 stop the run with nothing scored. Gate 3 is per
video: a video whose spans do not align is EXCLUDED, named in the output and in
the results file, and the rest are scored; if none aligns, nothing is.
(2026-10-12: V2 fails it. Its shipped skeletons are not at their annotated frames
of the video BoxingVI links as V2, at any shift or frame-rate ratio, nor anywhere
else in that video; V5 and V9 sit exactly at theirs. Vault ISSUES-LOG.)
1. the saved classifier, reading the corpus's own held-out clips, reproduces the
   held-out accuracy and macro F1 its manifest records -- so the model and the
   input path are the reported ones, and any drop below is the pose switch;
2. each video's YOLO pose is the downloaded source: its frame rate is the
   video's native one (O54) and it covers every annotated span;
3. each video's spans align with YOLO's frame numbering (poseswitch.align).

Readings of the same held-out punches by the same classifier:
  A      AlphaPose clips, as shipped
  Y_all  YOLO11-pose over the same frames, every detected joint kept: THE PIPELINE'S
         RULE since 2026-10-12 (classify.CLASSIFY_MIN_CONF). A - Y_all is the cost
         of the pose-model switch
  Y      YOLO11-pose with joints under confidence 0.3 blanked: the rule the pipeline
         used until then, kept as the record of what it cost (0.348 against 0.889)
  Y_clip YOLO11-pose through THE PIPELINE'S CUT (ADR-0014 as amended 2026-10-13,
         classify.clip_cut + clip_window): one length per video, the median of its
         punches' own extents, around the wrist-speed peak inside each span. Perfect
         punch timing, every detected joint kept
  Y_ext  the same peak, each punch cut by ITS OWN extent (punchfind.extent): the
         design the amendment replaced
  Y_fix  the same peak, the old FIXED 0.375 s window (classify.cut_window)
         Y_all - Y_clip is what the pipeline's cut costs; Y_clip - Y_fix what it buys
         over the old window; Y_ext - Y_clip what sizing each punch on its own adds
         (it does not). All on the punch types these videos hold (V6 had hooks and
         uppercuts only). The per-punch extent's overlap with each labelled span is
         reported by punch type.
Which punches each comparison is read on (`aggregate`, 2026-10-21):
  all_seen       every punch YOLO saw in >= 2 frames with every detected joint kept, the
                 pipeline's rule. A, Y_all and A - Y_all (the pose-switch cost) are read here,
                 and the cut readings start from here.
  legacy_common  the punches of all_seen that the legacy 0.3 rule ALSO sees in >= 2 frames.
                 Y is compared only here, with A and Y_all re-read on the same punches.
Until 2026-10-21 every reading used legacy_common, so A - Y_all left out the punches the
old rule blanks and the pipeline reads. Both n's are reported, pooled, for the same person
only, and per video; clips YOLO missed are counted. Paired bootstrap intervals throughout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from boxingvi import classify, poseswitch, punchfind  # noqa: E402
from boxingvi.data import CLASSES, NATIVE_FPS, load_corpus, split_pipeline, to_arrays  # noqa: E402
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


SUBSETS = {
    "all_seen": "every punch YOLO saw in >= 2 frames with every detected joint kept (the pipeline's rule); "
                "A - Y_all, the cost of the pose switch, is read here",
    "legacy_common": "the punches of all_seen that the legacy 0.3 rule (Y) also sees in >= 2 frames; "
                     "Y is compared only here, with A and Y_all re-read on the same punches",
}
CUTS = ("Y_fix", "Y_ext", "Y_clip")


def _block(y, preds: dict, mask) -> dict:
    """Scores of every reading in `preds` on the punches `mask` picks, and A minus each other one."""
    out = {"n": int(mask.sum())}
    out.update({k: scores(y[mask], p[mask]) for k, p in preds.items()})
    out.update({f"A_minus_{k}": paired(y[mask], preds["A"][mask], p[mask]) for k, p in preds.items() if k != "A"})
    return out


def _readings(sub: list, predict_fn, keys: tuple, scored: list) -> dict:
    y = np.array([r["label"] for r in sub], dtype=int)
    preds = {k: np.asarray(predict_fn([r[k] for r in sub]), dtype=int) for k in keys}
    same = np.array([r["same"] is True for r in sub], dtype=bool)
    vid = np.array([r["video"] for r in sub], dtype=str)
    return {"pooled": _block(y, preds, np.ones(len(sub), bool)), "same_person": _block(y, preds, same),
            "by_video": {v: _block(y, preds, vid == v) for v in scored}}


def aggregate(rows: list, predict_fn, scored: list) -> dict:
    """Every score in the results file, from the per-punch rows. `predict_fn` maps a
    list of clips to predicted labels; nothing else here touches the classifier.

    A vs Y_all is read on all_seen; the legacy rule Y only on legacy_common (see
    SUBSETS). Any reading can be empty (n 0): `scores` then gives {"n": 0}."""
    seen_all = [r for r in rows if r["seen_all"]]
    common = [r for r in seen_all if r["seen"]]
    s = _readings(seen_all, predict_fn, ("A", "Y_all"), scored)
    c = _readings(common, predict_fn, ("A", "Y", "Y_all"), scored)
    wseen = [r for r in seen_all if all(r[k] is not None for k in CUTS)]
    yw = np.array([r["label"] for r in wseen], dtype=int)
    pw = {k: np.asarray(predict_fn([r[k] for r in wseen]), dtype=int) for k in ("Y_all",) + CUTS}
    iou_by = {c_: [r["iou"] for r in rows if r["iou"] is not None and r["label"] == n] for n, c_ in enumerate(CLASSES)}
    return {
        "n_clips": len(rows), "n_yolo_missed": len(rows) - len(seen_all),
        "n_legacy_blanked": len(seen_all) - len(common),
        "n_same_person": sum(r["same"] is True for r in seen_all),
        "n_other_person": sum(r["same"] is False for r in seen_all),
        "subsets": SUBSETS,
        "all_seen": s["pooled"], "legacy_common": c["pooled"],
        "same_person": {"all_seen": s["same_person"], "legacy_common": c["same_person"]},
        "by_video": {v: {"all_seen": s["by_video"][v], "legacy_common": c["by_video"][v]} for v in scored},
        "window": {"subset": "the all_seen punches every cut exists for", "n": len(wseen),
                   **{k: scores(yw, v) for k, v in pw.items()},
                   "Y_all_minus_Y_clip": paired(yw, pw["Y_all"], pw["Y_clip"]),
                   "Y_ext_minus_Y_fix": paired(yw, pw["Y_ext"], pw["Y_fix"]),
                   "Y_ext_minus_Y_clip": paired(yw, pw["Y_ext"], pw["Y_clip"]),
                   "Y_clip_minus_Y_fix": paired(yw, pw["Y_clip"], pw["Y_fix"]),
                   "n_uncuttable": len(seen_all) - len(wseen),
                   "extent_iou_with_label": {c_: {"n": len(v), "mean": float(np.mean(v)) if v else None}
                                             for c_, v in iou_by.items()}}}


def _acc(s: dict) -> str:
    return f"{s['accuracy']:.3f}" if s.get("n") else "n/a"


def _accf1(s: dict) -> str:
    return f"{s['accuracy']:.3f} F1 {s['macro_f1']:.3f}" if s.get("n") else "n/a"


def _gap(block: dict, key: str) -> str:
    g = (block.get(key) or {}).get("accuracy")
    return f"{g['point']:+.3f} [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}]" if g else "n/a"


def summary_lines(report: dict) -> "list[str]":
    """What the run prints. Every reading may be empty; an empty one prints as n/a."""
    out = [f"SCORED: {'/'.join(report['scored_videos'])} of the held-out {'/'.join(report['held_out_videos'])}"
           + (f"; EXCLUDED: {'/'.join(report['excluded_videos'])}" if report["excluded_videos"] else "")]
    for v in report["scored_videos"]:
        s, c = report["by_video"][v]["all_seen"], report["by_video"][v]["legacy_common"]
        out.append(f"  {v}: A acc {_acc(s['A'])}, Y_all acc {_acc(s['Y_all'])} on {s['n']}; "
                   f"A - Y_all {_gap(s, 'A_minus_Y_all')}. Old 0.3 rule {_acc(c['Y'])} on the {c['n']} it sees")
    for name, s, c in (("every punch YOLO saw", report["all_seen"], report["legacy_common"]),
                       ("same person only", report["same_person"]["all_seen"],
                        report["same_person"]["legacy_common"])):
        out.append(f"{name} (all_seen, {s['n']}): A acc {_accf1(s['A'])}, Y_all acc {_accf1(s['Y_all'])}; "
                   f"A - Y_all {_gap(s, 'A_minus_Y_all')}")
        out.append(f"  legacy_common ({c['n']} of those {s['n']}): A acc {_accf1(c['A'])}, Y acc {_accf1(c['Y'])}, "
                   f"Y_all acc {_accf1(c['Y_all'])}; A - Y {_gap(c, 'A_minus_Y')}, A - Y_all {_gap(c, 'A_minus_Y_all')}")
    w = report["window"]
    if not w["n"]:
        out.append(f"cut readings: no punch had every cut ({w['n_uncuttable']} uncuttable); nothing compared")
    else:
        out.append("cut readings, every detected joint kept: " + ", ".join(
            f"{k} acc {_acc(w[k])}" for k in ("Y_all",) + CUTS) + f" on {w['n']}")
        for key, name in (("Y_all_minus_Y_clip", "what the pipeline's cut costs (Y_all - Y_clip)"),
                          ("Y_clip_minus_Y_fix", "what the per-clip window buys over the old window (Y_clip - Y_fix)"),
                          ("Y_ext_minus_Y_clip", "what sizing each punch on its own adds (Y_ext - Y_clip)"),
                          ("Y_ext_minus_Y_fix", "per-punch extent against the old window (Y_ext - Y_fix)")):
            out.append(f"{name}: accuracy {_gap(w, key)} on {w[key]['n']}")
    out.append("extent overlap with the labelled span, by type: " + (", ".join(
        f"{c} {v['mean']:.2f}" for c, v in w["extent_iou_with_label"].items() if v["mean"] is not None) or "none"))
    out.append(f"{report['n_yolo_missed']} clips YOLO did not see; {report['n_legacy_blanked']} it saw that the "
               f"old 0.3 rule blanks; {report['n_other_person']} where it followed someone else")
    return out


def y_clip(track, peak_frame: int, cut, geom, torso: float) -> np.ndarray:
    """Y_clip's keypoints: exactly what the pipeline reads under the clip's cut. A
    training-cut fallback (too few punches) is cut_window's own frames; the old
    clip_window + cut_span read held 8 of its 9 frames at 25, 29.97 and 59.94 fps."""
    return classify.read_clip_window(track, peak_frame, cut, geom, torso)[0].keypoints


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--poses", type=Path, required=True, help="folder with <video>.pose.npz from run_ingest.py")
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--videos", default="V2,V5,V9")
    ap.add_argument("--extent", type=Path, default=Path("results/punch_extent.json"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    videos = a.videos.split(",")
    model, man = classify.load_classifier(a.classifier)
    ex = classify.read_extent(a.extent, man["geometry"].variant)

    # Gate 1: the reported model and input path, on the corpus's own held-out clips.
    held = (man.get("held_out") or {})
    # The checkpoint's own held-out clips: scoring it on the athlete-disjoint set
    # would score it on V3, which it trained on.
    _, val = split_pipeline(load_corpus(a.data))
    held_out = sorted({c.source for c in val})
    if not set(videos) <= set(held_out) or len(set(videos)) != len(videos):
        raise SystemExit(f"--videos must be distinct held-out videos ({held_out}), got {videos}")
    xv, yv = to_arrays(val)
    gate = scores(yv, predict(model, man, list(xv)))
    print(f"gate 1: corpus held-out clips, accuracy {gate['accuracy']:.4f} macro F1 {gate['macro_f1']:.4f} "
          f"(manifest {held.get('accuracy')} / {held.get('macro_f1')})")
    if not (isinstance(held.get("accuracy"), float) and abs(gate["accuracy"] - held["accuracy"]) < TOL
            and abs(gate["macro_f1"] - held["macro_f1"]) < TOL):
        raise SystemExit("GATE 1 FAILED: this classifier does not reproduce its own held-out numbers; nothing scored")

    rows, report = [], {"classifier_sha256": man["sha256"], "gate_corpus_held_out": gate, "videos": {},
                        "held_out_videos": held_out, "excluded_videos": {}}
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
            why = "its spans do not align with this video's frames (no offset is clearly better than the others)"
            report["excluded_videos"][v] = why
            print(f"GATE 3 FAILED for {v}: {why}. {v} IS EXCLUDED; nothing from it is scored.")
            continue
        o = al["best"]
        geom = man["geometry"]
        track = punchfind.bridge(pose.to_track(usable_only=False), int(round(punchfind.BRIDGE_S * pose.fps)))
        torso = punchfind.body_scale(track)
        sig = punchfind.wrist_signal(track, geom.variant)
        row_of = {int(f): i for i, f in enumerate(track.frame_ids)}
        peaks = []
        for c in clips:
            pk = span_peak(track, sig, c.start + o, c.end + o)
            i = row_of.get(pk)
            peaks.append((pk, int(np.argmax(sig[i]))) if i is not None and sig[i].max() > 0 else None)
        clip_cut = classify.clip_cut(track, sig, [p_ for p_ in peaks if p_ is not None], ex, geom)
        info["clip_cut"] = None if clip_cut is None else clip_cut._asdict()
        for c, pk_hand in zip(clips, peaks):
            pk = span_peak(track, sig, c.start + o, c.end + o)
            win = classify.cut_window(track, pk, geom, torso)
            ext_kp, long_kp, iou = None, None, None
            if pk_hand is not None:
                e = classify.punch_extent(track, sig, pk, pk_hand[1], ex, geom)
                ext_kp = classify.cut_span(track, e.start, e.end, geom, torso).keypoints
                iou = punchfind.span_iou((e.start, e.end), (c.start + o, c.end + o))
                long_kp = y_clip(track, pk, clip_cut, geom, torso)
            y = poseswitch.yolo_clip(pose, c.start + o, c.end + o)
            y_all = poseswitch.yolo_clip(pose, c.start + o, c.end + o, None)
            rows.append({"video": v, "label": c.label, "A": c.alphapose, "Y": y, "Y_all": y_all,
                         "Y_fix": win.keypoints, "Y_ext": ext_kp, "Y_clip": long_kp, "iou": iou,
                         "seen": int((y[..., 2] > 0).any(axis=1).sum()) >= 2,
                         "seen_all": int((y_all[..., 2] > 0).any(axis=1).sum()) >= 2,
                         "same": poseswitch.same_person(c, y_all)})

    scored = [v for v in videos if v not in report["excluded_videos"]]
    report["scored_videos"] = scored
    if not rows:
        raise SystemExit("GATE 3 FAILED for every video; nothing scored")
    report.update(aggregate(rows, lambda clips: predict(model, man, clips), scored))
    report["limits"] = [
        "the per-clip cut (Y_clip) was CHOSEN looking at these videos' numbers and at V6's second half "
        "(ADR-0014, 2026-10-13): this is no longer a clean test of it",
        "held-out videos are coaches' YouTube footage at 24-30 fps, not phone footage",
        "V5 and V9 are each a person seen in no other video (O55): athlete-disjoint; gate 1 re-scores the "
        "checkpoint on V2/V5/V9, and V2 is NOT (its instructor trains in V3)",
        "YOLO keeps one person per frame; other-person clips are counted, and scored in all_seen"]
    for line in summary_lines(report):
        print(line)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, allow_nan=False, default=float), encoding="utf-8")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
