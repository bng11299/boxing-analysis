"""The solo pipeline end to end on continuous footage with all six punch types.

    python eval_pipeline_v5v9.py --data data --poses results/ingest \\
        --classifier weights/classifier/stgcnpp-boxingvi-seed0.json --stance orthodox \\
        [--videos-dir data/rgb]                       # also render the label-completeness audit
        [--judgments results/e2e_v5v9/audit/judgments.csv]   # ... and fold in its judgments

V5 and V9 are held-out BoxingVI videos: coaches' landscape YouTube footage at
29.97 and 23.976 fps, YOLO11-pose run on the downloaded source. This runs the
pipeline exactly as a user would, through its own entry points:
find_punches.main (the V6-tuned wrist-speed rule) then classify_punches.main (the
saved ST-GCN++, one window per clip at the median of its punches' extents), and
scores what they wrote against every annotated span. Until now the finder had
only been scored on V6, which holds hooks and uppercuts: its recall on jabs and
crosses had never been measured.

Gates, in order; each stops the run with nothing scored:
1. the classifier is the one results/pose_switch.json was measured with (hash);
2. each pose is at the source's native frame rate and covers every span;
3. the spans align with the pose's frame numbering at offset 0, sharply
   (poseswitch.align, eval_pose_switch.py's gate 3);
4. this script's own perfect-finder readings of the same punches reproduce
   pose_switch.json exactly: labelled spans (Y_all, 0.889) and the per-clip cut
   at each span's wrist-speed peak (Y_clip, 0.826), on the same 742 punches;
5. the classified CSV is the one classify_punches.py wrote in this run (its
   sidecar's hash, clip, stance and classifier).

Ground truth: every span poseswitch.held_out_clips returns. The 5 annotated rows
it leaves out (a span whose length does not match its clip, or that ends before
it starts) are real punches whose frames cannot be trusted: a detection on one
is neither a hit nor a false alarm, and is counted.

Precision counts every unmatched detection as false, which is only right if
every punch in the video is labelled. The audit tests that: a blinded sample of
detections more than 0.3 s from any label, with matched detections mixed in as
controls, rendered as frame strips for a person to judge.

Intervals: finder numbers by a block bootstrap over 30 s of video, resampled
within each video; type numbers by a paired clip-level bootstrap (as
pose_switch.json) and by the same paired block bootstrap, which is the honest
one when neighbouring punches are not independent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import classify_punches  # noqa: E402
import find_punches  # noqa: E402
from boxingvi import classify, poseswitch, punchfind  # noqa: E402
from boxingvi.data import (CLASSES, N_CLASSES, NATIVE_FPS, Span, _class_column, _normalise_label,  # noqa: E402
                           _numeric_columns_left_of)
from boxingvi.pose_yolo import PoseSequence  # noqa: E402
from boxingvi.stats import bootstrap_gap_ci  # noqa: E402

VIDEOS = ("V5", "V9")
DEFAULT_OUT = Path("results/e2e_v5v9")
DEFAULT_REFERENCE = Path("results/pose_switch.json")
FAR_S = 0.3            # an unmatched detection this far from every label is audited
BLOCK_S = 30.0         # block length for the block bootstrap (V6 used 719 frames, 30 s)
MIN_SHIFT_S = find_punches.MIN_SHIFT_S
AUDIT_FAR = 25         # far detections sampled per video
AUDIT_CONTROLS = 5     # matched detections mixed in per video, blind
AUDIT_SEED = 20261019
JUDGMENTS = ("punch", "not_punch", "unclear")
REF_TOL = 1e-9
MAX_CSV_BYTES = 16 * 1024 * 1024
SIDE = np.array([classify.class_side(c) for c in CLASSES])


# ------------------------------------------------------------ ground truth ---

def unusable_rows(data: Path, video: str, kept: "set[int]") -> "list[tuple[int, int, int]]":
    """(row, first, last) frames of the annotated punches held_out_clips left out. A row
    with a label and no frame numbers at all cannot be placed and stops the run: silently
    losing it would count a detection on a real punch as a false alarm."""
    import pandas as pd

    df = pd.read_excel(Path(data) / "Annotation_files" / f"{video}.xlsx")
    cc = _class_column(df)
    sc, ec = _numeric_columns_left_of(df, cc)
    out = []
    for i in range(len(df)):
        if i in kept or _normalise_label(df[cc].iloc[i]) is None:
            continue
        s, e = pd.to_numeric(df[sc].iloc[i], errors="coerce"), pd.to_numeric(df[ec].iloc[i], errors="coerce")
        if not (np.isfinite(s) and np.isfinite(e)):
            raise ValueError(f"{video} row {i}: a labelled punch with no frames ({s!r}..{e!r})")
        out.append((i, int(min(s, e)), int(max(s, e))))
    return out


def check_alignment(video: str, al: dict) -> None:
    if not al["sharp"] or al["best"] != 0:
        raise SystemExit(f"GATE 3 FAILED: {video} spans are not aligned at offset 0 (best {al['best']}, "
                         f"sharpness {al['ratio']}); the finder's frames cannot be scored against them")


def check_reference(ref: dict, n: int, acc_all: float, acc_clip: float, sha: str) -> None:
    """Gate 4: the perfect-finder readings computed here are pose_switch.json's, to the bit."""
    w = ref.get("window") or {}
    want = {"Y_all": acc_all, "Y_clip": acc_clip}
    for k, got in want.items():
        r = w.get(k) or {}
        if r.get("n") != n or not isinstance(r.get("accuracy"), float) or abs(r["accuracy"] - got) > REF_TOL:
            raise SystemExit(f"GATE 4 FAILED: {k} here is {got:.6f} on {n}; the reference has "
                             f"{r.get('accuracy')} on {r.get('n')}: not the same readings, nothing scored")
    if ref.get("classifier_sha256") != sha:
        raise SystemExit("GATE 1 FAILED: the reference was measured with another classifier")


# ------------------------------------------------- what the pipeline wrote ---

def read_classified(csv_path: Path, sidecar: Path, clip: str, n_frames: int, fps: float, stance: str,
                    classifier_sha: str, rule: "dict | None" = None) -> "list[dict]":
    """classify_punches.py's CSV, bound to its sidecar: the hash of these exact bytes, this
    clip, this stance and this classifier. Every field this script uses is re-validated."""
    raw = csv_path.read_bytes()
    if len(raw) > MAX_CSV_BYTES:
        raise ValueError(f"{csv_path.name}: larger than {MAX_CSV_BYTES} bytes")
    side = json.loads(sidecar.read_text(encoding="utf-8"), parse_constant=classify._no_constant)
    if not isinstance(side, dict):
        raise ValueError(f"{sidecar.name}: expected a JSON object")
    want = {"csv_sha256": hashlib.sha256(raw).hexdigest(), "clip": clip, "stance": stance,
            "classifier_sha256": classifier_sha, "pose_frames": n_frames}
    if rule is not None:
        want["rule"] = rule
    for k, v in want.items():
        if side.get(k) != v:
            raise ValueError(f"{sidecar.name}: {k} is {side.get(k)!r}, expected {v!r}: "
                             "this CSV is not the one classified in this run")
    rows = list(csv.reader(raw.decode("utf-8").splitlines()))
    if not rows or rows[0] != classify_punches.OUT_HEADER:
        raise ValueError(f"{csv_path.name}: not a classify_punches.py CSV")
    out = []
    for i, r in enumerate(rows[1:], start=2):
        if len(r) != len(classify_punches.OUT_HEADER):
            raise ValueError(f"{csv_path.name} line {i}: {len(r)} fields")
        d = dict(zip(classify_punches.OUT_HEADER, r))
        if not classify_punches._DIGITS.fullmatch(d["frame"]) or not 0 <= int(d["frame"]) < n_frames:
            raise ValueError(f"{csv_path.name} line {i}: frame {d['frame']!r} is not a frame of {clip}")
        try:
            t_ = float(d["time_s"])
        except ValueError:
            raise ValueError(f"{csv_path.name} line {i}: time {d['time_s']!r} is not a number") from None
        if not math.isfinite(t_) or abs(t_ - int(d["frame"]) / fps) > 2e-3:
            raise ValueError(f"{csv_path.name} line {i}: time does not match the frame at {fps:.3f} fps")
        if d["hand"] not in ("left", "right") or d["lead_rear"] != classify.hand_side(d["hand"], stance):
            raise ValueError(f"{csv_path.name} line {i}: hand {d['hand']!r} / {d['lead_rear']!r}")
        if d["punch_type"] != "unclassified" and d["punch_type"] not in CLASSES:
            raise ValueError(f"{csv_path.name} line {i}: punch type {d['punch_type']!r}")
        out.append({"frame": int(d["frame"]), "hand": d["hand"], "lead_rear": d["lead_rear"],
                    "type": None if d["punch_type"] == "unclassified" else CLASSES.index(d["punch_type"]),
                    "note": d["note"]})
    return out


# ----------------------------------------------------------------- scoring ---

def distance_to(frames, intervals: "list[tuple[int, int]]") -> np.ndarray:
    """Frames from each frame to the nearest inclusive interval (0 inside one)."""
    f = np.asarray(frames, dtype=np.int64)
    if not intervals:
        return np.full(len(f), np.iinfo(np.int64).max)
    iv = np.asarray(intervals, dtype=np.int64)
    d = np.maximum(np.maximum(iv[None, :, 0] - f[:, None], f[:, None] - iv[None, :, 1]), 0)
    return d.min(axis=1)


def match_nearest(det, spans: "list[Span]", tol: int) -> "list[tuple[int, int]]":
    """A maximum one-to-one matching of detections to spans, as punchfind.match, that
    among all maximum matchings has the least total distance from each detection to its
    span's MIDDLE (ties broken by the solver, deterministically).

    Why not punchfind.match's: it takes spans in order of end and hands each the earliest
    free detection that fits, which maximises the count but not who gets what. In a fast
    jab-cross the finder (0.4 s minimum gap) reports one peak, usually the cross; that
    peak also fits inside the jab's span plus tolerance, and the jab ends first, so the
    jab is credited with the cross's detection. On V5 147 of 356 matched detections sat
    off their span's own wrist-speed peak, 115 of them on another labelled punch's: a
    window correctly read as a cross was scored against 'jab'. Counts (recall, precision)
    are the same under both; per-type recall and type accuracy are not, and both are
    reported. Solved as an assignment: an infeasible pair costs more than every feasible
    pair together, so the count is maximised first and the distance second."""
    det = np.asarray(det, dtype=np.int64)
    if not len(det) or not spans:
        return []
    from scipy.optimize import linear_sum_assignment

    st = np.array([s.start for s in spans], dtype=np.int64)
    en = np.array([s.end for s in spans], dtype=np.int64)
    ok = (det[None, :] >= st[:, None] - tol) & (det[None, :] <= en[:, None] + tol)
    dist = np.abs(det[None, :] - (st[:, None] + en[:, None]) / 2)
    big = float(dist[ok].sum()) + 1.0 if ok.any() else 1.0
    cost = np.where(ok, dist, big)
    ri, ci = linear_sum_assignment(cost)
    pairs = [(int(s), int(d)) for s, d in zip(ri, ci) if ok[s, d]]
    if len(pairs) != len(punchfind.match(det, spans, tol)):   # both are maximum matchings
        raise AssertionError("nearest matching is not a maximum matching")
    return sorted(pairs)


def split_detections(det, spans: "list[Span]", unusable: "list[tuple[int, int]]", tol: int, far: int,
                     matcher=match_nearest) -> dict:
    """Which detections are hits, which sit on an unusable labelled row (ignored), and
    which unmatched ones are far (> `far` frames) from every label of either kind."""
    det = np.asarray(det, dtype=np.int64)
    pairs = matcher(det, spans, tol)
    hit = np.zeros(len(det), bool)
    hit[[j for _, j in pairs]] = True
    on_unusable = ~hit & (distance_to(det, unusable) <= tol)
    every = [(s.start, s.end) for s in spans] + list(unusable)
    far_mask = ~hit & ~on_unusable & (distance_to(det, every) > far)
    return {"pairs": pairs, "hit": hit, "ignored": on_unusable, "far": far_mask}


def confusion(y, p) -> np.ndarray:
    cm = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    np.add.at(cm, (np.asarray(y, int), np.asarray(p, int)), 1)
    return cm


def cm_accuracy(cm: np.ndarray) -> "float | None":
    n = cm.sum()
    return float(np.trace(cm) / n) if n else None


def cm_macro_f1(cm: np.ndarray) -> "float | None":
    """Macro F1 over all six classes, zero for a class with no support and no
    prediction: sklearn's f1_score(labels=range(6), zero_division=0), as pose_switch.json."""
    if not cm.sum():
        return None
    tp = np.diag(cm).astype(float)
    den = cm.sum(axis=0) + cm.sum(axis=1)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))


def cm_hand(cm: np.ndarray) -> "float | None":
    n = cm.sum()
    return float(cm[SIDE[:, None] == SIDE[None, :]].sum() / n) if n else None


def block_ci(counts: "dict[str, np.ndarray]", stat, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile interval for stat(sum of per-block count vectors), resampling whole
    blocks WITH replacement inside each stratum (a video): each draw holds as many
    blocks of V5 as V5 has, and of V9 as V9 has. Every statistic here is a function of
    summed counts (hits, labels, confusion cells), so a block carries its punches'
    pairing with it. Draws where the statistic is undefined are dropped and counted."""
    point = stat(sum(c.sum(axis=0) for c in counts.values()))
    if point is None:
        raise ValueError("the statistic is undefined on the data itself")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        tot = sum(c[rng.integers(0, len(c), size=len(c))].sum(axis=0) for c in counts.values())
        v = stat(tot)
        if v is not None:
            vals.append(v)
    if len(vals) < n_boot // 2:
        raise ValueError(f"only {len(vals)} of {n_boot} resamples were defined")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_boot": n_boot,
            "n_undefined_dropped": n_boot - len(vals), "blocks": {k: len(c) for k, c in counts.items()}}


def wilson(k: int, n: int, z: float = 1.959964) -> "list[float] | None":
    if n == 0:
        return None
    p = k / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    m = (p + z * z / (2 * n)) / (1 + z * z / n)
    return [max(0.0, m - c), min(1.0, m + c)]


def _ratio(a, b):
    return float(a / b) if b else None


def tally(v: str, pairs, det: np.ndarray, scored_det: np.ndarray, spans: "list[Span]", rows: "list[dict]",
          refs: dict, block: int, n_blocks: int) -> dict:
    """Per-block counts for one pairing of detections with labelled spans.

    fin: detections, labels, hits (by the detection's block), hits (by the span's block),
         then labels per type and hits per type (by the span's block).
    typ: three 6x6 confusion matrices flattened (pipeline, Y_clip, Y_all) on matched
         detections the classifier named, by the span's block: the same labelled punch
         read three ways, so every gap between them is paired."""
    fin = np.zeros((n_blocks, 4 + 2 * N_CLASSES), np.int64)
    np.add.at(fin[:, 0], scored_det // block, 1)
    for s in spans:
        fin[s.start // block, 1] += 1
        fin[s.start // block, 4 + s.label] += 1
    typ = np.zeros((n_blocks, 3 * N_CLASSES * N_CLASSES), np.int64)
    out_pairs, unclassified, no_reference, stance = [], 0, 0, []
    for si, di in pairs:
        sp_, r = spans[si], rows[di]
        b = sp_.start // block
        fin[det[di] // block, 2] += 1
        fin[b, 3] += 1
        fin[b, 4 + N_CLASSES + sp_.label] += 1
        stance.append(r["lead_rear"] == SIDE[sp_.label])
        if r["type"] is None:
            unclassified += 1
            continue
        if (v, si) not in refs:   # no perfect-finder reading of this span: not pairable, counted
            no_reference += 1
            continue
        pa, pc = refs[(v, si)]
        for m, p in enumerate((r["type"], pc, pa)):
            typ[b, m * 36 + sp_.label * N_CLASSES + p] += 1
        out_pairs.append((v, sp_.label, r["type"], pc, pa))
    return {"fin": fin, "typ": typ, "pairs": out_pairs, "unclassified": unclassified, "no_reference": no_reference,
            "stance": stance}


def cm_from(t: np.ndarray, m: int) -> np.ndarray:
    return t[m * 36:(m + 1) * 36].reshape(N_CLASSES, N_CLASSES)


def type_summary(typ: "dict[str, np.ndarray]") -> dict:
    tot = sum(c.sum(axis=0) for c in typ.values())
    return {name: {"accuracy": cm_accuracy(cm_from(tot, m)), "macro_f1": cm_macro_f1(cm_from(tot, m)),
                   "hand_accuracy": cm_hand(cm_from(tot, m))}
            for m, name in enumerate(("pipeline", "Y_clip", "Y_all"))} | {"n": int(cm_from(tot, 0).sum())}


def recall_by_type(fin: "dict[str, np.ndarray]", with_ci: bool) -> dict:
    tot = sum(c.sum(axis=0) for c in fin.values())
    out = {}
    for t in range(N_CLASSES):
        lab, hit = 4 + t, 4 + N_CLASSES + t
        out[CLASSES[t]] = {"labelled": int(tot[lab]), "found": int(tot[hit]), "recall": _ratio(tot[hit], tot[lab])}
        if with_ci and tot[lab]:
            out[CLASSES[t]]["ci"] = block_ci(fin, lambda x, h=hit, l_=lab: _ratio(x[h], x[l_]))
    return out


# ------------------------------------------------------------------- audit ---

def audit_sample(per_video: "dict[str, dict]", n_far: int = AUDIT_FAR, n_controls: int = AUDIT_CONTROLS,
                 seed: int = AUDIT_SEED) -> "list[dict]":
    """Far detections and matched controls per video, shuffled together and numbered, so
    whoever judges the frames cannot tell which is which. per_video[v] = {"far": [(frame,
    hand)], "matched": [(frame, hand)]}."""
    rng = np.random.default_rng(seed)
    items = []
    for v in sorted(per_video):
        for kind, n in (("far", n_far), ("control", n_controls)):
            pool = per_video[v]["far" if kind == "far" else "matched"]
            for k in sorted(rng.choice(len(pool), size=min(n, len(pool)), replace=False).tolist()):
                items.append({"video": v, "frame": int(pool[k][0]), "hand": pool[k][1], "kind": kind})
    order = rng.permutation(len(items))
    return [{"id": f"s{i:03d}", **items[j]} for i, j in enumerate(order)]


def key_digest(items: "list[dict]") -> str:
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


JUDGMENT_HEADER = ["id", "sample_digest", "judgment", "note"]


def read_judgments(path: Path, items: "list[dict]") -> "dict[str, str]":
    """One judgment per sampled id, nothing more, nothing less, from the fixed set, and
    every row carrying the digest of THIS sample: ids s000.. repeat in every sample, so a
    file judged on another sample would otherwise be counted against other detections."""
    digest = key_digest(items)
    with path.open(newline="", encoding="utf-8-sig") as f:   # -sig: a file saved by Excel starts with a BOM
        r = csv.reader(f)
        if next(r, None) != JUDGMENT_HEADER:
            raise ValueError(f"{path.name}: header must be {','.join(JUDGMENT_HEADER)}")
        got = {}
        for i, row in enumerate(r, start=2):
            if len(got) > len(items):
                raise ValueError(f"{path.name}: more rows than the {len(items)} sampled strips")
            if len(row) != len(JUDGMENT_HEADER):
                raise ValueError(f"{path.name} line {i}: {len(row)} fields")
            if row[1] != digest:
                raise ValueError(f"{path.name} line {i}: judged on another audit sample (digest {row[1][:12]!r}, "
                                 f"this run's is {digest[:12]!r}); re-render and judge again")
            if row[0] in got:
                raise ValueError(f"{path.name} line {i}: {row[0]} judged twice")
            if row[2] not in JUDGMENTS:
                raise ValueError(f"{path.name} line {i}: judgment {row[2]!r} is not one of {JUDGMENTS}")
            got[row[0]] = row[2]
    ids = {it["id"] for it in items}
    if set(got) != ids:
        raise ValueError(f"{path.name}: judged {sorted(set(got) - ids)} not in the sample, "
                         f"missing {sorted(ids - set(got))}")
    return got


def write_png(path: Path, img: np.ndarray) -> None:
    """(H, W, 3) uint8 as an 8-bit RGB PNG, stdlib only (no imaging library in this env)."""
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _resize(img: np.ndarray, h: int) -> np.ndarray:
    w = max(1, round(img.shape[1] * h / img.shape[0]))
    ys = np.minimum((np.arange(h) * img.shape[0] / h).astype(int), img.shape[0] - 1)
    xs = np.minimum((np.arange(w) * img.shape[1] / w).astype(int), img.shape[1] - 1)
    return img[ys][:, xs]


def render_audit(video_path: Path, pose: PoseSequence, items: "list[dict]", out_dir: Path, tile_h: int = 300) -> None:
    """One strip per sampled detection: the whole frame at the peak (the tracked person
    boxed), then the person cropped at -0.3 .. +0.3 s in 0.1 s steps, YOLO's skeleton
    drawn and the detected wrist ringed. Nothing on it says whether it was labelled."""
    from boxingvi import overlay, video

    if not items:
        return
    info = video.probe(video_path)
    # Another encode of the "same" video puts every strip at the wrong moment, and a judge
    # would call the strips not-punches: the labels would look more complete than they are.
    if (info.width, info.height) != (pose.width, pose.height) or abs(info.fps - pose.fps) > 1e-3 or info.vfr:
        raise SystemExit(f"{video_path.name} is {info.width}x{info.height} at {info.fps:.3f} fps"
                         f"{' (variable rate)' if info.vfr else ''}; the pose is {pose.width}x{pose.height} at "
                         f"{pose.fps:.3f}: not the video the pose was made from")
    step = max(1, int(round(0.1 * pose.fps)))
    plan = {}
    for it in items:
        frames = [it["frame"] + k * step for k in range(-3, 4)]
        frames = [f for f in frames if 0 <= f < len(pose)]
        pts = np.concatenate([pose.xy[f][pose.conf[f] > 0.3] for f in frames if pose.detected[f]] or [np.zeros((0, 2))])
        if len(pts):
            (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
            pad = 0.25 * max(x1 - x0, y1 - y0, 50)
            box = (int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(pose.width, x1 + pad)),
                   int(min(pose.height, y1 + pad)))
        if not len(pts) or box[2] - box[0] < 8 or box[3] - box[1] < 8:   # nothing to crop to: the whole frame
            box = (0, 0, pose.width, pose.height)
        plan[it["id"]] = {"frames": frames, "box": box, "tiles": {}, "full": None}
    want = {}
    for sid, p in plan.items():
        for f in p["frames"]:
            want.setdefault(f, []).append(sid)
    wrist = {"left": 9, "right": 10}
    for i, fr in enumerate(video.iter_frames(video_path, max_frames=max(want) + 1)):
        for sid in want.get(i, []):
            p, it = plan[sid], next(x for x in items if x["id"] == sid)
            img = fr.copy()
            if pose.detected[i]:
                overlay.draw_skeleton(img, pose.xy[i], pose.conf[i])
                overlay.draw_line(img, pose.xy[i, wrist[it["hand"]]], pose.xy[i, wrist[it["hand"]]], (255, 230, 0), 9)
            x0, y0, x1, y1 = p["box"]
            p["tiles"][i] = _resize(img[y0:y1, x0:x1], tile_h)
            if i == it["frame"]:
                full = img.copy()
                full[y0:y0 + 6, x0:x1] = full[max(y1 - 6, 0):y1, x0:x1] = (255, 0, 255)
                full[y0:y1, x0:x0 + 6] = full[y0:y1, max(x1 - 6, 0):x1] = (255, 0, 255)
                p["full"] = _resize(full, tile_h)
    for sid, p in plan.items():
        tiles = [p["full"]] + [p["tiles"][f] for f in p["frames"]]
        if any(t is None for t in tiles):
            raise RuntimeError(f"{video_path.name}: a frame of sample {sid} was not decoded")
        mid = 1 + p["frames"].index(next(x for x in items if x["id"] == sid)["frame"])
        tiles[mid] = tiles[mid].copy()
        tiles[mid][:5] = tiles[mid][-5:] = (0, 255, 0)
        gap = np.full((tile_h, 6, 3), 255, np.uint8)
        strip = np.concatenate([x for t in tiles for x in (t, gap)][:-1], axis=1)
        write_png(out_dir / f"{sid}.png", strip)


# -------------------------------------------------------------------- main ---

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--poses", type=Path, required=True)
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--stance", choices=sorted(classify.STANCES), required=True)
    ap.add_argument("--rule", type=Path, default=find_punches.DEFAULT_RULE)
    ap.add_argument("--extent", type=Path, default=classify_punches.DEFAULT_EXTENT)
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--videos-dir", type=Path, help="data/rgb: render the audit strips from <video>.mp4")
    ap.add_argument("--judgments", type=Path, help="id,judgment,note for every audit sample")
    a = ap.parse_args(argv)

    model, man = classify.load_classifier(a.classifier)
    geom = man["geometry"]
    ex = classify.read_extent(a.extent, geom.variant)
    ref = json.loads(a.reference.read_text(encoding="utf-8"), parse_constant=classify._no_constant)
    if not isinstance(ref, dict) or not isinstance(ref.get("window"), dict):
        raise SystemExit(f"GATE 4 FAILED: {a.reference.name} is not a pose_switch.json results file")
    if ref.get("classifier_sha256") != man["sha256"]:
        raise SystemExit("GATE 1 FAILED: results/pose_switch.json was measured with another classifier")

    punches_dir = a.out_dir / "punches"
    per = {}
    ref_rows = []      # (video, span index, label, Y_all keypoints, Y_clip keypoints, frames seen under 0.3)
    for v in VIDEOS:
        pose_path = a.poses / f"{v}.pose.npz"
        pose = PoseSequence.load(pose_path)
        clips, mismatched = poseswitch.held_out_clips(a.data, v)
        if abs(pose.fps - NATIVE_FPS[v]) > 1e-3:
            raise SystemExit(f"GATE 2 FAILED: {v} pose is {pose.fps:.3f} fps, the source is {NATIVE_FPS[v]:.3f}")
        bad = unusable_rows(a.data, v, {c.index for c in clips})
        if len(bad) != mismatched or max(c.end for c in clips) >= len(pose) or any(e >= len(pose) for _, _, e in bad):
            raise SystemExit(f"GATE 2 FAILED: {v} spans run past the pose's {len(pose)} frames, "
                             "or the left-out rows do not reconcile")
        al = poseswitch.align(pose, clips)
        check_alignment(v, al)

        # The perfect-finder readings, exactly as eval_pose_switch.py makes them.
        track = punchfind.bridge(pose.to_track(usable_only=False), int(round(punchfind.BRIDGE_S * pose.fps)))
        torso = punchfind.body_scale(track)
        sig = punchfind.wrist_signal(track, geom.variant)
        row_of = {int(f): i for i, f in enumerate(track.frame_ids)}
        peaks = []
        for c in clips:
            pk = classify_punches.span_peak(track, sig, c.start, c.end)
            i = row_of.get(pk)
            peaks.append((pk, int(np.argmax(sig[i]))) if i is not None and sig[i].max() > 0 else None)
        cut = classify.clip_cut(track, sig, [p for p in peaks if p is not None], ex, geom)
        for k, (c, p) in enumerate(zip(clips, peaks)):
            y_all = poseswitch.yolo_clip(pose, c.start, c.end, None)
            y_clip = (classify.cut_span(track, *classify.clip_window(p[0], cut, pose.fps), geom, torso).keypoints
                      if p is not None else None)
            seen = int((poseswitch.yolo_clip(pose, c.start, c.end)[..., 2] > 0).any(axis=1).sum())
            ref_rows.append((v, k, c.label, y_all, y_clip, seen))

        # The pipeline, through its own entry points, into this run's folder. Old outputs go first,
        # so everything read below was written by this run.
        for old in punches_dir.glob(f"{v}.*") if punches_dir.is_dir() else []:
            old.unlink()
        if find_punches.main(["--pose", str(pose_path), "--rule", str(a.rule), "--out-dir", str(punches_dir)]):
            raise SystemExit(f"find_punches.py failed on {v}")
        if classify_punches.main(["--pose", str(pose_path), "--punches", str(punches_dir / f"{v}.punches.csv"),
                                  "--classifier", str(a.classifier), "--stance", a.stance, "--rule", str(a.rule),
                                  "--extent", str(a.extent), "--out-dir", str(punches_dir)]):
            raise SystemExit(f"classify_punches.py failed on {v}")
        rows = read_classified(punches_dir / f"{v}.classified.csv", punches_dir / f"{v}.classified.json", v,
                               len(pose), pose.fps, a.stance, man["sha256"], find_punches.load_rule(a.rule))
        per[v] = {"pose": pose, "clips": clips, "unusable": bad, "rows": rows, "alignment": al,
                  "perfect_finder_cut": None if cut is None else cut._asdict(),
                  "pipeline_cut": json.loads((punches_dir / f"{v}.classified.json").read_text(encoding="utf-8"))["cut"]}

    # Gate 4: the perfect-finder readings, pooled as pose_switch.json pools them.
    # eval_pose_switch.py's own selection: YOLO saw >= 2 frames under the old 0.3 rule (`seen`) AND
    # with every joint (`seen_all`), and every cut exists. Y_fix / Y_ext are not read here; their
    # cuts exist wherever Y_clip's does (same peak), which gate 4's count then confirms.
    keep = [r for r in ref_rows if r[5] >= 2 and int((r[3][..., 2] > 0).any(axis=1).sum()) >= 2 and r[4] is not None]
    pred_all = np.argmax(classify.predict_proba(model, [r[3] for r in keep], man["t_out"]), axis=1)
    pred_clip = np.argmax(classify.predict_proba(model, [r[4] for r in keep], man["t_out"]), axis=1)
    lab = np.array([r[2] for r in keep])
    check_reference(ref, len(keep), float((pred_all == lab).mean()), float((pred_clip == lab).mean()), man["sha256"])
    print(f"gate 4: perfect-finder readings reproduce pose_switch.json: Y_all {(pred_all == lab).mean():.3f}, "
          f"Y_clip {(pred_clip == lab).mean():.3f} on {len(keep)}")
    refs = {(r[0], r[1]): (int(pa), int(pc)) for r, pa, pc in zip(keep, pred_all, pred_clip)}

    # ---- score
    report = {"stance": a.stance, "classifier_sha256": man["sha256"], "rule": find_punches.load_rule(a.rule),
              "extent": ex, "matching": "nearest (primary): " + match_nearest.__doc__.split("\n")[0].strip(),
              "videos": {}, "limits": []}
    tallies = {"nearest": {}, "greedy_v6_convention": {}}
    audit_pools = {}
    for v, d in per.items():
        pose, clips, rows = d["pose"], d["clips"], d["rows"]
        fps, n = pose.fps, len(pose)
        tol, far = int(round(find_punches.TOL_S * fps)), int(round(FAR_S * fps))
        spans = [Span(c.start, c.end, c.label) for c in clips]
        det = np.array([r["frame"] for r in rows], dtype=np.int64)
        unusable = [(s, e) for _, s, e in d["unusable"]]
        sp = split_detections(det, spans, unusable, tol, far)
        scored_det = det[~sp["ignored"]]
        whole = (0, n - 1)
        s = punchfind.score(scored_det, spans, tol, whole)
        if s["tp"] != len(sp["pairs"]):
            raise AssertionError(f"{v}: punchfind.score and the nearest matching disagree on the hit count")
        floor = punchfind.shift_floor(scored_det, spans, tol, whole, int(round(MIN_SHIFT_S * fps)))
        block = int(round(BLOCK_S * fps))
        n_blocks = (n - 1) // block + 1
        t = tally(v, sp["pairs"], det, scored_det, spans, rows, refs, block, n_blocks)
        g = tally(v, [(si, di) for si, di in punchfind.match(det, spans, tol) if not sp["ignored"][di]],
                  det, scored_det, spans, rows, refs, block, n_blocks)
        tallies["nearest"][v], tallies["greedy_v6_convention"][v] = t, g
        unmatched = ~sp["hit"] & ~sp["ignored"]
        audit_pools[v] = {"far": [(int(det[j]), rows[j]["hand"]) for j in np.flatnonzero(sp["far"])],
                          "matched": [(int(det[j]), rows[j]["hand"]) for _, j in sp["pairs"]]}
        first, last = min(x.start for x in spans), max(x.end for x in spans)
        report["videos"][v] = {
            "frames": n, "fps": fps, "duration_s": n / fps, "tolerance_frames": tol,
            "labelled_spans": len(spans), "unusable_labelled_rows": [list(x) for x in d["unusable"]],
            "first_label_s": first / fps, "last_label_s": last / fps,
            "alignment": {"best": d["alignment"]["best"], "sharpness": d["alignment"]["ratio"]},
            "pipeline_cut": d["pipeline_cut"], "perfect_finder_cut": d["perfect_finder_cut"],
            "detections": int(len(det)), "detections_on_unusable_rows_ignored": int(sp["ignored"].sum()),
            "unclassified_detections": int(sum(r["type"] is None for r in rows)),
            "finder": {**s, "shift_floor": floor,
                       "block_ci": {"precision": block_ci({v: t["fin"]}, lambda x: _ratio(x[2], x[0])),
                                    "recall": block_ci({v: t["fin"]}, lambda x: _ratio(x[3], x[1]))},
                       "recall_by_type": recall_by_type({v: t["fin"]}, False),
                       "recall_by_type_greedy_v6_convention": recall_by_type({v: g["fin"]}, False),
                       "unmatched": int(unmatched.sum()),
                       "unmatched_within_far_s_of_a_label": int((unmatched & ~sp["far"]).sum()),
                       "unmatched_far_from_every_label": int(sp["far"].sum()),
                       "far_detections_before_first_label": int((det[sp["far"]] < first).sum()),
                       "far_detections_after_last_label": int((det[sp["far"]] > last).sum())},
            "matched_unclassified": t["unclassified"], "matched_without_reference_reading": t["no_reference"],
            "type": type_summary({v: t["typ"]}),
            "type_greedy_v6_convention": type_summary({v: g["typ"]}),
            "finder_hand_as_lead_rear_matches_label_side": {
                "stance": a.stance, "n": len(t["stance"]),
                "share": float(np.mean(t["stance"])) if t["stance"] else None}}

    # ---- pooled
    fin = {v: x["fin"] for v, x in tallies["nearest"].items()}
    typ = {v: x["typ"] for v, x in tallies["nearest"].items()}
    allfc = sum(c.sum(axis=0) for c in fin.values())
    pooled = {"detections_scored": int(allfc[0]), "labelled": int(allfc[1]), "hits": int(allfc[2]),
              "precision": _ratio(allfc[2], allfc[0]), "recall": _ratio(allfc[3], allfc[1]),
              "count_ratio": _ratio(allfc[0], allfc[1]),
              "precision_ci": block_ci(fin, lambda x: _ratio(x[2], x[0])),
              "recall_ci": block_ci(fin, lambda x: _ratio(x[3], x[1])),
              "recall_by_type": recall_by_type(fin, True),
              "recall_by_type_greedy_v6_convention": recall_by_type(
                  {v: x["fin"] for v, x in tallies["greedy_v6_convention"].items()}, True)}
    pairs_all = [p for x in tallies["nearest"].values() for p in x["pairs"]]
    y = np.array([p[1] for p in pairs_all])
    pp, pc, pa = (np.array([p[i] for p in pairs_all]) for i in (2, 3, 4))

    def metric(m: int, f):
        return lambda x: f(cm_from(x, m))

    def gap(m1: int, m2: int, f):
        def g_(x):
            u, w_ = f(cm_from(x, m1)), f(cm_from(x, m2))
            return None if u is None or w_ is None else u - w_
        return g_

    major = int(np.bincount(y, minlength=N_CLASSES).argmax())
    all_labels = np.array([c.label for d_ in per.values() for c in d_["clips"]])
    typed = {
        **type_summary(typ),
        "matched_unclassified": int(sum(x["unclassified"] for x in tallies["nearest"].values())),
        "confusion_rows_true_cols_pred": {name: confusion(y, p).tolist()
                                          for name, p in (("pipeline", pp), ("Y_clip", pc), ("Y_all", pa))},
        "pipeline_block_ci": {k: block_ci(typ, metric(0, f)) for k, f in (
            ("accuracy", cm_accuracy), ("macro_f1", cm_macro_f1), ("hand_accuracy", cm_hand))},
        "commonest_type_floor": {"type": CLASSES[major], "accuracy_on_matched": float((y == major).mean()),
                                 "accuracy_on_all_labelled": float((all_labels == np.bincount(all_labels).argmax()).mean())},
        "reference_on_all_742": {"Y_all": ref["window"]["Y_all"]["accuracy"], "Y_clip": ref["window"]["Y_clip"]["accuracy"],
                                 "n": ref["window"]["Y_all"]["n"]},
        "greedy_v6_convention": type_summary({v: x["typ"] for v, x in tallies["greedy_v6_convention"].items()}),
        "gaps": {}}
    for name, m, refp in (("pipeline_minus_Y_clip", 1, pc), ("pipeline_minus_Y_all", 2, pa)):
        typed["gaps"][name] = {
            "paired_clip_bootstrap": {"accuracy": bootstrap_gap_ci(y, pp, refp, metric="accuracy"),
                                      "macro_f1": bootstrap_gap_ci(y, pp, refp, metric="macro_f1")},
            "paired_block_bootstrap": {k: block_ci(typ, gap(0, m, f)) for k, f in (
                ("accuracy", cm_accuracy), ("macro_f1", cm_macro_f1), ("hand_accuracy", cm_hand))}}
    named_right = int((y == pp).sum())
    pooled["labelled_punches_found_and_named_right"] = {"n": named_right, "of": int(allfc[1]),
                                                       "share": _ratio(named_right, allfc[1])}
    report["pooled"] = {"finder": pooled, "type": typed}

    # ---- audit
    items = audit_sample(audit_pools)
    audit_dir = a.out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    # The key says which strips are controls: it lives OUTSIDE the folder the judge works in.
    key_path = a.out_dir / "audit_key.json"
    key = {"digest": key_digest(items), "far_s": FAR_S, "items": items}
    old = json.loads(key_path.read_text(encoding="utf-8")) if key_path.is_file() else {}
    if not isinstance(old, dict):
        raise SystemExit(f"{key_path.name} is not an audit key")
    if a.judgments:
        # read_judgments binds every row to this run's sample digest; the key on disk must agree,
        # so the strips the judge looked at are this sample's.
        if old.get("digest") != key["digest"]:
            raise SystemExit("the audit sample of this run is not the one rendered on disk; "
                             "re-render (--videos-dir, no --judgments) and judge again")
        j = read_judgments(a.judgments, items)
        aud = {}
        for kind in ("far", "control"):
            for v in list(VIDEOS) + ["pooled"]:
                sel = [it for it in items if it["kind"] == kind and (v == "pooled" or it["video"] == v)]
                c = {x: sum(j[it["id"]] == x for it in sel) for x in JUDGMENTS}
                aud.setdefault(kind, {})[v] = {"n": len(sel), **c, "punch_share": _ratio(c["punch"], len(sel)),
                                               "punch_share_wilson": wilson(c["punch"], len(sel))}
        # Post hoc (seen only after unblinding): does completeness differ inside the stretch of
        # video that carries labels and outside it (intros, talk, conditioning work)?
        stretch = {v: (min(c.start for c in per[v]["clips"]), max(c.end for c in per[v]["clips"])) for v in VIDEOS}
        for where in ("inside", "outside"):
            sel = [it for it in items if it["kind"] == "far"
                   and (stretch[it["video"]][0] <= it["frame"] <= stretch[it["video"]][1]) == (where == "inside")]
            c = {x: sum(j[it["id"]] == x for it in sel) for x in JUDGMENTS}
            aud.setdefault("far_by_labelled_stretch_post_hoc", {})[where] = {
                "n": len(sel), **c, "punch_share_wilson": wilson(c["punch"], len(sel)),
                "by_video": {v: sum(it["video"] == v for it in sel) for v in VIDEOS}}
        report["label_completeness_audit"] = aud
        for v in VIDEOS:
            f = report["videos"][v]["finder"]
            a_ = aud["far"][v]
            if a_["n"]:
                lo, hi = a_["punch_share_wilson"]
                n_far = f["unmatched_far_from_every_label"]
                f["precision_if_far_detections_are_real_at_the_audited_share"] = {
                    "point": (f["tp"] + a_["punch_share"] * n_far) / f["n_detected"],
                    "range_from_wilson": [(f["tp"] + lo * n_far) / f["n_detected"], (f["tp"] + hi * n_far) / f["n_detected"]],
                    "note": "unclear judgments count as not punches; detections within 0.3 s of a label stay false"}
    else:
        if old.get("digest") not in (None, key["digest"]) and not a.videos_dir:
            raise SystemExit("this run's audit sample differs from the one rendered on disk; re-render it "
                             "(--videos-dir) rather than leave strips that belong to another sample")
        tmpl = audit_dir / "judgments_template.csv"
        if a.videos_dir:   # strips of an older sample must not survive next to this one's
            for f_ in list(audit_dir.glob("s[0-9][0-9][0-9].png")) + [tmpl]:
                f_.unlink(missing_ok=True)
        key_path.write_text(json.dumps(key, indent=1), encoding="utf-8")
        with tmpl.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(JUDGMENT_HEADER)
            for it in sorted(items, key=lambda x: x["id"]):
                w.writerow([it["id"], key["digest"], "", ""])
        if a.videos_dir:
            for v in VIDEOS:
                render_audit(a.videos_dir / f"{v}.mp4", per[v]["pose"], [it for it in items if it["video"] == v], audit_dir)
        print(f"audit: {len(items)} strips sampled ({'rendered to ' + str(audit_dir) if a.videos_dir else 'not rendered'}); "
              f"judge them blind into a copy of {tmpl.name} (do not open {key_path.name} first)")

    report["limits"] = [
        f"stance: run with --stance {a.stance}; BoxingVI's coaches' stance is unknown, so every lead/rear on these "
        "videos is unverified. Type accuracy does not depend on it",
        "the per-clip cut was chosen looking at V5's and V9's numbers (ADR-0014, 2026-10-13): these videos are not "
        "a clean test of it",
        "V2 is excluded: its skeletons are not from the video linked as V2 (vault ISSUES-LOG O61)",
        "coaches' landscape YouTube footage at 24-30 fps, not phone footage",
        "the finder's rule was tuned on V6 only; V5 and V9 never informed it",
        "precision counts every unmatched detection as false: a lower bound unless the labels are complete "
        "(see label_completeness_audit)",
        "the audit is one judge (a reviewer, from frame strips), blind to which strips were labelled; its controls "
        "measure that judge on known punches",
        "the video-disjoint split may not be athlete-disjoint (O55)"]
    out = a.out_dir / "pipeline_v5v9.json"
    out.write_text(json.dumps(report, indent=1, allow_nan=False, default=float), encoding="utf-8")

    # ---- print
    print("\nFINDER (V6 rule) on V5 + V9")
    for v in VIDEOS:
        f = report["videos"][v]["finder"]
        print(f"  {v}: {f['n_detected']} detections vs {f['n_labelled']} labels: precision {f['precision']:.3f} "
              f"recall {f['recall']:.3f} F1 {f['f1']:.3f} count ratio {f['count_ratio']:.2f}; shift floor F1 "
              f"{f['shift_floor']['f1_mean']:.3f} (97.5th {f['shift_floor']['f1_p97_5']:.3f}); "
              f"{f['unmatched_far_from_every_label']} unmatched > {FAR_S} s from any label")
    pr, rc = pooled["precision_ci"], pooled["recall_ci"]
    print(f"  pooled: precision {pooled['precision']:.3f} [{pr['ci_low']:.3f}, {pr['ci_high']:.3f}], recall "
          f"{pooled['recall']:.3f} [{rc['ci_low']:.3f}, {rc['ci_high']:.3f}], count ratio {pooled['count_ratio']:.2f}")
    print("  recall by type: " + ", ".join(f"{c} {x['recall']:.2f} ({x['found']}/{x['labelled']}) "
                                          f"[{x['ci']['ci_low']:.2f}, {x['ci']['ci_high']:.2f}]"
                                          for c, x in pooled["recall_by_type"].items()))
    print("  ... by punchfind.match's attribution (V6 convention): " + ", ".join(
        f"{c} {x['recall']:.2f}" for c, x in pooled["recall_by_type_greedy_v6_convention"].items()))
    print(f"TYPE on {typed['n']} matched detections ({typed['matched_unclassified']} unclassified)")
    for name in ("pipeline", "Y_clip", "Y_all"):
        print(f"  {name}: accuracy {typed[name]['accuracy']:.3f} macro F1 {typed[name]['macro_f1']:.3f} "
              f"hand {typed[name]['hand_accuracy']:.3f}"
              + (f"   (V6-convention attribution: {typed['greedy_v6_convention'][name]['accuracy']:.3f})"))
    for v in VIDEOS:
        t_ = report["videos"][v]["type"]
        print(f"  {v}: pipeline {t_['pipeline']['accuracy']:.3f}, Y_clip {t_['Y_clip']['accuracy']:.3f}, "
              f"Y_all {t_['Y_all']['accuracy']:.3f} on {t_['n']}")
    print(f"  commonest-type floor ({typed['commonest_type_floor']['type']}) "
          f"{typed['commonest_type_floor']['accuracy_on_matched']:.3f}")
    for name, g in typed["gaps"].items():
        c_, b_ = g["paired_clip_bootstrap"]["accuracy"], g["paired_block_bootstrap"]["accuracy"]
        print(f"  {name.replace('_minus_', ' - ')}: accuracy {c_['point']:+.3f} clip [{c_['ci_low']:+.3f}, "
              f"{c_['ci_high']:+.3f}] block [{b_['ci_low']:+.3f}, {b_['ci_high']:+.3f}]")
    nr = pooled["labelled_punches_found_and_named_right"]
    print(f"END TO END: {nr['n']} of {nr['of']} labelled punches found and named right ({nr['share']:.3f})")
    if "label_completeness_audit" in report:
        for kind in ("far", "control"):
            x = report["label_completeness_audit"][kind]["pooled"]
            print(f"  audit {kind}: {x['punch']} of {x['n']} judged real punches {x['punch_share_wilson']}")
        for where, x in report["label_completeness_audit"]["far_by_labelled_stretch_post_hoc"].items():
            print(f"  audit far, {where} the labelled stretch (post hoc): {x['punch']} of {x['n']} real punches")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
