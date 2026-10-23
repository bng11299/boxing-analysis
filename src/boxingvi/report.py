"""The output a boxer reads: a spreadsheet and a written summary, built from the
classified punches (classify_punches.py) and the measured reliability of each stage.

The summary is fixed sentence templates filled with computed numbers, so it
cannot invent one. Every number goes into the text through `_Writer`, which
records it; tests check that the text holds exactly those numbers and no other.
Rounding never flips a claim: a share that is not exactly 0 or 100 is never
printed as 0% or 100%, a rate under 10 keeps a decimal, and a change in punch
rate is described from the numbers as printed, and only when it is bigger than
chance would produce.

Reliability comes from measurements, linked by hashes rather than trusted:
  classified CSV --(sidecar: csv sha256, classifier sha256, pose frames and fps,
  finder rule)--> classifier manifest --(the held-out confusion matrix, cell for
  cell, and per-class F1 recomputed from it)--> the reported run's held-out
  confusion matrix (results/stgcnpp.json).
* punch type: that confusion matrix.
* the hand: the share of held-out punches whose predicted TYPE is thrown by the
  same hand as the true one. That measures the classifier's hand. The finder's
  hand (which wrist moved) has never been measured on its own; the report shows
  how often the two disagree and says which number belongs to which reading.
* punch finding: the tuned rule's held-out score on V6, recall AND count ratio,
  only if the sidecar records that rule (variant, threshold, minimum gap).
  V6 labels only hooks and uppercuts, so its precision understates the truth and
  its count ratio overstates the over-count; how far counts are off on your
  footage is unknown, and the summary says so.
* your own footage: the clip's classify_score.json, when you labelled it, and
  only if it scored the same classifier and the same classified CSV.

Fixed rules, stated in the report: a punch type the classifier names correctly
under LOW_RECALL of the time is "named least reliably"; a count more than
COUNT_BIAS off on the benchmark is reported as running high or low; a change in
punch rate counts only beyond two standard errors of a Poisson count.

The one label-free check on THIS clip's punch types: how often the hand a punch
type implies is the wrist that moved, against the agreement two independent
readings with the same proportions would reach by chance (classify.hand_agreement).
The summary leads with a warning when that agreement is under HAND_FLOOR, or is not
above chance by either of two one-sided exact tests at HAND_ALPHA (binomial against
the chance agreement; Fisher's, which holds both proportions fixed). When the types
fall so heavily on one hand, or are so few, that even full agreement could not pass
both tests, it says the check cannot confirm them instead (unless agreement is already
under the floor, which is a warning). Without this, a broken
classify step (2026-10-12: 63% and 41% lead uppercuts) produced reports that were
faithful to the broken input and said nothing about it. The check sees the hand
only: a jab read as a lead hook passes it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from .classify import class_side, hand_agreement, read_manifest
from .data import CLASSES, HELD_OUT_VIDEOS
from .stats import binomial_greater_p, fisher_greater_p

CLASSIFIED_HEADER = ["frame", "time_s", "hand", "peak_speed_bl_per_s", "peak_on_interpolated_frame",
                     "lead_rear", "punch_type", "confidence", "type_hand", "hands_agree",
                     "punch_start_frame", "punch_end_frame", "window_frames_observed", "note"]
UNCLASSIFIED = "unclassified"
LOW_RECALL = 0.70
COUNT_BIAS = 0.20
CHANCE_SE = 2.0
HAND_ALPHA = 0.05
# The agreement the report requires, whatever the tests say. Set by Brendan 2026-10-20, to rise to 0.80;
# chosen, not fitted to any clip.
HAND_FLOOR = 0.70



def _under_alpha(p: float) -> bool:
    """p < HAND_ALPHA, ignoring float noise at the boundary: scipy returns Fisher's exact 1/20
    as 0.04999999999999999. A p within a billionth of alpha counts as not under it."""
    return p < HAND_ALPHA * (1 - 1e-9)
MAX_ROWS = 100_000
MAX_ROUNDS = 100
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_COUNT = 10 ** 9   # a punch count or confusion cell; far above any real one, far below int64
TIME_TOL_S = 2e-3   # find_punches.py writes times to the millisecond; classify_punches.py checks the same bound
PLURAL = {"jab": "jabs", "cross": "crosses", "lead_hook": "lead hooks", "lead_uppercut": "lead uppercuts",
          "rear_hook": "rear hooks", "rear_uppercut": "rear uppercuts"}
# The only notes classify_punches.py writes: an unnamed punch carries one REASON, a named
# punch carries none. (Per-punch caveats about a punch's start and end went away with the
# per-clip cut, 2026-10-13: a punch's window no longer depends on its own peak width.)
_REASONS = (re.compile(r"pose in only [0-9]+ of the window's [0-9]+ frames"),
            re.compile(r"no frame in the window shows both hips"),
            re.compile(r"joints fall far outside the frame after scaling"),
            re.compile(r"the punch is shorter than two frames at the training frame rate"))
_INT = re.compile(r"-?[0-9]+")   # a punch's estimated start can fall before the clip's first frame
_DIGITS = re.compile(r"[0-9]+")
_DECIMAL = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_NUMBER = re.compile(r"[0-9]+(?:[.:][0-9]+)?")
_FORMULA_LEAD = ("=", "+", "-", "@", "\t", "\r")

assert set(PLURAL) == set(CLASSES)


# ------------------------------------------------------------------ inputs ---

def _read_json(path: Path) -> dict:
    path = Path(path)
    with path.open("rb") as f:
        raw = f.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"{path.name}: over {MAX_JSON_BYTES} bytes")

    def no_constant(name):
        raise ValueError(f"{path.name} contains {name}, which is not valid JSON")
    d = json.loads(raw.decode("utf-8"), parse_constant=no_constant)
    if not isinstance(d, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return d


def _prob(x, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not 0 <= x <= 1:
        raise ValueError(f"{what} must be a number in [0, 1], got {x!r}")
    return float(x)


def _count(x, what: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= MAX_COUNT:   # bounded: no int64 overflow
        raise ValueError(f"{what} must be a whole number in 0..{MAX_COUNT}, got {str(x)[:20]!r}")
    return x


def read_classified(path: Path, duration_s: float, *, fps: float) -> "list[dict]":
    """classify_punches.py's CSV, checked field by field: it sits on disk between
    the two tools, and everything in it ends up in front of the user. Every frame
    must lie in this clip and every time must equal frame / this clip's fps, as
    find_punches.py wrote it: a CSV made at another frame rate is refused even when
    its sidecar has been made to agree."""
    path = Path(path)
    if (isinstance(fps, bool) or not isinstance(fps, (int, float)) or (isinstance(fps, int) and not 0 < fps <= 10_000)
            or not (math.isfinite(fps) and fps > 0)):
        raise ValueError("fps must be a positive number")
    n_frames = round(duration_s * fps)
    rows, leads = [], set()
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        if next(r, None) != CLASSIFIED_HEADER:
            raise ValueError(f"{path.name}: not a classify_punches.py CSV")
        for i, row in enumerate(r, start=2):
            if len(rows) >= MAX_ROWS:
                raise ValueError(f"{path.name}: more than {MAX_ROWS} rows")
            if len(row) != len(CLASSIFIED_HEADER):
                raise ValueError(f"{path.name} line {i}: {len(row)} fields")
            d = dict(zip(CLASSIFIED_HEADER, row))
            where = f"{path.name} line {i}"
            if not _DIGITS.fullmatch(d["frame"]) or not _DIGITS.fullmatch(d["window_frames_observed"]):
                raise ValueError(f"{where}: frame numbers must be whole numbers")
            if not _DECIMAL.fullmatch(d["time_s"]) or not float(d["time_s"]) <= duration_s:
                raise ValueError(f"{where}: time {d['time_s']!r} is not inside this {duration_s:.1f} s clip")
            if not int(d["frame"]) < n_frames:
                raise ValueError(f"{where}: frame {d['frame']} is past this {n_frames}-frame clip")
            if abs(float(d["time_s"]) - int(d["frame"]) / fps) > TIME_TOL_S:
                raise ValueError(f"{where}: time {d['time_s']} is not frame {d['frame']} at {fps:.3f} fps: "
                                 "this CSV was not made from this clip")
            if d["hand"] not in ("left", "right") or d["lead_rear"] not in ("lead", "rear"):
                raise ValueError(f"{where}: hand {d['hand']!r} / {d['lead_rear']!r}")
            leads.add(d["hand"] if d["lead_rear"] == "lead" else ("right" if d["hand"] == "left" else "left"))
            if d["peak_on_interpolated_frame"] not in ("yes", "no"):
                raise ValueError(f"{where}: interpolated flag {d['peak_on_interpolated_frame']!r}")
            a_, b_ = d["punch_start_frame"], d["punch_end_frame"]
            if a_ or b_:
                if not (_INT.fullmatch(a_) and _INT.fullmatch(b_)) or not int(a_) <= int(d["frame"]) <= int(b_):
                    raise ValueError(f"{where}: punch frames {a_!r}-{b_!r} do not contain its peak")
            elif d["punch_type"] != UNCLASSIFIED:
                raise ValueError(f"{where}: a classified punch without the frames it was read from")
            if d["punch_type"] == UNCLASSIFIED:
                if (d["confidence"] or d["type_hand"] or d["hands_agree"]
                        or not any(p.fullmatch(d["note"]) for p in _REASONS)):
                    raise ValueError(f"{where}: an unclassified punch carries a reading or lacks a known reason")
                conf = None
            elif d["punch_type"] in CLASSES:
                if d["note"]:
                    raise ValueError(f"{where}: a named punch carries a note; classify_punches.py writes none")
                if not _DECIMAL.fullmatch(d["confidence"]):
                    raise ValueError(f"{where}: confidence {d['confidence']!r} is not a number")
                conf = _prob(float(d["confidence"]), f"{where}: confidence")
                side = class_side(d["punch_type"])
                if d["type_hand"] != side or d["hands_agree"] != ("yes" if side == d["lead_rear"] else "no"):
                    raise ValueError(f"{where}: type hand and agreement do not follow from the type")
            else:
                raise ValueError(f"{where}: unknown punch type {d['punch_type']!r}")
            rows.append({"frame": int(d["frame"]), "time_s": float(d["time_s"]), "hand": d["hand"],
                         "lead_rear": d["lead_rear"], "punch_type": d["punch_type"], "confidence": conf,
                         "type_hand": d["type_hand"] or None, "hands_agree": d["hands_agree"] or None,
                         "note": d["note"], "interpolated": d["peak_on_interpolated_frame"] == "yes",
                         "span": (int(a_), int(b_)) if a_ else None})
    if len(leads) > 1:
        raise ValueError(f"{path.name}: rows disagree about which hand leads (two stances in one file)")
    return sorted(rows, key=lambda x: x["frame"])


def verify_provenance(csv_path: Path, sidecar: Path, manifest: Path, clip: str) -> dict:
    """The CSV is the one classify_punches.py wrote, by the classifier this manifest
    describes. Returns the validated manifest."""
    side = _read_json(sidecar)
    if side.get("clip") != clip:
        raise ValueError(f"{Path(sidecar).name} describes clip {side.get('clip')!r}, not {clip}")
    if not isinstance(side.get("csv_sha256"), str) or not _HEX64.fullmatch(side["csv_sha256"]):
        raise ValueError(f"{Path(sidecar).name}: no csv sha256")
    if hashlib.sha256(Path(csv_path).read_bytes()).hexdigest() != side["csv_sha256"]:
        raise ValueError(f"{Path(csv_path).name} has changed since classify_punches.py wrote it")
    man = read_manifest(manifest)   # classify's own schema checks
    if side.get("classifier_sha256") != man["sha256"]:
        raise ValueError("these punches were named by another classifier than this manifest describes; "
                         "its reliability does not apply")
    man["_sidecar"] = side
    return man


def check_pose(sidecar: dict, n_frames: int, fps: float) -> None:
    """The pose the report reads is the one the CSV was classified on: the same number
    of frames AND the same frame rate. A clip classified at 60 fps is refused against
    a same-named, same-length pose at 30 fps; its times would all be wrong."""
    frames, rate = sidecar.get("pose_frames"), sidecar.get("fps")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames != n_frames:
        raise ValueError(f"these punches were classified on a {frames!r}-frame pose; this one has {n_frames}")
    # Python compares int with float exactly, so a huge JSON integer is refused here instead of
    # overflowing math.isfinite; NaN and inf are never equal to a real pose's fps.
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate != fps:
        raise ValueError(f"these punches were classified on a pose at {rate!r} fps; this one is at {fps!r}")


def parse_rounds(text: "str | None", duration_s: float) -> "list[tuple[float, float]]":
    """'0-180,240-420' in seconds; the whole clip when not given. Rounds must be
    inside the clip, in order and not overlapping."""
    if not (math.isfinite(duration_s) and duration_s > 0):
        raise ValueError("clip duration must be positive")
    if text is None:
        return [(0.0, duration_s)]
    parts = text.split(",")
    if len(parts) > MAX_ROUNDS:
        raise ValueError(f"at most {MAX_ROUNDS} rounds")
    out = []
    for part in parts:
        m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*", part)
        if not m:
            raise ValueError(f"round {part!r} is not start-end in seconds")
        a, b = float(m.group(1)), float(m.group(2))
        if not a < b <= duration_s + 1e-6:
            raise ValueError(f"round {part.strip()} is empty or runs past the {duration_s:.1f} s clip")
        if out and a < out[-1][1]:
            raise ValueError(f"round {part.strip()} starts before the previous one ends")
        out.append((a, min(b, duration_s)))
    return out


@dataclass(frozen=True)
class TypeReliability:
    cls: str
    true: int
    predicted: int
    recall: float
    precision: "float | None"
    f1: float

    @property
    def count_ratio(self) -> float:
        return self.predicted / self.true


def benchmark_reliability(reference: Path, manifest: dict) -> dict:
    """Per-type reliability from the reported run's held-out confusion matrix. Used
    only if the matrix is, cell for cell, the one the classifier's manifest records
    (train_classifier.py writes the matrix of the very model it saves), and it
    reproduces the per-class F1 both files record. Per-class F1 alone is not enough:
    it is unchanged when the matrix is transposed, while recall, precision and every
    over- and under-count warning reverse."""
    ref = _read_json(reference)
    runs = [r for r in ref.get("results", []) if isinstance(r, dict) and r.get("model") == "st-gcn++ pretrained"]
    if len(runs) != 1:
        raise ValueError(f"{Path(reference).name}: no single 'st-gcn++ pretrained' result")
    r = runs[0]
    raw = r.get("confusion")
    if not (isinstance(raw, list) and len(raw) == len(CLASSES)
            and all(isinstance(row, list) and len(row) == len(CLASSES) for row in raw)):
        raise ValueError(f"{Path(reference).name}: confusion is not {len(CLASSES)}x{len(CLASSES)}")
    c = np.array([[_count(v, "confusion cell") for v in row] for row in raw], dtype=np.int64)
    if (c.sum(axis=1) == 0).any():
        raise ValueError(f"{Path(reference).name}: a punch type has no held-out examples")
    held = manifest.get("held_out") if isinstance(manifest.get("held_out"), dict) else {}
    pinned = held.get("confusion")
    if not (isinstance(pinned, list) and len(pinned) == len(CLASSES)
            and all(isinstance(row, list) and len(row) == len(CLASSES) for row in pinned)):
        raise ValueError("the classifier's manifest records no held-out confusion matrix: re-run "
                         "train_classifier.py, which records it, so the reliability can be tied to this model")
    mine = np.array([[_count(v, "manifest confusion cell") for v in row] for row in pinned], dtype=np.int64)
    if not np.array_equal(mine, c):
        how = ("is the manifest's with rows and columns swapped" if np.array_equal(mine, c.T)
               else "is not the one the classifier's manifest records")
        raise ValueError(f"{Path(reference).name}'s confusion matrix {how}: it does not describe this classifier")
    want, got = r.get("per_class_f1"), held.get("per_class_f1")
    table = []
    for i, cls in enumerate(CLASSES):
        true, pred, tp = int(c[i].sum()), int(c[:, i].sum()), int(c[i, i])
        f1 = 2 * tp / (true + pred)
        for name, d in (("reference", want), ("manifest", got)):
            v = d.get(cls) if isinstance(d, dict) else None
            if isinstance(v, bool) or not isinstance(v, (int, float)) or abs(v - f1) > 1e-9:
                raise ValueError(f"the {name}'s {cls} F1 ({v!r}) is not the confusion matrix's ({f1:.4f}): "
                                 "this matrix does not describe this classifier")
        table.append(TypeReliability(cls, true, pred, tp / true, tp / pred if pred else None, f1))
    side = np.array([class_side(k) == "lead" for k in CLASSES])
    same = side[:, None] == side[None, :]
    videos = ref.get("val_videos")
    if not (isinstance(videos, list) and videos and all(isinstance(v, str) and _DIGITS.sub("", v) == "V"
                                                         for v in videos)):
        videos = list(HELD_OUT_VIDEOS)
    return {"types": table, "side_accuracy": float(c[same].sum() / c.sum()),
            "accuracy": float(np.trace(c) / c.sum()), "n": int(c.sum()), "videos": videos}


RULE_KEYS = ("variant", "threshold", "min_gap_s")


def _rule_of(d, what: str) -> dict:
    """The finder rule's three settings, as find_punches.load_rule returns them and
    classify_punches.py records them in the sidecar."""
    if not isinstance(d, dict) or not isinstance(d.get("variant"), str):
        raise ValueError(f"{what}: no punch-finder rule (variant, threshold, min_gap_s)")
    out = {"variant": d["variant"]}
    for k in RULE_KEYS[1:]:
        v = d.get(k)
        if (isinstance(v, bool) or not isinstance(v, (int, float))
                or (isinstance(v, int) and abs(v) > 1e6) or not math.isfinite(v)):   # int bound: no float overflow
            raise ValueError(f"{what}: rule {k} {str(v)[:20]!r} is not a number")
        out[k] = float(v)
    return out


def finder_reliability(rule: Path, used: dict) -> dict:
    """The tuned rule's held-out score, only if `used` -- the rule the sidecar says
    found these punches -- is this rule: the same variant, threshold and minimum gap.
    Another rule's recall says nothing about the punches in front of the user."""
    d = _read_json(rule)
    if _rule_of(d.get("chosen"), Path(rule).name) != _rule_of(used, "the sidecar"):
        raise ValueError(f"these punches were found with another rule than {Path(rule).name} "
                         f"({_rule_of(used, 'the sidecar')}); its held-out score does not apply")
    t = (d.get("chosen") or {}).get("test") if isinstance(d.get("chosen"), dict) else None
    if not isinstance(t, dict):
        raise ValueError(f"{Path(rule).name}: no held-out test score for the punch finder")
    ratio = t.get("count_ratio")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 <= ratio < 100:
        raise ValueError(f"{Path(rule).name}: count ratio {ratio!r}")
    video = d.get("video") if isinstance(d.get("video"), str) and _DIGITS.sub("", d["video"]) == "V" else None
    return {"recall": _prob(t.get("recall"), "finder recall"), "count_ratio": float(ratio), "video": video}


def own_reliability(score: Path, clip: str, classifier_sha256: str, csv_sha256: str) -> dict:
    """The clip's score against your labels, only if it scored THESE punches: the same
    clip, the same classifier, and the classified CSV whose sha256 the sidecar records.
    Re-classifying a clip (another stance, rule or cut) leaves the old score file on
    disk; without the CSV hash its accuracy would be reported for punches it never saw."""
    d = _read_json(score)
    if d.get("clip") != clip:
        raise ValueError(f"{Path(score).name} scores clip {d.get('clip')!r}, not {clip}")
    if d.get("classifier_sha256") != classifier_sha256:
        raise ValueError(f"{Path(score).name} scored another classifier; re-run classify_punches.py --labels")
    if not (isinstance(csv_sha256, str) and _HEX64.fullmatch(csv_sha256)):
        raise ValueError("no classified-CSV sha256 to check the score against")
    if "csv_sha256" not in d:
        raise ValueError(f"{Path(score).name} does not say which classified CSV it scored (written before scores "
                         "recorded it); re-run classify_punches.py --labels")
    if d["csv_sha256"] != csv_sha256:
        raise ValueError(f"{Path(score).name} scored another classified CSV than this one (the clip was classified "
                         "again since); re-run classify_punches.py --labels")
    m = d.get("matched_detections")
    if not isinstance(m, dict):
        raise ValueError(f"{Path(score).name}: no matched-detection score")
    n = _count(m.get("n"), "matched punch count")
    out = {"n": n, "unclassifiable": _count(d.get("matched_unclassifiable", 0), "unclassifiable count")}
    if n:
        out["accuracy"] = _prob(m.get("accuracy"), "own accuracy")
        out["floor"] = _prob(m.get("majority_floor_accuracy"), "own majority floor")
    return out


# ------------------------------------------------------------------- facts ---

def round_of(t: float, rounds) -> "int | None":
    for k, (a, b) in enumerate(rounds):
        if a <= t < b or (k == len(rounds) - 1 and t == b):
            return k
    return None


def _could_pass(n: int, a: int) -> bool:
    """Whether the check could pass at all on n named punches whose types put `a` on the
    lead hand: both tests, if the wrist agreed with every one."""
    return (_under_alpha(binomial_greater_p(n, n, (a * a + (n - a) ** 2) / n ** 2))
            and _under_alpha(fisher_greater_p(a, 0, 0, n - a)))


def hand_check(named) -> dict:
    """Type-implied hand against the wrist that moved, over the named punches.

    verdict, in this order: "warn" if agreement is under HAND_FLOOR; "pass" if both
    one-sided exact tests (binomial, Fisher) are under HAND_ALPHA; "warn" if they are
    not but full agreement could have passed them; otherwise the check cannot confirm
    the types: "concentrated" (full agreement on a 50/50 split of this many punches
    would have passed; at alpha 0.05 that needs 7 or more) or "too_few". "none" with no
    named punch: no type to warn about."""
    ag = hand_agreement([r["type_hand"] for r in named], [r["lead_rear"] for r in named])
    n = ag["n"]
    out = {**ag, "k": 0, "table": None, "p_binomial": None, "p_fisher": None, "below_floor": False,
           "dominant": None, "dominant_share": None, "most_often": [], "most_often_n": 0}
    if not n:
        return {**out, "verdict": "none", "warn": False}
    t = Counter((r["type_hand"], r["lead_rear"]) for r in named)
    table = (t[("lead", "lead")], t[("lead", "rear")], t[("rear", "lead")], t[("rear", "rear")])
    k, a = table[0] + table[3], table[0] + table[1]
    pb, pf = binomial_greater_p(k, n, ag["chance"]), fisher_greater_p(*table)
    below = Fraction(k, n) < Fraction(str(HAND_FLOOR))
    if below:
        verdict = "warn"
    elif _under_alpha(pb) and _under_alpha(pf):
        verdict = "pass"
    elif _could_pass(n, a):
        verdict = "warn"
    else:
        verdict = "concentrated" if _could_pass(n, n // 2) else "too_few"
    hand = "lead" if 2 * a >= n else "rear"
    types = Counter(r["punch_type"] for r in named if r["type_hand"] == hand)
    top = max(types.values())
    return {**out, "k": k, "table": table, "p_binomial": pb, "p_fisher": pf, "below_floor": below,
            "dominant": hand, "dominant_share": max(a, n - a) / n,
            "most_often": [c for c in CLASSES if types[c] == top], "most_often_n": top,   # every tied type
            "verdict": verdict, "warn": verdict == "warn"}


def compute_facts(rows, rounds, duration_s: float) -> dict:
    """Every number the report states, computed once."""
    which = [round_of(r["time_s"], rounds) for r in rows]
    per = []
    for k, (a, b) in enumerate(rounds):
        rr = [r for r, w in zip(rows, which) if w == k]
        per.append({"round": k + 1, "start_s": a, "end_s": b, "n": len(rr), "minutes": (b - a) / 60,
                    "per_min": len(rr) / ((b - a) / 60),
                    "types": {c: sum(r["punch_type"] == c for r in rr) for c in CLASSES},
                    "unnamed": sum(r["punch_type"] == UNCLASSIFIED for r in rr)})
    inside = [r for r, w in zip(rows, which) if w is not None]
    named = [r for r in inside if r["punch_type"] != UNCLASSIFIED]
    minutes = sum(b - a for a, b in rounds) / 60
    return {"duration_s": duration_s, "rounds": per, "n_found": len(inside),
            "n_outside_rounds": len(rows) - len(inside), "n_named": len(named),
            "n_unnamed": len(inside) - len(named), "per_min": len(inside) / minutes,
            "round_minutes": minutes,
            "types": {c: sum(r["punch_type"] == c for r in named) for c in CLASSES},
            "n_type_lead": sum(r["type_hand"] == "lead" for r in named),
            "n_hands_disagree": sum(r["hands_agree"] == "no" for r in named),
            "hand_check": hand_check(named)}


def shares(counts: "dict[str, int]") -> "dict[str, int]":
    """Whole-percent shares that add up to 100 (largest remainder)."""
    total = sum(counts.values())
    if not total:
        return {k: 0 for k in counts}
    exact = {k: 100 * v / total for k, v in counts.items()}
    out = {k: int(math.floor(x)) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - out[k]), k))[:100 - sum(out.values())]:
        out[k] += 1
    return out


# ----------------------------------------------------------------- summary ---

class _Writer:
    """Collects sentences; every number enters through a method here, which records it."""

    def __init__(self):
        self.parts: "list[str]" = []
        self.numbers: "list[str]" = []

    def num(self, s: str) -> str:
        self.numbers.append(s)
        return s

    def pct(self, x: float) -> str:
        """Never 0% or 100% unless exact."""
        if 0 < x < 0.005:
            return "under " + self.num("1") + "%"
        if 0.995 <= x < 1:
            return "over " + self.num("99") + "%"
        return self.num(f"{100 * x:.0f}") + "%"

    def share(self, p: int, count: int, total: int) -> str:
        if p == 0 and count:
            return "under " + self.num("1") + "%"
        if p == 100 and count < total:
            return "over " + self.num("99") + "%"
        return self.num(str(p)) + "%"

    def count(self, n: int) -> str:
        return self.num(str(int(n)))

    def rate(self, x: float) -> str:
        return self.num(f"{x:.1f}" if x < 10 else f"{x:.0f}")

    def pval(self, p: float) -> str:
        """'p = 0.29'; three decimals under 0.1, so a p just above 0.05 never prints as 0.05;
        a p under alpha that would print as alpha says 'just under'; never 'p = 0.000' or
        'p = 1.00' unless exact."""
        if _under_alpha(p) and f"{p:.3f}" == f"{HAND_ALPHA:.3f}":
            return "p just under " + self.num(f"{HAND_ALPHA:.2f}")
        if p == 1:
            return "p = " + self.num("1")
        if p >= 0.995:
            return "p over " + self.num("0.99")
        if 0 < p < 0.0005:
            return "p under " + self.num("0.001")
        return "p = " + self.num(f"{p:.3f}" if p < 0.1 else f"{p:.2f}")

    def mmss(self, s: float) -> str:
        s = int(round(s))
        return self.num(f"{s // 60}:{s % 60:02d}")

    def say(self, text: str) -> None:
        self.parts.append(text)


def _punches(n: int) -> str:
    return "punch" if n == 1 else "punches"


def _join(items: "list[str]") -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def summary(facts: dict, bench: dict, finder: dict, own: "dict | None") -> "tuple[str, list[str]]":
    """(text, the numbers it states). Fixed templates only."""
    w = _Writer()
    hc = facts["hand_check"]
    if hc["verdict"] == "warn":
        w.say("Warning: the punch types in this summary may be wrong for this clip.")
        w.say(f"For {w.count(hc['k'])} of {w.count(hc['n'])} named {_punches(hc['n'])} the hand the punch type "
              f"implies is the wrist that moved ({w.pct(hc['agree'])}), against {w.pct(hc['chance'])} expected by "
              f"chance from how often each reading says lead hand.")
        ok_b, ok_f = _under_alpha(hc["p_binomial"]), _under_alpha(hc["p_fisher"])
        tests = (f"One-sided exact tests of whether that is above chance give {w.pval(hc['p_binomial'])} "
                 f"(binomial) and {w.pval(hc['p_fisher'])} (Fisher)")
        if ok_b and ok_f:
            w.say(tests + ".")
        elif ok_b or ok_f:
            w.say(tests + f"; only the {'binomial' if ok_b else 'Fisher'} test is under "
                  f"{w.num(f'{HAND_ALPHA:.2f}')}, so that agreement is not shown to be better than chance.")
        else:
            w.say(tests + f", neither under {w.num(f'{HAND_ALPHA:.2f}')}, so that agreement is not shown to be "
                  f"better than chance.")
        if hc["below_floor"]:
            w.say(f"The {w.pct(hc['agree'])} agreement is under the {w.pct(HAND_FLOOR)} this report requires.")
        if hc["agree"] <= hc["chance"]:
            w.say("Agreement no higher than chance can also mean the clip was read with the wrong stance "
                  "(orthodox or southpaw), which swaps lead and rear.")
    elif hc["verdict"] == "concentrated":
        w.say(f"Caution: there appears to be an extreme concentration of {hc['dominant']}-hand punch types: "
              f"{w.pct(hc['dominant_share'])} of the {w.count(hc['n'])} named punches, most often "
              f"{_join([PLURAL[c] for c in hc['most_often']])} ({w.count(hc['most_often_n'])}"
              f"{' each' if len(hc['most_often']) > 1 else ''}). On a split like this the hand check "
              f"cannot confirm the types even if every punch agreed with the wrist that moved. If this is not "
              f"expected, checking the footage is recommended.")
    elif hc["verdict"] == "too_few":
        w.say(f"Caution: only {w.count(hc['n'])} named {_punches(hc['n'])}, too few to check the punch types "
              f"against the wrist that moved.")
    rounds = facts["rounds"]
    nr = len(rounds)
    where = "one round" if nr == 1 else f"{w.count(nr)} rounds"
    footage = w.mmss(facts["round_minutes"] * 60)
    if facts["n_found"] == 0:
        w.say(f"In {where} ({footage} of footage) the tool found no punches.")
    else:
        w.say(f"In {where} ({footage} of footage) the tool found {w.count(facts['n_found'])} "
              f"{_punches(facts['n_found'])}, {w.rate(facts['per_min'])} a minute.")
        if facts["n_named"]:
            sh = shares(facts["types"])
            order = sorted((c for c in CLASSES if facts["types"][c]), key=lambda c: (-facts["types"][c], c))
            parts = [f"{w.share(sh[c], facts['types'][c], facts['n_named'])} {PLURAL[c]} "
                     f"({w.count(facts['types'][c])})" for c in order]
            w.say(f"Of the {w.count(facts['n_named'])} it could name, {_join(parts)}.")
            lead = facts["n_type_lead"] / facts["n_named"]
            w.say(f"By punch type, {w.pct(lead)} were thrown with your lead hand.")
            if facts["n_hands_disagree"]:
                k = facts["n_hands_disagree"]
                w.say(f"For {w.count(k)} {_punches(k)} the punch type and the wrist that moved point to "
                      f"different hands; one of the two readings is wrong there.")
            if hc["verdict"] == "pass":
                w.say(f"Across the clip the hand each punch type implies is the wrist that moved for "
                      f"{w.count(hc['k'])} of {w.count(hc['n'])} ({w.pct(hc['agree'])}) against "
                      f"{w.pct(hc['chance'])} by chance ({w.pval(hc['p_binomial'])} binomial, "
                      f"{w.pval(hc['p_fisher'])} Fisher), and at least the {w.pct(HAND_FLOOR)} this report "
                      f"requires. That checks the hand only: it cannot tell a jab from a lead hook or uppercut, or "
                      f"a cross from a rear hook or uppercut.")
        if facts["n_unnamed"]:
            w.say(f"{w.count(facts['n_unnamed'])} it could not name: the pose was too incomplete to read.")
        first, last = rounds[0], rounds[-1]
        if nr >= 2 and first["n"] and last["n"]:
            a_s, b_s = w.rate(first["per_min"]), w.rate(last["per_min"])
            a, b = float(a_s), float(b_s)
            se = math.sqrt(first["n"] / first["minutes"] ** 2 + last["n"] / last["minutes"] ** 2)
            span = f"from {a_s} a minute in round {w.count(1)} to {b_s} in round {w.count(last['round'])}"
            if abs(last["per_min"] - first["per_min"]) <= CHANCE_SE * se or a == b:
                w.say(f"Your punch rate went {span}: a difference chance alone could produce.")
            else:
                trend = "dropped" if b < a else "rose"
                w.say(f"Your punch rate {trend} by {w.pct(abs(b / a - 1))}, {span}.")
        if facts["n_outside_rounds"]:
            k = facts["n_outside_rounds"]
            w.say(f"{w.count(k)} {_punches(k)} fell outside the rounds and {'is' if k == 1 else 'are'} not counted.")

    low = [t for t in bench["types"] if t.recall < LOW_RECALL]
    if low:
        w.say(f"The tool names these least reliably: {_join([PLURAL[t.cls] for t in low])}. On held-out benchmark "
              f"footage it named only {_join([f'{w.pct(t.recall)} of {PLURAL[t.cls]}' for t in low])} correctly.")
    high = [t for t in bench["types"] if t.count_ratio - 1 > COUNT_BIAS]
    under = [t for t in bench["types"] if 1 - t.count_ratio > COUNT_BIAS]
    bias = []
    if high:
        bias.append("high for " + _join([f"{PLURAL[t.cls]} (by {w.pct(t.count_ratio - 1)})" for t in high]))
    if under:
        bias.append("low for " + _join([f"{PLURAL[t.cls]} (by {w.pct(1 - t.count_ratio)})" for t in under]))
    if bias:
        w.say(f"On the benchmark, type counts ran {', and '.join(bias)}.")
    w.say(f"The hand a punch type implies was right {w.pct(bench['side_accuracy'])} of the time on the benchmark; "
          f"which wrist moved has not been checked against labels.")
    src = f"benchmark video {w.num(finder['video'])}" if finder["video"] else "the benchmark video"
    ratio = w.num(f"{finder['count_ratio']:.1f}")
    w.say(f"Finding punches is the least tested stage. On {src}, the only one with continuous labels, it found "
          f"{w.pct(finder['recall'])} of the labelled punches but flagged {ratio} "
          f"times as many punches as were labelled; part of that is real punches that were never labelled there, "
          f"so how far the totals above are off is not known.")
    if own is None:
        w.say("None of this has been measured on your own footage yet: label a clip to measure it.")
    elif not own["n"]:
        w.say("You labelled this clip, but no labelled punch with a nameable type matched a punch the tool found.")
    else:
        w.say(f"Against your own labels for this clip it named {w.pct(own['accuracy'])} of {w.count(own['n'])} "
              f"matched {_punches(own['n'])} correctly; always guessing your commonest punch would score "
              f"{w.pct(own['floor'])}.")
        if own["unclassifiable"]:
            w.say(f"{w.count(own['unclassifiable'])} more matched {_punches(own['unclassifiable'])} could not be named.")
    return " ".join(w.parts), w.numbers


def numbers_in(text: str) -> "list[str]":
    return _NUMBER.findall(text)


# -------------------------------------------------------------- spreadsheet ---

def _cell(v):
    """Values only, never formulas: a string that a spreadsheet would evaluate is refused."""
    if isinstance(v, str) and v.startswith(_FORMULA_LEAD):
        raise ValueError(f"refusing to write a cell that starts like a formula: {v[:20]!r}")
    return v


def cut_line(sidecar: dict) -> str:
    """One sentence for the About sheet on how this clip's punches were cut, from the sidecar
    classify_punches.py wrote. Checked field by field: the sidecar is a file on disk."""
    c = sidecar.get("cut")
    if c is None:
        return "No punch was found in this clip, so nothing was cut."
    if not isinstance(c, dict) or c.get("per") != "clip":
        raise ValueError("the sidecar does not describe a per-clip cut: re-run classify_punches.py")
    length = c.get("length_s")
    if isinstance(length, bool) or not isinstance(length, (int, float)) or not 0 < length <= 5:
        raise ValueError(f"sidecar cut length {length!r} is not a punch length in seconds")
    used, left = _count(c.get("n_used"), "cut n_used"), _count(c.get("n_left_out"), "cut n_left_out")
    if not isinstance(c.get("capped"), bool):
        raise ValueError("sidecar cut 'capped' must be true or false")
    if c.get("source") == "training":
        return (f"Too few punches in this clip to read its tempo: every punch was read over {length:.2f} s around "
                f"its wrist-speed peak, the median punch length in the benchmark's training clips.")
    if c.get("source") != "clip":
        raise ValueError(f"sidecar cut source {c.get('source')!r} is not one classify_punches.py writes")
    return (f"Every punch was read over {length:.2f} s around its wrist-speed peak: the median punch length in "
            f"this clip, from {used} {_punches(used)}"
            + (f" ({left} left out: the wrist left view beside the peak)" if left else "")
            + ("; capped at the longest the tool can read." if c["capped"] else "."))


def _check_result(hc: dict) -> str:
    """The Reliability sheet's one-cell verdict of the hand check."""
    if hc["verdict"] == "warn":
        why = ([f"under {100 * HAND_FLOOR:.0f}%"] if hc["below_floor"] else []) + [
            f"{name} not under {HAND_ALPHA:.2f}" for name, q in (("binomial", "p_binomial"), ("Fisher", "p_fisher"))
            if not _under_alpha(hc[q])]
        return "warning: " + "; ".join(why)
    return {"pass": "passed", "concentrated": "cannot check: types concentrated on one hand",
            "too_few": "cannot check: too few named punches", "none": "no named punch"}[hc["verdict"]]


def write_workbook(path: Path, clip: str, rows, rounds, facts: dict, bench: dict, text: str,
                   fps: float, how_cut: str = "") -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Punches"
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError("fps must be positive")
    ws.append(["round", "time (s)", "time (m:ss)", "hand (wrist that moved)", "lead/rear", "punch type",
               "confidence", "type agrees with wrist", "punch length (s)", "note"])
    for r in rows:
        k = round_of(r["time_s"], rounds)
        s = int(r["time_s"])
        ws.append([_cell(v) for v in (
            k + 1 if k is not None else "outside rounds", round(r["time_s"], 3), f"{s // 60}:{s % 60:02d}",
            r["hand"], r["lead_rear"], r["punch_type"].replace("_", " "),
            None if r["confidence"] is None else round(r["confidence"], 3),
            r["hands_agree"] or "",
            None if r["span"] is None else round((r["span"][1] - r["span"][0] + 1) / fps, 2), r["note"])])
    ws = wb.create_sheet("Rounds")
    ws.append(["round", "start (s)", "end (s)", "punches", "per minute"]
              + [PLURAL[c] for c in CLASSES] + ["not named"])
    for p in facts["rounds"]:
        ws.append([p["round"], p["start_s"], round(p["end_s"], 3), p["n"], round(p["per_min"], 2)]
                  + [p["types"][c] for c in CLASSES] + [p["unnamed"]])
    ws.append(["all", rounds[0][0], round(rounds[-1][1], 3), facts["n_found"], round(facts["per_min"], 2)]
              + [facts["types"][c] for c in CLASSES] + [facts["n_unnamed"]])
    ws = wb.create_sheet("Reliability")
    ws.append(["punch type", "benchmark punches", "tool counted", "count off by", "named correctly (recall)",
               "precision", "F1", "named least reliably"])
    for t in bench["types"]:
        ws.append([PLURAL[t.cls], t.true, t.predicted, f"{100 * (t.count_ratio - 1):+.0f}%", round(t.recall, 3),
                   None if t.precision is None else round(t.precision, 3), round(t.f1, 3),
                   "yes" if t.recall < LOW_RECALL else ""])
    hc = facts["hand_check"]
    ws.append([])
    ws.append(["this clip's check", "named punches", "type hand = wrist", "agreement", "chance agreement",
               "p, binomial (one-sided exact)", "p, Fisher (one-sided exact)", "agreement required", "result"])
    ws.append(["punch type's hand vs the wrist that moved", hc["n"], hc["k"],
               None if hc["agree"] is None else round(hc["agree"], 3),
               None if hc["chance"] is None else round(hc["chance"], 3),
               *(None if hc[q] is None else float(f"{hc[q]:.3g}") for q in ("p_binomial", "p_fisher")),  # never 0
               HAND_FLOOR, _check_result(hc)])
    ws = wb.create_sheet("About")
    for line in (f"clip: {clip}", "", "summary:", text, "", how_cut,
                 f"'named least reliably' = named correctly under {100 * LOW_RECALL:.0f}% of the time on held-out "
                 f"benchmark footage; type counts reported when more than {100 * COUNT_BIAS:.0f}% off; a change in "
                 f"punch rate reported only beyond {CHANCE_SE:.0f} standard errors of a Poisson count.",
                 f"A warning leads the summary when the hand each punch type implies matches the wrist that moved "
                 f"under {100 * HAND_FLOOR:.0f}% of the time (a fixed requirement), or not significantly more often "
                 f"than chance by either of two one-sided exact tests at {HAND_ALPHA:.2f}: binomial, against the "
                 f"chance agreement for the two readings' proportions, and Fisher's, which holds both proportions "
                 f"fixed. When the punch types fall so heavily on one hand, or are so few, that even full agreement "
                 f"could not pass both tests, the summary says the check cannot confirm them instead, unless "
                 f"agreement is already under {100 * HAND_FLOOR:.0f}%.",
                 "That check sees only which hand threw: it cannot tell a jab from a lead hook or uppercut, or a "
                 "cross from a rear hook or uppercut. Lead and rear follow the stance given to classify_punches.py; "
                 "a wrong stance swaps them and shows up as low agreement.",
                 f"Benchmark = BoxingVI, held-out videos {'/'.join(bench['videos'])}; skeletons from AlphaPose, "
                 "not the YOLO11-pose used on your footage. Counts are estimates."):
        ws.append([_cell(line)])
    wb.save(path)
