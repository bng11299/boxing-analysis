"""The output a boxer reads: a spreadsheet and a written summary, built from the
classified punches (classify_punches.py) and the measured reliability of each stage.

The summary is fixed sentence templates filled with computed numbers, so it
cannot invent one. Every number goes into the text through `_Writer.num`, which
records it; tests check that the text holds exactly those numbers and no other.

Reliability comes from measurements, never from prose:
* punch type: the held-out confusion matrix of the reported classifier run
  (results/stgcnpp.json). The report refuses to use it unless the classifier's
  manifest records the same per-class F1, i.e. it is the model that classified
  these punches.
* punch finding: the tuned rule's held-out score on V6 (results/punch_finder_v6.json).
* your own footage: the clip's classify_score.json, when you have labelled it.
Until the last exists, the summary says plainly that nothing was measured on
your footage.

Which punch types to "trust least" is a fixed rule, not a judgement per report:
a type the classifier named correctly under LOW_RECALL of the time on held-out
benchmark footage. Count bias is reported when the predicted count was more than
COUNT_BIAS off the true count. Both thresholds are stated in the report.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import CLASSES

CLASSIFIED_HEADER = ["frame", "time_s", "hand", "peak_speed_bl_per_s", "peak_on_interpolated_frame",
                     "lead_rear", "punch_type", "confidence", "type_hand", "hands_agree",
                     "window_frames_observed", "note"]
UNCLASSIFIED = "unclassified"
LOW_RECALL = 0.70
COUNT_BIAS = 0.20
STEADY = 0.05          # a change in punch rate within this is "held steady"
MAX_ROWS = 100_000
MAX_JSON_BYTES = 4 * 1024 * 1024
PLURAL = {"jab": "jabs", "cross": "crosses", "lead_hook": "lead hooks", "lead_uppercut": "lead uppercuts",
          "rear_hook": "rear hooks", "rear_uppercut": "rear uppercuts"}
_DIGITS = re.compile(r"[0-9]+")
_NUMBER = re.compile(r"[0-9]+(?:[.:][0-9]+)?")
# A cell starting with one of these is a formula to a spreadsheet program.
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


def read_classified(path: Path, duration_s: float) -> "list[dict]":
    """classify_punches.py's CSV, checked field by field: it sits on disk between
    the two tools, and everything in it ends up in front of the user."""
    path = Path(path)
    rows = []
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
            try:
                t = float(d["time_s"])
            except ValueError:
                raise ValueError(f"{where}: time {d['time_s']!r} is not a number") from None
            if not (math.isfinite(t) and 0 <= t <= duration_s):
                raise ValueError(f"{where}: time {d['time_s']} is not inside this {duration_s:.1f} s clip")
            if d["hand"] not in ("left", "right") or d["lead_rear"] not in ("lead", "rear"):
                raise ValueError(f"{where}: hand {d['hand']!r} / {d['lead_rear']!r}")
            if d["peak_on_interpolated_frame"] not in ("yes", "no"):
                raise ValueError(f"{where}: interpolated flag {d['peak_on_interpolated_frame']!r}")
            if d["punch_type"] == UNCLASSIFIED:
                if d["confidence"] or d["type_hand"] or d["hands_agree"]:
                    raise ValueError(f"{where}: an unclassified punch carries a type reading")
                conf = None
            elif d["punch_type"] in CLASSES:
                try:
                    conf = _prob(float(d["confidence"]), f"{where}: confidence")
                except ValueError as e:
                    raise ValueError(f"{where}: confidence {d['confidence']!r}: {e}") from None
                side = "lead" if d["punch_type"] in ("jab", "lead_hook", "lead_uppercut") else "rear"
                if d["type_hand"] != side or d["hands_agree"] != ("yes" if side == d["lead_rear"] else "no"):
                    raise ValueError(f"{where}: type hand and agreement do not follow from the type")
            else:
                raise ValueError(f"{where}: unknown punch type {d['punch_type']!r}")
            if d["note"] and (d["note"].startswith(_FORMULA_LEAD) or len(d["note"]) > 200):
                raise ValueError(f"{where}: note is not one this pipeline writes")
            rows.append({"frame": int(d["frame"]), "time_s": t, "hand": d["hand"], "lead_rear": d["lead_rear"],
                         "punch_type": d["punch_type"], "confidence": conf,
                         "hands_agree": d["hands_agree"] or None, "note": d["note"],
                         "interpolated": d["peak_on_interpolated_frame"] == "yes"})
    return sorted(rows, key=lambda x: x["frame"])


def parse_rounds(text: "str | None", duration_s: float) -> "list[tuple[float, float]]":
    """'0-180,240-420' in seconds; the whole clip when not given. Rounds must be
    inside the clip, in order and not overlapping."""
    if not (math.isfinite(duration_s) and duration_s > 0):
        raise ValueError("clip duration must be positive")
    if text is None:
        return [(0.0, duration_s)]
    out = []
    for part in text.split(","):
        m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*", part)
        if not m:
            raise ValueError(f"round {part!r} is not start-end in seconds")
        a, b = float(m.group(1)), float(m.group(2))
        if not a < b <= duration_s + 1e-6:
            raise ValueError(f"round {part.strip()} is empty or runs past the {duration_s:.1f} s clip")
        if out and a < out[-1][1]:
            raise ValueError(f"round {part.strip()} starts before the previous one ends")
        out.append((a, min(b, duration_s)))
    if not out or len(out) > 100:
        raise ValueError("between 1 and 100 rounds")
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


def benchmark_reliability(reference: Path, manifest: Path) -> dict:
    """Per-type reliability from the reported run's held-out confusion matrix, but
    only if the classifier's manifest says it IS that run (same per-class F1)."""
    ref, man = _read_json(reference), _read_json(manifest)
    runs = [r for r in ref.get("results", []) if isinstance(r, dict) and r.get("model") == "st-gcn++ pretrained"]
    if len(runs) != 1:
        raise ValueError(f"{Path(reference).name}: no single 'st-gcn++ pretrained' result")
    r = runs[0]
    held = man.get("held_out") or {}
    want, got = r.get("per_class_f1"), held.get("per_class_f1")
    if not (isinstance(want, dict) and isinstance(got, dict) and set(want) == set(CLASSES) == set(got)
            and all(abs(want[c] - got[c]) < 1e-9 for c in CLASSES)):
        raise ValueError("the classifier's manifest does not match the reported run: its per-type "
                         "reliability is unmeasured, so the report will not borrow another model's")
    c = np.asarray(r.get("confusion"), dtype=np.int64)
    if c.shape != (len(CLASSES), len(CLASSES)) or (c < 0).any() or (c.sum(axis=1) == 0).any():
        raise ValueError(f"{Path(reference).name}: confusion matrix is not {len(CLASSES)}x{len(CLASSES)} counts")
    table = []
    for i, cls in enumerate(CLASSES):
        true, pred = int(c[i].sum()), int(c[:, i].sum())
        table.append(TypeReliability(cls, true, pred, float(c[i, i] / true),
                                     float(c[i, i] / pred) if pred else None, float(want[cls])))
    side = np.array([0 if k in ("jab", "lead_hook", "lead_uppercut") else 1 for k in CLASSES])
    same = side[:, None] == side[None, :]
    return {"types": table, "side_accuracy": float(c[same].sum() / c.sum()),
            "accuracy": float(np.trace(c) / c.sum()), "n": int(c.sum())}


def finder_reliability(rule: Path) -> dict:
    d = _read_json(rule)
    t = d.get("test") or (d.get("chosen") or {}).get("test")
    if not isinstance(t, dict):
        raise ValueError(f"{Path(rule).name}: no held-out test score for the punch finder")
    return {"recall": _prob(t.get("recall"), "finder recall"),
            "precision": _prob(t.get("precision"), "finder precision"),
            "n_labelled": int(t.get("n_labelled", 0))}


def own_reliability(score: Path, clip: str) -> "dict | None":
    d = _read_json(score)
    if d.get("clip") != clip:
        raise ValueError(f"{Path(score).name} scores clip {d.get('clip')!r}, not {clip}")
    m = d.get("matched_detections") or {}
    if not m.get("n"):
        return None
    n = m["n"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("matched punch count must be a positive integer")
    return {"n": n, "accuracy": _prob(m.get("accuracy"), "own accuracy"),
            "side_accuracy": _prob(m.get("side_accuracy"), "own hand accuracy")}


# ------------------------------------------------------------------- facts ---

def round_of(t: float, rounds) -> "int | None":
    for k, (a, b) in enumerate(rounds):
        if a <= t < b or (k == len(rounds) - 1 and t == b):
            return k
    return None


def compute_facts(rows, rounds, duration_s: float) -> dict:
    """Every number the report states, computed once."""
    per = []
    for k, (a, b) in enumerate(rounds):
        rr = [r for r in rows if round_of(r["time_s"], rounds) == k]
        per.append({"round": k + 1, "start_s": a, "end_s": b, "n": len(rr),
                    "per_min": len(rr) / ((b - a) / 60),
                    "types": {c: sum(r["punch_type"] == c for r in rr) for c in CLASSES},
                    "unnamed": sum(r["punch_type"] == UNCLASSIFIED for r in rr)})
    inside = [r for r in rows if round_of(r["time_s"], rounds) is not None]
    named = [r for r in inside if r["punch_type"] != UNCLASSIFIED]
    minutes = sum(b - a for a, b in rounds) / 60
    return {"duration_s": duration_s, "rounds": per, "n_found": len(inside),
            "n_outside_rounds": len(rows) - len(inside), "n_named": len(named),
            "n_unnamed": len(inside) - len(named), "per_min": len(inside) / minutes,
            "round_minutes": minutes,
            "types": {c: sum(r["punch_type"] == c for r in named) for c in CLASSES},
            "lead_share": (sum(r["lead_rear"] == "lead" for r in inside) / len(inside)) if inside else None}


# ----------------------------------------------------------------- summary ---

class _Writer:
    """Collects sentences; every number enters through num(), which records it."""

    def __init__(self):
        self.parts: "list[str]" = []
        self.numbers: "list[str]" = []

    def num(self, s: str) -> str:
        self.numbers.append(s)
        return s

    def pct(self, x: float) -> str:
        return self.num(f"{100 * x:.0f}") + "%"

    def count(self, n: int) -> str:
        return self.num(str(int(n)))

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
    """(text, the numbers it states in order). Fixed templates only."""
    w = _Writer()
    rounds = facts["rounds"]
    nr = len(rounds)
    where = "one round" if nr == 1 else f"{w.count(nr)} rounds"
    if facts["n_found"] == 0:
        w.say(f"In {where} ({w.mmss(facts['round_minutes'] * 60)} of footage) the tool found no punches.")
    else:
        rate = w.num(f"{facts['per_min']:.0f}")
        w.say(f"In {where} ({w.mmss(facts['round_minutes'] * 60)} of footage) the tool found "
              f"{w.count(facts['n_found'])} {_punches(facts['n_found'])}, {rate} a minute.")
        if facts["n_named"]:
            order = sorted((c for c in CLASSES if facts["types"][c]), key=lambda c: (-facts["types"][c], c))
            shares = [f"{w.pct(facts['types'][c] / facts['n_named'])} {PLURAL[c]} ({w.count(facts['types'][c])})"
                      for c in order]
            w.say(f"Of the {w.count(facts['n_named'])} it could name, {_join(shares)}.")
        if facts["n_unnamed"]:
            w.say(f"{w.count(facts['n_unnamed'])} it could not name: too little of you was in view.")
        w.say(f"{w.pct(facts['lead_share'])} came from your lead hand.")
        if nr >= 2 and rounds[0]["per_min"] > 0:
            first, last = rounds[0], rounds[-1]
            change = last["per_min"] / first["per_min"] - 1
            a, b = w.num(f"{first['per_min']:.0f}"), w.num(f"{last['per_min']:.0f}")
            if abs(change) <= STEADY:
                trend = "held steady"
            else:
                trend = ("dropped by " if change < 0 else "rose by ") + w.pct(abs(change))
            w.say(f"Your punch rate {trend}, from {a} a minute in round {w.count(1)} "
                  f"to {b} in round {w.count(last['round'])}.")
        if facts["n_outside_rounds"]:
            k = facts["n_outside_rounds"]
            w.say(f"{w.count(k)} {_punches(k)} fell between rounds and {'is' if k == 1 else 'are'} not counted.")

    low = [t for t in bench["types"] if t.recall < LOW_RECALL]
    if low:
        w.say(f"Trust these counts least: {_join([PLURAL[t.cls] for t in low])}. On benchmark footage the tool "
              f"named only {_join([f'{w.pct(t.recall)} of {PLURAL[t.cls]}' for t in low])} correctly.")
    high = [t for t in bench["types"] if t.count_ratio - 1 > COUNT_BIAS]
    under = [t for t in bench["types"] if 1 - t.count_ratio > COUNT_BIAS]
    bias = []
    if high:
        bias.append("high for " + _join([f"{PLURAL[t.cls]} (by {w.pct(t.count_ratio - 1)})" for t in high]))
    if under:
        bias.append("low for " + _join([f"{PLURAL[t.cls]} (by {w.pct(1 - t.count_ratio)})" for t in under]))
    if bias:
        w.say(f"On the benchmark, counts ran {', and '.join(bias)}.")
    w.say(f"Which hand threw a punch is the most reliable reading: right {w.pct(bench['side_accuracy'])} "
          f"of the time on the benchmark.")
    w.say(f"Finding punches is the least tested stage: on the one benchmark video with continuous labels it "
          f"found {w.pct(finder['recall'])} of the labelled punches, and only hooks and uppercuts were labelled there.")
    if own:
        w.say(f"Against your own labels for this clip it named {w.pct(own['accuracy'])} of "
              f"{w.count(own['n'])} matched punches correctly.")
    else:
        w.say("None of this has been measured on your own footage yet: label a clip to measure it.")
    return " ".join(w.parts), w.numbers


def numbers_in(text: str) -> "list[str]":
    return _NUMBER.findall(text)


# -------------------------------------------------------------- spreadsheet ---

def _cell(v):
    """Values only, never formulas: a string that a spreadsheet would evaluate is refused."""
    if isinstance(v, str) and v.startswith(_FORMULA_LEAD):
        raise ValueError(f"refusing to write a cell that starts like a formula: {v[:20]!r}")
    return v


def write_workbook(path: Path, clip: str, rows, rounds, facts: dict, bench: dict, text: str) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Punches"
    ws.append(["round", "time (s)", "time (m:ss)", "hand", "lead/rear", "punch type", "confidence",
               "type agrees with hand", "note"])
    for r in rows:
        k = round_of(r["time_s"], rounds)
        s = int(r["time_s"])
        ws.append([_cell(v) for v in (
            k + 1 if k is not None else "between rounds", round(r["time_s"], 3), f"{s // 60}:{s % 60:02d}",
            r["hand"], r["lead_rear"], r["punch_type"].replace("_", " "),
            None if r["confidence"] is None else round(r["confidence"], 3),
            r["hands_agree"] or "", r["note"])])
    ws = wb.create_sheet("Rounds")
    ws.append(["round", "start (s)", "end (s)", "punches", "per minute"]
              + [PLURAL[c] for c in CLASSES] + ["not named"])
    for p in facts["rounds"]:
        ws.append([p["round"], p["start_s"], round(p["end_s"], 3), p["n"], round(p["per_min"], 1)]
                  + [p["types"][c] for c in CLASSES] + [p["unnamed"]])
    ws.append(["all", rounds[0][0], round(rounds[-1][1], 3), facts["n_found"], round(facts["per_min"], 1)]
              + [facts["types"][c] for c in CLASSES] + [facts["n_unnamed"]])
    ws = wb.create_sheet("Reliability")
    ws.append(["punch type", "benchmark punches", "tool counted", "count off by", "named correctly (recall)",
               "precision", "F1", "trust least"])
    for t in bench["types"]:
        ws.append([PLURAL[t.cls], t.true, t.predicted, f"{100 * (t.count_ratio - 1):+.0f}%", round(t.recall, 3),
                   None if t.precision is None else round(t.precision, 3), round(t.f1, 3),
                   "yes" if t.recall < LOW_RECALL else ""])
    ws = wb.create_sheet("About")
    for line in (f"clip: {clip}", "", "summary:", text, "",
                 f"'trust least' = named correctly under {100 * LOW_RECALL:.0f}% of the time on held-out benchmark "
                 f"footage; counts flagged when more than {100 * COUNT_BIAS:.0f}% off.",
                 "Benchmark = BoxingVI, videos V2/V5/V9 held out; skeletons from AlphaPose, not the YOLO11-pose "
                 "used on your footage. Counts are estimates."):
        ws.append([_cell(line)])
    wb.save(path)
