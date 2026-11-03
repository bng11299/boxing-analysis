"""Punch combinations: runs of one person's punches, each within COMBO_GAP_S of the last (vault ADR-0019).

A COMBINATION is a maximal run of consecutive punches of ONE person in which each punch's peak is at most
COMBO_GAP_S after the previous one's, with two or more punches; a run of one is a single. A run never crosses a
round boundary, and in the per-boxer report never crosses an identity-segment edge, so a combination is always
one tracked person: a wrong join by the identity method cannot merge two boxers' punches into one combination
(it can still credit the whole combination to the wrong boxer). Callers say which "unit" (round, segment) each
punch belongs to; a punch with unit None (outside every round) is in no run.

Names are the six BoxingVI types; a punch the classifier could not name is "?", and a combination with a "?" is
counted but never named in a tally. Classifier confidence is not used (never calibrated).

The finder reports at most one punch per its minimum gap (0.4 s, whichever hand), so faster combinations are
counted short or as single punches: every count here is a lower bound, and the report says so.

The gate (docs/gates/combinations.md, committed before this code) scores the statements on labelled own footage;
`score_clip` / `gate_result` compute it, `read_gate` reads its result back for the report, and uses it only for
the classifier, finder rule and gap it was measured with.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from . import report
from .data import CLASSES

# Chosen by Brendan 2026-10-28 over 0.75 s (2 x the training punch length): a round number, NOT fitted to any
# clip. Changing it voids every gate result (each records the gap it was measured with).
COMBO_GAP_S = 1.0
UNKNOWN = "?"
GATE_FORMAT = "boxingvi-combo-gate-1"
GATE_DOC = "docs/gates/combinations.md"
GATE_SEED = 20261028
GATE_PERMUTATIONS = 2000
GATE_PERCENTILE = 97.5
GATE_CLIPS = ("IMG_4395", "IMG_4406")   # the registered clips; any other set is an exploratory run, never the gate
MAX_GATE_BYTES = 256 * 1024
SHOWN = 3            # at most this many names in the commonest list; a tie is never cut
NAME = {"jab": "jab", "cross": "cross", "lead_hook": "lead hook", "rear_hook": "rear hook",
        "lead_uppercut": "lead uppercut", "rear_uppercut": "rear uppercut", UNKNOWN: UNKNOWN}
NUMBER = {"jab": "1", "cross": "2", "lead_hook": "3", "rear_hook": "4", "lead_uppercut": "5",
          "rear_uppercut": "6", UNKNOWN: UNKNOWN}   # the boxing numbers, for the sheet only (never the summary)
_EPS = 1e-9
_GATE_KEYS = {"format", "document", "gap_s", "rule", "finder_per_hand", "classifier_sha256", "clips", "bars", "PASS",
              "pooled"}
_POOLED_KEYS = {"recovered_exactly", "named_label_combinations", "exact_recall", "floor_mean", "floor_p975",
                "permutations", "seed"}
_CLIP_COUNTS = ("label_combinations", "named_label_combinations", "recovered_exactly", "found_named_combinations",
                "found_named_that_recover", "structure_recovered", "label_combinations_with_a_gap_under_min_gap",
                "found_combinations", "found_punches_in_combinations", "found_punches",
                "label_punches_in_combinations", "label_punches", "tolerance_frames", "point_tolerance_frames")
_CLIP_LISTS = ("favourite_found", "favourite_labels", "starter_found", "starter_labels")
_CLIP_KEYS = set(_CLIP_COUNTS) | set(_CLIP_LISTS) | {"favourite_matches", "starter_matches", "fps",
                                                      "classified_csv_sha256", "labels_sha256"}
_BARS = ("1_better_than_chance", "2_favourite_matches", "3_starter_matches")
_HEX64 = re.compile(r"[0-9a-f]{64}")

assert set(NAME) - {UNKNOWN} == set(CLASSES) == set(NUMBER) - {UNKNOWN}


# --------------------------------------------------------------------------------------- grouping ---

def type_of(punch: dict) -> str:
    t = punch["punch_type"]
    if t == report.UNCLASSIFIED:
        return UNKNOWN
    if t not in CLASSES:
        raise ValueError(f"unknown punch type {str(t)[:20]!r}")
    return t


def group(punches, fps: float, gap_s: float = COMBO_GAP_S) -> "list[list[dict]]":
    """Every run (singles included), in time order. `punches`: dicts with "frame", "punch_type" and "unit"
    (hashable; None = in no unit, left out). Two punches are in one run iff they are consecutive among the
    punches of their unit, in the same unit, and the later peak is at most `gap_s` after the earlier."""
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not (math.isfinite(fps) and fps > 0):
        raise ValueError("fps must be a positive number")
    if isinstance(gap_s, bool) or not isinstance(gap_s, (int, float)) or not (math.isfinite(gap_s) and gap_s > 0):
        raise ValueError("the combination gap must be a positive number of seconds")
    for p in punches:
        f = p.get("frame")   # a detection's peak frame; a label's centre may be a half frame
        if (isinstance(f, bool) or not isinstance(f, (int, float)) or (isinstance(f, int) and f > 10 ** 12)
                or not (math.isfinite(f) and f >= 0)):
            raise ValueError("a punch's frame must be a number >= 0")
        type_of(p)
    by_unit: "dict[object, list]" = {}
    for i, p in enumerate(punches):
        if p.get("unit") is not None:
            by_unit.setdefault(p["unit"], []).append((p["frame"], i, p))
    runs = []
    for items in by_unit.values():
        items.sort(key=lambda x: (x[0], x[1]))
        run = [items[0][2]]
        for (f0, _i0, _p0), (f1, _i1, p1) in zip(items, items[1:]):
            if (f1 - f0) / fps <= gap_s + _EPS:
                run.append(p1)
            else:
                runs.append(run)
                run = [p1]
        runs.append(run)
    return sorted(runs, key=lambda r: (r[0]["frame"], str(r[0]["unit"])))


def solo_punches(rows, rounds) -> "list[dict]":
    """report.read_classified's rows with their unit: the round (None outside every round)."""
    return [{**r, "unit": report.round_of(r["time_s"], rounds)} for r in rows]


def boxer_punches(rows, rounds, segments, boxer: str, finder_rows) -> "list[dict]":
    """One boxer's rows with their unit: (round, identity segment). `segments`: the identity's segments (objects
    with track_id, first, last, boxer); `finder_rows`: boxers.read_boxers' (track id, finder columns) for this
    boxer, which say the track each punch was found on. A punch no segment of this boxer covers is refused:
    read_boxers has already checked that every punch lies in one."""
    tid_of = {}
    for tid, cols in finder_rows:
        f = int(cols[0])
        if f in tid_of and tid_of[f] != tid:
            raise ValueError(f"boxer {boxer} has punches on two tracks at frame {f}")
        tid_of[f] = int(tid)
    mine = [(k, s) for k, s in enumerate(segments) if s.boxer == boxer]
    out = []
    for r in rows:
        tid = tid_of.get(r["frame"])
        seg = [k for k, s in mine if s.track_id == tid and s.first <= r["frame"] <= s.last]
        if len(seg) != 1:
            raise ValueError(f"a punch of boxer {boxer} at frame {r['frame']} is not in exactly one of its segments")
        k = report.round_of(r["time_s"], rounds)
        out.append({**r, "unit": None if k is None else (k, seg[0])})
    return out


def name_of(run) -> "tuple[str, ...]":
    return tuple(type_of(p) for p in run)


def spoken(name: "tuple[str, ...]") -> str:
    """'jab-cross-lead hook'. No digits, so a name adds no number to a summary."""
    return "-".join(NAME[t] for t in name)


def numbered(name: "tuple[str, ...]") -> str:
    return "-".join(NUMBER[t] for t in name)


def _top(counter: Counter, at_least: int = 1) -> "tuple[list, int]":
    """(every key with the highest count, sorted; that count); ([], 0) when no count reaches `at_least`."""
    if not counter:
        return [], 0
    top = max(counter.values())
    if top < at_least:
        return [], 0
    return sorted(k for k, v in counter.items() if v == top), top


def favourite(runs) -> "list[tuple]":
    """The gate's favourite: among NAMED combinations seen at least twice, every name at the highest count."""
    return _top(Counter(name_of(r) for r in runs if len(r) >= 2 and UNKNOWN not in name_of(r)), 2)[0]


def starter(runs) -> "list[str]":
    """The gate's starter: among combinations whose first punch is named, every first type at the highest count."""
    return _top(Counter(type_of(r[0]) for r in runs if len(r) >= 2 and type_of(r[0]) != UNKNOWN))[0]


def compute(runs, min_gap_s: float, per_hand: bool = False) -> dict:
    """Every number the combination sentences state, from `group`'s runs."""
    if isinstance(min_gap_s, bool) or not isinstance(min_gap_s, (int, float)) or not (
            math.isfinite(min_gap_s) and min_gap_s >= 0) or not isinstance(per_hand, bool):
        raise ValueError("the finder's minimum gap must be a number of seconds >= 0")
    combos = [r for r in runs if len(r) >= 2]
    named = Counter(name_of(r) for r in combos if UNKNOWN not in name_of(r))
    # the commonest list: whole count groups, highest first, while they fit in SHOWN names (a tie is never cut)
    shown, groups = [], sorted({v for v in named.values() if v >= 2}, reverse=True)
    for v in groups:
        names = sorted(k for k, c in named.items() if c == v)
        if len(shown) + len(names) > SHOWN:
            break
        shown += [(k, v) for k in names]
    tie_n = 0 if shown or not groups else sum(1 for c in named.values() if c == groups[0])
    firsts = Counter(type_of(r[0]) for r in combos if type_of(r[0]) != UNKNOWN)
    st, st_n = _top(firsts)
    doubles = Counter(a for r in combos for a, b in zip(name_of(r), name_of(r)[1:]) if a == b and a != UNKNOWN)
    dbl, dbl_n = _top(doubles)
    return {"n_punches": sum(len(r) for r in runs), "n_combos": len(combos),
            "n_in_combos": sum(len(r) for r in combos), "longest": max((len(r) for r in combos), default=0),
            "shown": shown, "tie_n": tie_n, "tie_count": groups[0] if tie_n else 0,
            "n_named": sum(named.values()), "n_unknown": sum(UNKNOWN in name_of(r) for r in combos),
            "starter": st, "starter_n": st_n, "n_first_named": sum(firsts.values()),
            "n_doubles": sum(doubles.values()), "doubled": dbl, "doubled_n": dbl_n,
            "lengths": Counter(len(r) for r in combos), "min_gap_s": float(min_gap_s), "per_hand": per_hand,
            "gap_s": COMBO_GAP_S}


# ---------------------------------------------------------------------------------- the summary ---

def _each(k: int) -> str:
    return " each" if k > 1 else ""


def _or(items: "list[str]") -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def say_facts(w, cf: dict, per_boxer: bool = False) -> None:
    """The combination sentences, through report._Writer `w`. Says nothing when there was no punch."""
    if not cf["n_punches"]:
        return
    gap = w.num(f"{cf['gap_s']:.1f}")
    if not cf["n_combos"]:
        w.say(f"No two punches in a row came within {gap} s of each other, so the tool saw no combinations "
              f"(two or more punches, each within {w.num(f'{cf['gap_s']:.1f}')} s of the last).")
        return
    n, k = cf["n_combos"], cf["n_in_combos"]
    w.say(f"It saw {w.count(n)} {'combination' if n == 1 else 'combinations'} (two or more punches, each within "
          f"{gap} s of the last), taking {w.count(k)} of the {w.count(cf['n_punches'])} punches "
          f"({w.pct(k / cf['n_punches'])}); the longest was {w.count(cf['longest'])} punches.")
    if cf["shown"]:
        # "commonest" is exactly the gate's favourite (the highest count, ties kept); lower counts follow it
        top = cf["shown"][0][1]
        first = [f"{spoken(nm)} ({w.count(c)})" for nm, c in cf["shown"] if c == top]
        rest = [f"{spoken(nm)} ({w.count(c)})" for nm, c in cf["shown"] if c != top]
        w.say(f"{'The commonest combination was' if len(first) == 1 else 'The commonest combinations were'} "
              f"{report._join(first)}" + (f", followed by {report._join(rest)}" if rest else "") + ".")
    elif cf["tie_n"]:
        w.say(f"{w.count(cf['tie_n'])} different combinations tie as the commonest, seen {w.count(cf['tie_count'])} "
              f"times each.")
    elif cf["n_named"]:
        w.say("No named combination came up more than once.")
    if cf["starter"]:
        st = _or(["the " + NAME[t] for t in cf["starter"]])
        w.say(f"Of the {w.count(cf['n_first_named'])} {'combination' if cf['n_first_named'] == 1 else 'combinations'}"
              f" whose first punch it could name, {w.count(cf['starter_n'])}{_each(len(cf['starter']))} "
              f"({w.pct(cf['starter_n'] / cf['n_first_named'])}{_each(len(cf['starter']))}) started with {st}.")
    if cf["n_doubles"]:
        d = report._join(["the " + NAME[t] for t in cf["doubled"]])
        w.say(f"The same punch came twice in a row {w.count(cf['n_doubles'])} "
              f"{'time' if cf['n_doubles'] == 1 else 'times'} inside combinations, most often {d} "
              f"({w.count(cf['doubled_n'])}{_each(len(cf['doubled']))}).")
    if cf["n_unknown"]:
        u = cf["n_unknown"]
        w.say(f"{w.count(u)} {'combination includes' if u == 1 else 'combinations include'} a punch it could not "
              f"name, shown as {UNKNOWN}, and {'is' if u == 1 else 'are'} left out of the commonest.")
    if per_boxer:
        w.say("A combination never spans a change of tracked person or a round, so a wrong join between tracks "
              "cannot merge two people's punches into one; which boxer threw it is only as certain as the split "
              "between the boxers.")


_BAR_WORDS = {"1_better_than_chance": "exact recovery was not above chance",
              "2_favourite_matches": "the commonest combination did not match the labels in both clips",
              "3_starter_matches": "the punch combinations start with did not match the labels in both clips"}


def say_reliability(w, cf: dict, gate: "dict | None", per_boxer: bool = False) -> None:
    """The combination caveats, through report._Writer `w`: the finder's blind spot, the all-or-nothing nature of
    a sequence, and the gate's verdict when one applies. Says nothing when there was no combination."""
    if not cf["n_combos"]:
        return
    w.say(f"Combinations are the least certain part of this report: the tool sees at most one punch per "
          f"{w.num(f'{cf['min_gap_s']:.1f}')} s{' from each hand' if cf['per_hand'] else ''}, so faster "
          f"combinations are counted short or as single punches, "
          f"and a combination is right only if every punch in it was found and named right.")
    where = "two labelled clips of solo own footage (not two boxers)" if per_boxer else "two labelled clips of own footage"
    if gate is None:
        w.say("The combinations have not been checked against labelled footage.")
    elif gate["status"] == "not_applicable":
        w.say("The combination check on labelled footage was run with another classifier, punch finder or gap, so "
              "it does not apply here.")
    else:
        nums = (f"it recovered {w.count(gate['recovered'])} of {w.count(gate['n_label'])} labelled combinations "
                f"exactly ({w.pct(gate['recall'])}), against {w.pct(gate['floor'])} by chance with the punch types "
                f"shuffled "
                f"({w.num(f'{GATE_PERCENTILE:.1f}')}th percentile)")
        if gate["status"] == "pass":
            w.say(f"On {where} the combinations passed their pre-registered check: {nums}, and the commonest "
                  f"combination and the punch combinations start with matched the labels in both clips.")
        else:
            w.say(f"On {where} the combinations failed their pre-registered check ({_join_failed(gate)}): {nums}. "
                  f"Treat them as rough.")


def _join_failed(gate: dict) -> str:
    return report._join([_BAR_WORDS[b] for b in _BARS if not gate["bars"][b]])


def sheet(runs, rounds_of) -> list:
    """The Combinations sheet: one row per combination. `rounds_of(punch)` gives its round number (1-based)."""
    out = [["round", "start (s)", "start (m:ss)", "punches", "combination", "numbers", "includes an unnamed punch"]]
    for r in runs:
        if len(r) < 2:
            continue
        s = int(r[0]["time_s"])
        nm = name_of(r)
        out.append([rounds_of(r[0]), round(r[0]["time_s"], 3), f"{s // 60}:{s % 60:02d}", len(r), spoken(nm),
                    numbered(nm), "yes" if UNKNOWN in nm else ""])
    return out


# ----------------------------------------------------------------------------- the gate (scoring) ---

def label_punches(spans, centers, fps: float) -> "list[dict]":
    """Labelled punches as `group` takes them (one unit: the whole clip). `spans` exclude framing marks; an
    unclear label is a punch nobody could name ("?"). `centers`: punchfind.centers(spans)."""
    if len(spans) != len(centers):
        raise ValueError("one centre per labelled span")
    out = []
    for j, (s, c) in enumerate(zip(spans, centers)):
        lab = s.label
        if lab != -1 and not 0 <= lab < len(CLASSES):
            raise ValueError(f"label {lab!r} is not a punch type or unclear")
        c = float(c)
        out.append({"frame": c, "punch_type": CLASSES[lab] if lab >= 0 else report.UNCLASSIFIED,
                    "time_s": c / fps, "unit": 0, "_label": j})
    return out


def score_clip(rows, spans, centers, pairs, fps: float) -> dict:
    """One clip's runs and matching, everything the gate needs. `rows`: report.read_classified's rows (all
    detections, in that order); `pairs`: (span index, detection index) from classify_punches.attribute."""
    det = [{**r, "unit": 0, "_det": i} for i, r in enumerate(rows)]
    lab = label_punches(spans, centers, fps)
    span_of = {}
    for si, di in pairs:
        if not (0 <= si < len(spans) and 0 <= di < len(rows)) or di in span_of:
            raise ValueError("the matching is not a one-to-one map of these detections to these labels")
        span_of[di] = si
    if len(set(span_of.values())) != len(span_of):
        raise ValueError("the matching is not a one-to-one map of these detections to these labels")
    found, labelled = group(det, fps), group(lab, fps)
    return {"found": found, "labelled": labelled, "span_of": span_of,
            "det_types": [type_of(p) for p in det], "label_types": [type_of(p) for p in lab]}


def _recovered(sc: dict, det_types) -> "tuple[int, int, int]":
    """(named label combinations recovered exactly, named label combinations, found named combinations that
    recover one) with the detections' types `det_types` (permuted for the floor)."""
    lab_combo = {}
    for k, r in enumerate(sc["labelled"]):
        if len(r) >= 2:
            for p in r:
                lab_combo[p["_label"]] = k
    named_lab = [r for r in sc["labelled"] if len(r) >= 2 and UNKNOWN not in name_of(r)]
    hits = set()
    precise = 0
    for f in sc["found"]:
        if len(f) < 2 or any(det_types[p["_det"]] == UNKNOWN for p in f):
            continue
        spans = [sc["span_of"].get(p["_det"]) for p in f]
        if any(s is None for s in spans):
            continue
        k = lab_combo.get(spans[0])
        if k is None:
            continue
        lr = sc["labelled"][k]
        if (UNKNOWN in name_of(lr) or len(lr) != len(f)
                or [p["_label"] for p in lr] != spans
                or [det_types[p["_det"]] for p in f] != [sc["label_types"][s] for s in spans]):
            continue
        hits.add(k)
        precise += 1
    return len(hits), len(named_lab), precise


def _structure(sc: dict) -> "tuple[int, int]":
    """(label combinations whose punches are exactly one found combination's, label combinations)."""
    found_sets = {tuple(sc["span_of"].get(p["_det"]) for p in f) for f in sc["found"] if len(f) >= 2}
    labs = [r for r in sc["labelled"] if len(r) >= 2]
    return sum(tuple(p["_label"] for p in r) in found_sets for r in labs), len(labs)


def gate_result(clips: dict, rule: dict, classifier_sha256: str, fps_of: dict,
                n_perm: int = GATE_PERMUTATIONS, seed: int = GATE_SEED) -> dict:
    """The gate's result (docs/gates/combinations.md). `clips`: {clip: (score_clip's dict, extra provenance)}."""
    obs_hit = obs_n = 0
    per = {}
    for clip, (sc, prov) in clips.items():
        hit, n, precise = _recovered(sc, sc["det_types"])
        found_named = sum(1 for f in sc["found"] if len(f) >= 2 and UNKNOWN not in name_of(f))
        s_hit, s_n = _structure(sc)
        fav_f, fav_l = favourite(sc["found"]), favourite(sc["labelled"])
        st_f, st_l = starter(sc["found"]), starter(sc["labelled"])
        fps = fps_of[clip]
        blind = sum(1 for r in sc["labelled"] if len(r) >= 2
                    and any((b["frame"] - a["frame"]) / fps < rule["min_gap_s"] - _EPS for a, b in zip(r, r[1:])))
        obs_hit, obs_n = obs_hit + hit, obs_n + n
        per[clip] = {**prov, "label_combinations": s_n, "named_label_combinations": n, "recovered_exactly": hit,
                     "found_named_combinations": found_named, "found_named_that_recover": precise,
                     "structure_recovered": s_hit,
                     "label_combinations_with_a_gap_under_min_gap": blind,
                     "found_combinations": sum(1 for f in sc["found"] if len(f) >= 2),
                     "found_punches_in_combinations": sum(len(f) for f in sc["found"] if len(f) >= 2),
                     "found_punches": sum(len(f) for f in sc["found"]),
                     "label_punches_in_combinations": sum(len(r) for r in sc["labelled"] if len(r) >= 2),
                     "label_punches": sum(len(r) for r in sc["labelled"]),
                     "favourite_found": [spoken(x) for x in fav_f], "favourite_labels": [spoken(x) for x in fav_l],
                     "starter_found": [NAME[x] for x in st_f], "starter_labels": [NAME[x] for x in st_l],
                     "favourite_matches": bool(fav_f) and set(fav_f) <= set(fav_l),
                     "starter_matches": bool(st_f) and set(st_f) <= set(st_l)}
    rng = np.random.default_rng(seed)
    floor = []
    order = sorted(clips)
    for _ in range(n_perm):
        hit = 0
        for clip in order:
            sc = clips[clip][0]
            types = list(sc["det_types"])
            idx = [i for i, t in enumerate(types) if t != UNKNOWN]
            perm = rng.permutation(len(idx))
            shuffled = list(types)
            for a, b in zip(idx, perm):
                shuffled[a] = types[idx[b]]
            hit += _recovered(sc, shuffled)[0]
        floor.append(hit / obs_n if obs_n else 0.0)
    p975 = float(np.percentile(floor, GATE_PERCENTILE)) if floor else 0.0
    recall = obs_hit / obs_n if obs_n else 0.0
    bars = {"1_better_than_chance": bool(obs_n) and recall > p975 + _EPS,
            "2_favourite_matches": all(per[c]["favourite_matches"] for c in order),
            "3_starter_matches": all(per[c]["starter_matches"] for c in order)}
    return {"format": GATE_FORMAT, "document": GATE_DOC, "gap_s": COMBO_GAP_S, "rule": dict(rule),
            "finder_per_hand": False,   # score_combinations.py scores the pooled finder only
            "classifier_sha256": classifier_sha256, "clips": per, "bars": bars, "PASS": all(bars.values()),
            "pooled": {"recovered_exactly": obs_hit, "named_label_combinations": obs_n, "exact_recall": recall,
                       "floor_mean": float(np.mean(floor)) if floor else 0.0, "floor_p975": p975,
                       "permutations": n_perm, "seed": seed}}


# --------------------------------------------------------------------- the gate (read by reports) ---

def _num(x, what: str, lo: float, hi: float) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or (isinstance(x, int) and abs(x) > 10 ** 9) \
            or not math.isfinite(x) or not lo <= x <= hi:
        raise ValueError(f"combination gate: {what} {str(x)[:20]!r} is not a number in [{lo:g}, {hi:g}]")
    return float(x)


def _check_clip(name: str, c, fname: str) -> "tuple[int, int]":
    """One clip's block of a gate result, internally consistent: every count a count, every list a list of names,
    and the headline matches exactly what its lists say. Returns (recovered, named label combinations)."""
    if not isinstance(c, dict) or set(c) != _CLIP_KEYS:
        raise ValueError(f"{fname}: clip {name} is not a gate clip block")
    v = {k: report._count(c[k], f"combination gate: {name} {k}") for k in _CLIP_COUNTS}
    if not (v["recovered_exactly"] <= v["named_label_combinations"] <= v["label_combinations"]
            and v["structure_recovered"] <= v["label_combinations"]
            and v["label_combinations_with_a_gap_under_min_gap"] <= v["label_combinations"]
            and v["found_named_that_recover"] <= v["found_named_combinations"] <= v["found_combinations"]
            and v["recovered_exactly"] == v["found_named_that_recover"]
            and v["found_punches_in_combinations"] <= v["found_punches"]
            and v["label_punches_in_combinations"] <= v["label_punches"]):
        raise ValueError(f"{fname}: clip {name}'s counts contradict each other")
    names = set(NAME.values()) - {UNKNOWN}
    for k in _CLIP_LISTS:
        x = c[k]
        if not isinstance(x, list) or len(x) > 64 or not all(isinstance(e, str) and len(e) <= 200 for e in x):
            raise ValueError(f"{fname}: clip {name} {k} is not a list of names")
        if any(not set(e.split("-")) <= names for e in x):
            raise ValueError(f"{fname}: clip {name} {k} names something that is not a punch type")
    for k, a, b in (("favourite_matches", "favourite_found", "favourite_labels"),
                    ("starter_matches", "starter_found", "starter_labels")):
        if not isinstance(c[k], bool) or c[k] != (bool(c[a]) and set(c[a]) <= set(c[b])):
            raise ValueError(f"{fname}: clip {name} {k} is not what its lists say")
    return v["recovered_exactly"], v["named_label_combinations"]


def read_gate(path, rule: dict, classifier_sha256: str, per_hand: bool = False) -> dict:
    """The gate result for the report: {"status": "pass" | "fail", numbers} when it is the REGISTERED run
    (GATE_CLIPS, GATE_PERMUTATIONS, GATE_SEED; docs/gates/combinations.md) measured with this classifier, this
    finder rule (report._rule_of's three settings), the pooled finder (`per_hand` False) and COMBO_GAP_S;
    {"status": "not_applicable"} otherwise. A file that is not a well-formed, internally consistent result is
    refused (ValueError): PASS must be exactly its bars, bar 1 exactly its recall against its floor, bars 2 and 3
    exactly its clips' headline matches, each clip's headline exactly its lists, and the pooled counts the sums
    of the clips' (Astra, review 1)."""
    fname = Path(path).name
    with Path(path).open("rb") as f:
        raw = f.read(MAX_GATE_BYTES + 1)
    if len(raw) > MAX_GATE_BYTES:
        raise ValueError(f"{fname}: more than {MAX_GATE_BYTES} bytes")

    def no_constant(c):
        raise ValueError(f"{fname} contains {c}, which is not valid JSON")
    try:
        d = json.loads(raw.decode("utf-8"), parse_constant=no_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ValueError(f"{fname}: not valid JSON") from None
    if not isinstance(d, dict) or set(d) != _GATE_KEYS or d["format"] != GATE_FORMAT or d["document"] != GATE_DOC:
        raise ValueError(f"{fname}: not a {GATE_FORMAT} result")
    bars = d["bars"]
    if not isinstance(bars, dict) or set(bars) != set(_BARS) or not all(isinstance(bars[b], bool) for b in _BARS):
        raise ValueError(f"{fname}: every bar must be true or false")
    if not isinstance(d["PASS"], bool) or d["PASS"] != all(bars.values()):
        raise ValueError(f"{fname}: PASS is not what its bars say")
    if not isinstance(d["finder_per_hand"], bool):
        raise ValueError(f"{fname}: finder_per_hand must be true or false")
    clips = d["clips"]
    if not isinstance(clips, dict) or not clips or len(clips) > 64:
        raise ValueError(f"{fname}: no clips")
    per = [_check_clip(k[:64], clips[k], fname) for k in sorted(clips)]
    if (bars["2_favourite_matches"] != all(clips[k]["favourite_matches"] for k in clips)
            or bars["3_starter_matches"] != all(clips[k]["starter_matches"] for k in clips)):
        raise ValueError(f"{fname}: bars 2 and 3 are not what its clips say")
    p = d["pooled"]
    if not isinstance(p, dict) or set(p) != _POOLED_KEYS:
        raise ValueError(f"{fname}: no pooled result")
    n = report._count(p["named_label_combinations"], "combination gate: labelled combinations")
    hit = report._count(p["recovered_exactly"], "combination gate: recovered combinations")
    if (hit, n) != (sum(h for h, _ in per), sum(m for _, m in per)):
        raise ValueError(f"{fname}: the pooled counts are not the sums of its clips'")
    recall = _num(p["exact_recall"], "exact recall", 0, 1)
    if abs(recall - (hit / n if n else 0.0)) > 1e-9:
        raise ValueError(f"{fname}: exact recall is not recovered / labelled")
    floor = _num(p["floor_p975"], "floor", 0, 1)
    _num(p["floor_mean"], "floor mean", 0, 1)
    perms = report._count(p["permutations"], "combination gate: permutations")
    seed = report._count(p["seed"], "combination gate: seed")
    if bars["1_better_than_chance"] != (bool(n) and recall > floor + _EPS):
        raise ValueError(f"{fname}: bar 1 is not what its recall and floor say")
    if not isinstance(d["classifier_sha256"], str) or not _HEX64.fullmatch(d["classifier_sha256"]):
        raise ValueError(f"{fname}: no classifier sha256")
    gap = _num(d["gap_s"], "gap", 0, 60)
    try:
        theirs = report._rule_of(d["rule"], fname)
    except ValueError:
        raise ValueError(f"{fname}: no finder rule") from None
    registered = tuple(sorted(clips)) == GATE_CLIPS and perms == GATE_PERMUTATIONS and seed == GATE_SEED
    if (not registered or theirs != report._rule_of(rule, "the punches' finder rule")
            or d["classifier_sha256"] != classifier_sha256 or gap != COMBO_GAP_S
            or d["finder_per_hand"] is not False or per_hand is not False):
        return {"status": "not_applicable"}
    return {"status": "pass" if d["PASS"] else "fail", "bars": dict(bars), "recovered": hit, "n_label": n,
            "recall": recall, "floor": floor, "sha256": hashlib.sha256(raw).hexdigest()}
