"""Tests for the measured identity error in the per-boxer report (the linker's gate, ADR-0016 + ADR-0018).

The linker's identity sidecar may record its gate verdict with the join check's result.json; read with that file,
the identity carries the measured join error and the report's first sentence quotes it. Every number must come from
the checked result file, re-derived from its own counts: a sidecar or a result edited by hand is refused.
Synthetic only (a two-person table, a synthetic join-check sample); no Olympic file is read. A plain script.

  python tests/test_identity_measured.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import olympic_boxers  # noqa: E402
from boxingvi import boxers, report  # noqa: E402
from boxingvi import joincheck as jc  # noqa: E402
from boxingvi.data import N_JOINTS  # noqa: E402
from boxingvi.pose_yolo import PoseTable  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f"  ({detail})" if detail else ""))


def refused(name: str, fn, needle: str) -> None:
    try:
        fn()
    except ValueError as e:
        check(name, needle.lower() in str(e).lower(), f"refused, but with: {e}")
        return
    except Exception as e:     # noqa: BLE001 -- a crash is not a refusal
        check(name, False, f"{type(e).__name__}: {e}")
        return
    check(name, False, "was accepted")


tmp = Path(tempfile.mkdtemp(prefix="identity_measured_test_"))
N, FPS, W, H = 300, 50.0, 320, 180

# ---- a two-person tracks file and the linker's segments -------------------------------------------
rows = [(f, t) for f in range(N) for t in (1, 2)]
k = len(rows)
xy = np.zeros((k, N_JOINTS, 2), np.float32)
for i, (f, t) in enumerate(rows):
    xy[i, :, 0] = 80 if t == 1 else 240
    xy[i, :, 1] = np.linspace(20, 160, N_JOINTS)
box = np.stack([xy[:, :, 0].min(1) - 10, xy[:, :, 1].min(1), xy[:, :, 0].max(1) + 10, xy[:, :, 1].max(1)], 1)
table = PoseTable(np.array([r[0] for r in rows], np.int64), np.array([r[1] for r in rows], np.int64), box, xy,
                  np.full((k, N_JOINTS), 0.9, np.float32), N, FPS, W, H, source="BOUT01.mp4")
data = tmp / "data"
data.mkdir()
tracks = data / "BOUT01.tracks.npz"
table.save(tracks)
table = PoseTable.load(tracks)
SEGS = [boxers.Segment(1, 0, N - 1, "A"), boxers.Segment(2, 0, N - 1, "B")]

# ---- a real result.json, from joincheck.score on a synthetic sample ---------------------------------
MAN = frozenset({"V1", "V2", "V3"})
fj = [jc.Join(v, 2 * i, 2 * i + 1, 1000 + 300 * i, 1000 + 300 * i + 1 + i % 100) for v in sorted(MAN) for i in range(100)]


def sample_and_score(n_d: int, n_u: int, positions, name: str) -> Path:
    pv = {v: {"joins": [j for j in fj if j.video == v], "positions": positions, "chain_minutes": 10.0,
              "camera": "kam2", "fps": 50.0, "n_frames": 40_000, "tracks_sha256": "a" * 64, "video_fp": "b" * 64}
          for v in MAN}
    d = tmp / "results" / name
    d.mkdir(parents=True)
    s = jc.parse_sample(jc.sample_bytes(jc.build_sample(pv, MAN)), MAN)
    (d / "sample.json").write_bytes(jc.sample_bytes(s.raw))
    a = jc.AnswerFile.create(d / "answers.csv", s)
    for i in range(s.n):
        a.set_value(i, "D" if i < n_d else ("U" if i < n_d + n_u else "S"))
    a.save()
    r = jc.score(s, jc.AnswerFile.load(d / "answers.csv", s), draws=200) | {"sample_sha256": s.sha256,
                                                                              "gate": jc.GATE}
    p = d / "result.json"
    p.write_text(json.dumps(r, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    return p


CHAINS = [[k_, 1] for k_ in range(24)]          # ~23 joins per chain: p* about 1%
unreliable = sample_and_score(64, 2, CHAINS, "unreliable")
acceptable = sample_and_score(0, 0, [[1, 1]], "acceptable")      # one join per chain: p* = 10%
nothing = sample_and_score(0, 100, CHAINS, "nothing")

print("the result file")
v, m, sha = boxers.read_join_result(unreliable)
check("an unreliable result records 'fail' and its sha256", v == "fail" and sha == hashlib.sha256(
    unreliable.read_bytes()).hexdigest())
check("its numbers are re-derived from the counts", m["decided"] == 98 and m["different"] == 64
      and abs(m["join_error"] - 64 / 98) < 1e-12 and (m["lower"], m["upper"]) == jc.wilson(64, 98)
      and m["verdict"] == "measured, unreliable" and m["footage"] == "Olympic Boxing")
check("an acceptable result records 'pass'", boxers.read_join_result(acceptable)[0] == "pass")
refused("a result with nothing decided gives no measured error", lambda: boxers.read_join_result(nothing),
        "nothing was decided")


def edited(fn, name: str, sample_from: Path = unreliable) -> Path:
    d = json.loads(unreliable.read_text())
    fn(d)
    folder = tmp / "edited" / name
    folder.mkdir(parents=True)
    if sample_from is not None:
        shutil.copy(sample_from.with_name("sample.json"), folder / "sample.json")
    p = folder / "result.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


refused("a result with no sample.json beside it is refused (its p* could not be checked)",
        lambda: boxers.read_join_result(edited(lambda d: None, "nosample", None)), "no sample.json")
refused("a result beside another sample is refused",
        lambda: boxers.read_join_result(edited(lambda d: None, "swapped", acceptable)), "not the sample it was scored on")


for label, fn, needle in (
        ("an edited join error", lambda d: d.update(join_error=0.01), "not what the answer counts give"),
        ("an edited interval", lambda d: d.update(wilson95=[0.0, 0.005]), "not what the answer counts give"),
        ("an edited verdict", lambda d: d.update(verdict="measured, acceptable"), "verdict is not what"),
        ("an inflated p* (it could turn a result into a pass)", lambda d: d.update(p_star=0.5),
         "not the one its sample gives"),
        ("counts that do not add up", lambda d: d["answers"].update(total=99), "do not add up"),
        ("a decided count that disagrees", lambda d: d.update(decided=90), "do not add up"),
        ("a fractional count", lambda d: d["answers"].update(D=64.0), "whole numbers"),
        ("another gate's document", lambda d: d.update(gate="docs/gates/glove-identity.md"), "not the linker"),
        ("an extra key", lambda d: d.update(boxer_a="red"), "not the linker"),
        ("a missing key", lambda d: d.pop("wilson95"), "not the linker")):
    p = edited(fn, label.replace(" ", "_").replace("*", "").replace("(", "").replace(")", ""))
    refused(f"{label} is refused", lambda p=p: boxers.read_join_result(p), needle)

print("the identity")


def identity(name: str, gate: dict) -> Path:
    d = data / name
    d.mkdir()
    p = d / "BOUT01.identity.csv"
    boxers.write_identity(p, table, tracks, SEGS, "linker", gate)
    return p


fail_gate = olympic_boxers.linker_gate(unreliable)[0]
check("olympic_boxers records the linker's measured verdict, the gate document and the result's hash",
      fail_gate == {"verdict": "fail", "document": "docs/gates/linker-join-check.md", "result_sha256": sha})
check("... and 'ungated' without the join check", olympic_boxers.linker_gate(None) == ({"verdict": "ungated"}, None))
p_fail = identity("fail", fail_gate)
idn = boxers.read_identity(p_fail, table, tracks, unreliable)
check("read with its result, the identity carries the measured error", idn.verdict == "fail"
      and idn.measured == m)
idn_plain = boxers.read_identity(p_fail, table, tracks)
check("read without it, the identity is still a fail, with no number", idn_plain.verdict == "fail"
      and idn_plain.measured is None)
other = tmp / "other.json"
other.write_bytes(unreliable.read_bytes() + b" ")
refused("a different result file is refused (sha256)", lambda: boxers.read_identity(p_fail, table, tracks, other),
        "not the gate result the identity recorded")
p_lie = identity("lie", {**fail_gate, "verdict": "pass"})
refused("a sidecar claiming PASS on an unreliable result is refused",
        lambda: boxers.read_identity(p_lie, table, tracks, unreliable), "says FAIL")
refused("a sidecar claiming PASS without its result is refused", lambda: boxers.read_identity(p_lie, table, tracks),
        "give its gate result")
acc_gate = olympic_boxers.linker_gate(acceptable)[0]
p_pass = identity("pass", acc_gate)
idp = boxers.read_identity(p_pass, table, tracks, acceptable)
check("an acceptable result gives a checked PASS with its numbers", idp.verdict == "pass" and idp.measured["different"] == 0)
refused("a linker gate naming another document is refused",
        lambda: identity("wrongdoc", {**fail_gate, "document": "docs/gates/glove-identity.md"}), "not the gate")
refused("an ungated identity given a result file is refused",
        lambda: boxers.read_identity(identity("ungated", {"verdict": "ungated"}), table, tracks, unreliable),
        "records no gate")

print("the header")


def traced(text, numbers) -> bool:
    return Counter(report.numbers_in(text)) == Counter(t for n in numbers for t in report.numbers_in(n))


text, nums = boxers.header("A", "linker", "fail", 3, 40, 95.0, 1200.0, idn.measured)
first = text.split(". ")[0]
check("the first sentence warns and quantifies: the count, the rate and its interval",
      first.startswith("Warning: which boxer threw each punch is unverified")
      and "hand check of 98 of its joins on Olympic Boxing video" in first and "65% joined two different people" in first
      and f"95% interval {100 * m['lower']:.0f}% to {100 * m['upper']:.0f}%" in first, first)
check("every number in it came from the templates", traced(text, nums))
plain, _ = boxers.header("A", "linker", "fail", 3, 40, 95.0, 1200.0)
check("without a measured error the warning is the old one, with no number in it",
      "hand check" not in plain and plain.startswith("Warning:"))
t_pass, n_pass = boxers.header("B", "linker", "pass", 0, 10, 95.0, 1200.0, idp.measured)
check("a measured PASS has no warning and states the error", not t_pass.startswith("Warning")
      and "Its error was measured: in a hand check of 100 of its joins" in t_pass and traced(t_pass, n_pass), t_pass)
refused("an ungated method with a measured error is refused",
        lambda: boxers.header("A", "linker", "ungated", 0, 0, 1.0, 1.0, idn.measured), "ungated")
for label, bad in (("a rate outside its interval", {**m, "join_error": 0.9}),
                   ("a typed-in string", {**m, "join_error": "65%"}),
                   ("other footage", {**m, "footage": "sparring"}),
                   ("an extra field", {**m, "boxer": "A"}),
                   ("more different than decided", {**m, "different": 99})):
    refused(f"a measured error with {label} is refused",
            lambda bad=bad: boxers.header("A", "linker", "fail", 0, 0, 1.0, 1.0, bad), "measured identity error")

print("the Identity sheet")
sheet = dict((r[0], r[1]) for r in boxers.identity_sheet(idn, "A", {"identified_frames": N, "exposure_frames": N,
                                                                    "segments": 1, "short_segments": 0,
                                                                    "bodyless_segments": 0}, 0, 0))
check("the sheet records the measured error, its interval, the counts and the footage",
      sheet.get("identity join error") == round(64 / 98, 4) and sheet.get("joins decided (same or different)") == 98
      and sheet.get("identity join error, Wilson 95% upper") == round(jc.wilson(64, 98)[1], 4)
      and "Olympic Boxing" in sheet.get("measured on", ""))
sheet_plain = [r[0] for r in boxers.identity_sheet(idn_plain, "A", {"identified_frames": N, "exposure_frames": N,
                                                                    "segments": 1, "short_segments": 0,
                                                                    "bodyless_segments": 0}, 0, 0)]
check("without a measured error the sheet is unchanged", "identity join error" not in sheet_plain
      and sheet_plain[:3] == ["identity method", "gate verdict", "gate document"])

print(f"\n=== {PASS} passed, {FAIL} failed ===")
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(1 if FAIL else 0)
