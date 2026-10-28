"""Tests for the fast-combination tuning and held-out scripts and their scoring module.

What must hold: the grids in code are the gate's; the scorer counts as the end-to-end
evaluation does; the selection follows the gate's order and tie-breaks; the paired
bootstrap pairs; and both scripts REFUSE (no tuning before the O55 split, no held-out
video tuned on, no second held-out run, no rule file that tuning did not write).

Run:  python tests/test_fastcombo_eval.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import eval_fast_combinations as ev  # noqa: E402
import tune_fast_combinations as tune  # noqa: E402
from boxingvi import data as bdata  # noqa: E402
from boxingvi import fastcombo as fc  # noqa: E402
from boxingvi.data import CLASSES, N_JOINTS, Span, Track  # noqa: E402

# Nothing these tests do may touch the repo's real experiment state: the started marker points at a
# temp folder from the first line to the last (never restored: this process ends with the file), and
# the tests run from a temp working directory, so every default relative output lands there too.
REAL_STARTED = tune.STARTED
SANDBOX = Path(tempfile.mkdtemp())
tune.STARTED = SANDBOX / "results" / "fast_combinations_heldout.started"
REAL_RESULTS = ROOT / "results"
RESULTS_BEFORE = sorted(p.name for p in REAL_RESULTS.iterdir()) if REAL_RESULTS.is_dir() else []
os.chdir(SANDBOX)

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def raises(name: str, fn, exc=ValueError) -> None:
    try:
        fn()
    except exc:
        check(name, True)
    except BaseException as e:   # SystemExit included
        check(name, False, f"raised {type(e).__name__}: {e}")
    else:
        check(name, False, "did not raise")


J, C_, LH, LU, RH, RU = (CLASSES.index(c) for c in
                         ("jab", "cross", "lead_hook", "lead_uppercut", "rear_hook", "rear_uppercut"))

print("the grids in code are the ones the committed gate fixes")
gate = (ROOT / "docs" / "gates" / "fast-combinations.md").read_text(encoding="utf-8")


def row(label: str) -> "list[float]":
    line = next(ln for ln in gate.splitlines() if ln.startswith(f"| {label}"))
    return [float(x) for x in re.findall(r"\d+\.\d+", line.split("|")[2])]


check("thresholds: 2.0 to 6.0 in 0.1 steps, 41 values", list(fc.THRESHOLDS) == [round(2 + 0.1 * i, 1) for i in range(41)]
      and row("Threshold")[:2] == [2.0, 6.0] and "41 values" in gate)
check("arm 1 windows match the gate's table", list(fc.WINDOWS_S) == row("Arm 1"), str(row("Arm 1")))
check("arm 2 gaps match the gate's table", list(fc.CROSS_GAPS_S) == row("Arm 2"), str(row("Arm 2")))
check("the V6 rule's threshold lies inside the grid", fc.THRESHOLDS[0] < 3.62 < fc.THRESHOLDS[-1])
check("the margin, groups and fixed settings are the gate's",
      fc.MARGIN == 0.05 and "0.05 below" in gate and fc.MIN_GAP_S == 0.4 and fc.TOL_S == 0.1 and fc.BLOCK_S == 30.0
      and fc.N_BOOT == 2000 and fc.SEED == 0 and set(fc.GROUPS) == {"jab", "cross", "hooks", "uppercuts"})
check("the grid has 41 control + 5 x 41 arm 1 + 6 x 41 arm 2 points", len(list(fc.grid())) == 41 * 12)
check("V2 and V3 swap pose files (O61), every other video its own", fc.POSE_FILE == {"V2": "V3", "V3": "V2"})


def video(spans, unusable=(), n=900, fps=30.0) -> fc.Video:
    tr = Track(np.arange(n, dtype=np.int64), np.zeros((n, N_JOINTS, 2)), fps)
    return fc.Video("Vx", fps, n, tr, np.zeros((n, 2)), list(spans), list(unusable))


print("counts: scored as the end-to-end evaluation scores")
v = video([Span(100, 110, J), Span(200, 210, C_), Span(850, 860, LH)], unusable=[(400, 410)])    # blocks of 900 frames
c = fc.counts(v, [105, 207, 300, 405, 412, 600])
t = c.sum(axis=0)
check("one block at 30 fps over 900 frames", c.shape == (1, fc.N_COLS))
check("hits: the jab and the cross; labels 3", t[fc.HIT_S] == 2 and t[fc.HIT_D] == 2 and t[fc.LAB] == 3, str(t))
check("a detection on an unusable labelled row (405, and 412 within the 3-frame tolerance) is not scored",
      t[fc.DET] == 4, str(t[fc.DET]))
check("per type: jab 1/1, cross 1/1, lead hook 0/1",
      (t[fc.HIT_T + J], t[fc.LAB_T + J], t[fc.HIT_T + C_], t[fc.HIT_T + LH], t[fc.LAB_T + LH]) == (1, 1, 1, 0, 1))
check("precision 2/4, recall 2/3, F1 their harmonic mean",
      fc.precision(t) == 0.5 and abs(fc.recall(t) - 2 / 3) < 1e-12 and abs(fc.f1(t) - 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3)) < 1e-12)
v2 = video([Span(100, 110, J), Span(1000, 1010, C_)], n=2000)
c2 = fc.counts(v2, [105, 1950])
check("blocks: a span counts in its start's block, a detection in its own", c2.shape == (3, fc.N_COLS)
      and c2[0, fc.LAB] == 1 and c2[1, fc.LAB] == 1 and c2[2, fc.DET] == 1 and c2[0, fc.HIT_S] == 1)
raises("a detection outside the video is refused", lambda: fc.counts(v, [900]))
raises("a negative detection frame is refused", lambda: fc.counts(v, [-1]))
check("no detections: precision undefined, F1 undefined, recall 0", (lambda z: fc.precision(z) is None and fc.f1(z) is None
      and fc.recall(z) == 0.0)(fc.counts(v, []).sum(axis=0)))
check("detections but no hit: F1 is 0, not undefined", fc.f1(fc.counts(v, [600]).sum(axis=0)) == 0.0)


def totals(**rec) -> np.ndarray:
    """A count row with the given (found, labelled) per type; precision columns 1/1."""
    t = np.zeros(fc.N_COLS, np.int64)
    t[fc.DET] = t[fc.HIT_D] = 1
    for name, (f, n) in rec.items():
        i = CLASSES.index(name)
        t[fc.HIT_T + i], t[fc.LAB_T + i] = f, n
        t[fc.HIT_S] += f
        t[fc.LAB] += n
    return t


print("admissible: jab up, no guarded group down by more than 0.05")
base = totals(jab=(3, 10), cross=(6, 10), lead_hook=(5, 10), rear_hook=(5, 10), lead_uppercut=(4, 10), rear_uppercut=(4, 10))
ok_ = totals(jab=(5, 10), cross=(6, 10), lead_hook=(5, 10), rear_hook=(5, 10), lead_uppercut=(4, 10), rear_uppercut=(4, 10))
check("jab up, the rest equal: admissible", fc.admissible(ok_, base) == (True, []))
check("jab equal: not admissible (it must RISE)", not fc.admissible(base, base)[0])
edge = totals(jab=(5, 10), cross=(11, 20), lead_hook=(5, 10), rear_hook=(5, 10), lead_uppercut=(4, 10), rear_uppercut=(4, 10))
check("cross down by exactly 0.05 (0.60 -> 0.55): admissible", fc.admissible(edge, base)[0], str(fc.admissible(edge, base)))
low = totals(jab=(5, 10), cross=(6, 10), lead_hook=(4, 10), rear_hook=(4, 10), lead_uppercut=(4, 10), rear_uppercut=(4, 10))
check("hooks down by 0.10: not admissible, and the reason names hooks",
      (lambda r: not r[0] and any("hooks" in w for w in r[1]))(fc.admissible(low, base)))
nohook = totals(jab=(5, 10), cross=(6, 10), lead_uppercut=(4, 10), rear_uppercut=(4, 10))
check("a guarded group with no labels is not admissible (cannot be checked)", not fc.admissible(nohook, nohook)[0])

print("choose: highest F1 among admissible, ties to the higher threshold then the smaller parameter")
P = lambda thr, prm, f, ok=True: {"threshold": thr, "param": prm, "f1": f, "admissible": ok}  # noqa: E731
check("the highest admissible F1 wins over a higher inadmissible one",
      fc.choose([P(3.0, 0.1, 0.5), P(3.1, 0.1, 0.6, False), P(3.2, 0.1, 0.55)]) == P(3.2, 0.1, 0.55))
check("an F1 tie goes to the higher threshold", fc.choose([P(3.0, 0.1, 0.5), P(4.0, 0.1, 0.5)])["threshold"] == 4.0)
check("then to the smaller parameter", fc.choose([P(4.0, 0.2, 0.5), P(4.0, 0.1, 0.5)])["param"] == 0.1)
check("no admissible point: the highest F1 overall, still flagged inadmissible",
      fc.choose([P(3.0, 0.1, 0.4, False), P(3.1, 0.1, 0.45, False)]) == P(3.1, 0.1, 0.45, False))
check("an undefined F1 ranks last", fc.choose([P(3.0, 0.1, None), P(3.1, 0.1, 0.01)])["f1"] == 0.01)
check("nothing to choose from: None", fc.choose([]) is None)
check("choose_arm: the higher training F1", fc.choose_arm({"per_hand_extension": P(3, .1, .61),
                                                          "cross_hand_gap": P(3, .1, .60)}) == "per_hand_extension")
check("choose_arm: an exact tie goes to arm 2", fc.choose_arm({"per_hand_extension": P(3, .1, .6),
                                                              "cross_hand_gap": P(3, .1, .6)}) == "cross_hand_gap")

print("paired bootstrap")
rng = np.random.default_rng(1)
bb = {"V5": np.zeros((18, fc.N_COLS), np.int64), "V9": np.zeros((32, fc.N_COLS), np.int64)}
for k_ in bb:
    bb[k_][:, fc.LAB] = bb[k_][:, fc.DET] = 10
    bb[k_][:, fc.HIT_S] = bb[k_][:, fc.HIT_D] = rng.integers(3, 7, len(bb[k_]))
same = fc.paired_ci(bb, bb, fc.f1)
check("a finder against itself: difference 0, interval [0, 0], not excluding zero",
      same["point"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0 and not same["excludes_zero"])
better = {k_: x.copy() for k_, x in bb.items()}
for x in better.values():
    x[:, fc.HIT_S] += 1
    x[:, fc.HIT_D] += 1
up = fc.paired_ci(better, bb, fc.f1)
check("one more hit in every block: the interval excludes zero, above it (PAIRED: unpaired would not be this tight)",
      up["ci_low"] > 0 and up["excludes_zero"] and up["ci_high"] - up["ci_low"] < 0.02, str(up))
check("the same seed gives the same interval", fc.paired_ci(better, bb, fc.f1) == up)
check("blocks are reported per video", up["blocks"] == {"V5": 18, "V9": 32})
raises("different videos are refused", lambda: fc.paired_ci({"V5": bb["V5"]}, bb, fc.f1))
raises("different block counts are refused", lambda: fc.paired_ci({"V5": bb["V5"][:3], "V9": bb["V9"]}, bb, fc.f1))

print("verdict: all three conditions, each reported")
ci_up = {"point": 0.05, "ci_low": 0.01}
check("F1 up with an interval above zero, jab up, nothing down: PASS", fc.verdict(ok_, base, ci_up)["pass"])
check("an interval touching zero: FAIL", not fc.verdict(ok_, base, {"point": 0.05, "ci_low": 0.0})["pass"])
check("jab not up: FAIL", not fc.verdict(base, base, ci_up)["pass"])
v_low = fc.verdict(low, base, ci_up)
check("hooks down 0.10: FAIL, and the guard says which", not v_low["pass"] and not v_low["guarded_groups"]["hooks"]["ok"]
      and v_low["guarded_groups"]["cross"]["ok"])
check("cross down exactly 0.05: still passes the guard", fc.verdict(edge, base, ci_up)["no_group_down_more_than_margin"])

print("tune script: refusals")
had = hasattr(bdata, "PERSON_OF_VIDEO")
saved = getattr(bdata, "PERSON_OF_VIDEO", None)
if had:
    delattr(bdata, "PERSON_OF_VIDEO")
raises("without the O55 split (no data.PERSON_OF_VIDEO) tuning refuses", tune.split_ready, SystemExit)
raises("... and so does the held-out run", lambda: ev.main(["--data", "x", "--poses", "x", "--classifier", "x",
                                                             "--stance", "orthodox"]), SystemExit)
bdata.PERSON_OF_VIDEO = saved if had else {}
tmp = Path(tempfile.mkdtemp())
held_saved = bdata.HELD_OUT_VIDEOS
bdata.HELD_OUT_VIDEOS = ("V2", "V3", "V5", "V9")
check("training videos are every source not held out, in order",
      tune.training_videos() == ["V1", "V4", "V6", "V7", "V8", "V10"], str(tune.training_videos()))
check("the started marker is a fixed path under the repo, not an option",
      REAL_STARTED == ROOT / "results" / "fast_combinations_heldout.started"
      and "--heldout" not in tune.main.__code__.co_consts)
tune.STARTED.parent.mkdir(parents=True, exist_ok=True)
tune.STARTED.write_text("{}", encoding="utf-8")
raises("tuning again after the held-out run started is refused",
       lambda: tune.main(["--data", "x", "--poses", str(tmp)]), SystemExit)
raises("a second held-out run is refused whatever --out says",
       lambda: ev.main(["--data", "x", "--poses", "x", "--classifier", "x", "--stance", "orthodox",
                        "--out", str(tmp / "a-fresh-name.json")]), SystemExit)
tune.STARTED.unlink()
orig_git = tune._git
tune._git = lambda *a: " M docs/gates/fast-combinations.md" if a[0] == "status" else "abc"
raises("an uncommitted change to the gate is refused", tune.gate_commit, SystemExit)
tune._git = lambda *a: ""
raises("a gate that was never committed is refused", tune.gate_commit, SystemExit)
tune._git = orig_git
check("on this checkout the gate is committed and its commit is found", len(tune.gate_commit()) == 40)
v6rule = tmp / "v6.json"
v6rule.write_text(json.dumps({"chosen": {"variant": "relative", "threshold": fc.BASELINE_THRESHOLD,
                                         "min_gap_s": 0.4}}), encoding="utf-8")
check("control: the registered baseline (threshold 3.6239) is accepted", tune.baseline_rule(v6rule)["threshold"]
      == fc.BASELINE_THRESHOLD)
for thr_ in (3.62, 6.0, 2.0, float("nan")):
    alt = tmp / f"alt{thr_}.json"
    alt.write_text(json.dumps({"chosen": {"variant": "relative", "threshold": thr_, "min_gap_s": 0.4}}), encoding="utf-8")
    raises(f"a pooled baseline with threshold {thr_} (not the registered one) is refused",
           lambda a_=alt: tune.baseline_rule(a_), SystemExit)
raises("no pose for a tuning video: refused, naming it",
       lambda: tune.main(["--data", "x", "--poses", str(tmp), "--baseline", str(v6rule)]), SystemExit)
armrule = tmp / "arm.json"
armrule.write_text(json.dumps({"chosen": {"variant": "relative", "threshold": 3.62, "min_gap_s": 0.4,
                                          "finder": {"mode": "cross_hand_gap", "cross_gap_s": 0.15}}}), encoding="utf-8")
raises("an arm as the baseline is refused", lambda: tune.baseline_rule(armrule), SystemExit)
other = tmp / "other.json"
other.write_text(json.dumps({"chosen": {"variant": "relative", "threshold": 3.62, "min_gap_s": 0.25}}), encoding="utf-8")
raises("a pooled rule with another gap as the baseline is refused", lambda: tune.baseline_rule(other), SystemExit)

print("held-out script: refusals")
raises("no tune file: refused", lambda: ev.read_tune(tmp / "missing.json", armrule, tmp / "fz"), SystemExit)
tf = tmp / "tune.json"
tf.write_text(json.dumps({"chosen_arm": "learned"}), encoding="utf-8")
raises("a tune file naming no known arm: refused", lambda: ev.read_tune(tf, armrule, tmp / "fz"), SystemExit)
good = {"chosen_arm": "cross_hand_gap", "rule_file_sha256": tune.sha256(armrule),
        "chosen_rule": {"variant": "relative", "threshold": 3.62, "min_gap_s": 0.4,
                        "finder": {"mode": "cross_hand_gap", "cross_gap_s": 0.15}}}
tf.write_text(json.dumps(good), encoding="utf-8")
check("control: a consistent tune file and rule file are accepted", ev.read_tune(tf, armrule, tmp / "fz")[2]["finder"]["cross_gap_s"] == 0.15)
armbytes = armrule.read_bytes()
t4 = ev.read_tune(tf, armrule, tmp / "fz")
armrule.write_text(json.dumps({"chosen": {"variant": "relative", "threshold": 5.0, "min_gap_s": 0.4,
                                          "finder": {"mode": "per_hand_extension", "extension_window_s": 0.2}}}),
                   encoding="utf-8")       # a retune replaces the rule file AFTER validation
check("interleaving: the frozen copy still holds the validated arm after the rule file is replaced",
      t4[3].read_bytes() == armbytes and ev.find_punches.load_rule(t4[3]) == t4[2]
      and t4[3].resolve() != armrule.resolve(), str(t4[3]))
armrule.write_bytes(armbytes)
raises("freeze refuses a file whose sha256 is not the recorded one", lambda: ev.freeze(armrule, "0" * 64, tmp / "fz" / "x.json"),
       SystemExit)
tf.write_text(json.dumps({**good, "rule_file_sha256": "0" * 64}), encoding="utf-8")
raises("a rule file tuning did not write (sha256): refused", lambda: ev.read_tune(tf, armrule, tmp / "fz"), SystemExit)
tf.write_text(json.dumps({**good, "chosen_arm": "per_hand_extension"}), encoding="utf-8")
raises("a rule file holding the OTHER arm: refused", lambda: ev.read_tune(tf, armrule, tmp / "fz"), SystemExit)
tf.write_text(json.dumps({**good, "chosen_rule": {**good["chosen_rule"], "threshold": 3.0}}), encoding="utf-8")
raises("a rule file whose threshold is not the chosen one: refused", lambda: ev.read_tune(tf, armrule, tmp / "fz"), SystemExit)
(tmp / "out.json").write_text("{}", encoding="utf-8")
raises("an existing --out file: refused",
       lambda: ev.main(["--data", "x", "--poses", "x", "--classifier", "x", "--stance", "orthodox",
                        "--out", str(tmp / "out.json")]), SystemExit)
raises("a baseline that does not reproduce RESULTS 5a-5b stops the run",
       lambda: ev.check_finder_reproduction(np.zeros(fc.N_COLS, np.int64)), SystemExit)
raises("a baseline that does not reproduce RESULTS 5c stops the run",
       lambda: ev.check_named_reproduction({"V5": np.array([[351, 742]])}), SystemExit)
check("control: 352 named right reproduces", ev.check_named_reproduction({"V5": np.array([[300, 500]]),
                                                                          "V9": np.array([[52, 242]])}) == 352)
ev.mark_started({"chosen_arm": "cross_hand_gap"}, "a" * 64)
check("mark_started creates the marker", tune.STARTED.is_file())
raises("mark_started a second time is refused (atomic create)", lambda: ev.mark_started({"chosen_arm": "x"}, "a" * 64), SystemExit)
tune.STARTED.unlink()

print("held-out script: when the baseline fails to reproduce, the arm is never scored")
calls = []
saved_fns = (ev.read_tune, tune.gate_commit, tune.load_video, ev.run_e2e, fc.detections, ev.check_finder_reproduction,
             ev.freeze)
bdata.HELD_OUT_VIDEOS = ("V5", "V9")
pose_dir = tmp / "poses"; pose_dir.mkdir()
for v_ in ("V5", "V9"):
    (pose_dir / f"{v_}.pose.npz").write_bytes(b"")
tf.write_text(json.dumps({**good, "gate_commit": "c" * 40, "held_out_videos_at_tuning": ["V5", "V9"],
                          "baseline_rule_sha256": tune.sha256(v6rule), "tuned_on": ["V4"],
                          "control": {"threshold": 3.0}}), encoding="utf-8")
ev.read_tune = lambda tp, rp, fz: (json.loads(tf.read_text(encoding="utf-8")), "a" * 64,
                                  json.loads(tf.read_text(encoding="utf-8"))["chosen_rule"], fz / "arm_rule.json")
ev.freeze = lambda p_, s_, d_: (d_.parent.mkdir(parents=True, exist_ok=True), d_.write_bytes(p_.read_bytes()), d_)[2]
tune.gate_commit = lambda: "c" * 40
tune.load_video = lambda d, p, v_: (video([Span(100, 110, J)]), {"aligned": True})
ev.check_finder_reproduction = lambda ref: calls.append("finder-reproduced")
ev.run_e2e = lambda a_, rp, od, sha, vids, r_: (calls.append(("e2e", Path(rp).name)), {v_: [] for v_ in ("V5", "V9")})[1]
fc.detections = lambda vd, r, masks=None: (calls.append(("detect", "finder" in r)), np.array([], np.int64))[1]
classify_saved = ev.classify.read_manifest
ev.classify.read_manifest = lambda p: {"sha256": "s"}
raises("named-right 0 instead of 352: the run stops",
       lambda: ev.main(["--data", "x", "--poses", str(pose_dir), "--classifier", "x", "--stance", "orthodox",
                        "--tune", str(tf), "--rule", str(armrule), "--baseline", str(v6rule),
                        "--out", str(tmp / "never.json"), "--e2e-dir", str(tmp / "e2e-fail")]), SystemExit)
check("... after the baseline only: no arm detection, no arm end-to-end run",
      ("detect", True) not in calls and ("e2e", "arm_rule.json") not in calls
      and calls.count(("e2e", "baseline_rule.json")) == 1 and ("e2e", v6rule.name) not in calls, str(calls))
check("... and no started marker and no output were written", not tune.STARTED.exists() and not (tmp / "never.json").exists())
print("held-out script: the whole path, stubbed, runs to a written result")
calls.clear()
named_saved = ev.check_named_reproduction
ev.check_named_reproduction = lambda nb_: (calls.append("named-reproduced"), 352)[1]
fc.detections = lambda vd, r, masks=None: (calls.append(("detect", "finder" in r)), np.array([105], np.int64))[1]
rc_ = ev.main(["--data", "x", "--poses", str(pose_dir), "--classifier", "x", "--stance", "orthodox",
               "--tune", str(tf), "--rule", str(armrule), "--baseline", str(v6rule), "--out", str(tmp / "done.json"),
               "--e2e-dir", str(tmp / "e2e")])
res_ = json.loads((tmp / "done.json").read_text(encoding="utf-8")) if (tmp / "done.json").is_file() else {}
check("it returns 0 and writes the result with a verdict", rc_ == 0 and "verdict" in res_ and "end_to_end" in res_)
check("this invocation's frozen copies and outputs live in its own new folder under --e2e-dir",
      Path(res_.get("run_dir", "")).parent == tmp / "e2e" and (Path(res_.get("run_dir", "x")) / "frozen" / "baseline_rule.json").is_file())
check("order: baseline reproduced (finder, end to end) -> marker -> arm",
      calls.index("named-reproduced") < calls.index(("detect", True)) < calls.index(("e2e", "arm_rule.json")))
check("the arm's end-to-end stage gets the FROZEN rule file, never the one named on the command line",
      ("e2e", armrule.name) not in calls and ("e2e", "arm_rule.json") in calls)
check("the marker now exists, so a rerun is refused", tune.STARTED.is_file())
raises("... a rerun with a fresh --out is refused",
       lambda: ev.main(["--data", "x", "--poses", str(pose_dir), "--classifier", "x", "--stance", "orthodox",
                        "--out", str(tmp / "again.json")]), SystemExit)
tune.STARTED.unlink()
ev.check_named_reproduction = named_saved
(ev.read_tune, tune.gate_commit, tune.load_video, ev.run_e2e, fc.detections, ev.check_finder_reproduction,
 ev.freeze) = saved_fns
ev.classify.read_manifest = classify_saved

check("the reference is RESULTS rows 5a-5c", ev.REFERENCE == {"detections_scored": 686, "labelled": 742, "hits": 434,
                                                             "jab": (84, 264), "named_right": 352})

print("named right: found AND given the label's type")
vn = video([Span(100, 110, J), Span(200, 210, C_), Span(300, 310, LH)])
nc = ev.named_counts(vn, [{"frame": 105, "type": J}, {"frame": 205, "type": J}, {"frame": 500, "type": C_}])
check("jab found and named jab counts; cross found but named jab does not; a miss does not",
      nc.sum(axis=0).tolist() == [1, 3], str(nc.sum(axis=0)))
pc = ev.paired_named_ci({"Vx": nc}, {"Vx": nc})
check("named-right difference of a run against itself is 0", pc["point"] == 0 and pc["ci_low"] == 0 == pc["ci_high"])

print("tune script: the whole path, stubbed, on synthetic punches")
bdata.HELD_OUT_VIDEOS = ("V2", "V3", "V5", "V9")
tpose = tmp / "tposes"; tpose.mkdir()
for v_ in tune.training_videos():
    (tpose / f"{v_}.pose.npz").write_bytes(b"")


def synth(v_):
    n_, fps_ = 600, 30.0
    xy = np.zeros((n_, N_JOINTS, 2))
    xy[:, 5] = (400, 300); xy[:, 6] = (500, 300); xy[:, 11] = (400, 400); xy[:, 12] = (500, 400)
    xy[:, 9] = (400, 350); xy[:, 10] = (500, 350)
    t_ = np.arange(n_) / fps_
    spans_ = []
    # a cross, then a SHORTER jab 0.2 s later: the pooled rule keeps the taller peak and loses the jab
    for k_, (t0, h_, lab_, amp_) in enumerate([(3.0, 1, C_, 100), (3.2, 0, J, 70), (8.0, 0, LH, 100),
                                               (12.0, 1, RU, 100), (15.0, 0, J, 100)]):
        p_ = np.clip((t_ - t0) / 0.12, 0, 1); q_ = np.clip((t_ - t0 - 0.12) / 0.12, 0, 1)
        xy[:, 9 + h_, 0] += (-1 if h_ == 0 else 1) * amp_ * ((1 - np.cos(np.pi * p_)) / 2 - (1 - np.cos(np.pi * q_)) / 2)
        f0 = int(t0 * fps_)
        spans_.append(Span(f0, f0 + 7, lab_))
    tr_ = Track(np.arange(n_, dtype=np.int64), xy, fps_)
    from boxingvi import punchfind as pf_
    return (fc.Video(v_, fps_, n_, tr_, pf_.wrist_signal(tr_), spans_, []),
            {"aligned": v_ != "V8", "best_offset": 0, "sharpness": 2.0})


tune.load_video = lambda d, p, v_: synth(v_)
rc_ = tune.main(["--data", "x", "--poses", str(tpose), "--baseline", str(v6rule),
                 "--out", str(tmp / "tune_out.json"), "--rule-out", str(tmp / "rule_out.json")])
tj = json.loads((tmp / "tune_out.json").read_text(encoding="utf-8")) if (tmp / "tune_out.json").is_file() else {}
check("it returns 0 and writes the grid (492 points) and a chosen arm", rc_ == 0 and len(tj.get("grid", [])) == 492
      and tj.get("chosen_arm") in fc.ARMS, str({k: tj.get(k) for k in ("chosen_arm", "excluded")}))
check("a video failing alignment is excluded and listed, not dropped silently",
      tj.get("excluded") == ["V8"] and "V8" not in tj.get("tuned_on", []) and "V8" in tj.get("videos", {}))
check("V6 counts for recall, not F1", "V6" in tj.get("tuned_on", []) and "V6" not in tj.get("f1_videos", ["V6"]))
check("no held-out video is in the record", not set(tj.get("tuned_on", [])) & {"V2", "V3", "V5", "V9"})
check("the rule file is what find_punches loads, and the tune record pins it by sha256",
      tj.get("rule_file_sha256") == tune.sha256(tmp / "rule_out.json")
      and ev.find_punches.load_rule(tmp / "rule_out.json") == tj.get("chosen_rule"))
check("on these synthetic combinations both arms find more jabs than the pooled baseline",
      all(tj["best_per_arm"][k]["recall_by_group_all_videos"]["jab"]
          > tj["baseline_training"]["all_videos"]["recall_by_group"]["jab"] for k in fc.ARMS), str(tj.get("best_per_arm")))
choose_saved = fc.choose_arm
fc.choose_arm = lambda best: (tune.STARTED.parent.mkdir(parents=True, exist_ok=True), tune.STARTED.write_text("{}"),
                              choose_saved(best))[2]
raises("a held-out run that starts WHILE tuning runs stops tuning before it writes",
       lambda: tune.main(["--data", "x", "--poses", str(tpose), "--baseline", str(v6rule),
                          "--out", str(tmp / "tune_late.json"), "--rule-out", str(tmp / "rule_late.json")]), SystemExit)
check("... and nothing was written", not (tmp / "tune_late.json").exists() and not (tmp / "rule_late.json").exists())
fc.choose_arm = choose_saved
tune.STARTED.unlink()
tune.load_video = saved_fns[2]

bdata.HELD_OUT_VIDEOS = held_saved
check("the tests left the repo's real results/ exactly as they found it, and no real marker exists",
      (sorted(p.name for p in REAL_RESULTS.iterdir()) if REAL_RESULTS.is_dir() else []) == RESULTS_BEFORE
      and not REAL_STARTED.exists(), str(RESULTS_BEFORE))
if not had:
    delattr(bdata, "PERSON_OF_VIDEO")
print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
