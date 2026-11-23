"""The ONE held-out run of the chosen fast-combination arm against the pooled rule.

    python cli/eval_fast_combinations.py --data data --poses results/ingest \\
        --classifier weights/classifier/stgcnpp-boxingvi-seed0.json --stance orthodox

Follows docs/gates/fast-combinations.md. The arm is never an argument: it is read from
results/fast_combinations_tune.json, which tune_fast_combinations.py wrote from training
videos only. Refused, with nothing scored, when:
- the athlete-disjoint split (#38, O55) is not merged, or the held-out list differs from
  the one recorded at tuning;
- the gate changed since tuning, or the rule file is not the one tuning wrote (sha256);
- the held-out run has started before (results/fast_combinations_heldout.started, a fixed
  path created before the arm is scored) or the --out file exists: one run;
- the pooled baseline on V5+V9 does not reproduce RESULTS rows 5a-5c exactly (checked
  before the arm is touched).

Finder metrics on every held-out video that passes the alignment gate (V2 and V3 on each
other's pose, O61). Found-and-named-right on V5+V9 only, through eval_pipeline_v5v9.py,
because the pipeline checkpoint trained on V3. Intervals: paired block bootstrap, 30 s
blocks within each video, the same blocks for both finders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import eval_pipeline_v5v9 as e2e  # noqa: E402
import find_punches  # noqa: E402
import tune_fast_combinations as tune  # noqa: E402
from boxingvi import data as bdata  # noqa: E402
from boxingvi import fastcombo as fc  # noqa: E402
from boxingvi import classify, punchfind  # noqa: E402
from boxingvi.data import CLASSES  # noqa: E402

# RESULTS rows 5a-5c (#20, re-run on main in #30): the pooled rule on V5+V9, nearest matching.
REFERENCE = {"detections_scored": 686, "labelled": 742, "hits": 434, "jab": (84, 264), "named_right": 352}
E2E_VIDEOS = e2e.VIDEOS        # ("V5", "V9")
NAMED, NLAB = 0, 1             # columns of a named-right count row


def freeze(path: Path, want_sha256: "str | None", dest: Path) -> Path:
    """Read `path` ONCE, check its sha256, and write exactly those bytes to `dest`, which every
    later stage reads instead. A file replaced on disk mid-run (a retune, an edit) cannot change
    what is scored after it was checked."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != want_sha256:
        raise SystemExit(f"REFUSED: {path} is not the file tuning recorded (sha256 differs)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    return dest


def read_tune(tune_path: Path, rule_path: Path, frozen: Path) -> "tuple[dict, str, dict, Path]":
    """(tune record, its sha256, the chosen rule, the frozen copy of the rule file). The tune file
    is read once; the rule file is frozen into `frozen` and loaded from there."""
    if not tune_path.is_file():
        raise SystemExit(f"REFUSED: no {tune_path}; run cli/tune_fast_combinations.py on the training videos first")
    raw = tune_path.read_bytes()
    t = json.loads(raw.decode("utf-8"))
    if not isinstance(t, dict) or t.get("chosen_arm") not in fc.ARMS:
        raise SystemExit(f"REFUSED: {tune_path} names no chosen arm")
    rule_copy = freeze(rule_path, t.get("rule_file_sha256"), frozen / "arm_rule.json")
    rule = find_punches.load_rule(rule_copy)
    if rule.get("finder", {}).get("mode") != t["chosen_arm"] or rule != t.get("chosen_rule"):
        raise SystemExit(f"REFUSED: {rule_path} does not hold the arm {tune_path.name} chose")
    return t, hashlib.sha256(raw).hexdigest(), rule, rule_copy


def named_counts(v: fc.Video, rows: "list[dict]") -> np.ndarray:
    """(n_blocks, 2): labelled punches found AND given their label's type, and labels, by the
    span's block. Matching as fc.counts (nearest, the same tolerance)."""
    det = np.array([r["frame"] for r in rows], dtype=np.int64)
    c = np.zeros((v.n_blocks, 2), np.int64)
    for s in v.spans:
        c[s.start // v.block, NLAB] += 1
    for si, dj in punchfind.match_nearest(det, v.spans, v.tol):
        if rows[dj]["type"] == v.spans[si].label:
            c[v.spans[si].start // v.block, NAMED] += 1
    return c


def paired_named_ci(arm: "dict[str, np.ndarray]", base: "dict[str, np.ndarray]") -> dict:
    """Same bootstrap as fc.paired_ci, on the named-right share."""
    rng = np.random.default_rng(fc.SEED)
    point = (sum(a[:, NAMED].sum() for a in arm.values()) - sum(b[:, NAMED].sum() for b in base.values())) \
        / sum(b[:, NLAB].sum() for b in base.values())
    vals = []
    for _ in range(fc.N_BOOT):
        na = nb = lab = 0
        for v in sorted(arm):
            idx = rng.integers(0, len(arm[v]), size=len(arm[v]))
            na += arm[v][idx, NAMED].sum(); nb += base[v][idx, NAMED].sum(); lab += base[v][idx, NLAB].sum()
        if lab:
            vals.append((na - nb) / lab)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi), "excludes_zero": bool(lo > 0 or hi < 0),
            "n_boot": fc.N_BOOT, "n_undefined_dropped": fc.N_BOOT - len(vals)}


def check_finder_reproduction(ref: np.ndarray) -> dict:
    """The pooled baseline's V5+V9 counts must be RESULTS 5a-5b exactly, or the run stops."""
    jab = CLASSES.index("jab")
    got = {"detections_scored": int(ref[fc.DET]), "labelled": int(ref[fc.LAB]), "hits": int(ref[fc.HIT_S]),
           "jab": (int(ref[fc.HIT_T + jab]), int(ref[fc.LAB_T + jab]))}
    if any(got[k] != REFERENCE[k] for k in got):
        raise SystemExit(f"REPRODUCTION FAILED: the baseline on V5+V9 gives {got}, RESULTS 5a-5b say "
                         f"{ {k: REFERENCE[k] for k in got} }; the arm is not scored")
    print(f"reproduction: the baseline finder on V5+V9 matches RESULTS 5a-5b ({got})")
    return got


def check_named_reproduction(nb: "dict[str, np.ndarray]") -> int:
    """The pooled baseline's found-and-named-right on V5+V9 must be RESULTS 5c exactly."""
    named = int(sum(c[:, NAMED].sum() for c in nb.values()))
    if named != REFERENCE["named_right"]:
        raise SystemExit(f"REPRODUCTION FAILED: the baseline finds and names right {named}, RESULTS 5c says "
                         f"{REFERENCE['named_right']}; the arm is not scored")
    print(f"reproduction: the baseline end to end on V5+V9 matches RESULTS 5c ({named})")
    return named


def mark_started(t: dict, tune_sha256: str) -> None:
    """Create the started marker, atomically and only if absent: a second run, a run with another
    --out, a retune, and a run that crashed half way are all refused from now on."""
    tune.STARTED.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tune.STARTED.open("x", encoding="utf-8") as f:
            json.dump({"tune_file_sha256": tune_sha256, "arm": t["chosen_arm"],
                       "head": tune._git("rev-parse", "HEAD")}, f)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {tune.STARTED} exists; the held-out run has already started") from None


def run_e2e(a, rule_path: Path, out_dir: Path, sha: str, videos: "dict[str, fc.Video]",
            rule: dict) -> "dict[str, list[dict]]":
    """The pipeline end to end on V5+V9 through eval_pipeline_v5v9.main (finder, then the
    classifier), with the FROZEN rule file; the classified rows are read back checked against
    `rule`, the rule validated at the start, not a fresh read of any file."""
    if e2e.main(["--data", str(a.data), "--poses", str(a.poses), "--classifier", str(a.classifier),
                 "--stance", a.stance, "--rule", str(rule_path), "--out-dir", str(out_dir),
                 "--extent", str(a.extent), "--reference", str(a.reference)]):
        raise SystemExit(f"eval_pipeline_v5v9.py failed with {rule_path.name}")
    rows = {}
    for v in E2E_VIDEOS:
        rows[v] = e2e.read_classified(out_dir / "punches" / f"{v}.classified.csv",
                                      out_dir / "punches" / f"{v}.classified.json", v, videos[v].n_frames, videos[v].fps, a.stance,
                                      sha, rule)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--poses", type=Path, required=True)
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--stance", required=True)
    ap.add_argument("--baseline", type=Path, default=find_punches.DEFAULT_RULE)
    ap.add_argument("--tune", type=Path, default=tune.DEFAULT_TUNE)
    ap.add_argument("--rule", type=Path, default=tune.DEFAULT_RULE_OUT)
    ap.add_argument("--out", type=Path, default=tune.DEFAULT_HELDOUT)
    ap.add_argument("--e2e-dir", type=Path, default=Path("results/fast_combinations_e2e"))
    # Passed through to eval_pipeline_v5v9.py, whose own defaults are relative to the working folder:
    ap.add_argument("--extent", type=Path, default=e2e.classify_punches.DEFAULT_EXTENT)
    ap.add_argument("--reference", type=Path, default=e2e.DEFAULT_REFERENCE)
    a = ap.parse_args(argv)

    tune.split_ready()
    tune.refuse_if_started()
    if a.out.exists():
        raise SystemExit(f"REFUSED: {a.out} exists. The held-out run is done once; a second run with other "
                         "settings is not allowed by the gate")
    # Everything this invocation writes before the marker (frozen rule copies, pipeline outputs) goes
    # in a folder created for it alone, so a concurrent invocation can never overwrite what it scores.
    a.e2e_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=a.e2e_dir))
    frozen = run_dir / "frozen"
    t, tune_sha, arm_rule, arm_path = read_tune(a.tune, a.rule, frozen)
    commit = tune.gate_commit()
    if commit != t.get("gate_commit"):
        raise SystemExit(f"REFUSED: the gate changed since tuning ({t.get('gate_commit')} -> {commit})")
    if list(bdata.HELD_OUT_VIDEOS) != t.get("held_out_videos_at_tuning"):
        raise SystemExit("REFUSED: the held-out list differs from the one recorded at tuning")
    base_path = freeze(a.baseline, t.get("baseline_rule_sha256"), frozen / "baseline_rule.json")
    base_rule = tune.baseline_rule(base_path)
    held = sorted(bdata.HELD_OUT_VIDEOS, key=tune.video_order)
    if set(held) & set(t.get("tuned_on", [])):
        raise AssertionError("a held-out video was tuned on")
    missing = [v for v in held if not (a.poses / f"{fc.POSE_FILE.get(v, v)}.pose.npz").is_file()]
    if missing:
        raise SystemExit(f"REFUSED: no pose for held-out video(s) {', '.join(missing)}")

    loaded, checks, excluded = {}, {}, []
    for v in held:
        video, chk = tune.load_video(a.data, a.poses, v)
        checks[v] = chk
        if chk["aligned"]:
            loaded[v] = video
        else:
            excluded.append(v)
    if not set(E2E_VIDEOS) <= set(loaded):
        raise SystemExit("REFUSED: V5 or V9 fails the alignment gate; the reproduction check cannot run")

    # Gates before the arm is touched: the BASELINE reproduces RESULTS 5a-5c on V5+V9, finder and
    # end to end. Only the baseline has run when either fails; the arm is never scored.
    base_c = {v: fc.counts(vd, fc.detections(vd, base_rule)) for v, vd in loaded.items()}
    check_finder_reproduction(fc.total(base_c, E2E_VIDEOS))
    sha = classify.read_manifest(a.classifier)["sha256"]
    rows_b = run_e2e(a, base_path, run_dir / "baseline", sha, loaded, base_rule)
    nb = {v: named_counts(loaded[v], rows_b[v]) for v in E2E_VIDEOS}
    named_b = check_named_reproduction(nb)

    # From here the arm is scored: mark the one held-out run as started, at a path no option moves.
    mark_started(t, tune_sha)
    ctl = t["control"]
    ctl_rule = fc.rule(ctl["threshold"], None)
    arm_c = {v: fc.counts(vd, fc.detections(vd, arm_rule)) for v, vd in loaded.items()}
    ctl_c = {v: fc.counts(vd, fc.detections(vd, ctl_rule)) for v, vd in loaded.items()}
    arm_t, base_t = fc.total(arm_c), fc.total(base_c)
    diffs = {"f1": fc.paired_ci(arm_c, base_c, fc.f1),
             "recall": fc.paired_ci(arm_c, base_c, fc.recall),
             "precision_lower_bound": fc.paired_ci(arm_c, base_c, fc.precision),
             "recall_by_group": {g: fc.paired_ci(arm_c, base_c, lambda x, g=g: fc.group_recall(x, g))
                                 for g in fc.GROUPS if fc.group_recall(base_t, g) is not None}}
    ver = fc.verdict(arm_t, base_t, diffs["f1"])

    # End to end on V5+V9 (the classifier checkpoint's own held-out videos).
    rows_a = run_e2e(a, arm_path, run_dir / "arm", sha, loaded, arm_rule)
    na = {v: named_counts(loaded[v], rows_a[v]) for v in E2E_VIDEOS}
    labels_e2e = int(sum(c[:, NLAB].sum() for c in nb.values()))
    end_to_end = {"videos": list(E2E_VIDEOS), "labelled": labels_e2e,
                  "baseline_named_right": named_b, "arm_named_right": int(sum(c[:, NAMED].sum() for c in na.values())),
                  "difference_share": paired_named_ci(na, nb)}

    out = {"gate": str(tune.GATE), "gate_commit": commit, "head": tune._git("rev-parse", "HEAD"),
           "tune_file_sha256": tune_sha, "run_dir": str(run_dir), "arm": t["chosen_arm"], "arm_rule": arm_rule,
           "baseline_rule": base_rule, "control_rule": ctl_rule, "held_out": held, "videos": checks,
           "excluded": excluded, "reproduction": {"reference": {k: list(v) if isinstance(v, tuple) else v
                                                                for k, v in REFERENCE.items()}, "ok": True},
           "baseline": fc.summary(base_t), "arm_metrics": fc.summary(arm_t), "control": fc.summary(fc.total(ctl_c)),
           "per_video": {v: {"baseline": fc.summary(base_c[v].sum(axis=0)), "arm": fc.summary(arm_c[v].sum(axis=0))}
                         for v in loaded},
           "paired_differences": diffs, "end_to_end": end_to_end, "verdict": ver,
           "precision_note": "a lower bound: labels are incomplete (RESULTS row 5d: 19 of 50 far detections were "
                             "real punches; inside the labelled stretch 19 of 37, outside it 0 of 13)"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, allow_nan=False), encoding="utf-8")
    d = diffs["f1"]
    print(f"F1 baseline {fc.f1(base_t):.3f}, arm {fc.f1(arm_t):.3f}: difference {d['point']:+.3f} "
          f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")
    print(f"VERDICT: {'PASS' if ver['pass'] else 'FAIL'}  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
