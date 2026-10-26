"""Glove-colour boxer identity on participant sparring footage (docs/gates/glove-identity.md).

PARTICIPANT DATA. Every output goes under ~/comps-participant-data/ (deny by default, the face-blur tool's
location check: no git checkout, vault, Syncthing or cloud folder). Only BLURRED copies registered in the
manifest are read. Nothing prints a file name; output is counts and shares.

  roles     --pair BLURRED.mp4 BLURRED.blur.json (x3) --fail-case K --out M.json
            fixes DEV / TEST / FAIL-CASE before any frame is read (K = which pair has same-colour gloves;
            DEV = the other pair with the smaller ORIGINAL sha256, from the blur report)
  compute   --manifest M.json --video BLURRED.mp4 --tracks T.tracks.npz --faceboxes F.faceboxes.npz --out C.npz
            (F: face boxes measured on the ORIGINAL with the blur's detector; reads under a very large
            face mosaic are contested)
  template  --manifest M.json --tracks T.tracks.npz --out L.labels.csv      (Brendan fills it in)
  freeze    --manifest M.json            records the rule digest and commit; TEST / FAIL-CASE need it
  score     --manifest M.json --tracks T --counts C --labels L --out S.json   (TEST / FAIL-CASE once each)
  gate      --manifest M.json --test S.json --fail-case S.json --out G.json

Exit 0 = done (gate: PASS), 1 = refused input, 2 = gate FAILS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from boxingvi import faceblur, gloves, glovescore  # noqa: E402
from boxingvi.pose_yolo import PoseTable  # noqa: E402
from boxingvi.video import fingerprint  # noqa: E402

MANIFEST_FORMAT = "glove-identity-manifest-1"
SCORE_FORMAT = "glove-identity-score-1"
ROLES = ("dev", "test", "fail_case")
MAX_JSON_BYTES = 4 * 1024 * 1024
# Everything the result depends on: the whole library (the rule imports appearance, headgear, linking, video,
# npyread, faceblur, punchfind ...) and this file. The freeze pins all of it, not just the glove files.
RULE_FILES = ("src/boxingvi", "glove_identity.py")
_FP = re.compile(r"[0-9a-f]{64}")


class Refused(Exception):
    pass


def full_digest() -> str:
    both = {"rule": gloves.rule_constants(), "score": glovescore.score_constants()}
    return hashlib.sha256(json.dumps(both, sort_keys=True).encode()).hexdigest()


def code_digest(root: Path = ROOT) -> str:
    """sha256 of EVERY file under src/boxingvi (subfolders too: a package folder there would shadow a module)
    and this one, as they are on disk; only __pycache__ is skipped (it is derived and gitignored)."""
    lib = root / "src" / "boxingvi"
    files = sorted(p for p in lib.rglob("*") if p.is_file() and "__pycache__" not in p.relative_to(lib).parts)
    h = hashlib.sha256()
    for p in files + [root / "glove_identity.py"]:
        h.update(p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def check_imports(modules=None, root: Path = ROOT) -> None:
    """Every loaded boxingvi module must be one of the top-level .py files under src/boxingvi (or its
    __init__): a module loaded from anywhere else (an untracked package folder, another checkout on sys.path)
    is code the digest and the freeze do not describe."""
    lib = (root / "src" / "boxingvi").resolve()
    for name, mod in list((modules if modules is not None else sys.modules).items()):
        if name != "boxingvi" and not name.startswith("boxingvi."):
            continue
        if name == "boxingvi" and getattr(mod, "__file__", None) is None:
            # a namespace package (src/boxingvi has no __init__.py): it merges every `boxingvi` folder on
            # sys.path, so it must have exactly one portion, this library's folder
            portions = {Path(p).resolve() for p in getattr(mod, "__path__", [])}   # one folder twice is fine
            if portions != {lib}:
                raise Refused("the boxingvi package is assembled from more than this library's folder: "
                              "refusing to run")
            continue
        f = getattr(mod, "__file__", None)
        p = Path(f).resolve() if f else None
        if p is None or p.parent != lib or p.suffix != ".py" \
                or p.stem != (name.split(".")[-1] if "." in name else "__init__"):
            raise Refused(f"module {name} was not loaded from the library's own files: refusing to run")


def provenance() -> str:
    """What a counts file must carry to be scored: the rule's constants and the code that computed it."""
    check_imports()
    return hashlib.sha256(f"{full_digest()}|{code_digest()}".encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- files -------------------------------------------------------------------------------
def out_path(path, ctx, suffixes, overwrite=False) -> Path:
    try:
        return faceblur.check_output_location(path, overwrite=overwrite, allowed_root=ctx["root"],
                                              syncthing_configs=ctx["syncthing"], suffixes=suffixes)
    except faceblur.UnsafeOutput as e:      # the reason code only: the message names folders and files
        raise Refused(f"output refused ({e.reason})") from e


def write_atomic(path: Path, write, suffix: str) -> None:
    """Write through a fresh temp file in the same folder, then swap the directory entry: an existing file
    (or a hard link planted at `path`) is replaced, never written through, and a failure leaves nothing."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp" + suffix)
    os.close(fd)
    try:
        write(Path(tmp))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json(path: Path, obj) -> None:
    def w(p):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
    write_atomic(path, w, ".json")


def read_json(path, what: str) -> dict:
    p = Path(path)
    try:
        if p.stat().st_size > MAX_JSON_BYTES:
            raise Refused(f"{what}: more than {MAX_JSON_BYTES} bytes")
        obj = json.loads(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise Refused(f"{what}: cannot be read ({e.strerror})") from e
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise Refused(f"{what}: not valid JSON") from e
    if not isinstance(obj, dict):
        raise Refused(f"{what}: not a JSON object")
    return obj


def load_manifest(path) -> dict:
    m = read_json(path, "manifest")
    if m.get("format") != MANIFEST_FORMAT or set(m) != {"format", "roles", "frozen", "scored"}:
        raise Refused("manifest: not a glove-identity manifest")
    roles = m["roles"]
    if not isinstance(roles, dict) or set(roles) != set(ROLES):
        raise Refused("manifest: roles must be exactly dev, test and fail_case")
    fps = set()
    for r in ROLES:
        v = roles[r]
        if not isinstance(v, dict) or set(v) != {"source_sha256", "blurred_fp"} \
                or not all(isinstance(v[k], str) and _FP.fullmatch(v[k]) for k in v):
            raise Refused(f"manifest: role {r} must hold two sha256 digests")
        fps.add(v["blurred_fp"])
    if len(fps) != 3:
        raise Refused("manifest: the three roles must be three different videos")
    fr = m["frozen"]
    if fr is not None and (not isinstance(fr, dict) or set(fr) != {"rule_digest", "commit"}
                           or not isinstance(fr["rule_digest"], str) or not _FP.fullmatch(fr["rule_digest"])
                           or not isinstance(fr["commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", fr["commit"])):
        raise Refused("manifest: frozen must be null or a rule digest and a commit")
    sc = m["scored"]
    if not isinstance(sc, dict) or not set(sc) <= {"test", "fail_case"} \
            or not all(isinstance(v, str) and _FP.fullmatch(v) for v in sc.values()):
        raise Refused("manifest: scored must map test / fail_case to the score file's sha256")
    return m


def role_of(manifest: dict, table) -> str:
    if not table.video_fp:
        raise Refused("the pose table carries no video fingerprint: re-run run_ingest.py --track")
    for r in ROLES:
        if manifest["roles"][r]["blurred_fp"] == table.video_fp:
            return r
    raise Refused("this pose table is not of a video registered in the manifest (only blurred copies are read)")


def load_table(path) -> PoseTable:
    try:
        return PoseTable.load(path)
    except (ValueError, OSError, KeyError) as e:
        raise Refused(f"pose table refused: {e}") from e


def git_head() -> "tuple[str, bool]":
    """(commit, rule files clean) of this checkout; argument arrays, no shell."""
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all", "--",
                            *RULE_FILES], capture_output=True, text=True)
    if head.returncode != 0 or dirty.returncode != 0:
        raise Refused("not a git checkout: the freeze needs a commit to record")
    return head.stdout.strip(), dirty.stdout.strip() == ""


def git_same(commit: str, root: Path = ROOT) -> bool:
    """True when the library and CLI on disk are exactly `commit`'s: no tracked change (a logic change that
    leaves every constant alone still changes the rule) and no UNTRACKED file (an untracked package folder
    shadows a tracked module, and `git diff` never sees it)."""
    r = subprocess.run(["git", "-C", str(root), "diff", "--quiet", commit, "--", *RULE_FILES],
                       capture_output=True, text=True)
    s = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--",
                        *RULE_FILES], capture_output=True, text=True)
    return r.returncode == 0 and s.returncode == 0 and s.stdout.strip() == ""


# ---- commands -------------------------------------------------------------------------------
def cmd_roles(a, ctx) -> int:
    out = out_path(a.out, ctx, (".json",))
    if len(a.pair) != 3:
        raise Refused("give exactly three --pair BLURRED.mp4 REPORT.blur.json")
    if a.fail_case not in (1, 2, 3):
        raise Refused("--fail-case must be 1, 2 or 3 (the pair whose boxers wear same-colour gloves)")
    entries = []
    for i, (vid, rep) in enumerate(a.pair, 1):
        r = read_json(rep, f"blur report {i}")
        src, outsha = r.get("source_sha256"), r.get("output_sha256")
        if not (isinstance(src, str) and _FP.fullmatch(src) and isinstance(outsha, str) and _FP.fullmatch(outsha)):
            raise Refused(f"blur report {i}: no source / output sha256")
        if sha256_file(Path(vid)) != outsha:
            raise Refused(f"pair {i}: the video is not the output its blur report describes")
        entries.append({"source_sha256": src, "blurred_fp": fingerprint(vid)})
    if len({e["source_sha256"] for e in entries}) != 3:
        raise Refused("the three reports must come from three different original videos")
    fail = entries[a.fail_case - 1]
    rest = sorted((e for i, e in enumerate(entries, 1) if i != a.fail_case), key=lambda e: e["source_sha256"])
    write_json(out, {"format": MANIFEST_FORMAT, "roles": {"dev": rest[0], "test": rest[1], "fail_case": fail},
                     "frozen": None, "scored": {}})
    print(f"roles fixed: dev = pair {entries.index(rest[0]) + 1}, test = pair {entries.index(rest[1]) + 1}, "
          f"fail_case = pair {a.fail_case}")
    return 0


def cmd_compute(a, ctx) -> int:
    m = load_manifest(a.manifest)
    out = out_path(a.out, ctx, (".npz",), a.overwrite)
    table = load_table(a.tracks)
    role = role_of(m, table)
    if fingerprint(a.video) != table.video_fp:
        raise Refused("the video is not the one this pose table was extracted from")
    stats: dict = {}
    try:
        masks = gloves.FaceMasks.load(a.faceboxes)
        if masks.source_sha256 != m["roles"][role]["source_sha256"]:
            raise Refused("the face boxes were measured on another original than this video's")
        counts = gloves.compute_gloves(a.video, table, provenance=provenance(), masks=masks, stats=stats)
    except ValueError as e:
        raise Refused(str(e)) from e
    write_atomic(out, counts.save, ".npz")
    print(f"glove reads: {len(counts.rows)} ({int(counts.contested.sum())} contested); under a very large "
          f"face mosaic (contested): {stats.get('under_large_square', 0)}; under any face mosaic (counted, "
          f"not excluded): {stats.get('under_any_face_mosaic', 0)}")
    return 0


def cmd_template(a, ctx) -> int:
    m = load_manifest(a.manifest)
    out = out_path(a.out, ctx, (".csv",))
    table = load_table(a.tracks)
    role = role_of(m, table)
    try:
        write_atomic(out, lambda p: glovescore.write_template(table, p), ".csv")
        n = len(glovescore.label_times(table))
    except ValueError as e:
        raise Refused(str(e)) from e
    print(f"role: {role}; label rows: {n} (every {glovescore.LABEL_EVERY_S} s)")
    return 0


def cmd_freeze(a, ctx) -> int:
    mp = out_path(a.manifest, ctx, (".json",), overwrite=True)
    commit, clean = ctx["git"]()
    if not clean:
        raise Refused("commit the library and CLI first (no untracked files either): the freeze records a "
                      "commit that must hold this code")
    with manifest_lock(mp, ctx.get("lock_timeout_s", 30.0)):
        m = load_manifest(mp)
        if m["frozen"] is not None:
            raise Refused("already frozen: the rule is not changed after a freeze (the gate)")
        m["frozen"] = {"rule_digest": full_digest(), "commit": commit}
        write_json(mp, m)
    print(f"frozen at {commit[:12]}, rule digest {m['frozen']['rule_digest'][:12]}")
    return 0


class manifest_lock:
    """Serialise read-modify-write of the manifest (TEST and FAIL-CASE finishing at once must not overwrite
    each other's record): an exclusive-create lock file, waited for up to `timeout_s`, removed on exit."""

    def __init__(self, mp: Path, timeout_s: float = 30.0):
        self.path, self.timeout_s = mp.with_name(f"{mp.stem}.lock"), timeout_s

    def __enter__(self):
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return self
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise Refused("the manifest is locked by another run (or a stale lock file is left: "
                                  "remove it only if no run is going)") from None
                time.sleep(0.05)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
        return False


def claim_path(mp: Path, role: str) -> Path:
    return mp.with_name(f"{mp.stem}.{role}.claim")


def claim(mp: Path, role: str) -> None:
    """Take the role's one run, atomically (exclusive create), BEFORE its labels are read. A run that dies
    after this has used the role up: rerunning needs the owner to delete the claim deliberately and say why."""
    try:
        fd = os.open(claim_path(mp, role), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as e:
        raise Refused(f"{role} has already been claimed by a run (finished, running or interrupted): "
                      "it is scored once (the gate)") from e
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"claimed {datetime.now(timezone.utc).isoformat(timespec='seconds')} pid {os.getpid()}\n")


def cmd_score(a, ctx) -> int:
    mp = Path(a.manifest)
    m = load_manifest(mp)
    table = load_table(a.tracks)
    role = role_of(m, table)
    if role != "dev":
        if m["frozen"] is None:
            raise Refused(f"{role} is read only after `freeze` (the gate)")
        if m["frozen"]["rule_digest"] != full_digest() or not ctx["git_same"](m["frozen"]["commit"]):
            raise Refused("the rule changed since the freeze: TEST / FAIL-CASE are never scored by another rule")
        if role in m["scored"]:
            raise Refused(f"{role} was already scored once (the gate)")
        mp = out_path(mp, ctx, (".json",), overwrite=True)
    out = out_path(a.out, ctx, (".json",), overwrite=a.overwrite and role == "dev")
    try:
        counts = gloves.GloveCounts.load(a.counts)
        if counts.provenance != provenance():
            raise Refused("the glove counts were computed by another rule or another version of the code: "
                          "recompute them with this checkout")
        if not counts.mask_sha256:
            raise Refused("the glove counts were computed without the face-mosaic mask (`compute --faceboxes`)")
        counts.check_table(table)
        if role != "dev":
            claim(mp, role)
        labels = glovescore.parse_labels(a.labels, table)
        ident = gloves.identify(table, counts)
    except ValueError as e:
        raise Refused(str(e)) from e
    ms = glovescore.moments(table, labels)
    methods = {"glove": glovescore.score_method(ms, glovescore.glove_names(ident)),
               "bytetrack": glovescore.score_method(ms, glovescore.bytetrack_names(table))}
    try:
        methods["linker"] = glovescore.score_method(ms, glovescore.linker_names(table))
    except ValueError as e:     # the linker refuses (no usable fragment, MILP not proven optimal): recorded, and
        methods["linker"] = {"error": str(e), "correct_share": None}   # bar 3 is then not met
    res = {"format": SCORE_FORMAT, "role": role, "design_data_not_evidence": role == "dev",
           "rule_digest": full_digest(), "frozen": m["frozen"], "label_minutes": labels.minutes,
           "moments": glovescore.status_counts(ms), "glove_verdict": ident.verdict,
           "glove_diagnostics": ident.diagnostics, "methods": methods}
    write_json(out, res)
    if role != "dev":
        with manifest_lock(mp, ctx.get("lock_timeout_s", 30.0)):
            m = load_manifest(mp)             # re-read under the lock: record onto the manifest as it is now
            m["scored"][role] = sha256_file(out)
            write_json(mp, m)
    g = methods["glove"]
    print(f"{role}: verdict {ident.verdict}; moments {res['moments']}; glove decided {g['decided']} of "
          f"{g['scored']}, correct {g['correct']}, swaps {g['swaps']}")
    for b in ("bytetrack", "linker"):
        x = methods[b]
        if "error" in x:
            print(f"  baseline {b}: FAILED to run ({x['error']}); bar 3 cannot be met")
        else:
            print(f"  baseline {b}: decided {x['decided']} of {x['scored']}, correct {x['correct']}")
    return 0


def _count(d, k) -> int:
    v = d.get(k)
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise Refused(f"score: {k} must be a whole number >= 0")
    return v


def _score_file(path, role, m) -> dict:
    s = read_json(path, f"{role} score")
    if s.get("format") != SCORE_FORMAT or s.get("role") != role or s.get("design_data_not_evidence") is not False:
        raise Refused(f"{role} score: not a {role} score file")
    if m["frozen"] is None or s.get("frozen") != m["frozen"] or s.get("rule_digest") != m["frozen"]["rule_digest"]:
        raise Refused(f"{role} score: not made by the frozen rule")
    if not claim_path(Path(m["_path"]), role).exists():
        raise Refused(f"{role} score: no claim was taken for it")
    if m["scored"].get(role) != sha256_file(Path(path)):
        raise Refused(f"{role} score: not the file the frozen run recorded")
    meth = s.get("methods")
    if not isinstance(meth, dict) or set(meth) != {"glove", "bytetrack", "linker"} \
            or not all(isinstance(meth[k], dict) for k in meth):
        raise Refused(f"{role} score: methods must be glove, bytetrack and linker")
    g = meth["glove"]
    sc, dec, cor, wr = (_count(g, k) for k in ("scored", "decided", "correct", "wrong"))
    if not (cor + wr == dec <= sc):
        raise Refused(f"{role} score: glove counts are inconsistent")
    spr = g.get("swaps_per_round")
    if spr is not None and (isinstance(spr, bool) or not isinstance(spr, (int, float)) or not spr >= 0):
        raise Refused(f"{role} score: swaps_per_round must be a number >= 0 or null")
    for b in ("bytetrack", "linker"):
        cs = meth[b].get("correct_share")
        if cs is not None and (isinstance(cs, bool) or not isinstance(cs, (int, float)) or not 0 <= cs <= 1):
            raise Refused(f"{role} score: {b} correct_share must be in [0, 1] or null")
    return s


def cmd_gate(a, ctx) -> int:
    m = load_manifest(a.manifest)
    m["_path"] = str(Path(a.manifest))
    out = out_path(a.out, ctx, (".json",), a.overwrite)
    v = glovescore.gate_verdict(_score_file(a.test, "test", m), _score_file(a.fail_case, "fail_case", m))
    write_json(out, v)
    for k, val in v.items():
        print(f"  {k}: {val}")
    return 0 if v["PASS"] else 2

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("roles")
    p.add_argument("--pair", nargs=2, action="append", type=Path, metavar=("BLURRED", "REPORT"), required=True)
    p.add_argument("--fail-case", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("compute")
    for k in ("--manifest", "--video", "--tracks", "--faceboxes", "--out"):
        p.add_argument(k, type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("template")
    for k in ("--manifest", "--tracks", "--out"):
        p.add_argument(k, type=Path, required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--manifest", type=Path, required=True)
    p = sub.add_parser("score")
    for k in ("--manifest", "--tracks", "--counts", "--labels", "--out"):
        p.add_argument(k, type=Path, required=True)
    p.add_argument("--overwrite", action="store_true", help="DEV only")
    p = sub.add_parser("gate")
    for k in ("--manifest", "--test", "--fail-case", "--out"):
        p.add_argument(k, type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return ap


def main(argv=None, participant_root=None, syncthing_configs=None, git=None, same=None,
         lock_timeout_s: float = 30.0) -> int:
    a = build_parser().parse_args(argv)
    ctx = {"root": participant_root or faceblur.PARTICIPANT_ROOT, "syncthing": syncthing_configs,
           "git": git or git_head, "git_same": same or git_same, "lock_timeout_s": lock_timeout_s}
    try:
        return {"roles": cmd_roles, "compute": cmd_compute, "template": cmd_template, "freeze": cmd_freeze,
                "score": cmd_score, "gate": cmd_gate}[a.cmd](a, ctx)
    except Refused as e:
        print(scrub(f"refused: {e}", argv), file=sys.stderr)
        return 1
    except OSError as e:          # never a traceback: it would print a path, and a path may carry a name
        print(scrub(f"refused: a file could not be read or written ({e.strerror or type(e).__name__})", argv),
              file=sys.stderr)
        return 1


def scrub(msg: str, argv) -> str:
    """Remove every path given on the command line (as typed, resolved, and its file name) from a message:
    a file name may carry a person's name, and loaders name the file they refuse."""
    names = set()
    for tok in list(argv if argv is not None else sys.argv[1:])[1:]:     # [0] is the subcommand
        tok = str(tok)
        if tok.startswith("-"):
            continue
        p = Path(tok)
        names.update({tok, str(p), str(p.resolve()), p.name})
    for n in sorted((n for n in names if len(n) >= 3), key=len, reverse=True):
        msg = msg.replace(n, "<file>")
    return msg


if __name__ == "__main__":
    sys.exit(main())
