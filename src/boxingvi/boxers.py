"""Two boxers, one report each: the identity input, punches per boxer, and the checks that bind them.

The report must not depend on WHICH identity method wins (glove colour, headgear colour, the pose-only
linker, a hand-made ground truth), so identity comes in as a file, in one format every method can write:

IDENTITY FILE (vault ADR-0016): `<clip>.identity.csv`, one row per tracked-person SEGMENT,
    track_id, first_frame, last_frame, boxer        (boxer: A, B or unsure)
meaning "every row of the tracks file with this track id and a frame in [first_frame, last_frame] is boxer A
(B; a boxer, but which one the method cannot say)". Rows of the tracks file that no segment covers are nobody
the method calls a boxer (a referee, a coach) and are never scanned. Beside it, `<clip>.identity.json`, the
provenance sidecar: the CSV's sha256, the method's name, its GATE verdict (pass / fail / ungated) and the tracks
file it was made from (file name, sha256, source video, frame count, fps, video fingerprint).

Why segments: every method this project has produced reduces to them without loss. The linker names a track id
per boxer per frame (linking.Linking.membership); headgear and kit colour decide per fragment (a track id); glove
colour names each pose-table row (`glovescore.glove_names`), and a run of rows of one id with one name is a
segment. `segments_from_row_names` is the one converter, from the per-row name function the glove gate already
scores. A per-punch identity would make the identity depend on the finder's output; per row is lossless but
unreadable by a person. And a segment can never span a fragment handover, which is the finder's own rule.

Deny by default. An identity is refused unless its CSV is the one the sidecar hashed, the sidecar is exactly the
known keys, the tracks file is the very one it names (sha256, frame count, fps, source, fingerprint), every
segment names a person the tracks file has in those frames, no id is in two segments at once, neither boxer is
two people in one frame, and both A and B are somebody. A recorded gate PASS is refused unless the gate's result
file is given, hashes to what the sidecar recorded and says PASS, and the gate's pre-registered document is in
this repo's docs/gates/; a FAIL whose result file says PASS is refused too. "Ungated" carries no gate fields.

PUNCHES PER BOXER: the finder (completeness._scan_segments, the one scan implementation) on each boxer's rows,
split into segments wherever the track id changes or a hole exceeds the finder's bridge, exactly as
completeness.boxer_segments splits a linked boxer: nothing is differenced across a handover. Each punch is then
named from the very segment track and signal it was found on, under one cut per boxer pooled from that boxer's
own punches (classify.cut_from_widths), with that boxer's own stance. Unsure segments are scanned per track id and
their punches are counted, by wrist, and never named or assigned.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import classify, completeness, faceblur, punchfind, report
from .data import CLASSES
from .pose_yolo import PoseTable

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "boxingvi-identity-1"
BOXERS_FORMAT = "boxingvi-boxers-1"
IDENTITY_HEADER = ["track_id", "first_frame", "last_frame", "boxer"]
PUNCH_HEADER = ["frame", "time_s", "hand", "peak_speed_bl_per_s", "peak_on_interpolated_frame"]
BOXERS_HEADER = ["boxer", "track_id"] + report.CLASSIFIED_HEADER
UNSURE_HEADER = ["track_id"] + PUNCH_HEADER
BOXERS = ("A", "B")
UNSURE = "unsure"
NAMES = BOXERS + (UNSURE,)
VERDICTS = ("pass", "fail", "ungated")
MAX_CSV_BYTES = 32 * 1024 * 1024
MAX_SIDECAR_BYTES = 64 * 1024
MAX_SEGMENTS = 500_000
MAX_ROWS = 100_000
_ID = re.compile(r"[0-9]{1,12}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
# A method name is printed in the summary, whose numbers are all checked: no digits, so it adds none.
_METHOD = re.compile(r"[a-z][a-z-]{0,39}")
_GATE_DOC = re.compile(r"docs/gates/[a-z0-9-]{1,60}\.md")
_SIDECAR_KEYS = {"format", "csv_sha256", "method", "gate", "tracks"}
_TRACKS_KEYS = {"file", "sha256", "source", "n_frames", "fps", "video_fp"}
_BOXERS_KEYS = {"format", "clip", "csv_sha256", "unsure_csv_sha256", "classifier_sha256", "stances", "rule",
                "extent", "cuts", "pose_frames", "fps", "tracks", "identity", "scan"}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clip_of(path) -> str:
    """The clip a file belongs to: its name before the first dot (every stage names outputs <clip>.<what>)."""
    return Path(path).name.split(".")[0]


def _read_capped(path, cap: int, what: str) -> bytes:
    p = Path(path)
    try:
        with p.open("rb") as f:
            raw = f.read(cap + 1)
    except OSError as e:
        raise ValueError(f"{what} {p.name}: cannot be read ({e.strerror or type(e).__name__})") from None
    if len(raw) > cap:
        raise ValueError(f"{what} {p.name}: more than {cap} bytes")
    return raw


def _json_object(raw: bytes, name: str) -> dict:
    def no_constant(c):
        raise ValueError(f"{name} contains {c}, which is not valid JSON")
    try:
        d = json.loads(raw.decode("utf-8"), parse_constant=no_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ValueError(f"{name}: not valid JSON") from None
    if not isinstance(d, dict):
        raise ValueError(f"{name}: expected a JSON object")
    return d


def _csv_rows(raw: bytes, name: str):
    """A CSV file's rows, as a list. `csv.Error` (a NUL byte, a field over the csv module's size limit) and
    non-UTF-8 bytes become the ValueError every other refusal is, never a traceback."""
    try:
        return list(csv.reader(io.StringIO(raw.decode("utf-8"), newline="")))
    except (UnicodeDecodeError, csv.Error) as e:
        raise ValueError(f"{name}: not a readable UTF-8 CSV ({type(e).__name__})") from None


def replace_file(path, write) -> None:
    """Write an output without following anything already at `path`: `write(tmp)` fills a NEW file created
    exclusively in the same folder, which then replaces `path` (os.replace swaps the directory entry: a link
    planted at `path` is replaced, never written through; Astra, review 1). The temporary file is removed on any
    failure."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        write(Path(tmp))
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def replace_bytes(path, data: bytes) -> None:
    replace_file(path, lambda p: p.write_bytes(data))


# Participant footage lives here only (encrypted desktop; vault ISSUES-LOG O63-O65). Both paths the project uses.
PARTICIPANT_ROOTS = (Path.home() / "comps-participant-data", Path.home() / "repos" / "comps-participant-data")


def check_output_location(inputs, out_dir, roots=PARTICIPANT_ROOTS) -> None:
    """Deny by default: when any input lies under a participant root, the output folder must lie under that same
    root, so nothing derived from participant footage is written anywhere else (results/, the vault, a sync
    folder). Paths are resolved first, so a relative path, `..` or a link cannot step around the check."""
    out = Path(out_dir).resolve()
    for p in inputs:
        if p is None:
            continue
        rp = Path(p).resolve()
        for root in roots:
            r = Path(root).resolve()
            if faceblur._under(rp, r) and not faceblur._under(out, r):
                raise ValueError(f"{Path(p).name} is participant data: its outputs must stay under the participant "
                                 "folder, and --out-dir is outside it")


# ------------------------------------------------------------------ binding to the tracks file ---

def tracks_binding(table: PoseTable, tracks_path) -> dict:
    """What an identity (and the punches made with it) records about the tracks file it was made from."""
    return {"file": Path(tracks_path).name, "sha256": sha256_file(tracks_path), "source": table.source,
            "n_frames": int(table.n_frames), "fps": float(table.fps), "video_fp": table.video_fp}


def check_binding(recorded, table: PoseTable, tracks_path, what: str) -> None:
    """`recorded` names exactly this tracks file: every field, compared exactly. Frame count and fps are
    reported first (a clear reason); the sha256 then catches any other difference in content."""
    if not isinstance(recorded, dict) or set(recorded) != _TRACKS_KEYS:
        raise ValueError(f"{what}: the tracks binding must have exactly the keys {sorted(_TRACKS_KEYS)}")
    want = tracks_binding(table, tracks_path)
    n, fps = recorded["n_frames"], recorded["fps"]
    if isinstance(n, bool) or not isinstance(n, int) or n != want["n_frames"]:
        raise ValueError(f"{what} was made from a tracks file of {str(n)[:20]} frames; "
                         f"{want['file']} has {want['n_frames']}")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps != want["fps"]:
        raise ValueError(f"{what} was made from a tracks file at {str(fps)[:20]} fps; {want['file']} is at "
                         f"{want['fps']!r}")
    if recorded["source"] != want["source"] or recorded["video_fp"] != want["video_fp"]:
        raise ValueError(f"{what} was made from the tracks of another video than {want['file']}")
    if recorded["file"] != want["file"]:
        raise ValueError(f"{what} was made from another tracks file than {want['file']}")
    if recorded["sha256"] != want["sha256"]:
        raise ValueError(f"{what} was made from a tracks file with other content than {want['file']} "
                         "(re-tracked or edited since)")


# ------------------------------------------------------------------------- the identity file ---

@dataclass(frozen=True)
class Segment:
    track_id: int
    first: int
    last: int
    boxer: str      # A / B / unsure


@dataclass(frozen=True)
class Identity:
    method: str
    gate: dict              # {"verdict"} or {"verdict", "document", "result_sha256"}
    segments: tuple
    csv_sha256: str
    sidecar_sha256: str
    measured: "dict | None" = None   # the method's measured identity error, from its checked gate result only

    @property
    def verdict(self) -> str:
        return self.gate["verdict"]


def _rows_by_track(table: PoseTable) -> "dict[int, np.ndarray]":
    """Row indices of each track id, in frame order."""
    order = np.lexsort((table.frame, table.track_id))
    tids = table.track_id[order]
    cut = np.flatnonzero(np.diff(tids)) + 1
    return {int(tids[a]): order[a:b] for a, b in zip(np.r_[0, cut], np.r_[cut, len(order)]) if b > a}


def segment_rows(table: PoseTable, seg: Segment, by_track: "dict | None" = None) -> np.ndarray:
    """The rows a segment names, in frame order."""
    rows = (by_track or _rows_by_track(table)).get(seg.track_id)
    if rows is None:
        return np.zeros(0, dtype=np.int64)
    fr = table.frame[rows]
    lo, hi = np.searchsorted(fr, seg.first, "left"), np.searchsorted(fr, seg.last, "right")
    return rows[lo:hi]


def check_segments(table: PoseTable, segments) -> None:
    """Every rule an identity's segments must keep against THIS tracks table (module docstring)."""
    if len(segments) > MAX_SEGMENTS:
        raise ValueError(f"more than {MAX_SEGMENTS} segments")
    by_track = _rows_by_track(table)
    seen: "dict[str, list]" = {b: [] for b in NAMES}
    for s in segments:
        if s.boxer not in NAMES:
            raise ValueError(f"boxer must be one of {NAMES}")
        if not 0 <= s.first <= s.last < table.n_frames:
            raise ValueError(f"a segment's frames {s.first}-{s.last} are not inside this "
                             f"{table.n_frames}-frame video")
        rows = segment_rows(table, s, by_track)
        if not len(rows):
            raise ValueError(f"a segment names track {s.track_id} at frames {s.first}-{s.last}, where the tracks "
                             "file has no such person: an identity made from another tracks file?")
        seen[s.boxer].append(rows)
    order = sorted(segments, key=lambda s: (s.track_id, s.first))
    for p, q in zip(order, order[1:]):
        if p.track_id == q.track_id and q.first <= p.last:
            raise ValueError(f"track {q.track_id} is in two segments at frame {q.first}: one person, two names")
    for b in BOXERS:
        if not seen[b]:
            raise ValueError(f"the identity names nobody as boxer {b}: there is nothing to split between two "
                             "boxers" + (" (every segment is unsure)" if not seen["A"] and not seen["B"] else ""))
        fr = table.frame[np.concatenate(seen[b])]
        if len(np.unique(fr)) != len(fr):
            raise ValueError(f"boxer {b} is two people in one frame: the identity contradicts itself")


# The identity methods with a PRE-REGISTERED gate (deny by default: any other method can only be "ungated"), the
# gate document committed before its rule, and the bars its result file must carry. Glove colour's result is what
# `glove_identity.py gate` writes (glovescore.gate_verdict): four bars and PASS = all four. Headgear colour (#29)
# failed its gate and has no result format registered, so it can record "fail" without a result, never "pass".
GATES = {"glove-colour": {"document": "docs/gates/glove-identity.md",
                          "bars": ("1_accuracy", "2_unsure", "3_better_than_tracking", "4_fails_safe")},
         "headgear-colour": {"document": "docs/gates/appearance-identity.md", "bars": None},
         # The pose-only linker (#11): Brendan's hand check of 100 sampled joins on Olympic Boxing (ADR-0018). Its
         # result is `linker_join_check.py score`'s result.json, re-derived from its counts on every read.
         # `manifest` None = joincheck.HELD_OUT_MANIFEST, the only videos a counted sample may come from.
         "linker": {"document": "docs/gates/linker-join-check.md", "bars": None, "result": "linker-join",
                    "manifest": None}}
MEASURED_VERDICTS = ("measured, acceptable", "measured, unreliable", "measured, inconclusive", "not established")
JOIN_RESULT_BASE = {"answers", "unsure_share", "decided", "p_star", "minutes", "join_error", "wilson95", "verdict",
                    "sample_sha256", "gate"}        # joincheck.score with nothing decided writes only these
JOIN_RESULT_KEYS = JOIN_RESULT_BASE | {"misattribution", "by_stratum", "by_camera", "wrong_joins_per_chain_minute",
                                       "bootstrap"}
MEASURED_KEYS = {"verdict", "decided", "different", "join_error", "lower", "upper", "footage"}


def _whole(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _real(x) -> bool:
    """A JSON float (joincheck.score writes floats): no bool, no int (a huge one would overflow a comparison)."""
    return isinstance(x, float) and math.isfinite(x)


def _check_join_result(raw: bytes, name: str, result_path) -> "tuple[bool, dict | None]":
    """The linker join check's result.json, accepted only as a faithful score of the gate's own sample and of
    the answers given to it. Nothing is taken from the result's numbers: beside it must lie the sample.json it
    names (sha256), drawn from the gate's 23 held-out videos (joincheck.HELD_OUT_MANIFEST, so a sample of made-up
    videos cannot be counted), and the complete answers.csv pinned to that sample. S / D / U are counted from the
    answers; the rate, Wilson interval, p* (from the sample's chains) and three-way verdict are recomputed; the
    result must state exactly those. Returns (PASS, the measured numbers a report may quote), or (False, None)
    when nothing was decided."""
    from . import joincheck as jc
    d = _json_object(raw, name)
    if set(d) not in (JOIN_RESULT_KEYS, JOIN_RESULT_BASE) or d["gate"] != GATES["linker"]["document"]:
        raise ValueError(f"{name}: not the linker join check's result")
    folder = Path(result_path).parent
    sp, ap_ = folder / "sample.json", folder / "answers.csv"
    if not sp.is_file() or not ap_.is_file():
        raise ValueError(f"{name}: the sample.json and answers.csv it was scored from must lie beside it")
    sample_raw = _read_capped(sp, jc.MAX_SAMPLE_BYTES, "join-check sample")
    if d["sample_sha256"] != hashlib.sha256(sample_raw).hexdigest():
        raise ValueError(f"{name}: the sample.json beside it is not the sample it was scored on")
    _json_object(sample_raw, "sample.json")          # refuses NaN / Infinity before the sample parser sees it
    try:
        sample = jc.parse_sample(sample_raw, GATES["linker"]["manifest"] or jc.HELD_OUT_MANIFEST)
        answers = jc.AnswerFile.load(ap_, sample)
    except ValueError as e:
        raise ValueError(f"{name}: its sample or answers are not the gate's ({e})") from None
    c = answers.counts()
    if c["empty"]:
        raise ValueError(f"{name}: the answers beside it are not complete")
    s, dd, u, total = c["S"], c["D"], c["U"], sample.n
    decided = s + dd
    a = d["answers"]
    if not isinstance(a, dict) or a != {"S": s, "D": dd, "U": u, "total": total} or not all(
            _whole(v) for v in a.values()) or not _whole(d["decided"]) or d["decided"] != decided:
        raise ValueError(f"{name}: its answer counts are not the answers beside it")
    ps = sample.raw["frame"]["p_star"]
    if not _real(d["p_star"]) or d["p_star"] != ps:
        raise ValueError(f"{name}: p* is not the one its sample gives")
    if (decided == 0) != (set(d) == JOIN_RESULT_BASE):
        raise ValueError(f"{name}: not the linker join check's result (its sections do not match its counts)")
    if decided == 0:
        if d["verdict"] != "not established" or d["join_error"] is not None or d["wilson95"] is not None:
            raise ValueError(f"{name}: with nothing decided the verdict can only be 'not established'")
        return False, None
    rate = dd / decided
    lo, hi = jc.wilson(dd, decided)
    w = d["wilson95"]
    if (not _real(d["join_error"]) or abs(d["join_error"] - rate) > 1e-12 or not isinstance(w, list)
            or len(w) != 2 or not all(_real(x) for x in w) or abs(w[0] - lo) > 1e-12 or abs(w[1] - hi) > 1e-12):
        raise ValueError(f"{name}: the join error or its interval is not what the answers give")
    verdict = jc.verdict(decided, u, total, lo, hi, ps)
    if d["verdict"] != verdict:
        raise ValueError(f"{name}: the verdict is not what the answers and p* give")
    return verdict == "measured, acceptable", {"verdict": verdict, "decided": decided, "different": dd,
                                               "join_error": rate, "lower": lo, "upper": hi,
                                               "footage": "Olympic Boxing"}


def read_join_result(path) -> "tuple[str, dict, str]":
    """For a method writing a linker identity: (the sidecar verdict, "pass" or "fail"; the measured numbers;
    the result file's sha256). Checked exactly as read_identity will check it."""
    raw = _read_capped(path, MAX_SIDECAR_BYTES, "gate result")
    passed, measured = _check_join_result(raw, Path(path).name, path)
    if measured is None:
        raise ValueError(f"{Path(path).name}: nothing was decided, so there is no measured error to record")
    return ("pass" if passed else "fail"), measured, hashlib.sha256(raw).hexdigest()


def _check_result(raw: bytes, name: str, bars) -> bool:
    """A gate result file's verdict, accepted only in the registered format: every bar a true / false, and PASS
    exactly all of them. A file that only says PASS is not evidence of anything (Astra, review 1)."""
    d = _json_object(raw, name)
    if bars is None:
        raise ValueError(f"{name}: no result format is registered for this gate, so its result cannot be checked")
    vals = [d.get(b) for b in bars]
    passed = d.get("PASS")
    if not all(isinstance(v, bool) for v in vals) or not isinstance(passed, bool):
        raise ValueError(f"{name}: not this gate's result (every bar and PASS must be true or false)")
    if passed != all(vals):
        raise ValueError(f"{name}: PASS is not what its bars say")
    return passed


def _check_gate(gate, gate_result, method: str) -> dict:
    """The sidecar's gate block, and the gate result file when one is given (module docstring)."""
    return _gate_and_measure(gate, gate_result, method)[0]


def _gate_and_measure(gate, gate_result, method: str) -> "tuple[dict, dict | None]":
    """_check_gate, plus the measured numbers when the method's result format carries them (the linker's), taken
    only from the checked result file: nothing typed into a sidecar can put a number in a report."""
    if not isinstance(gate, dict) or not isinstance(gate.get("verdict"), str) or gate["verdict"] not in VERDICTS:
        raise ValueError(f"the gate verdict must be one of {VERDICTS}")
    v = gate["verdict"]
    if v == "ungated":
        if set(gate) != {"verdict"}:
            raise ValueError("an ungated identity carries no gate document or result")
        if gate_result is not None:
            raise ValueError("a gate result was given for an identity that records no gate")
        return {"verdict": v}, None
    if set(gate) != {"verdict", "document", "result_sha256"}:
        raise ValueError("a gated identity records exactly verdict, document and result_sha256")
    doc, sha = gate["document"], gate["result_sha256"]
    if not isinstance(doc, str) or not _GATE_DOC.fullmatch(doc):
        raise ValueError("the gate document must be docs/gates/<name>.md")
    if method not in GATES:
        raise ValueError(f"no gate is registered for the method {method}: record it as ungated")
    if doc != GATES[method]["document"]:
        raise ValueError(f"{doc} is not the gate pre-registered for {method}")
    path = (ROOT / doc).resolve()
    if not path.is_relative_to((ROOT / "docs" / "gates").resolve()) or not path.is_file():
        raise ValueError(f"no pre-registered gate {doc} in this repository")
    if not isinstance(sha, str) or not _HEX64.fullmatch(sha):
        raise ValueError("the gate result_sha256 must be a sha256 hex digest")
    if gate_result is None:
        if v == "pass":
            raise ValueError("the identity records a gate PASS: give its gate result file (--gate-result) so the "
                             "pass can be checked, or record the identity as ungated")
        return {"verdict": v, "document": doc, "result_sha256": sha}, None
    raw = _read_capped(gate_result, MAX_SIDECAR_BYTES, "gate result")
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"{Path(gate_result).name} is not the gate result the identity recorded")
    measured = None
    if GATES[method].get("result") == "linker-join":
        passed, measured = _check_join_result(raw, Path(gate_result).name, gate_result)
    else:
        passed = _check_result(raw, Path(gate_result).name, GATES[method]["bars"])
    if passed != (v == "pass"):
        raise ValueError(f"the identity records a gate {v.upper()}, its gate result says "
                         f"{'PASS' if passed else 'FAIL'}")
    return {"verdict": v, "document": doc, "result_sha256": sha}, measured


def sidecar_path(csv_path) -> Path:
    return Path(csv_path).with_suffix(".json")


def _parse_identity_csv(raw: bytes, name: str) -> "list[Segment]":
    r = iter(_csv_rows(raw, name))
    if next(r, None) != IDENTITY_HEADER:
        raise ValueError(f"{name}: not an identity CSV (header {IDENTITY_HEADER})")
    out = []
    for i, row in enumerate(r, start=2):
        if len(out) >= MAX_SEGMENTS:
            raise ValueError(f"{name}: more than {MAX_SEGMENTS} segments")
        if len(row) != len(IDENTITY_HEADER):
            raise ValueError(f"{name} line {i}: {len(row)} fields, expected {len(IDENTITY_HEADER)}")
        if not all(_ID.fullmatch(v) for v in row[:3]):
            raise ValueError(f"{name} line {i}: track id and frames must be whole numbers")
        if row[3] not in NAMES:
            raise ValueError(f"{name} line {i}: boxer must be one of {NAMES}")
        a, b = int(row[1]), int(row[2])
        if a > b:
            raise ValueError(f"{name} line {i}: first frame after last frame")
        out.append(Segment(int(row[0]), a, b, row[3]))
    return out


def read_identity(csv_path, table: PoseTable, tracks_path, gate_result=None) -> Identity:
    """An identity file and its sidecar, checked against this tracks table and file. Raises ValueError on
    anything it cannot vouch for (module docstring); never repairs or guesses."""
    csv_path, tracks_path = Path(csv_path), Path(tracks_path)
    if clip_of(csv_path) != clip_of(tracks_path):
        raise ValueError(f"{csv_path.name} does not name the clip of {tracks_path.name}")
    side_p = sidecar_path(csv_path)
    if not side_p.is_file():
        raise ValueError(f"no {side_p.name} beside {csv_path.name}: an identity needs its provenance sidecar")
    raw_side = _read_capped(side_p, MAX_SIDECAR_BYTES, "identity sidecar")
    side = _json_object(raw_side, side_p.name)
    if set(side) != _SIDECAR_KEYS:
        raise ValueError(f"{side_p.name}: expected exactly the keys {sorted(_SIDECAR_KEYS)}")
    if side["format"] != FORMAT:
        raise ValueError(f"{side_p.name}: not a {FORMAT} sidecar")
    if not isinstance(side["method"], str) or not _METHOD.fullmatch(side["method"]):
        raise ValueError(f"{side_p.name}: the method must be lower-case letters and hyphens (at most 40)")
    gate, measured = _gate_and_measure(side["gate"], gate_result, side["method"])
    check_binding(side["tracks"], table, tracks_path, f"identity {csv_path.name}")
    raw = _read_capped(csv_path, MAX_CSV_BYTES, "identity")
    if not isinstance(side["csv_sha256"], str) or hashlib.sha256(raw).hexdigest() != side["csv_sha256"]:
        raise ValueError(f"{csv_path.name} is not the CSV its sidecar describes (changed since it was written)")
    segments = _parse_identity_csv(raw, csv_path.name)
    check_segments(table, segments)
    return Identity(side["method"], gate, tuple(segments), side["csv_sha256"],
                    hashlib.sha256(raw_side).hexdigest(), measured)


def write_identity(csv_path, table: PoseTable, tracks_path, segments, method: str, gate: dict) -> None:
    """Write `<clip>.identity.csv` and its sidecar. Checked by the reader's own rules first, so a method
    cannot write an identity the report would refuse (a gate result file is checked when it is read)."""
    csv_path = Path(csv_path)
    if clip_of(csv_path) != clip_of(tracks_path):
        raise ValueError(f"{csv_path.name} does not name the clip of {Path(tracks_path).name}")
    if not isinstance(method, str) or not _METHOD.fullmatch(method):
        raise ValueError("the method must be lower-case letters and hyphens (at most 40)")
    gate = dict(gate)
    if gate.get("verdict") == "pass":   # the result file is checked on reading; here only its shape
        _check_gate({**gate, "verdict": "fail"}, None, method)
    else:
        _check_gate(gate, None, method)
    segments = sorted(segments, key=lambda s: (s.track_id, s.first))
    check_segments(table, segments)
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(IDENTITY_HEADER)
    for s in segments:
        w.writerow([int(s.track_id), int(s.first), int(s.last), s.boxer])
    data = buf.getvalue().encode("utf-8")
    replace_bytes(csv_path, data)
    replace_bytes(sidecar_path(csv_path), json.dumps(
        {"format": FORMAT, "csv_sha256": hashlib.sha256(data).hexdigest(), "method": method, "gate": gate,
         "tracks": tracks_binding(table, tracks_path)}, indent=1, allow_nan=False).encode("utf-8"))


# ---------------------------------------------------------------------- adapters from methods ---

def segments_from_row_names(table: PoseTable, names) -> "list[Segment]":
    """Per pose-table row a name -- "A", "B", "unsure", or None (nobody the method calls a boxer) -- as maximal
    runs of one name within one track id, in frame order. Lossless: every row of the id inside a segment's
    frames carries that segment's name. This is how a per-row method (glove colour: `glovescore.glove_names`,
    with its internal 0 / 1 mapped to A / B by the caller) emits an identity."""
    names = list(names)
    if len(names) != len(table.frame):
        raise ValueError("one name per pose-table row")
    if any(n is not None and n not in NAMES for n in names):
        raise ValueError(f"a row's name must be one of {NAMES} or None")
    out = []
    for tid, rows in _rows_by_track(table).items():
        run = None
        for r in rows.tolist():
            n, f = names[r], int(table.frame[r])
            if run is not None and run[0] == n:
                run[2] = f
                continue
            if run is not None and run[0] is not None:
                out.append(Segment(tid, run[1], run[2], run[0]))
            run = [n, f, f]
        if run is not None and run[0] is not None:
            out.append(Segment(tid, run[1], run[2], run[0]))
    return out


def row_names_from_linking(table: PoseTable, lk, unsure_tracks=()) -> list:
    """The pose-only linker (#11) as per-row names: a row is A (B) where the linking names its track id as
    identity 0 (1) in its frame; a row of a track id in `unsure_tracks` that the linking does not name is
    unsure (Olympic: a fragment credited with a labelled punch -- a boxer -- left out of both chains); every
    other row is nobody."""
    if lk.n_frames != table.n_frames or lk.source != table.source:
        raise ValueError("the linking was made from a different pose table than this one")
    unsure = {int(t) for t in unsure_tracks}
    names = []
    for f, t in zip(table.frame.tolist(), table.track_id.tolist()):
        m = lk.membership[f]
        names.append("A" if m[0] == t else "B" if m[1] == t else (UNSURE if t in unsure else None))
    return names


# ----------------------------------------------------------------------- punches per boxer ---

def boxer_rows(table: PoseTable, identity: Identity, who: str) -> np.ndarray:
    by_track = _rows_by_track(table)
    rows = [segment_rows(table, s, by_track) for s in identity.segments if s.boxer == who]
    return np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)


def scan_segments_of(table: PoseTable, identity: Identity, who: str, max_gap: int) -> "list[np.ndarray]":
    """The row segments the finder runs on: each identity segment of `who` on its own, cut further wherever a
    hole exceeds `max_gap` (completeness._split_rows, as boxer_segments cuts a linked boxer). An identity
    segment's edge is ALWAYS a cut: frames the method left unsure, gave the other boxer, or did not name are
    never bridged back into this boxer, however short (Astra, review 1). In frame order."""
    by_track = _rows_by_track(table)
    out = []
    for s in identity.segments:
        if s.boxer == who:
            out += completeness._split_rows(table, segment_rows(table, s, by_track), max_gap)
    return sorted(out, key=lambda r: (int(table.frame[r[0]]), int(table.track_id[r[0]])))


def finder_params(fps: float) -> "tuple[int, int]":
    """(bridge, min_rows): the finder's own, as completeness.find_bursts sets them."""
    return (int(round(punchfind.BRIDGE_S * fps)),
            max(2, int(math.ceil(completeness.MIN_SEGMENT_S * fps - 1e-9))))


def find(table: PoseTable, identity: Identity, rule: dict) -> dict:
    """Per name (A, B, unsure): the scan's statistics and every processed segment as (rows, bridged track,
    signal, peak frames, peak hands)."""
    bridge, min_rows = finder_params(table.fps)
    out = {}
    for who in NAMES:
        kept: list = []
        scan = completeness._scan_segments(table, scan_segments_of(table, identity, who, bridge), rule,
                                           f"boxer {who}", bridge, min_rows, keep=kept)
        stats = {"segments": scan["segments"], "short_segments": scan["short_segments"],
                 "bodyless_segments": scan["bodyless_segments"], "exposure_frames": int(scan["mask"].sum()),
                 "identified_frames": int(len(np.unique(table.frame[boxer_rows(table, identity, who)])))}
        out[who] = {"stats": stats, "kept": kept}
    return out


def _peak_rows(table: PoseTable, kept) -> list:
    """Each punch of the kept segments: (frame, hand, track id, segment index, finder columns)."""
    out = []
    for k, (seg, track, sig, frames, hands) in enumerate(kept):
        observed = set(table.frame[seg].tolist())
        valid = punchfind.hand_valid(track)
        row_of = {int(f): i for i, f in enumerate(track.frame_ids)}
        tid = int(table.track_id[seg[0]])
        for f, h in zip(frames, hands):
            i, h = row_of[int(f)], int(h)
            interp = int(f) not in observed or not valid[i, h]
            out.append((int(f), h, tid, k, [str(int(f)), f"{int(f) / table.fps:.3f}", ("left", "right")[h],
                                            f"{sig[i, h]:.3f}", "yes" if interp else "no"]))
    return sorted(out, key=lambda p: (p[0], p[1], p[2]))


def name_punches(table: PoseTable, kept, model, man: dict, ex: dict, stance: str) -> "tuple[list, object]":
    """One boxer's punches, named as classify_punches.py names a clip's: one cut pooled from this boxer's own
    punches, each punch read around its own peak on the segment track it was found on, lead / rear from this
    boxer's stance. Returns ([BOXERS_HEADER-minus-boxer rows as strings], the cut)."""
    if stance not in classify.STANCES:
        raise ValueError(f"stance must be one of {sorted(classify.STANCES)}")
    geom = man["geometry"]
    peaks = _peak_rows(table, kept)
    torso = {k: punchfind.body_scale(kept[k][1]) for k in {p[3] for p in peaks}}
    cut = classify.cut_from_widths([punchfind.peak_width(kept[k][1], kept[k][2], f, h, ex["frac"])
                                    for f, h, _t, k, _c in peaks], ex, geom)
    reads = [classify.read_clip_window(kept[k][1], f, cut, geom, torso[k]) for f, _h, _t, k, _c in peaks]
    ok = [i for i, (w, _s) in enumerate(reads) if w.keypoints is not None]
    probs = classify.predict_proba(model, [reads[i][0].keypoints for i in ok], man["t_out"]) if ok else []
    pred = dict(zip(ok, probs))
    rows = []
    for i, (f, h, tid, _k, cols) in enumerate(peaks):
        side = classify.hand_side(cols[2], stance)
        win, span = reads[i]
        base = [str(tid)] + cols + [side]
        if i not in pred:
            rows.append(base + [report.UNCLASSIFIED, "", "", ""] + [str(span[0]), str(span[1]),
                                                                       str(win.observed), win.reason])
            continue
        c = int(np.argmax(pred[i]))
        t_side = classify.class_side(CLASSES[c])
        rows.append(base + [CLASSES[c], f"{pred[i][c]:.3f}", t_side, "yes" if t_side == side else "no",
                            str(span[0]), str(span[1]), str(win.observed), ""])
    return rows, cut


def unsure_rows(table: PoseTable, kept) -> list:
    return [[str(tid)] + cols for _f, _h, tid, _k, cols in _peak_rows(table, kept)]


def _csv_bytes(header, rows) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def write_boxers(out_dir, clip: str, table: PoseTable, tracks_path, identity: Identity, found: dict,
                 named: dict, cuts: dict, stances: dict, man: dict, rule: dict, ex: dict) -> "tuple[Path, Path]":
    """`<clip>.boxers.csv` (A's and B's named punches, a boxer and track id column in front of the classified
    columns), `<clip>.unsure.csv` (punches on unsure segments: found, counted, never named) and the sidecar
    `<clip>.boxers.json` binding both to the tracks file, the identity, the classifier, rule, extent and stances."""
    out_dir = Path(out_dir)
    rows = sorted(([b] + r for b in BOXERS for r in named[b]), key=lambda r: (int(r[2]), r[0]))
    data = _csv_bytes(BOXERS_HEADER, rows)
    udata = _csv_bytes(UNSURE_HEADER, unsure_rows(table, found[UNSURE]["kept"]))
    p, up = out_dir / f"{clip}.boxers.csv", out_dir / f"{clip}.unsure.csv"
    replace_bytes(p, data)
    replace_bytes(up, udata)
    side = {"format": BOXERS_FORMAT, "clip": clip, "csv_sha256": hashlib.sha256(data).hexdigest(),
            "unsure_csv_sha256": hashlib.sha256(udata).hexdigest(), "classifier_sha256": man["sha256"],
            "stances": dict(stances), "rule": rule,
            # classify.read_extent has checked the rule was measured with the classifier's variant
            "extent": {**{k: ex[k] for k in ("frac", "lead_widths", "tail_widths")},
                       "variant": man["geometry"].variant},
            "cuts": {b: None if cuts[b] is None else {"per": "clip", **cuts[b]._asdict()} for b in BOXERS},
            "pose_frames": int(table.n_frames), "fps": float(table.fps),
            "tracks": tracks_binding(table, tracks_path),
            "identity": {"csv_sha256": identity.csv_sha256, "sidecar_sha256": identity.sidecar_sha256,
                         "method": identity.method, "gate_verdict": identity.verdict},
            "scan": {w: found[w]["stats"] for w in NAMES}}
    replace_bytes(out_dir / f"{clip}.boxers.json", json.dumps(side, indent=1, allow_nan=False).encode("utf-8"))
    return p, up


# ------------------------------------------------------------ reading the punches back (report) ---

def _in_segment(segs: "dict[tuple, list]", who: str, tid: int, frame: int) -> bool:
    return any(a <= frame <= b for a, b in segs.get((who, tid), ()))


def read_boxers(csv_path, table: PoseTable, tracks_path, identity: Identity, man: dict) -> dict:
    """The punches per boxer, checked against everything they were made from: the CSVs are the ones the
    sidecar hashed; the sidecar is exactly the known keys, names this clip, this tracks file (sha256, frames,
    fps), this identity (its CSV and sidecar hashes, method and verdict) and this classifier; every row parses as
    classify_punches.py's own (report.read_classified, per boxer), its lead / rear follows that boxer's recorded
    stance, and it lies inside one of that boxer's identity segments of its track id. Returns
    {"side", "rows": {A, B}, "unsure": [{frame, time_s, hand}]}."""
    csv_path = Path(csv_path)
    clip = clip_of(csv_path)
    if clip != clip_of(tracks_path):
        raise ValueError(f"{csv_path.name} does not name the clip of {Path(tracks_path).name}")
    side_p = csv_path.with_suffix(".json")
    if not side_p.is_file():
        raise ValueError(f"no {side_p.name} beside {csv_path.name}: re-run the per-boxer punch stage")
    side = _json_object(_read_capped(side_p, MAX_SIDECAR_BYTES, "boxers sidecar"), side_p.name)
    if set(side) != _BOXERS_KEYS or side["format"] != BOXERS_FORMAT:
        raise ValueError(f"{side_p.name}: not a {BOXERS_FORMAT} sidecar")
    if side["clip"] != clip:
        raise ValueError(f"{side_p.name} describes another clip than {clip}")
    check_binding(side["tracks"], table, tracks_path, f"{csv_path.name}")
    want_id = {"csv_sha256": identity.csv_sha256, "sidecar_sha256": identity.sidecar_sha256,
               "method": identity.method, "gate_verdict": identity.verdict}
    if not _same(side["identity"], want_id):
        raise ValueError(f"{csv_path.name} was split between the boxers by another identity file than this one")
    if side["classifier_sha256"] != man["sha256"]:
        raise ValueError("these punches were named by another classifier than this manifest describes; its "
                         "reliability does not apply")
    report.check_pose(side, table.n_frames, table.fps)
    st = side["stances"]
    if (not isinstance(st, dict) or set(st) != set(BOXERS)
            or any(not isinstance(st[b], str) or st[b] not in classify.STANCES for b in BOXERS)):
        raise ValueError(f"{side_p.name}: a stance per boxer, each one of {sorted(classify.STANCES)}")
    if not isinstance(side["cuts"], dict) or set(side["cuts"]) != set(BOXERS):
        raise ValueError(f"{side_p.name}: a cut (or null) per boxer")
    for b in BOXERS:
        report.cut_line({"cut": side["cuts"][b]})   # its own field checks
    raw = _read_capped(csv_path, MAX_CSV_BYTES, "punches")
    if hashlib.sha256(raw).hexdigest() != side["csv_sha256"]:
        raise ValueError(f"{csv_path.name} has changed since the per-boxer punch stage wrote it")
    up = csv_path.with_name(f"{clip}.unsure.csv")
    uraw = _read_capped(up, MAX_CSV_BYTES, "unsure punches")
    if hashlib.sha256(uraw).hexdigest() != side["unsure_csv_sha256"]:
        raise ValueError(f"{up.name} has changed since the per-boxer punch stage wrote it")

    segs: "dict[tuple, list]" = {}
    for s in identity.segments:
        segs.setdefault((s.boxer, s.track_id), []).append((s.first, s.last))
    duration = table.n_frames / table.fps
    per = {b: [] for b in BOXERS}
    finder: "dict[str, list]" = {w: [] for w in NAMES}
    r = iter(_csv_rows(raw, csv_path.name))
    if next(r, None) != BOXERS_HEADER:
        raise ValueError(f"{csv_path.name}: not a per-boxer punches CSV")
    for i, row in enumerate(r, start=2):
        if sum(len(v) for v in per.values()) >= MAX_ROWS:
            raise ValueError(f"{csv_path.name}: more than {MAX_ROWS} rows")
        if len(row) != len(BOXERS_HEADER) or row[0] not in BOXERS or not _ID.fullmatch(row[1]) \
                or not _ID.fullmatch(row[2]):
            raise ValueError(f"{csv_path.name} line {i}: not a punch of boxer A or B with a track id and frame")
        if not _in_segment(segs, row[0], int(row[1]), int(row[2])):
            raise ValueError(f"{csv_path.name} line {i}: a punch of boxer {row[0]} on track {row[1]} at frame "
                             f"{row[2]}, where the identity does not name that track as boxer {row[0]}")
        per[row[0]].append(row[2:])
        finder[row[0]].append((int(row[1]), row[2:7]))
    out = {"side": side, "rows": {}, "unsure": []}
    for b in BOXERS:
        rows = report.read_classified(csv_path, duration, data=_csv_bytes(report.CLASSIFIED_HEADER, per[b]),
                                      fps=table.fps)
        for d in rows:
            if classify.hand_side(d["hand"], st[b]) != d["lead_rear"]:
                raise ValueError(f"{csv_path.name}: a punch of boxer {b} at frame {d['frame']} is not "
                                 f"{d['lead_rear']} for a {st[b]} boxer")
        out["rows"][b] = rows
    r = iter(_csv_rows(uraw, up.name))
    if next(r, None) != UNSURE_HEADER:
        raise ValueError(f"{up.name}: not an unsure-punches CSV")
    for i, row in enumerate(r, start=2):
        if len(out["unsure"]) >= MAX_ROWS:
            raise ValueError(f"{up.name}: more than {MAX_ROWS} rows")
        if (len(row) != len(UNSURE_HEADER) or not _ID.fullmatch(row[0]) or not _ID.fullmatch(row[1])
                or row[3] not in ("left", "right") or row[5] not in ("yes", "no")):
            raise ValueError(f"{up.name} line {i}: not a punch on an unsure segment")
        f = int(row[1])
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", row[2]) or abs(float(row[2]) - f / table.fps) > report.TIME_TOL_S:
            raise ValueError(f"{up.name} line {i}: time {row[2][:20]!r} is not frame {f} of this clip")
        if not _in_segment(segs, UNSURE, int(row[0]), f):
            raise ValueError(f"{up.name} line {i}: not on an unsure segment of the identity")
        out["unsure"].append({"frame": f, "time_s": float(row[2]), "hand": row[3]})
        finder["unsure"].append((int(row[0]), row[1:6]))
    out["finder"] = finder
    return out


def _same(a, b) -> bool:
    """Equal as JSON, types included: Python's == calls True equal to 1, JSON does not, and a recorded count of
    true is not a count."""
    try:
        return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def _extent_of(d, variant: str) -> dict:
    """The sidecar's punch-extent rule: exactly the three numbers a cut is computed from and the wrist-speed
    variant, which must be the classifier's."""
    if not isinstance(d, dict) or set(d) != {"frac", "lead_widths", "tail_widths", "variant"}:
        raise ValueError("the sidecar's extent is not a punch-extent rule")
    if d["variant"] != variant:
        raise ValueError("the sidecar's extent was measured with another wrist-speed variant than the classifier's")
    out = {}
    # classify.read_extent's own bounds, so the report accepts every rule the punch stage accepted (Astra, review
    # 2); each range is checked before any float operation: a 400-digit JSON integer must not reach isfinite
    for k, lo, hi in (("frac", 0.0, 1.0), ("lead_widths", 0.0, 20.0), ("tail_widths", 0.0, 20.0)):
        v = d.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi or not math.isfinite(v):
            raise ValueError(f"the sidecar's extent {k} is not a number in [{lo:g}, {hi:g}]")
        out[k] = float(v)
    if not (0 < out["frac"] < 1 and out["lead_widths"] + out["tail_widths"] > 0):
        raise ValueError("the sidecar's extent: frac must be strictly inside (0, 1) and the extent non-empty")
    return out


def check_against_finder(got: dict, found: dict, table: PoseTable, man: dict) -> None:
    """The punches read back are EXACTLY what the finder gives on this tracks file with this identity and rule:
    every boxer's and the unsure segments' punches (track id, frame, time, hand, peak speed, interpolated flag),
    the scan statistics, and each boxer's cut, recomputed from those punches and the recorded extent rule. A
    sidecar re-hashed over an edited CSV is refused here: the hashes detect staleness, this detects anything
    the finder did not find."""
    ex = _extent_of(got["side"]["extent"], man["geometry"].variant)
    for b in BOXERS:
        peaks = _peak_rows(table, found[b]["kept"])
        kept = found[b]["kept"]
        cut = classify.cut_from_widths([punchfind.peak_width(kept[k][1], kept[k][2], f, h, ex["frac"])
                                        for f, h, _t, k, _c in peaks], ex, man["geometry"])
        if not _same(got["side"]["cuts"][b], None if cut is None else {"per": "clip", **cut._asdict()}):
            raise ValueError(f"the cut the sidecar records for boxer {b} is not the one its punches give")
    for who in NAMES:
        want = [(tid, cols) for _f, _h, tid, _k, cols in _peak_rows(table, found[who]["kept"])]
        have = sorted(got["finder"][who], key=lambda p: (int(p[1][0]), p[1][2], p[0]))
        if have != want:
            raise ValueError(f"the punches recorded for {'boxer ' + who if who in BOXERS else 'unsure segments'} "
                             "are not what the finder finds on these tracks with this identity and rule")
    if not _same(got["side"]["scan"], {w: found[w]["stats"] for w in NAMES}):
        raise ValueError("the scan the sidecar records is not the finder's on these tracks with this identity")


# ---------------------------------------------------------------------------- the report ---

VERDICT_WORDS = {"pass": "passed its pre-registered gate", "fail": "failed its pre-registered gate",
                 "ungated": "has not been tested against a gate"}


def _check_measured(m) -> None:
    ok = (isinstance(m, dict) and set(m) == MEASURED_KEYS and m["footage"] == "Olympic Boxing"
          and all(isinstance(m[k], int) and not isinstance(m[k], bool) for k in ("decided", "different"))
          and 0 <= m["different"] <= m["decided"] and m["decided"] > 0
          and all(isinstance(m[k], float) and 0.0 <= m[k] <= 1.0 for k in ("join_error", "lower", "upper"))
          and m["lower"] <= m["join_error"] <= m["upper"] and m["verdict"] in MEASURED_VERDICTS)
    if not ok:
        raise ValueError("header: the measured identity error is not what a checked gate result gives")


def _check_measured_matches(verdict: str, m) -> None:
    """A gate PASS goes only with a measured "acceptable", and a measured "acceptable" only with a PASS."""
    if (verdict == "pass") != (m["verdict"] == "measured, acceptable"):
        raise ValueError("header: the gate verdict and the measured verdict disagree")


def _measured_clause(w, m) -> str:
    """'in a hand check of 98 of its joins on Olympic Boxing video, 65% joined two different people (95%
    interval 55% to 74%)'. Numbers only through the writer."""
    return (f"in a hand check of {w.count(m['decided'])} of its joins on {m['footage']} video, "
            f"{w.pct(m['join_error'])} joined two different people ({w.num('95')}% interval {w.pct(m['lower'])} to "
            f"{w.pct(m['upper'])})"
            + (", too few clear answers for its gate to judge" if m["verdict"] == "not established" else ""))


def header(boxer: str, method: str, verdict: str, n_unsure: int, n_all: int, identified_s: float,
           duration_s: float, measured: "dict | None" = None) -> "tuple[str, list[str]]":
    """The block that leads each boxer's summary. Fixed templates; every number through report._Writer. When
    the identity method failed its gate or has none, the FIRST sentence says the split is unverified, and, when
    the method's error was measured (`Identity.measured`, from its checked gate result), how large it was and
    on what footage."""
    if boxer not in BOXERS or verdict not in VERDICTS or not _METHOD.fullmatch(method):
        raise ValueError("header: boxer, verdict or method out of range")
    if measured is not None:
        _check_measured(measured)
        if verdict == "ungated":
            raise ValueError("header: an ungated method has no measured error")
        _check_measured_matches(verdict, measured)
    w = report._Writer()
    if verdict != "pass":
        w.say(f"Warning: which boxer threw each punch is unverified in this report: the method that told the two "
              f"boxers apart ({method}) {VERDICT_WORDS[verdict]}"
              + (f": {_measured_clause(w, measured)}" if measured else "")
              + ", so some punches counted here may have been thrown by the other boxer, and some of this "
                "boxer's may be counted in the other report.")
    w.say(f"This is boxer {boxer}'s report. The boxers were told apart by {method}, which {VERDICT_WORDS[verdict]}.")
    if verdict == "pass" and measured:
        w.say(f"Its error was measured: {_measured_clause(w, measured)}.")
    w.say(f"Boxer {boxer} was identified for {w.mmss(identified_s)} of the {w.mmss(duration_s)} of video.")
    if n_all == 0:
        w.say("The tool found no punch on either boxer, so none was left unassigned.")
    elif n_unsure == 0:
        w.say(f"Every one of the {w.count(n_all)} {report._punches(n_all)} found on the two boxers was assigned "
              f"to one of them.")
    else:
        w.say(f"Of the {w.count(n_all)} punches found on the two boxers, {w.count(n_unsure)} "
              f"({w.pct(n_unsure / n_all)}) fell where {method} could not say which boxer it was, and "
              f"{'is' if n_unsure == 1 else 'are'} counted in neither report.")
    return " ".join(w.parts), w.numbers


def identity_sheet(identity: Identity, boxer: str, stats: dict, n_unsure: int, n_all: int) -> list:
    """The Identity sheet's rows: what split the punches, and how much of the video it covered for this boxer."""
    g = identity.gate
    m = identity.measured
    measured = [] if not m else [["measured on", f"{m['footage']}, hand check of sampled joins"],
                                 ["measured verdict", m["verdict"]],
                                 ["joins decided (same or different)", m["decided"]],
                                 ["of them joining two different people", m["different"]],
                                 ["identity join error", round(m["join_error"], 4)],
                                 ["identity join error, Wilson 95% lower", round(m["lower"], 4)],
                                 ["identity join error, Wilson 95% upper", round(m["upper"], 4)]]
    return [["identity method", identity.method],
            ["gate verdict", g["verdict"]],
            ["gate document", g.get("document", "")]] + measured + [
            ["identity file sha256", identity.csv_sha256],
            ["punches found on the two boxers (inside the rounds)", n_all],
            ["of them on unsure segments, counted in neither report", n_unsure],
            ["unassigned share", None if not n_all else round(n_unsure / n_all, 4)],
            [f"boxer {boxer}: frames identified", stats["identified_frames"]],
            [f"boxer {boxer}: frames the finder ran on", stats["exposure_frames"]],
            [f"boxer {boxer}: segments (track id changes or holes split them)", stats["segments"]],
            [f"boxer {boxer}: segments too short for the finder", stats["short_segments"]],
            [f"boxer {boxer}: segments with no usable body", stats["bodyless_segments"]]]
