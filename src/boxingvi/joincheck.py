"""Brendan's hand check of the linker's joins on Olympic Boxing (option A; docs/gates/linker-join-check.md).

A JOIN is two consecutive fragments (track ids) of one linker chain (linking.py). The linker can only change
person at a join, so the share of joins that connect two different people is its identity error. This module
holds everything except the window: the sampling frame, the seeded gap-stratified sample, the sample and
answer files, the clip with the two fragments' boxes, and the score against the pre-registered bar.

Blinding. Nothing the checker shows or reads may say which boxer (A or B) a chain is, what a link cost, or
which stratum a join is in. `Join` therefore has no chain field; the sample file stores per join only the
video, the two track ids and the two boundary frames. The stratum is not stored at all: the score derives it
from the join's gap and the frame's median, and the window shows no gap length.

Everything derived from Olympic Boxing (the sample, the answers, the result) stays under the gitignored
results/; it is research-use only and never redistributed. Messages never repeat a path or an input value.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import labelmoments as lm

FORMAT_SAMPLE = "linker-join-sample-1"
FORMAT_ANSWERS = "linker-join-answers-1"
GATE = "docs/gates/linker-join-check.md"
SEED = 20261027                 # the gate's seed (sample draw, shuffle, bootstrap)
N_SAMPLE = 100                  # Brendan, 2026-10-28
TARGET_MISATTRIBUTION = 0.10    # the 0.90 identity bar of ADR-0013 / #29, stated for punches (ADR-0018)
MAX_UNSURE_SHARE = 0.30
MIN_DECIDED = 70
BOOT_DRAWS = 10_000
MAX_BOOT_DROPPED = 0.01
BREAK_MIN = 50.0                # the window refuses answers after this many minutes in one sitting
EDGE_S = 1.0                    # fragment 1's last 1.0 s and fragment 2's first 1.0 s
MAX_DISPLAY_W = 640
STILL_SEP = 8                   # black pixels between the two stills
BOX_PX = 3
CYAN = (0, 255, 255)            # fragment 1, on every join
YELLOW = (255, 255, 0)          # fragment 2, on every join
VALUES = ("S", "D", "U")
MAX_SAMPLE_BYTES = 4 * 1024 * 1024
MAX_ANSWER_BYTES = 64 * 1024
MAX_MINUTES = lm.MAX_MINUTES
MAX_JOINS = 10_000              # far above any real sample: a file claiming more is not one of ours
VIDEO_ID = re.compile(r"[A-Za-z0-9]{1,32}")
_FP = re.compile(r"[0-9a-f]{64}")

Refused = lm.Refused


# ---- joins and the frame ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Join:
    """One join, and nothing about the chain it came from."""

    video: str
    tid1: int
    tid2: int
    last1: int      # fragment 1's last frame
    first2: int     # fragment 2's first frame

    @property
    def gap(self) -> int:
        return self.first2 - self.last1


def joins_of(lk, video: str) -> "list[Join]":
    """Every consecutive pair of every chain of one Linking, in a canonical order that does not follow
    the chains (sorted by track ids)."""
    frag = {f.tid: f for f in lk.fragments}
    out = [Join(video, int(a), int(b), int(frag[a].last), int(frag[b].first))
           for chain in lk.identities for a, b in zip(chain, chain[1:])]
    return sorted(out, key=lambda j: (j.video, j.tid1, j.tid2))


def positions_of(lk, credited: "dict[int, int]") -> "list[list[int]]":
    """[k, punches] for every fragment of every chain holding a credited punch; k = joins before it in its
    chain. Pooled over both chains, so it does not say which chain is which."""
    out = [[k, int(credited[t])] for chain in lk.identities for k, t in enumerate(chain) if credited.get(t, 0)]
    return sorted(out)


def chain_minutes(lk) -> float:
    return float(lk.identified(0).sum() + lk.identified(1).sum()) / lk.fps / 60.0


def camera_of(task_name: str) -> str:
    """'task_kam2_gh078416' -> 'kam2'."""
    m = re.search(r"kam\d", task_name or "")
    return m.group(0) if m else "unknown"


def max_gap_frames(fps: float) -> int:
    """The linker's own bound (linking.candidate_links, no re-acquisition): no join is longer."""
    from .linking import MAX_GAP_S
    return max(1, int(round(MAX_GAP_S * fps)))


# ---- the estimate ----------------------------------------------------------------------------------
def wilson(x: int, n: int, z: float = 1.959963984540054) -> "tuple[float, float]":
    if n <= 0 or not 0 <= x <= n:
        raise ValueError("wilson needs 0 <= x <= n and n > 0")
    p = x / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = 0.0 if x == 0 else max(0.0, (c - h) / d)      # exact at the ends (floating point leaves ~1e-18)
    hi = 1.0 if x == n else min(1.0, (c + h) / d)
    return lo, hi


def misattribution(p: float, positions) -> float:
    """Expected share of credited punches on someone other than their chain's first person, if every join
    is wrong independently with probability p and a wrong join switches between two people: a fragment k
    joins in is switched an odd number of times with probability (1 - (1 - 2p)^k) / 2."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    w = sum(n for _, n in positions)
    if w <= 0:
        raise ValueError("no credited punch in any chain: misattribution is undefined")
    return sum(n * (1 - (1 - 2 * p) ** k) / 2 for k, n in positions) / w


def p_star(positions, target: float = TARGET_MISATTRIBUTION) -> float:
    """The join error at which misattribution reaches `target` (bisection; monotone on [0, 0.5])."""
    if misattribution(0.5, positions) < target:
        return 0.5
    lo, hi = 0.0, 0.5
    for _ in range(80):
        mid = (lo + hi) / 2
        if misattribution(mid, positions) < target:
            lo = mid
        else:
            hi = mid
    return hi


# ---- the sample ------------------------------------------------------------------------------------
def draw(joins: "list[Join]", n: int = N_SAMPLE, seed: int = SEED) -> "tuple[list[Join], dict, list[str]]":
    """The gate's draw. Strata: short = gap <= the frame's median gap, long = above. Proportional
    allocation (self-weighting). random.Random(seed); each stratum sampled without replacement from the
    canonical order (video, tid1, tid2); then the n are shuffled with the same generator.
    Returns (the joins in the order shown, the frame's counts, each shown join's stratum)."""
    if len(set(joins)) != len(joins):
        raise Refused("the frame holds the same join twice")
    frame = sorted(joins, key=lambda j: (j.video, j.tid1, j.tid2))
    if len(frame) < n:
        raise Refused(f"the frame has {len(frame)} joins, fewer than the {n} to draw")
    med = statistics.median(j.gap for j in frame)
    short = [j for j in frame if j.gap <= med]
    long_ = [j for j in frame if j.gap > med]
    n_short = int(round(n * len(short) / len(frame)))
    n_short = min(max(n_short, n - len(long_)), len(short))
    n_long = n - n_short
    rng = random.Random(seed)
    picked = [(j, "short") for j in rng.sample(short, n_short)] + [(j, "long") for j in rng.sample(long_, n_long)]
    rng.shuffle(picked)
    counts = {"joins": len(frame), "short": len(short), "long": len(long_), "median_gap": float(med),
              "n_short": n_short, "n_long": n_long}
    return [j for j, _ in picked], counts, [s for _, s in picked]


def build_sample(per_video: dict, manifest, n: int = N_SAMPLE, seed: int = SEED) -> dict:
    """The sample file's content from every manifest video's linking.

    per_video[vid] = {"joins": [Join], "positions": [[k, n]], "chain_minutes": float, "camera": str,
                      "fps": float, "n_frames": int, "tracks_sha256": hex, "video_fp": hex}
    All manifest videos must be present and no other: otherwise nothing is drawn (the gate)."""
    have, want = set(per_video), set(manifest)
    if have != want:
        raise Refused(f"the sample needs exactly the {len(want)} manifest videos: {len(want - have)} missing, "
                      f"{len(have - want)} not in the manifest")
    joins = [j for v in sorted(per_video) for j in per_video[v]["joins"]]
    for j in joins:
        _check_join(j, per_video[j.video]["fps"], per_video[j.video]["n_frames"])
    positions = [p for v in sorted(per_video) for p in per_video[v]["positions"]]
    shown, counts, _ = draw(joins, n, seed)
    ps = p_star(positions)
    videos = {}
    for v in sorted(per_video):
        d = per_video[v]
        videos[v] = {"camera": d["camera"], "fps": float(d["fps"]), "n_frames": int(d["n_frames"]),
                     "tracks_sha256": d["tracks_sha256"], "video_fp": d["video_fp"],
                     "joins": len(d["joins"]), "chain_minutes": round(float(d["chain_minutes"]), 6),
                     "positions": [[int(k), int(c)] for k, c in d["positions"]]}
    return {"format": FORMAT_SAMPLE, "gate": GATE, "seed": seed, "n": n,
            "frame": counts | {"p_star": ps, "target_misattribution": TARGET_MISATTRIBUTION,
                               "misattribution_at_p_star": misattribution(ps, positions)},
            "videos": videos,
            "joins": [{"video": j.video, "tid1": j.tid1, "tid2": j.tid2, "last1": j.last1, "first2": j.first2}
                      for j in shown]}


def stratum(j: Join, median_gap: float) -> str:
    return "short" if j.gap <= median_gap else "long"


def _int(x, name: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int):
        raise Refused(f"{name} must be a whole number")
    return x


def _check_join(j: Join, fps: float, n_frames: int) -> None:
    """A join must lie inside its video, with fragment 2 after fragment 1 by 1 .. the linker's longest gap."""
    if not (0 <= j.last1 < j.first2 < n_frames):
        raise Refused("a join lies outside its video or its fragments are not in time order")
    if not 1 <= j.gap <= max_gap_frames(fps):
        raise Refused("a join's gap is longer than the linker allows")
    if j.tid1 == j.tid2 or j.tid1 < 0 or j.tid2 < 0:
        raise Refused("a join must connect two different, non-negative track ids")


SAMPLE_KEYS = {"format", "gate", "seed", "n", "frame", "videos", "joins"}
FRAME_KEYS = {"joins", "short", "long", "median_gap", "n_short", "n_long", "p_star", "target_misattribution",
              "misattribution_at_p_star"}
VIDEO_KEYS = {"camera", "fps", "n_frames", "tracks_sha256", "video_fp", "joins", "chain_minutes", "positions"}
JOIN_KEYS = {"video", "tid1", "tid2", "last1", "first2"}


@dataclass(frozen=True)
class Sample:
    raw: dict
    sha256: str
    joins: tuple        # Join, in the order shown

    @property
    def n(self) -> int:
        return len(self.joins)


def _num(x, name: str, lo: float = 0.0, hi: float = math.inf) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not lo <= x <= hi:
        raise Refused(f"{name} is not a number in range")
    return float(x)


def load_sample(path, manifest) -> Sample:
    p = Path(path)
    try:
        if p.stat().st_size > MAX_SAMPLE_BYTES:
            raise Refused("the sample file is too large to be a sample")
        raw_bytes = p.read_bytes()
    except OSError:
        raise Refused("the sample file cannot be read") from None
    return parse_sample(raw_bytes, manifest)


def parse_sample(raw_bytes: bytes, manifest) -> Sample:
    """The sample is checked whole before use: exact keys (so nothing extra, such as a chain name, can ride
    along), the gate's constants (format, seed, n, target), every video in the manifest, every join inside its
    video and within the linker's gap, and every number the score relies on: p* is recomputed from the
    positions, the frame's totals must add up, and the strata implied by the median must match n_short/n_long."""
    if len(raw_bytes) > MAX_SAMPLE_BYTES:
        raise Refused("the sample file is too large to be a sample")
    try:
        d = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise Refused("the sample file is not JSON") from None
    if not isinstance(d, dict) or set(d) != SAMPLE_KEYS or d.get("format") != FORMAT_SAMPLE:
        raise Refused("the sample file is not a linker-join sample (unknown format or keys)")
    if not isinstance(d["frame"], dict) or set(d["frame"]) != FRAME_KEYS:
        raise Refused("the sample's frame summary has unexpected keys")
    if not isinstance(d["videos"], dict) or not isinstance(d["joins"], list):
        raise Refused("the sample file's sections have the wrong types")
    if d["gate"] != GATE or _int(d["seed"], "seed") != SEED or _int(d["n"], "n") != N_SAMPLE:
        raise Refused("the sample was not drawn with the gate's seed and n")
    if len(d["joins"]) != N_SAMPLE:
        raise Refused("the sample's join count does not match n")
    man = set(manifest)
    for v, info in d["videos"].items():
        if v not in man or not VIDEO_ID.fullmatch(v):
            raise Refused("the sample names a video outside the manifest")
        if not isinstance(info, dict) or set(info) != VIDEO_KEYS:
            raise Refused("a video entry has unexpected keys")
        if not _FP.fullmatch(str(info["tracks_sha256"])) or not _FP.fullmatch(str(info["video_fp"])):
            raise Refused("a video entry's hashes are not sha256 hex digests")
        fps = info["fps"]
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not (math.isfinite(fps) and 1 <= fps <= 240):
            raise Refused("a video entry's frame rate is out of range")
        if not 1 <= _int(info["n_frames"], "n_frames") <= 10_000_000:
            raise Refused("a video entry's frame count is out of range")
        if not 0 <= _int(info["joins"], "joins") <= MAX_JOINS * 100:
            raise Refused("a video entry's join count is out of range")
        _num(info["chain_minutes"], "chain minutes", 0.0, 10_000.0)
        if not isinstance(info["camera"], str) or not re.fullmatch(r"kam\d|unknown", info["camera"]):
            raise Refused("a video entry's camera is not kamN")
        pos = info["positions"]
        if not isinstance(pos, list) or len(pos) > 1_000_000 or not all(
                isinstance(x, list) and len(x) == 2 and _int(x[0], "position") >= 0 and _int(x[1], "punches") >= 1
                for x in pos):
            raise Refused("a video entry's positions are not [joins before, punches] pairs")
    if set(d["videos"]) != man:
        raise Refused("the sample does not cover exactly the manifest's videos")
    joins = []
    for e in d["joins"]:
        if not isinstance(e, dict) or set(e) != JOIN_KEYS:
            raise Refused("a join has unexpected keys")
        v = e["video"]
        if v not in d["videos"]:
            raise Refused("a join names a video outside the manifest")
        j = Join(v, _int(e["tid1"], "tid1"), _int(e["tid2"], "tid2"), _int(e["last1"], "last1"),
                 _int(e["first2"], "first2"))
        _check_join(j, d["videos"][v]["fps"], d["videos"][v]["n_frames"])
        joins.append(j)
    if len(set(joins)) != len(joins):
        raise Refused("the sample holds the same join twice")
    fr = d["frame"]
    total = _int(fr["joins"], "frame joins")
    if total != sum(i["joins"] for i in d["videos"].values()) or total < N_SAMPLE \
            or _int(fr["short"], "short") + _int(fr["long"], "long") != total:
        raise Refused("the frame's join counts do not add up")
    med = _num(fr["median_gap"], "median gap", 1.0, float(max_gap_frames(240.0)))
    n_short = sum(stratum(j, med) == "short" for j in joins)
    if (_int(fr["n_short"], "n_short"), _int(fr["n_long"], "n_long")) != (n_short, N_SAMPLE - n_short):
        raise Refused("the sample's strata do not match its median gap")
    positions = [x for v in sorted(d["videos"]) for x in d["videos"][v]["positions"]]
    if _num(fr["target_misattribution"], "target") != TARGET_MISATTRIBUTION:
        raise Refused("the sample's misattribution target is not the gate's")
    try:
        ps = p_star(positions)
    except ValueError:
        raise Refused("the sample's chains hold no credited punch") from None
    if abs(_num(fr["p_star"], "p*", 0.0, 0.5) - ps) > 1e-9:
        raise Refused("the sample's p* does not follow from its positions")
    return Sample(d, hashlib.sha256(raw_bytes).hexdigest(), tuple(joins))


def write_new(path: Path, data: bytes) -> None:
    """Create a file that must not exist yet (the sample and the result are written once, never redrawn)."""
    try:
        with open(path, "xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        raise Refused("that file already exists: a sample or result is written once and never replaced") from None


def sample_bytes(sample: dict) -> bytes:
    return (json.dumps(sample, indent=1, allow_nan=False) + "\n").encode("utf-8")


# ---- the answers file ------------------------------------------------------------------------------
_HEAD = re.compile(r"# (format|sample_sha256|minutes): ?(.*)")
_ROW = re.compile(r"([1-9][0-9]{0,4}),([SDU]?)")


class AnswerFile:
    """`# format`, `# sample_sha256`, `# minutes`, then `k,answer` for k = 1 .. n. The interface Session
    (labelmoments) uses: values, first_empty, set_value, prior_minutes, set_minutes, counts, save."""

    def __init__(self, path: Path, sample_sha: str, values: list, minutes: str, on_disk: bytes):
        self.path, self.sample_sha, self.values, self._minutes, self.on_disk = path, sample_sha, values, minutes, on_disk

    @staticmethod
    def template(sample_sha: str, n: int) -> bytes:
        return ("\n".join([f"# format: {FORMAT_ANSWERS}", f"# sample_sha256: {sample_sha}", "# minutes: ",
                           "k,answer"] + [f"{k}," for k in range(1, n + 1)]) + "\n").encode("utf-8")

    @classmethod
    def create(cls, path, sample: Sample) -> "AnswerFile":
        write_new(Path(path), cls.template(sample.sha256, sample.n))
        return cls.load(path, sample)

    @classmethod
    def load(cls, path, sample: Sample) -> "AnswerFile":
        p = Path(path)
        try:
            if p.stat().st_size > MAX_ANSWER_BYTES:
                raise Refused("the answers file is too large")
            raw = p.read_bytes()
            text = raw.decode("utf-8")
        except OSError:
            raise Refused("the answers file cannot be read") from None
        except UnicodeDecodeError:
            raise Refused("the answers file is not UTF-8 text") from None
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        head = {}
        for line in lines[:3]:
            m = _HEAD.fullmatch(line)
            if not m or m.group(1) in head:
                raise Refused("the answers file's header is not as the tool wrote it")
            head[m.group(1)] = m.group(2)
        if set(head) != {"format", "sample_sha256", "minutes"} or head["format"] != FORMAT_ANSWERS:
            raise Refused("the answers file is not a linker-join answers file")
        if head["sample_sha256"] != sample.sha256:
            raise Refused("the answers file belongs to a different sample (sha256 differs)")
        if len(lines) < 4 or lines[3] != "k,answer":
            raise Refused("the column header must be k,answer")
        rows = lines[4:]
        if len(rows) != sample.n:
            raise Refused(f"the answers file must have exactly {sample.n} rows")
        values = []
        for k, line in enumerate(rows, start=1):
            m = _ROW.fullmatch(line)
            if not m or int(m.group(1)) != k:
                raise Refused(f"row {k} is not '{k},' or '{k},S' / D / U")
            values.append(m.group(2))
        af = cls(p, sample.sha256, values, head["minutes"], raw)
        af.prior_minutes()
        return af

    def to_bytes(self) -> bytes:
        return ("\n".join([f"# format: {FORMAT_ANSWERS}", f"# sample_sha256: {self.sample_sha}",
                           f"# minutes: {self._minutes}", "k,answer"]
                          + [f"{k},{v}" for k, v in enumerate(self.values, start=1)]) + "\n").encode("utf-8")

    def prior_minutes(self) -> float:
        s = self._minutes.strip()
        if not s:
            return 0.0
        try:
            m = float(s)
        except ValueError:
            raise Refused("'# minutes:' holds something that is not a number") from None
        if not (math.isfinite(m) and 0 <= m <= MAX_MINUTES):
            raise Refused(f"'# minutes:' must be between 0 and {MAX_MINUTES:.0f}")
        return m

    def set_minutes(self, minutes: float) -> None:
        if not (math.isfinite(minutes) and 0 < minutes <= MAX_MINUTES):
            raise ValueError("minutes out of range")
        self._minutes = f"{minutes:.2f}"

    def set_value(self, k: int, value: str) -> None:
        if value not in VALUES:
            raise ValueError("an answer must be S, D or U")
        self.values[k] = value

    def first_empty(self):
        return next((k for k, v in enumerate(self.values) if not v), None)

    def counts(self) -> dict:
        return {k: sum(v == k for v in self.values) for k in VALUES} | {"empty": sum(not v for v in self.values)}

    def save(self) -> None:
        data = self.to_bytes()
        lm.replace_file(self.path, data, self.on_disk, "answers file", prefix=".linker_join_check-")
        self.on_disk = data


class SessionLock:
    """One check window per answers file. replace_file refuses a stale copy, but two windows that read the
    same bytes could both pass that comparison and the second rename would drop the first's answer (Astra).
    The OS lock on `<answers>.lock` is held for the whole session and released by the OS if the process dies,
    so a crash never leaves the file locked."""

    def __init__(self, answers_path):
        p = Path(answers_path)
        self.path = p.with_name(p.name + ".lock")
        try:
            self._f = open(self.path, "a+b")
        except OSError:
            self._f = None
            raise Refused("the answers file's lock cannot be opened") from None
        try:
            if os.name == "nt":
                import msvcrt
                self._f.seek(0)
                msvcrt.locking(self._f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._f.close()
            self._f = None
            raise Refused("another check window already has this answers file open") from None

    def close(self) -> None:
        if self._f is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self._f.seek(0)
                msvcrt.locking(self._f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        self._f.close()
        self._f = None


class Session(lm.Session):
    """labelmoments.Session, unchanged except that there is no `# A:` to ask for, and it can say when a
    break is due (BREAK_MIN of active time in this sitting)."""

    def needs_description(self) -> bool:
        return False

    def sitting_minutes(self) -> float:
        return self.active_s / 60.0

    def break_due(self) -> bool:
        self.tick()
        return self.sitting_minutes() >= BREAK_MIN


# ---- the clip --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class JoinClip:
    k: int               # the join's index in the sample (0-based)
    width: int
    height: int
    fps: float           # the video's rate
    start_f: int
    frames: tuple        # RGB24 bytes with the boxes drawn
    part: tuple          # per frame: 1 (fragment 1), 0 (gap), 2 (fragment 2)
    ms: tuple            # per frame: how long it stays on screen (real time: the source frames it stands for)
    still: bytes         # fragment 1's last frame | fragment 2's first frame
    still_width: int


def boxes_for(table, join: Join, fps: float) -> "tuple[dict, dict]":
    """{frame: xyxy} of fragment 1 over its last EDGE_S, and of fragment 2 over its first EDGE_S. Refuses a
    join the tracks do not bear out: fragment 1 must END at last1 and fragment 2 must START at first2."""
    e = max(1, int(round(EDGE_S * fps)))
    out = []
    for tid, lo, hi, edge in ((join.tid1, join.last1 - e + 1, join.last1, join.last1),
                              (join.tid2, join.first2, join.first2 + e - 1, join.first2)):
        rows = np.flatnonzero(table.track_id == tid)
        if not len(rows):
            raise Refused("a sampled track id is not in its tracks file")
        fr = table.frame[rows]
        if (tid == join.tid1 and int(fr.max()) != join.last1) or (tid == join.tid2 and int(fr.min()) != join.first2):
            raise Refused("a sampled join does not match its tracks file (a fragment's end or start moved)")
        keep = rows[(fr >= lo) & (fr <= hi)]
        out.append({int(table.frame[r]): tuple(float(v) for v in table.box[r]) for r in keep})
        if edge not in out[-1]:
            raise Refused("a sampled join does not match its tracks file")
    return out[0], out[1]


def display_size(width: int, height: int, max_w: int = MAX_DISPLAY_W) -> "tuple[int, int]":
    return lm.display_size(lm.VideoMeta(width, height, 1.0, 1, 1.0), max_w)


def draw_box(rgb: np.ndarray, box, scale: float, colour, px: int = BOX_PX) -> None:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = (int(round(v * scale)) for v in box)
    x0, x1 = sorted((min(max(x0, 0), w - 1), min(max(x1, 0), w - 1)))
    y0, y1 = sorted((min(max(y0, 0), h - 1), min(max(y1, 0), h - 1)))
    c = np.asarray(colour, np.uint8)
    rgb[y0:min(y0 + px, y1 + 1), x0:x1 + 1] = c
    rgb[max(y1 - px + 1, y0):y1 + 1, x0:x1 + 1] = c
    rgb[y0:y1 + 1, x0:min(x0 + px, x1 + 1)] = c
    rgb[y0:y1 + 1, max(x1 - px + 1, x0):x1 + 1] = c


def build_clip(k: int, buf: bytes, n: int, start_f: int, join: Join, b1: dict, b2: dict, src_w: int, w: int,
               h: int, fps: float, step: int) -> JoinClip:
    """From ffmpeg's raw output (`buf`, n frames from start_f; viewed, not copied): draw the two boxes (cyan on
    fragment 1's frames, yellow on fragment 2's, none in the gap and no other box ever), keep every `step`-th
    frame plus both boundary frames, time each kept frame by the source frames it stands for (so playback is
    real speed even where a boundary frame falls off the step grid), and build the side-by-side still."""
    view = np.frombuffer(buf, np.uint8, count=n * h * w * 3).reshape(n, h, w, 3)
    scale = w / src_w
    keep_f, out, part = [], [], []
    for i in range(n):
        f = start_f + i
        if not ((f - start_f) % step == 0 or f in (join.last1, join.first2)):
            continue
        a = view[i].copy()
        if f <= join.last1:
            if f in b1:
                draw_box(a, b1[f], scale, CYAN)
            part.append(1)
        elif f >= join.first2:
            if f in b2:
                draw_box(a, b2[f], scale, YELLOW)
            part.append(2)
        else:
            part.append(0)
        keep_f.append(f)
        out.append(a)
    if join.last1 not in keep_f or join.first2 not in keep_f:
        raise Refused("the clip does not reach both boundary frames")
    sep = np.zeros((h, STILL_SEP, 3), np.uint8)
    still = np.concatenate([out[keep_f.index(join.last1)], sep, out[keep_f.index(join.first2)]], axis=1)
    ms = tuple(max(1, int(round((b - a) * 1000 / fps))) for a, b in zip(keep_f, keep_f[1:] + [start_f + n]))
    return JoinClip(k, w, h, fps, start_f, tuple(a.tobytes() for a in out), tuple(part), ms,
                    still.tobytes(), 2 * w + STILL_SEP)


def clip_range(join: Join, fps: float, n_frames: int) -> "tuple[int, int]":
    e = max(1, int(round(EDGE_S * fps)))
    return max(0, join.last1 - e + 1), min(n_frames - 1, join.first2 + e - 1)


def step_for(fps: float) -> int:
    """Every other frame above 30 fps: a 4 s clip at 50 fps is 200 frames, and the window keeps two in memory."""
    return 2 if fps > 30 else 1


# ---- the score -------------------------------------------------------------------------------------
def verdict(decided: int, unsure: int, total: int, lo: float, hi: float, ps: float) -> str:
    if total == 0 or unsure / total > MAX_UNSURE_SHARE or decided < MIN_DECIDED:
        return "not established"
    if hi <= ps:
        return "measured, acceptable"
    if lo > ps:
        return "measured, unreliable"
    return "measured, inconclusive"


def score(sample: Sample, answers: AnswerFile, draws: int = BOOT_DRAWS, seed: int = SEED) -> dict:
    """The gate's metric, computed once from a complete answers file."""
    if answers.sample_sha != sample.sha256 or len(answers.values) != sample.n:
        raise Refused("the answers do not belong to this sample")
    if answers.first_empty() is not None:
        raise Refused("not every join is answered: the score is computed once, from a complete file")
    vals = answers.values
    c = answers.counts()
    S, D, U = c["S"], c["D"], c["U"]
    decided = S + D
    videos = sample.raw["videos"]
    positions = [p for v in sorted(videos) for p in videos[v]["positions"]]
    ps = float(sample.raw["frame"]["p_star"])
    out = {"answers": {"S": S, "D": D, "U": U, "total": sample.n},
           "unsure_share": U / sample.n, "decided": decided, "p_star": ps,
           "minutes": answers.prior_minutes()}
    if decided == 0:
        out |= {"join_error": None, "wilson95": None, "verdict": "not established"}
        return out
    rate = D / decided
    lo, hi = wilson(D, decided)
    out |= {"join_error": rate, "wilson95": [lo, hi],
            "verdict": verdict(decided, U, sample.n, lo, hi, ps),
            "misattribution": {"at_rate": misattribution(rate, positions), "at_lower": misattribution(lo, positions),
                               "at_upper": misattribution(hi, positions),
                               "assumptions": "independent joins; a wrong join switches between two people; "
                                              "measured against the chain's first person; weighted by credited punches"}}

    def part(sel):
        d = sum(1 for i in sel if vals[i] == "D")
        s = sum(1 for i in sel if vals[i] == "S")
        u = sum(1 for i in sel if vals[i] == "U")
        r = {"S": s, "D": d, "U": u}
        if s + d:
            r |= {"join_error": d / (s + d), "wilson95": list(wilson(d, s + d))}
        return r

    idx = range(sample.n)
    med = float(sample.raw["frame"]["median_gap"])
    out["by_stratum"] = {st: part([i for i in idx if stratum(sample.joins[i], med) == st]) for st in ("short", "long")}
    cams = sorted({videos[v]["camera"] for v in videos})
    out["by_camera"] = {}
    for cam in cams:
        r = part([i for i in idx if videos[sample.joins[i].video]["camera"] == cam])
        if r["S"] + r["D"] < 20:
            r["note"] = "not established (under 20 decided)"
        out["by_camera"][cam] = r
    J = sum(videos[v]["joins"] for v in videos)
    T = sum(videos[v]["chain_minutes"] for v in videos)
    per_min = rate * J / T if T > 0 else None
    out["wrong_joins_per_chain_minute"] = per_min
    out["bootstrap"] = bootstrap_per_minute(sample, vals, draws, seed)
    return out


def bootstrap_per_minute(sample: Sample, vals, draws: int = BOOT_DRAWS, seed: int = SEED) -> dict:
    """Video-cluster bootstrap of wrong joins per chain-minute: resample the manifest's videos with
    replacement; a draw with no decided sampled join is dropped and counted."""
    videos = sample.raw["videos"]
    vids = sorted(videos)
    dv = {v: 0 for v in vids}
    nv = {v: 0 for v in vids}
    for j, a in zip(sample.joins, vals):
        if a in ("S", "D"):
            nv[j.video] += 1
            dv[j.video] += a == "D"
    rng = np.random.default_rng(seed)
    D = np.array([dv[v] for v in vids], float)
    N = np.array([nv[v] for v in vids], float)
    J = np.array([videos[v]["joins"] for v in vids], float)
    T = np.array([videos[v]["chain_minutes"] for v in vids], float)
    stats, dropped = [], 0
    for _ in range(draws):
        pick = rng.integers(0, len(vids), len(vids))
        n, t = N[pick].sum(), T[pick].sum()
        if n == 0 or t <= 0:
            dropped += 1
            continue
        stats.append(D[pick].sum() / n * J[pick].sum() / t)
    out = {"draws": draws, "dropped": dropped, "seed": seed}
    if dropped > MAX_BOOT_DROPPED * draws or not stats:
        out["interval95"] = None
        out["note"] = "more than 1% of draws had no decided join: no interval reported"
    else:
        out["interval95"] = [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]
    return out
