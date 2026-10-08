"""Link the fragments of one video's ByteTrack ids into TWO boxer identities.

Why: ByteTrack splits each boxer into hundreds of short ids (median 541 ids per
Olympic Boxing video; two boxers, a referee, judges, coaches, spectators). A
per-boxer measurement -- the completeness test of the referee's labels -- needs
"boxer A in frame f" and "boxer B in frame f", not ids. ADR-0012: link first.

The model, in one paragraph. A FRAGMENT is one track id. A candidate link a -> b
is allowed only FORWARD in time (b starts after a ends, within MAX_GAP_S) and only
if the jump between a's last and b's first position is something a person can do
in that gap (MAX_SPEED_BL_S body lengths per second, plus JUMP_SLACK_BL). A
fragment is worth including by its EVIDENCE: labelled punches credited to it, and
the share of its frames inside the ring (a region read off where credited gloves
are). At most two vertex-disjoint chains of fragments, EACH with at least one credited
punch, that maximise evidence minus link costs are found exactly, as an integer
program (scipy's HiGHS `milp`) over a two-state layered path model: a chain is in
state 0 until it has passed a fragment with a credited punch and in state 1 after,
and only state 1 may end. The punch requirement is part of the optimisation, so a
punch-free chain (a referee) can never take a slot from a boxer. A solution that is
not proven optimal within the time limit is an error, never an answer. Fragments no
chain wants are left out.

What it cannot do (residuals, tested or reported rather than hidden):
  * It has no appearance model. Where two people cross and the id is lost during
    the crossing, the continuation nearest in space wins, which can be the OTHER
    person: an identity SWAP. tests/test_linking.py asserts this behaviour.
  * The referee is in the ring too. Only the punch evidence tells a referee from
    a boxer, so a referee fragment next to a boxer's gap can be absorbed (a chain needs
    at least one credited punch, so a referee alone is never an identity).
  * Punch evidence is also what decides which chains win, so "most credited punches
    sit inside the two identities" is partly true BY CONSTRUCTION; olympic_link.py
    therefore also reports it with half the punches held out of the linking.

Every constant below was fixed from the physics of the footage and the tracker's
own configuration BEFORE any linking result was looked at; none was tuned to a
real-data number. Where one changes later, docs/quality-loop.md says so.
"""

from __future__ import annotations

import bisect
import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .pose_yolo import PoseTable
from .punchfind import L_HIP, L_SHOULDER, MIN_CONF, R_HIP, R_SHOULDER

# ---- link plausibility ------------------------------------------------------------
# 2 s = ByteTrack's own track buffer (config/bytetrack_50fps.yaml, 100 frames at 50 fps).
# The tracker already re-attaches a lost person that reappears within that window, so
# what reaches us are the cases it gave up on; beyond 2 s the box could be anywhere in
# the ring and "same person" is a guess.
MAX_GAP_S = 2.0
# 5 torso lengths per second: a torso (shoulder-mid to hip-mid) is ~0.5 m, so ~2.5 m/s
# averaged over the whole gap. Boxing footwork is ~1-2 m/s and a lunge is brief; this
# lets a boxer cover real ground yet rejects a hop to a different person across the ring.
MAX_SPEED_BL_S = 5.0
# One torso of slack on top of the speed limit: a box centre moves by up to half an arm
# length when a punch is thrown (the box spans the extended arm), and a re-detected
# person's box is not drawn identically to the one that was lost.
JUMP_SLACK_BL = 1.0
# RE-ACQUISITION (added after the first real-data run, see docs/quality-loop.md). A chain is
# one path, so a stretch longer than MAX_GAP_S in which the boxer is not tracked (hidden behind
# the referee, out of frame) ends it for good and everything later is lost. A link across a longer
# gap is allowed up to this many seconds, with NO speed limit (after 2 s a boxer can be
# anywhere in a ring ~12 torsos wide) but a flat price, so it is used only when it pays for
# itself in punches. 0 switches it off: the model exactly as first specified. The price
# is the cost of being less sure it is the same person; the identity labels A and B are then
# only reliable between re-acquisitions, because nothing here recognises a face or a shirt.
REACQUIRE_S = 0.0
REACQUIRE_ON_S = 60.0     # the value olympic_link.py uses with --reacquire: one round's rest
REACQUIRE_POINTS = 4.0
REACQUIRE_PER_S = 0.05    # a longer wait is slightly less certain: +3 points at 60 s
# Fragment ends are read as the MEDIAN of the first/last 0.1 s of boxes (5 frames at
# 50 fps): robust to a single flickering box, short enough that a mover has not moved.
EDGE_S = 0.1

# ---- what counts as a fragment worth considering -----------------------------------
# 0.2 s = 10 frames at 50 fps: a labelled punch lasts ~0.08 s, so a fragment shorter
# than 2.5 punches cannot carry evidence worth a link, and most are detector flicker.
MIN_FRAGMENT_S = 0.2
# Fewer confident torsos than this and the body scale falls back to the box height.
MIN_TORSO_ROWS = 3
# Shoulder-mid to hip-mid is ~0.30 of a standing person's height (0.5 m of 1.75 m); a
# boxer in stance with raised arms has a taller-than-wide box, so 0.3 is not far off.
TORSO_PER_BOX_HEIGHT = 0.3
# Below this a "body" is a few pixels: not a person to link (also keeps divisions safe).
MIN_SCALE_PX = 4.0

# ---- the ring prior ----------------------------------------------------------------
# The ring is where credited gloves are. The 1st-99th percentile of their centres,
# per axis, ignores the rare mislabelled or mis-credited box; ...
RING_QUANTILES = (0.01, 0.99)
# ... and the region is widened by two torso lengths: a glove sits about an arm's reach
# (~1.4 torsos) in front of the body, and the box centre is half a torso behind the
# shoulder, so a boxer standing at the edge of the punch region is ~2 torsos outside it.
# Spectators, judges and coaches sit metres beyond the ropes.
RING_MARGIN_BL = 2.0
# A fragment must spend at least half its frames inside the region to count as in the ring.
RING_MIN_SHARE = 0.5
# Fewer credited punches than this and the region is a guess: refuse, do not invent one.
MIN_RING_PUNCHES = 10

# ---- scoring (evidence points; a labelled punch is the unit of account) ------------
# One credited punch = 10 points; one second inside the ring = 0.5. A boxer fragment
# with even a single punch therefore outweighs 20 s of punch-less ring time, so a
# referee (in the ring, throwing nothing) cannot outbid a boxer who threw. Ring time is
# worth little on purpose: it only has to pay for a link to a punch-less fragment in
# between, because a wrong inclusion puts someone else's arm movements into a boxer's
# track, which is worse than a hole.
PUNCH_POINTS = 10.0
RING_POINTS_PER_S = 0.5
# Every link costs 1 point (a chain should not hop through fragments for nothing); a
# link at the plausibility limit costs up to 3 more for the jump and 1 for the gap
# (a longer gap leaves more room for someone else to have taken the place); 2 points
# per unit of |ln(scale ratio)| penalises joining a near person to a far one.
LINK_POINTS = 1.0
JUMP_POINTS = 3.0
GAP_POINTS = 1.0
SCALE_POINTS = 2.0
# A chain with no credited punch is not a boxer, whatever its ring time: the referee is in
# the ring too and throws nothing. The requirement is enforced INSIDE the optimisation (a punch-free
# chain may not take a slot a boxer could have used). The layered model encodes exactly ">= 1"
# (state 0 = no punch yet, state 1 = at least one), so this constant must stay 1.
MIN_CHAIN_PUNCHES = 1
MAX_CHAINS = 2
# The solver is exact (HiGHS branch and bound, zero relative gap) or it fails: past this many
# seconds the answer is an error, not the best solution found so far.
MILP_TIME_LIMIT_S = 300.0
# Opening a chain costs 8 points: more than the dearest plausible link (1 + 3 + 1, plus up to
# 3 for a 4x scale mismatch), so one boxer is never cut in two just to save a link, and less
# than one credited punch (10), so a second boxer who threw even one still opens a chain.
# Without it the second chain is free, and the cheapest use of a free chain is to split
# the only boxer found (found by the first test of a one-boxer scene).
CHAIN_POINTS = 8.0
# Rewards and costs are rounded to 1/1000 of a point before solving, so every objective value is an
# integer number of those units and the solver's absolute gap tolerance (1e-6) can never hide a
# difference between two solutions; 1/1000 of a point is far below anything meaningful.
COST_SCALE = 1000


def _finite(name: str, v, lo: float = 0.0, strict: bool = False) -> None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    if v < lo or (strict and v <= lo):
        raise ValueError(f"{name} must be {'>' if strict else '>='} {lo}, got {v!r}")


@dataclass(frozen=True)
class LinkParams:
    max_gap_s: float = MAX_GAP_S
    reacquire_s: float = REACQUIRE_S
    reacquire_points: float = REACQUIRE_POINTS
    reacquire_per_s: float = REACQUIRE_PER_S
    max_speed_bl_s: float = MAX_SPEED_BL_S
    jump_slack_bl: float = JUMP_SLACK_BL
    edge_s: float = EDGE_S
    min_fragment_s: float = MIN_FRAGMENT_S
    torso_per_box_height: float = TORSO_PER_BOX_HEIGHT
    ring_quantiles: tuple = RING_QUANTILES
    ring_margin_bl: float = RING_MARGIN_BL
    ring_min_share: float = RING_MIN_SHARE
    min_ring_punches: int = MIN_RING_PUNCHES
    punch_points: float = PUNCH_POINTS
    ring_points_per_s: float = RING_POINTS_PER_S
    link_points: float = LINK_POINTS
    jump_points: float = JUMP_POINTS
    gap_points: float = GAP_POINTS
    scale_points: float = SCALE_POINTS
    chain_points: float = CHAIN_POINTS

    def __post_init__(self) -> None:
        _finite("max_gap_s", self.max_gap_s, 0.0, strict=True)
        _finite("reacquire_s", self.reacquire_s)
        _finite("reacquire_points", self.reacquire_points)
        _finite("reacquire_per_s", self.reacquire_per_s)
        _finite("max_speed_bl_s", self.max_speed_bl_s, 0.0, strict=True)
        _finite("jump_slack_bl", self.jump_slack_bl)
        _finite("edge_s", self.edge_s, 0.0, strict=True)
        _finite("min_fragment_s", self.min_fragment_s)
        _finite("torso_per_box_height", self.torso_per_box_height, 0.0, strict=True)
        _finite("ring_margin_bl", self.ring_margin_bl)
        _finite("ring_min_share", self.ring_min_share)
        if self.ring_min_share > 1.0:
            raise ValueError(f"ring_min_share must be in [0, 1], got {self.ring_min_share!r}")
        q = self.ring_quantiles
        if (not isinstance(q, tuple) or len(q) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x)
                                                               for x in q) or not 0.0 <= q[0] < q[1] <= 1.0):
            raise ValueError(f"ring_quantiles must be (lo, hi) with 0 <= lo < hi <= 1, got {q!r}")
        if isinstance(self.min_ring_punches, bool) or not isinstance(self.min_ring_punches, int) \
                or self.min_ring_punches < 1:
            raise ValueError(f"min_ring_punches must be a whole number >= 1, got {self.min_ring_punches!r}")
        for n in ("punch_points", "ring_points_per_s", "link_points", "jump_points", "gap_points", "scale_points",
                  "chain_points"):
            _finite(n, getattr(self, n))


@dataclass(frozen=True)
class Region:
    """Axis-aligned pixel rectangle: where the ring is, as far as glove boxes say."""

    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x, y):
        return (x >= self.x0) & (x <= self.x1) & (y >= self.y0) & (y <= self.y1)


@dataclass(frozen=True)
class Fragment:
    """One track id, summarised."""

    tid: int
    first: int            # first / last video frame it is seen on
    last: int
    n_rows: int           # frames it is actually present on (<= last - first + 1)
    start_xy: tuple       # box centre over its first EDGE_S, pixels
    end_xy: tuple         # ... and over its last EDGE_S
    scale: float          # body length (shoulder-mid to hip-mid), pixels; NaN when unusable
    scale_from_joints: bool
    ring_share: float = 0.0
    punches: int = 0

    @property
    def span(self) -> int:
        return self.last - self.first + 1


def _groups(table: PoseTable):
    """Row indices of every track id, in frame order: {tid: rows}."""
    order = np.lexsort((table.frame, table.track_id))
    tids = table.track_id[order]
    cut = np.flatnonzero(np.diff(tids)) + 1
    return {int(tids[s]): order[s:e] for s, e in zip(np.r_[0, cut], np.r_[cut, len(order)])} if len(order) else {}


def summarise_fragments(table: PoseTable, params: LinkParams = LinkParams()) -> "tuple[list[Fragment], dict]":
    """One Fragment per track id (ring share and punches still zero), and the rows of each."""
    rows_of = _groups(table)
    edge = max(1, int(round(params.edge_s * table.fps)))
    cx = (table.box[:, 0].astype(np.float64) + table.box[:, 2]) / 2
    cy = (table.box[:, 1].astype(np.float64) + table.box[:, 3]) / 2
    out = []
    for tid in sorted(rows_of):
        r = rows_of[tid]
        c = np.stack([cx[r], cy[r]], axis=1)
        fr = table.frame[r]
        ok = (table.conf[r][:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]] >= MIN_CONF).all(axis=1)
        from_joints = bool(ok.sum() >= MIN_TORSO_ROWS)
        if from_joints:
            xy = table.xy[r][ok].astype(np.float64)
            sh = (xy[:, L_SHOULDER] + xy[:, R_SHOULDER]) / 2
            hp = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
            scale = float(np.median(np.linalg.norm(sh - hp, axis=1)))
            if not scale >= MIN_SCALE_PX:      # a degenerate torso: use the box instead
                from_joints = False
        if not from_joints:
            h = table.box[r, 3].astype(np.float64) - table.box[r, 1]
            scale = float(np.median(h)) * params.torso_per_box_height
        if not scale >= MIN_SCALE_PX:
            scale = float("nan")
        out.append(Fragment(tid, int(fr[0]), int(fr[-1]), len(r),
                            tuple(float(v) for v in np.median(c[:edge], axis=0)),
                            tuple(float(v) for v in np.median(c[-edge:], axis=0)),
                            scale, from_joints))
    return out, rows_of


def estimate_region(boxes, scale_px: float, params: LinkParams = LinkParams()) -> Region:
    """The ring: the central part of where credited glove boxes are, widened by a margin.
    `boxes` is (n, 4) xyxy of glove boxes; fewer than `min_ring_punches` is refused."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(b) < params.min_ring_punches:
        raise ValueError(f"{len(b)} credited glove boxes: too few to estimate the ring "
                         f"(need {params.min_ring_punches})")
    if not np.isfinite(b).all() or not (np.isfinite(scale_px) and scale_px > 0):
        raise ValueError("glove boxes and body scale must be finite and positive")
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    lo, hi = params.ring_quantiles
    m = params.ring_margin_bl * scale_px
    return Region(float(np.quantile(cx, lo)) - m, float(np.quantile(cy, lo)) - m,
                  float(np.quantile(cx, hi)) + m, float(np.quantile(cy, hi)) + m)


@dataclass(frozen=True)
class Link:
    a: int            # track id the chain leaves
    b: int            # track id it continues with
    gap: int          # frames between a's last and b's first (>= 1)
    jump_bl: float    # distance between a's end and b's start, in body lengths
    cost: float       # points


def plausible_jump(jump_bl: float, gap_frames: int, fps: float, params: LinkParams = LinkParams()) -> bool:
    """Can a person cover `jump_bl` body lengths in `gap_frames`? Speed limit plus slack."""
    return jump_bl <= params.jump_slack_bl + params.max_speed_bl_s * gap_frames / fps


def candidate_links(frags: "list[Fragment]", fps: float, params: LinkParams = LinkParams()) -> "list[Link]":
    """Every plausible a -> b, FORWARD in time only. b must start strictly after a ends
    (so two fragments that overlap in time can never be linked, and a chain's fragments
    are pairwise time-disjoint: the first of one starts after the last of the one before),
    within `max_gap_s`, with a jump a person could make in that gap. With `reacquire_s` >
    `max_gap_s`, longer gaps up to that are also linked, with no speed limit and a flat price."""
    max_gap = max(1, int(round(params.max_gap_s * fps)))
    reach = max(max_gap, int(round(params.reacquire_s * fps)))
    by_first = sorted(frags, key=lambda f: (f.first, f.tid))
    firsts = [f.first for f in by_first]
    links = []
    for a in frags:
        if not a.scale > 0:
            continue
        lo = bisect.bisect_left(firsts, a.last + 1)
        hi = bisect.bisect_right(firsts, a.last + reach)
        for b in by_first[lo:hi]:
            gap = b.first - a.last
            if gap < 1 or gap > reach or b.tid == a.tid or not b.scale > 0:
                continue
            scale = (a.scale + b.scale) / 2
            jump = math.hypot(b.start_xy[0] - a.end_xy[0], b.start_xy[1] - a.end_xy[1]) / scale
            allowed_max = params.jump_slack_bl + params.max_speed_bl_s * params.max_gap_s
            if gap <= max_gap:
                if not plausible_jump(jump, gap, fps, params):
                    continue
                allowed = params.jump_slack_bl + params.max_speed_bl_s * gap / fps
                cost = (params.link_points + params.jump_points * jump / allowed
                        + params.gap_points * gap / max_gap
                        + params.scale_points * abs(math.log(a.scale / b.scale)))
            else:      # re-acquisition: unobserved longer than the buffer, so anywhere is reachable
                cost = (params.link_points + params.reacquire_points
                        + params.reacquire_per_s * (gap - max_gap) / fps
                        + params.jump_points * min(jump / allowed_max, 1.0) + params.gap_points
                        + params.scale_points * abs(math.log(a.scale / b.scale)))
            links.append(Link(a.tid, b.tid, int(gap), float(jump), float(cost)))
    return links


def eligible(f: Fragment, fps: float, params: LinkParams = LinkParams()) -> bool:
    """Long enough, a usable body, and in the ring for at least `ring_min_share` of its frames."""
    return (f.n_rows >= max(2, int(round(params.min_fragment_s * fps))) and f.scale > 0
            and f.ring_share >= params.ring_min_share)


def reward(f: Fragment, fps: float, params: LinkParams = LinkParams()) -> float:
    """Evidence that this fragment is a boxer: credited punches, and seconds in the ring."""
    return params.punch_points * f.punches + params.ring_points_per_s * f.ring_share * f.n_rows / fps


@dataclass(frozen=True)
class Linking:
    """The two boxer identities of one video.

    `identities[k]` is identity k's fragments (track ids) in time order, empty when
    nothing qualified. `membership[f, k]` is the track id that is identity k in
    frame f, or -1: the identity is simply not identified there (a gap, or a
    fragment left out). Identity 0 is the one with more credited punches (A), only
    so that the labels are reproducible: which boxer is "A" means nothing.
    """

    identities: tuple
    membership: np.ndarray
    n_frames: int
    fps: float
    source: str
    region: "Region | None"
    fragments: tuple = field(default=(), repr=False)
    objective: float = 0.0
    solve_s: float = field(default=0.0, compare=False)     # wall time of the integer program

    def ids(self, k: int) -> frozenset:
        return frozenset(self.identities[k])

    def tid_at(self, frame: int, k: int) -> "int | None":
        t = int(self.membership[frame, k])
        return None if t < 0 else t

    def identified(self, k: int) -> np.ndarray:
        return self.membership[:, k] >= 0


def _units(x: float) -> int:
    return int(round(x * COST_SCALE))


def solve_chains(usable: "list[Fragment]", links: "list[Link]", fps: float,
                 params: LinkParams = LinkParams(), time_limit_s: float = MILP_TIME_LIMIT_S
                 ) -> "tuple[list[tuple], float]":
    """The best set of at most two vertex-disjoint forward chains, each holding a credited punch.

    Returns (chains as tuples of track ids, objective in points), objective = the sum of the
    fragment rewards in the chains - `chain_points` per chain - the cost of every link used
    (all rounded to 1/COST_SCALE point), or no chains and 0 when nothing is worth including.

    A two-state layered path model on the forward-only DAG. Binary variables: start[f];
    x[f, s] (f is used, the chain's state AFTER f is s; s = 1 means "the chain already holds a
    punch"); y[l, s] for each candidate link l = a -> b carrying state s (the state after a);
    end[f], allowed only from state 1. A fragment with punches exists only in state 1; a
    punch-free one in both. Flow in = flow out at every node, x[f,0] + x[f,1] <= 1 (a fragment
    is in at most one chain and counted once), sum start <= 2. Exact: any status but "optimal"
    raises RuntimeError.
    """
    if MIN_CHAIN_PUNCHES != 1:
        raise RuntimeError("the layered model encodes exactly 'at least one punch'; MIN_CHAIN_PUNCHES must be 1")
    if not usable:
        return [], 0.0
    idx = {f.tid: i for i, f in enumerate(usable)}
    if len(idx) != len(usable):
        raise ValueError("duplicate fragment ids")
    punchy = [f.punches > 0 for f in usable]
    nvar = 0
    col: "dict[tuple, int]" = {}
    obj: "list[int]" = []

    def var(key, cost: int) -> None:
        nonlocal nvar
        col[key] = nvar
        obj.append(cost)
        nvar += 1

    chain_u = _units(params.chain_points)
    for i, f in enumerate(usable):
        r = _units(reward(f, fps, params))
        var(("start", i), chain_u)
        if not punchy[i]:
            var(("x", i, 0), -r)       # the reward is counted once, in whichever state holds the fragment
            var(("x", i, 1), -r)
        else:
            var(("x", i, 1), -r)
        var(("end", i), 0)
    lk = []
    for l in links:
        if l.a not in idx or l.b not in idx:
            raise ValueError(f"link {l.a} -> {l.b} names a fragment that is not usable")
        ia, ib = idx[l.a], idx[l.b]
        c = _units(l.cost)
        for s in ((1,) if punchy[ia] else (0, 1)):
            var(("y", ia, ib, s), c)
        lk.append((ia, ib))

    inn: "dict[tuple, list]" = {}      # (fragment, state carried by the incoming link) -> columns
    out: "dict[tuple, list]" = {}      # (fragment, state after it) -> columns
    for (ia, ib) in lk:
        for s in (0, 1):
            k = col.get(("y", ia, ib, s))
            if k is not None:
                out.setdefault((ia, s), []).append(k)
                inn.setdefault((ib, s), []).append(k)

    rows_i, cols_i, vals = [], [], []
    nrow = 0

    def eq(terms) -> None:
        nonlocal nrow
        for k, v in terms:
            rows_i.append(nrow)
            cols_i.append(k)
            vals.append(v)
        nrow += 1

    for i in range(len(usable)):
        st, en = col[("start", i)], col[("end", i)]
        if punchy[i]:
            x1 = col[("x", i, 1)]
            eq([(x1, 1), (st, -1)] + [(k, -1) for s in (0, 1) for k in inn.get((i, s), [])])
            eq([(x1, 1), (en, -1)] + [(k, -1) for k in out.get((i, 1), [])])
        else:
            x0, x1 = col[("x", i, 0)], col[("x", i, 1)]
            eq([(x0, 1), (st, -1)] + [(k, -1) for k in inn.get((i, 0), [])])
            eq([(x1, 1)] + [(k, -1) for k in inn.get((i, 1), [])])
            eq([(x0, 1)] + [(k, -1) for k in out.get((i, 0), [])])
            eq([(x1, 1), (en, -1)] + [(k, -1) for k in out.get((i, 1), [])])
    n_eq = nrow
    cap_rows = [[(col[("x", i, 0)], 1), (col[("x", i, 1)], 1)] for i in range(len(usable)) if not punchy[i]]
    ub = [1.0] * len(cap_rows)
    cap_rows.append([(col[("start", i)], 1) for i in range(len(usable))])      # sum start <= 2
    ub.append(float(MAX_CHAINS))
    ri, ci, vv = [], [], []
    for r_, terms in enumerate(cap_rows):
        for k, v in terms:
            ri.append(r_)
            ci.append(k)
            vv.append(v)
    a_eq = coo_matrix((vals, (rows_i, cols_i)), shape=(n_eq, nvar)).tocsr()
    a_ub = coo_matrix((vv, (ri, ci)), shape=(len(cap_rows), nvar)).tocsr()
    t0 = time.perf_counter()
    res = milp(c=np.asarray(obj, dtype=np.float64),
               constraints=[LinearConstraint(a_eq, 0.0, 0.0), LinearConstraint(a_ub, -np.inf, np.asarray(ub))],
               integrality=np.ones(nvar), bounds=Bounds(0.0, 1.0),
               options={"time_limit": float(time_limit_s), "mip_rel_gap": 0.0, "disp": False})
    solve_s = time.perf_counter() - t0
    if res.status != 0 or not res.success or res.x is None:
        raise RuntimeError(f"the chain optimisation was not solved to proven optimality "
                           f"(status {res.status}: {res.message}) after {solve_s:.1f} s")
    x = np.rint(res.x).astype(np.int64)
    if np.abs(res.x - x).max() > 1e-4:
        raise RuntimeError("the chain optimisation returned a fractional solution")
    nxt: "dict[int, int]" = {}
    link_units = 0
    for key, k in col.items():
        if key[0] == "y" and x[k] == 1:
            if key[1] in nxt:
                raise RuntimeError("the solution leaves one fragment twice")
            nxt[key[1]] = key[2]
            link_units += obj[k]
    starts = [i for i in range(len(usable)) if x[col[("start", i)]] == 1]
    if len(starts) > MAX_CHAINS:
        raise RuntimeError("the solution opens more than two chains")
    chains, seen = [], set()
    total_units = -len(starts) * chain_u - link_units
    for i0 in starts:
        chain, cur = [i0], i0
        while cur in nxt:
            cur = nxt[cur]
            if cur in chain or cur in seen:       # cannot happen on a forward-only graph
                raise RuntimeError("the solution contains a cycle or shares a fragment")
            chain.append(cur)
        seen.update(chain)
        if sum(usable[j].punches for j in chain) < MIN_CHAIN_PUNCHES:
            raise RuntimeError("the solution holds a chain with no credited punch")
        total_units += sum(_units(reward(usable[j], fps, params)) for j in chain)
        chains.append(tuple(usable[j].tid for j in chain))
    if abs(total_units + res.fun) > 0.5:       # res.fun = -(objective in units), up to solver tolerance
        raise RuntimeError(f"the extracted chains are worth {total_units / COST_SCALE} but the solver reports "
                           f"{-res.fun / COST_SCALE}")
    return chains, total_units / COST_SCALE


def link(table: PoseTable, credited: "dict[int, int]", region: Region,
         params: LinkParams = LinkParams()) -> Linking:
    """Two vertex-disjoint chains of fragments with the most evidence.

    `credited` maps a track id to the number of labelled punches credited to it.
    """
    if not isinstance(params, LinkParams):
        raise ValueError("params must be a LinkParams")
    if not isinstance(region, Region):
        raise ValueError("region must be a Region")
    for t, n in credited.items():
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError(f"punch count for id {t!r} must be a whole number >= 0, got {n!r}")
    frags, rows_of = summarise_fragments(table, params)
    stale = sorted(int(t) for t in credited if int(t) not in rows_of)
    if stale:
        raise ValueError(f"punches are credited to ids {stale[:5]} that are not in this table: "
                         "attributions from another video?")
    cx = (table.box[:, 0].astype(np.float64) + table.box[:, 2]) / 2
    cy = (table.box[:, 1].astype(np.float64) + table.box[:, 3]) / 2
    inside = region.contains(cx, cy)
    full = []
    for f in frags:
        r = rows_of[f.tid]
        full.append(Fragment(f.tid, f.first, f.last, f.n_rows, f.start_xy, f.end_xy, f.scale, f.scale_from_joints,
                             float(inside[r].mean()), int(credited.get(f.tid, 0))))
    fps = table.fps
    usable = [f for f in full if eligible(f, fps, params) and reward(f, fps, params) > 0]
    by_tid = {f.tid: f for f in usable}
    links = candidate_links(usable, fps, params)

    t0 = time.perf_counter()
    chains, objective = solve_chains(usable, links, fps, params)
    solve_s = time.perf_counter() - t0
    while len(chains) < MAX_CHAINS:
        chains.append(())

    def key(c):   # more punches first, then earlier start, then lower id: reproducible labels
        if not c:
            return (1, 0, 0, 0)
        return (0, -sum(by_tid[t].punches for t in c), by_tid[c[0]].first, c[0])

    chains.sort(key=key)
    membership = np.full((table.n_frames, 2), -1, dtype=np.int64)
    for k, chain in enumerate(chains):
        for t in chain:
            membership[table.frame[rows_of[t]], k] = t
    return Linking(tuple(chains), membership, table.n_frames, table.fps, table.source, region,
                   tuple(full), objective, solve_s)


def credited_counts(attrs) -> "dict[int, int]":
    out: "dict[int, int]" = {}
    for a in attrs:
        if a.track_id is not None:
            out[int(a.track_id)] = out.get(int(a.track_id), 0) + 1
    return out


def link_video(table: PoseTable, attrs, params: LinkParams = LinkParams()) -> Linking:
    """Link one video from its attributions (`olympic.attribute` output). The ring is
    estimated from the glove boxes of the CREDITED punches, in this video."""
    frags, _ = summarise_fragments(table, params)
    credited = credited_counts(attrs)
    boxes = []
    for a in attrs:
        if a.track_id is not None:
            e = a.event
            boxes += [b for _, b in (e.keyboxes or ((e.first, e.box),))]
    ref = [f.scale for f in frags if f.tid in credited and f.scale > 0] or [f.scale for f in frags if f.scale > 0]
    if not ref:
        raise ValueError("no fragment has a usable body: nothing to link")
    region = estimate_region(boxes, float(np.median(ref)), params)
    return link(table, credited, region, params)


def _longest_false(mask: np.ndarray) -> int:
    if mask.all():
        return 0
    d = np.diff(np.r_[0, (~mask).astype(int), 0])
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def summarise(lk: Linking, attrs) -> dict:
    """The numbers that say whether a linking is any good (see olympic_link.py)."""
    n = lk.n_frames
    a, b = lk.identified(0), lk.identified(1)
    both, either = a & b, a | b
    credited = [x for x in attrs if x.track_id is not None]
    ids_a, ids_b = lk.ids(0), lk.ids(1)
    ia = sum(x.track_id in ids_a for x in credited)
    ib = sum(x.track_id in ids_b for x in credited)
    nc = max(len(credited), 1)
    return {
        "frames": n,
        "identified_a": round(float(a.mean()), 4), "identified_b": round(float(b.mean()), 4),
        "both_identified": round(float(both.mean()), 4), "either_identified": round(float(either.mean()), 4),
        # Two real boxers are on screen together: if A and B were one person split in two,
        # they would never be identified in the same frame and this would sit near 0.
        "co_presence": round(float(both.sum() / max(min(a.sum(), b.sum()), 1)), 4),
        "credited_punches": len(credited),
        "credited_in_a": ia, "credited_in_b": ib,
        "credited_in_either_share": round((ia + ib) / nc, 4),
        "fragments_a": len(lk.identities[0]), "fragments_b": len(lk.identities[1]),
        "longest_unidentified_a_s": round(_longest_false(a) / lk.fps, 2),
        "longest_unidentified_b_s": round(_longest_false(b) / lk.fps, 2),
        "longest_neither_identified_s": round(_longest_false(either) / lk.fps, 2),
    }
