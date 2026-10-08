"""Tests for linking ByteTrack fragments into two boxer identities (linking.py), the linked
overlay, olympic_link.py, and the completeness test (completeness.py, olympic_completeness.py).

No model, no GPU, no Olympic data: every scene is planted. Two boxers are each split into
several track ids with gaps, a referee moves between them, spectators stand still outside the
ring, and the labelled punches come out of the real `olympic.attribute`, so "who are the
boxers" has a known answer.

Run:  python tests/test_linking.py
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import olympic_completeness  # noqa: E402
import olympic_link  # noqa: E402
from boxingvi import completeness, linking, olympic, overlay, punchfind, video  # noqa: E402
from boxingvi.data import N_JOINTS  # noqa: E402
from boxingvi.linking import Fragment, LinkParams, Linking, Region  # noqa: E402
from boxingvi.pose_yolo import PoseTable  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def raises(name: str, fn, pattern: str = "") -> None:
    try:
        fn()
    except ValueError as e:
        check(name, pattern in str(e), f"wrong reason: {e}")
    except Exception as e:
        check(name, False, f"raised {type(e).__name__}: {e}")
    else:
        check(name, False, "did not raise")


tmp = Path(tempfile.mkdtemp())
W, H, FPS, N = 640, 360, 50.0, 1000
TORSO = 40.0     # shoulder-mid to hip-mid, pixels: one body length in this scene


def pose(cx, cy, ext=(0.0, 0.0)):
    """A person centred at (cx, cy): torso 40 px, wrists 24 px out to each side."""
    xy = np.tile([cx, cy], (N_JOINTS, 1)).astype(float)
    xy[:, 0] += np.linspace(-6, 6, N_JOINTS)
    xy[5], xy[6] = (cx - 12, cy - 20), (cx + 12, cy - 20)
    xy[11], xy[12] = (cx - 8, cy + 20), (cx + 8, cy + 20)
    xy[9], xy[10] = (cx - 24 - ext[0], cy - 4), (cx + 24 + ext[1], cy - 4)
    return xy, np.array([cx - 28, cy - 52, cx + 28, cy + 56], float)


class Scene:
    """People with a path and the track ids that follow them; builds the PoseTable."""

    def __init__(self, n=N):
        self.n, self.people = n, {}

    def add(self, name, path, frags, bursts=()):
        self.people[name] = {"path": path, "frags": frags, "bursts": list(bursts)}
        return self

    def ext(self, name, f):
        e = [0.0, 0.0]
        for b, hand in self.people[name]["bursts"]:
            e[hand] = max(e[hand], 120.0 * max(0.0, 1 - abs(f - b) / 4))   # out and back in 8 frames
        return tuple(e)

    def wrist(self, name, f, hand):
        cx, cy = self.people[name]["path"](f)
        return pose(cx, cy, self.ext(name, f))[0][9 + hand]

    def table(self, source="GH000001.mp4", fps=FPS, conf=0.9):
        rows = []
        for name, p in self.people.items():
            for tid, f0, f1 in p["frags"]:
                for f in range(f0, f1 + 1):
                    cx, cy = p["path"](f)
                    xy, box = pose(cx, cy, self.ext(name, f))
                    rows.append((f, tid, box, xy))
        rows.sort(key=lambda r: (r[0], r[1]))
        k = len(rows)
        return PoseTable(np.array([r[0] for r in rows], np.int64), np.array([r[1] for r in rows], np.int64),
                         np.array([r[2] for r in rows], np.float32).reshape(k, 4),
                         np.array([r[3] for r in rows], np.float32).reshape(k, N_JOINTS, 2),
                         np.full((k, N_JOINTS), conf, np.float32), self.n, fps, W, H, source=source)

    def punches(self, names, per_frag=4):
        """Labelled punches on each named person's fragments: the glove box on a wrist, at
        frames where nobody else's wrist is within 30 px (so attribution is unambiguous)."""
        events = []
        for name in names:
            for tid, f0, f1 in self.people[name]["frags"]:
                got, last = 0, -999
                for f in range(f0 + 10, f1 - 10, 3):
                    if got >= per_frag or f - last < 25 or any(
                            abs(f - b) < 15 for q in self.people.values() for b, _ in q["bursts"]):
                        continue      # keep the planted labels clear of the planted wrist bursts
                    for hand in (got % 2, 1 - got % 2):
                        w = self.wrist(name, f, hand)
                        if any(np.hypot(*(self.wrist(o, f, h) - w)) < 30 for o in self.people if o != name
                               for h in (0, 1) if any(a <= f <= b for _, a, b in self.people[o]["frags"])):
                            continue
                        events.append(olympic.Event(f, f + 2, "head", ("left", "right")[hand],
                                                    (w[0] - 8, w[1] - 8, w[0] + 8, w[1] + 8), 1))
                        got, last = got + 1, f
                        break
                assert got >= 3, f"scene cannot plant punches on {name} {tid}"
        return events


def sway(cx, cy, period, phase=0.0, amp=40.0):
    return lambda f: (cx + amp * np.sin(2 * np.pi * f / period + phase), cy)


A_FR = [(1, 0, 199), (2, 215, 420), (3, 431, 640), (4, 655, 999)]
B_FR = [(11, 0, 139), (12, 150, 399), (13, 420, 700), (14, 712, 999)]
REF_FR = [(21, 0, 499), (22, 520, 999)]
SPEC_FR = [(31, 0, 999)], [(32, 0, 999)]
A_IDS, B_IDS = frozenset({1, 2, 3, 4}), frozenset({11, 12, 13, 14})


def base_scene(boxer_b=True, referee=True, bursts=None):
    bursts = bursts or {}
    s = Scene().add("A", sway(230, 200, 200), A_FR, bursts.get("A", ()))
    if boxer_b:
        s.add("B", sway(420, 190, 230, 1.0), B_FR, bursts.get("B", ()))
    if referee:
        s.add("ref", lambda f: (325 + 50 * np.sin(f / 150), 215), REF_FR, bursts.get("ref", ()))
    s.add("spec1", lambda f: (40, 300), SPEC_FR[0]).add("spec2", lambda f: (600, 310), SPEC_FR[1])
    return s


def run(scene, boxers=("A", "B"), per_frag=4, params=LinkParams()):
    table = scene.table()
    events = scene.punches(boxers, per_frag)
    attrs = olympic.attribute(events, table)
    return table, events, attrs, linking.link_video(table, attrs, params)


def sets(lk):
    return {frozenset(lk.identities[0]), frozenset(lk.identities[1])}


print("a planted scene: two boxers in fragments, a referee, spectators")
sc = base_scene()
table, events, attrs, lk = run(sc)
check("the planted punches are credited to the right fragments by the real attribution",
      all(a.track_id is not None for a in attrs) and len(attrs) == 32, str([a.track_id for a in attrs]))
check("linking recovers EXACTLY the two boxers' fragment sets", sets(lk) == {frozenset(A_IDS), frozenset(B_IDS)},
      str(lk.identities))
check("neither the referee nor a spectator is in an identity",
      not ({21, 22, 31, 32} & (lk.ids(0) | lk.ids(1))))
check("the two identities share no fragment", not (lk.ids(0) & lk.ids(1)))
check("each identity's fragments are in time order, ending before the next begins",
      all(all(frag[a].last < frag[b].first for a, b in zip(ch, ch[1:]))
          for frag in [{f.tid: f for f in lk.fragments}] for ch in lk.identities))
check("identity 0 has the most credited punches (the labels are reproducible, not arbitrary)",
      sum(f.punches for f in lk.fragments if f.tid in lk.ids(0)) >= sum(f.punches for f in lk.fragments if f.tid in lk.ids(1)))
k_a = 0 if lk.ids(0) == A_IDS else 1
check("per-frame membership: inside a fragment the identity IS that track id",
      lk.tid_at(100, k_a) == 1 and lk.tid_at(300, k_a) == 2 and lk.tid_at(500, k_a) == 3 and lk.tid_at(900, k_a) == 4
      and lk.tid_at(100, 1 - k_a) == 11 and lk.tid_at(900, 1 - k_a) == 14)
check("... and in the gap between two fragments the identity is None, not carried over",
      lk.tid_at(207, k_a) is None and lk.tid_at(425, k_a) is None and lk.tid_at(145, 1 - k_a) is None)
check("a membership entry always names a track id that has a row in that frame",
      all(((table.frame == f) & (table.track_id == lk.membership[f, k])).any()
          for f in range(0, N, 37) for k in (0, 1) if lk.membership[f, k] >= 0))
st = linking.summarise(lk, attrs)
check("both identified in most frames; every credited punch is inside the two identities",
      st["both_identified"] > 0.9 and st["credited_in_either_share"] == 1.0
      and st["credited_in_a"] + st["credited_in_b"] == 32 and st["credited_in_a"] == 16, str(st))
check("co-presence is high for two real boxers", st["co_presence"] > 0.95, str(st["co_presence"]))
check("the longest unidentified stretch is the longest planted gap (A: 15 frames = 0.3 s; B: 20 frames = 0.4 s)",
      st["longest_unidentified_a_s"] == 0.3 and st["longest_unidentified_b_s"] == 0.4, str(st))
check("the ring region is estimated from the credited gloves and excludes the spectators",
      lk.region is not None and not lk.region.contains(40, 300) and not lk.region.contains(600, 310)
      and lk.region.contains(325, 200), str(lk.region))

print("fragment summaries")
frags, rows_of = linking.summarise_fragments(table)
f2 = next(f for f in frags if f.tid == 2)
check("first, last, rows, scale from the joints (the torso, 40 px), ends at the box centre",
      f2.first == 215 and f2.last == 420 and f2.n_rows == 206 and abs(f2.scale - 40.0) < 1e-6 and f2.scale_from_joints
      and abs(f2.start_xy[0] - (230 + 40 * np.sin(2 * np.pi * 215 / 200))) < 3
      and abs(f2.end_xy[1] - 202) < 1e-6, str(f2))
lowconf = sc.table(conf=0.05)
fl, _ = linking.summarise_fragments(lowconf)
f2l = next(f for f in fl if f.tid == 2)
check("with no confident torso the scale falls back to the box height (108 px x 0.3)",
      not f2l.scale_from_joints and abs(f2l.scale - 108 * 0.3) < 1e-6, str(f2l))
tiny = PoseTable(np.arange(12), np.ones(12, np.int64), np.tile([0, 0, 2, 2], (12, 1)).astype(np.float32),
                 np.zeros((12, N_JOINTS, 2), np.float32), np.zeros((12, N_JOINTS), np.float32), 20, FPS, W, H)
check("a body a couple of pixels tall has no usable scale (NaN), so it is never linked or counted",
      np.isnan(linking.summarise_fragments(tiny)[0][0].scale))

print("candidate links: forward in time only, within the gap limit, at a speed a person can go")
FP = 50.0


def fr(tid, first, last, x, y=100.0, scale=40.0):
    return Fragment(tid, first, last, last - first + 1, (x, y), (x, y), scale, True)


MAXGAP = int(round(LinkParams().max_gap_s * FP))
pairs = lambda fs: {(l.a, l.b) for l in linking.candidate_links(fs, FP)}  # noqa: E731
check("a fragment that starts the frame after another ends can be linked (gap 1)",
      (1, 2) in pairs([fr(1, 0, 99, 100), fr(2, 100, 200, 100)]))
check("overlapping fragments are never linked, either way round",
      pairs([fr(1, 0, 99, 100), fr(2, 50, 200, 100)]) == set())
check("overlapping by a few frames (a duplicate id picked up 5 frames early) is not a link either, whatever the speed rule says",
      pairs([fr(1, 0, 99, 100), fr(2, 95, 200, 100)]) == set() and pairs([fr(1, 0, 99, 100), fr(2, 91, 200, 100)]) == set())
check("starting on the same frame the other ends is overlap (gap 0): no link",
      pairs([fr(1, 0, 99, 100), fr(2, 99, 200, 100)]) == set())
check("never backwards: only (earlier, later) pairs exist", all(
    a_.last < b_.first for a_, b_ in [(fr(1, 0, 10, 0), fr(2, 20, 30, 0))]) and
    pairs([fr(2, 20, 30, 100), fr(1, 0, 10, 100)]) == {(1, 2)})
check(f"the gap limit is inclusive: {MAXGAP} frames (2 s) links, one more does not",
      pairs([fr(1, 0, 99, 100), fr(2, 99 + MAXGAP, 400, 100)]) == {(1, 2)}
      and pairs([fr(1, 0, 99, 100), fr(2, 100 + MAXGAP, 400, 100)]) == set())
# 25 frames = 0.5 s: allowed jump = 1 + 5 x 0.5 = 3.5 body lengths = 140 px at scale 40
check("the speed limit: 139 px in 0.5 s is plausible, 141 px is not",
      pairs([fr(1, 0, 99, 100), fr(2, 124, 300, 239)]) == {(1, 2)}
      and pairs([fr(1, 0, 99, 100), fr(2, 124, 300, 241)]) == set())
check("the same jump is farther for a smaller body: 100 px is 5 body lengths at scale 20",
      pairs([fr(1, 0, 99, 100, scale=20), fr(2, 124, 300, 200, scale=20)]) == set()
      and pairs([fr(1, 0, 99, 100, scale=40), fr(2, 124, 300, 200, scale=40)]) == {(1, 2)})
check("plausible_jump is the speed limit plus the slack",
      linking.plausible_jump(3.5, 25, FP) and not linking.plausible_jump(3.51, 25, FP)
      and linking.plausible_jump(1.0, 1, FP) is True)
check("a link costs more the farther the jump, the longer the gap, the worse the scale match",
      all(c1 < c2 for c1, c2 in [
          (linking.candidate_links([fr(1, 0, 99, 100), fr(2, 110, 300, 105)], FP)[0].cost,
           linking.candidate_links([fr(1, 0, 99, 100), fr(2, 110, 300, 140)], FP)[0].cost),
          (linking.candidate_links([fr(1, 0, 99, 100), fr(2, 110, 300, 100)], FP)[0].cost,
           linking.candidate_links([fr(1, 0, 99, 100), fr(2, 190, 300, 100)], FP)[0].cost),
          (linking.candidate_links([fr(1, 0, 99, 100), fr(2, 110, 300, 100)], FP)[0].cost,
           linking.candidate_links([fr(1, 0, 99, 100), fr(2, 110, 300, 100, scale=60)], FP)[0].cost)]))

print("evidence: what a fragment is worth")
f_half = Fragment(7, 0, 99, 100, (0, 0), (0, 0), 40.0, True, ring_share=0.6, punches=2)
check("reward = 10 points a credited punch + 0.5 a second in the ring, weighted by the share inside the ring",
      abs(linking.reward(f_half, FP) - (20 + 0.5 * 0.6 * 2.0)) < 1e-9, str(linking.reward(f_half, FP)))
check("a fragment wholly outside the ring earns nothing for its time",
      linking.reward(dataclasses.replace(f_half, ring_share=0.0, punches=0), FP) == 0.0)
check("eligible: long enough, a body, at least half its frames in the ring",
      linking.eligible(f_half, FP) and not linking.eligible(dataclasses.replace(f_half, ring_share=0.49), FP)
      and not linking.eligible(dataclasses.replace(f_half, n_rows=9), FP)
      and not linking.eligible(dataclasses.replace(f_half, scale=float("nan")), FP))

print("the guards, in a scene")
# A far spatial jump: boxer B's third fragment appears 200 px (5 body lengths) from where the second
# one vanished, 0.4 s later. Allowed: 1 + 5 x 0.4 = 3 body lengths.
far = base_scene()
far.people["B"]["path"] = lambda f: ((420 + 40 * np.sin(2 * np.pi * f / 230 + 1.0)) - (200 if 420 <= f <= 700 else 0), 190)
t_far, _, a_far, lk_far = run(far)
fr_far, _ = linking.summarise_fragments(t_far)
check("a far spatial jump is not even a candidate link",
      (12, 13) not in {(l.a, l.b) for l in linking.candidate_links(fr_far, FPS)})
check("... and the two fragments never end up in one identity",
      not any({12, 13} <= lk_far.ids(k) for k in (0, 1)), str(lk_far.identities))
# A gap beyond 2 s: boxer B vanishes for 3 s between fragments 12 and 13.
gap = base_scene()
gap.people["B"]["frags"] = [(11, 0, 139), (12, 150, 399), (13, 550, 700), (14, 712, 999)]
t_gap, _, a_gap, lk_gap = run(gap)
check("a gap longer than the limit (3 s) is never bridged: 12 and 13 are not one identity",
      not any({12, 13} <= lk_gap.ids(k) for k in (0, 1)), str(lk_gap.identities))
# Two ids on the SAME person at the same time (a tracker double): they overlap in time.
dup = base_scene()
dup.people["A"]["frags"] = A_FR + [(41, 100, 600)]
t_dup, _, a_dup, lk_dup = run(dup)
check("two ids that overlap in time are never in one identity (41 overlaps 1, 2 and 3)",
      not any(41 in lk_dup.ids(k) and (lk_dup.ids(k) & {1, 2, 3}) for k in (0, 1)), str(lk_dup.identities))
# Disjointness of the two units: ONE boxer only. A flow that let two chains share fragments would
# return the same fragments twice (each counted twice); the second identity must be empty instead.
solo = base_scene(boxer_b=False, referee=False)
t_solo, _, a_solo, lk_solo = run(solo, boxers=("A",))
check("the two identities never share a fragment: with one boxer, the second identity is empty",
      lk_solo.ids(0) == A_IDS and lk_solo.identities[1] == () and not (lk_solo.ids(0) & lk_solo.ids(1)),
      str(lk_solo.identities))
check("... and its membership column is entirely unidentified", (lk_solo.membership[:, 1] == -1).all())
# Punch evidence: the referee stands in the ring for 20 s, more than boxer B's ring time, and has no punches.
refb = base_scene()
t_ref, ev_ref, a_ref, lk_ref = run(refb)
check("a referee with more ring time but no punches does not displace a boxer who threw them",
      {21, 22} & (lk_ref.ids(0) | lk_ref.ids(1)) == set() and B_IDS <= (lk_ref.ids(0) | lk_ref.ids(1)))
nob = base_scene(boxer_b=True)
t_nob, ev_nob, a_nob, lk_nob = run(nob, boxers=("A",))     # nobody credits B: no punch evidence for B
check("with no punch evidence for the second boxer, the referee is NOT made an identity (a chain needs a punch)",
      lk_nob.ids(0) == A_IDS and lk_nob.identities[1] == () and not ({21, 22, 31, 32} & lk_nob.ids(1)),
      str(lk_nob.identities))
lk_nob5 = run(nob, boxers=("A",), params=LinkParams(ring_points_per_s=5.0))[3]
# REPLACED (was: ids(0) == A_IDS and identities[1] == ()): that encoded the old post-filter, which emptied a
# punch-free chain AFTER solving. The punch requirement is now inside the optimisation, so with ring time
# worth 10x the best answer may legitimately use a referee fragment as a tail of a punching chain, but never
# as a chain of its own: every non-empty identity must hold a credited punch.
punchy_ids = {f.tid for f in lk_nob5.fragments if f.punches > 0}
check("... even if ring time were valued 10x: no identity is punch-free (a referee-only chain is not reported)",
      all(set(c) & punchy_ids for c in lk_nob5.identities if c) and not any(c and set(c) <= {21, 22, 31, 32}
                                                                           for c in lk_nob5.identities),
      str(lk_nob5.identities))
# The ring prior: with no second boxer and no referee, the spectators (20 s each, static, outside the
# ring) are the only candidates for the second identity, and they carry no punches or ring time.
spec = base_scene(boxer_b=False, referee=False)
t_sp, _, a_sp, lk_sp = run(spec, boxers=("A",))
check("static spectators outside the ring are never an identity", not ({31, 32} & (lk_sp.ids(0) | lk_sp.ids(1))))
# Punches credited to a spectator (a wrongly credited punch) cannot make them a boxer: outside the ring.
lk_noisy = linking.link(t_sp, dict(linking.credited_counts(a_sp)) | {31: 2}, lk_sp.region)
check("... even with two punches wrongly credited to one: a fragment outside the ring is not eligible",
      31 not in (lk_noisy.ids(0) | lk_noisy.ids(1)) and lk_noisy.ids(0) == A_IDS, str(lk_noisy.identities))

print("re-acquisition (off by default): linking across a gap longer than the tracker's buffer")
RE = LinkParams(reacquire_s=60.0)
far_pair = [fr(1, 0, 99, 100), fr(2, 400, 600, 900)]       # 6 s later, 800 px (20 body lengths) away
check("off by default: a 6 s gap with a huge jump is not a candidate", pairs(far_pair) == set())
check("on: it is, with NO speed limit across the long gap",
      {(l.a, l.b) for l in linking.candidate_links(far_pair, FP, RE)} == {(1, 2)})
ordinary = linking.candidate_links([fr(1, 0, 99, 100), fr(2, 110, 300, 105)], FP, RE)[0].cost
long_ = linking.candidate_links(far_pair, FP, RE)[0].cost
check("... but it is priced higher than any ordinary link", long_ > ordinary + RE.reacquire_points, f"{long_} {ordinary}")
check("an ordinary gap is still held to the speed limit with re-acquisition on",
      {(l.a, l.b) for l in linking.candidate_links([fr(1, 0, 99, 100), fr(2, 124, 300, 241)], FP, RE)} == set())
reach = int(round(60.0 * FP))
check("the re-acquisition reach is inclusive, 60 s = 3000 frames: one more is out",
      {(l.a, l.b) for l in linking.candidate_links([fr(1, 0, 99, 100), fr(2, 99 + reach, 4000, 100)], FP, RE)} == {(1, 2)}
      and {(l.a, l.b) for l in linking.candidate_links([fr(1, 0, 99, 100), fr(2, 100 + reach, 4000, 100)], FP, RE)} == set())
check("overlapping fragments are still never linked with re-acquisition on",
      {(l.a, l.b) for l in linking.candidate_links([fr(1, 0, 99, 100), fr(2, 50, 400, 100)], FP, RE)} == set())
lk_re = run(gap, params=RE)[3]
check("the 3 s hole in boxer B (unlinkable by default) is bridged with re-acquisition on: B is whole again",
      sets(lk_re) == {A_IDS, B_IDS}, str(lk_re.identities))
raises("a negative re-acquisition window is refused", lambda: LinkParams(reacquire_s=-1), "reacquire_s")
raises("a NaN re-acquisition price is refused", lambda: LinkParams(reacquire_points=float("nan")), "reacquire_points")

print("crossing boxers: what the linking does when two people swap places during an id handover")
cross = Scene(250).add("A", lambda f: (150 + 1.5 * f, 200), [(1, 0, 98), (2, 110, 249)]).add(
    "B", lambda f: (450 - 1.5 * f, 200), [(11, 0, 98), (12, 110, 249)])
# A walks right, B walks left; they pass at frame 100 and both ids are lost for 11 frames. At the
# handover A's new id starts nearer where B was, and B's nearer where A was.
t_x = cross.table()
ev_x = cross.punches(("A", "B"), per_frag=3)
lk_x = linking.link_video(t_x, olympic.attribute(ev_x, t_x))
swap = sets(lk_x) == {frozenset({1, 12}), frozenset({11, 2})}
straight = sets(lk_x) == {frozenset({1, 2}), frozenset({11, 12})}
# DOCUMENTED LIMITATION, asserted as the actual behaviour: this IS an identity swap. Spatial
# continuity alone cannot tell who is who after a crossing, so the shorter jump wins.
check("crossing paths with ids lost at the crossing: the nearer continuation wins, which SWAPS the boxers",
      swap and not straight, str(lk_x.identities))
check("... yet the two identities stay disjoint and each stays inside the ring and in time order",
      not (lk_x.ids(0) & lk_x.ids(1)) and all(lk_x.region.contains(*f.start_xy) for f in lk_x.fragments
                                              if f.tid in lk_x.ids(0) | lk_x.ids(1)))

print("any table: chains are time-disjoint (property)")
rng = np.random.default_rng(3)
ok_disjoint = ok_count = ok_two = True
for trial in range(12):
    s = Scene(600)
    for tid in range(1, 25):
        f0 = int(rng.integers(0, 520)); f1 = int(min(f0 + rng.integers(10, 150), 599))
        x0, y0 = rng.uniform(150, 500), rng.uniform(150, 250)
        vx, vy = rng.uniform(-1, 1, 2)
        s.add(f"p{tid}", (lambda x0, y0, vx, vy, f0: lambda f: (x0 + vx * (f - f0), y0 + vy * (f - f0)))(x0, y0, vx, vy, f0),
              [(tid, f0, f1)])
    tb = s.table()
    reg = Region(100, 100, 560, 300)
    cr = {tid: int(rng.integers(0, 4)) for tid in range(1, 25)}
    lkp = linking.link(tb, cr, reg, LinkParams(reacquire_s=60.0 if trial % 2 else 0.0))
    fm = {f.tid: f for f in lkp.fragments}
    for ch in lkp.identities:
        ok_disjoint &= all(fm[a].last < fm[b].first for a, b in zip(ch, ch[1:]))
        ok_disjoint &= all(fm[a].last < fm[b].first for i, a in enumerate(ch) for b in ch[i + 1:])
    ok_two &= not (lkp.ids(0) & lkp.ids(1))
    ok_count &= all(int((lkp.membership[:, k] >= 0).sum()) == sum(fm[t].n_rows for t in lkp.identities[k]) for k in (0, 1))
check("in every random table (half with re-acquisition on), a chain's fragments are pairwise disjoint in time", ok_disjoint)
check("... the two chains share no fragment", ok_two)
check("... and each frame is counted once: identified frames == the rows of its fragments", ok_count)

print("the punch requirement is part of the optimisation (Astra 6, finding 1)")


def raises_as(name, fn, exc, pattern=""):
    try:
        fn()
    except exc as e:
        check(name, pattern in str(e), f"wrong reason: {e}")
    except Exception as e:
        check(name, False, f"raised {type(e).__name__}: {e}")
    else:
        check(name, False, "did not raise")


def fq(tid, first, last, x, punches, ring_share=1.0, y=100.0, scale=40.0):
    return Fragment(tid, first, last, last - first + 1, (x, y), (x, y), scale, True, ring_share, punches)


# Astra's example, in the units of the model: boxer A 60 s with 10 punches, boxer B 10 s with 1 punch, a
# referee 60 s with none, all overlapping in time and in the ring. Net of the 8-point chain fee: 122, 7, 22.
fa, fb, fref = fq(1, 0, 2999, 100, 10), fq(2, 0, 499, 140, 1), fq(3, 0, 2999, 180, 0)
check("the three fragments are worth 130 / 15 / 30 points (net 122 / 7 / 22 of the chain fee)",
      [linking.reward(f, FP) for f in (fa, fb, fref)] == [130.0, 15.0, 30.0])
ch3, obj3 = linking.solve_chains([fa, fb, fref], [], FP)
check("A and B are chosen and the referee is not: A + B = 122 + 7 = 129 (the old post-filter returned A + an emptied slot)",
      sorted(ch3) == [(1,), (2,)] and obj3 == 129.0, str((ch3, obj3)))
sc3 = Scene(3000).add("A", lambda f: (150, 200), [(1, 0, 2999)]).add("B", lambda f: (250, 200), [(2, 0, 499)]).add(
    "ref", lambda f: (350, 200), [(3, 0, 2999)])
lk3 = linking.link(sc3.table(), {1: 10, 2: 1}, Region(0, 0, W, H))
check("... through link(): identity A is the 60 s boxer, identity B is the 10 s boxer, nobody else",
      lk3.identities == ((1,), (2,)) and abs(lk3.objective - 129.0) < 1e-9, str((lk3.identities, lk3.objective)))
check("... with the solve time recorded", lk3.solve_s > 0)
ch3n, obj3n = linking.solve_chains([fa, fref], [], FP)
check("with no second boxer the referee still does not get the slot: A alone = 122",
      ch3n == [(1,)] and obj3n == 122.0, str((ch3n, obj3n)))
ch0, obj0 = linking.solve_chains([fref], [], FP)
check("a lone punch-free fragment is no chain at all (value 0, nothing returned)", ch0 == [] and obj0 == 0.0)
check("no usable fragment: no chains", linking.solve_chains([], [], FP) == ([], 0.0))
# A punch-free CONNECTOR is allowed inside a chain that holds a punch (state 0 -> state 1 and 1 -> 1).
bridge_frags = [fq(1, 0, 299, 100, 0), fq(2, 310, 500, 105, 2)]
ch_b, _ = linking.solve_chains(bridge_frags, linking.candidate_links(bridge_frags, FP), FP)
check("a punch-free fragment BEFORE a punching one is a legitimate head of the same chain", ch_b == [(1, 2)], str(ch_b))
tail_frags = [fq(1, 0, 99, 100, 2), fq(2, 110, 600, 105, 0)]
ch_t, _ = linking.solve_chains(tail_frags, linking.candidate_links(tail_frags, FP), FP)
check("... and a punch-free tail after it (ring time pays for the link)", ch_t == [(1, 2)], str(ch_t))


def brute(frags, links, params, require_punch=True):
    """The optimum by enumeration: every pair of vertex-disjoint forward chains (and every single
    chain, and none), each with >= 1 credited punch, valued exactly as the integer program values them."""
    by = {f.tid: f for f in frags}
    nxt = {}
    for l in links:
        nxt.setdefault(l.a, []).append((l.b, linking._units(l.cost)))
    chains = []

    def grow(path, val):
        chains.append((frozenset(path), val - linking._units(params.chain_points), sum(by[t].punches for t in path)))
        for b, c in nxt.get(path[-1], []):
            grow(path + [b], val + linking._units(linking.reward(by[b], FP, params)) - c)

    for f in frags:
        grow([f.tid], linking._units(linking.reward(f, FP, params)))
    good = [c for c in chains if c[2] >= 1 or not require_punch]
    best = max([0] + [c[1] for c in good])
    for i, c in enumerate(good):
        for d in good[i + 1:]:
            if not (c[0] & d[0]):
                best = max(best, c[1] + d[1])
    return best / linking.COST_SCALE


rng = np.random.default_rng(20261007)
variants = [LinkParams(), LinkParams(reacquire_s=60.0), LinkParams(chain_points=0.0),
            LinkParams(punch_points=1.0), LinkParams(punch_points=1.0, reacquire_s=60.0, chain_points=2.0),
            # ring time worth so much that a punch-free chain (a referee) is worth opening: the punch rule must bind
            LinkParams(ring_points_per_s=20.0), LinkParams(ring_points_per_s=20.0, punch_points=2.0, reacquire_s=60.0)]
n_inst = n_exact = n_binds = n_valid = 0
worst = 0.0
for inst in range(800):
    prm = variants[inst % len(variants)]
    nf = int(rng.integers(2, 9))
    frs = []
    for tid in range(1, nf + 1):
        f0 = int(rng.integers(0, 400))
        f1 = f0 + int(rng.integers(12, 160))
        frs.append(fq(tid, f0, f1, float(rng.uniform(100, 220)), int(rng.choice([0, 0, 1, 2, 3])),
                      ring_share=float(rng.uniform(0.5, 1.0)), scale=float(rng.choice([36.0, 40.0, 44.0]))))
    lks = linking.candidate_links(frs, FP, prm)
    got_chains, got_obj = linking.solve_chains(frs, lks, FP, prm)
    want = brute(frs, lks, prm)
    n_inst += 1
    worst = max(worst, abs(got_obj - want))
    n_exact += abs(got_obj - want) <= 1e-9
    n_binds += abs(brute(frs, lks, prm, require_punch=False) - want) > 1e-9      # instances where the punch rule matters
    pairs_ok = {(l.a, l.b) for l in lks}
    by_t = {f.tid: f for f in frs}
    flat = [t for c in got_chains for t in c]
    n_valid += (len(got_chains) <= 2 and len(flat) == len(set(flat))
                and all(sum(by_t[t].punches for t in c) >= 1 for c in got_chains)
                and all((a, b) in pairs_ok for c in got_chains for a, b in zip(c, c[1:])))
check("brute force: on 800 random small instances (7 parameter sets) the integer program's objective equals the "
      "enumerated optimum exactly", n_exact == n_inst == 800, f"{n_exact}/{n_inst}, worst difference {worst}")
check("... and every returned solution is valid: <= 2 chains, disjoint, each with a punch, only candidate links",
      n_valid == n_inst, f"{n_valid}/{n_inst}")
check("... and the punch rule CHANGED the optimum in at least 20 of them (the cross-check has teeth)",
      n_binds >= 20, str(n_binds))


class _Stub:
    def __init__(self, status):
        self.status, self.success, self.x, self.fun, self.message = status, False, np.zeros(3), 0.0, "stub"


_real_milp = linking.milp
linking.milp = lambda *a, **k: _Stub(1)
raises_as("a solver that stops at the time limit is an error, never an answer",
          lambda: linking.solve_chains([fa, fb], [], FP), RuntimeError, "proven optimality")
linking.milp = lambda *a, **k: _Stub(4)
raises_as("... and so is any other non-optimal status", lambda: linking.solve_chains([fa, fb], [], FP), RuntimeError,
          "proven optimality")
linking.milp = _real_milp
_real_min = linking.MIN_CHAIN_PUNCHES
linking.MIN_CHAIN_PUNCHES = 2
raises_as("the model encodes exactly '>= 1 punch': any other MIN_CHAIN_PUNCHES is refused",
          lambda: linking.solve_chains([fa], [], FP), RuntimeError, "MIN_CHAIN_PUNCHES")
linking.MIN_CHAIN_PUNCHES = _real_min
raises("a link naming a fragment that is not usable is refused",
       lambda: linking.solve_chains([fa], [linking.Link(1, 99, 5, 0.0, 1.0)], FP), "not usable")
raises("two fragments with one id are refused", lambda: linking.solve_chains([fa, fa], [], FP), "duplicate")
check("networkx is gone: no module in src/ or entry point mentions it, and requirements.txt does not pin it",
      not any("networkx" in q.read_text(encoding="utf-8")
              for q in list((ROOT / "src" / "boxingvi").glob("*.py")) + list(ROOT.glob("*.py")) + [ROOT / "requirements.txt"]))

print("the ring region")
rng = np.random.default_rng(1)
gl = np.column_stack([rng.normal(300, 5, 100), rng.normal(200, 5, 100)])
boxes = np.column_stack([gl[:, 0] - 8, gl[:, 1] - 8, gl[:, 0] + 8, gl[:, 1] + 8])
boxes = np.vstack([boxes, [4992, 992, 5008, 1008]])         # one mislabelled glove far away
reg = linking.estimate_region(boxes, 40.0)
check("the region ignores one wild glove box (1st-99th percentile) and adds two body lengths",
      reg.x1 < 340 + 80 and reg.x0 > 260 - 80 - 1 and reg.contains(300, 200) and not reg.contains(2000, 1000), str(reg))
raises("too few credited gloves to estimate a ring is refused, not guessed",
       lambda: linking.estimate_region(boxes[:9], 40.0), "too few")
raises("a non-finite glove box is refused", lambda: linking.estimate_region(np.full((12, 4), np.nan), 40.0), "finite")

print("negative: bad input fails closed")
empty = PoseTable(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 4), np.float32),
                  np.zeros((0, N_JOINTS, 2), np.float32), np.zeros((0, N_JOINTS), np.float32), 50, FPS, W, H)
lk_e = linking.link(empty, {}, Region(0, 0, W, H))
check("an empty table links to nothing: two empty identities, nobody identified",
      lk_e.identities == ((), ()) and (lk_e.membership == -1).all() and lk_e.membership.shape == (50, 2))
raises("... but link_video on it refuses (no body to link, no ring to estimate)",
       lambda: linking.link_video(empty, []), "nothing to link")
one = Scene(200).add("A", lambda f: (300, 200), [(5, 20, 150)])
t1 = one.table()
lk1 = linking.link(t1, {5: 3}, Region(0, 0, W, H))
check("one fragment: it is the first identity, the second is empty", lk1.identities == ((5,), ()))
lk1b = linking.link(t1, {5: 3}, Region(0, 0, 10, 10))
check("one fragment outside the ring: no identity at all", lk1b.identities == ((), ()))
lk1c = linking.link(t1, {5: 0}, Region(0, 0, W, H))
check("one fragment in the ring with no punch: no identity (a chain needs a credited punch)", lk1c.identities == ((), ()))
raises("punches credited to an id that is not in the table (attributions from another video) are refused",
       lambda: linking.link(t1, {99: 3}, Region(0, 0, W, H)), "not in this table")
raises("a negative punch count is refused", lambda: linking.link(t1, {5: -1}, Region(0, 0, W, H)), "whole number")
raises("link_video with too few credited punches is refused",
       lambda: linking.link_video(t1, olympic.attribute([olympic.Event(30, 32, "head", "left", (270, 190, 290, 210), 1)], t1)),
       "too few")
raises("link wants a Region, not None", lambda: linking.link(t1, {5: 3}, None), "Region")
raises("link wants LinkParams, not a dict", lambda: linking.link(t1, {5: 3}, Region(0, 0, W, H), {"max_gap_s": 2}),
       "LinkParams")
for label, kw, pat in [
    ("a zero gap limit", {"max_gap_s": 0}, "max_gap_s"), ("a negative gap limit", {"max_gap_s": -1.0}, "max_gap_s"),
    ("a NaN gap limit", {"max_gap_s": float("nan")}, "max_gap_s"), ("an infinite speed", {"max_speed_bl_s": float("inf")}, "max_speed"),
    ("a zero speed limit", {"max_speed_bl_s": 0}, "max_speed"), ("a boolean slack", {"jump_slack_bl": True}, "jump_slack"),
    ("a negative reward", {"punch_points": -1}, "punch_points"), ("a share above 1", {"ring_min_share": 1.5}, "ring_min_share"),
    ("reversed ring quantiles", {"ring_quantiles": (0.9, 0.1)}, "ring_quantiles"),
    ("ring quantiles beyond 1", {"ring_quantiles": (0.0, 1.5)}, "ring_quantiles"),
    ("zero credited punches to make a ring", {"min_ring_punches": 0}, "min_ring_punches"),
    ("a string as a number", {"edge_s": "0.1"}, "edge_s")]:
    raises(f"invalid parameters: {label}", lambda kw=kw: LinkParams(**kw), pat)

print("linked overlay")
clip = tmp / "GH000001.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r=50",
                "-frames:v", str(N), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)
ov = tmp / "linked.mp4"
n_out = overlay.render_linked_overlay(clip, table, lk, ov, max_frames=240)
fr_ov = list(video.iter_frames(ov))
check("the overlay has the frames asked for", n_out == 240 and len(fr_ov) == 240)


def has(frame, x, y, pred, r=14):
    win = frame[int(y) - r:int(y) + r, int(x) - r:int(x) + r].astype(int)
    return int(pred(win[..., 0], win[..., 1], win[..., 2]).sum())


green = lambda R, G, B: (G > 170) & (R < 110) & (B < 130)       # noqa: E731
magenta = lambda R, G, B: (R > 170) & (B > 160) & (G < 120)     # noqa: E731
grey = lambda R, G, B: (abs(R - G) < 25) & (abs(G - B) < 25) & (R > 70) & (R < 190)   # noqa: E731
a_pt, b_pt, ref_pt = sc.wrist("A", 100, 0), sc.wrist("B", 100, 0), sc.wrist("ref", 100, 0)
want_a = green if k_a == 0 else magenta
want_b = magenta if k_a == 0 else green
check("each boxer is drawn in their own colour, at their own wrist",
      has(fr_ov[100], *a_pt, want_a) > 5 and has(fr_ov[100], *b_pt, want_b) > 5
      and has(fr_ov[100], *a_pt, want_b) == 0 and has(fr_ov[100], *b_pt, want_a) == 0)
check("the referee is thin grey, not in a boxer's colour",
      has(fr_ov[100], *ref_pt, grey) > 3 and has(fr_ov[100], *ref_pt, green) == 0 and has(fr_ov[100], *ref_pt, magenta) == 0)
red = lambda f_: f_[2, W // 2, 0] > 150 and f_[2, W // 2, 1] < 90   # noqa: E731
check("the border is red only in frames where a boxer is not identified (A has a gap at 200-214)",
      red(fr_ov[205]) and not red(fr_ov[100]))
wrong_fps = PoseTable(table.frame, table.track_id, table.box, table.xy, table.conf, N, 25.0, W, H, source=table.source)
raises("the linked overlay refuses a table from another video (frame rate)",
       lambda: overlay.render_linked_overlay(clip, wrong_fps, lk, tmp / "n1.mp4"), "different video")
wrong_size = PoseTable(table.frame, table.track_id, table.box, table.xy, table.conf, N, FPS, W + 2, H, source=table.source)
raises("... or a different frame size",
       lambda: overlay.render_linked_overlay(clip, wrong_size, lk, tmp / "n2.mp4"), "different video")
raises("... or a linking made from another table (other source)",
       lambda: overlay.render_linked_overlay(clip, table, dataclasses.replace(lk, source="GH999999.MP4"), tmp / "n3.mp4"),
       "different pose table")
raises("... or a linking of a different length",
       lambda: overlay.render_linked_overlay(clip, table, dataclasses.replace(lk, n_frames=N + 5), tmp / "n4.mp4"),
       "different pose table")
longer = PoseTable(table.frame, table.track_id, table.box, table.xy, table.conf, N + 5, FPS, W, H, source=table.source)
raises("... or a table longer than the video (a different, longer recording)",
       lambda: overlay.render_linked_overlay(clip, longer, dataclasses.replace(lk, n_frames=N + 5, membership=np.full((N + 5, 2), -1)),
                                             tmp / "n5.mp4"), "different video")
raises("it will not overwrite the input video", lambda: overlay.render_linked_overlay(clip, table, lk, clip), "overwrite")

print("olympic_link.py end to end, on a planted dataset")
ds = tmp / "ds"
task = ds / "task_kam2_gh000001"
(task / "data").mkdir(parents=True)
clip.replace(task / "data" / "GH000001.mp4")
pl = {v: k for k, v in olympic.TARGETS.items()}
ph = {v: k for k, v in olympic.HANDS.items()}


def cvat(evs):
    tracks = []
    for e in evs:
        x1, y1, x2, y2 = e.box
        shp = lambda frame, out: {"type": "rectangle", "frame": frame, "outside": out, "points": [x1, y1, x2, y2]}  # noqa: E731
        tracks.append({"label": f"{pl[e.target]} {ph[e.hand]}", "shapes": [shp(e.first, False), shp(e.last + 1, True)]})
    return [{"tracks": tracks}]


all_events = events + [olympic.Event(302, 304, "head", "left", (100, 100, 116, 116), 1)]   # next to nobody
(task / "annotations.json").write_text(json.dumps(cvat(all_events)), encoding="utf-8")
trk = tmp / "tracks"
trk.mkdir()
table.save(trk / "GH000001.tracks.npz")
out_json = tmp / "res" / "link.json"
rc = olympic_link.main(["--data", str(ds), "--tracks", str(trk), "--out", str(out_json), "--overlay", "GH000001"])
rep = json.loads(out_json.read_text(encoding="utf-8"))
r1 = rep["videos"]["task_kam2_gh000001"]
check("exit code 0, and the report holds the linking numbers", rc == 0 and "error" not in r1)
check("REAL FACT: every credited punch is on boxer A or B, none on the referee or the crowd",
      r1["credited_in_either_share"] == 1.0 and r1["credited_in_a"] + r1["credited_in_b"] == r1["credited_punches"] == 32,
      str(r1))
check("REAL FACT: the two identities are on screen together (co-presence), both identified most of the time",
      r1["co_presence"] > 0.95 and r1["both_identified"] > 0.9, str(r1))
check("... and it still holds with half the punches held out of the linking",
      r1["credited_in_either_share_heldout"] == 1.0 and r1["credited_heldout"] > 0, str(r1))
check("the identities are the boxers' ids, the report says which fragments",
      {frozenset(x) for x in r1["identity_fragments"]} == {frozenset(A_IDS), frozenset(B_IDS)})
check("the overlay was written next to the tracks, with every frame",
      (trk / "GH000001.linked.overlay.mp4").is_file() and video.probe(trk / "GH000001.linked.overlay.mp4").timing.n_frames == N)
check("pooled numbers agree with the video's", rep["all"]["credited_punches"] == 32 and rep["all"]["credited_in_either_share"] == 1.0)
out_re = tmp / "res" / "link_re.json"
rc_re = olympic_link.main(["--data", str(ds), "--tracks", str(trk), "--out", str(out_re), "--reacquire"])
r_re = json.loads(out_re.read_text(encoding="utf-8"))["videos"]["task_kam2_gh000001"]
check("--reacquire runs end to end and keeps every credited punch inside the two identities",
      rc_re == 0 and r_re["credited_in_either_share"] == 1.0, str(r_re))
bad_trk = tmp / "bad_tracks"
bad_trk.mkdir()
PoseTable(table.frame, table.track_id, table.box, table.xy, table.conf, N, 25.0, W, H,
          source=table.source).save(bad_trk / "GH000001.tracks.npz")
raises("a tracks file from another video (frame rate) links nothing, loudly",
       lambda: olympic_link.main(["--data", str(ds), "--tracks", str(bad_trk), "--out", str(tmp / "x.json")]), "could be linked")
short_trk = tmp / "short_tracks"
short_trk.mkdir()
PoseTable(table.frame[table.frame < 500], table.track_id[table.frame < 500], table.box[table.frame < 500],
          table.xy[table.frame < 500], table.conf[table.frame < 500], 500, FPS, W, H,
          source=table.source).save(short_trk / "GH000001.tracks.npz")
raises("an incomplete tracks file (a --max-frames run) is refused, not linked as if it were whole",
       lambda: olympic_link.main(["--data", str(ds), "--tracks", str(short_trk), "--out", str(tmp / "y.json")]), "could be linked")
for bad_id in ("../GH000001", "GH000001/../../x", "GH 1", "", "GH000001.mp4"):
    raises(f"--overlay {bad_id!r} is not a GoPro id: refused before anything is written",
           lambda b=bad_id: olympic_link.main(["--data", str(ds), "--tracks", str(trk), "--out", str(tmp / "z.json"),
                                               "--overlay", b]), "GoPro id")
raises("--overlay of a video that is not in the dataset is refused",
       lambda: olympic_link.main(["--data", str(ds), "--tracks", str(trk), "--out", str(tmp / "z.json"), "--overlay", "GH123456"]),
       "no such video")

print("the held-out evaluation can fail (Astra 6, finding 7)")
# B's last fragment (15) is only 1.2 s long and its four labelled punches all sit at ODD list positions, i.e. in the
# held-out half. Linked on the even half alone, fragment 15 has no punch evidence and 0.6 points of ring time,
# less than the cost of the link, so it is left out: the held-out share must be below 100%.
HB_FR = [(11, 0, 139), (12, 150, 399), (13, 420, 700), (14, 712, 930), (15, 940, 999)]
ho_scene = base_scene()
ho_scene.people["B"]["frags"] = HB_FR
ho_base = base_scene()
ho_base.people["B"]["frags"] = HB_FR[:4]
tab_ho = ho_scene.table()
ev_base = ho_base.punches(("A", "B"))
extra = []
for f_ in (945, 960, 975, 990):
    w_ = ho_scene.wrist("B", f_, 1)
    extra.append(olympic.Event(f_, f_ + 2, "head", "right", (w_[0] - 8, w_[1] - 8, w_[0] + 8, w_[1] + 8), 1))
ordered, it_b, it_x = [], iter(ev_base), list(extra)
for pos in range(len(ev_base) + len(extra)):
    ordered.append(it_x.pop(0) if pos % 2 == 1 and it_x else next(it_b))
attrs_ho = olympic.attribute(ordered, tab_ho)
check("setup: 36 punches, all credited; the four on fragment 15 are credited to it and sit at odd positions",
      len(attrs_ho) == 36 and all(a.track_id is not None for a in attrs_ho)
      and [i for i, a in enumerate(attrs_ho) if a.track_id == 15] == [1, 3, 5, 7],
      str([(i, a.track_id) for i, a in enumerate(attrs_ho) if a.track_id in (15, None)]))
lk_ho, out_ho = olympic_link.analyse(tab_ho, attrs_ho)
train = linking.link_video(tab_ho, attrs_ho[0::2])
check("fragment 15 is in an identity when every label is used, and NOT when only the even half is",
      15 in (lk_ho.ids(0) | lk_ho.ids(1)) and 15 not in (train.ids(0) | train.ids(1)),
      str((lk_ho.identities, train.identities)))
check("the full in-sample share is 100% (36 of 36)", out_ho["credited_in_either_share"] == 1.0)
check("the held-out share is BELOW 100%: 14 of the 18 held-out punches (the four on fragment 15 are lost)",
      out_ho["credited_heldout"] == 18 and out_ho["credited_in_either_share_heldout"] == round(14 / 18, 4)
      and out_ho["credited_in_either_share_heldout"] < 1.0, str(out_ho))
leaked = linking.link_video(tab_ho, attrs_ho)
leak_share = sum(a.track_id in leaked.ids(0) | leaked.ids(1) for a in attrs_ho[1::2]) / 18
check("... and had all the labels leaked into the linking it would read 100%: the test can tell the difference",
      leak_share == 1.0 and out_ho["credited_in_either_share_heldout"] < leak_share, str(leak_share))
check("the report carries the ring the held-out linking used",
      out_ho["ring_heldout"] == [round(v, 1) for v in (train.region.x0, train.region.y0, train.region.x1, train.region.y1)],
      str((out_ho["ring_heldout"], train.region)))
# Changing the held-out events' glove boxes (bigger, shifted 40 px to the right; boxer B's punches, which are the
# rightmost) must not move the held-out ring.
moved = []
for i_, e_ in enumerate(ordered):
    if i_ % 2 == 1 and attrs_ho[i_].track_id in {11, 12, 13, 14, 15}:
        cx_, cy_ = (e_.box[0] + e_.box[2]) / 2 + 40, (e_.box[1] + e_.box[3]) / 2
        e_ = olympic.Event(e_.first, e_.last, e_.target, e_.hand, (cx_ - 24, cy_ - 24, cx_ + 24, cy_ + 24), 1)
    moved.append(e_)
attrs_mv = olympic.attribute(moved, tab_ho)
_, out_mv = olympic_link.analyse(tab_ho, attrs_mv)
check("setup: the moved held-out punches are still credited to the same ids",
      [a.track_id for a in attrs_mv[1::2]] == [a.track_id for a in attrs_ho[1::2]])
check("changing the held-out events' glove boxes changes the full ring but NOT the held-out ring",
      out_mv["ring"] != out_ho["ring"] and out_mv["ring_heldout"] == out_ho["ring_heldout"],
      str((out_ho["ring"], out_mv["ring"], out_ho["ring_heldout"], out_mv["ring_heldout"])))

check("analyse keeps the held-out count as an INTEGER: 14 of 18 (not just the rounded share)",
      out_ho["credited_heldout_inside"] == 14 and isinstance(out_ho["credited_heldout_inside"], int)
      and out_ho["credited_heldout"] == 18, str(out_ho))
# Two videos: 1 of 5 and 4 of 6 held-out punches inside. Rebuilding counts from the ROUNDED shares (0.2 x 5, 0.6667 x 6)
# gives 0.4546; the integer ratio 5 / 11 is 0.4545.
rows_p = [{"credited_heldout_inside": 1, "credited_heldout": 5, "credited_in_either_share_heldout": round(1 / 5, 4)},
          {"credited_heldout_inside": 4, "credited_heldout": 6, "credited_in_either_share_heldout": round(4 / 6, 4)}]
old_way = round(sum(r["credited_in_either_share_heldout"] * r["credited_heldout"] for r in rows_p) / 11, 4)
check("setup: the old pooling (rounded shares times denominators) would say 0.4546", old_way == 0.4546, str(old_way))
check("the pooled held-out share is the integer ratio 5 / 11 = 0.4545", olympic_link.pooled_heldout_share(rows_p) == 0.4545
      and olympic_link.pooled_heldout_share(rows_p) == round(5 / 11, 4))
check("a video with no held-out punches (ring not estimable) adds nothing to either count",
      olympic_link.pooled_heldout_share(rows_p + [{"credited_heldout_inside": 0, "credited_heldout": 0,
                                                   "credited_in_either_share_heldout": None}]) == 0.4545)

print("the pose table must belong to its video (Astra 6, finding 8)")
tk = olympic.find_tasks(ds)[0]
other_clip = tmp / "other.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=white:s={W}x{H}:r=50",
                "-frames:v", str(N), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(other_clip)], check=True)
fp_real, fp_other = video.fingerprint(tk.video), video.fingerprint(other_clip)
info_real = video.probe(tk.video)
check("setup: the other recording has the same size, rate and frame count (so the old checks cannot tell them apart), "
      "a different fingerprint",
      (video.probe(other_clip).width, video.probe(other_clip).height) == (info_real.width, info_real.height)
      and video.probe(other_clip).fps == info_real.fps and video.probe(other_clip).timing.n_frames == info_real.timing.n_frames
      and fp_real != fp_other and len(fp_real) == 64)


def table_dir(name, **over):
    d = tmp / name
    d.mkdir()
    dataclasses.replace(table, **over).save(d / "GH000001.tracks.npz")
    return d


raises("same size, same fps, same length, DIFFERENT NAME: refused, naming both",
       lambda: olympic_link.load_video(tk, table_dir("t_renamed", source="GH999999.mp4")), "'GH999999.mp4', not 'GH000001.mp4'")
raises("same name, different content, with a fingerprint: refused",
       lambda: olympic_link.load_video(tk, table_dir("t_wrongfp", video_fp=fp_other)), "fingerprint")
got_legacy = olympic_link.load_video(tk, table_dir("t_legacy"))
check("a legacy table with no fingerprint is accepted on the name alone", got_legacy[0].video_fp == "" and len(got_legacy[2]) > 0)
got_bound = olympic_link.load_video(tk, table_dir("t_bound", video_fp=fp_real))
check("a table whose fingerprint matches is accepted, and keeps it", got_bound[0].video_fp == fp_real)
for tag, kw, want_bound in (("t_legacy", {}, False), ("t_bound", {"video_fp": fp_real}, True)):
    o_ = tmp / "res" / f"bind_{tag}.json"
    olympic_link.main(["--data", str(ds), "--tracks", str(tmp / tag), "--out", str(o_)])
    rb = json.loads(o_.read_text(encoding="utf-8"))
    check(f"the report says video_bound = {want_bound} for {tag}",
          rb["videos"]["task_kam2_gh000001"]["video_bound"] is want_bound
          and rb["all"]["videos_content_bound"] == int(want_bound), str(rb["all"]))
raises("the CLI links nothing from a renamed table, loudly",
       lambda: olympic_link.main(["--data", str(ds), "--tracks", str(tmp / "t_renamed"), "--out", str(tmp / "x2.json")]),
       "could be linked")
renamed_t = dataclasses.replace(table, source="GH999999.mp4")
wrongfp_t = dataclasses.replace(table, video_fp=fp_other)
raises("the tracks overlay refuses a renamed table", lambda: overlay.render_tracks_overlay(tk.video, renamed_t, tmp / "b1.mp4", max_frames=5),
       "'GH999999.mp4', not 'GH000001.mp4'")
raises("... and a table with another recording's fingerprint",
       lambda: overlay.render_tracks_overlay(tk.video, wrongfp_t, tmp / "b2.mp4", max_frames=5), "fingerprint")
check("... but renders a table whose fingerprint matches",
      overlay.render_tracks_overlay(tk.video, dataclasses.replace(table, video_fp=fp_real), tmp / "b3.mp4", max_frames=5) == 5)
raises("the linked overlay refuses a renamed table", lambda: overlay.render_linked_overlay(tk.video, renamed_t, lk, tmp / "b4.mp4", max_frames=5),
       "'GH999999.mp4', not 'GH000001.mp4'")
raises("... and a table with another recording's fingerprint",
       lambda: overlay.render_linked_overlay(tk.video, wrongfp_t, lk, tmp / "b5.mp4", max_frames=5), "fingerprint")
check("... but renders a table whose fingerprint matches",
      overlay.render_linked_overlay(tk.video, dataclasses.replace(table, video_fp=fp_real), lk, tmp / "b6.mp4",
                                    max_frames=5) == 5)

print("completeness: wrist-speed bursts per linked boxer, matched one-to-one to labels")
RULE = {"variant": "relative", "threshold": 3.62, "min_gap_s": 0.4}


def manual_linking(tab, chains):
    """A Linking naming exactly these chains of track ids (no optimisation), for building scenes by hand."""
    mem = np.full((tab.n_frames, 2), -1, np.int64)
    for k, ch in enumerate(chains):
        for t in ch:
            mem[tab.frame[tab.track_id == t], k] = t
    return Linking((tuple(chains[0]), tuple(chains[1])), mem, tab.n_frames, tab.fps, tab.source, None)


def bursts_near(res, lo, hi):
    return sorted((b for b in res["bursts"] if lo <= b.frame <= hi), key=lambda b: b.boxer)


# Planted bursts: A's right wrist at 100 (nothing labelled near), A's left at 300 (a label at 302-304, thrown by
# nobody we can credit), B's right at 500 (nothing near), and the REFEREE's at 250 (not a boxer: must not be counted).
cs = base_scene(bursts={"A": [(100, 1), (300, 0)], "B": [(500, 1)], "ref": [(250, 0)]})
t_c = cs.table()
ev_c = cs.punches(("A", "B"))
lab = ev_c + [olympic.Event(302, 304, "head", "left", (100, 100, 116, 116), 1)]
attrs_lab = olympic.attribute(lab, t_c)
lk_c = linking.link_video(t_c, attrs_lab)
res = completeness.find_bursts(t_c, lk_c, lab, attrs_lab, RULE, video="GH000001")
kA = "A" if lk_c.ids(0) == A_IDS else "B"      # which letter the linking gave the planted boxer A
fa = sorted((b.frame, b.hand, b.explained) for b in res["bursts"] if b.boxer == kA)
fb = sorted((b.frame, b.hand, b.explained) for b in res["bursts"] if b.boxer != kA)
near = lambda got, want: len(got) == len(want) and all(abs(g[0] - w[0]) <= 6 and g[1:] == w[1:] for g, w in zip(got, want))  # noqa: E731
check("setup: the extra label is unattributed (nobody's wrist is near it); the other 32 are credited",
      attrs_lab[-1].track_id is None and all(a.track_id is not None for a in attrs_lab[:-1]))
check("A: both planted bursts are found at the right frame (+-6, the speed peak is on the rise) and hand; the one next to a label is explained",
      near(fa, [(100, "right", False), (300, "left", True)]), str(fa))
check("B: its planted burst is found and unexplained", near(fb, [(500, "right", False)]), str(fb))
b300 = bursts_near(res, 290, 310)[0]
check("the burst at 300 is explained by the UNATTRIBUTED label, and its hand agrees (left, left)",
      b300.match == "unattributed" and b300.hand_agrees is True, str(b300))
check("the referee's burst (not an identity) is not counted for anybody",
      not any(240 <= b.frame <= 260 for b in res["bursts"]))
check("per-boxer counts and minutes: identified minutes are the identified frames, unexplained counted",
      res["boxers"][kA]["unexplained"] == 1 and res["boxers"][kA]["bursts"] == 2
      and abs(res["boxers"][kA]["identified_min"] - int(lk_c.identified(0 if kA == "A" else 1).sum()) / FPS / 60) < 1e-12
      and res["boxers"][kA]["exposure_min"] > 0)
check("the label counts: 33 labels, 16 owned by each boxer, 1 unattributed; one label matched a burst (unattributed)",
      res["labels"] == {"total": 33, "owned_a": 16, "owned_b": 16, "unattributed": 1, "unattributed_in_exposure": 1,
                        "matched_own": 0, "matched_unattributed": 1}, str(res["labels"]))
check("a burst's time is its frame over the frame rate", all(abs(b.time_s - b.frame / FPS) < 1e-9 for b in res["bursts"]))
check("per-hand counts: A threw one burst per hand, B one with the right",
      res["boxers"][kA]["bursts_by_hand"] == {"left": 1, "right": 1} and res["boxers"][kA]["unexplained_by_hand"] == {"left": 0, "right": 1}
      and res["boxers"]["B" if kA == "A" else "A"]["bursts_by_hand"] == {"left": 0, "right": 1})
quiet = base_scene()
t_q = quiet.table()
ev_q = quiet.punches(("A", "B"))
at_q = olympic.attribute(ev_q, t_q)
lk_q = linking.link_video(t_q, at_q)
res_q = completeness.find_bursts(t_q, lk_q, ev_q, at_q, RULE)
check("no planted burst, no burst: a boxer who only sways gives none", res_q["bursts"] == [])
ev5 = lambda a_, b_: olympic.Event(a_, b_, "head", "left", (1, 1, 2, 2), 1)   # noqa: E731
check("the tolerance is 0.1 s = 5 frames: a label 5 frames away is within it, 6 are not (inside the span is 0 away)",
      completeness.span_gap(100, ev5(105, 107)) == 5 and completeness.span_gap(100, ev5(106, 108)) == 6
      and completeness.span_gap(100, ev5(90, 95)) == 5 and completeness.span_gap(100, ev5(80, 94)) == 6
      and completeness.span_gap(100, ev5(95, 105)) == 0)
raises("a linking from another table is refused", lambda: completeness.boxer_track(t_c, dataclasses.replace(lk_c, source="x"), 0),
       "different pose table")
check("a boxer with under two identified frames has no track (None), not a crash",
      completeness.boxer_track(t_c, dataclasses.replace(lk_c, membership=np.full_like(lk_c.membership, -1)), 0) is None)
raises("a negative tolerance is refused", lambda: completeness.find_bursts(t_c, lk_c, lab, attrs_lab, RULE, tol_s=-1), "tol_s")
raises("attributions that are not one per event, in order, are refused",
       lambda: completeness.find_bursts(t_c, lk_c, lab, attrs_lab[:-1], RULE), "one attribution per event")
raises("... or are for other events", lambda: completeness.find_bursts(t_c, lk_c, lab[::-1], attrs_lab, RULE), "one attribution per event")
raises("a Burst with an unknown match kind is refused", lambda: completeness.Burst("v", 1, 0.1, "A", "left", 5.0, "maybe"), "match")

print("completeness: a label thrown by the OTHER boxer never explains this one (Astra 6, finding 2)")
# A punches with the left hand (extended far from the referee), B with the right
both = base_scene(bursts={"A": [(500, 0)], "B": [(500, 1)]})
t_b = both.table()
ev_b = both.punches(("A", "B"))
wr = both.wrist("A", 500, 0)
mine_ev = olympic.Event(496, 504, "head", "left", (wr[0] - 30, wr[1] - 30, wr[0] + 30, wr[1] + 30), 1)
ev_b2 = ev_b + [mine_ev]
attrs_b = olympic.attribute(ev_b2, t_b)
lk_b = linking.link_video(t_b, attrs_b)
kA_b = 0 if lk_b.ids(0) == A_IDS else 1
check("setup: the extra label is credited to a fragment of boxer A, and the linking owns it to A",
      attrs_b[-1].track_id in A_IDS and kA_b == 0 and completeness.event_owner(attrs_b[-1], lk_b) == 0,
      str((attrs_b[-1].track_id, completeness.event_owner(attrs_b[-1], lk_b))))
res_b = completeness.find_bursts(t_b, lk_b, ev_b2, attrs_b, RULE)
nb = bursts_near(res_b, 485, 515)
check("both boxers burst at the same moment, only A's punch is labelled: A's burst is explained by its OWN label ...",
      len(nb) == 2 and nb[0].boxer == "A" and nb[0].match == "own" and nb[0].explained, str(nb))
check("... and B's burst is UNEXPLAINED (the label is A's; temporal proximity is not a match)",
      nb[1].boxer == "B" and nb[1].match == "none" and not nb[1].explained and res_b["boxers"]["B"]["unexplained"] == 1, str(nb))
check("the label count: exactly one label matched, it is an own match",
      res_b["labels"]["matched_own"] == 1 and res_b["labels"]["matched_unattributed"] == 0, str(res_b["labels"]))
check("hand evidence is recorded and not required: the label says left, the burst is the left hand",
      nb[0].hand == "left" and nb[0].hand_agrees is True and nb[1].hand_agrees is None and nb[1].hand == "right")
ev_b3 = ev_b + [olympic.Event(496, 504, "head", "right", mine_ev.box, 1)]
res_b3 = completeness.find_bursts(t_b, lk_b, ev_b3, olympic.attribute(ev_b3, t_b), RULE)
nb3 = bursts_near(res_b3, 485, 515)
check("... a label with the other hand still explains the burst (the 78% convention is not a requirement) and says so",
      nb3[0].match == "own" and nb3[0].hand_agrees is False, str(nb3[0]))

print("completeness: one-to-one (Astra 6, finding 2)")
EV = lambda a_, b_: olympic.Event(a_, b_, "head", "left", (1, 1, 2, 2), 1)   # noqa: E731
m1 = completeness._match([("b", 100)], [0, 1], [EV(103, 104), EV(95, 95)], 5, set())
check("one burst between two labels matches exactly ONE of them, the nearer (3 frames, not 5)", m1 == {"b": 0}, str(m1))
taken = set()
m2 = completeness._match([("x", 100), ("y", 102)], [0], [EV(101, 101)], 5, taken)
check("two bursts near one label: only one gets it (the earlier on a tie), the label is then used up",
      m2 == {"x": 0} and taken == {0}, str(m2))
m3 = completeness._match([("x", 100)], [0], [EV(101, 101)], 5, {0})
check("a label already taken is not available", m3 == {})
m4 = completeness._match([("x", 100), ("y", 101)], [0, 1], [EV(101, 101), EV(95, 95)], 5, set())
check("nearest-first still holds where it costs nothing: the burst on the label (gap 0) takes it; the other takes the farther label",
      m4 == {"y": 0, "x": 1}, str(m4))
# Astra's counterexample: tolerance 5, bursts at 100 and 120, owned labels spanning [99,120] and [95,95]. Greedy gave the
# broad label to the burst at 100 (gap 0) and left 120 unexplained; the optimal assignment explains both.
m5 = completeness._match([("a", 100), ("b", 120)], [0, 1], [EV(99, 120), EV(95, 95)], 5, set())
check("maximum matching: bursts at 100 and 120, labels [99,120] and [95,95], tol 5 -> BOTH explained (100 takes the narrow label, "
      "120 the broad one)", m5 == {"a": 1, "b": 0}, str(m5))
tk5 = set()
completeness._match([("a", 100), ("b", 120)], [0, 1], [EV(99, 120), EV(95, 95)], 5, tk5)
check("... and both labels are used up", tk5 == {0, 1})
m6 = completeness._match([("a", 100)], [0], [EV(106, 108)], 5, set())
check("a label beyond the tolerance is never matched, even when it is the only candidate", m6 == {})
m7 = completeness._match([("a", 100), ("b", 101)], [0], [EV(100, 101)], 5, set())
check("two bursts, one label: exactly one is matched, the lower (frame, key) on an exact tie", m7 == {"a": 0}, str(m7))
m8 = completeness._match([((1, 0), 100), ((0, 0), 100)], [0], [EV(100, 100)], 5, set())
check("simultaneous bursts of both boxers competing for ONE label: boxer 0 (A) wins the exact tie, whatever the list order",
      m8 == {(0, 0): 0}, str(m8))
m9 = completeness._match([("a", 100)], [3, 1], [EV(0, 0), EV(98, 98), EV(0, 0), EV(102, 102)], 5, set())
check("one burst, two labels at an equal gap of 2: the earlier label (by frame) wins", m9 == {"a": 1}, str(m9))


def brute(bursts, labels, events, tol):
    """The best (matches, -total gap) over every partial one-to-one assignment, by enumeration."""
    best = (0, 0)

    def go(i, used, n, g):
        nonlocal best
        if i == len(bursts):
            best = max(best, (n, -g))
            return
        go(i + 1, used, n, g)
        for li in labels:
            gap_ = completeness.span_gap(bursts[i][1], events[li])
            if li not in used and gap_ <= tol:
                go(i + 1, used | {li}, n + 1, g + gap_)

    go(0, frozenset(), 0, 0)
    return best


rng_m = np.random.default_rng(5)
bad_m, shown = 0, ""
for trial in range(400):
    nb_, nl_ = int(rng_m.integers(1, 6)), int(rng_m.integers(1, 6))
    bs_ = [(f"b{i}", int(rng_m.integers(0, 40))) for i in range(nb_)]
    evs_ = []
    for _ in range(nl_):
        a_ = int(rng_m.integers(0, 40))
        evs_.append(EV(a_, a_ + int(rng_m.integers(0, 12))))
    got_ = completeness._match(bs_, list(range(nl_)), evs_, 5, set())
    fr_ = dict(bs_)
    ok_ = (len(set(got_.values())) == len(got_)
           and all(completeness.span_gap(fr_[k_], evs_[v_]) <= 5 for k_, v_ in got_.items()))
    score_ = (len(got_), -sum(completeness.span_gap(fr_[k_], evs_[v_]) for k_, v_ in got_.items()))
    if not ok_ or score_ != brute(bs_, list(range(nl_)), evs_, 5):
        bad_m += 1
        shown = str((bs_, [(e_.first, e_.last) for e_ in evs_], got_))
check("400 random small instances: the matching is one-to-one, within the tolerance, and equals the brute-force optimum "
      "(most matches, then least total gap)", bad_m == 0, shown)
# Integrated through find_bursts: A's left hand bursts at ~303 and ~337 (34 frames apart).
two = base_scene(bursts={"A": [(300, 0), (335, 0)]})
t_2 = two.table()
ev_t = two.punches(("A", "B"))
ws_ = np.array([two.wrist("A", f_, 0) for f_ in range(302, 338)])
broad = olympic.Event(302, 337, "head", "left", (ws_[:, 0].min() - 10, ws_[:, 1].min() - 10, ws_[:, 0].max() + 10, ws_[:, 1].max() + 10), 1)
w298 = two.wrist("A", 298, 0)
narrow = olympic.Event(298, 298, "head", "left", (w298[0] - 8, w298[1] - 8, w298[0] + 8, w298[1] + 8), 1)
ev_t2 = ev_t + [broad, narrow]
at_t2 = olympic.attribute(ev_t2, t_2)
lk_t2 = linking.link_video(t_2, at_t2)
res_t2 = completeness.find_bursts(t_2, lk_t2, ev_t2, at_t2, RULE)
pt = [b for b in res_t2["bursts"] if 280 <= b.frame <= 340]
check("setup: two A bursts (303 and 337), a broad and a narrow label both credited to A's id 2, the narrow one 5 frames from the first burst",
      [b.frame for b in pt] == [303, 337] and [a_.track_id for a_ in at_t2[-2:]] == [2, 2]
      and completeness.span_gap(303, narrow) == 5 and completeness.span_gap(337, narrow) > 5
      and completeness.span_gap(303, broad) == 0 and completeness.span_gap(337, broad) == 0, str(pt))
check("in the scene: BOTH bursts are explained by the two owned labels (greedy left 337 unexplained)",
      [b.match for b in pt] == ["own", "own"] and res_t2["labels"]["matched_own"] == 2, str(pt))
# The documented LIMITATION, pinned: the matching is optimal within each phase (own labels first, then unattributed), not the
# global maximum. Same two bursts, but the narrow label is now UNATTRIBUTED. Owned phase: the one owned label ties (gap 0) for
# both bursts and goes to the first by (frame, boxer): 303. 337 is then out of reach of the unattributed label (gap > 5).
# Giving the owned label to 337 and the unattributed one to 303 would explain BOTH; this implementation deliberately does not.
narrow_u = olympic.Event(298, 298, "head", "left", (1, 1, 2, 2), 1)
ev_t3 = ev_t + [broad, narrow_u]
at_t3 = olympic.attribute(ev_t3, t_2)
res_t3 = completeness.find_bursts(t_2, linking.link_video(t_2, at_t3), ev_t3, at_t3, RULE)
pt3 = [b for b in res_t3["bursts"] if 280 <= b.frame <= 340]
check("setup: the broad label is A's, the narrow one unattributed, and a global assignment WOULD explain both bursts "
      "(narrow -> 303 at gap 5, broad -> 337 at gap 0)",
      at_t3[-2].track_id == 2 and at_t3[-1].track_id is None and [b.frame for b in pt3] == [303, 337]
      and completeness.span_gap(303, narrow_u) == 5 and completeness.span_gap(337, broad) == 0
      and completeness.span_gap(337, narrow_u) > 5, str(pt3))
check("LIMITATION pinned: the owned phase takes burst 303, so 337 stays unexplained and the unattributed label goes unused "
      "(optimal within each phase, not the global maximum)",
      [b.match for b in pt3] == ["own", "none"] and res_t3["labels"]["matched_own"] == 1
      and res_t3["labels"]["matched_unattributed"] == 0, str((pt3, res_t3["labels"])))
# In a scene: A's left burst at 300 with two A-owned labels 5 frames either side of its speed peak.
one = base_scene(bursts={"A": [(300, 0)]})
t_1 = one.table()
ev_1 = one.punches(("A", "B"))
res_1a = completeness.find_bursts(t_1, linking.link_video(t_1, olympic.attribute(ev_1, t_1)), ev_1, olympic.attribute(ev_1, t_1), RULE)
pk = [b for b in res_1a["bursts"] if 285 <= b.frame <= 315]
wl = one.wrist("A", 300, 0)
bx = (wl[0] - 10, wl[1] - 70, wl[0] + 140, wl[1] + 70)       # contains the left wrist from full extension (wl) to rest (wl + 120)
ev_2 = ev_1 + [olympic.Event(pk[0].frame - 9, pk[0].frame - 5, "head", "left", bx, 1),
               olympic.Event(pk[0].frame + 5, pk[0].frame + 9, "head", "left", bx, 1)]
at_2 = olympic.attribute(ev_2, t_1)
lk_1 = linking.link_video(t_1, at_2)
res_1 = completeness.find_bursts(t_1, lk_1, ev_2, at_2, RULE)
pk2 = [b for b in res_1["bursts"] if 285 <= b.frame <= 315]
check("setup: the burst is one burst, both extra labels are A's, both exactly 5 frames from its peak",
      len(pk) == 1 and len(pk2) == 1 and pk2[0].frame == pk[0].frame
      and all(a.track_id in A_IDS for a in at_2[-2:])
      and [completeness.span_gap(pk[0].frame, e) for e in ev_2[-2:]] == [5, 5], str((pk, pk2)))
check("in the scene: that one burst is matched to exactly one of the two labels",
      pk2[0].match == "own" and res_1["labels"]["matched_own"] == 1, str(res_1["labels"]))
# Two boxers burst together; ONE unattributed label near both: it explains one burst, not both.
uev = olympic.Event(494, 506, "head", "left", (100, 100, 116, 116), 1)
ev_u = ev_b + [uev]
at_u = olympic.attribute(ev_u, t_b)
res_u = completeness.find_bursts(t_b, linking.link_video(t_b, at_u), ev_u, at_u, RULE)
nu = bursts_near(res_u, 485, 515)
check("setup: that label is unattributed", at_u[-1].track_id is None)
check("an unattributed label near two boxers' coincident bursts is matched ONCE: one 'unattributed', one 'none'",
      sorted(b.match for b in nu) == ["none", "unattributed"] and res_u["labels"]["matched_unattributed"] == 1
      and sum(b.explained for b in nu) == 1, str(nu))
win_ = min(nu, key=lambda b: (b.frame, b.boxer))
check("... and on that tie (equal gap 0) the burst that sorts first by (frame, boxer) gets the label, not whichever boxer is named A: "
      "here B's, one frame earlier than A's",
      win_.match == "unattributed" and all(b.match == "none" for b in nu if b is not win_) and win_.boxer == "B", str(nu))
check("... and the unmatched one is counted as unexplained for its boxer",
      sum(res_u["boxers"][n]["unexplained"] for n in "AB") == 1, str(res_u["boxers"]))
check("the hand of an unattributed label is information only: it is recorded (label left) and the match stands either way",
      all(b.hand_agrees == (b.hand == "left") for b in nu if b.explained) and sum(b.explained for b in nu) == 1)

print("completeness: no signal across a fragment join (Astra 6, finding 3)")


def two_fragment_table(n=400, split=200, shift=40.0, burst_at=None, drop=(), left_conf=0.9, n2=None):
    """One stationary person, id 1 until `split`, id 2 after; id 2's wrists sit `shift` px (one torso) further out
    relative to the shoulders. `burst_at`: a real right-hand punch inside id 1."""
    rows = []
    for f in range(n if n2 is None else n2):
        if f in drop:
            continue
        e = 120.0 * max(0.0, 1 - abs(f - burst_at) / 4) if burst_at is not None else 0.0
        xy, box = pose(300, 200, ext=(0.0, e))
        tid = 1 if f < split else 2
        if tid == 2:
            xy[9, 0] += shift
            xy[10, 0] += shift
        rows.append((f, tid, box, xy))
    k = len(rows)
    cf = np.full((k, N_JOINTS), 0.9, np.float32)
    cf[:, 9] = left_conf
    return PoseTable(np.array([r[0] for r in rows], np.int64), np.array([r[1] for r in rows], np.int64),
                     np.array([r[2] for r in rows], np.float32).reshape(k, 4),
                     np.array([r[3] for r in rows], np.float32).reshape(k, N_JOINTS, 2), cf,
                     n, FPS, W, H, source="GH000001.mp4")


t_j = two_fragment_table()
lk_j = manual_linking(t_j, [(1, 2), ()])
joined = completeness.boxer_track(t_j, lk_j, 0)
sg_j = punchfind.wrist_signal(joined, "relative")
jf, _ = punchfind.detect(joined, sg_j, RULE["threshold"], RULE["min_gap_s"])
check("setup: JOINED into one track the handover reads as a punch (this is the fault: both fragments are stationary)",
      any(195 <= int(f) <= 205 for f in jf) and sg_j.max() > 5 * RULE["threshold"] / 3, str((jf, sg_j.max())))
res_j = completeness.find_bursts(t_j, lk_j, [], [], RULE)
check("per SEGMENT there is no burst at all: nothing is differenced across the handover",
      res_j["bursts"] == [] and res_j["boxers"]["A"]["segments"] == 2 and res_j["boxers"]["A"]["short_segments"] == 0,
      str(res_j["boxers"]["A"]))
t_r = two_fragment_table(burst_at=100)
res_r = completeness.find_bursts(t_r, manual_linking(t_r, [(1, 2), ()]), [], [], RULE)
check("a real punch inside one fragment is still found, exactly one, at the right frame and hand",
      len(res_r["bursts"]) == 1 and abs(res_r["bursts"][0].frame - 100) <= 6 and res_r["bursts"][0].hand == "right"
      and res_r["bursts"][0].boxer == "A", str(res_r["bursts"]))
BRIDGE_F = int(round(punchfind.BRIDGE_S * FPS))
check("setup: the rule bridges 2 missing frames at 50 fps", BRIDGE_F == 2, str(BRIDGE_F))
t_g = two_fragment_table(n=210, split=200, drop=(100, 101), n2=203)
res_g = completeness.find_bursts(t_g, manual_linking(t_g, [(1, 2), ()]), [], [], RULE)
check("a hole of 2 frames inside ONE id is bridged, not a segment end (same person, as on the tuning footage); "
      "the 3-frame fragment after the handover is under the smoothing window, skipped and counted",
      res_g["boxers"]["A"]["segments"] == 2 and res_g["boxers"]["A"]["short_segments"] == 1
      and abs(res_g["boxers"]["A"]["identified_min"] - (198 + 3) / FPS / 60) < 1e-12
      # 198 rows + the 2 bridged frames carry a signal: the bridge really ran inside the segment
      and abs(res_g["boxers"]["A"]["exposure_min"] - 200 / FPS / 60) < 1e-12, str(res_g["boxers"]["A"]))
t_g4 = two_fragment_table(n=210, split=200, drop=(100, 101, 102), n2=203)
res_g4 = completeness.find_bursts(t_g4, manual_linking(t_g4, [(1, 2), ()]), [], [], RULE)
check("a hole of 3 frames (longer than the bridge) inside one id DOES end a segment",
      res_g4["boxers"]["A"]["segments"] == 3 and res_g4["boxers"]["A"]["short_segments"] == 1, str(res_g4["boxers"]["A"]))
lk_g = manual_linking(t_g, [(1, 2), ()])
check("boxer_segments: an id change splits even with no missing frame and a generous max_gap",
      [len(x) for x in completeness.boxer_segments(t_j, lk_j, 0, max_gap=50)] == [200, len(t_j.frame) - 200],
      str([len(x) for x in completeness.boxer_segments(t_j, lk_j, 0, max_gap=50)]))
raises("boxer_segments refuses a negative max_gap", lambda: completeness.boxer_segments(t_g, lk_g, 0, max_gap=-1), "max_gap")
raises("boxer_segments refuses a bool max_gap", lambda: completeness.boxer_segments(t_g, lk_g, 0, max_gap=True), "max_gap")
res_bl = completeness.find_bursts(dataclasses.replace(t_j, conf=np.zeros_like(t_j.conf)), lk_j, [], [], RULE)
check("... and a segment with no confident body at all is counted as bodyless, not a crash",
      res_bl["boxers"]["A"]["bodyless_segments"] == 2 and res_bl["bursts"] == [] and res_bl["boxers"]["A"]["exposure_min"] == 0.0)

print("completeness: signal minutes are elapsed minutes with EITHER hand valid (Astra 6, finding 5)")
t_h = two_fragment_table(n=400, split=400, left_conf=0.05, burst_at=200)
lk_h = manual_linking(t_h, [(1,), ()])
res_h = completeness.find_bursts(t_h, lk_h, [], [], RULE)
bi = res_h["boxers"]["A"]
check("the left wrist is invalid all the time, the right all the time: signal minutes = the 400 elapsed frames (not half)",
      abs(bi["exposure_min"] - 400 / FPS / 60) < 1e-12 and bi["signal_min_left"] == 0.0
      and abs(bi["signal_min_right"] - 400 / FPS / 60) < 1e-12, str(bi))
check("... with the burst counted for the right hand only",
      bi["bursts_by_hand"] == {"left": 0, "right": 1} and bi["unexplained_by_hand"] == {"left": 0, "right": 1}, str(bi))
trk_h = completeness.boxer_track(t_h, lk_h, 0)
sf = punchfind.signal_frames(trk_h)
cov = punchfind.signal_coverage(trk_h, span=(0, 399))
check("signal_frames agrees with signal_coverage, hand by hand (one definition of where a hand has a signal)",
      abs(sf[:, 0].sum() / 400 - cov["left"]) < 1e-12 and abs(sf[:, 1].sum() / 400 - cov["right"]) < 1e-12 and sf[:, 1].all()
      and not sf[:, 0].any())

print("completeness: the -1 sentinel never selects rows (Astra 6, finding 9)")
t_n = sc.table()
t_n.track_id[t_n.track_id == 31] = -1          # a malformed table that bypassed validation: the spectator became id -1
check("setup: the table really holds id -1 rows", int((t_n.track_id == -1).sum()) == N)
lk_none = manual_linking(t_n, [(), ()])
check("an EMPTY linking selects no rows for either boxer, although -1 == -1",
      all(len(completeness.boxer_rows(t_n, lk_none, k)) == 0 for k in (0, 1)))
res_n = completeness.find_bursts(t_n, lk_none, [], [], RULE)
check("... so no bursts and no minutes", res_n["bursts"] == [] and res_n["boxers"]["A"]["identified_min"] == 0.0
      and res_n["boxers"]["B"]["identified_min"] == 0.0)
lk_real = manual_linking(t_n, [(1, 2, 3, 4), (11, 12, 13, 14)])
check("a real linking beside -1 rows never selects one",
      all((t_n.track_id[completeness.boxer_rows(t_n, lk_real, k)] >= 0).all() and len(completeness.boxer_rows(t_n, lk_real, k)) > 0
          for k in (0, 1)))
raises("PoseTable refuses a negative track id at construction",
       lambda: dataclasses.replace(t_n), "negative track id")
np.savez(tmp / "neg.npz", frame=np.array([0, 1]), track_id=np.array([1, -1]), box=np.zeros((2, 4), np.float32),
         xy=np.zeros((2, N_JOINTS, 2), np.float32), conf=np.zeros((2, N_JOINTS), np.float32), n_frames=5, fps=50.0,
         width=W, height=H, source="x", model="x")
raises("... and loading a file that holds one", lambda: PoseTable.load(tmp / "neg.npz"), "negative track id")

print("completeness: every rate divides by the EXPOSURE, unrounded (Astra 6, findings 4 and 1 of the second review)")


def fake_res(expo_a, expo_b, ident_a, ident_b, in_a, in_b, out_a, out_b, unattr, n_unexpl=1):
    z = {"signal_min_left": 0.0, "signal_min_right": 0.0, "segments": 1, "short_segments": 0, "bodyless_segments": 0,
         "note": "", "labels_matched_own": 0, "bursts_by_hand": {"left": n_unexpl, "right": 0},
         "unexplained_by_hand": {"left": n_unexpl, "right": 0}, "bursts": n_unexpl, "unexplained": n_unexpl}

    def box(expo, ident, inn, out):
        return z | {"exposure_min": expo, "exposure_frames": round(expo * 3000), "identified_min": ident,
                    "identified_frames": round(ident * 3000), "own_labels": inn + out, "own_labels_in_exposure": inn,
                    "own_labels_outside_exposure": out}

    return {"bursts": [completeness.Burst("V", i, 0.1 * i, "A", "left", 5.0) for i in range(2 * n_unexpl)],
            "boxers": {"A": box(expo_a, ident_a, in_a, out_a), "B": box(expo_b, ident_b, in_b, out_b)},
            "labels": {"total": in_a + out_a + in_b + out_b + unattr, "owned_a": in_a + out_a, "owned_b": in_b + out_b,
                       "unattributed": unattr, "unattributed_in_exposure": 0, "matched_own": 0, "matched_unattributed": 0}}


# A one-minute video, 60 labels; each boxer identified 0.5 min but the rule ran on only 0.1 min of it. Of each boxer's
# 23 own labels only 3 touch the exposure; 14 labels are unattributed.
row_a = olympic_completeness.video_row(fake_res(0.1, 0.1, 0.5, 0.5, 3, 3, 20, 20, 14), 60, 1.0)
tot_a = olympic_completeness.totals([row_a])
check("own labels IN exposure per exposure minute = 6 / 0.2 = 30 (not 46 / 0.2, not over the 1.0 identified minutes)",
      abs(tot_a["own_labels_in_exposure_per_exposure_min"] - 30.0) < 1e-9 and tot_a["own_labels_in_exposure_boxers"] == 6
      and tot_a["own_labels_outside_exposure_boxers"] == 40, str(tot_a))
check("unexplained and bursts per exposure minute = 2 / 0.2 = 10: the identified 1.0 min (which would give 2) divides nothing",
      abs(tot_a["unexplained_per_exposure_min"] - 10.0) < 1e-9 and abs(tot_a["bursts_per_exposure_min"] - 10.0) < 1e-9
      and abs(tot_a["identified_boxer_min"] - 1.0) < 1e-12 and abs(tot_a["exposure_boxer_min"] - 0.2) < 1e-12, str(tot_a))
check("per boxer too: 1 / 0.1 = 10", abs(tot_a["A"]["unexplained_per_exposure_min"] - 10.0) < 1e-9
      and abs(row_a["boxers"]["B"]["unexplained_per_exposure_min"] - 10.0) < 1e-9)
check("labels per VIDEO boxer-minute = 60 / (2 x 1 min) = 30, a separate population; 14 of 60 unattributed",
      abs(tot_a["labels_per_video_boxer_min"] - 30.0) < 1e-9 and tot_a["labels"]["unattributed"] == 14
      and abs(tot_a["labels"]["unattributed_share"] - 14 / 60) < 1e-12 and tot_a["labels"]["total"] == 60)


def keys(x, out=None):
    out = set() if out is None else out
    if isinstance(x, dict):
        for k_, v_ in x.items():
            out.add(k_)
            keys(v_, out)
    return out


check("no rate anywhere in the output is named for, or divided by, the identified minutes",
      not any("per_identified" in k_ for k_ in keys(tot_a) | keys(row_a)))
check("the old mismatched rates are gone from the output",
      "labelled_punches_per_identified_boxer_min" not in tot_a and "own_labels_per_identified_boxer_min" not in tot_a)
# Rounding comes last: two videos with 0.0004 exposure minutes per boxer each round to 0.0 at 3 decimals.
rows_u = [olympic_completeness.video_row(fake_res(0.0004, 0.0004, 0.0004, 0.0004, 1, 1, 0, 0, 0), 2, 0.5) for _ in range(2)]
tot_u = olympic_completeness.totals(rows_u)
check("aggregation runs on UNROUNDED minutes: 4 unexplained over 0.0016 exposure min = 2500 (rounded first it would be n/a)",
      abs(tot_u["unexplained_per_exposure_min"] - 4 / 0.0016) < 1e-6, str(tot_u["unexplained_per_exposure_min"]))
check("... and only the OUTPUT is rounded", olympic_completeness._round(tot_u)["exposure_boxer_min"] == 0.002
      and olympic_completeness._round(tot_u)["unexplained_per_exposure_min"] == 2500.0)
check("the per-video rows keep floats unrounded", rows_u[0]["boxers"]["A"]["exposure_min"] == 0.0004)
tot_0 = olympic_completeness.totals([olympic_completeness.video_row(fake_res(0.0, 0.0, 0.0, 0.0, 0, 0, 3, 3, 0, n_unexpl=0), 6, 1.0)])
check("no exposure at all: the rates are None (not a crash, not zero), the counts are exact",
      tot_0["unexplained_per_exposure_min"] is None and tot_0["bursts_per_exposure_min"] is None
      and tot_0["own_labels_in_exposure_per_exposure_min"] is None and tot_0["own_labels_outside_exposure_boxers"] == 6)

print("completeness: numerator and denominator are the same observation, through find_bursts (second review, finding 1)")
# (a) a burst whose peak lands on a frame the bridge INTERPOLATED (frames 101 and 102 are missing from the table)
t_x = two_fragment_table(n=400, split=400, burst_at=100, drop=(101, 102))
res_x = completeness.find_bursts(t_x, manual_linking(t_x, [(1,), ()]), [], [], RULE)
ax = res_x["boxers"]["A"]
check("setup: the peak is at frame 102, a frame with NO row in the table",
      [b.frame for b in res_x["bursts"]] == [102] and 102 not in set(t_x.frame.tolist()), str(res_x["bursts"]))
check("(a) the bridged-frame peak is in the numerator AND in the exposure: 398 identified rows, 400 exposure frames",
      ax["bursts"] == 1 and ax["unexplained"] == 1 and ax["identified_frames"] == 398 and ax["exposure_frames"] == 400
      and bool(res_x["exposure"]["A"][102]) and res_x["exposure"]["A"].sum() == 400
      and abs(ax["exposure_min"] - 400 / FPS / 60) < 1e-12 and ax["exposure_min"] > ax["identified_min"], str(ax))
ex_ids = {n_: int(res_x["exposure"][n_].sum()) for n_ in "AB"}
check("... and boxer B (nobody) has an empty exposure", ex_ids == {"A": 400, "B": 0})
# the guard itself: if the exposure and the detector ever disagree, the run stops
real_sf = punchfind.signal_frames
punchfind.signal_frames = lambda track, *a_, **k_: np.zeros((len(track.frame_ids), 2), dtype=bool)
try:
    completeness.find_bursts(t_x, manual_linking(t_x, [(1,), ()]), [], [], RULE)
    raised = None
except RuntimeError as e_:
    raised = str(e_)
finally:
    punchfind.signal_frames = real_sf
check("a burst outside the exposure it is counted against is a RuntimeError, not a quiet mismatch",
      raised is not None and "outside the exposure" in raised, str(raised))
# (b) and (c): a 3-row second fragment is too short for the rule; labels on it and on the long fragment
t_s = two_fragment_table(n=210, split=200, n2=203)


def label_on(tab, first, last, tid, hand=0):
    r = np.flatnonzero((tab.frame == (first + last) // 2) & (tab.track_id == tid))[0]
    w = tab.xy[r, 9 + hand]
    return olympic.Event(first, last, "head", ("left", "right")[hand], (w[0] - 8, w[1] - 8, w[0] + 8, w[1] + 8), 1)


ev_s = [label_on(t_s, 100, 102, 1), label_on(t_s, 200, 202, 2), olympic.Event(150, 152, "head", "left", (1, 1, 2, 2), 1),
        olympic.Event(205, 207, "head", "left", (1, 1, 2, 2), 1)]
at_s = olympic.attribute(ev_s, t_s)
lk_s = manual_linking(t_s, [(1, 2), ()])
res_s = completeness.find_bursts(t_s, lk_s, ev_s, at_s, RULE)
bs = res_s["boxers"]["A"]
check("setup: labels 0 and 1 are credited to ids 1 and 2 and both are boxer A's; labels 2 and 3 are unattributed",
      [a_.track_id for a_ in at_s] == [1, 2, None, None] and [completeness.event_owner(a_, lk_s) for a_ in at_s] == [0, 0, None, None])
check("(b) the skipped 3-row segment adds to identified minutes (203 rows) but NOT to the exposure (200 frames)",
      bs["short_segments"] == 1 and bs["identified_frames"] == 203 and bs["exposure_frames"] == 200
      and bs["identified_min"] > bs["exposure_min"] and int(res_s["exposure"]["A"][200:].sum()) == 0, str(bs))
check("(c) the owned label wholly inside the skipped segment is OUTSIDE the exposure, the other is inside",
      bs["own_labels"] == 2 and bs["own_labels_in_exposure"] == 1 and bs["own_labels_outside_exposure"] == 1, str(bs))
check("unattributed labels: the one at 150-152 touches the exposure, the one at 205-207 (the skipped segment) does not",
      res_s["labels"]["unattributed"] == 2 and res_s["labels"]["unattributed_in_exposure"] == 1, str(res_s["labels"]))
rates_s = olympic_completeness.video_row(res_s, len(ev_s), 210 / FPS / 60)["boxers"]["A"]
check("the video row's own-labels rate counts only the label in the exposure: 1 over the 200 exposure frames, not 2 over 203",
      abs(rates_s["own_labels_in_exposure_per_exposure_min"] - 1 / (200 / FPS / 60)) < 1e-9, str(rates_s))

print("completeness: the review sample is a simple random sample (Astra 6, finding 6)")
mk = lambda v, n: [completeness.Burst(v, 10 * i, 0.2 * i, "AB"[i % 2], "left", 5.0) for i in range(n)]  # noqa: E731
small = {"V1": mk("V1", 5), "V2": mk("V2", 3), "V3": mk("V3", 4), "V4": []}
s1, pop1 = completeness.random_sample(small, 4, 7)
check("the sample has exactly 4 distinct bursts out of a population of 12",
      len(s1) == 4 and pop1 == 12 and len({(b.video, b.frame, b.boxer) for b in s1}) == 4)
check("the same seed gives the same sample, whatever the dict order",
      s1 == completeness.random_sample(dict(reversed(list(small.items()))), 4, 7)[0])
check("a different seed gives a different sample", s1 != completeness.random_sample(small, 4, 8)[0])
cnt, vids_seen = {}, set()
for sd in range(2000):
    smp, _ = completeness.random_sample(small, 4, sd)
    for b in smp:
        cnt[(b.video, b.frame)] = cnt.get((b.video, b.frame), 0) + 1
        vids_seen.add(b.video)
p_ = 4 / 12
se = (p_ * (1 - p_) / 2000) ** 0.5
rates = {k: v / 2000 for k, v in cnt.items()}
check("over 2000 seeds all 12 bursts appear, and every burst's inclusion rate is within 3 standard errors of n/N = 1/3",
      len(rates) == 12 and all(abs(r - p_) <= 3 * se for r in rates.values()),
      str(sorted((abs(r - p_) / se for r in rates.values()))[-3:]))
check("... every video can appear (none has zero inclusion probability), and V4 with nothing never does",
      vids_seen == {"V1", "V2", "V3"})
# Astra's case: 60 equal videos and a sample of 40: the old quota always took the first 40 video names.
many = {f"V{i:02d}": mk(f"V{i:02d}", 3) for i in range(60)}
seen_many, firsts = {}, 0
for sd in range(600):
    smp, popm = completeness.random_sample(many, 40, sd)
    for v in {b.video for b in smp}:
        seen_many[v] = seen_many.get(v, 0) + 1
check("60 videos x 3 bursts, sample 40: across seeds EVERY video is sampled, and none is sampled always or never "
      "(the old quota gave the first 40 names every time)",
      len(seen_many) == 60 and 0 < min(seen_many.values()) and max(seen_many.values()) < 600 and popm == 180,
      str((len(seen_many), min(seen_many.values()), max(seen_many.values()))))
tiny_s, tiny_n = completeness.random_sample({"V": mk("V", 5)}, 40, 7)
check("fewer bursts than the sample size: all of them, nothing invented, population 5", len(tiny_s) == 5 and tiny_n == 5)
check("nothing unexplained: an empty sample, population 0", completeness.random_sample({"V": []}, 40, 7) == ([], 0))
raises("a zero sample size is refused", lambda: completeness.random_sample(small, 0, 7), "whole number")
raises("a boolean sample size is refused", lambda: completeness.random_sample(small, True, 7), "whole number")

print("olympic_completeness.py end to end")
(task / "annotations.json").write_text(json.dumps(cvat(lab)), encoding="utf-8")
cs.table().save(trk / "GH000001.tracks.npz")
rule_f = tmp / "rule.json"
rule_f.write_text(json.dumps({"chosen": RULE}), encoding="utf-8")
cj, cc = tmp / "res" / "comp.json", tmp / "res" / "sample.csv"
args = ["--data", str(ds), "--tracks", str(trk), "--rule", str(rule_f), "--out", str(cj), "--sample-out", str(cc),
        "--sample-size", "2"]
rc = olympic_completeness.main(args)
comp = json.loads(cj.read_text(encoding="utf-8"))
csv1 = cc.read_text(encoding="utf-8")
ca = comp["all"]
check("exit 0; the totals are the planted ones (3 bursts, 2 unexplained)",
      rc == 0 and ca["bursts"] == 3 and ca["unexplained"] == 2, str(ca))
check("REAL FACT: unexplained bursts per exposure minute is a positive rate from real minutes",
      ca["unexplained_per_exposure_min"] > 0 and ca["A"]["exposure_min"] > 0.2 and ca["A"]["identified_min"] >= ca["A"]["exposure_min"])
check("REAL FACT: the 32 credited labels are all owned, some touch the exposure and the rest do not; the rate is over the "
      "exposure minutes; whole-video labels per boxer-minute (a different population) are 33 over 2 x 20 s",
      ca["labels"]["total"] == 33 and ca["labels"]["unattributed"] == 1 and ca["own_labels_boxers"] == 32
      and ca["own_labels_in_exposure_boxers"] + ca["own_labels_outside_exposure_boxers"] == 32
      and ca["own_labels_in_exposure_boxers"] > 0
      and abs(ca["own_labels_in_exposure_per_exposure_min"] - ca["own_labels_in_exposure_boxers"] / ca["exposure_boxer_min"]) < 0.01
      and abs(ca["labels_per_video_boxer_min"] - 33 / (2 * 1000 / FPS / 60)) < 0.01 and ca["labels"]["unattributed_share"] == round(1 / 33, 3),
      str(ca))
check("the bursts split by how they matched: 1 unattributed, 2 none", ca["bursts_by_match"] == {"own": 0, "unattributed": 1, "none": 2})
check("the output no longer calls the count a bound or a recall",
      "events_with_burst" not in json.dumps(comp) and "labelled_with_a_burst" not in json.dumps(comp)
      and "upper bound" not in json.dumps(comp) and "recall" not in json.dumps(comp))
check("the sample records its population and design", comp["sample"]["population_unexplained"] == 2
      and comp["sample"]["n"] == 2 and comp["sample"]["inclusion_prob"] == 1.0 and "simple random" in comp["sample"]["design"])
check("the report says the table is not content-bound (a legacy table)", comp["videos"]["task_kam2_gh000001"]["video_bound"] is False)
lines = csv1.strip().splitlines()
check("the sample file lists video, frame, time, boxer, hand, the inclusion probability and an EMPTY label column",
      lines[0] == "video,frame,time_s,boxer,hand,inclusion_prob,user_label" and len(lines) == 3
      and all(l.endswith(",") for l in lines[1:]) and all(l.startswith("GH000001,") for l in lines[1:])
      and all(l.split(",")[5] == "1.000000" for l in lines[1:]), csv1)
args1 = args[:-1] + ["1"]
olympic_completeness.main(args1)
csv_one = cc.read_text(encoding="utf-8").strip().splitlines()
check("a sample of 1 from the 2 unexplained bursts has inclusion probability 1/2 in every row",
      len(csv_one) == 2 and csv_one[1].split(",")[5] == "0.500000", str(csv_one))
olympic_completeness.main(args)
check("running it again writes the identical sample (reproducible)", cc.read_text(encoding="utf-8") == csv1)
print("olympic_completeness.py: a video with labels but no eligible identity runs to the end (second review, finding 4)")
t_shred = cs.table()      # every person re-identified every 5 frames (0.1 s, under the 0.2 s minimum): credited punches, but no fragment is long enough
dataclasses.replace(t_shred, track_id=t_shred.track_id * 1000 + t_shred.frame // 5).save(trk / "GH000001.tracks.npz")
cj0, cc0 = tmp / "res" / "comp0.json", tmp / "res" / "sample0.csv"
import contextlib  # noqa: E402
import io  # noqa: E402
buf0 = io.StringIO()
with contextlib.redirect_stdout(buf0):
    rc0 = olympic_completeness.main(["--data", str(ds), "--tracks", str(trk), "--rule", str(rule_f), "--out", str(cj0),
                                     "--sample-out", str(cc0)])
c0 = json.loads(cj0.read_text(encoding="utf-8"))
a0 = c0["all"]
check("setup: 33 labels, but no fragment is long enough to be a boxer, so neither boxer is identified at all",
      a0["labels"]["total"] == 33 and a0["A"]["identified_frames"] == 0 and a0["B"]["identified_frames"] == 0
      and a0["exposure_boxer_frames"] == 0, str(a0))
check("it runs to the end with exit code 0 and writes both files", rc0 == 0 and cj0.is_file() and cc0.is_file())
check("the rates are null in the json (absent, not zero), the counts are exact zeros",
      a0["unexplained_per_exposure_min"] is None and a0["bursts_per_exposure_min"] is None
      and a0["own_labels_in_exposure_per_exposure_min"] is None and a0["bursts"] == 0 and a0["unexplained"] == 0)
check("... and they print as n/a, the unattributed share still printed (all 33 labels are unattributed)",
      "n/a per exposure minute" in buf0.getvalue() and "33 of 33" in buf0.getvalue(), buf0.getvalue())
check("the sample is empty: header only", cc0.read_text(encoding="utf-8").strip() == "video,frame,time_s,boxer,hand,inclusion_prob,user_label"
      and c0["sample"]["population_unexplained"] == 0 and c0["sample"]["inclusion_prob"] is None)
print("olympic_link.py: no held-out observations is not 0% (third review, B2)")
ev_few = ev_c[::2]                                   # 16 labels: enough for a ring, 8 per half: too few for the held-out ring
(task / "annotations.json").write_text(json.dumps(cvat(ev_few)), encoding="utf-8")
cs.table().save(trk / "GH000001.tracks.npz")
tk_u = olympic.find_tasks(ds)[0]
tb_u, _, evu, atu = olympic_link.load_video(tk_u, trk)
lk_u, out_u = olympic_link.analyse(tb_u, atu)
check("setup: 16 labels link, and the held-out half cannot estimate a ring: no share, no evaluable punch, ring None",
      len(evu) == 16 and out_u["credited_heldout"] == 0 and out_u["credited_in_either_share_heldout"] is None
      and out_u["credited_heldout_inside"] == 0 and out_u["ring_heldout"] is None and out_u["credited_in_either_share"] > 0, str(out_u))
cl_u, bufu = tmp / "res" / "link_unevaluable.json", io.StringIO()
with contextlib.redirect_stdout(bufu):
    rcu = olympic_link.main(["--data", str(ds), "--tracks", str(trk), "--out", str(cl_u)])
lu = json.loads(cl_u.read_text(encoding="utf-8"))
check("an all-unevaluable pool: exit 0, the pooled held-out share is null (not 0.0), the counts are exact zeros",
      rcu == 0 and lu["all"]["credited_in_either_share_heldout"] is None and lu["all"]["credited_heldout"] == 0
      and lu["all"]["credited_heldout_inside"] == 0, str(lu["all"]))
check("... the per-video row says null too, and the CLI prints n/a, never 0.0%",
      lu["videos"]["task_kam2_gh000001"]["credited_in_either_share_heldout"] is None
      and "n/a with half held out" in bufu.getvalue() and "0.0% with half" not in bufu.getvalue()
      and "n/a held out" in bufu.getvalue(), bufu.getvalue())
check("pooled_heldout_share: an empty pool and an all-zero pool are None; a pool with data is a number",
      olympic_link.pooled_heldout_share([]) is None
      and olympic_link.pooled_heldout_share([{"credited_heldout_inside": 0, "credited_heldout": 0}]) is None
      and olympic_link.pooled_heldout_share([{"credited_heldout_inside": 0, "credited_heldout": 4}]) == 0.0)
at_full = olympic.attribute(ev_c, tb_u)
at_odd_blank = [dataclasses.replace(a_, track_id=None) if i_ % 2 else a_ for i_, a_ in enumerate(at_full)]
_, out_blank = olympic_link.analyse(tb_u, at_odd_blank)
check("a video whose held-out half has a ring but NO credited punch: share None too, evaluable count 0 (not 0%)",
      out_blank["credited_in_either_share_heldout"] is None and out_blank["credited_heldout"] == 0
      and out_blank["ring_heldout"] is not None, str(out_blank))

raises("a missing rule file is refused", lambda: olympic_completeness.main(args[:6] + ["--rule", str(tmp / "nope.json")] + args[8:]),
       "no tuned rule")

print("summarise: no credited punch is no measurement (None), never 0%")
from types import SimpleNamespace  # noqa: E402
st_none = linking.summarise(lk_j, [SimpleNamespace(track_id=None)] * 3)
check("three uncredited punches: the share is None, credited_punches 0",
      st_none["credited_in_either_share"] is None and st_none["credited_punches"] == 0, str(st_none))
st_one = linking.summarise(lk_j, [SimpleNamespace(track_id=1), SimpleNamespace(track_id=99)])
check("one of two credited punches on boxer A: the share is exactly 0.5",
      st_one["credited_in_either_share"] == 0.5 and st_one["credited_in_a"] == 1, str(st_one))
st_zero = linking.summarise(lk_j, [SimpleNamespace(track_id=99)])
check("a credited punch on neither boxer: the share is a measured 0.0, not None",
      st_zero["credited_in_either_share"] == 0.0, str(st_zero))

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
