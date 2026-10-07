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
from boxingvi import completeness, linking, olympic, overlay, video  # noqa: E402
from boxingvi.data import N_JOINTS  # noqa: E402
from boxingvi.linking import Fragment, LinkParams, Region  # noqa: E402
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

    def table(self, source="GH000001.MP4", fps=FPS, conf=0.9):
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
check("... even if ring time were valued 10x: a chain with no credited punch is emptied, not reported",
      lk_nob5.ids(0) == A_IDS and lk_nob5.identities[1] == (), str(lk_nob5.identities))
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
PoseTable(table.frame, table.track_id, table.box, table.xy, table.conf, N, 25.0, W, H).save(bad_trk / "GH000001.tracks.npz")
raises("a tracks file from another video (frame rate) links nothing, loudly",
       lambda: olympic_link.main(["--data", str(ds), "--tracks", str(bad_trk), "--out", str(tmp / "x.json")]), "could be linked")
short_trk = tmp / "short_tracks"
short_trk.mkdir()
PoseTable(table.frame[table.frame < 500], table.track_id[table.frame < 500], table.box[table.frame < 500],
          table.xy[table.frame < 500], table.conf[table.frame < 500], 500, FPS, W, H).save(short_trk / "GH000001.tracks.npz")
raises("an incomplete tracks file (a --max-frames run) is refused, not linked as if it were whole",
       lambda: olympic_link.main(["--data", str(ds), "--tracks", str(short_trk), "--out", str(tmp / "y.json")]), "could be linked")
for bad_id in ("../GH000001", "GH000001/../../x", "GH 1", "", "GH000001.mp4"):
    raises(f"--overlay {bad_id!r} is not a GoPro id: refused before anything is written",
           lambda b=bad_id: olympic_link.main(["--data", str(ds), "--tracks", str(trk), "--out", str(tmp / "z.json"),
                                               "--overlay", b]), "GoPro id")
raises("--overlay of a video that is not in the dataset is refused",
       lambda: olympic_link.main(["--data", str(ds), "--tracks", str(trk), "--out", str(tmp / "z.json"), "--overlay", "GH123456"]),
       "no such video")

print("completeness: wrist-speed bursts per linked boxer, away from every label")
RULE = {"variant": "relative", "threshold": 3.62, "min_gap_s": 0.4}
# Planted bursts: A's right wrist at 100 (nothing labelled near), A's left at 300 (a label at 302-304),
# B's right at 500 (nothing near), and the REFEREE's at 250 (not a boxer: must not be counted).
cs = base_scene(bursts={"A": [(100, 1), (300, 0)], "B": [(500, 1)], "ref": [(250, 0)]})
t_c = cs.table()
ev_c = cs.punches(("A", "B"))
attrs_c = olympic.attribute(ev_c, t_c)
lk_c = linking.link_video(t_c, attrs_c)
lab = ev_c + [olympic.Event(302, 304, "head", "left", (100, 100, 116, 116), 1)]
res = completeness.find_bursts(t_c, lk_c, lab, RULE, video="GH000001")
kA = "A" if lk_c.ids(0) == A_IDS else "B"      # which letter the linking gave the planted boxer A
fa = sorted((b.frame, b.hand, b.explained) for b in res["bursts"] if b.boxer == kA)
fb = sorted((b.frame, b.hand, b.explained) for b in res["bursts"] if b.boxer != kA)
near = lambda got, want: len(got) == len(want) and all(abs(g[0] - w[0]) <= 6 and g[1:] == w[1:] for g, w in zip(got, want))  # noqa: E731
check("A: both planted bursts are found at the right frame (+-6, the speed peak is on the rise) and hand; the one next to a label is explained",
      near(fa, [(100, "right", False), (300, "left", True)]), str(fa))
check("B: its planted burst is found and unexplained", near(fb, [(500, "right", False)]), str(fb))
check("the referee's burst (not an identity) is not counted for anybody",
      not any(240 <= b.frame <= 260 for b in res["bursts"]))
check("per-boxer counts and minutes: identified minutes are the identified frames, unexplained counted",
      res["boxers"][kA]["unexplained"] == 1 and res["boxers"][kA]["bursts"] == 2
      and abs(res["boxers"][kA]["identified_min"] - int(lk_c.identified(0 if kA == "A" else 1).sum()) / FPS / 60) < 1e-9
      and res["boxers"][kA]["signal_min"] > 0)
check("the labelled punches the rule also sees: exactly the one with a burst within 0.1 s",
      res["events_with_burst"] == 1, str(res["events_with_burst"]))
check("a burst's time is its frame over the frame rate", all(abs(b.time_s - b.frame / FPS) < 1e-9 for b in res["bursts"]))
quiet = base_scene()
t_q = quiet.table()
lk_q = linking.link_video(t_q, olympic.attribute(quiet.punches(("A", "B")), t_q))
res_q = completeness.find_bursts(t_q, lk_q, quiet.punches(("A", "B")), RULE)
check("no planted burst, no burst: a boxer who only sways gives none", res_q["bursts"] == [])
check("the tolerance is 0.1 s = 5 frames: a label 5 frames away explains a burst, 6 do not",
      completeness._near_label(np.array([100]), [olympic.Event(105, 107, "head", "left", (1, 1, 2, 2), 1)], 5)[0]
      and not completeness._near_label(np.array([100]), [olympic.Event(106, 108, "head", "left", (1, 1, 2, 2), 1)], 5)[0]
      and completeness._near_label(np.array([100]), [olympic.Event(90, 95, "head", "left", (1, 1, 2, 2), 1)], 5)[0]
      and not completeness._near_label(np.array([100]), [olympic.Event(80, 94, "head", "left", (1, 1, 2, 2), 1)], 5)[0])
check("a label ANYWHERE in the video explains a burst (not just the boxer's own)",
      completeness._near_label(np.array([300]), [olympic.Event(302, 304, "head", "left", (1, 1, 2, 2), 1)], 5)[0])
raises("a linking from another table is refused", lambda: completeness.boxer_track(t_c, dataclasses.replace(lk_c, source="x"), 0),
       "different pose table")
check("a boxer with under two identified frames has no track (None), not a crash",
      completeness.boxer_track(t_c, dataclasses.replace(lk_c, membership=np.full_like(lk_c.membership, -1)), 0) is None)
raises("a negative tolerance is refused", lambda: completeness.find_bursts(t_c, lk_c, lab, RULE, tol_s=-1), "tol_s")

print("completeness: the random sample")
mk = lambda v, n: [completeness.Burst(v, 10 * i, 0.2 * i, "AB"[i % 2], "left", 5.0, False) for i in range(n)]  # noqa: E731
pool = {"V1": mk("V1", 100), "V2": mk("V2", 10), "V3": mk("V3", 1), "V4": [], "V5": mk("V5", 50)}
s1 = completeness.stratified_sample(pool, 40, 7)
check("the sample has exactly 40 distinct bursts", len(s1) == 40 and len({(b.video, b.frame, b.boxer) for b in s1}) == 40)
check("every video that has an unexplained burst is in it; one with none is not",
      {b.video for b in s1} == {"V1", "V2", "V3", "V5"} and sum(b.video == "V3" for b in s1) == 1)
check("bigger videos get more: stratified in proportion",
      sum(b.video == "V1" for b in s1) > sum(b.video == "V5" for b in s1) > sum(b.video == "V2" for b in s1))
check("the same seed gives the same sample, whatever the dict order",
      s1 == completeness.stratified_sample(dict(reversed(list(pool.items()))), 40, 7))
check("a different seed gives a different sample", s1 != completeness.stratified_sample(pool, 40, 8))
check("fewer bursts than the sample size: all of them, nothing invented",
      len(completeness.stratified_sample({"V": mk("V", 5)}, 40, 7)) == 5)
check("more videos than the sample size: still exactly n, no video twice over its supply",
      len(completeness.stratified_sample({f"V{i}": mk(f"V{i}", 3) for i in range(60)}, 40, 7)) == 40)
raises("a zero sample size is refused", lambda: completeness.stratified_sample(pool, 0, 7), "whole number")

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
check("exit 0; the totals are the planted ones (3 bursts, 2 unexplained)",
      rc == 0 and comp["all"]["bursts"] == 3 and comp["all"]["unexplained"] == 2, str(comp["all"]))
check("REAL FACT: unexplained bursts per identified boxer-minute is a positive rate from real minutes",
      comp["all"]["unexplained_per_identified_boxer_min"] > 0 and comp["all"]["A"]["identified_min"] > 0.2)
lines = csv1.strip().splitlines()
check("the sample file lists video, frame, time, boxer, hand and an EMPTY label column for the user",
      lines[0] == "video,frame,time_s,boxer,hand,user_label" and len(lines) == 3
      and all(l.endswith(",") for l in lines[1:]) and all(l.startswith("GH000001,") for l in lines[1:]), csv1)
olympic_completeness.main(args)
check("running it again writes the identical sample (reproducible)", cc.read_text(encoding="utf-8") == csv1)
raises("a missing rule file is refused", lambda: olympic_completeness.main(args[:6] + ["--rule", str(tmp / "nope.json")] + args[8:]),
       "no tuned rule")

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
