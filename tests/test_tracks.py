"""Tests for two-boxer pose tracking (pose_yolo.PoseTable / extract_tracks), the
tracks overlay, `run_ingest.py --track`, and crediting Olympic punches to people.

No model or GPU: the tracker is injected. Two boxers and a referee are planted,
so "who threw it" has a known answer.

Run:  python tests/test_tracks.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import run_ingest  # noqa: E402
from boxingvi import olympic, overlay, pose_yolo, video  # noqa: E402
from boxingvi.data import N_JOINTS  # noqa: E402
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
W, H, N = 320, 180, 20
clip = tmp / "GH000001.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=gray:s={W}x{H}:r=50",
                "-frames:v", str(N), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)


def person(cx, cy):
    """A stick figure centred at (cx, cy): COCO joints spread over a 40x80 box."""
    xy = np.zeros((N_JOINTS, 2))
    xy[:, 0] = cx + np.linspace(-15, 15, N_JOINTS)
    xy[:, 1] = cy + np.linspace(-35, 35, N_JOINTS)
    xy[9] = (cx - 25, cy - 5)    # left wrist out to the left
    xy[10] = (cx + 25, cy - 5)   # right wrist out to the right
    return xy, np.array([cx - 20, cy - 40, cx + 20, cy + 40], float)


BOXER_A, BOXER_B, REF = (80, 90), (200, 90), (290, 60)
calls = []


def fake_track(frame):
    i = len(calls)
    calls.append(i)
    people = [(1, *person(*BOXER_A)), (2, *person(*BOXER_B)), (7, *person(*REF))]
    if i == 0:
        people[2] = (-1, *people[2][1:])            # the referee not yet given an id
    if i == 3:
        people.append((2, *person(150, 120)))       # a duplicate id in one frame
    if i == 5:
        people = []                                 # nobody tracked
    if i == 6:
        people = people[:1]                         # one boxer lost: fewer than two people
    if i == 8:
        people.append((12, *person(322, 130)))      # a two-digit id at the right edge: box x1 302, digits would end at 323 > 320
    ids = np.array([p[0] for p in people], np.int64)
    kxy = np.array([p[1] for p in people]).reshape(-1, N_JOINTS, 2)
    boxes = np.array([p[2] for p in people]).reshape(-1, 4)
    kc = np.full((len(people), N_JOINTS), 0.9)
    return boxes, kxy, kc, ids


print("extract_tracks")
table = pose_yolo.extract_tracks(clip, fake_track, "fake")
check("every person with an id is kept, every frame", table.n_frames == N and set(table.ids()) == {1, 2, 7, 12})
check("a detection with no id yet (-1) is dropped, not guessed",
      not ((table.frame == 0) & (table.track_id == 7)).any() and ((table.frame == 1) & (table.track_id == 7)).any())
check("a second row with the same id in one frame is dropped (the first is kept)",
      ((table.frame == 3) & (table.track_id == 2)).sum() == 1
      and np.allclose(table.xy[(table.frame == 3) & (table.track_id == 2)][0, 0], person(*BOXER_B)[0][0]))
check("a frame with nobody tracked has no rows", not (table.frame == 5).any())
check("ids are ordered most-seen first", table.ids()[0] == 1 and table.ids()[-1] == 12)
tr = table.track(7)
check("one id as a Track: its frames in order, with confidence",
      list(tr.frame_ids) == [i for i in range(N) if i not in (0, 5, 6)] and tr.conf is not None
      and tr.conf.shape == (N - 3, N_JOINTS))
f = tmp / "t.tracks.npz"
table.save(f)
back = PoseTable.load(f)
check("npz round-trip is exact", np.array_equal(back.frame, table.frame) and np.array_equal(back.xy, table.xy)
      and back.n_frames == N and back.model == "fake")
np.savez(tmp / "evil.npz", frame=np.array([{"a": 1}], dtype=object))
try:
    PoseTable.load(tmp / "evil.npz")
    check("a pickled array in a tracks file is refused", False, "loaded")
except (ValueError, KeyError):
    check("a pickled array in a tracks file is refused", True)


def bad_len(frame):
    b, k, c, i = fake_track(frame)
    return b, k, c, i[:-1]


calls.clear()
raises("a tracker returning mismatched boxes/poses/ids is an error, not a misaligned row",
       lambda: pose_yolo.extract_tracks(clip, bad_len), "returned")
z = lambda n: (np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros((n, 4)), np.zeros((n, N_JOINTS, 2)), np.zeros((n, N_JOINTS)))  # noqa: E731
fr, ti, bx, xy, cf = z(2)
raises("the table refuses one id twice in one frame",
       lambda: PoseTable(fr, ti, bx, xy, cf, 5, 50.0, W, H), "twice")
raises("the table refuses a frame past the video", lambda: PoseTable(np.array([0, 9]), np.array([1, 1]), bx, xy, cf,
                                                                       5, 50.0, W, H), "outside the video")
raises("the table refuses NaN coordinates",
       lambda: PoseTable(np.array([0, 1]), np.array([1, 1]), bx, np.full((2, N_JOINTS, 2), np.nan), cf, 5, 50.0, W, H),
       "non-finite")
raises("the table refuses confidence above 1",
       lambda: PoseTable(np.array([0, 1]), np.array([1, 1]), bx, xy, np.full((2, N_JOINTS), 2.0), 5, 50.0, W, H),
       "[0, 1]")

print("tracks overlay")
ov = tmp / "ov.mp4"
check("the overlay has every frame", overlay.render_tracks_overlay(clip, table, ov) == N
      and video.probe(ov).timing.n_frames == N)
frames = list(video.iter_frames(ov))
red = lambda fr_: fr_[2, W // 2, 0] > 150 and fr_[2, W // 2, 1] < 90  # noqa: E731
check("frames with fewer than two people tracked are bordered red (none, or one boxer lost)",
      red(frames[5]) and red(frames[6]))
check("frames with both boxers (and the referee) tracked are not", not red(frames[4]) and not red(frames[0]))
yellow = lambda p: ((p[..., 0] > 190) & (p[..., 1] > 190) & (p[..., 2] < 90)).sum()  # noqa: E731
x1 = int(person(*REF)[1][0])
check("each person's id is drawn at their box, below the frame counter",
      yellow(frames[2][26:45, x1:x1 + 12]) > 20)
# Clamped, "12" starts at x 295 and its "2" fills columns 307-315; unclamped it would start
# at 302, the "2" would not fit and be skipped, and columns 313-316 would hold no digit.
# Rows 72-86: the id sits 18 px above the box top (y 90), clear of the skeleton below.
check("a two-digit id at the right edge is drawn whole: its second digit is on screen",
      yellow(frames[8][72:87, 313:317]) > 5, str(yellow(frames[8][72:87, 313:317])))
wrong = PoseTable(table.frame, table.track_id, table.box, table.xy, table.conf, N, 25.0, W, H)
raises("the overlay refuses a table from another video (frame rate)",
       lambda: overlay.render_tracks_overlay(clip, wrong, tmp / "no.mp4"), "different video")
longer = PoseTable(table.frame, table.track_id, table.box, table.xy, table.conf, N + 5, 50.0, W, H)
raises("the overlay refuses a table longer than the video",
       lambda: overlay.render_tracks_overlay(clip, longer, tmp / "no2.mp4"), "different video")

print("run_ingest --track")
calls.clear()
rc = run_ingest.main(["--video", str(clip), "--out-dir", str(tmp / "o"), "--track"], track=fake_track)
check("end to end: tracks file and overlay written",
      rc == 0 and (tmp / "o" / "GH000001.tracks.npz").is_file() and (tmp / "o" / "GH000001.tracks.overlay.mp4").is_file())
check("... and the single-boxer pose file is not", not (tmp / "o" / "GH000001.pose.npz").exists())

print("crediting punches to people (Olympic): the label's hand is NOT used to choose")
import olympic_attribute  # noqa: E402

gx, gy = BOXER_A[0] - 25, BOXER_A[1] - 5          # boxer A's LEFT wrist
glove = (gx - 8, gy - 8, gx + 8, gy + 8)
left_a = olympic.Event(10, 12, "head", "left", glove, 2)
right_b = olympic.Event(14, 15, "body", "right", (BOXER_B[0] + 17, BOXER_B[1] - 13, BOXER_B[0] + 33, BOXER_B[1] + 3), 2)
nowhere = olympic.Event(16, 17, "miss", "left", (150, 160, 166, 176), 1)
late = olympic.Event(N + 3, N + 4, "head", "left", glove, 1)
at = olympic.attribute([left_a, right_b, nowhere, late], table)
check("a punch on boxer A's left glove is credited to A (id 1), nearest wrist his left",
      at[0].track_id == 1 and at[0].wrist == "left" and at[0].dist == 0.0 and at[0].hand_agrees)
check("a punch on B's right glove is credited to B (id 2)", at[1].track_id == 2 and at[1].hand_agrees)
check("a glove box no wrist reaches is credited to nobody, and is not called ambiguous",
      at[2].track_id is None and at[2].dist > 0.5 and not at[2].ambiguous)
check("an event past the tracked frames is credited to nobody", at[3].track_id is None and at[3].dist is None)


def table_of(people, n=4):
    """people: [(id, xy (17,2))] in every frame 0..n-1, all joints confident."""
    rows = [(f, t, xy) for f in range(n) for t, xy in people]
    return PoseTable(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
                     np.tile([0, 0, 10, 10], (len(rows), 1)).astype(np.float32),
                     np.array([r[2] for r in rows], np.float32), np.full((len(rows), N_JOINTS), 0.9, np.float32),
                     n, 50.0, W, H)


ev = olympic.Event(0, 3, "head", "left", glove, 2)
a_xy = person(*BOXER_A)[0]
mirrored = a_xy.copy(); mirrored[[9, 10]] = mirrored[[10, 9]]   # a pose model whose "left" is the boxer's right
s_ok = olympic_attribute.summarise(olympic.attribute([ev], table_of([(1, a_xy)])))
s_mir = olympic_attribute.summarise(olympic.attribute([ev], table_of([(1, mirrored)])))
check("the convention test CAN fail: a mirrored pose model gives 0% nearest-wrist agreement (100% when not)",
      s_ok["nearest_wrist_is_labelled_hand"] == 1.0 and s_mir["nearest_wrist_is_labelled_hand"] == 0.0
      and s_mir["credited"] == 1, f"{s_ok} {s_mir}")
blocker = a_xy.copy(); blocker[10] = (gx + 3, gy + 2)            # the defender's glove inside the box too
amb = olympic.attribute([ev], table_of([(1, a_xy), (2, blocker)]))[0]
check("both boxers' gloves at the box on every frame (a block): too close to call, not settled arbitrarily",
      amb.track_id is None and amb.ambiguous)
# The defender's glove touches the box on ONE contact frame of four, elsewhere a box away.
away = a_xy.copy(); away[[9, 10]] = (gx + 40, gy)
rows4 = [(f, 1, a_xy) for f in range(4)] + [(f, 2, blocker if f == 2 else away) for f in range(4)]
t4 = PoseTable(np.array([r[0] for r in rows4]), np.array([r[1] for r in rows4]),
               np.tile([0, 0, 10, 10], (8, 1)).astype(np.float32), np.array([r[2] for r in rows4], np.float32),
               np.full((8, N_JOINTS), 0.9, np.float32), 4, 50.0, W, H)
touch = olympic.attribute([ev], t4)[0]
check("a defender's glove at the box on the contact frame only does not veto the thrower (median, not minimum)",
      touch.track_id == 1 and not touch.ambiguous, str(touch))
both_in = a_xy.copy(); both_in[10] = (gx + 2, gy + 1)             # the thrower's two wrists both inside the box
tied = olympic.attribute([ev], table_of([(1, both_in)]))[0]
check("when the thrower's two wrists are equally near, the punch is credited but gives no hand evidence",
      tied.track_id == 1 and tied.hand_agrees is None
      and olympic_attribute.summarise([tied])["hand_tied_excluded"] == 1)
near_ref = a_xy.copy(); near_ref[[9, 10]] = (gx + 30, gy)         # someone else's wrist 1.3 box diagonals away
clear = olympic.attribute([ev], table_of([(1, a_xy), (7, near_ref)]))[0]
check("a wrist inside the box with the next person over a box away: credited, with a margin",
      clear.track_id == 1 and clear.runner_up is not None and clear.dist < 0.5 * clear.runner_up)
moving = olympic.Event(0, 3, "head", "left", (200, 200, 216, 216), 2, keyboxes=((0, (200, 200, 216, 216)),
                                                                              (3, (gx - 8, gy - 8, gx + 8, gy + 8))))
# A real punch: the wrist travels WITH the glove box. Frame by frame it is always
# inside; compared only with the first box it would be far on frames 1-3.
rows_mv = []
for f in range(4):
    b = moving.box_at(f)
    xy_f = a_xy.copy(); xy_f[9] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    rows_mv.append((f, 1, xy_f))
t_mv = PoseTable(np.arange(4), np.ones(4, np.int64), np.tile([0, 0, 10, 10], (4, 1)).astype(np.float32),
                 np.array([r[2] for r in rows_mv], np.float32), np.full((4, N_JOINTS), 0.9, np.float32), 4, 50.0, W, H)
mv = olympic.attribute([moving], t_mv)[0]
check("the glove box is followed frame by frame: a wrist travelling with it is at distance 0 throughout",
      mv.track_id == 1 and mv.dist == 0.0, str(mv.dist))
check("box_at interpolates between keyframes",
      np.allclose(moving.box_at(0), (200, 200, 216, 216)) and np.allclose(moving.box_at(3), glove)
      and np.allclose(moving.box_at(1)[0], 200 + (gx - 8 - 200) / 3))
low = PoseTable(table.frame, table.track_id, table.box, table.xy, np.where(np.isin(np.arange(N_JOINTS), [9, 10]), 0.05, 0.9)
                .astype(np.float32)[None].repeat(len(table.frame), 0), N, 50.0, W, H)
check("unsure wrists are not used to credit a punch", olympic.attribute([left_a], low)[0].track_id is None)

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
