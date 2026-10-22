"""Tests for the Olympic Boxing loader (src/boxingvi/olympic.py).

The annotation files are third-party input, so most of these are the files the
loader must refuse. A real-data check runs too when the release is present.

Run:  python tests/test_olympic.py
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from boxingvi import olympic  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def refuses(name: str, fn, pattern: str) -> None:
    try:
        fn()
    except ValueError as e:
        check(name, pattern in str(e), f"wrong reason: {e}")
    except Exception as e:
        check(name, False, f"raised {type(e).__name__}: {e}")
    else:
        check(name, False, "accepted")


HEAD_L, BODY_R, MISS_L, BLOCK_R = ("Głowa lewą ręką", "Korpus prawą ręką",
                                   "Chybienie lewą ręką", "Blok prawą ręką")


def rect(frame, box=(100.0, 200.0, 150.0, 260.0), outside=False):
    return {"type": "rectangle", "occluded": False, "z_order": 0, "points": list(box),
            "frame": frame, "outside": outside, "attributes": []}


def track(label, shapes):
    return {"frame": shapes[0]["frame"], "group": 0, "source": "manual", "label": label, "shapes": shapes,
            "attributes": []}


def doc(tracks):
    return [{"version": 18, "tags": [], "shapes": [], "tracks": tracks}]


N = 1000
GOOD = doc([track(HEAD_L, [rect(500), rect(502), rect(503), rect(504, outside=True)]),
            track(BODY_R, [rect(100, (10.0, 20.0, 40.0, 60.0)), rect(101)]),
            track(MISS_L, [rect(800)]),
            track(BLOCK_R, [rect(900, outside=True)])])          # no visible shape: CVAT leaves a few

print("labels")
check("the eight Polish labels map to target x hand",
      olympic.parse_label(HEAD_L) == ("head", "left") and olympic.parse_label(BODY_R) == ("body", "right")
      and olympic.parse_label(MISS_L) == ("miss", "left") and olympic.parse_label(BLOCK_R) == ("block", "right"))
refuses("an unknown label is refused", lambda: olympic.parse_label("Głowa obiema rękami"), "unknown label")
refuses("a label that is not text is refused", lambda: olympic.parse_label(None), "not text")

print("events")
ev, empty = olympic.load_events(GOOD, N)
check("one event per track with a visible shape, in time order",
      [(e.first, e.last, e.target, e.hand) for e in ev]
      == [(100, 101, "body", "right"), (500, 503, "head", "left"), (800, 800, "miss", "left")], str(ev))
check("a shape marked outside ends the punch and is not part of it", ev[1].last == 503 and ev[1].n_keyframes == 3)
check("a track with no visible shape is counted, not turned into a punch", empty == 1)
check("the box is the one at the first visible frame", ev[0].box == (10.0, 20.0, 40.0, 60.0))
check("the event frame is the first frame the glove box appears", ev[1].frame == 500)

print("refused files")


def bad(mutate):
    d = copy.deepcopy(GOOD)
    mutate(d)
    return lambda: olympic.load_events(d, N)


refuses("a shape past the end of the video", bad(lambda d: d[0]["tracks"][0]["shapes"].__setitem__(0, rect(N))),
        "outside the video")
refuses("a negative frame", bad(lambda d: d[0]["tracks"][0]["shapes"].__setitem__(0, rect(-1))), "outside the video")
refuses("a boolean frame", bad(lambda d: d[0]["tracks"][0]["shapes"][0].__setitem__("frame", True)),
        "outside the video")
refuses("a polygon instead of a rectangle", bad(lambda d: d[0]["tracks"][0]["shapes"][0].__setitem__("type", "polygon")),
        "unexpected shape")
refuses("a box with three numbers", bad(lambda d: d[0]["tracks"][0]["shapes"][0].__setitem__("points", [1, 2, 3])),
        "four numbers")
refuses("an inverted box", bad(lambda d: d[0]["tracks"][0]["shapes"][0].__setitem__("points", [150, 260, 100, 200])),
        "degenerate")
refuses("a box far outside the frame",
        bad(lambda d: d[0]["tracks"][0]["shapes"][0].__setitem__("points", [5000, 5000, 5100, 5100])), "not inside")
refuses("a box poking one pixel past the edge (none in the release does)",
        bad(lambda d: d[0]["tracks"][0]["shapes"][0].__setitem__("points", [1800, 500, 1921, 560])), "not inside")
refuses("a box with a negative corner",
        bad(lambda d: d[0]["tracks"][0]["shapes"][0].__setitem__("points", [-5000, 200, 150, 260])), "not inside")
refuses("two keyframes on one frame", bad(lambda d: d[0]["tracks"][0]["shapes"].append(rect(502))), "two keyframes")

print("CVAT track semantics")
ev2, _ = olympic.load_events(doc([track(HEAD_L, [rect(500), rect(503), rect(510, outside=True)])]), N)
check("a punch runs until the frame before its outside keyframe, not its last visible keyframe",
      (ev2[0].first, ev2[0].last) == (500, 509) and not ev2[0].open_ended, str(ev2))
check("a track with no closing outside keyframe ends at its last visible one, and is flagged",
      ev[0].open_ended and ev[0].last == 101)
ev3, _ = olympic.load_events(doc([track(HEAD_L, [rect(500), rect(501, outside=True), rect(502),
                                                  rect(504, outside=True)])]), N)
check("hidden for one frame then visible again: one punch, merged, flagged",
      len(ev3) == 1 and (ev3[0].first, ev3[0].last) == (500, 503) and ev3[0].reappeared, str(ev3))
refuses("hidden for longer than 2 frames then visible again: refused, not merged",
        lambda: olympic.load_events(doc([track(HEAD_L, [rect(500), rect(501, outside=True), rect(520),
                                                         rect(522, outside=True)])]), N), "Refusing to merge")
ev4, _ = olympic.load_events(doc([track(HEAD_L, [rect(500, outside=True), rect(505), rect(507, outside=True)])]), N)
check("a leading outside keyframe is ignored", (ev4[0].first, ev4[0].last) == (505, 506), str(ev4))
refuses("an unknown label in a track", bad(lambda d: d[0]["tracks"][0].__setitem__("label", "Punch")), "unknown label")
refuses("free-standing shapes (not punch tracks)", bad(lambda d: d[0].__setitem__("shapes", [rect(5)])),
        "free-standing")
refuses("an empty file", lambda: olympic.load_events([], N), "non-empty list")
tmp = Path(tempfile.mkdtemp())
(tmp / "nan.json").write_text('[{"tracks": [{"label": "x", "shapes": [{"frame": NaN}]}]}]', encoding="utf-8")
refuses("NaN in the file", lambda: olympic.load_events(tmp / "nan.json", N), "not a number")

print("task discovery")
root = tmp / olympic.DATASET_DIR
(root / "task_kam2_gh000001" / "data").mkdir(parents=True)
(root / "task_kam2_gh000001" / "data" / "GH000001.mp4").write_bytes(b"")
(root / "task_kam2_gh000001" / "annotations.json").write_text("[]", encoding="utf-8")
t = olympic.find_tasks(tmp)
check("tasks are found from the dataset's parent folder, with their camera", len(t) == 1 and t[0].camera == "kam2")
(root / "stray").mkdir()
refuses("an unexpected folder in the dataset", lambda: olympic.find_tasks(tmp), "unexpected folder")

real = ROOT / "data" / "olympic"
if (real / olympic.DATASET_DIR).is_dir():
    print("the real release (data/olympic)")
    from boxingvi.video import probe
    tasks = olympic.find_tasks(real)
    total = n_ev = n_empty = pf = re = oe = 0
    for tk in tasks:
        info = probe(tk.video)
        e, m = olympic.load_events(tk.annotations, info.timing.n_frames, info.width, info.height)
        total += info.timing.n_frames; n_ev += len(e); n_empty += m
        pf += sum(x.last - x.first + 1 for x in e); re += sum(x.reappeared for x in e); oe += sum(x.open_ended for x in e)
    check("29 videos, 312,774 frames: exactly the paper's count", len(tasks) == 29 and total == 312_774, str(total))
    check("4,605 punch events and 4 empty tracks, every file parsed strictly", n_ev == 4605 and n_empty == 4,
          f"{n_ev} {n_empty}")
    check("21,564 punch frames to the outside keyframe (the paper's 11,345 matches no count: recorded, not fitted)",
          pf == 21_564, str(pf))
    check("exactly one track reappears after a 1-frame hide (kam4_gh069681), none is open-ended",
          re == 1 and oe == 0, f"{re} {oe}")
else:
    print("(the real release is not in data/olympic; real-data checks skipped)")

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
