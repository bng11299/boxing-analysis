"""Tests for the linker join check (joincheck.py, linker_join_check.py). Synthetic videos and tracks only:
no Olympic Boxing file is read. A plain script, like the rest of tests/.

  python tests/test_linker_join_check.py
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))

import label_moments  # noqa: E402
import linker_join_check as cli  # noqa: E402
from boxingvi import joincheck as jc  # noqa: E402
from boxingvi import labelmoments as lm  # noqa: E402
from boxingvi import linking  # noqa: E402
from boxingvi.data import N_JOINTS  # noqa: E402
from boxingvi.pose_yolo import PoseTable  # noqa: E402
from boxingvi.video import fingerprint  # noqa: E402

PASS = FAIL = SKIP = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f"  ({detail})" if detail else ""))


def raises(name: str, fn, exc=lm.Refused, contains: str = "") -> None:
    try:
        fn()
    except exc as e:
        check(name, contains.lower() in str(e).lower(), f"message: {e}")
        return
    except Exception as e:     # noqa: BLE001
        check(name, False, f"raised {type(e).__name__}: {e}")
        return
    check(name, False, "did not raise")


def skip(name: str, why: str) -> None:
    global SKIP
    SKIP += 1
    print(f"  SKIP {name} ({why})")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# Words that would say what the linker decided, or how sure it was. None may appear in anything shown.
REVEALING = re.compile(r"\b((?i:boxers?|chains?|identity|cost|jump|confidence|stratum|short|long)|[AB])\b")
REVEALING_KEYS = {"chain", "chains", "boxer", "identity", "identities", "cost", "jump", "k_chain", "a", "b",
                  "membership", "objective"}

tmp = Path(tempfile.mkdtemp(prefix="linker_join_check_test_"))
cli.RESULTS_ROOTS = [tmp / "results"]          # the tests' own results/ folder stands in for the repo's
MAN = frozenset({"V1", "V2", "V3"})
H64 = "a" * 64

# ---- 1. the estimate ---------------------------------------------------------------------------------
print("the estimate")
lo, hi = jc.wilson(0, 100)
check("Wilson 0/100 is [0, 0.037]", lo == 0.0 and abs(hi - 0.0370) < 5e-4, f"{lo}, {hi}")
lo, hi = jc.wilson(10, 100)
check("Wilson 10/100 is [0.055, 0.174]", abs(lo - 0.0552) < 5e-4 and abs(hi - 0.1744) < 5e-4, f"{lo}, {hi}")
raises("Wilson refuses n = 0", lambda: jc.wilson(0, 0), ValueError)
raises("Wilson refuses x > n", lambda: jc.wilson(5, 4), ValueError)
check("one join before every punch: misattribution = p", abs(jc.misattribution(0.07, [[1, 3]]) - 0.07) < 1e-12)
check("no join before a punch: never misattributed", jc.misattribution(0.3, [[0, 5]]) == 0.0)
check("misattribution is 0 at p = 0", jc.misattribution(0.0, [[5, 1], [20, 2]]) == 0.0)
uniform = [[k, 1] for k in range(24)]
ms = [jc.misattribution(p, uniform) for p in (0.0, 0.01, 0.05, 0.2, 0.5)]
check("misattribution rises with p", all(a < b for a, b in zip(ms, ms[1:])), str(ms))
check("p* for one join per punch is the target itself", abs(jc.p_star([[1, 1]]) - 0.10) < 1e-9)
ps = jc.p_star(uniform)
check("p* for ~23 joins per chain is about 1% (the gate's figure)", 0.008 < ps < 0.012, str(ps))
check("misattribution at p* is the target", abs(jc.misattribution(ps, uniform) - 0.10) < 1e-9)
raises("misattribution needs a credited punch", lambda: jc.misattribution(0.1, []), ValueError)
check("verdict: interval below p* is acceptable", jc.verdict(100, 0, 100, 0.0, 0.009, 0.01) == "measured, acceptable")
check("verdict: interval above p* is unreliable", jc.verdict(100, 0, 100, 0.02, 0.1, 0.01) == "measured, unreliable")
check("verdict: interval across p* is inconclusive",
      jc.verdict(100, 0, 100, 0.0, 0.037, 0.01) == "measured, inconclusive")
check("verdict: over 30% unsure is not established", jc.verdict(69, 31, 100, 0.0, 0.0, 0.5) == "not established")
check("verdict: under 70 decided is not established", jc.verdict(69, 0, 69, 0.0, 0.0, 0.5) == "not established")
check("verdict: exactly 30% unsure and 70 decided still counts",
      jc.verdict(70, 30, 100, 0.0, 0.0, 0.5) == "measured, acceptable")

# ---- 2. joins from a linking, with nothing about the chain -----------------------------------------
print("joins from a linking")


def frag(tid, first, last):
    return linking.Fragment(tid, first, last, last - first + 1, (0.0, 0.0), (0.0, 0.0), 10.0, True)


frags = (frag(10, 0, 49), frag(11, 60, 99), frag(12, 110, 199), frag(20, 0, 89), frag(21, 95, 199), frag(30, 0, 199))
mem = np.full((200, 2), -1, np.int64)
for k, chain in enumerate(((10, 11, 12), (20, 21))):
    for t in chain:
        f = next(x for x in frags if x.tid == t)
        mem[f.first:f.last + 1, k] = t
lk = linking.Linking(((10, 11, 12), (20, 21)), mem, 200, 50.0, "syn", None, frags)
J = jc.joins_of(lk, "V1")
check("every consecutive pair of every chain is a join",
      [(j.tid1, j.tid2, j.last1, j.first2) for j in J] == [(10, 11, 49, 60), (11, 12, 99, 110), (20, 21, 89, 95)],
      str(J))
check("a Join carries no chain, boxer or cost field",
      set(jc.Join.__dataclass_fields__) == {"video", "tid1", "tid2", "last1", "first2"})
check("gap = first frame of 2 minus last frame of 1", J[0].gap == 11)
pos = jc.positions_of(lk, {10: 2, 12: 1, 21: 3, 30: 7})
check("positions: [joins before the fragment, its punches], pooled over both chains, unchained ids ignored",
      pos == [[0, 2], [1, 3], [2, 1]], str(pos))
check("chain minutes count both chains' identified frames", abs(jc.chain_minutes(lk) - (50 + 40 + 90 + 90 + 105) / 50 / 60) < 1e-9)
check("camera from the task name", jc.camera_of("task_kam2_gh078416") == "kam2" and jc.camera_of("x") == "unknown")

# ---- 3. the draw ---------------------------------------------------------------------------------
print("the draw")


def frame_joins(n_per=100, vids=("V1", "V2", "V3"), gap_of=lambda i: 1 + i % 100):
    out = []
    for v in vids:
        for i in range(n_per):
            out.append(jc.Join(v, 2 * i, 2 * i + 1, 1000 + 300 * i, 1000 + 300 * i + gap_of(i)))
    return out


fj = frame_joins()
s1, c1, st1 = jc.draw(fj)
s2, c2, st2 = jc.draw(list(reversed(fj)))
check("the same seed draws the same sample, whatever order the frame arrives in", s1 == s2 and st1 == st2)
s3, _, _ = jc.draw(fj, seed=1)
check("another seed draws another sample", s3 != s1)
check("n = 100 exactly, short + long", len(s1) == 100 and c1["n_short"] + c1["n_long"] == 100)
check("both strata are represented about evenly (median split, proportional)", 45 <= c1["n_short"] <= 55, str(c1))
med = c1["median_gap"]
check("each shown join's stratum matches its gap", all((j.gap <= med) == (s == "short") for j, s in zip(s1, st1)))
check("the shown order is shuffled, not the frame's order",
      s1 != sorted(s1, key=lambda j: (j.video, j.tid1, j.tid2)))
check("no join is drawn twice", len(set(s1)) == 100)
raises("a frame smaller than n is refused", lambda: jc.draw(fj[:50]), contains="fewer than")
raises("a frame with a repeated join is refused", lambda: jc.draw(fj + fj[:1]), contains="twice")
same, cs, _ = jc.draw(frame_joins(gap_of=lambda i: 5))
check("all gaps equal: everything is short, still 100", len(same) == 100 and cs["n_long"] == 0)

# ---- 4. the sample file -------------------------------------------------------------------------
print("the sample file")


def per_video(joins_by_vid, positions=None, fps=50.0, n_frames=40_000, minutes=10.0):
    pv = {}
    for v, js in joins_by_vid.items():
        pv[v] = {"joins": js, "positions": positions if positions is not None else [[k, 1] for k in range(24)],
                 "chain_minutes": minutes, "camera": "kam2" if v != "V3" else "kam4", "fps": fps,
                 "n_frames": n_frames, "tracks_sha256": H64, "video_fp": "b" * 64}
    return pv


byv = {v: [j for j in fj if j.video == v] for v in MAN}
smp = jc.build_sample(per_video(byv), MAN)
raises("a missing manifest video means nothing is drawn",
       lambda: jc.build_sample(per_video({v: byv[v] for v in ("V1", "V2")}), MAN), contains="manifest")
raises("a video outside the manifest is refused",
       lambda: jc.build_sample(per_video(byv | {"V9": []}), MAN), contains="manifest")
bad = dict(byv)
bad["V1"] = byv["V1"][:-1] + [jc.Join("V1", 900, 901, 100, 100 + 101)]
raises("a join longer than the linker's 2 s is refused (out of range)",
       lambda: jc.build_sample(per_video(bad), MAN), contains="gap")
bad["V1"] = byv["V1"][:-1] + [jc.Join("V1", 900, 901, 39_990, 40_005)]
raises("a join past the end of its video is refused", lambda: jc.build_sample(per_video(bad), MAN), contains="outside")
bad["V1"] = byv["V1"][:-1] + [jc.Join("V1", 900, 900, 10, 20)]
raises("a join of a track id to itself is refused", lambda: jc.build_sample(per_video(bad), MAN), contains="different")


def walk_keys(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from walk_keys(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk_keys(v)


keys = {k.lower() for k in walk_keys(smp)}
check("nothing in the sample file names a chain, a boxer, a cost or a jump", not (keys & REVEALING_KEYS),
      str(keys & REVEALING_KEYS))
check("each join stores only video, the two ids and the two boundary frames",
      all(set(e) == jc.JOIN_KEYS for e in smp["joins"]))
check("the strata are not stored at all (the score derives them from the gaps)",
      "strata" not in smp and "stratum" not in smp["joins"][0])

d = tmp / "results" / "ljc"
d.mkdir(parents=True)
sp = d / cli.SAMPLE_NAME
jc.write_new(sp, jc.sample_bytes(smp))
S = jc.load_sample(sp, MAN)
check("the sample loads back whole", S.n == 100 and list(S.joins) == [jc.Join(**e) for e in smp["joins"]])
check("the loaded sample offers no stratum to the window", not hasattr(S, "strata"))
raises("a sample is written once (no redraw)", lambda: jc.write_new(sp, b"{}"), contains="already exists")


def tampered(fn, name="t.json"):
    x = json.loads(sp.read_text())
    fn(x)
    p = d / name
    p.write_text(json.dumps(x))
    return lambda: jc.load_sample(p, MAN)


raises("a join carrying a chain field is refused", tampered(lambda x: x["joins"][0].update(chain=0)), contains="keys")
raises("an extra top-level key is refused", tampered(lambda x: x.update(boxers={})), contains="keys")
raises("a join with fragment 2 before fragment 1 is refused",
       tampered(lambda x: x["joins"][0].update(first2=x["joins"][0]["last1"])), contains="outside")
raises("a join past its video's end is refused",
       tampered(lambda x: x["joins"][0].update(last1=39_999, first2=40_000)), contains="outside")
raises("a join with a gap over 100 frames at 50 fps is refused",
       tampered(lambda x: x["joins"][0].update(first2=x["joins"][0]["last1"] + 101)), contains="gap")
raises("a fractional frame is refused", tampered(lambda x: x["joins"][0].update(last1=1.5)), contains="whole number")
raises("a boolean id is refused", tampered(lambda x: x["joins"][0].update(tid1=True)), contains="whole number")
raises("a video outside the manifest is refused",
       tampered(lambda x: x["videos"].update(V9=x["videos"]["V1"])), contains="manifest")
raises("a join naming an unknown video is refused",
       tampered(lambda x: x["joins"][0].update(video="V9")), contains="manifest")
raises("a duplicated join is refused", tampered(lambda x: x["joins"].__setitem__(1, x["joins"][0])), contains="twice")
raises("n other than the gate's 100 is refused", tampered(lambda x: x.update(n=99)), contains="seed and n")
raises("a seed other than the gate's is refused", tampered(lambda x: x.update(seed=7)), contains="seed and n")
raises("a float seed is refused", tampered(lambda x: x.update(seed=20261027.0)), contains="whole number")
raises("a dropped join is refused", tampered(lambda x: x["joins"].pop()), contains="does not match")
raises("an edited p* is refused (an inflated p* would make any answers 'acceptable')",
       tampered(lambda x: x["frame"].update(p_star=0.5)), contains="p*")
raises("positions edited to move p* are refused", tampered(lambda x: x["videos"]["V1"].update(positions=[[1, 1]])),
       contains="p*")
raises("a non-numeric position is refused",
       tampered(lambda x: x["videos"]["V1"].update(positions=[["1", 1]])), contains="whole number")
raises("frame totals that do not add up are refused", tampered(lambda x: x["frame"].update(joins=301)),
       contains="add up")
raises("strata counts that do not follow from the median are refused",
       tampered(lambda x: x["frame"].update(n_short=x["frame"]["n_short"] + 1, n_long=x["frame"]["n_long"] - 1)),
       contains="strata")
raises("a moved median gap is refused", tampered(lambda x: x["frame"].update(median_gap=99)), contains="strata")
raises("a changed misattribution target is refused",
       tampered(lambda x: x["frame"].update(target_misattribution=0.5)), contains="target")
raises("negative chain minutes are refused", tampered(lambda x: x["videos"]["V1"].update(chain_minutes=-1)),
       contains="chain minutes")
raises("an odd camera name is refused", tampered(lambda x: x["videos"]["V1"].update(camera="Brown")),
       contains="camera")
raises("a bad hash is refused", tampered(lambda x: x["videos"]["V1"].update(tracks_sha256="zz")), contains="sha256")
raises("a sample is checked before it is written (parse_sample on bytes)",
       lambda: jc.parse_sample(b'{"format": "x"}', MAN), contains="not a linker-join sample")
(d / "nj.json").write_text("not json")
raises("not JSON is refused", lambda: jc.load_sample(d / "nj.json", MAN), contains="JSON")
(d / "big.json").write_bytes(b" " * (jc.MAX_SAMPLE_BYTES + 1))
raises("an oversized file is refused before parsing", lambda: jc.load_sample(d / "big.json", MAN), contains="large")

# ---- 5. answers and resume ----------------------------------------------------------------------
print("answers and resume")
ap_ = d / cli.ANSWERS_NAME
A = jc.AnswerFile.create(ap_, S)
check("a new answers file has every row empty and pins the sample", A.first_empty() == 0
      and A.counts()["empty"] == 100 and S.sha256 in ap_.read_text())
raises("the answers file is created once", lambda: jc.AnswerFile.create(ap_, S), contains="already exists")
clk = Clock()
ses = jc.Session(A, clock=clk)
check("there is no `# A:` question for joins", not ses.needs_description())
clk.t += 4
ses.label("S")
clk.t += 4
ses.label("D")
check("each answer is saved at once", ap_.read_text().count("\n1,S\n") == 1 and "\n2,D\n" in ap_.read_text())
A2 = jc.AnswerFile.load(ap_, S)
ses2 = jc.Session(A2, clock=Clock())
check("a new session resumes at the first unanswered join", ses2.i == 2 and A2.values[:2] == ["S", "D"])
check("time spent so far is kept across sessions", 0.1 < A2.prior_minutes() < 0.2, str(A2.prior_minutes()))
raises("an answer other than S / D / U is refused", lambda: A2.set_value(3, "L"), ValueError)
other = jc.build_sample(per_video(byv, minutes=11.0), MAN)
(d / "other.json").write_bytes(jc.sample_bytes(other))
S_other = jc.load_sample(d / "other.json", MAN)
raises("answers cannot be resumed into a different sample", lambda: jc.AnswerFile.load(ap_, S_other),
       contains="different sample")
text = ap_.read_text()
for name, broken in (("a row out of order", text.replace("\n3,\n", "\n4,\n", 1)),
                     ("an answer that is not S / D / U", text.replace("\n3,\n", "\n3,X\n", 1)),
                     ("a missing row", text.replace("\n100,\n", "\n")),
                     ("an extra row", text + "101,\n"),
                     ("a changed header", text.replace("# format: linker-join-answers-1", "# format: other"))):
    (d / "a2.csv").write_text(broken, newline="")
    raises(f"{name} is refused", lambda: jc.AnswerFile.load(d / "a2.csv", S))
(d / "a3.csv").write_text(text.replace("# minutes: ", "# minutes: 9999", 1).replace("9999.", "9999"), newline="")
raises("an out-of-range minutes value is refused", lambda: jc.AnswerFile.load(d / "a3.csv", S), contains="minutes")
stale = jc.Session(jc.AnswerFile.load(ap_, S), clock=Clock())
ses2.label("U")
raises("a stale session cannot overwrite newer answers", lambda: stale.label("S"), lm.FileChanged)
lock = jc.SessionLock(ap_)
raises("a second check window on the same answers file is refused (no lost-update race)",
       lambda: jc.SessionLock(ap_), contains="already has")
lock.close()
lock2 = jc.SessionLock(ap_)
check("once the first window lets go, a new one may open", lock2._f is not None)
lock2.close()
(d / "blocked.csv.lock").mkdir()
raises("a lock file that cannot be opened is a refusal, not a traceback with a path",
       lambda: jc.SessionLock(d / "blocked.csv"), contains="cannot be opened")
check("... and the newer answer is still there", jc.AnswerFile.load(ap_, S).values[2] == "U")
bclk = Clock()
bs = jc.Session(jc.AnswerFile.load(ap_, S), clock=bclk)
check("no break is due at the start", not bs.break_due())
for _ in range(25):
    bclk.t += 119.0
    bs.tick()
check("no break before 50 minutes of this sitting", not bs.break_due(), str(bs.sitting_minutes()))
bclk.t += 119.0
check("a break is due at 50 minutes of this sitting", bs.break_due(), str(bs.sitting_minutes()))
idle = jc.Session(jc.AnswerFile.load(ap_, S), clock=(ic := Clock()))
ic.t += 3 * 3600
check("an idle hour counts as at most 2 minutes (#35's cap), so walking away is not checking time",
      not idle.break_due() and idle.sitting_minutes() <= 2.0001)

# ---- 6. the clip ---------------------------------------------------------------------------------
print("the clip")
W, H, FPS, BANDS = 320, 180, 50, 10


def frame_for(i: int) -> np.ndarray:
    f = np.zeros((H, W, 3), np.uint8)
    f[:, :] = 40
    bw = W // BANDS
    for b in range(BANDS):
        if (i >> b) & 1:
            f[H - 30:, b * bw:(b + 1) * bw] = 230      # the index sits in the bottom strip, below every box
    return f


def read_index(a: np.ndarray) -> int:
    bw = a.shape[1] / BANDS
    return sum(1 << b for b in range(BANDS) if a[a.shape[0] - 15, int((b + 0.5) * bw)].mean() > 128)


def make_video(path: Path, n: int) -> None:
    cmd = [shutil.which("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-g", "50", "-bf", "2", "-crf", "12", "-pix_fmt", "yuv420p",
           str(path)]
    subprocess.run(cmd, input=b"".join(frame_for(i).tobytes() for i in range(n)), check=True, capture_output=True)


BOX1, BOX2, BOX3 = (20, 10, 100, 80), (180, 10, 260, 80), (120, 100, 160, 140)


def table_for(n=260):
    rows = []
    for f in range(0, 101):
        rows.append((f, 1, BOX1))
    for f in range(120, n):
        rows.append((f, 2, BOX2))
    for f in range(0, n):
        rows.append((f, 3, BOX3))          # a bystander, never to be drawn
    rows.sort()
    k = len(rows)
    return PoseTable(np.array([r[0] for r in rows], np.int64), np.array([r[1] for r in rows], np.int64),
                     np.array([r[2] for r in rows], np.float32).reshape(k, 4),
                     np.zeros((k, N_JOINTS, 2), np.float32), np.full((k, N_JOINTS), 0.9, np.float32),
                     n, float(FPS), W, H, source="syn.mp4")


have_ffmpeg = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
CYAN, YELLOW = np.array(jc.CYAN, np.uint8), np.array(jc.YELLOW, np.uint8)


def has(a, colour, box=None):
    m = (a == colour).all(axis=2)
    if box is not None:
        x0, y0, x1, y1 = box
        m = m[y0:y1 + 1, x0:x1 + 1]
    return bool(m.any())


join = jc.Join("V1", 1, 2, 100, 120)
table = table_for()
b1, b2 = jc.boxes_for(table, join, FPS)
check("fragment 1's boxes cover its last 1 s, fragment 2's its first 1 s",
      sorted(b1) == list(range(51, 101)) and sorted(b2) == list(range(120, 170)))
check("the bystander's box is not among them", all(b == tuple(map(float, BOX1)) for b in b1.values()))
raises("a join whose fragment 1 does not end at last1 is refused (out of range)",
       lambda: jc.boxes_for(table, jc.Join("V1", 1, 2, 99, 120), FPS), contains="does not match")
raises("a join whose fragment 2 does not start at first2 is refused",
       lambda: jc.boxes_for(table, jc.Join("V1", 1, 2, 100, 121), FPS), contains="does not match")
raises("a track id not in the table is refused", lambda: jc.boxes_for(table, jc.Join("V1", 1, 9, 100, 120), FPS),
       contains="not in its tracks")

clip = None
if have_ffmpeg:
    vid = tmp / "data" / "task_kam2_v1" / "data" / "V1.mp4"
    vid.parent.mkdir(parents=True)
    make_video(vid, 260)
    start, end = jc.clip_range(join, FPS, 260)
    check("the clip runs from 1 s before fragment 1 ends to 1 s after fragment 2 starts",
          (start, end) == (51, 169), f"{start}, {end}")
    w, h = jc.display_size(W, H)
    buf, got = lm.decode_raw(vid, FPS, start, end - start + 1, w, h, "join 1", min_frames=end - start + 1)
    clip = jc.build_clip(0, buf, got, start, join, b1, b2, W, w, h, FPS, jc.step_for(FPS))
    arr = [np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in clip.frames]
    idx = [read_index(a) for a in arr]
    check("every other frame at 50 fps", len(arr) < 70, str(len(arr)))
    check("playback is real time: the frames' durations add up to the clip's length",
          sum(clip.ms) == (end - start + 1) * 1000 // FPS, f"{sum(clip.ms)} ms")
    b = idx.index(100)
    check("a boundary frame off the step grid is not shown at half speed (Astra)",
          clip.ms[b - 1] == 20 and clip.ms[b] == 20, str(clip.ms[b - 1:b + 2]))
    check("the last frame of fragment 1 and the first of fragment 2 are both in the clip",
          100 in idx and 120 in idx, str(idx[:5]))
    check("the frames are in order and are the frames asked for", idx == sorted(idx) and idx[0] == 51, str(idx[:3]))
    p1 = [a for a, p in zip(arr, clip.part) if p == 1]
    p0 = [a for a, p in zip(arr, clip.part) if p == 0]
    p2 = [a for a, p in zip(arr, clip.part) if p == 2]
    check("fragment 1's frames carry a cyan box on its box, and no yellow",
          all(has(a, CYAN, BOX1) and not has(a, YELLOW) for a in p1))
    check("fragment 2's frames carry a yellow box on its box, and no cyan",
          all(has(a, YELLOW, BOX2) and not has(a, CYAN) for a in p2))
    check("the gap frames carry no box", p0 and all(not has(a, CYAN) and not has(a, YELLOW) for a in p0))
    check("the bystander's box is never drawn",
          all(not has(a, CYAN, BOX3) and not has(a, YELLOW, BOX3) for a in arr))
    still = np.frombuffer(clip.still, np.uint8).reshape(h, clip.still_width, 3)
    left, right = still[:, :w], still[:, w + jc.STILL_SEP:]
    check("the still is fragment 1's last frame | fragment 2's first frame",
          read_index(left) == 100 and read_index(right) == 120)
    check("... cyan on the left, yellow on the right", has(left, CYAN) and not has(left, YELLOW)
          and has(right, YELLOW) and not has(right, CYAN))
    raises("a decode that stops before the clip's end is refused (a truncated clip is U-only)",
           lambda: lm.decode_raw(vid, FPS, 250, 30, w, h, "join 1", min_frames=30), contains="could not decode")
else:
    skip("the clip tests", "ffmpeg not on PATH")

# ---- 7. the video guard on Olympic paths ----------------------------------------------------------
print("paths")
if have_ffmpeg:
    data_root = tmp / "data"
    g = lm.VideoGuard(vid, data_root, path_check=cli.data_path_check(data_root))
    check("a video under --data is accepted and held open", g.check() == vid.resolve())
    g.close()
    outside = tmp / "elsewhere.mp4"
    shutil.copy(vid, outside)
    raises("a video outside --data is refused",
           lambda: lm.VideoGuard(outside, data_root, path_check=cli.data_path_check(data_root)),
           contains="under the data folder")
raises("the output folder must be under results/", lambda: cli.check_out_dir(tmp / "docs"), contains="results")
raises("a results/ folder somewhere else is not this repo's", lambda: cli.check_out_dir(tmp / "x" / "results" / "y"),
       contains="results")
check("a folder under this repo's results/ is accepted", cli.check_out_dir(d) == d.resolve())

# ---- 8. the window --------------------------------------------------------------------------------
print("the window")
try:
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
except Exception as e:     # noqa: BLE001
    root = None
    skip("the window tests", f"no display: {type(e).__name__}")


class FakeVideos:
    """Stands in for linker_join_check.Videos: every join shows the synthetic clip; join 3 cannot decode."""

    def __init__(self):
        self.calls = []

    def decode(self, sample, k):
        self.calls.append(k)
        if k == 2:
            raise lm.Refused("ffmpeg could not decode join 3")
        return jc.JoinClip(k, clip.width, clip.height, clip.fps, clip.start_f, clip.frames, clip.part,
                           clip.ms, clip.still, clip.still_width)

    def max_display(self):
        return clip.width, clip.height


def pump(app, until, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.root.update()
        if until():
            return True
        time.sleep(0.01)
    return False


class Ev:
    CODES = {"s": 83, "d": 68, "u": 85, "l": 76, "r": 82, "question": 191}

    def __init__(self, keysym):
        self.keysym, self.keycode = keysym, self.CODES[keysym]


shown_texts = []
if root is not None and clip is not None:
    d2 = tmp / "results" / "win"
    d2.mkdir(parents=True)
    jc.write_new(d2 / cli.SAMPLE_NAME, jc.sample_bytes(smp))
    S2 = jc.load_sample(d2 / cli.SAMPLE_NAME, MAN)
    A3 = jc.AnswerFile.create(d2 / cli.ANSWERS_NAME, S2)
    wclk = Clock()
    ses3 = jc.Session(A3, clock=wclk)
    app = cli.JoinApp(root, ses3, S2, FakeVideos(), clock=wclk)

    def grab():
        shown_texts.extend([app.caption.cget("text"), app.status.cget("text"), root.title()])

    check("the first join loads", pump(app, lambda: app.clip is not None))
    grab()
    check("it opens by playing the clip, not on the still", "still" not in app.photos and app.shown_at is None)
    check("answers are ignored while the clip plays the first time", app.handle("S") == "too-soon")
    check("the still is held after one full play", pump(app, lambda: app.photos.get("still") is not None)
          and str(app.image.cget("image")) == str(app.photos["still"]) and app.shown_at is not None)
    grab()
    wclk.t += 1.0
    check("#35's L key answers nothing here", app.on_key(Ev("l")) is None and A3.values[0] == "")
    check("? is not an answer here either", app.on_key(Ev("question")) is None)
    check("S answers 'same' and saves", app.on_key(Ev("s")) == "S" and "\n1,S\n" in (d2 / cli.ANSWERS_NAME).read_text())
    check("the window moves to join 2", ses3.i == 1 and pump(app, lambda: app.clip is not None and app.clip.k == 1))
    shown_texts.extend(app.play_view(app.clip, j)[1] for j in range(len(app.clip.frames)))
    check("the clip's captions name the fragments and the gap only",
          {app.play_view(app.clip, j)[1].split(" (")[0].split(":")[0] for j in range(len(app.clip.frames))}
          == {"end of fragment 1", "gap", "start of fragment 2"})
    check("no caption gives the gap's length (it would reveal the stratum)",
          not any(re.search(r"\d", app.play_view(app.clip, j)[1].replace("fragment 1", "").replace("fragment 2", ""))
                  for j in range(len(app.clip.frames))))
    app.on_release(Ev("s"))
    pump(app, lambda: app.armed and app.shown_at is not None)
    wclk.t += 1.0
    check("D answers 'different'", app.on_key(Ev("d")) == "D" and A3.values[1] == "D")
    app.on_release(Ev("d"))
    pump(app, lambda: app.armed)
    pump(app, lambda: app.shown_at is not None)
    wclk.t += 1.0
    grab()
    check("a join that cannot be decoded shows nothing, and says Space retries",
          app.clip is None and "cannot show this join" in app.caption.cget("text")
          and "Space retries" in app.caption.cget("text"))
    check("... and only U can be given for it", app.handle("S") == "not-shown" and app.handle("U") == "U"
          and A3.values[2] == "U")
    app.on_release(Ev("u"))
    pump(app, lambda: app.armed and app.shown_at is not None)
    for _ in range(26):
        wclk.t += 119.0
        ses3.tick()
    wclk.t += 1.0
    check("after 50 minutes in one sitting no answer is taken", app.on_key(Ev("s")) == "blocked"
          and A3.values[3] == "" and "break" in app.status.cget("text"))
    grab()
    check("... but quitting still saves", app.handle("quit") == "quit" and app.closed
          and jc.AnswerFile.load(d2 / cli.ANSWERS_NAME, S2).prior_minutes() > 49)
    leaks = [t for t in shown_texts if REVEALING.search(t)]
    check("no caption, status or title says boxer, chain, cost, stratum, A or B", not leaks, str(leaks[:3]))
elif root is not None:
    skip("the window tests", "no clip (ffmpeg missing)")

# ---- 9. the score -----------------------------------------------------------------------------------
print("the score")


def scored(n_d, n_u=0, positions=None, sub="s"):
    dd = tmp / "results" / f"score-{sub}-{n_d}-{n_u}"
    dd.mkdir(parents=True)
    smp_ = jc.build_sample(per_video(byv, positions), MAN)
    jc.write_new(dd / cli.SAMPLE_NAME, jc.sample_bytes(smp_))
    s_ = jc.load_sample(dd / cli.SAMPLE_NAME, MAN)
    a_ = jc.AnswerFile.create(dd / cli.ANSWERS_NAME, s_)
    for k in range(s_.n):
        a_.set_value(k, "D" if k < n_d else ("U" if k < n_d + n_u else "S"))
    a_.save()
    return dd, s_, jc.AnswerFile.load(dd / cli.ANSWERS_NAME, s_)


_, s0, a0 = scored(0)
r0 = jc.score(s0, a0)
check("0 wrong of 100 with ~23 joins per chain is inconclusive (cannot reach acceptable at n = 100)",
      r0["verdict"] == "measured, inconclusive" and r0["join_error"] == 0.0, r0["verdict"])
_, s10, a10 = scored(10)
r10 = jc.score(s10, a10)
check("10 wrong of 100 is unreliable", r10["verdict"] == "measured, unreliable")
check("the misattribution estimate is reported with its bounds and assumptions",
      r10["misattribution"]["at_lower"] <= r10["misattribution"]["at_rate"] <= r10["misattribution"]["at_upper"]
      and "independent" in r10["misattribution"]["assumptions"])
_, s1j, a1j = scored(0, positions=[[1, 1]], sub="one")
check("with one join per chain (p* = 10%), 0 of 100 is acceptable", jc.score(s1j, a1j)["verdict"]
      == "measured, acceptable")
_, su, au = scored(5, n_u=31)
ru = jc.score(su, au)
check("31 unsure of 100 is not established, and unsure is excluded from the rate",
      ru["verdict"] == "not established" and ru["decided"] == 69 and abs(ru["join_error"] - 5 / 69) < 1e-12)
check("the rate per stratum adds up", sum(ru["by_stratum"][x]["D"] for x in ("short", "long")) == 5)
check("a camera with under 20 decided is 'not established'",
      any("note" in v for v in ru["by_camera"].values()) or all(v["S"] + v["D"] >= 20 for v in ru["by_camera"].values()))
check("the bootstrap is reproducible from its seed", jc.score(s10, a10)["bootstrap"] == r10["bootstrap"])
check("the bootstrap gives an interval around the per-minute rate",
      r10["bootstrap"]["interval95"] is not None
      and r10["bootstrap"]["interval95"][0] <= r10["wrong_joins_per_chain_minute"] <= r10["bootstrap"]["interval95"][1])
_, sall, aall = scored(0, n_u=100, sub="u")
check("all unsure: no rate, not established", jc.score(sall, aall)["verdict"] == "not established"
      and jc.score(sall, aall)["join_error"] is None)
inc_d, sinc, ainc = scored(0, sub="inc")
ainc.values[5] = ""
raises("an incomplete answers file is not scored", lambda: jc.score(sinc, ainc), contains="not every join")

# ---- 10. the command line -------------------------------------------------------------------------
print("the command line")
real_manifest = cli.manifest
cli.manifest = lambda: MAN
try:
    dd, _, _ = scored(3, sub="cli")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["verify", "--dir", str(dd)])
    out = buf.getvalue()
    check("verify reports answered / empty / minutes only, never S / D / U counts",
          rc == 0 and "answered 100" in out and not re.search(r"\b[SDU]\b", out), out)
    with redirect_stdout(io.StringIO()):
        rc1 = cli.main(["score", "--dir", str(dd)])
    res = json.loads((dd / cli.RESULT_NAME).read_text())
    check("score writes the result once", rc1 == 0 and res["answers"]["D"] == 3 and res["sample_sha256"])
    with redirect_stdout(io.StringIO()):
        rc2 = cli.main(["score", "--dir", str(dd)])
    check("a second score is refused (the metric is computed once)", rc2 == 2)
    with redirect_stdout(io.StringIO()):
        rc3 = cli.main(["sample", "--dir", str(dd), "--data", str(tmp), "--tracks", str(tmp)])
    check("a second sample is refused before anything is linked", rc3 == 2)
    with redirect_stdout(io.StringIO()):
        rc4 = cli.main(["verify", "--dir", str(tmp / "nowhere")])
    check("a folder outside results/ is refused", rc4 == 2)
finally:
    cli.manifest = real_manifest
check("the real manifest is #29's 23 held-out videos", len(real_manifest()) == 23)
check("the library's copy of the manifest is exactly #29's", jc.HELD_OUT_MANIFEST == real_manifest())

print(f"\n=== {PASS} passed, {FAIL} failed, {SKIP} skipped ===")
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(1 if FAIL else 0)
