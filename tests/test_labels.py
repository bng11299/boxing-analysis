"""Tests for the Label Studio import (src/boxingvi/lsimport.py, import_labels.py).

The export is untrusted and becomes the ground truth every punch-finding number
is scored against, so most of these are the exports it must refuse.

Run:  python tests/test_labels.py
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))

import import_labels  # noqa: E402
from boxingvi import lsimport  # noqa: E402
from boxingvi.data import CLASSES, Span  # noqa: E402

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
    """Must raise ValueError whose message contains `pattern` -- another error is a failure."""
    try:
        fn()
    except ValueError as e:
        check(name, pattern in str(e), f"wrong reason: {e}")
    except Exception as e:
        check(name, False, f"raised {type(e).__name__}: {e}")
    else:
        check(name, False, "accepted")


def region(label, start, end):
    return {"type": "timelinelabels", "from_name": "punches", "to_name": "video",
            "value": {"ranges": [{"start": start, "end": end}], "timelinelabels": [label]}}


def export(regions, offset=1, lead_time=600.5, extra_ann=None):
    """A Label Studio "JSON" export of one task. LS 1.23 frames are 1-based, so the default
    offset is +1, as a real export has; `offset` shifts every frame as LS would."""
    res = [region(lab, s + offset, e + offset) for lab, s, e in regions]
    anns = [{"id": 1, "result": res, "lead_time": lead_time, "was_cancelled": False}]
    if extra_ann:
        anns.append(extra_ann)
    return [{"id": 7, "data": {"video": "/data/upload/1/IMG_4406.overlay.mp4"}, "annotations": anns}]


N = 2353
CHECKS = [("frame-check", 100, 100), ("frame-check", 2000, 2000)]
PUNCHES = [("jab", 300, 310), ("cross", 312, 325), ("lead_hook", 330, 345),
           ("rear_uppercut", 400, 418), ("unclear", 500, 506), ("jab", 311, 320)]

print("good exports")
ls = lsimport.parse_export(export(CHECKS + PUNCHES), N)
check("Label Studio's +1 offset is measured from the frame checks and removed", ls.offset == 1
      and Span(300, 310, CLASSES.index("jab")) in ls.spans, str(ls.spans[:2]))
check("frame checks are not punches", len(ls.spans) == len(PUNCHES))
check("an unclear punch is kept as a punch with label -1",
      Span(500, 506, lsimport.UNCLEAR_LABEL) in ls.spans)
check("counts per class, unclear included",
      ls.counts["jab"] == 2 and ls.counts["cross"] == 1 and ls.counts["unclear"] == 1)
check("spans come out in time order", [s.start for s in ls.spans] == sorted(s.start for s in ls.spans))
check("Label Studio's lead_time is reported as the labelling time", ls.lead_time_s == 600.5)
check("a missing lead_time is None, not 0",
      lsimport.parse_export(export(CHECKS + PUNCHES, lead_time=None), N).lead_time_s is None)
check("a cancelled annotation is ignored, not counted",
      len(lsimport.parse_export(export(CHECKS + PUNCHES, extra_ann={"was_cancelled": True, "result": []}), N).spans)
      == len(PUNCHES))
check("lead and rear punches may overlap (a cross can start before the jab lands)",
      len(lsimport.parse_export(export(CHECKS + [("jab", 300, 320), ("cross", 310, 330)]), N).spans) == 2)

print("frame numbering")
refuses("no frame checks -> refused", lambda: lsimport.parse_export(export(PUNCHES), N), "frame-check")
refuses("one frame check -> refused", lambda: lsimport.parse_export(export(CHECKS[:1] + PUNCHES), N), "found 1")
bad = export(CHECKS + PUNCHES)
bad[0]["annotations"][0]["result"][1]["value"]["ranges"][0] = {"start": 2005, "end": 2005}
refuses("checks that disagree (a wrong frameRate drifts) -> refused", lambda: lsimport.parse_export(bad, N),
        "disagree")
refuses("both checks off by 5 frames -> refused", lambda: lsimport.parse_export(export(CHECKS + PUNCHES, offset=5), N),
        "expected +1")
dragged = [("frame-check", 100, 102), ("frame-check", 2000, 2000)]
refuses("a dragged (multi-frame) check -> refused", lambda: lsimport.parse_export(export(dragged + PUNCHES), N),
        "single frame")

print("ranges and labels")
refuses("a punch past the last frame -> refused",
        lambda: lsimport.parse_export(export(CHECKS + [("jab", 2350, 2360)]), N), "outside the video")
one_based = [("frame-check", 101, 101), ("frame-check", 2001, 2001), ("jab", 0, 3)]  # LS frame 0 -> video -1
refuses("a punch before frame 0 once the offset is removed -> refused",
        lambda: lsimport.parse_export(export(one_based, offset=0), N), "outside the video")
refuses("end before start -> refused",
        lambda: lsimport.parse_export(export(CHECKS + [("jab", 310, 300)]), N), "before it starts")
refuses("an unknown label -> refused",
        lambda: lsimport.parse_export(export(CHECKS + [("haymaker", 300, 310)]), N), "unknown label")
refuses("two jabs overlapping -> refused (one hand, one punch at a time)",
        lambda: lsimport.parse_export(export(CHECKS + [("jab", 300, 320), ("lead_hook", 315, 330)]), N), "overlap")
frac = export(CHECKS + PUNCHES)
frac[0]["annotations"][0]["result"][2]["value"]["ranges"][0]["start"] = 300.5
refuses("a fractional frame -> refused", lambda: lsimport.parse_export(frac, N), "whole frame")
boo = export(CHECKS + PUNCHES)
boo[0]["annotations"][0]["result"][2]["value"]["ranges"][0]["start"] = True
refuses("a boolean frame (True == 1 in Python) -> refused", lambda: lsimport.parse_export(boo, N), "whole frame")
two_lab = export(CHECKS + PUNCHES)
two_lab[0]["annotations"][0]["result"][2]["value"]["timelinelabels"] = ["jab", "cross"]
refuses("a region with two labels -> refused", lambda: lsimport.parse_export(two_lab, N), "exactly one label")
other = export(CHECKS + PUNCHES)
other[0]["annotations"][0]["result"].append({"type": "choices", "value": {"choices": ["x"]}})
refuses("a result type the config does not produce -> refused", lambda: lsimport.parse_export(other, N),
        "unexpected result type")

print("export shape")
refuses("two tasks -> refused", lambda: lsimport.parse_export(export(CHECKS) + export(CHECKS), N), "exactly one task")
refuses("an empty export -> refused", lambda: lsimport.parse_export([], N), "exactly one task")
refuses("not a list -> refused", lambda: lsimport.parse_export({"annotations": []}, N), "exactly one task")
mini = [{"video": "x.mp4", "punches": []}]
refuses("JSON-MIN format -> refused with the fix named", lambda: lsimport.parse_export(mini, N), "JSON-MIN")
second = export(CHECKS + PUNCHES, extra_ann={"id": 2, "result": [], "was_cancelled": False})
refuses("two completed annotations -> refused (which is ground truth?)",
        lambda: lsimport.parse_export(second, N), "exactly one completed annotation")
refuses("a video with no frames -> refused", lambda: lsimport.parse_export(export(CHECKS), 0), "no frames")

print("Label Studio 1.23 specifics (from its source)")
refuses("offset 0 (checks marked one frame early) -> refused, not silently shifted",
        lambda: lsimport.parse_export(export(CHECKS + PUNCHES, offset=0), N), "expected +1")
refuses("offset -1 -> refused", lambda: lsimport.parse_export(export(CHECKS + PUNCHES, offset=-1), N), "expected +1")
nul = export(CHECKS + PUNCHES)
nul[0]["annotations"][0]["result"][2]["value"]["ranges"][0]["end"] = None  # LS types start/end as maybeNull
refuses("a null range end -> refused", lambda: lsimport.parse_export(nul, N), "whole frame")
wrong_tag = export(CHECKS + PUNCHES)
wrong_tag[0]["annotations"][0]["result"][2]["from_name"] = "videoLabels"
refuses("a region from another config's tag -> refused", lambda: lsimport.parse_export(wrong_tag, N), "punch-labels.xml")
draft = export(CHECKS + PUNCHES)
draft[0]["drafts"] = [{"result": [], "lead_time": 30}]
draft[0]["annotations"] = []
refuses("only a draft saved -> refused, and the message says to press Submit",
        lambda: lsimport.parse_export(draft, N), "press Submit")
inf_lead = export(CHECKS + PUNCHES, lead_time=float("inf"))
check("an infinite lead_time is dropped, not written as Infinity",
      lsimport.parse_export(inf_lead, N).lead_time_s is None)

print("unclear duplicates (unclear has no hand, so the hand rule cannot catch them)")
refuses("two unclear regions overlapping -> refused",
        lambda: lsimport.parse_export(export(CHECKS + [("unclear", 500, 510), ("unclear", 508, 515)]), N), "duplicates")
refuses("an unclear mostly on top of a named punch -> refused",
        lambda: lsimport.parse_export(export(CHECKS + [("jab", 500, 510), ("unclear", 501, 511)]), N), "duplicates")
ok = lsimport.parse_export(export(CHECKS + [("cross", 500, 520), ("unclear", 515, 530)]), N)
check("an unclear partly overlapping another punch (IoU < 0.5) is allowed", len(ok.spans) == 2)
ok2 = lsimport.parse_export(export(CHECKS + [("unclear", 500, 510), ("unclear", 511, 520)]), N)
check("two unclear regions back to back (no overlap) are allowed", len(ok2.spans) == 2)

print("hostile JSON")
tmpj = Path(tempfile.mkdtemp())
(tmpj / "inf.json").write_text('[{"x": Infinity}]', encoding="utf-8")
refuses("Infinity in the file -> refused at load", lambda: lsimport.load_export(tmpj / "inf.json"), "not a number")
(tmpj / "nan.json").write_text('[{"x": NaN}]', encoding="utf-8")
refuses("NaN in the file -> refused at load", lambda: lsimport.load_export(tmpj / "nan.json"), "not a number")
(tmpj / "deep.json").write_text("[" * 100000 + "]" * 100000, encoding="utf-8")
refuses("pathological nesting -> refused at load with a clean error",
        lambda: lsimport.load_export(tmpj / "deep.json"), "nested too deeply")

print("import_labels CLI")
tmp = Path(tempfile.mkdtemp())
clip = tmp / "IMG_9999.overlay.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x96:r=60",
                "-frames:v", "2100", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)
ex = tmp / "export.json"
ex.write_text(json.dumps(export(CHECKS + [("jab", 300, 310), ("cross", 2050, 2060)], offset=1)), encoding="utf-8")
rc = import_labels.main(["--export", str(ex), "--video", str(clip), "--out-dir", str(tmp / "out")])
out = tmp / "out" / "IMG_9999.labels.json"
got = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
check("end to end: labels written under the clip's name, offset removed",
      rc == 0 and got.get("spans") == [{"start": 300, "end": 310, "label": "jab"},
                                        {"start": 2050, "end": 2060, "label": "cross"}]
      and got.get("n_frames") == 2100 and got.get("label_studio_offset") == 1, str(got)[:200])
ex.write_text(json.dumps(export(CHECKS + [("cross", 2150, 2160)])), encoding="utf-8")
try:
    import_labels.main(["--export", str(ex), "--video", str(clip), "--out-dir", str(tmp / "out2")])
    check("the CLI measures the video's real length (2100 frames) and refuses a punch past it", False, "accepted")
except ValueError as e:
    check("the CLI measures the video's real length (2100 frames) and refuses a punch past it",
          "outside the video" in str(e), str(e))
check("... and writes nothing", not (tmp / "out2" / "IMG_9999.labels.json").exists())
check("the export's sha256 and the Label Studio convention are recorded with the labels",
      len(got.get("export_sha256", "")) == 64 and "+1" in got.get("label_studio", ""))
ex.write_text(json.dumps(export(CHECKS + [("jab", 300, 310)], offset=1)), encoding="utf-8")
try:
    import_labels.main(["--export", str(ex), "--video", str(clip), "--out-dir", str(tmp / "out")])
    check("existing ground-truth labels are not overwritten without --force", False, "overwrote")
except ValueError as e:
    check("existing ground-truth labels are not overwritten without --force", "--force" in str(e), str(e))
rc = import_labels.main(["--export", str(ex), "--video", str(clip), "--out-dir", str(tmp / "out"), "--force"])
check("... and are with --force", rc == 0 and len(json.loads(out.read_text(encoding="utf-8"))["spans"]) == 1)
vfr = tmp / "IMG_8888.overlay.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x96:r=60",
                "-frames:v", "2100", "-vf", "select='not(eq(n\\,1000))'", "-fps_mode", "passthrough",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vfr)], check=True)
try:
    import_labels.main(["--export", str(ex), "--video", str(vfr), "--out-dir", str(tmp / "out3")])
    check("a video with a dropped frame (VFR) is refused: LS frame numbers would drift", False, "accepted")
except ValueError as e:
    check("a video with a dropped frame (VFR) is refused: LS frame numbers would drift", "irregular" in str(e), str(e))
big = tmp / "big.json"
big.write_bytes(b" " * (lsimport.MAX_EXPORT_BYTES + 1))
refuses("an export over the size cap is refused before parsing", lambda: lsimport.load_export(big), "MB")

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
