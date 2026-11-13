"""Tests for the local demo app's logic (src/boxingvi/demo.py) and its config.

No streamlit, no browser, no pose model, no ffmpeg run: the two subprocess seams (analyse_clip.py and
the web-preview transcode) are replaced by fakes that record what they were handed. Weighted toward
the trust boundaries: a path that climbs out of its folder, a participant folder, an upload that is
not a video or is called something clever, a stance or rounds string that is not what it claims, a
clip with one output missing, and an argument list that carries anything a shell would read.

The real-fact assertions count a fixture whose punches are known and compare the result with those
counts; nothing here says anything about boxing.

Run:  python tests/test_demo.py
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import analyse_clip  # noqa: E402
from boxingvi import demo  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def refused(name: str, fn, *args, **kwargs) -> None:
    """`fn` must raise DemoError, and nothing else."""
    try:
        fn(*args, **kwargs)
    except demo.DemoError:
        check(name, True)
    except Exception as e:   # noqa: BLE001 - the wrong exception is the failure being reported
        check(name, False, f"raised {type(e).__name__}: {e}")
    else:
        check(name, False, "did not raise")


def make_link(link: Path, target: Path) -> bool:
    """A symlink, else a directory junction (no privilege needed on Windows); False if neither works."""
    try:
        os.symlink(target, link, target_is_directory=target.is_dir())
        return True
    except OSError:
        pass
    if os.name == "nt" and target.is_dir():
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
        return r.returncode == 0
    return False


def write_clip(folder: Path, name: str, skip=()) -> None:
    for suffix in demo.OUTPUT_SUFFIXES:
        if suffix not in skip:
            (folder / f"{name}.{suffix}").write_bytes(b"x")


tmp = Path(tempfile.mkdtemp(prefix="demo-test-"))
SHELL_CHARS = set(";|&$`<>()\"'*?!{}[]\\\n\r\t")

# ------------------------------------------------------------------ containment ---
print("containment")
root = tmp / "root"
(root / "sub").mkdir(parents=True)
check("a plain name resolves inside the root", demo.contained(root, "a.txt") == (root / "a.txt").resolve())
check("a nested name resolves inside the root", demo.contained(root, "sub/a.txt") == (root / "sub" / "a.txt").resolve())
check("a .. that stays inside is fine", demo.contained(root, "sub/../a.txt") == (root / "a.txt").resolve())
for bad in ("../x", "sub/../../x", "..", "../root/../../x", "..\\x", "sub/../..", "a/../../" + root.name + "x"):
    refused(f"traversal {bad!a} is refused", demo.contained, root, bad)
for bad in ("C:\\Windows\\win.ini", "C:/Windows/win.ini", "/etc/passwd", "\\\\host\\share\\x", "\\x", "/x",
            str(tmp / "elsewhere.txt")):
    refused(f"absolute path {bad!a} is refused", demo.contained, root, bad)
refused("the root itself is refused", demo.contained, root, ".")
refused("an empty name is refused", demo.contained, root, "")
refused("a NUL in the name is refused", demo.contained, root, "a\x00.txt")
refused("a non-string is refused", demo.contained, root, None)
outside = tmp / "outside"
outside.mkdir()
(outside / "secret.txt").write_text("s")
if make_link(root / "escape", outside):
    refused("a link inside the root that leads out of it is refused", demo.contained, root, "escape/secret.txt")
    check("is_inside follows the link too", not demo.is_inside(root / "escape" / "secret.txt", root))
else:
    print("  SKIP  link escape (cannot create a link here)")
check("is_inside: a sibling folder with the root's name as prefix is not inside",
      not demo.is_inside(tmp / "root2" / "x", root) and demo.is_inside(root / "sub", root))

# ------------------------------------------------------------------ participant folders ---
print("participant folders")
proot = tmp / "comps-participant-data"
(proot / "P01").mkdir(parents=True)
roots = (proot,)
refused("a file under a participant folder is refused", demo.refuse_participant, proot / "P01" / "R1.mp4", roots)
refused("the participant folder itself is refused", demo.refuse_participant, proot, roots)
refused(".. out of a normal folder into the participant folder is refused", demo.refuse_participant,
        tmp / "root" / ".." / "comps-participant-data" / "P01", roots)
check("an ordinary folder is accepted", demo.refuse_participant(root, roots) == root.resolve())
refused("the real default participant folder is refused (it need not exist)", demo.refuse_participant,
        Path.home() / "comps-participant-data" / "x.mp4")
refused("so is the second default location", demo.refuse_participant,
        Path.home() / "repos" / "comps-participant-data" / "x.mp4")
refused("a results folder pointed at a participant folder is refused", demo.find_clips, proot, roots)
refused("an uploads folder pointed at a participant folder is refused", demo.save_upload, proot / "up", "a.mp4",
        io.BytesIO(b"v"), roots=roots)
check("nothing was created in the participant folder", sorted(p.name for p in proot.iterdir()) == ["P01"])
full_p = proot / "P01"
write_clip(full_p, "P01R1")
check("a complete set inside a participant folder is not 'complete' for the gallery",
      not demo.is_complete(full_p, "P01R1", roots))
if make_link(root / "to-participant", proot / "P01"):
    refused("a link from a normal folder into a participant folder is refused", demo.check_folder,
            root / "to-participant", roots)
else:
    print("  SKIP  link to participant folder (cannot create a link here)")
refused("deleting uploads in a participant folder is refused", demo.delete_upload, proot, "upload_20260101_000000_abcdef",
        roots)

# ------------------------------------------------------------------ gallery: complete clips only ---
print("gallery")
res = tmp / "results"
res.mkdir()
write_clip(res, "IMG_0001")
write_clip(res, "IMG_0002")
check("two complete clips are found, by name", [c.name for c in demo.find_clips(res)] == ["IMG_0001", "IMG_0002"])
for suffix in demo.OUTPUT_SUFFIXES:
    write_clip(res, "IMG_0003", skip=(suffix,))
    got = [c.name for c in demo.find_clips(res)]
    check(f"a clip missing {suffix} is excluded", "IMG_0003" not in got and got == ["IMG_0001", "IMG_0002"], str(got))
    for f in res.glob("IMG_0003.*"):
        f.unlink()
write_clip(res, "IMG_0004", skip=("overlay.mp4",))
check("a --no-overlay run (six of seven) is excluded", "IMG_0004" not in [c.name for c in demo.find_clips(res)])
for f in res.glob("IMG_0004.*"):
    f.unlink()
write_clip(res, "IMG_0005", skip=("report.xlsx",))
(res / "IMG_0005.report.xlsx").mkdir()
check("a folder with an output's name does not count as that output",
      "IMG_0005" not in [c.name for c in demo.find_clips(res)])
(res / ".IMG_0006.abc123").mkdir()
(res / ".IMG_0006.abc123" / "IMG_0006.pose.npz").write_bytes(b"x")
(res / "IMG_0001.overlay.web.mp4").write_bytes(b"x")
(res / "run.log").write_text("log")
(res / "bad name.summary.txt").write_text("x")
(res / "a.b.summary.txt").write_text("x")
got = [c.name for c in demo.find_clips(res)]
check("staging folders, web previews, logs and oddly named files are not clips", got == ["IMG_0001", "IMG_0002"], str(got))
check("a results folder that does not exist is an empty gallery", demo.find_clips(tmp / "nope") == [])
check("Clip.path builds only known outputs", demo.find_clips(res)[0].path("summary.txt") == (res / "IMG_0001.summary.txt").resolve())
refused("Clip.path refuses anything else", demo.find_clips(res)[0].path, "../x")
# an output that is a link out of the folder (file link where possible)
linked = tmp / "linked"
linked.mkdir()
write_clip(linked, "IMG_0008", skip=("summary.txt",))
(outside / "elsewhere.txt").write_text("not yours")
try:
    os.symlink(outside / "elsewhere.txt", linked / "IMG_0008.summary.txt")
    check("an output that is a link leading outside the folder is excluded",
          "IMG_0008" not in [c.name for c in demo.find_clips(linked)])
except OSError:
    print("  SKIP  file symlink (cannot create one here)")

# ------------------------------------------------------------------ upload names and extensions ---
print("upload names and extensions")
names = [demo.new_upload_name() for _ in range(300)]
check("every generated name matches analyse_clip's clip-name rule", all(analyse_clip._CLIP_NAME.fullmatch(n) for n in names))
check("every generated name is upload_<date>_<time>_<6 hex>",
      all(re.fullmatch(r"upload_\d{8}_\d{6}_[0-9a-f]{6}", n) for n in names))
check("300 generated names are all different", len(set(names)) == 300)
check("the name carries the time it was made",
      demo.new_upload_name(datetime(2026, 11, 12, 9, 5, 7)).startswith("upload_20261112_090507_"))
check("is_upload_name accepts a generated name", demo.is_upload_name(names[0]))
for bad in ("IMG_4406", "upload", "upload_../x", "upload_a b", "upload_a.b", "", None, "upload_" + "a" * 100):
    check(f"is_upload_name refuses {bad!a}", not demo.is_upload_name(bad))
check("accepted extensions, any case: .mp4 .MOV .m4v",
      [demo.upload_extension(n) for n in ("a.mp4", "B.MOV", "c.M4v")] == [".mp4", ".mov", ".m4v"])
for bad in ("a.avi", "a.mkv", "a.exe", "a.mp4.exe", "a", "", ".mp4x", "a.mp4 ", "a.mp4\x00", "a.mp4.", "video.webm",
            "a.zip", None, 5):
    refused(f"extension of {bad!a} is refused", demo.upload_extension, bad)
refused("a file called just '.mp4' has no extension and is refused", demo.upload_extension, ".mp4")

# ------------------------------------------------------------------ saving an upload ---
print("saving an upload")
up = tmp / "uploads"
p = demo.save_upload(up, "../../Windows/evil name; rm -rf.MP4", io.BytesIO(b"abc" * 100))
check("the file lands inside the uploads folder under a generated name",
      p.parent == up.resolve() and analyse_clip._CLIP_NAME.fullmatch(p.stem) and p.stem.startswith("upload_")
      and p.suffix == ".mp4", str(p))
check("nothing of the uploaded name is kept", "evil" not in p.name and "Windows" not in p.name and ";" not in p.name)
check("the bytes are written as sent", p.read_bytes() == b"abc" * 100)
check("nothing else appeared outside the uploads folder", not (tmp / "Windows").exists() and sorted(x.name for x in up.iterdir()) == [p.name])
p2 = demo.save_upload(up, "a.mov", io.BytesIO(b"z"))
check("two uploads never share a path", p2 != p and p2.suffix == ".mov")
before = sorted(x.name for x in up.iterdir())
refused("a bad extension is refused before anything is written", demo.save_upload, up, "clip.exe", io.BytesIO(b"v"))
refused("a double extension is refused", demo.save_upload, up, "clip.mp4.exe", io.BytesIO(b"v"))
refused("an empty file is refused", demo.save_upload, up, "a.mp4", io.BytesIO(b""))
refused("a file over the size cap is refused", demo.save_upload, up, "a.mp4", io.BytesIO(b"x" * 5000), max_bytes=4096)
check("a refused upload leaves no file behind", sorted(x.name for x in up.iterdir()) == before)
check("the cap in the code equals the cap in .streamlit/config.toml",
      tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))["server"]["maxUploadSize"]
      == demo.MAX_UPLOAD_BYTES // (1024 * 1024))
check("list_uploads lists stored uploads, newest name first, and nothing else",
      [x.name for x in demo.list_uploads(up)] == sorted((p.name, p2.name), reverse=True))
check("find_upload finds a stored upload by its name", demo.find_upload(up, p.stem) == p.resolve())
check("find_upload refuses a name this app did not make", demo.find_upload(up, "IMG_4406") is None
      and demo.find_upload(up, "upload_../x") is None)

# ------------------------------------------------------------------ rounds, stance, device ---
print("rounds, stance and device")
check("rounds: empty and None mean the whole clip", demo.validate_rounds("") is None and demo.validate_rounds(None) is None
      and demo.validate_rounds("   ") is None)
check("rounds: the documented example is accepted", demo.validate_rounds("0-180,240-420") == "0-180,240-420")
check("rounds: decimals and spaces are accepted", demo.validate_rounds(" 0.5-10.25 , 12 - 30 ") == "0.5-10.25 , 12 - 30")
check("analyse_clip's own parser accepts every string this validator accepts",
      all(analyse_clip.report.parse_rounds(demo.validate_rounds(s), 1000.0)
          for s in ("0-180,240-420", "1-2", "0.5-10.25 , 12 - 30")))
for bad in ("abc", "0-180;calc", "0-180 && dir", "0-180\n240-420", "$(whoami)", "`id`", "180-0", "0-0", "0-100,50-200",
            "-5-10", "1e3-5", "0-", "-10", "0-10,", ",0-10", "0..5-10", "0-180|x", "0–180", "٠-١٨٠",
            "0-1," * 50, "9" * 300, 5):
    refused(f"rounds {bad!a} is refused", demo.validate_rounds, bad)
check("stance: both real stances accepted", [demo.validate_stance(s) for s in ("orthodox", "southpaw")] == ["orthodox", "southpaw"])
for bad in (None, "", "Orthodox", "orthodox ", "orthodox; calc", "left", "--video", "ambi"):
    refused(f"stance {bad!a} is refused (no default, no guess)", demo.validate_stance, bad)
check("device: cpu is always valid", demo.validate_device("cpu") == "cpu")
refused("device: cuda:0 is refused when CUDA is not available", demo.validate_device, "cuda:0")
check("device: cuda:0 is accepted when it is", demo.validate_device("cuda:0", True) == "cuda:0")
for bad in ("gpu", "cuda", "cuda:", "cuda:100", "cpu; x", "cuda:0 --weights x", "", None):
    refused(f"device {bad!a} is refused", demo.validate_device, bad, True)
check("the device list always starts with cpu", demo.available_devices()[0] == "cpu")

# ------------------------------------------------------------------ argv for analyse_clip ---
print("analyse_clip argv")
vid = up / f"{demo.new_upload_name()}.mp4"
vid.write_bytes(b"v")
out = demo.out_dir_for(up)
argv = demo.build_analyse_argv(vid, "southpaw", "cpu", "0-30, 40-60", out)
check("argv is a list of strings, starting with this interpreter",
      isinstance(argv, list) and all(isinstance(a, str) for a in argv) and argv[0] == sys.executable)
check("argv runs analyse_clip.py by absolute path", Path(argv[argv.index("-u") + 1]) == (ROOT / "analyse_clip.py").resolve()
      or Path(argv[2]).name == "analyse_clip.py")
check("--stance carries the stance exactly", argv[argv.index("--stance") + 1] == "southpaw" and argv.count("--stance") == 1)
check("--video and --out-dir are absolute", Path(argv[argv.index("--video") + 1]).is_absolute()
      and Path(argv[argv.index("--out-dir") + 1]).is_absolute())
check("--device is passed", argv[argv.index("--device") + 1] == "cpu")
check("rounds are one --rounds=<text> element", "--rounds=0-30, 40-60" in argv)
check("no --rounds when none was given", not any(a.startswith("--rounds") for a in
                                                 demo.build_analyse_argv(vid, "orthodox", "cpu", "", out)))
user_text = [argv[argv.index("--stance") + 1], argv[argv.index("--device") + 1]]
check("no user-supplied element contains a shell metacharacter",
      not any(set(t) & SHELL_CHARS for t in user_text) and not (set("0-30, 40-60") & SHELL_CHARS - set()))
for stance, device, rounds in (("orthodox; calc", "cpu", None), ("orthodox", "cpu && calc", None),
                               ("orthodox", "cpu", "0-5; calc"), ("orthodox", "cpu", "0-5 | more"), (None, "cpu", None)):
    refused(f"argv is not built for {stance!a}/{device!a}/{rounds!a}", demo.build_analyse_argv, vid, stance, device, rounds, out)
for bad_video in ("IMG_4406.mov", "my video.mp4", "upload_20260101_000000_abcdef.mp4.exe", "../upload_x.mp4",
                  "upload_20260101_000000_abcdef.avi", "upload_x$(id).mp4", "-upload_x.mp4"):
    refused(f"argv is not built for the video {bad_video!a}", demo.build_analyse_argv, up / bad_video, "orthodox", "cpu", None, out)
check("a path that starts with - cannot read as an option: video and out-dir come after their flags, absolute",
      all(not a.startswith("-") or a.startswith(("--", "-u")) for a in argv))

# the subprocess seam: a fake Popen records exactly what it is handed
calls = []


class FakeProc:
    def __init__(self, lines, rc=0):
        self.stdout = iter(lines)
        self._rc = rc

    def wait(self):
        return self._rc

    def poll(self):
        return self._rc

    def kill(self):
        pass


def fake_popen(argv_, **kw):
    calls.append((argv_, kw))
    return FakeProc(["== ingest ==\r\n", "frames 10\n", "done\n"], rc=0)


seen = []
rc = demo.run_analyse(argv, on_line=seen.append, popen=fake_popen)
check("run_analyse returns the exit code and streams the lines, line endings stripped",
      rc == 0 and seen == ["== ingest ==", "frames 10", "done"], str(seen))
a_, kw = calls[0]
check("it passed the argument list unchanged", a_ is argv or a_ == argv)
check("it never used a shell and gave the child no stdin", kw.get("shell") is False and kw.get("stdin") == subprocess.DEVNULL)
check("stderr is merged into the stream so errors show in order", kw.get("stderr") == subprocess.STDOUT)
check("the child makes no network call of its own (YOLO_OFFLINE=1) and is unbuffered",
      kw["env"].get("YOLO_OFFLINE") == "1" and kw["env"].get("PYTHONUNBUFFERED") == "1")
check("a failing child's exit code is returned", demo.run_analyse(argv, popen=lambda a, **k: FakeProc(["err\n"], rc=1)) == 1)
refused("a command string is refused (list only)", demo.run_analyse, "python analyse_clip.py --video x")
refused("a list with a non-string is refused", demo.run_analyse, ["python", Path("x")])

# ------------------------------------------------------------------ web preview ---
print("web preview")
ov = res / "IMG_0001.overlay.mp4"
ov.write_bytes(b"o" * 10)
fa = demo.ffmpeg_web_argv("ffmpeg", ov, res / "o.tmp.mp4")
check("the ffmpeg argv is exactly the specified transcode",
      fa[:1] == ["ffmpeg"] and fa[fa.index("-vf") + 1] == "scale=-2:720" and fa[fa.index("-c:v") + 1] == "libx264"
      and fa[fa.index("-crf") + 1] == "28" and fa[fa.index("-preset") + 1] == "veryfast"
      and fa[fa.index("-movflags") + 1] == "+faststart" and "-an" in fa, str(fa))
check("ffmpeg gets the source after -i and the destination last, both absolute",
      fa[fa.index("-i") + 1] == str(ov.resolve()) and fa[-1] == str((res / "o.tmp.mp4").resolve()))
check("ffmpeg's argv is a list of strings, never reads stdin", all(isinstance(x, str) for x in fa) and "-nostdin" in fa)
check("the preview sits next to the overlay as <clip>.overlay.web.mp4",
      demo.web_preview_path(ov) == res / "IMG_0001.overlay.web.mp4")
refused("a path that is not an overlay has no preview name", demo.web_preview_path, res / "IMG_0001.report.xlsx")
for f in res.glob("IMG_0001.overlay.web.mp4"):
    f.unlink()
made = []


def fake_run(argv_, **kw):
    made.append((argv_, kw))
    Path(argv_[-1]).write_bytes(b"web")
    return subprocess.CompletedProcess(argv_, 0, "", "")


w = demo.ensure_web_preview(ov, ffmpeg="ffmpeg", run=fake_run)
check("first view makes the preview with ffmpeg", w == res / "IMG_0001.overlay.web.mp4" and w.read_bytes() == b"web" and len(made) == 1)
check("ffmpeg ran without a shell", made[0][1].get("shell") is False and isinstance(made[0][0], list))
check("no temporary file is left next to it", [f.name for f in res.glob("*.tmp.mp4")] == [])
demo.ensure_web_preview(ov, ffmpeg="ffmpeg", run=fake_run)
check("a fresh cached preview is not made again", len(made) == 1)
os.utime(ov, (w.stat().st_mtime + 100, w.stat().st_mtime + 100))
demo.ensure_web_preview(ov, ffmpeg="ffmpeg", run=fake_run)
check("a preview older than the overlay is made again", len(made) == 2)
w.unlink()
check("ffmpeg missing: None, so the page offers the original instead of crashing",
      demo.ensure_web_preview(ov, ffmpeg=None, run=fake_run) is None and len(made) == 2)
refused("a failing ffmpeg is a refusal, not a crash", demo.ensure_web_preview, ov, "ffmpeg",
        lambda a, **k: subprocess.CompletedProcess(a, 1, "", "boom"))
refused("an ffmpeg that exits 0 but wrote nothing is a refusal", demo.ensure_web_preview, ov, "ffmpeg",
        lambda a, **k: subprocess.CompletedProcess(a, 0, "", ""))


def timeout_run(a, **k):
    raise subprocess.TimeoutExpired(a, 1)


refused("an ffmpeg that times out is a refusal", demo.ensure_web_preview, ov, "ffmpeg", timeout_run)
check("failures left no preview and no temporary file", not list(res.glob("*.tmp.mp4")) and not (res / "IMG_0001.overlay.web.mp4").exists())

# ------------------------------------------------------------------ reading what is shown ---
print("what the page shows")
text = "In one round the tool found 3 punches.  \n\nCaveat: not checked against labels.\u00e9 \u2014 end\n\n"
(res / "S.summary.txt").write_bytes(text.encode("utf-8"))
check("summary.txt is returned verbatim: spaces, blank lines, non-ASCII, trailing newlines",
      demo.read_summary(res / "S.summary.txt") == text)
big = res / "BIG.summary.txt"
big.write_bytes(b"x" * (demo.MAX_TEXT_BYTES + 1))
refused("a summary over the size cap is refused", demo.read_summary, big)

header = "frame,time_s,hand,peak_speed_bl_per_s,peak_on_interpolated_frame,lead_rear,punch_type,confidence,type_hand,hands_agree,note"
known = [
    ("10,0.5,left,4.0,no,lead,jab,0.9,lead,yes,"),
    ("20,1.0,right,5.0,no,rear,cross,0.8,rear,yes,"),
    ("30,1.5,left,4.5,no,lead,jab,0.7,lead,yes,"),
    ("40,2.0,left,6.0,no,lead,lead_hook,0.6,lead,no,"),
    ("50,2.5,left,4.1,no,lead,jab,0.9,lead,yes,"),
    ("60,3.0,right,5.5,no,rear,cross,0.95,rear,yes,"),
    ("70,3.5,right,5.0,no,rear,,0.1,rear,no,"),
]
csvp = tmp / "known.classified.csv"
csvp.write_text(header + "\n" + "\n".join(known) + "\n", encoding="utf-8")
s = demo.summarise_punches(csvp)
check("REAL FACT: a fixture of 3 jabs, 2 crosses, 1 lead hook and 1 unnamed counts exactly so",
      s.total == 7 and s.counts == {"jab": 3, "cross": 2, "lead_hook": 1, "unnamed": 1}, str(s.counts))
check("counts come most common first, ties by name", list(s.counts) == ["jab", "cross", "lead_hook", "unnamed"])
check("the counts add up to the rows", sum(s.counts.values()) == s.total == len(s.rows))
check("rows hold exactly the display columns, in file order",
      list(s.rows[0]) == list(demo.TABLE_COLUMNS) and s.rows[3]["punch_type"] == "lead_hook" and s.rows[3]["hands_agree"] == "no"
      and s.rows[0]["time_s"] == "0.5" and s.rows[6]["confidence"] == "0.1")
no_col = tmp / "nocol.classified.csv"
no_col.write_text("frame,time_s,hand\n1,0.1,left\n", encoding="utf-8")
refused("a CSV missing a display column is refused, not shown with a gap", demo.summarise_punches, no_col)
empty = tmp / "empty.classified.csv"
empty.write_text(header + "\n", encoding="utf-8")
e = demo.summarise_punches(empty)
check("a CSV with no punches summarises to zero", e.total == 0 and e.counts == {} and e.rows == [])
real = demo.MAX_CSV_ROWS
demo.MAX_CSV_ROWS = 3
refused("a CSV over the row cap is refused", demo.summarise_punches, csvp)
demo.MAX_CSV_ROWS = real
demo.MAX_CSV_BYTES, real_b = 10, demo.MAX_CSV_BYTES
refused("a CSV over the size cap is refused", demo.summarise_punches, csvp)
demo.MAX_CSV_BYTES = real_b

# ------------------------------------------------------------------ deleting an upload ---
print("deleting an upload")
name = demo.new_upload_name()
(up / f"{name}.mov").write_bytes(b"v")
out.mkdir(exist_ok=True)
write_clip(out, name)
(out / f"{name}.overlay.web.mp4").write_bytes(b"w")
(out / f".{name}.kept").mkdir()
(out / f".{name}.kept" / f"{name}.pose.npz").write_bytes(b"p")
other = demo.new_upload_name()
(up / f"{other}.mp4").write_bytes(b"v")
write_clip(out, other)
(out / "IMG_4406.summary.txt").write_text("not an upload")
(tmp / "precious.txt").write_text("keep me")
gone = demo.delete_upload(up, name)
check("the video, seven outputs, web preview and staging folder are removed",
      len(gone) == 1 + 7 + 1 + 1 and not (up / f"{name}.mov").exists() and not list(out.glob(f"{name}.*"))
      and not (out / f".{name}.kept").exists(), str(gone))
check("another upload and a non-upload file in the same folder are untouched",
      (up / f"{other}.mp4").exists() and (out / f"{other}.summary.txt").exists() and (out / "IMG_4406.summary.txt").exists())
for bad in ("IMG_4406", "../precious", "upload_../../precious", "upload_x/../..", "", None, "upload_a.b", "*"):
    refused(f"delete refuses the name {bad!a}", demo.delete_upload, up, bad)
check("and nothing outside was touched", (tmp / "precious.txt").read_text() == "keep me" and (out / "IMG_4406.summary.txt").exists())
check("deleting something already gone removes nothing and does not raise", demo.delete_upload(up, name) == [])
name2 = demo.new_upload_name()
write_clip(out, name2, skip=("summary.txt",))
try:
    os.symlink(tmp / "precious.txt", out / f"{name2}.summary.txt")
    demo.delete_upload(up, name2)
    check("a link in the outputs is removed as a link; its target survives",
          not (out / f"{name2}.summary.txt").is_symlink() and (tmp / "precious.txt").read_text() == "keep me")
except OSError:
    print("  SKIP  file symlink delete (cannot create one here)")

# ------------------------------------------------------------------ config and source hygiene ---
print("config and source")
cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
check("the server binds to 127.0.0.1 only", cfg["server"]["address"] == "127.0.0.1")
check("usage statistics are off", cfg["browser"]["gatherUsageStats"] is False)
check("the upload cap is set and sane (1..2048 MB)", 1 <= cfg["server"]["maxUploadSize"] <= 2048)
check("CORS and XSRF protection stay on (not disabled in config)",
      cfg.get("server", {}).get("enableCORS", True) is True and cfg.get("server", {}).get("enableXsrfProtection", True) is True)
check("headless is on: no browser opened behind the app's back, no first-run e-mail prompt",
      cfg["server"].get("headless") is True)
sources = {n: (ROOT / n).read_text(encoding="utf-8") for n in ("src/boxingvi/demo.py", "demo_app.py")}
for n, src in sources.items():
    check(f"{n}: no shell=True, no os.system, no eval/exec",
          not re.search(r"shell\s*=\s*True|os\.system|\beval\(|\bexec\(", src))
    check(f"{n}: no network library is imported", not re.search(r"^\s*(import|from)\s+(requests|urllib|http|socket|httpx|aiohttp)\b", src, re.M))
check("the demo has no external assets: no http(s) URL is loaded by either file",
      not re.search(r"https?://", sources["demo_app.py"] + sources["src/boxingvi/demo.py"]))
check("demo.py is importable without streamlit having been imported", "streamlit" not in sys.modules)
check("requirements-demo.txt pins exactly; requirements.txt does not mention the demo",
      all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.+-]*", ln)
          for ln in (ROOT / "requirements-demo.txt").read_text(encoding="utf-8").splitlines()
          if ln.strip() and not ln.lstrip().startswith("#"))
      and "streamlit" not in (ROOT / "requirements.txt").read_text(encoding="utf-8").lower())

import shutil  # noqa: E402

shutil.rmtree(tmp, ignore_errors=True)
print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
