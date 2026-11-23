"""Plain assert tests; run with Python 3.13. ffmpeg/ffprobe are required.

python tests/test_sparring_labels.py
python tests/test_sparring_labels.py --mutations
Mutations run isolated source copies; production files are never modified.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "cli")]

import import_labels
from boxingvi import lsimport, sparring_io
from boxingvi.video import iter_frames

spec = importlib.util.spec_from_file_location("number_frames", ROOT / "scripts/number_frames.py")
number_frames = importlib.util.module_from_spec(spec)
spec.loader.exec_module(number_frames)


def refuse(fn, reason):
    try:
        fn()
    except ValueError as exc:
        assert reason in str(exc), str(exc)
        assert "PRIVATE_CANARY" not in str(exc), "error leaked export contents"
    else:
        raise AssertionError(f"accepted input that should fail: {reason}")


def fixture(punches=(("jab", 300, 310, "A"), ("cross", 312, 325, "B"))):
    results = []
    for i, (label, start, end, boxer) in enumerate(
            [("frame-check", 100, 100, None), ("frame-check", 2000, 2000, None), *punches]):
        ranges = [{"start": start + 1, "end": end + 1}]
        results.append({"id": f"r{i}", "type": "timelinelabels", "from_name": "punches",
                        "to_name": "video", "value": {"timelinelabels": [label], "ranges": ranges}})
        if boxer is not None:
            results.append({"id": f"r{i}", "type": "choices", "from_name": "boxer",
                            "to_name": "video", "value": {"choices": [boxer], "ranges": copy.deepcopy(ranges)}})
    return [{"id": 7, "annotations": [{"result": results, "was_cancelled": False, "lead_time": 4.5}]}]


def rows(ex):
    return ex[0]["annotations"][0]["result"]


def parse(ex):
    return lsimport.parse_sparring_export(ex, 2100)


def test_boxers():
    ex = fixture()
    del rows(ex)[3]
    refuse(lambda: parse(ex), "missing boxer")
    ex = fixture()
    rows(ex)[3]["value"]["choices"] = []
    refuse(lambda: parse(ex), "missing boxer")
    for values in (["A", "B"], ["A", "A"]):
        ex = fixture()
        rows(ex)[3]["value"]["choices"] = values
        refuse(lambda: parse(ex), "two or more boxer")
    ex = fixture()
    rows(ex).append(copy.deepcopy(rows(ex)[3]))
    refuse(lambda: parse(ex), "two boxer results")
    for value in ("C", None, True, {}, [], "PRIVATE_CANARY"):
        ex = fixture()
        rows(ex)[3]["value"]["choices"] = [value]
        refuse(lambda: parse(ex), "unknown boxer")
    ex = fixture()
    row = copy.deepcopy(rows(ex)[3])
    row["id"] = "r0"
    row["value"]["ranges"] = rows(ex)[0]["value"]["ranges"]
    rows(ex).append(row)
    refuse(lambda: parse(ex), "frame-check region must not carry a boxer")


def test_join():
    ex = fixture()
    rows(ex).reverse()
    got = parse(ex).to_json("clip", 30, 2100)
    assert got["spans"] == [{"start": 300, "end": 310, "type": "jab", "boxer": "A"},
                            {"start": 312, "end": 325, "type": "cross", "boxer": "B"}]
    ex = fixture()
    rows(ex)[3]["id"] = "orphan"
    refuse(lambda: parse(ex), "orphan boxer")
    ex = fixture()
    rows(ex)[3]["value"]["ranges"][0]["start"] += 1
    refuse(lambda: parse(ex), "boxer ranges do not match")
    ex = fixture()
    rows(ex).append(copy.deepcopy(rows(ex)[2]))
    refuse(lambda: parse(ex), "duplicate timeline")
    ex = fixture()
    rows(ex)[2]["id"] = ""
    refuse(lambda: parse(ex), "nonempty string id")
    ex = fixture()
    rows(ex)[2]["value"]["boxer"] = "B"
    refuse(lambda: parse(ex), "separate per-region")


def test_overlap():
    refuse(lambda: parse(fixture((("jab", 300, 320, "A"), ("lead_hook", 320, 330, "A")))), "overlap")
    for boxer in ("B", "?"):
        refuse(lambda: parse(fixture((("cross", 300, 320, boxer), ("rear_hook", 310, 330, boxer)))), "overlap")
    assert len(parse(fixture((("jab", 300, 320, "A"), ("lead_hook", 300, 320, "B")))).spans) == 2
    assert len(parse(fixture((("jab", 300, 320, "A"), ("cross", 300, 320, "A")))).spans) == 2
    refuse(lambda: parse(fixture((("unclear", 300, 320, "A"), ("jab", 300, 320, "A")))), "duplicates")
    assert len(parse(fixture((("unclear", 300, 320, "A"), ("unclear", 300, 320, "B")))).spans) == 2
    assert parse(fixture((("unclear", 400, 400, "?"),))).boxers == ["?"]
    # The same punch marked once unsure and once decided is a phantom duplicate.
    refuse(lambda: parse(fixture((("unclear", 300, 320, "?"), ("jab", 300, 320, "A")))), "duplicates a punch")
    refuse(lambda: parse(fixture((("jab", 301, 321, "?"), ("jab", 300, 320, "B")))), "duplicates a punch")
    # A "?" punch that only touches a named one (IoU < 0.5) is a separate punch.
    assert len(parse(fixture((("jab", 300, 310, "?"), ("cross", 308, 330, "A")))).spans) == 2


def test_old_guards():
    for delta, reason in ((-1, "expected +1"), (2, "expected +1")):
        ex = fixture()
        for row in rows(ex):
            for r in row["value"]["ranges"]:
                r["start"] += delta
                r["end"] += delta
        refuse(lambda: parse(ex), reason)
    ex = fixture()
    rows(ex)[1]["value"]["ranges"][0] = {"start": 2002, "end": 2002}
    refuse(lambda: parse(ex), "disagree")
    ex = fixture()
    rows(ex)[0]["value"]["ranges"][0]["end"] += 1
    refuse(lambda: parse(ex), "single frame")
    ex = fixture()
    rows(ex).pop(0)
    refuse(lambda: parse(ex), "frame-check")
    refuse(lambda: parse(fixture((("jab", 2090, 2100, "A"),))), "outside the video")
    refuse(lambda: parse(fixture((("jab", -1, 10, "A"),))), "outside the video")
    refuse(lambda: parse(fixture((("jab", 20, 10, "A"),))), "before it starts")
    refuse(lambda: parse(fixture((("PRIVATE_CANARY", 1, 2, "A"),))), "unknown label")
    ex = fixture()
    ex[0]["annotations"].append({"result": [], "was_cancelled": False})
    refuse(lambda: parse(ex), "exactly one completed")
    ex = fixture()
    ex[0]["drafts"] = [{"result": []}]
    refuse(lambda: parse(ex), "press Submit")
    refuse(lambda: parse(fixture() + fixture()), "exactly one task")
    refuse(lambda: lsimport.parse_sparring_export(fixture(), 2000), "frame-check is outside")


def test_shapes():
    for value in (True, None, float("nan"), float("inf"), "PRIVATE_CANARY", [], {}, 1.5):
        ex = fixture()
        rows(ex)[2]["value"]["ranges"][0]["start"] = value
        refuse(lambda: parse(ex), "finite whole")
    for field, value, reason in (("type", "PRIVATE_CANARY", "unexpected result type"),
                                  ("from_name", "other", "control names"),
                                  ("to_name", "other", "control names"),
                                  ("value", [], "malformed region value")):
        ex = fixture()
        rows(ex)[2][field] = value
        refuse(lambda: parse(ex), reason)
    ex = fixture()
    rows(ex)[2]["value"]["ranges"].append({"start": 400, "end": 410})
    refuse(lambda: parse(ex), "exactly one range")
    ex = fixture()
    rows(ex)[2]["value"]["timelinelabels"] = ["jab", "cross"]
    refuse(lambda: parse(ex), "exactly one label")
    for value in (None, {}, "PRIVATE_CANARY", [None]):
        ex = fixture()
        ex[0]["annotations"] = value
        refuse(lambda: parse(ex), "annotations must be")
    ex = fixture()
    ex[0]["annotations"][0]["was_cancelled"] = "no"
    refuse(lambda: parse(ex), "cancellation flag")


def test_loader():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "export.json"
        for content, reason in (("[{\"x\":NaN}]", "not a number"),
                                ("[{\"x\":Infinity}]", "not a number"),
                                ("[{\"x\":1e999}]", "non-finite"),
                                ('{"x":1,"x":2}', "duplicate JSON keys"),
                                ("[" * 10000 + "]" * 10000, "nested too deeply"),
                                ('{"PRIVATE_CANARY":', "invalid JSON")):
            path.write_text(content, encoding="utf-8")
            refuse(lambda: lsimport.load_sparring_export(path), reason)
        path.write_bytes(b" " * (lsimport.MAX_EXPORT_BYTES + 1))
        refuse(lambda: lsimport.load_sparring_export(path), "size cap")
        raw = json.dumps(fixture()).encode()
        path.write_bytes(raw)
        got, digest = lsimport.load_sparring_export(path)
        assert got == fixture() and digest == hashlib.sha256(raw).hexdigest()


def test_paths():
    refuse(lambda: sparring_io.external_path(ROOT / "results" / "../private.mp4"), "inside the git repo")
    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder)
        repo = tmp / "other-repo"
        repo.mkdir()
        (repo / ".git").write_text("gitdir: somewhere", encoding="utf-8")
        refuse(lambda: sparring_io.external_path(repo / "clip.mp4"), "inside a git repo")
        # Junctions need no symlink privilege on Windows. Remove the link only.
        link = tmp / "linked"
        if os.name == "nt":
            script = tmp / "junction.ps1"
            script.write_text('param($LinkPath, $TargetPath)\n$ErrorActionPreference = "Stop"\n'
                              'New-Item -ItemType Junction -Path $LinkPath -Value $TargetPath | Out-Null\n',
                              encoding="utf-8")
            completed = subprocess.run(["powershell", "-NoProfile", "-File", str(script),
                str(link), str(ROOT / "tests")], capture_output=True)
            assert completed.returncode == 0, "could not create junction test fixture"
        else:
            link.symlink_to(ROOT / "tests", target_is_directory=True)
        try:
            refuse(lambda: sparring_io.external_path(link / "private.mp4"), "inside the git repo")
        finally:
            if os.name == "nt":
                os.rmdir(link)
            else:
                link.unlink()
        out = tmp / "labels.json"
        out.write_text("original", encoding="utf-8")
        def write(force=False):
            with sparring_io.staged_output(out, force) as stage:
                stage.write_text("new", encoding="utf-8")
        refuse(write, "--force")
        assert out.read_text() == "original"
        # The refusal must come BEFORE the body runs (no 10-minute encode wasted,
        # no stray temp file), not only at the final atomic link.
        ran = []
        def early():
            with sparring_io.staged_output(out) as stage:
                ran.append(stage)
        refuse(early, "--force")
        assert not ran, "existing output was only refused after the work was done"
        assert not list(tmp.glob(".sparring-*")), "temporary file left behind"
        write(True)
        assert out.read_text() == "new"
        out.unlink()
        def race():
            with sparring_io.staged_output(out) as stage:
                stage.write_text("loser", encoding="utf-8")
                out.write_text("winner", encoding="utf-8")
        refuse(race, "--force")
        assert out.read_text() == "winner"
        assert not list(tmp.glob(".sparring-*"))


def video(path, frames=2100, fps=30, vfr=False, audio=False, mild=False, drift=False):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=gray:s=320x96:r={fps}"]
    if drift:  # every step within one tick of 1/30 s and a mean of exactly 30 fps,
        # but the error accumulates to 12 ticks mid-clip (use frames=25)
        cmd += ["-vf", "settb=1/300000,setpts=N*10000+if(lt(N\\,12)\\,N\\,24-N)",
                "-enc_time_base", "1/300000", "-fps_mode", "passthrough"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-c:a", "aac", "-shortest"]
    if vfr:
        cmd += ["-vf", "select=not(eq(n\\,5))", "-fps_mode", "passthrough"]
    if mild:
        cmd += ["-vf", "settb=1/30000,setpts=PTS+100*eq(N\\,5)",
                "-enc_time_base", "1/30000", "-fps_mode", "passthrough"]
    cmd += ["-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)


def test_numbering():
    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder)
        source, out = tmp / "input.mp4", tmp / "numbered.mp4"
        video(source, frames=2010, audio=True)
        args = ["--video", str(source), "--output", str(out)]
        assert number_frames.main(args) == 0
        info = sparring_io.strict_cfr(out)
        assert info.timing.n_frames == 2010 and info.fps_frac == 30
        streams = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(out)],
                                            capture_output=True, check=True).stdout)["streams"]
        assert len(streams) == 1 and streams[0]["codec_name"] == "h264", "audio retained or codec changed"
        frames = list(iter_frames(out, None, 12))
        assert len(frames) == 12
        assert (frames[0][:60, :80] != frames[1][:60, :80]).any(), "frame number did not advance"
        assert (frames[0][:60, :80].max(axis=2) > 220).any(), "no white frame-number pixels"
        assert abs(float(frames[0][70:, 100:].mean()) - 128) < 5, "content outside overlay changed"
        # Read the digits back: frame i must show i (0-based), the convention the
        # importer's frame checks assume. Compare each frame to every candidate index.
        import numpy as np
        from boxingvi.overlay import draw_number
        def mask(k):
            blank = np.zeros((96, 320, 3), np.uint8)
            draw_number(blank, k)
            return blank[..., 0] > 0
        def yellow(frame):
            f = frame.astype(int)
            return (f[..., 0] > 180) & (f[..., 1] > 180) & (f[..., 2] < 110)
        for i, frame in enumerate(frames):
            scores = {k: (yellow(frame) == mask(k)).mean() for k in range(12)}
            assert max(scores, key=scores.get) == i, f"frame {i} reads as {max(scores, key=scores.get)}"
        # Frame 2000 is where the labeller places the second frame check.
        import itertools
        f2000 = next(itertools.islice(iter_frames(out), 2000, None))
        scores = {k: (yellow(f2000) == mask(k)).mean() for k in (1999, 2000, 2001, 200, 20000)}
        assert max(scores, key=scores.get) == 2000, "frame 2000 does not read 2000"
        short, slow = tmp / "short.mp4", tmp / "slow.mp4"
        video(short, frames=12)
        refuse(lambda: number_frames.main(["--video", str(short), "--output", str(tmp / "s.mp4")]), "frame check")
        video(slow, frames=2010, fps=25)
        refuse(lambda: number_frames.main(["--video", str(slow), "--output", str(tmp / "s.mp4")]), "30 fps")
        refuse(lambda: number_frames.main(args), "--force")
        assert number_frames.main(args + ["--force"]) == 0
        refuse(lambda: number_frames.main(["--video", str(source), "--output", str(source), "--force"]), "input video")
        refuse(lambda: number_frames.main(["--video", str(source), "--output", str(ROOT / "private.mp4")]), "inside the git repo")
        for mild in (False, True):
            bad = tmp / "vfr.mp4"
            video(bad, frames=12, vfr=not mild, mild=mild)
            refuse(lambda: number_frames.main(["--video", str(bad), "--output", str(tmp / "bad.mp4")]), "VFR")
            assert not (tmp / "bad.mp4").exists()
        drifting = tmp / "drift.mp4"
        video(drifting, frames=25, drift=True)
        refuse(lambda: number_frames.main(["--video", str(drifting), "--output", str(tmp / "d.mp4")]), "VFR")


def test_cli():
    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder)
        clip, export = tmp / "bout.one.numbered.mp4", tmp / "export.json"
        video(clip)
        export.write_text(json.dumps(fixture((("jab", 0, 10, "A"), ("jab", 0, 10, "B"),
                                              ("unclear", 2099, 2099, "?")))), encoding="utf-8")
        base = ["--sparring", "--video", str(clip), "--export", str(export)]
        refuse(lambda: import_labels.main(base), "--out-dir is required")
        refuse(lambda: import_labels.main(base + ["--out-dir", str(ROOT)]), "inside the git repo")
        refuse(lambda: import_labels.main(base + ["--out-dir", str(tmp / "elsewhere")]), "next to the video")
        args = base + ["--out-dir", str(tmp)]
        assert import_labels.main(args) == 0
        out = tmp / "bout.one.numbered.labels.json"
        got = json.loads(out.read_text(encoding="utf-8"))
        assert got["spans"] == [{"start": 0, "end": 10, "type": "jab", "boxer": "A"},
                                {"start": 0, "end": 10, "type": "jab", "boxer": "B"},
                                {"start": 2099, "end": 2099, "type": "unclear", "boxer": "?"}]
        assert got["export_sha256"] == hashlib.sha256(export.read_bytes()).hexdigest()
        assert got["n_frames"] == 2100 and got["label_studio_offset"] == 1 and got["fps"] == 30
        original = out.read_bytes()
        refuse(lambda: import_labels.main(args), "--force")
        assert out.read_bytes() == original
        assert import_labels.main(args + ["--force"]) == 0
        assert out.read_bytes() == original
        export.write_text(json.dumps(fixture((("jab", 10, 20, None),))), encoding="utf-8")
        refuse(lambda: import_labels.main(args + ["--force"]), "missing boxer")
        assert out.read_bytes() == original
        video(clip, fps=60)
        refuse(lambda: import_labels.main(args + ["--force"]), "frameRate is 30")
        video(clip, vfr=True)
        refuse(lambda: import_labels.main(args + ["--force"]), "VFR")


def test_config():
    solo = ET.parse(ROOT / "labelstudio/punch-labels.xml").getroot()
    sparring = ET.parse(ROOT / "labelstudio/sparring-punch-labels.xml").getroot()
    assert [e.attrib for e in solo.iter("Label")] == [e.attrib for e in sparring.iter("Label")]
    assert sparring.find("Video").get("frameRate") == "30"
    choice = sparring.find("Choices")
    assert choice.get("perRegion") == "true"
    # These hid the box in LS 1.23.2 (10-28); the importer enforces the boxer rule instead.
    assert all(choice.get(k) is None for k in ("required", "visibleWhen", "whenLabelValue"))
    assert [e.get("value") for e in choice] == ["A", "B", "?"]


TESTS = {fn.__name__: fn for fn in (test_boxers, test_join, test_overlap, test_old_guards,
         test_shapes, test_loader, test_paths, test_numbering, test_cli, test_config)}


def mutations():
    # Remove individual guards, then require a behavioural assertion to fail.
    # Exact-match anchors also refuse to silently mutate the wrong code after edits.
    cases = [
        ("src/boxingvi/lsimport.py", "if rid not in choices:", "if False:", "test_boxers"),
        ("src/boxingvi/lsimport.py", "if len(selection) != 1:", "if False:", "test_boxers"),
        ("src/boxingvi/lsimport.py", "if rid in choices:", "if False:", "test_boxers"),
        ("src/boxingvi/lsimport.py", "if rid in destination:", "if False:", "test_boxers"),
        ("src/boxingvi/lsimport.py", "if not isinstance(boxer, str) or boxer not in BOXERS:", "if False:", "test_boxers"),
        ("src/boxingvi/lsimport.py", "if choices.keys() - timelines.keys():", "if False:", "test_join"),
        ("src/boxingvi/lsimport.py", "if json.dumps(choices[rid].get(\"ranges\"), sort_keys=True) != json.dumps(ranges, sort_keys=True):", "if False:", "test_join"),
        ("src/boxingvi/lsimport.py", "if not isinstance(rid, str) or not rid.strip():", "if False:", "test_join"),
        ("src/boxingvi/lsimport.py", "if \"boxer\" in value or \"choices\" in value:", "if False:", "test_join"),
        ("src/boxingvi/lsimport.py", "if task.get(\"drafts\"):", "if False:", "test_old_guards"),
        ("src/boxingvi/lsimport.py", "if any(not 1 <= ranges[0][key] <= n_frames for key in (\"start\", \"end\")):", "if False:", "test_old_guards"),
        ("src/boxingvi/lsimport.py", "if key in result:", "if False:", "test_loader"),
        ("src/boxingvi/lsimport.py", "if not math.isfinite(value):", "if False:", "test_loader"),
        ("src/boxingvi/lsimport.py", "if len(raw) > MAX_EXPORT_BYTES:", "if False:", "test_loader"),
        ("src/boxingvi/lsimport.py", "if hand and hand in last_by_hand and s <= last_by_hand[hand][2]:", "if False:", "test_overlap"),
        ("src/boxingvi/lsimport.py", "if (label == pl and iou > 0) or iou >= DUPLICATE_IOU:", "if False:", "test_overlap"),
        ("src/boxingvi/lsimport.py", "if _iou(_span(unsure), _span(named)) >= DUPLICATE_IOU:", "if False:", "test_overlap"),
        ("scripts/number_frames.py", "if info.fps_frac != 30:", "if False:", "test_numbering"),
        ("scripts/number_frames.py", "if info.timing.n_frames <= max(FRAME_CHECKS):", "if False:", "test_numbering"),
        ("src/boxingvi/lsimport.py", "by_boxer[boxer].append(normalized)", "by_boxer['A'].append(normalized)", "test_join"),
        ("src/boxingvi/sparring_io.py", "if candidate.is_relative_to(REPO):", "if False:", "test_paths"),
        ("src/boxingvi/sparring_io.py", "if any((parent / \".git\").exists() for parent in (candidate, *candidate.parents)):", "if False:", "test_paths"),
        ("src/boxingvi/sparring_io.py", "resolved = raw.resolve()", "resolved = raw", "test_paths"),
        ("src/boxingvi/sparring_io.py", "if out.exists() and not force:", "if False:", "test_paths"),
        ("src/boxingvi/sparring_io.py", "os.link(temporary, out)", "os.replace(temporary, out)", "test_paths"),
        ("src/boxingvi/sparring_io.py", "if abs((timestamp - pts[0]) * den - i * num) > den:", "if False:", "test_numbering"),
        ("scripts/number_frames.py", "info = strict_cfr(source)", "from boxingvi.video import probe\n    info = probe(source)", "test_numbering"),
        ("scripts/number_frames.py", "if out == source or (out.exists() and source.exists() and out.samefile(source)):", "if False:", "test_numbering"),
        ("scripts/number_frames.py", "draw_number(frame, i)", "draw_number(frame, i + 1)", "test_numbering"),
        ("import_labels.py", "if args.out_dir is None:", "if False:", "test_cli"),
        ("import_labels.py", "if directory != video.parent:", "if False:", "test_cli"),
        ("import_labels.py", "if info.fps_frac != 30:", "if False:", "test_cli"),
    ]
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder) / "checkout"
        root.mkdir()
        for name in ("src", "scripts", "labelstudio", "tests"):
            shutil.copytree(ROOT / name, root / name, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT / "import_labels.py", root / "import_labels.py")
        for i, (file, old, new, test) in enumerate(cases, 1):
            path = root / file
            original = path.read_text(encoding="utf-8")
            assert original.count(old) == 1, (file, old, "mutation anchor drifted")
            path.write_text(original.replace(old, new), encoding="utf-8")
            try:
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
                result = subprocess.run([sys.executable, "-B", str(root / "tests/test_sparring_labels.py"), "--case", test],
                                        cwd=root, capture_output=True, env=env, timeout=90)
                assert result.returncode != 0, f"SURVIVED {file}: {old}"
                assert b"SyntaxError" not in result.stderr, "invalid mutant"
                print(f"KILLED {i}: {file}: {old}")
            finally:
                path.write_text(original, encoding="utf-8")
    print(f"{len(cases)} mutations killed")


if __name__ == "__main__":
    if "--mutations" in sys.argv:
        mutations()
    else:
        selected = [sys.argv[sys.argv.index("--case") + 1]] if "--case" in sys.argv else TESTS
        for name in selected:
            TESTS[name]()
            print(f"PASS {name}")
        print(f"{len(selected)} test groups passed")
