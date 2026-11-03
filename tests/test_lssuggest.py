"""Tests for Label Studio sparring suggestions (plain script, like the others)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boxingvi import lsimport, lssuggest  # noqa: E402


def refuse(fn, reason):
    try:
        fn()
    except ValueError as e:
        assert reason in str(e), (reason, str(e))
        return
    raise AssertionError(f"expected refusal: {reason}")


def check(region_id, frame):
    return {"id": region_id, "type": "timelinelabels", "from_name": "punches", "to_name": "video",
            "value": {"ranges": [{"start": frame, "end": frame}], "timelinelabels": ["frame-check"]}}


def as_export(regions, boxer_for=None, retype=None):
    """What Label Studio would export after Brendan submits the pre-filled task."""
    result = [check("c1", 101), check("c2", 2001)]
    for r in copy.deepcopy(regions):
        r.pop("origin")
        if retype:
            r["value"]["timelinelabels"] = [retype]
        result.append(r)
        if boxer_for:
            result.append({"id": r["id"], "type": "choices", "from_name": "boxer", "to_name": "video",
                           "value": {"ranges": r["value"]["ranges"], "choices": [boxer_for]}})
    return [{"annotations": [{"result": result, "was_cancelled": False}], "drafts": []}]


def test_results_shape():
    regs = lssuggest.results([299, 400], 3000)
    assert [r["value"]["ranges"][0]["start"] for r in regs] == [300, 401]   # LS frames are 1-based
    assert all(r["value"]["timelinelabels"] == ["unclear"] for r in regs)
    assert all("boxer" not in r["value"] and "choices" not in r["value"] for r in regs)
    assert len({r["id"] for r in regs}) == 2
    print("PASS results_shape")


def test_bad_frames():
    refuse(lambda: lssuggest.results([3000], 3000), "outside the video")
    refuse(lambda: lssuggest.results([-1], 3000), "outside the video")
    refuse(lambda: lssuggest.results([1.5], 3000), "outside the video")
    refuse(lambda: lssuggest.results([True], 3000), "outside the video")
    print("PASS bad_frames")


def test_url():
    ok = lssuggest.task("/data/upload/1/abc-gg1022-v2.numbered.mp4", [])
    assert ok[0]["predictions"][0]["result"] == [] and ok[0]["data"]["video"].startswith("/data/upload/")
    for bad in ("javascript:alert(1)", "file:///C:/x.mp4", "/etc/passwd", "", "/data/upload/a b.mp4",
                'https://x/"><script>', None):
        refuse(lambda b=bad: lssuggest.task(b, []), "video URL")
    print("PASS url")


def test_no_boxer_in_suggestion():
    regs = lssuggest.results([500], 3000)
    regs[0]["value"]["boxer"] = "A"
    refuse(lambda: lssuggest.task("/data/upload/1/v.mp4", regs), "must not carry a boxer")
    print("PASS no_boxer_in_suggestion")


def test_round_trip():
    regs = lssuggest.results([299, 900], 3000)
    # Kept unchanged without a boxer: the importer refuses, so a suggestion is never silently accepted.
    refuse(lambda: lsimport.parse_sparring_export(as_export(regs), 3000), "missing boxer")
    # Brendan picks a boxer and a type: accepted, at the right video frames.
    got = lsimport.parse_sparring_export(as_export(regs, "A", "jab"), 3000)
    assert [(s.start, s.end) for s in got.spans] == [(299, 299), (900, 900)], got.spans
    assert got.boxers == ["A", "A"]
    print("PASS round_trip")


def test_two_largest():
    # Frame 0: three people (areas 100, 400, 25); frame 1: one person.
    box = np.array([[0, 0, 10, 10], [0, 0, 20, 20], [0, 0, 5, 5], [0, 0, 1, 1]], dtype=float)
    t = SimpleNamespace(frame=np.array([0, 0, 0, 1]), box=box)
    assert lssuggest.two_largest_names(t) == ["unsure", "unsure", None, "unsure"]
    t.box[0, 2] = np.nan
    refuse(lambda: lssuggest.two_largest_names(t), "non-finite box")
    print("PASS two_largest")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all lssuggest tests passed")
