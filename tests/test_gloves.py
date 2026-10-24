"""Tests for glove-colour identity (gloves.py, glovescore.py) and glove_identity.py.

Synthetic only: every frame is painted in memory (two boxers in black, coloured gloves, skin faces, a grey
wall, pixel noise). No study footage and no model. The only pixels written to disk are the tiny painted clip
the CLI test builds, inside a temp folder. Most checks are negative: counts, labels, manifest and score files
are inputs, every output location is checked, and the rule must be able to say "not separable".

Run:  python tests/test_gloves.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import zipfile
import io
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import glove_identity  # noqa: E402
from boxingvi import faceblur, gloves, glovescore, video  # noqa: E402
from boxingvi.data import N_JOINTS  # noqa: E402
from boxingvi.gloves import GloveCounts  # noqa: E402
from boxingvi.glovescore import Labels, Moment  # noqa: E402
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


def raises(name: str, fn, pattern: str = "", exc=ValueError) -> None:
    try:
        fn()
    except exc as e:
        check(name, pattern in str(e), f"wrong reason: {e}")
    except Exception as e:
        check(name, False, f"raised {type(e).__name__}: {e}")
    else:
        check(name, False, "did not raise")


tmp = Path(tempfile.mkdtemp())
W, H, FPS = 320, 240, 30.0
BG, BLACK, SKIN = (110, 110, 110), (20, 20, 20), (225, 170, 140)
RED, BLUE, WHITE = (220, 30, 30), (30, 60, 230), (245, 245, 245)
FP = "ab" * 32


def person_xy(cx, cy):
    """COCO-17: shoulders at cy - 30, elbows 30 px out, wrists 50 px out (forearm 20 px, horizontal)."""
    xy = np.tile([cx, cy], (N_JOINTS, 1)).astype(np.float64)
    xy[0] = (cx, cy - 50)
    xy[1], xy[2], xy[3], xy[4] = (cx - 4, cy - 53), (cx + 4, cy - 53), (cx - 8, cy - 50), (cx + 8, cy - 50)
    xy[5], xy[6] = (cx - 20, cy - 30), (cx + 20, cy - 30)
    xy[7], xy[8] = (cx - 30, cy - 10), (cx + 30, cy - 10)
    xy[9], xy[10] = (cx - 50, cy - 10), (cx + 50, cy - 10)
    xy[11], xy[12] = (cx - 15, cy + 30), (cx + 15, cy + 30)
    xy[13], xy[14], xy[15], xy[16] = (cx - 15, cy + 60), (cx + 15, cy + 60), (cx - 15, cy + 90), (cx + 15, cy + 90)
    return xy


def disc(img, cx, cy, r, col):
    Y, X = np.ogrid[:H, :W]
    img[(X - cx) ** 2 + (Y - cy) ** 2 <= r * r] = col


def paint(img, cx, cy, gl, gr):
    x0, x1 = int(cx - 20), int(cx + 20)
    img[max(0, int(cy - 30)):int(cy + 30), max(0, x0):max(0, x1)] = BLACK
    img[int(cy - 14):int(cy - 6), max(0, int(cx - 50)):max(0, int(cx + 50))] = BLACK
    disc(img, cx, cy - 50, 10, SKIN)
    disc(img, cx - 55, cy - 10, 8, gl)     # glove centre = wrist + 0.25 x (wrist - elbow)
    disc(img, cx + 55, cy - 10, 8, gr)


def scene(n=900, ga=(RED, RED), gb=(BLUE, BLUE), swap_at=450, wrist_conf=0.9, light=None, gap=0.0,
          clinch=None, seed=1, hop=None, hop_drop=False, hop_hidden=False, b_reads_as=None):
    """A starts left and walks right, B the reverse; they cross at the middle. ByteTrack swaps their ids at
    `swap_at` (A = id 1 before, id 2 after). `light(f)` scales a frame; `clinch` = frames they stand together.
    Returns (table, frames(), truth) with truth[row] = "A" / "B"."""
    rows = {"frame": [], "tid": [], "box": [], "xy": [], "conf": []}
    truth, pos = [], {}
    for f in range(n):
        u = f / (n - 1)
        ax, bx = 90 + 140 * u, 230 - 140 * u
        if clinch is not None and f in clinch:
            ax, bx = 110, 210 + gap      # A's right wrist and B's left wrist at the same point: gloves together
        pos[f] = (ax, bx)
        for who, cx in (("A", ax), ("B", bx)):
            tid = (1 if who == "A" else 2) if (swap_at is None or f < swap_at) else (2 if who == "A" else 1)
            c = np.full(N_JOINTS, 0.9)
            c[[9, 10]] = wrist_conf
            if hop is not None and hop[0] <= f < hop[1]:
                if hop_drop != "B":                          # ByteTrack lends A's id to B for a moment
                    tid = 5 if who == "A" else 1
                else:                                        # ... or B's id to A
                    tid = 2 if who == "A" else 6
                if hop_drop == who:                          # ... while the lender is not detected at all
                    continue
                if hop_hidden and hop_drop and who != hop_drop:   # ... and the borrower's gloves are hidden
                    c[[9, 10]] = 0.2
            rows["frame"].append(f)
            rows["tid"].append(tid)
            rows["box"].append((cx - 55, 58, cx + 55, 210))
            rows["xy"].append(person_xy(cx, 120))
            rows["conf"].append(c)
            truth.append(who)
    t = PoseTable(frame=np.array(rows["frame"], dtype=np.int64), track_id=np.array(rows["tid"], dtype=np.int64),
                  box=np.array(rows["box"], dtype=np.float64), xy=np.array(rows["xy"]),
                  conf=np.array(rows["conf"]), n_frames=n, fps=FPS, width=W, height=H, source="synthetic",
                  model="painted", video_fp=FP)

    def frames():
        rng = np.random.default_rng(seed)
        for f in range(n):
            img = np.zeros((H, W, 3), dtype=np.float64)
            img[:] = BG
            ax, bx = pos[f]
            paint(img, ax, 120, *ga)
            if b_reads_as is not None and b_reads_as[0] <= f < b_reads_as[1]:
                paint(img, bx, 120, *ga)          # B's gloves read as A's colour for a stretch (a colour cast)
            else:
                paint(img, bx, 120, *gb)
            img += rng.normal(0, 6, img.shape)
            if light is not None:
                img *= light(f)
            yield np.clip(img, 0, 255).astype(np.uint8)
    return t, frames, np.array(truth)


def named_agreement(ident, truth):
    """(named share, wrong share of named) after the best cluster->boxer mapping (an upper bound check)."""
    lab = ident.label
    named = lab >= 0
    if not named.any():
        return 0.0, 0.0
    a = truth[named] == "A"
    ok = max(((lab[named] == 0) == a).mean(), ((lab[named] == 1) == a).mean())
    return float(named.mean()), float(1 - ok)


# ---- pixels and geometry ----------------------------------------------------------------
print("pixel classes and glove discs")
px = np.array([BLACK, SKIN, WHITE, (128, 128, 128), RED, BLUE], dtype=np.uint8)
cls = [int(np.argmax(gloves.pixel_classes(p[None]))) for p in px]
check("black -> dark", gloves.CLASS_NAMES[cls[0]] == "dark", gloves.CLASS_NAMES[cls[0]])
check("skin -> skin", gloves.CLASS_NAMES[cls[1]] == "skin", gloves.CLASS_NAMES[cls[1]])
check("white -> white", gloves.CLASS_NAMES[cls[2]] == "white")
check("grey -> grey", gloves.CLASS_NAMES[cls[3]] == "grey")
check("red -> hue000", gloves.CLASS_NAMES[cls[4]] == "hue000", gloves.CLASS_NAMES[cls[4]])
check("blue -> a blue hue bin", gloves.CLASS_NAMES[cls[5]] in ("hue210", "hue240"), gloves.CLASS_NAMES[cls[5]])
rnd = np.random.default_rng(0).integers(0, 256, (5000, 3)).astype(np.uint8)
check("every pixel lands in exactly one class", int(gloves.pixel_classes(rnd).sum()) == 5000)
check("hue 355 (red wrapping round) is bin 0", int(np.argmax(gloves.pixel_classes(np.array([[230, 20, 40]], np.uint8)))) == 0)

xy = person_xy(100, 120)
cf = np.full(N_JOINTS, 0.9)
d = gloves.glove_disc(xy, cf, 1, W, H)
check("disc sits beyond the wrist, radius 0.3 x forearm", d is not None and abs(d[0] - 155) < 1e-9 and abs(d[2] - 6) < 1e-9, str(d))
c2 = cf.copy(); c2[10] = 0.45
check("wrist under 0.5 confidence is not read", gloves.glove_disc(xy, c2, 1, W, H) is None)
c3 = cf.copy(); c3[8] = 0.1
check("no confident elbow, no direction, not read", gloves.glove_disc(xy, c3, 1, W, H) is None)
x2 = xy.copy(); x2[10] = x2[8] + (3, 0)
check("a forearm under 6 px is not read", gloves.glove_disc(x2, cf, 1, W, H) is None)
x3 = xy.copy(); x3[10] = x3[8] + (400, 0)
check("an implausibly long forearm is not read", gloves.glove_disc(x3, cf, 1, W, H) is None)
x4 = xy.copy(); x4[10] = (np.nan, np.nan)
check("a NaN wrist is not read", gloves.glove_disc(x4, cf, 1, W, H) is None)
far = np.zeros((H, W, 3), np.uint8)
check("a disc outside the frame is not read", gloves.disc_classes(far, (-50.0, -50.0, 6.0)) is None)
check("sample every 3 frames at 30 fps, 5 at 50", gloves.sample_every(30.0) == 3 and gloves.sample_every(50.0) == 5)
raises("sample_every refuses fps 0", lambda: gloves.sample_every(0.0), "fps")
raises("sample_every refuses NaN fps", lambda: gloves.sample_every(float("nan")), "fps")

# ---- counts --------------------------------------------------------------------------------
print("counts of one video")
table, frames, truth = scene()
cnt = gloves.compute_gloves(None, table, frames=frames())
check("reads on every 3rd frame, both hands of both boxers (most frames)", len(cnt.rows) > 0.8 * 300 * 4, str(len(cnt.rows)))
check("each read's classes add up to its pixels", bool((cnt.bins.sum(axis=1) == cnt.n).all()))
check("(row, hand) strictly increasing", bool((np.diff(cnt.rows * 2 + cnt.hand) > 0).all()))
check("only sampled frames are read", bool((table.frame[cnt.rows] % 3 == 0).all()))
check("some reads near the crossing are contested", 0 < int(cnt.contested.sum()) < 0.2 * len(cnt.rows), str(int(cnt.contested.sum())))
short = (fr for i, fr in enumerate(frames()) if i < 100)
raises("a video shorter than the table is refused", lambda: gloves.compute_gloves(None, table, frames=short), "ended after")
wrong = (np.zeros((H + 2, W, 3), np.uint8) for _ in range(900))
raises("a frame of another size is refused", lambda: gloves.compute_gloves(None, table, frames=wrong), "the table is")

p = tmp / "ok.gloves.npz"
cnt.save(p)
back = GloveCounts.load(p)
check("save -> load round trip", np.array_equal(back.bins, cnt.bins) and back.table_sha256 == cnt.table_sha256
      and back.video_fp == FP and back.every == 3)
back.check_table(table)
check("check_table accepts its own table", True)


def variant(name, **over):
    base = {"format": np.array(gloves.FORMAT), "source": np.array(cnt.source), "video_fp": np.array(cnt.video_fp),
            "table_sha256": np.array(cnt.table_sha256), "provenance": np.array(cnt.provenance),
            "meta": np.array([cnt.n_frames, cnt.n_table_rows, cnt.every], dtype=np.int64),
            "rows": cnt.rows, "hand": cnt.hand, "bins": cnt.bins, "n": cnt.n, "contested": cnt.contested}
    base.update(over)
    base = {k: v for k, v in base.items() if v is not None}
    q = tmp / name
    np.savez(q, **base)
    return q


bad_bins = cnt.bins.copy(); bad_bins[0, 0] += 1
neg_bins = cnt.bins.copy(); neg_bins[0, 0] = -1; neg_bins[0, 1] += 1
unsorted = cnt.rows.copy(); unsorted[[0, 2]] = unsorted[[2, 0]]
for nm, over, why in [
        ("wrong format", {"format": np.array("boxingvi-headgear-2")}, "unknown format"),
        ("missing array", {"contested": None}, "wrong arrays"),
        ("extra array", {"extra": np.zeros(1, np.int64)}, "wrong arrays"),
        ("float bins", {"bins": cnt.bins.astype(np.float64)}, "dtype"),
        ("bins not adding up", {"bins": bad_bins}, "add up"),
        ("negative bin", {"bins": neg_bins}, "impossible"),
        ("hand 2", {"hand": np.where(cnt.hand == 0, 2, 1).astype(np.int64)}, "0 or 1"),
        ("contested 5", {"contested": np.full_like(cnt.contested, 5)}, "0 or 1"),
        ("unsorted rows", {"rows": unsorted}, "strictly increasing"),
        ("row beyond the table", {"meta": np.array([cnt.n_frames, 5, cnt.every], np.int64)}, "row indices"),
        ("bad digest text", {"table_sha256": np.array("xyz")}, "sha256"),
        ("bad fingerprint", {"video_fp": np.array("Z" * 64)}, "video_fp"),
        ("meta every 0", {"meta": np.array([cnt.n_frames, cnt.n_table_rows, 0], np.int64)}, "meta"),
        ("2-D rows", {"rows": cnt.rows.reshape(-1, 1)}, "1-D"),
        ("bins wrong width", {"bins": cnt.bins[:, :15]}, "shape"),
        ("tiny disc", {"n": np.full_like(cnt.n, 3)}, "impossible")]:
    raises(f"counts refused: {nm}", lambda o=over, nm=nm: GloveCounts.load(variant(nm.replace(" ", "_") + ".npz", **o)), why)
obj = tmp / "obj.npz"
np.savez(obj, **{k: np.array([None], dtype=object) for k in gloves._KEYS})
raises("counts refused: object arrays (pickles)", lambda: GloveCounts.load(obj), "not allowed")
nz = tmp / "notzip.npz"; nz.write_bytes(b"not a zip")
raises("counts refused: not a zip", lambda: GloveCounts.load(nz), "not a readable npz")
raises("counts refused: missing file", lambda: GloveCounts.load(tmp / "nope.npz"), "cannot be read")
lie = tmp / "lie.npz"
with zipfile.ZipFile(lie, "w") as zf:
    for k in gloves._KEYS:
        b = io.BytesIO()
        if k == "bins":
            np.lib.format.write_array_header_1_0(b, {"descr": "<i8", "fortran_order": False, "shape": (10 ** 9, 16)})
            b.write(b"\0" * 64)
        else:
            np.lib.format.write_array(b, np.zeros(1, np.int64))
        zf.writestr(k + ".npy", b.getvalue())
raises("counts refused: a header declaring 128 GB before any allocation", lambda: GloveCounts.load(lie), "disagree")
t2 = PoseTable(**{**table.__dict__, "conf": np.where(table.conf > 0.5, 0.91, table.conf)})
raises("check_table: same shape, different content", lambda: back.check_table(t2), "different content")
raises("check_table: other source", lambda: back.check_table(PoseTable(**{**table.__dict__, "source": "x"})), "sources")
raises("check_table: other fingerprint", lambda: back.check_table(PoseTable(**{**table.__dict__, "video_fp": "cd" * 32})), "fingerprints")
raises("check_table: other frame count", lambda: back.check_table(PoseTable(**{**table.__dict__, "n_frames": 901})), "frames")
raises("check_table: same rows, other fps (sampling and smoothing depend on it)",
       lambda: back.check_table(PoseTable(**{**table.__dict__, "fps": 29.97})), "different content")
check("provenance survives save -> load", GloveCounts.load(
    (lambda q: (gloves.GloveCounts(**{**cnt.__dict__, "provenance": "cd" * 32}).save(q), q)[1])(tmp / "prov.npz")
).provenance == "cd" * 32)
raises("counts refused: provenance not a digest", lambda: GloveCounts.load(variant("prov_bad.npz", provenance=np.array("x"))), "provenance")

# ---- the rule --------------------------------------------------------------------------------
print("identify: when it should and should not speak")
ident = gloves.identify(table, cnt)
share, wrong_named = named_agreement(ident, truth)
check("red vs blue gloves: separable", ident.verdict == "separable", ident.verdict)
check("red vs blue: >= 85% of rows named", share >= 0.85, f"{share:.3f}")
check("red vs blue: no named row wrong, through the id swap", wrong_named == 0.0, f"{wrong_named:.3f}")
check("red vs blue: co-present boxers fall in different clusters", ident.diagnostics["copresent_split"] >= 0.95,
      str(ident.diagnostics["copresent_split"]))
fr_lab = {}
ok_frames = True
for r in range(len(table.frame)):
    if ident.label[r] >= 0:
        f = int(table.frame[r])
        if (f, ident.label[r]) in fr_lab:
            ok_frames = False
        fr_lab[(f, ident.label[r])] = r
check("never the same name twice in one frame", ok_frames)

for nm, kw in [("same-colour gloves (FAIL-CASE)", {"ga": (RED, RED), "gb": (RED, RED)}),
               ("each boxer one red one blue glove", {"ga": (RED, BLUE), "gb": (RED, BLUE)}),
               ("same colour, light drops to 20% halfway", {"ga": (BLUE, BLUE), "gb": (BLUE, BLUE),
                                                          "light": lambda f: 1.0 if f < 450 else 0.2})]:
    tb, fs, tr = scene(**kw)
    idn = gloves.identify(tb, gloves.compute_gloves(None, tb, frames=fs()))
    sh, wr = named_agreement(idn, tr)
    check(f"{nm}: not separable, nobody named", idn.verdict == "not_separable" and sh == 0.0,
          f"{idn.verdict}, named {sh:.2f}, diag sep {idn.diagnostics.get('separation')}, "
          f"split {idn.diagnostics.get('copresent_split')}")

tb, fs, tr = scene(ga=(BLACK, BLACK), gb=(RED, RED))
idn = gloves.identify(tb, gloves.compute_gloves(None, tb, frames=fs()))
sh, wr = named_agreement(idn, tr)
check("black vs red gloves (black is a colour too): separable, none wrong",
      idn.verdict == "separable" and wr == 0.0 and sh > 0.8, f"{idn.verdict} {sh:.2f} {wr:.2f}")
tb, fs, tr = scene(ga=(WHITE, WHITE), gb=(BLACK, BLACK))
idn = gloves.identify(tb, gloves.compute_gloves(None, tb, frames=fs()))
sh, wr = named_agreement(idn, tr)
check("white vs black gloves: separable, none wrong", idn.verdict == "separable" and wr == 0.0, f"{idn.verdict} {wr:.2f}")
tb, fs, tr = scene(light=lambda f: 1.0 if f < 450 else 0.2)
idn = gloves.identify(tb, gloves.compute_gloves(None, tb, frames=fs()))
sh, wr = named_agreement(idn, tr)
check("red vs blue, light drops to 20% halfway: never confidently wrong (<= 10% of named)", wr <= 0.10,
      f"{idn.verdict} named {sh:.2f} wrong {wr:.2f}")
tb, fs, tr = scene(wrist_conf=0.4)
idn = gloves.identify(tb, gloves.compute_gloves(None, tb, frames=fs()))
check("wrists under 0.5 confidence everywhere: too few rows, nobody named",
      idn.verdict == "too_few_rows" and not (idn.label >= 0).any(), idn.verdict)
tb, fs, tr = scene(clinch=set(range(300, 600)))
cn = gloves.compute_gloves(None, tb, frames=fs())
clinch_rows = np.isin(tb.frame[cn.rows], np.arange(300, 600))
who = tr[cn.rows]
inner = ((who == "A") & (cn.hand == 1)) | ((who == "B") & (cn.hand == 0))
check("clinch: the two touching gloves are contested", bool(cn.contested[clinch_rows & inner].all())
      and int((clinch_rows & inner).sum()) > 100, str(int((clinch_rows & inner).sum())))
check("clinch: the outer gloves are still read", not bool(cn.contested[clinch_rows & ~inner].any()))
idn = gloves.identify(tb, cn)
sh, wr = named_agreement(idn, tr)
check("clinch: still no named row wrong", wr == 0.0, f"{idn.verdict} {wr:.2f}")

tb, fs, tr = scene(swap_at=None, hop=(300, 330))
idn = gloves.identify(tb, gloves.compute_gloves(None, tb, frames=fs()))
sh, wr = named_agreement(idn, tr)
check("a track id lent to the other boxer for 1 s: no named row wrong",
      idn.verdict == "separable" and wr == 0.0, f"{idn.verdict} wrong {wr:.4f}")
for lender in ("A", "B"):
    tb, fs, tr = scene(swap_at=None, hop=(300, 312), hop_drop=lender)
    idn = gloves.identify(tb, gloves.compute_gloves(None, tb, frames=fs()))
    sh, wr = named_agreement(idn, tr)
    check(f"an id lent for 0.4 s ({lender} undetected): the window says the old boxer, the row's own glove "
          "says the other -> unsure, none wrong", idn.verdict == "separable" and wr == 0.0, f"wrong {wr:.4f}")
    tb, fs, tr = scene(swap_at=None, hop=(300, 330), hop_drop=lender)
    idn = gloves.identify(tb, gloves.compute_gloves(None, tb, frames=fs()))
    sh, wr = named_agreement(idn, tr)
    check(f"... {lender}'s id lent while {lender} goes undetected (no row to clash with): none wrong "
          "(the 75% majority, both directions)", idn.verdict == "separable" and wr == 0.0,
          f"{idn.verdict} wrong {wr:.4f}")

tb, fs, tr = scene(swap_at=None, hop=(300, 330), hop_drop="A", hop_hidden=True)
cnh = gloves.compute_gloves(None, tb, frames=fs())
in_hop = (tb.frame[cnh.rows] >= 300) & (tb.frame[cnh.rows] < 330)
check("fixture: during the lend the borrower's gloves give no read at all, and A has no row",
      not in_hop.any() and not ((tb.frame >= 300) & (tb.frame < 330) & (tr == "A")).any(), str(int(in_hop.sum())))
idn = gloves.identify(tb, cnh)
lab_, fr_ = idn.label, tb.frame
named_ = lab_ >= 0
nm0 = (tr[named_ & (fr_ < 250)] == "A")
c_a = int(np.bincount(lab_[named_ & (fr_ < 250)][nm0]).argmax())
wrong_frames = fr_[named_ & (((lab_ == c_a) & (tr == "B")) | ((lab_ != c_a) & (tr == "A")))]
edge = int(round(gloves.SUPPORT_S * FPS)) + gloves.sample_every(FPS)
check("an id lent while the borrower's gloves are hidden: wrong names only within 0.3 s (+1 sample) of glove "
      "evidence, never in the middle of the lend", all((300 <= f < 300 + edge) or (330 - edge <= f < 330)
                                                      for f in wrong_frames.tolist()),
      f"wrong at frames {sorted(set(wrong_frames.tolist()))[:20]}")
mid = (fr_ >= 300 + edge) & (fr_ < 330 - edge)
check("... and in the middle of the lend nobody is named at all (no glove evidence, no guess)",
      mid.any() and not named_[mid].any(), f"{int(named_[mid].sum())} named of {int(mid.sum())}")

print("one name per frame is judged before the evidence filters")
tiny = PoseTable(frame=np.array([0, 0, 1, 1], np.int64), track_id=np.array([1, 2, 1, 2], np.int64),
                 box=np.tile([0.0, 0.0, 10.0, 10.0], (4, 1)), xy=np.zeros((4, N_JOINTS, 2)),
                 conf=np.full((4, N_JOINTS), 0.9), n_frames=2, fps=FPS, width=W, height=H, source="t", video_fp=FP)
res = gloves.resolve_labels(tiny, np.array([0, 0, 0, 1]), np.array([False, True, True, True]))
check("frame 0: both smoothed to the same name; a filter removed one; the other is NOT left named",
      res[0] == -1 and res[1] == -1, str(res))
check("frame 1: two different names, both kept", res[2] == 0 and res[3] == 1, str(res))
res = gloves.resolve_labels(tiny, np.array([-1, 0, 1, -1]), np.array([True, False, True, True]))
check("a filter alone makes a row unsure; an unsure row never causes a conflict", list(res) == [-1, -1, 1, -1], str(res))

tb, fs, tr = scene(swap_at=None, b_reads_as=(150, 300))
idn = gloves.identify(tb, gloves.compute_gloves(None, tb, frames=fs()))
sh, wr = named_agreement(idn, tr)
check("B's gloves read as A's colour for 5 s: both would be named A, so neither is (one name per frame)",
      idn.verdict == "separable" and wr == 0.0, f"{idn.verdict} wrong {wr:.4f} split {idn.diagnostics.get('copresent_split')}")

# two glove colours that are barely different: a clean split, but too fragile to name anybody
tb, _, tr = scene(n=300, swap_at=None)
pick = [r for r in range(len(tb.frame)) if tb.frame[r] % 3 == 0]
bins = np.zeros((2 * len(pick), gloves.N_CLASSES), np.int64)
for i, r in enumerate(pick):
    for hnd in (0, 1):
        bins[2 * i + hnd, 0] = 100
        bins[2 * i + hnd, gloves.HUE_BINS + 1] = 2 if tr[r] == "A" else 6
hc = GloveCounts(np.repeat(np.array(pick, np.int64), 2), np.tile(np.array([0, 1], np.int64), len(pick)), bins,
                 bins.sum(axis=1), np.zeros(2 * len(pick), np.int64), tb.source, tb.n_frames, len(tb.frame),
                 gloves.sample_every(FPS), gloves.table_digest(tb, gloves.sample_every(FPS)), tb.video_fp)
idn = gloves.identify(tb, hc)
check("near-identical colours: split perfectly but not separable (separation under 0.30)",
      idn.verdict == "not_separable" and idn.diagnostics["copresent_split"] == 1.0
      and idn.diagnostics["separation"] < gloves.SEP_MIN, str(idn.diagnostics))

# ---- scoring -------------------------------------------------------------------------------
print("scoring, exactly as the gate says")
lab = Labels("the boxer in red gloves", 3.0, tuple((t, "L" if t < 15 else "R") for t in range(0, 30, 5)))
ms = glovescore.moments(table, lab)
check("moments: 0-10 s scored, 15 s overlapping (the crossing), 20-25 s scored",
      [m.status for m in ms] == ["scored", "scored", "scored", "overlapping", "scored", "scored"],
      str([m.status for m in ms]))
check("moments: truth comes from the label, left/right from the boxes",
      all(m.truth_left == ("A" if m.t < 15 else "B") for m in ms if m.status == "scored"))
g = glovescore.score_method(ms, glovescore.glove_names(ident))
check("glove: anchor 0 s, 4 scored, 4 correct, 0 swaps", (g["anchor_t"], g["scored"], g["decided"], g["correct"], g["swaps"])
      == (0, 4, 4, 4, 0), str(g))
b1 = glovescore.score_method(ms, glovescore.bytetrack_names(table))
check("ByteTrack: the id swap makes 20 s and 25 s wrong, 1 swap", (b1["correct"], b1["wrong"], b1["swaps"]) == (2, 2, 1), str(b1))
b2 = glovescore.score_method(ms, glovescore.linker_names(table))
check("linker baseline runs and decides the same moments as ByteTrack here", b2["decided"] == 4 and b2["correct"] == 2, str(b2))
check("swaps per round = swaps / (span / 180)", abs(b1["swaps_per_round"] - 1 / (20 / 180)) < 1e-12, str(b1["swaps_per_round"]))

mk = lambda t, tl="A": Moment(t, t, "scored", 0, 1, tl)  # noqa: E731
seq = [mk(0), mk(5), mk(10), mk(15), mk(20), mk(25), Moment(30, 30, "unclear")]


class Scripted:
    """name(row) returns the scripted pair for the moment being scored (score_method calls left then right)."""

    def __init__(self, pairs):
        self.pairs, self.i, self.half = list(pairs), 0, 0

    def __call__(self, row):
        v = self.pairs[self.i][row]
        if row == 1:
            self.i += 1
        return v


s = glovescore.score_method(seq, Scripted([(None, 1), (7, 8), (7, 8), (8, 7), (9, 8), (7, 7)]))
check("an undecided first moment is not the anchor; the next decided one is", s["anchor_t"] == 5, str(s))
check("after the anchor: 4 scored; unknown name 9 and the same name twice are unsure",
      (s["scored"], s["decided"], s["correct"], s["wrong"]) == (4, 2, 1, 1), str(s))
check("unsure share and correct share", abs(s["unsure_share"] - 0.5) < 1e-12 and abs(s["correct_share"] - 0.25) < 1e-12)
check("one swap (correct then wrong)", s["swaps"] == 1)
none = glovescore.score_method(seq, Scripted([(None, None)] * 6))
check("a method that never decides: no anchor, nothing decided, accuracy None",
      none["anchor_t"] is None and none["decided"] == 0 and none["accuracy"] is None and none["wilson95"] is None)
check("... and every scored moment counts as unsure (6 of 6)", none["scored"] == 6 and none["unsure_share"] == 1.0)

def G(scored, decided, correct, spr=0.5):
    return {"scored": scored, "decided": decided, "correct": correct, "wrong": decided - correct,
            "swaps_per_round": spr}


good = G(50, 40, 38)
base = {"correct_share": 0.5}
fc_ok = G(40, 3, 0, None)
T = lambda gl, b=base: {"methods": {"glove": gl, "bytetrack": b, "linker": {"correct_share": 0.3}}}  # noqa: E731
check("gate: all four bars met -> PASS", glovescore.gate_verdict(T(good), T(fc_ok))["PASS"] is True)
for nm, gl, bar in [("accuracy 35 / 40 = 0.875", G(50, 40, 35), "1_accuracy"),
                    ("Wilson lower bound under 0.80 (27 / 30 = 0.90)", G(40, 30, 27), "1_accuracy"),
                    ("29 decided", G(35, 29, 29), "1_accuracy"),
                    ("unsure 18 / 58 = 0.31", G(58, 40, 38), "2_unsure"),
                    ("1.1 swaps a round", G(50, 40, 38, 1.1), "3_better_than_tracking"),
                    ("no span for swaps", G(50, 40, 38, None), "3_better_than_tracking"),
                    ("correct share 25 / 50, not above the baseline 0.5", G(50, 40, 25), "3_better_than_tracking")]:
    v = glovescore.gate_verdict(T(gl), T(fc_ok))
    check(f"gate fails on {nm}", v[bar] is False and v["PASS"] is False, str(v))
v = glovescore.gate_verdict(T(G(50, 35, 34)), T(fc_ok))
check("bar 2 holds at exactly 0.30 unsure (15 / 50; 1 - 35/50 is 0.30000000000000004 in floats)", v["2_unsure"] is True, str(v))
v = glovescore.gate_verdict(T(G(100, 80, 72)), T(fc_ok))
check("bar 1 accuracy holds at exactly 0.90 (72 / 80)", v["1_accuracy"] is True, str(v))
v = glovescore.gate_verdict(T(good), T(G(30, 3, 0, None)))
check("bar 4 holds at exactly 10% wrong (3 / 30)", v["4_fails_safe"] is True, str(v))
v = glovescore.gate_verdict(T(good), T(G(40, 5, 0)))
check("gate fails when FAIL-CASE is confidently wrong on > 10%", v["4_fails_safe"] is False and not v["PASS"])
v = glovescore.gate_verdict(T(good), T(G(0, 0, 0, None)))
check("gate fails (not evaluated) when FAIL-CASE has no scored moment", v["4_fails_safe"] is False)
broken = {"methods": {"glove": good, "bytetrack": base, "linker": {"error": "no fragment", "correct_share": None}}}
v = glovescore.gate_verdict(broken, T(fc_ok))
check("gate bar 3 not met when a baseline failed to run (never compared against zero)",
      v["3_better_than_tracking"] is False and not v["PASS"])

# ---- labels ---------------------------------------------------------------------------------
print("label files")
tp = tmp / "t.labels.csv"
glovescore.write_template(table, tp)
txt = tp.read_text(encoding="utf-8")
check("template: header, column row and one row per 5 s", txt.count("\n") == 8 + 6 and "t_s,a_side" in txt)
raises("template refuses a table without a fingerprint",
       lambda: glovescore.write_template(PoseTable(**{**table.__dict__, "video_fp": ""}), tmp / "x.csv"), "fingerprint")


def filled(sides=("L", "L", "L", "?", "R", "R"), **hdr):
    h = {"format": glovescore.LABEL_FORMAT, "video_fp": FP, "every_s": "5", "A": "red gloves", "minutes": "4.5"}
    h.update(hdr)
    lines = [f"# {k}: {v}" for k, v in h.items() if v is not None] + ["t_s,a_side"]
    lines += [f"{t},{s}" for t, s in zip(range(0, 30, 5), sides)]
    q = tmp / "filled.labels.csv"
    q.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return q


lb = glovescore.parse_labels(filled(), table)
check("filled labels parse", lb.items[3] == (15, "?") and lb.minutes == 4.5 and lb.a_words == "red gloves")
lower = filled(sides=("l", "l", "l", "?", "r", "r"))
check("lower-case l / r accepted", glovescore.parse_labels(lower, table).items[0] == (0, "L"))
for nm, kw, why in [("another video", {"video_fp": "cd" * 32}, "different video"),
                    ("no fingerprint header", {"video_fp": None}, "missing header"),
                    ("every_s 10", {"every_s": "10"}, "every_s"),
                    ("wrong format", {"format": "v0"}, "format"),
                    ("A empty", {"A": ""}, "who A is"),
                    ("A too long", {"A": "x" * 81}, "who A is"),
                    ("minutes blank", {"minutes": ""}, "number"),
                    ("minutes 0", {"minutes": "0"}, "minutes"),
                    ("minutes nan", {"minutes": "nan"}, "minutes"),
                    ("minutes 9999", {"minutes": "9999"}, "minutes")]:
    raises(f"labels refused: {nm}", lambda kw=kw: glovescore.parse_labels(filled(**kw), table), why)
raises("labels refused: a blank answer", lambda: glovescore.parse_labels(filled(sides=("L", "", "L", "?", "R", "R")), table), "a_side")
raises("labels refused: an answer that is not L/R/?", lambda: glovescore.parse_labels(filled(sides=("L", "X", "L", "?", "R", "R")), table), "a_side")
raises("labels refused: a row missing", lambda: glovescore.parse_labels(filled(sides=("L", "L", "L", "?", "R")), table), "expected 6")
q = filled()
q.write_text(q.read_text().replace("\n10,", "\n11,"), encoding="utf-8")
raises("labels refused: a time off the grid", lambda: glovescore.parse_labels(q, table), "not the template")
q = filled()
q.write_text(q.read_text().replace("# A: red gloves", "# A: red gloves\n# A: blue"), encoding="utf-8")
raises("labels refused: a header twice", lambda: glovescore.parse_labels(q, table), "twice")
q = filled()
q.write_text(q.read_text().replace("\n10,L", "\nFirstname-Lastname.mp4,L").replace("\n15,?", "\n15,Firstname"),
             encoding="utf-8")
try:
    glovescore.parse_labels(q, table)
    check("labels refused: a value off the grid", False)
except ValueError as e:
    check("labels refused without echoing the bad value (it may carry a name)", "firstname" not in str(e).lower(), str(e))
q2 = filled()
q2.write_text(q2.read_text().replace("\n15,?", "\n15,Firstname"), encoding="utf-8")
try:
    glovescore.parse_labels(q2, table)
    check("labels refused: a bad side", False)
except ValueError as e:
    check("a bad side is refused without echoing it (in any case)", "firstname" not in str(e).lower() and "a_side" in str(e), str(e))
q = tmp / "bin.csv"; q.write_bytes(b"\xff\xfe\x00junk")
raises("labels refused: not UTF-8", lambda: glovescore.parse_labels(q, table), "UTF-8")
q = tmp / "big.csv"; q.write_bytes(b"#" * (glovescore.MAX_LABEL_BYTES + 1))
raises("labels refused: over 1 MB", lambda: glovescore.parse_labels(q, table), "bytes")
q = filled()
q.write_text(q.read_text().replace("\n5,L", "\n5,\"" + "L" * 140_000 + "\""), encoding="utf-8")
raises("labels refused: a 140,000-character field (csv's own limit), not a crash", lambda: glovescore.parse_labels(q, table), "CSV")
absurd = PoseTable(**{**table.__dict__, "fps": 1e-300, "n_frames": 900})
raises("template refused for a table claiming an absurd duration (fps 1e-300), before any grid is built",
       lambda: glovescore.write_template(absurd, tmp / "absurd.csv"), "s of video")

# ---- faceblur's location check, now with suffixes --------------------------------------------
print("output location (shared with the face blur)")
proot = tmp / "comps-participant-data"
proot.mkdir()
raises("default suffix still .mp4 only", lambda: faceblur.check_output_location(
    proot / "a.npz", allowed_root=proot, syncthing_configs=[]), "mp4", faceblur.UnsafeOutput)
check("an allowed suffix passes", faceblur.check_output_location(
    proot / "a.npz", allowed_root=proot, syncthing_configs=[], suffixes=(".npz",)) == (proot / "a.npz").resolve())

# ---- what the freeze and the provenance cover ------------------------------------------------
print("freeze boundary: untracked code, subfolders, import origins, the manifest lock")
from types import SimpleNamespace  # noqa: E402
import subprocess  # noqa: E402

repo = tmp / "minirepo"
(repo / "src" / "boxingvi").mkdir(parents=True)
(repo / "src" / "boxingvi" / "appearance.py").write_text("def hsv(x):\n    return x\n")
(repo / "glove_identity.py").write_text("# cli\n")
g = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True)  # noqa: E731
g("init", "-q"); g("-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
head = g("rev-parse", "HEAD").stdout.strip()
check("git_same: a clean checkout at the frozen commit", glove_identity.git_same(head, repo))
d0 = glove_identity.code_digest(repo)
shadow = repo / "src" / "boxingvi" / "appearance"
shadow.mkdir()
(shadow / "__init__.py").write_text("def hsv(x):\n    return 0\n")
check("git_same refuses an UNTRACKED package folder that shadows a tracked module (git diff never sees it)",
      not glove_identity.git_same(head, repo))
check("code_digest covers subfolders (the shadow changes it)", glove_identity.code_digest(repo) != d0)
shutil.rmtree(shadow)
(repo / "src" / "boxingvi" / "__pycache__").mkdir()
(repo / "src" / "boxingvi" / "__pycache__" / "x.pyc").write_bytes(b"\0")
check("code_digest ignores __pycache__ (derived)", glove_identity.code_digest(repo) == d0)
(repo / "src" / "boxingvi" / "appearance.py").write_text("def hsv(x):\n    return 1\n")
check("git_same refuses a tracked change", not glove_identity.git_same(head, repo))
check("git_same refuses a commit that does not exist", not glove_identity.git_same("f" * 40, repo))

lib = ROOT / "src" / "boxingvi"
glove_identity.check_imports()
check("check_imports passes for this process's own library", True)
for nm, mods in [("a module from another folder", {"boxingvi.appearance": SimpleNamespace(__file__=str(tmp / "x" / "appearance.py"))}),
                 ("a module from a package folder inside the library", {"boxingvi.appearance": SimpleNamespace(__file__=str(lib / "appearance" / "__init__.py"))}),
                 ("a module file under another name", {"boxingvi.gloves": SimpleNamespace(__file__=str(lib / "headgear.py"))}),
                 ("a module with no file", {"boxingvi.gloves": SimpleNamespace()}),
                 ("a namespace package with a second portion", {"boxingvi": SimpleNamespace(__path__=[str(lib), str(tmp / "other" / "boxingvi")])})]:
    raises(f"check_imports refuses {nm}", lambda m=mods: glove_identity.check_imports(m), "refusing to run", glove_identity.Refused)

lk = proot / "lockme.json"
with glove_identity.manifest_lock(lk, 0.2):
    raises("a second writer waits, then is refused, while the manifest is locked",
           lambda: glove_identity.manifest_lock(lk, 0.2).__enter__(), "locked", glove_identity.Refused)
check("the lock is released on exit", not (proot / "lockme.lock").exists())

# ---- the CLI ----------------------------------------------------------------------------------
print("glove_identity.py")
ctx = {"participant_root": proot, "syncthing_configs": [], "same": lambda c: True}
clip = tmp / "clip.mp4"
with video.VideoWriter(clip, W, H, Fraction(30), private=True) as vw:
    for fr in frames():
        vw.write(fr)
vids = [clip, tmp / "v2.mp4", tmp / "v3.mp4"]
vids[1].write_bytes(b"second video bytes" * 100)
vids[2].write_bytes(b"third video bytes" * 100)
reports = []
for i, (v, src) in enumerate(zip(vids, ("3" * 64, "1" * 64, "2" * 64)), 1):
    rp = tmp / f"r{i}.blur.json"
    rp.write_text(json.dumps({"source_sha256": src, "output_sha256": glove_identity.sha256_file(v)}))
    reports.append(rp)
man = proot / "m.json"


def run(*argv, **kw):
    return glove_identity.main([str(a) for a in argv], **{**ctx, **kw})


pairs = sum((["--pair", v, r] for v, r in zip(vids, reports)), [])
check("roles refused outside the participant root", run("roles", *pairs, "--fail-case", "3", "--out", tmp / "m.json") == 1)
check("roles refused with --fail-case 4", run("roles", *pairs, "--fail-case", "4", "--out", man) == 1)
bad_pairs = ["--pair", vids[0], reports[1]] + pairs[3:]
check("roles refused when a video is not its report's output", run("roles", *bad_pairs, "--fail-case", "3", "--out", man) == 1)
check("roles refused with two pairs", run("roles", *pairs[:6], "--fail-case", "2", "--out", man) == 1)
check("roles written", run("roles", *pairs, "--fail-case", "3", "--out", man) == 0)
m = json.loads(man.read_text())
check("DEV = the non-fail pair with the smaller original sha256 (pair 2), TEST = pair 1",
      m["roles"]["dev"]["source_sha256"] == "1" * 64 and m["roles"]["test"]["source_sha256"] == "3" * 64
      and m["roles"]["fail_case"]["source_sha256"] == "2" * 64)
check("roles refused when the manifest exists", run("roles", *pairs, "--fail-case", "3", "--out", man) == 1)

fps_ = {i: video.fingerprint(v) for i, v in enumerate(vids)}
tab = {}
for i, role in ((1, "dev"), (0, "test"), (2, "fail_case")):
    t_, f_, _ = scene(ga=(RED, RED), gb=(BLUE, BLUE) if role != "fail_case" else (RED, RED))
    t_ = PoseTable(**{**t_.__dict__, "video_fp": fps_[i]})
    tp_ = tmp / f"{role}.tracks.npz"
    t_.save(tp_)
    c_ = gloves.compute_gloves(None, t_, frames=f_(), provenance=glove_identity.provenance())
    cp_ = tmp / f"{role}.gloves.npz"
    c_.save(cp_)
    tab[role] = (t_, tp_, cp_)

# compute through the real decoder: the clip is pair 1 = TEST role
clip2 = tmp / "unregistered.mp4"
small, small_frames, _ = scene(n=60, swap_at=None)
with video.VideoWriter(clip2, W, H, Fraction(30), private=True) as vw:
    for fr in small_frames():
        vw.write(fr)
PoseTable(**{**small.__dict__, "video_fp": video.fingerprint(clip2)}).save(tmp / "unregistered.tracks.npz")
check("compute refused for an unregistered video, even with its own matching table", run(
    "compute", "--manifest", man, "--video", clip2, "--tracks", tmp / "unregistered.tracks.npz",
    "--out", proot / "c.npz") == 1 and not (proot / "c.npz").exists())
check("template refused for a table of an unregistered video", run(
    "template", "--manifest", man, "--tracks", tmp / "unregistered.tracks.npz", "--out", proot / "u.csv") == 1)
check("compute refused outside the root", run("compute", "--manifest", man, "--video", clip, "--tracks", tab["test"][1],
                                              "--out", tmp / "c.npz") == 1)
check("compute refused for a video that is not the table's", run("compute", "--manifest", man, "--video", vids[2],
                                                                 "--tracks", tab["test"][1], "--out", proot / "c.npz") == 1)
err = io.StringIO()
named = tmp / "Firstname-Lastname-round1.mp4"
real_err, sys.stderr = sys.stderr, err
try:
    rc = run("compute", "--manifest", man, "--video", named, "--tracks", tab["test"][1], "--out", proot / "c2.npz")
finally:
    sys.stderr = real_err
check("a missing video is refused without a traceback and without printing its path or name",
      rc == 1 and "Firstname" not in err.getvalue() and str(tmp) not in err.getvalue(), err.getvalue())
check("compute through ffmpeg decode", run("compute", "--manifest", man, "--video", clip, "--tracks", tab["test"][1],
                                           "--out", proot / "test.gloves.npz") == 0)
outside = tmp / "outside-shared.npz"
outside.write_bytes(b"keep me")
import os  # noqa: E402
os.link(outside, proot / "linked.gloves.npz")
check("compute --overwrite onto a planted hard link replaces the entry, never writes through it",
      run("compute", "--manifest", man, "--video", clip, "--tracks", tab["test"][1], "--out",
          proot / "linked.gloves.npz", "--overwrite") == 0 and outside.read_bytes() == b"keep me")
dec = GloveCounts.load(proot / "test.gloves.npz")
check("counts written by compute carry this code's provenance", dec.provenance == glove_identity.provenance())
idd = gloves.identify(tab["test"][0], dec)
check("decoded H.264 clip: still separable (a real decode, not painted arrays)", idd.verdict == "separable",
      f"{idd.verdict} {idd.diagnostics}")
gitdir = proot / "Firstname-repo"; (gitdir / ".git").mkdir(parents=True)
err = io.StringIO()
real_err, sys.stderr = sys.stderr, err
try:
    rc = run("template", "--manifest", man, "--tracks", tab["dev"][1], "--out", gitdir / "l.csv")
finally:
    sys.stderr = real_err
check("template refused inside a git checkout, and the refusal names no folder", rc == 1
      and "git_checkout" in err.getvalue() and "Firstname" not in err.getvalue(), err.getvalue())
check("template refused with a .txt name", run("template", "--manifest", man, "--tracks", tab["dev"][1],
                                               "--out", proot / "l.txt") == 1)
for role in ("dev", "test", "fail_case"):
    check(f"template written ({role})", run("template", "--manifest", man, "--tracks", tab[role][1],
                                            "--out", proot / f"{role}.labels.csv") == 0)
    lp = proot / f"{role}.labels.csv"
    t_ = lp.read_text(encoding="utf-8").replace("# A: ", "# A: red gloves").replace("# minutes: ", "# minutes: 2")
    rows_ = [ln for ln in t_.splitlines()]
    out_ = []
    for ln in rows_:
        if ln and ln[0].isdigit():
            tt = int(ln.split(",")[0])
            ln = f"{tt},{'L' if tt < 15 else ('?' if tt == 15 else 'R')}"
        out_.append(ln)
    lp.write_text("\n".join(out_) + "\n", encoding="utf-8")


def score(role, **kw):
    return run("score", "--manifest", man, "--tracks", tab[role][1], "--counts", tab[role][2], "--labels",
               proot / f"{role}.labels.csv", "--out", proot / f"{role}.score.json", **kw)


old = gloves.GloveCounts.load(tab["dev"][2])
stale = tmp / "dev.stale.gloves.npz"
gloves.GloveCounts(**{**old.__dict__, "provenance": ""}).save(stale)
check("counts without this code's provenance are refused (computed by another rule or version)", run(
    "score", "--manifest", man, "--tracks", tab["dev"][1], "--counts", stale, "--labels",
    proot / "dev.labels.csv", "--out", proot / "stale.score.json") == 1)
nm_labels = tmp / "Firstname-Lastname-sparring.labels.csv"
nm_labels.write_text("", encoding="utf-8")
err = io.StringIO()
real_err, sys.stderr = sys.stderr, err
try:
    rc = run("score", "--manifest", man, "--tracks", tab["dev"][1], "--counts", tab["dev"][2], "--labels",
             nm_labels, "--out", proot / "named.score.json")
finally:
    sys.stderr = real_err
check("a refused label file is not named in the message (names may be people's)",
      rc == 1 and "Firstname" not in err.getvalue() and "<file>" in err.getvalue(), err.getvalue())
err = io.StringIO()
real_err, sys.stderr = sys.stderr, err
try:
    rc = run("template", "--manifest", man, "--tracks", tab["dev"][1], "--out", proot / "Firstname.txt")
finally:
    sys.stderr = real_err
check("an output refusal gives its reason code only", rc == 1 and "Firstname" not in err.getvalue()
      and "bad_suffix" in err.getvalue(), err.getvalue())
check("TEST refused before the freeze", score("test") == 1)
check("FAIL-CASE refused before the freeze", score("fail_case") == 1)
check("DEV scored before the freeze", score("dev") == 0)
dv = json.loads((proot / "dev.score.json").read_text())
check("DEV score is stamped design data", dv["design_data_not_evidence"] is True and dv["role"] == "dev")
check("DEV score names no file", "synthetic" not in json.dumps(dv) and str(tmp) not in json.dumps(dv))
check("DEV refused a second time without --overwrite", score("dev") == 1)
check("DEV rescored with --overwrite", run("score", "--manifest", man, "--tracks", tab["dev"][1], "--counts",
                                           tab["dev"][2], "--labels", proot / "dev.labels.csv", "--out",
                                           proot / "dev.score.json", "--overwrite") == 0)
real_link = glovescore.link


def no_link(*a, **k):
    raise ValueError("no fragment has a usable body")


glovescore.link = no_link
rc = run("score", "--manifest", man, "--tracks", tab["dev"][1], "--counts", tab["dev"][2], "--labels",
         proot / "dev.labels.csv", "--out", proot / "dev.score.json", "--overwrite")
glovescore.link = real_link
dl = json.loads((proot / "dev.score.json").read_text())["methods"]["linker"]
check("a linker that refuses is recorded as an error, not a crash", rc == 0 and "error" in dl, str(dl))
try:
    glove_identity.write_json(proot / "half.json", {"x": object()})   # not serialisable: fails mid-write
    check("a failing JSON write raises", False)
except TypeError:
    pass
leftovers = list(proot.glob("*.tmp*"))
check("no temp files left in the participant folder after a failed write", not leftovers
      and not (proot / "half.json").exists(), str(leftovers))
check("freeze refused with uncommitted rule files", run("freeze", "--manifest", man, git=lambda: ("a" * 40, False)) == 1)
check("freeze", run("freeze", "--manifest", man, git=lambda: ("a" * 40, True)) == 0)
check("a second freeze refused", run("freeze", "--manifest", man, git=lambda: ("a" * 40, True)) == 1)
saved = gloves.MARGIN_MIN
gloves.MARGIN_MIN = 0.21
recomputed = tmp / "test.recomputed.gloves.npz"
gloves.GloveCounts(**{**gloves.GloveCounts.load(tab["test"][2]).__dict__,
                      "provenance": glove_identity.provenance()}).save(recomputed)
check("TEST refused when the rule changed after the freeze, even with counts recomputed under the new rule",
      run("score", "--manifest", man, "--tracks", tab["test"][1], "--counts", recomputed, "--labels",
          proot / "test.labels.csv", "--out", proot / "test.score.json") == 1
      and not (proot / "m.test.claim").exists())
gloves.MARGIN_MIN = saved
check("TEST refused when the rule files differ from the frozen commit (logic change, same constants)",
      score("test", same=lambda c: False) == 1)
check("TEST scored once", score("test") == 0)
check("the manifest records the TEST score's sha256",
      json.loads(man.read_text())["scored"].get("test") == glove_identity.sha256_file(proot / "test.score.json"))
(proot / "test.score.json").rename(proot / "test.score.moved")
check("TEST refused a second time, even with its first output moved away", score("test") == 1
      and not (proot / "test.score.json").exists())
check("a claim file marks the TEST run", (proot / "m.test.claim").exists())
(proot / "m.test.claim").rename(proot / "m.test.claim.x")
check("TEST refused a second time even with its claim removed too (the manifest's record)",
      score("test") == 1 and not (proot / "test.score.json").exists())
(proot / "m.test.claim.x").rename(proot / "m.test.claim")
(proot / "test.score.moved").rename(proot / "test.score.json")
(proot / "m.fail_case.claim").write_text("claimed by a run that was interrupted\n")
check("FAIL-CASE refused while a claim exists (a concurrent or interrupted run), labels never read",
      score("fail_case") == 1 and not (proot / "fail_case.score.json").exists())
(proot / "m.fail_case.claim").unlink()       # the owner's deliberate decision, in a real run recorded with why
ts = json.loads((proot / "test.score.json").read_text())
check("TEST: glove 4 of 4 correct, ByteTrack 2 of 4 (the id swap)",
      ts["methods"]["glove"]["correct"] == 4 and ts["methods"]["bytetrack"]["correct"] == 2, json.dumps(ts["methods"]))
check("FAIL-CASE scored once", score("fail_case") == 0)
fc = json.loads((proot / "fail_case.score.json").read_text())
check("FAIL-CASE: not separable, nothing wrong", fc["glove_verdict"] == "not_separable" and fc["methods"]["glove"]["wrong"] == 0)
gate = lambda out: run("gate", "--manifest", man, "--test", proot / "test.score.json", "--fail-case",  # noqa: E731
                       proot / "fail_case.score.json", "--out", out)
rc = gate(proot / "gate.json")
gv = json.loads((proot / "gate.json").read_text())
check("gate runs; 4 decided moments cannot pass bar 1 (needs 30): exit 2", rc == 2 and gv["1_accuracy"] is False
      and gv["4_fails_safe"] is True, str(gv))
mj = json.loads(man.read_text())
man.write_text(json.dumps({**mj, "frozen": None}))
check("gate refused when the manifest is not frozen (score hashes alone prove nothing)", gate(proot / "g3.json") == 1)
man.write_text(json.dumps({**mj, "frozen": {"rule_digest": "e" * 64, "commit": "b" * 40}}))
check("gate refused when the scores were made under another freeze", gate(proot / "g4.json") == 1)
man.write_text(json.dumps(mj))
(proot / "m.fail_case.claim").rename(proot / "m.fail_case.claim.x")
check("gate refused when a score has no claim behind it", gate(proot / "g5.json") == 1)
(proot / "m.fail_case.claim.x").rename(proot / "m.fail_case.claim")
check("gate runs again on the restored manifest", gate(proot / "g6.json") == 2)
shutil.copy(proot / "test.score.json", tmp / "keep.json")
tsx = dict(ts); tsx["methods"] = {**ts["methods"], "glove": {**ts["methods"]["glove"], "correct": 99}}
(proot / "test.score.json").write_text(json.dumps(tsx))
check("gate refused for a score file edited after the run (inconsistent counts)", gate(proot / "gate2.json") == 1)
(proot / "test.score.json").write_text(json.dumps({**ts, "label_minutes": 1.0}))
check("gate refused for a score file edited after the run (consistent, but not the recorded file)",
      gate(proot / "gate2b.json") == 1)
mj = json.loads(man.read_text())
for nm, bad in [("an extra key", {**mj, "x": 1}),
                ("two roles on one video", {**mj, "roles": {**mj["roles"], "test": mj["roles"]["dev"]}}),
                ("a non-hex fingerprint", {**mj, "roles": {**mj["roles"], "dev": {**mj["roles"]["dev"], "blurred_fp": "z" * 64}}}),
                ("frozen without a commit", {**mj, "frozen": {"rule_digest": "a" * 64}}),
                ("scored for dev", {**mj, "scored": {**mj["scored"], "dev": "a" * 64}})]:
    man.write_text(json.dumps(bad))
    check(f"manifest refused: {nm}", run("template", "--manifest", man, "--tracks", tab["dev"][1],
                                         "--out", proot / "z.csv") == 1)
man.write_text("[" * 100_000 + "]" * 100_000)
check("manifest refused: 100,000 levels of nesting (no RecursionError traceback)", run(
    "template", "--manifest", man, "--tracks", tab["dev"][1], "--out", proot / "z.csv") == 1)
man.write_text("{not json")
check("manifest refused: not JSON", run("template", "--manifest", man, "--tracks", tab["dev"][1],
                                        "--out", proot / "z.csv") == 1)

print(f"\n{PASS} passed, {FAIL} failed")
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(1 if FAIL else 0)
