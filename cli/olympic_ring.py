"""Ring-floor filter on Olympic Boxing: drop the tracked people who stand outside the ring, then link and
test completeness exactly as before. Gate (pre-registered before any held-out number):
docs/gates/ring-floor-filter.md.

  1. python cli/olympic_ring.py prepare --data data/olympic
       per video: a person-free background image and how still the camera was (pixels only, no labels)
       -> results/olympic_ring/<id>.background.png, results/olympic_ring/stillness.json
  2. python cli/olympic_ring.py click --floors data/olympic_ring/floors.json
       a person clicks each ring floor on its background image (timed); development videos first
  3. python cli/olympic_ring.py evaluate --data data/olympic --tracks results/ingest/olympic \
         --floors data/olympic_ring/floors.json --split dev
       the margin grid on the 4 development videos, and the margin the gate's rule picks
  4. ... evaluate ... --split heldout
       the verdict on ADR-0013's 25 held-out videos; refused until the margin is frozen in this file
       (FROZEN_MARGIN_PX). Exit 2 when the gate fails.

Baseline and filtered arms run the SAME library calls as olympic_attribute.py, olympic_link.py and
olympic_completeness.py, on the same tables; the only difference is the ids `ring.ids_outside` drops.
Everything written here stays in the gitignored results/ and data/ folders: Olympic Boxing is
research-use only, so only aggregate numbers ever reach docs.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import find_punches  # noqa: E402
import olympic_completeness  # noqa: E402
import olympic_link  # noqa: E402
from boxingvi import completeness, linking, olympic, ring  # noqa: E402
from boxingvi.boxers import replace_bytes, replace_file  # noqa: E402
from boxingvi.video import MAX_PIXELS, _tool, fingerprint, probe  # noqa: E402
from olympic_colour import DEV_VIDEOS  # noqa: E402

DEFAULT_PREP = Path("results/olympic_ring")
DEFAULT_FLOORS = Path("data/olympic_ring/floors.json")
STILLNESS_FORMAT = "boxing-comps camera stillness v1"
STILL_KEYS = frozenset({"video_fp", "width", "height", "motion_px", "max_px", "samples"})

# ---- pre-registered (docs/gates/ring-floor-filter.md); changing any of these after a held-out number is a new gate
# ADR-0013's held-out manifest: the 29 Olympic videos minus DEV_VIDEOS. A held-out run needs EXACTLY these ids
# (Astra, review 4: a count of 25 would let a substitute stand in for a missing video).
HELD_OUT_VIDEOS = frozenset({
    "GH088416", "GH098416", "GH108416", "GH118416", "GH138416", "GH148416", "GH158416", "GH168416", "GH178416",
    "GH188416", "GH198416", "GH208416", "GH218416", "GH079681", "GH089681", "GH099681", "GH109681", "GH119681",
    "GH129681", "GH139681", "GH149681", "GH169681", "GH179681", "GH189681", "GH199681"})
STILL_MAX_PX = 20.0                     # camera motion above this: the video is left out of BOTH arms
MARGIN_GRID = (-30.0, -15.0, 0.0, 15.0, 30.0, 60.0)
DEV_MAX_DROPPED_CREDITED = 0.005         # dev rule: the margin may drop at most 0.5% of credited punches
FROZEN_MARGIN_PX: "float | None" = None  # set by the commit that freezes the dev choice; None = held-out refused
# The punch-finder rule rows 7c/7d were made with (results/punch_finder_v6.json "chosen", as find_punches.load_rule
# returns it). Held-out runs refuse any other: a different rule would be a different experiment (Astra, review 3).
REGISTERED_RULE = {"variant": "relative", "threshold": 3.6238817996457016, "min_gap_s": 0.4}
G1_MAX_RATIO = 0.90                      # linked unexplained bursts per exposure-minute, filtered / baseline
G1_MAX_UPPER = 1.00                      # ... and the bootstrap 95% upper bound of that ratio stays below 1
G2_MIN_REMOVED = 0.20                    # share of baseline non-boxer person-frames the filter removes
G3A_MAX_DROPPED = 0.01                   # credited punches whose person is dropped, pooled
G3A_MAX_DROPPED_VIDEO = 0.05             # ... and in any one video
G3B_MAX_FALL = 0.01                      # credited share may fall by at most 1 percentage point
S1_MAX_FALL = 0.01                       # linking coverage (credited punches on A or B), secondary
S4_MAX_FALL = 0.02                       # median both-identified, secondary
BOOT_N, BOOT_SEED = 2000, 20261027
# Exactly what olympic_completeness.py builds without --reacquire (row 7c), spelled out so a change to the
# library default cannot quietly make the two disagree.
LINK_PARAMS = linking.LinkParams(reacquire_s=0.0)


# ================================================================ small checks

def _inside_repo_ok(path: Path) -> None:
    """Derived Olympic files may live only in the gitignored data/ or results/ (or outside the repo). The folder
    is resolved, the file name is not: the writer replaces a link planted at the name instead of following it,
    so the name's own location is what must be checked (Astra, review 3). The fully resolved path is checked too:
    a folder output (`results/..`) or a junction under results/ pointing into docs/ must not pass (review 4)."""
    path = Path(path)
    if path.name in ("", ".", ".."):
        raise ValueError(f"{path}: name a file or folder, not '{path.name or path}'")
    for p in (path.parent.resolve() / path.name, path.resolve()):
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            continue
        if not rel.parts or rel.parts[0] not in ("data", "results"):
            raise ValueError(f"{path}: derived Olympic files go under data/ or results/, never elsewhere in the repo")


def load_stillness(path) -> dict:
    """Read prepare's stillness file; every field is checked (it is read back from disk)."""
    p = Path(path)
    with open(p, "rb") as fh:
        raw = fh.read(ring.MAX_FILE_BYTES + 1)
    if len(raw) > ring.MAX_FILE_BYTES:
        raise ValueError(f"{p.name}: too large")
    try:
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=ring._no_duplicate_keys,
                         parse_constant=ring._refuse_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as e:
        raise ValueError(f"{p.name}: not readable JSON ({type(e).__name__})") from None
    except ValueError as e:                # a duplicated key, NaN, Infinity
        raise ValueError(f"{p.name}: {e}") from None
    if not isinstance(doc, dict) or set(doc) != {"format", "videos"} or doc["format"] != STILLNESS_FORMAT \
            or not isinstance(doc["videos"], dict) or len(doc["videos"]) > ring.MAX_VIDEOS:
        raise ValueError(f"{p.name}: not a {STILLNESS_FORMAT!r} file")
    out = {}
    for vid, e in doc["videos"].items():
        if not isinstance(vid, str) or not ring.VIDEO_ID.fullmatch(vid) or not isinstance(e, dict) or set(e) != STILL_KEYS:
            raise ValueError(f"{p.name}: bad entry {vid!r}")
        if not isinstance(e["video_fp"], str) or not ring.FINGERPRINT.fullmatch(e["video_fp"]):
            raise ValueError(f"{p.name}: {vid}: bad video_fp")
        w = ring._whole(e["width"], f"{vid}: width", 1, ring.MAX_SIDE)
        h = ring._whole(e["height"], f"{vid}: height", 1, ring.MAX_SIDE)
        if w * h > MAX_PIXELS:
            raise ValueError(f"{p.name}: {vid}: {w}x{h} is larger than any video this project reads")
        if ring._whole(e["samples"], f"{vid}: samples", 1, 1000) != ring.STILL_SAMPLES:
            raise ValueError(f"{p.name}: {vid}: measured on {e['samples']!r} samples, the rule needs {ring.STILL_SAMPLES}")
        motion, mx = ring._number(e["motion_px"], f"{vid}: motion_px"), ring._number(e["max_px"], f"{vid}: max_px")
        if not 0 <= motion <= mx:
            raise ValueError(f"{p.name}: {vid}: needs 0 <= motion_px <= max_px")
        # No shift can exceed the frame's diagonal (the unmeasurable fallback IS the diagonal): a larger value
        # was not measured by `prepare`, and would silently leave the video out (Astra, review 6)
        if mx > math.hypot(w, h):
            raise ValueError(f"{p.name}: {vid}: max_px {mx} is larger than the {w}x{h} frame's diagonal")
        out[vid] = e
    return out


# ================================================================ prepare

def write_png(path: Path, rgb: np.ndarray) -> None:
    h, w = rgb.shape[:2]
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or w * h > MAX_PIXELS:
        raise ValueError("expected a uint8 RGB image")

    def enc(tmp: Path) -> None:
        cmd = [_tool("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-i", "-", "-frames:v", "1", "-map_metadata", "-1", "-fflags", "+bitexact",
               "-f", "image2", "-update", "1", "-c:v", "png", str(tmp)]
        done = subprocess.run(cmd, input=np.ascontiguousarray(rgb).tobytes(), capture_output=True, timeout=120)
        if done.returncode != 0:
            raise RuntimeError(f"ffmpeg could not write {path.name}")
    replace_file(path, enc)


def cmd_prepare(a) -> int:
    _inside_repo_ok(a.out_dir)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    still_path = a.out_dir / "stillness.json"
    videos = load_stillness(still_path) if still_path.is_file() else {}
    only = {olympic_link._check_video_id(v) for v in a.only} if a.only else None
    for t in olympic.find_tasks(a.data):
        vid = t.video.stem
        if only is not None and vid not in only:
            continue
        info = probe(t.video)
        frames = ring.sample_frames(t.video, info)
        bg = ring.background(frames)
        m = ring.camera_motion(frames)
        write_png(a.out_dir / f"{vid}.background.png", bg)
        videos[vid] = {"video_fp": fingerprint(t.video), "width": info.width, "height": info.height,
                       # full precision: the 20 px rule reads this number, and 20.0003 rounded is 20.00 (Astra, review 6)
                       "motion_px": float(m["motion_px"]), "max_px": float(m["max_px"]), "samples": m["samples"]}
        print(f"{vid}: camera motion {m['motion_px']:.1f} px (p{ring.STILL_QUANTILE}), max {m['max_px']:.1f} px "
              f"-> {'still' if m['motion_px'] <= STILL_MAX_PX else 'MOVES: left out of both arms'}")
    doc = {"format": STILLNESS_FORMAT, "videos": {v: videos[v] for v in sorted(videos)}}
    replace_bytes(still_path, json.dumps(doc, indent=1, allow_nan=False).encode("utf-8"))
    return 0


# ================================================================ click

SNAP_DISPLAY_PX = 10
# Display sizes tried, largest first, as (zoom, subsample): Tk resizes photos only by whole numbers, so 3/4 is
# zoom 3 then subsample 4. On a 1920x1080 screen a 1920x1080 frame shows at 3/4 (1440x810); at 1/2 (the old
# rule) the corner posts were too small to click on.
DISPLAY_RATIOS = ((1, 1), (3, 4), (2, 3), (1, 2), (1, 3), (1, 4))
WINDOW_CHROME_PX = 180  # title bar, the instruction lines above the picture, and the taskbar


def display_fit(width: int, height: int, screen_w: int, screen_h: int) -> "tuple[int, int]":
    """The largest (zoom, subsample) whose picture fits the screen with the window's own lines around it."""
    avail_w, avail_h = screen_w * 0.95, max(screen_h - WINDOW_CHROME_PX, 1)
    for z, s in DISPLAY_RATIOS:
        if width * z / s <= avail_w and height * z / s <= avail_h:
            return z, s
    return 1, max(1, math.ceil(max(width / avail_w, height / avail_h)))


def display_to_frame(x: float, y: float, scale: float, width: int, height: int) -> "tuple[float, float]":
    """A click on the resized picture (`scale` frame pixels per screen pixel) -> frame pixels, held inside the
    frame; within SNAP_DISPLAY_PX of an edge it snaps onto the edge (that is how a floor that runs off the frame
    is closed)."""
    dw, dh = width / scale, height / scale
    x = 0.0 if x <= SNAP_DISPLAY_PX else float(width) if x >= dw - SNAP_DISPLAY_PX else x * scale
    y = 0.0 if y <= SNAP_DISPLAY_PX else float(height) if y >= dh - SNAP_DISPLAY_PX else y * scale
    return min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height))


def click_order(vids) -> "list[str]":
    """Development videos first, so the margin can be chosen while the rest are clicked."""
    return sorted(vids, key=lambda v: (v not in DEV_VIDEOS, v))


INSTRUCTIONS = ("Outline the RING FLOOR (the canvas inside the ropes) at floor level, going round it in order. "
                "Click where each corner post stands on the floor (estimate it when the post hides it). The corner "
                "nearest the camera is usually below the picture: there, click where each floor edge leaves the "
                "picture, on the picture's edge (a click within 10 px of an edge snaps onto it, and near a corner of "
                "the picture onto that corner). Usually 4 to 6 points.\n"
                "Left click: add point   |   Right click or Backspace: undo   |   Enter: save and next video   |   "
                "Esc: skip video   |   q: quit (run the command again to resume)")


class FloorClicker:
    """The click window. Its actions (add / undo / save / skip) are plain methods, so a test can drive them
    on a hidden window; cmd_click only binds them to the mouse and keys and runs the loop."""

    def __init__(self, root, todo, still: dict, floors: dict, floors_path: Path, prep_dir: Path, clock=time.monotonic):
        import tkinter as tk
        self.tk, self.root, self.todo, self.still = tk, root, list(todo), still
        self.floors, self.floors_path, self.prep_dir, self.clock = floors, floors_path, prep_dir, clock
        self.i, self.pts, self.t0, self.img, self.scale, self.done = 0, [], 0.0, None, 1, False
        self.help = tk.Label(root, text=INSTRUCTIONS, wraplength=1200, justify="left", font=("Segoe UI", 10))
        self.help.pack(anchor="w", padx=6)
        self.count = tk.Label(root, text="", font=("Segoe UI", 10, "bold"))
        self.count.pack(anchor="w", padx=6)
        self.status = tk.Label(root, text="", fg="red", font=("Segoe UI", 10, "bold"))
        self.status.pack(anchor="w", padx=6)
        self.canvas = tk.Canvas(root, cursor="crosshair", highlightthickness=0)
        self.canvas.pack()
        self.show()

    @property
    def video(self) -> "str | None":
        return self.todo[self.i] if self.i < len(self.todo) else None

    def show(self) -> None:
        if self.video is None:
            self.done = True
            return
        e = self.still[self.video]
        img = self.tk.PhotoImage(file=str(self.prep_dir / f"{self.video}.background.png"))
        if (img.width(), img.height()) != (e["width"], e["height"]):
            raise ValueError(f"{self.video}: background image is not {e['width']}x{e['height']}: run prepare again")
        z, s = display_fit(e["width"], e["height"], self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        self.scale = s / z
        big = img.zoom(z) if z > 1 else img
        self.img = big.subsample(s) if s > 1 else big
        self.help.config(wraplength=max(self.img.width() - 12, 400))
        self.pts, self.t0 = [], self.clock()
        self.canvas.config(width=self.img.width(), height=self.img.height())
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.img)
        split = "development" if self.video in DEV_VIDEOS else "held-out"
        self.root.title(f"Ring floor: {self.video} ({split}), {self.i + 1} of {len(self.todo)}")
        self.status.config(text="")
        self.redraw()

    def redraw(self) -> None:
        self.canvas.delete("poly")
        d = [(x / self.scale, y / self.scale) for x, y in self.pts]
        for k, (x, y) in enumerate(d):
            if k:
                self.canvas.create_line(*d[k - 1], x, y, fill="yellow", width=2, tags="poly")
        if len(d) > 2:
            self.canvas.create_line(*d[-1], *d[0], fill="yellow", dash=(4, 4), tags="poly")
        for k, (x, y) in enumerate(d):
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline="black", fill="yellow", width=1, tags="poly")
            self.canvas.create_text(x + 9, y - 9, text=str(k + 1), fill="yellow", anchor="sw",
                                    font=("Segoe UI", 11, "bold"), tags="poly")
        n = len(d)
        self.count.config(text=f"{n} point{'s' if n != 1 else ''}" + (" - Enter saves" if n >= ring.MIN_POINTS else
                                                                      f" - at least {ring.MIN_POINTS} needed"))

    def add(self, x: float, y: float) -> None:
        if self.video is None:
            return
        e = self.still[self.video]
        self.pts.append(display_to_frame(x, y, self.scale, e["width"], e["height"]))
        self.redraw()

    def undo(self) -> None:
        if self.pts:
            self.pts.pop()
            self.redraw()

    def save(self) -> bool:
        """Check the outline exactly as the reader will, write the whole file, move on. False (and a red
        message, nothing written) when the outline would be refused."""
        if self.video is None:
            return False
        e = self.still[self.video]
        try:
            f = ring.check_floor(self.video, [list(p) for p in self.pts], e["width"], e["height"], e["video_fp"],
                                 round(self.clock() - self.t0, 1))
        except ValueError as err:
            self.status.config(text=f"Not saved: {err}")
            return False
        self.floors[self.video] = f
        ring.dump_floors(self.floors, self.floors_path)
        print(f"{self.video}: {len(f.points)} points in {f.click_s:.0f} s")
        self.i += 1
        self.show()
        return True

    def skip(self) -> None:
        if self.video is not None:
            print(f"{self.video}: skipped")
            self.i += 1
            self.show()


def cmd_click(a) -> int:
    import tkinter as tk

    _inside_repo_ok(a.floors)
    still = load_stillness(a.prep_dir / "stillness.json")
    floors = ring.load_floors(a.floors) if a.floors.is_file() else {}
    moving = sorted(v for v in still if still[v]["motion_px"] > STILL_MAX_PX)
    if moving:
        print(f"not shown (the camera moves; left out of the experiment): {moving}")
    todo = [v for v in click_order(still) if v not in moving and (a.redo or v not in floors)]
    if a.only:
        todo = [v for v in todo if v in set(a.only)]
    if not todo:
        print("every video already has a floor (use --redo to click again)")
        return 0
    root = tk.Tk()
    fc = FloorClicker(root, todo, still, floors, a.floors, a.prep_dir)
    # Started from a terminal, a Tk window can open BEHIND it: pull it to the front once, then let it behave
    # like any other window.
    root.geometry("+0+0")
    root.lift()
    root.attributes("-topmost", True)
    root.after(800, lambda: root.attributes("-topmost", False))
    root.focus_force()
    print(f"the click window is open ('Ring floor: {todo[0]} ...'; if you cannot see it, look in the taskbar)")

    def after(action):
        def handler(ev=None):
            action(ev)
            if fc.done:
                root.destroy()
        return handler

    fc.canvas.bind("<Button-1>", after(lambda ev: fc.add(ev.x, ev.y)))
    fc.canvas.bind("<Button-3>", after(lambda _ev: fc.undo()))
    root.bind("<BackSpace>", after(lambda _ev: fc.undo()))
    root.bind("<Return>", after(lambda _ev: fc.save()))
    root.bind("<Escape>", after(lambda _ev: fc.skip()))
    root.bind("q", lambda _e: root.destroy())
    root.mainloop()
    print(f"{len(floors)} floors in {a.floors}; saved or skipped this session: {fc.i} of {len(todo)}")
    return 0


# ================================================================ evaluate

def _arm(table, events, attrs, rule, vid) -> dict:
    """One arm of one video through the existing pipeline. Errors are recorded, never hidden."""
    out = {"events": len(events), "credited": sum(x.track_id is not None for x in attrs)}
    credited_ids = set(linking.credited_counts(attrs))
    out["rows"] = len(table.frame)
    out["noncredited_rows"] = int((~np.isin(table.track_id, list(credited_ids))).sum())
    minutes = table.n_frames / table.fps / 60
    try:
        lk = linking.link_video(table, attrs, LINK_PARAMS)
        res = completeness.find_bursts(table, lk, events, attrs, rule, video=vid)
        out["linked"] = olympic_completeness.video_row(res, len(events), minutes)
        s = linking.summarise(lk, attrs)
        out["link"] = {k: s[k] for k in ("both_identified", "credited_punches", "credited_in_a", "credited_in_b")}
        m = lk.membership
        on = m >= 0
        out["link"]["ab_person_frames"] = int(on.sum())
        out["link"]["ab_noncredited_person_frames"] = int((on & ~np.isin(m, list(credited_ids))).sum())
    except (ValueError, RuntimeError) as e:
        out["linked_error"] = str(e)
    try:
        res = completeness.find_bursts_credited(table, events, attrs, rule, video=vid, min_punches=1)
        out["credited_path"] = olympic_completeness.video_row_credited(res, len(events), minutes)
    except (ValueError, RuntimeError) as e:
        out["credited_path_error"] = str(e)
    return out


def evaluate_video(task, tracks: Path, floor: "ring.Floor", margins, rule, full: bool = True) -> dict:
    table, _, events, attrs = olympic_link.load_video(task, tracks)
    vid = task.video.stem
    base_cred = linking.credited_counts(attrs)
    noncred = ~np.isin(table.track_id, list(base_cred))
    out = {"video": vid, "base": _arm(table, events, attrs, rule, vid) if full else None, "margins": {}}
    for m in margins:
        drop = ring.ids_outside(table, floor, m)
        removed = np.isin(table.track_id, list(drop))
        r = {"ids_dropped": len(drop), "rows_dropped": int(removed.sum()),
             "noncredited_rows": int(noncred.sum()), "noncredited_rows_dropped": int((removed & noncred).sum()),
             "credited": int(sum(base_cred.values())),
             "credited_dropped": int(sum(n for t, n in base_cred.items() if t in drop))}
        if full:
            ft = ring.without_ids(table, drop)
            r["arm"] = _arm(ft, events, olympic.attribute(events, ft), rule, vid)
        out["margins"][m] = r
    return out


def _linked_ue(row) -> "tuple[int, float]":
    """(unexplained bursts, exposure minutes) of a linked row, summed over A and B."""
    return (sum(row["boxers"][n]["unexplained"] for n in "AB"), sum(row["boxers"][n]["exposure_min"] for n in "AB"))


def bootstrap_ratio(base: "list[tuple]", filt: "list[tuple]", n: int = BOOT_N, seed: int = BOOT_SEED) -> "tuple[float, float]":
    """Video bootstrap of (pooled filtered rate) / (pooled baseline rate); each item is (unexplained,
    exposure_min) for one video, paired by index. Returns the 2.5th and 97.5th percentiles."""
    if len(base) != len(filt) or not base:
        raise ValueError("paired, non-empty video lists")
    b, f = np.asarray(base, np.float64), np.asarray(filt, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), size=(n, len(b)))
    bs, fs = b[idx].sum(axis=1), f[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (fs[:, 0] / fs[:, 1]) / (bs[:, 0] / bs[:, 1])
    ratio = np.where(np.isfinite(ratio), ratio, np.inf)
    # nearest-rank, not interpolated: two infinite neighbours would interpolate to NaN (inf - inf)
    return (float(np.percentile(ratio, 2.5, method="inverted_cdf")),
            float(np.percentile(ratio, 97.5, method="inverted_cdf")))


def _share(num, den):
    return num / den if den else None


def summarise_margin(rows: "list[dict]", m: float) -> dict:
    """Pooled numbers of one margin over the given per-video results. Videos whose BASELINE fails are left
    out of both arms and listed; a video that fails only when filtered is counted (gate G3c)."""
    out = {"margin_px": m, "videos": len(rows), "baseline_failed": [], "newly_failed": []}
    full = bool(rows) and rows[0]["base"] is not None     # no rows: every bar is unmeasured (Astra, review 2)
    failed = lambda arm: "linked_error" in arm or "credited_path_error" in arm  # noqa: E731
    if full:
        base = [r["base"] for r in rows]
        filt = [r["margins"][m]["arm"] for r in rows]
        out["baseline_failed"] = [r["video"] for r, b in zip(rows, base) if failed(b)]
        out["newly_failed"] = [r["video"] for r, b, f in zip(rows, base, filt) if not failed(b) and failed(f)]
    # Every metric pools over the same videos: a video whose BASELINE fails is in no arm. A video that fails
    # only when filtered stays in the filter's cost (G2, G3a) and is G3c's failure, but cannot give a G1 rate.
    kept = [r for r in rows if r["video"] not in out.get("baseline_failed", ())]
    cheap = [r["margins"][m] for r in kept]
    out["credited"] = sum(c["credited"] for c in cheap)
    out["credited_dropped"] = sum(c["credited_dropped"] for c in cheap)
    out["credited_dropped_share"] = _share(out["credited_dropped"], out["credited"])
    out["credited_dropped_share_worst_video"] = max(((_share(c["credited_dropped"], c["credited"]) or 0.0) for c in cheap),
                                                    default=0.0)
    out["noncredited_rows"] = sum(c["noncredited_rows"] for c in cheap)
    out["noncredited_removed_share"] = _share(sum(c["noncredited_rows_dropped"] for c in cheap), out["noncredited_rows"])
    out["per_video"] = {r["video"]: {k: r["margins"][m][k] for k in ("ids_dropped", "rows_dropped", "noncredited_rows",
                                                                    "noncredited_rows_dropped", "credited", "credited_dropped")}
                        for r in rows}
    if not full:
        return out
    for r, b, f in zip(rows, base, filt):
        if not failed(b) and not failed(f):
            out["per_video"][r["video"]] |= {"linked_ue_baseline": list(_linked_ue(b["linked"])),
                                             "linked_ue_filtered": list(_linked_ue(f["linked"]))}
    ok = [i for i, r in enumerate(rows) if r["video"] not in out["baseline_failed"] and r["video"] not in out["newly_failed"]]
    out["videos_compared"] = len(ok)
    # G3b's credited shares exist whenever a baseline ran: attribution precedes linking, so a video that fails only
    # when filtered still has them. Computed BEFORE the early return, so G3b is measured even if no video could be
    # compared on G1 (Astra, reviews 4 and 5).
    g3 = [i for i, r in enumerate(rows) if r["video"] not in out["baseline_failed"]]
    out["credited_share"] = {name: _share(sum(arm[i]["credited"] for i in g3), sum(arm[i]["events"] for i in g3))
                             for name, arm in (("baseline", base), ("filtered", filt))}
    if not ok:
        return out
    for name, arm in (("baseline", base), ("filtered", filt)):
        lt = olympic_completeness.totals([arm[i]["linked"] for i in ok])
        ct = olympic_completeness.totals_credited([arm[i]["credited_path"] for i in ok])
        ue = [_linked_ue(arm[i]["linked"]) for i in ok]
        rate, theirs = _share(sum(u for u, _ in ue), sum(e for _, e in ue)), lt["unexplained_per_exposure_min"]
        if (rate is None) != (theirs is None) or (rate is not None and not math.isclose(rate, theirs, rel_tol=1e-9)):
            raise AssertionError("the bootstrap's rate is not the completeness rate")   # same definition, or no gate
        links = [arm[i]["link"] for i in ok]
        cp = sum(l["credited_punches"] for l in links)
        out[name] = {
            "linked_unexplained_per_exposure_min": lt["unexplained_per_exposure_min"],
            "linked_unexplained": lt["unexplained"], "linked_exposure_boxer_min": lt["exposure_boxer_min"],
            "credited_path_unexplained_per_exposure_min": ct["unexplained_per_exposure_min"],
            "credited_share": out["credited_share"][name],     # over every video whose baseline ran (G3b)
            "rows": sum(arm[i]["rows"] for i in ok),
            "link_coverage": _share(sum(l["credited_in_a"] + l["credited_in_b"] for l in links), cp),
            "ab_noncredited_share": _share(sum(l["ab_noncredited_person_frames"] for l in links),
                                           sum(l["ab_person_frames"] for l in links)),
            "median_both_identified": float(np.median([l["both_identified"] for l in links])),
        }
    b, f = out["baseline"], out["filtered"]
    rb, rf = b["linked_unexplained_per_exposure_min"], f["linked_unexplained_per_exposure_min"]
    # No baseline rate, or a baseline of 0 unexplained: there is nothing to cut, so no ratio (G1 cannot pass).
    out["ratio"] = rf / rb if rb and rf is not None else None
    # An unbounded resample (a zero baseline rate) is written as null: strict JSON has no Infinity, and the
    # report must still be written so the FAIL is on record (Astra, review 2).
    out["ratio_ci95"] = [x if math.isfinite(x) else None
                         for x in bootstrap_ratio([_linked_ue(base[i]["linked"]) for i in ok],
                                                  [_linked_ue(filt[i]["linked"]) for i in ok])]
    return out


def _ge(x, y, slack=0.0) -> bool:
    """x >= y - slack, False when either side was not measured (None)."""
    return x is not None and y is not None and x >= y - slack


def verdict(s: dict) -> dict:
    """Pass/fail of every pre-registered bar from one margin's summary. PASS needs G1, G2 and G3a-c. A bar
    that could not be measured (no compared video, a zero baseline) fails; it never passes by default."""
    b, f = s.get("baseline"), s.get("filtered")
    measured = b is not None and f is not None
    g = {
        "G1_linked_unexplained": measured and s["ratio"] is not None and s["ratio"] <= G1_MAX_RATIO
                                 and s["ratio_ci95"][1] is not None and s["ratio_ci95"][1] < G1_MAX_UPPER,
        "G2_nonboxer_removed": s["noncredited_removed_share"] is not None and s["noncredited_removed_share"] >= G2_MIN_REMOVED,
        "G3a_boxers_kept": s["credited_dropped_share"] is not None and s["credited_dropped_share"] <= G3A_MAX_DROPPED
                           and s["credited_dropped_share_worst_video"] <= G3A_MAX_DROPPED_VIDEO,
        "G3b_credited_share": _ge(s.get("credited_share", {}).get("filtered"), s.get("credited_share", {}).get("baseline"),
                                  G3B_MAX_FALL),
        # unmeasured (no video compared, e.g. every baseline failed) fails; a new failure fails
        "G3c_no_new_failures": s.get("videos_compared", 0) > 0 and not s.get("newly_failed", []),
    }
    sec = {} if not measured else {
        "S1_link_coverage": _ge(f["link_coverage"], b["link_coverage"], S1_MAX_FALL),
        "S2_absorption_lower": (f["ab_noncredited_share"] is not None and b["ab_noncredited_share"] is not None
                                and f["ab_noncredited_share"] < b["ab_noncredited_share"]),
        "S3_credited_path_not_worse": _ge(b["credited_path_unexplained_per_exposure_min"],
                                          f["credited_path_unexplained_per_exposure_min"]),
        "S4_both_identified": _ge(f["median_both_identified"], b["median_both_identified"], S4_MAX_FALL),
    }
    return {"gate": g, "secondary": sec, "pass": all(g.values())}


def select_margin(summaries: "list[dict]") -> "float | None":
    """The dev rule: among margins dropping at most DEV_MAX_DROPPED_CREDITED of credited punches, the one
    removing the most non-boxer person-frames; ties to the larger (more forgiving) margin. None: no margin
    qualifies, and the experiment stops."""
    ok = [s for s in summaries if s["credited_dropped_share"] is not None
          and s["credited_dropped_share"] <= DEV_MAX_DROPPED_CREDITED]
    if not ok:
        return None
    return max(ok, key=lambda s: ((s["noncredited_removed_share"] or 0.0), s["margin_px"]))["margin_px"]


def _fmt(x, pct=False):
    return "n/a" if x is None else (f"{x:.1%}" if pct else f"{x:.3f}")


def cmd_evaluate(a) -> int:
    if a.out is not None:
        _inside_repo_ok(a.out)
    if a.write_filtered is not None:
        _inside_repo_ok(a.write_filtered)
        if a.split == "dev" and a.margin is None:
            raise ValueError("--write-filtered writes one margin: pass --margin on --split dev")
        if a.write_filtered.resolve() == a.tracks.resolve():
            raise ValueError("--write-filtered would overwrite the baseline tables")
    if a.margin is not None and not math.isfinite(float(a.margin)):
        raise ValueError(f"--margin must be a finite number of pixels, got {a.margin}")
    floors = ring.load_floors(a.floors)
    still = load_stillness(a.prep_dir / "stillness.json")
    rule = find_punches.load_rule(a.rule)
    if a.split == "heldout":
        if FROZEN_MARGIN_PX is None:
            raise ValueError("the margin is not frozen yet: choose it on --split dev, commit FROZEN_MARGIN_PX, then run held-out")
        if a.margin is not None and float(a.margin) != FROZEN_MARGIN_PX:
            raise ValueError(f"held-out runs only at the frozen margin {FROZEN_MARGIN_PX} px")
        if rule != REGISTERED_RULE:      # a different rule would be a different experiment (Astra, review 3)
            raise ValueError(f"held-out runs only with the registered punch rule {REGISTERED_RULE}, got {rule}")
    tasks = [t for t in olympic.find_tasks(a.data) if (a.tracks / f"{t.video.stem}.tracks.npz").is_file()]
    # One video, one task, one recording, over the WHOLE dataset, before any split is taken: a copy under a
    # second name would otherwise be pooled twice or stand in for a missing video (Astra, review 2).
    stems = [t.video.stem for t in tasks]
    if len(set(stems)) != len(stems):
        raise ValueError(f"a video id appears in more than one task: {sorted({s for s in stems if stems.count(s) > 1})}")
    all_fps = {t.video.stem: fingerprint(t.video) for t in tasks}
    if len(set(all_fps.values())) != len(all_fps):
        raise ValueError("two tasks hold the same recording (equal fingerprints): it would be pooled twice")
    if a.split == "dev":
        tasks = [t for t in tasks if t.video.stem in DEV_VIDEOS]
        margins = tuple(MARGIN_GRID) if a.margin is None else (float(a.margin),)
        have = {t.video.stem for t in tasks}
        if have != DEV_VIDEOS:     # the margin is chosen over the fixed four, never a subset (Astra, review 3)
            raise ValueError(f"the development split needs all of {sorted(DEV_VIDEOS)}; missing {sorted(DEV_VIDEOS - have)}")
    else:
        tasks = [t for t in tasks if t.video.stem not in DEV_VIDEOS]
        margins = (FROZEN_MARGIN_PX,)
        have = {t.video.stem for t in tasks}
        if have != HELD_OUT_VIDEOS:      # membership, not a count
            raise ValueError(f"held-out needs exactly the {len(HELD_OUT_VIDEOS)} held-out tables: missing "
                             f"{sorted(HELD_OUT_VIDEOS - have)}, not in the manifest {sorted(have - HELD_OUT_VIDEOS)}")
    # Every input is bound to the video it describes BEFORE anything is left out: a stale stillness entry from
    # another recording must stop the run, not quietly drop a video (Astra, review 1).
    fps = {t.video.stem: all_fps[t.video.stem] for t in tasks}
    no_still = [t.video.stem for t in tasks if t.video.stem not in still]
    if no_still:
        raise ValueError(f"no stillness entry for {no_still}: run prepare first")
    stale = [v for v, fp in fps.items() if still[v]["video_fp"] != fp]
    if stale:
        raise ValueError(f"{stale}: the stillness entry was measured on a different recording: run prepare again")
    moving = sorted(v for v in fps if still[v]["motion_px"] > STILL_MAX_PX)
    tasks = [t for t in tasks if t.video.stem not in moving]
    no_floor = [t.video.stem for t in tasks if t.video.stem not in floors]
    if no_floor:
        raise ValueError(f"no floor for {no_floor}: click them first")
    wrong = [t.video.stem for t in tasks if floors[t.video.stem].video_fp != fps[t.video.stem]]
    if wrong:
        raise ValueError(f"{wrong}: the floor was clicked on different recordings than these videos")
    rows = []
    for t in tasks:
        r = evaluate_video(t, a.tracks, floors[t.video.stem], margins, rule)
        rows.append(r)
        print(f"{t.video.stem}: " + " | ".join(
            f"m={m:+.0f}: drop {c['ids_dropped']} ids, non-boxer rows {_fmt(_share(c['noncredited_rows_dropped'], c['noncredited_rows']), True)}, "
            f"credited punches lost {c['credited_dropped']}/{c['credited']}" for m, c in r["margins"].items()))
        if a.write_filtered is not None:
            m = margins[0]
            table, *_ = olympic_link.load_video(t, a.tracks)
            ft = ring.without_ids(table, ring.ids_outside(table, floors[t.video.stem], m), f"+ringfloor margin={m:g}px")
            a.write_filtered.mkdir(parents=True, exist_ok=True)

            def save(p, ft=ft):
                with open(p, "wb") as fh:
                    ft.save(fh)
            replace_file(a.write_filtered / f"{t.video.stem}.tracks.npz", save)
    summaries = [summarise_margin(rows, m) for m in margins]
    report = {"split": a.split, "rule": rule, "left_out_camera_moves": moving, "still_max_px": STILL_MAX_PX,
              "videos": [r["video"] for r in rows], "margins": summaries}
    if a.split == "dev":
        # Only the registered grid is a selection; one --margin is a look at that margin, and must not read as
        # the chosen one (Astra, review 6: `--margin 100000` drops nobody and would have been "selected")
        grid = a.margin is None
        report["selection"] = "registered grid" if grid else "single margin: not a selection"
        report["selected_margin_px"] = select_margin(summaries) if grid else None
        for s in summaries:
            line = (f"margin {s['margin_px']:+.0f} px: credited punches lost {_fmt(s['credited_dropped_share'], True)} "
                    f"(worst video {_fmt(s['credited_dropped_share_worst_video'], True)}), non-boxer person-frames "
                    f"removed {_fmt(s['noncredited_removed_share'], True)}")
            if "baseline" in s:
                line += (f", linked unexplained/exp-min {_fmt(s['baseline']['linked_unexplained_per_exposure_min'])} -> "
                         f"{_fmt(s['filtered']['linked_unexplained_per_exposure_min'])}")
            print(line)
        print(f"selected margin (dev rule): {report['selected_margin_px']}" if grid else
              "one margin only: no margin is selected (run without --margin for the registered grid)")
        code = 0
    else:
        s = summaries[0]
        report["verdict"] = verdict(s)
        print(json.dumps(report["verdict"], indent=1))
        if "baseline" in s:
            print(f"linked unexplained per exposure-minute {_fmt(s['baseline']['linked_unexplained_per_exposure_min'])} -> "
                  f"{_fmt(s['filtered']['linked_unexplained_per_exposure_min'])} (ratio {_fmt(s['ratio'])}, 95% "
                  f"{_fmt(s['ratio_ci95'][0])}-{_fmt(s['ratio_ci95'][1])})")
        print(f"{'PASS' if report['verdict']['pass'] else 'FAIL'}; videos compared {s.get('videos_compared', 0)}; "
              f"left out (camera moves): {moving}; baseline failed: {s.get('baseline_failed', [])}; "
              f"failed only when filtered: {s.get('newly_failed', [])}")
        code = 0 if report["verdict"]["pass"] else 2
    out = a.out or Path(f"results/olympic_ring_eval.{a.split}.json")
    _inside_repo_ok(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    replace_bytes(out, json.dumps(report, indent=1, allow_nan=False).encode("utf-8"))
    print(f"-> {out}")
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_PREP)
    p.add_argument("--only", nargs="+", metavar="ID")
    c = sub.add_parser("click")
    c.add_argument("--floors", type=Path, default=DEFAULT_FLOORS)
    c.add_argument("--prep-dir", type=Path, default=DEFAULT_PREP)
    c.add_argument("--only", nargs="+", metavar="ID")
    c.add_argument("--redo", action="store_true", help="click videos that already have a floor again")
    e = sub.add_parser("evaluate")
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--tracks", type=Path, required=True)
    e.add_argument("--floors", type=Path, default=DEFAULT_FLOORS)
    e.add_argument("--prep-dir", type=Path, default=DEFAULT_PREP)
    e.add_argument("--split", choices=("dev", "heldout"), default="dev")
    e.add_argument("--margin", type=float, default=None, help="dev: one margin instead of the grid")
    e.add_argument("--rule", type=Path, default=find_punches.DEFAULT_RULE)
    e.add_argument("--out", type=Path, default=None)
    e.add_argument("--write-filtered", type=Path, default=None, metavar="DIR",
                   help="also write the filtered tables (selected margin) for the existing CLIs and overlays")
    a = ap.parse_args(argv)
    try:
        return {"prepare": cmd_prepare, "click": cmd_click, "evaluate": cmd_evaluate}[a.cmd](a)
    except ValueError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
