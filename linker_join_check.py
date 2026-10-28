"""Brendan's hand check of the linker's joins on Olympic Boxing (option A; docs/gates/linker-join-check.md).

  python linker_join_check.py sample --data data/olympic --tracks results/ingest/olympic   draw the 100, once
  python linker_join_check.py check  --data data/olympic --tracks results/ingest/olympic   the window
  python linker_join_check.py check  ... --dry                                            every check, one clip, no window
  python linker_join_check.py verify                                                      answered / empty / minutes only
  python linker_join_check.py score                                                       the metric, once, when complete

`sample` links the 23 held-out videos exactly as the per-boxer report does (linking.link_video, default
parameters), lists every join, and draws the gate's seeded, gap-stratified 100. It refuses to run twice.

`check` shows one join at a time: fragment 1's last 1 s with a CYAN box, the gap with no box, fragment 2's
first 1 s with a YELLOW box, then holds the two boundary frames side by side. Answer for the person in the
cyan box at its end and the person in the yellow box at its start:

  S  same person     D  different person (the other boxer, the referee, anyone)     U  can't tell
  <- or Backspace  previous join     ->  next join     Space  replay     E  first unanswered     Q or Esc  quit (saved)

Every answer is saved at once; running the same command again resumes at the first unanswered join. After
50 minutes in one sitting the window takes no more answers until it is restarted (a break). The window never
shows which boxer the linker chose, the link cost, or anything about other joins.

Everything this writes (sample, answers, result) is derived from Olympic Boxing: it stays under results/
(gitignored), research-use only, never redistributed. `verify` deliberately does not split answers by value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import label_moments  # noqa: E402
from boxingvi import joincheck as jc  # noqa: E402
from boxingvi import labelmoments as lm  # noqa: E402

DEFAULT_DIR = Path("results/linker_join_check")
SAMPLE_NAME, ANSWERS_NAME, RESULT_NAME = "sample.json", "answers.csv", "result.json"
CYAN_HEX, YELLOW_HEX = "#00ffff", "#ffff00"
# S / D / U, plus #35's movement keys (its own L / R / ? answers are not keys here)
KEYS = {k: v for k, v in lm.KEYS.items() if v not in lm.VALUES} | {
    "s": "S", "S": "S", "d": "D", "D": "D", "u": "U", "U": "U"}


def manifest() -> frozenset:
    from olympic_headgear import HELD_OUT      # #29's frozen 23 held-out videos
    return HELD_OUT


# Olympic-derived files may only go under one of these results/ folders (both gitignored): this checkout's, and
# the main checkout's (a worktree is deleted when its session is archived; the answers must outlive it).
RESULTS_ROOTS = [Path(__file__).resolve().parent / "results", Path.home() / "repos" / "boxing-comps" / "results"]


def check_out_dir(d: Path) -> Path:
    r = Path(d).resolve()
    if not any(r == root.resolve() or lm._under(r, root.resolve()) for root in RESULTS_ROOTS):
        raise lm.Refused("the output folder must be under this repo's results/ (gitignored): Olympic-derived "
                         "files stay there")
    return r


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def data_path_check(root):
    """VideoGuard's path rule for Olympic videos: the file must resolve under the --data folder."""
    def check(video, _root=None):
        try:
            r = Path(video).resolve(strict=True)
            base = Path(root).resolve(strict=True)
        except OSError:
            raise lm.Refused("the video (or the data folder) does not exist") from None
        if not lm._under(r, base) or not r.is_file():
            raise lm.Refused("the video is not a file under the data folder")
        return r
    return check


# ---- sample ----------------------------------------------------------------------------------------
def cmd_sample(a) -> int:
    import olympic_link
    from boxingvi import linking, olympic
    from boxingvi.video import fingerprint
    out = check_out_dir(a.dir)
    if (out / SAMPLE_NAME).exists():
        raise lm.Refused("a sample already exists: it is drawn once and never redrawn")
    man = manifest()
    inside = data_path_check(a.data)
    tasks = {t.video.stem: t for t in olympic.find_tasks(a.data) if t.video.stem in man}
    per_video = {}
    for vid in sorted(man):
        t = tasks.get(vid)
        tracks = a.tracks / f"{vid}.tracks.npz"
        if t is None or not tracks.is_file():
            print(f"{vid}: missing")
            continue
        inside(t.video)                 # a task folder linked elsewhere is refused before anything is read
        try:
            table, info, events, attrs = olympic_link.load_video(t, a.tracks)
            lk = linking.link_video(table, attrs)
        except (ValueError, OSError, RuntimeError) as e:
            print(f"{vid}: did not link ({type(e).__name__})")
            continue
        joins = jc.joins_of(lk, vid)
        per_video[vid] = {"joins": joins, "positions": jc.positions_of(lk, linking.credited_counts(attrs)),
                          "chain_minutes": jc.chain_minutes(lk), "camera": jc.camera_of(t.name),
                          "fps": table.fps, "n_frames": table.n_frames, "tracks_sha256": sha256_file(tracks),
                          "video_fp": table.video_fp or fingerprint(t.video)}
        print(f"{vid}: {len(joins)} joins, solve {lk.solve_s:.1f} s")
    sample = jc.build_sample(per_video, man)
    data = jc.sample_bytes(sample)
    s = jc.parse_sample(data, man)          # checked before it is written: a bad sample never blocks a rerun
    out.mkdir(parents=True, exist_ok=True)
    jc.write_new(out / SAMPLE_NAME, data)
    jc.AnswerFile.create(out / ANSWERS_NAME, s)
    f = sample["frame"]
    # No gap length is printed: with the median and a join's gap, its stratum could be worked out.
    print(f"frame: {f['joins']} joins in {len(per_video)} videos; drew {s.n}; p* = {f['p_star']:.4f}; "
          f"sample sha256 {s.sha256[:12]}")
    return 0


# ---- check -----------------------------------------------------------------------------------------
class Videos:
    """Every video the sample shows, checked once at the start and held open (labelmoments.VideoGuard):
    under --data, the same file the sample was drawn from (fingerprint), regular timing, and its tracks file
    unchanged (sha256), with each join's boxes read and checked against it."""

    def __init__(self, sample: jc.Sample, data: Path, tracks: Path, say=print):
        from boxingvi import olympic
        from boxingvi.pose_yolo import PoseTable
        from boxingvi.video import fingerprint, probe
        self.guards, self.meta, self.boxes = {}, {}, {}
        tasks = {t.video.stem: t for t in olympic.find_tasks(data)}
        info = sample.raw["videos"]
        try:
            for vid in sorted({j.video for j in sample.joins}):
                v = info[vid]
                if vid not in tasks:
                    raise lm.Refused("a sampled video is not in the data folder")
                g = lm.VideoGuard(tasks[vid].video, data, path_check=data_path_check(data))
                self.guards[vid] = g
                if fingerprint(g.check()) != v["video_fp"]:
                    raise lm.Refused("a video differs from the one the sample was drawn from (fingerprint)")
                p = probe(g.check())
                if p.vfr or p.timing is None:
                    raise lm.Refused("a video's frame timing is irregular")
                if abs(p.fps - v["fps"]) > 1e-6 or int(p.timing.n_frames) != v["n_frames"]:
                    raise lm.Refused("a video's rate or length differs from the sample's")
                self.meta[vid] = (p.width, p.height, float(v["fps"]), int(v["n_frames"]))
                tf = tracks / f"{vid}.tracks.npz"
                if not tf.is_file() or sha256_file(tf) != v["tracks_sha256"]:
                    raise lm.Refused("a tracks file is missing or changed since the sample was drawn")
                table = PoseTable.load(tf)
                if (table.width, table.height) != (p.width, p.height):
                    raise lm.Refused("a tracks file's frame size differs from its video's")
                for k, j in enumerate(sample.joins):
                    if j.video == vid:
                        self.boxes[k] = jc.boxes_for(table, j, v["fps"])
                del table
                say(f"  checked video {len(self.guards)} of {len({j.video for j in sample.joins})}")
        except BaseException:
            self.close()
            raise

    def close(self):
        for g in self.guards.values():
            g.close()

    def max_display(self):
        sizes = [jc.display_size(w, h) for w, h, _, _ in self.meta.values()]
        return max(s[0] for s in sizes), max(s[1] for s in sizes)

    def decode(self, sample: jc.Sample, k: int) -> jc.JoinClip:
        j = sample.joins[k]
        w0, h0, fps, n = self.meta[j.video]
        start, end = jc.clip_range(j, fps, n)
        w, h = jc.display_size(w0, h0)
        n_want = end - start + 1
        buf, got = lm.decode_raw(self.guards[j.video].check(), fps, start, n_want, w, h, f"join {k + 1}",
                                 min_frames=n_want)      # the whole clip, or the U-only "cannot show" state
        b1, b2 = self.boxes[k]
        return jc.build_clip(k, buf, got, start, j, b1, b2, w0, w, h, fps, jc.step_for(fps))


class JoinApp(label_moments.App):
    """#35's window with S / D / U, the two-fragment clip and a break at 50 minutes. Shows nothing about the
    chain, the boxer, the cost or the stratum: only "join k of n"."""

    TITLE = "Linker join check"
    VALUES = jc.VALUES
    CANT_TELL = "U"
    HOLD_MS = 2500
    ITEM = "join"
    HELP = ("S = SAME person     D = DIFFERENT person (other boxer, referee, anyone)     U = can't tell\n"
            "<- / Backspace = previous     -> = next     Space = replay     E = first unanswered     "
            "Q / Esc = quit (saved)")

    def __init__(self, root, session, sample, videos, clock=None):
        self.sample, self.videos = sample, videos
        kw = {} if clock is None else {"clock": clock}
        super().__init__(root, session, None, None, f"{sample.n} joins", **kw)
        # one fixed picture area, as wide as the side-by-side still, so the window does not jump between the
        # clip and the still (a tk Label's width counts characters while it has no image, so not the label)
        w, h = videos.max_display()
        b = label_moments.BORDER
        self.frame.config(width=2 * w + jc.STILL_SEP + 2 * b, height=h + 2 * b)
        self.frame.pack_propagate(False)

    def key(self, i):
        return i

    def decode(self, k):
        return self.videos.decode(self.sample, k)

    def action_for(self, keysym):
        return KEYS.get(keysym)

    def loading_caption(self, k):
        return f"loading join {k + 1} ..."

    def hold_view(self, c):
        if "still" not in self.photos:
            ppm = b"P6 %d %d 255\n" % (c.still_width, c.height) + c.still
            self.photos["still"] = self.tk.PhotoImage(data=ppm, format="PPM")
        return (self.photos["still"], label_moments.GREY,
                f"Join {c.k + 1}: LEFT = end of the CYAN box,  RIGHT = start of the YELLOW box.  "
                "Same person?  S / D / U", "white")

    def play_view(self, c, j):
        part = c.part[j]
        if part == 1:
            return CYAN_HEX, "end of fragment 1 (cyan box)"
        if part == 2:
            return YELLOW_HEX, "start of fragment 2 (yellow box)"
        return label_moments.GREY, "gap: neither fragment is tracked"

    def frame_ms(self, c, j):
        return c.ms[j]

    def first_view(self):
        """The clip plays once before the still is held; answers count only from the still (the gate's order)."""
        self._play(0)

    def status_text(self):
        s = self.s
        cur = s.lf.values[s.i] or "-"
        done = s.n - s.lf.counts()["empty"]
        return (f"join {s.i + 1} of {s.n}    current answer: {cur}    answered {done}/{s.n}    "
                f"this sitting {s.sitting_minutes():.0f} min")

    def answer_blocked(self):
        if self.s.break_due():
            return (f"{jc.BREAK_MIN:.0f} minutes this sitting: time for a break. Q quits (everything is saved); "
                    "run the same command after the break to carry on.")
        return None


def open_sample(d: Path):
    s = jc.load_sample(d / SAMPLE_NAME, manifest())
    return s, jc.AnswerFile.load(d / ANSWERS_NAME, s)


def cmd_check(a) -> int:
    d = check_out_dir(a.dir)
    sample, answers = open_sample(d)
    lock = None if a.dry else jc.SessionLock(d / ANSWERS_NAME)
    print(f"checking {len({j.video for j in sample.joins})} videos and their tracks ...")
    try:
        videos = Videos(sample, a.data, a.tracks)
    except BaseException:
        if lock:
            lock.close()
        raise
    try:
        if a.dry:
            k = answers.first_empty() or 0
            c = videos.decode(sample, k)
            print(f"ok: {sample.n} joins, {sample.n - answers.counts()['empty']} answered; join {k + 1} decodes to "
                  f"{len(c.frames)} frames at {c.width}x{c.height}; no window opened")
            return 0
        import tkinter as tk
        session = jc.Session(answers)
        root = tk.Tk()
        app = JoinApp(root, session, sample, videos)
        root.mainloop()
        if not app.closed:
            print("the window closed without its final save (answers given before then are saved). Run `verify`, "
                  "then the same command again.", file=sys.stderr)
            return 3
        c = answers.counts()
        print(f"saved: {sample.n - c['empty']} of {sample.n} answered; minutes so far {session.minutes():.1f}")
        print("every join answered." if c["empty"] == 0 else "run the same command again to carry on.")
        return 0
    finally:
        videos.close()
        if lock:
            lock.close()


def cmd_verify(a) -> int:
    sample, answers = open_sample(check_out_dir(a.dir))
    c = answers.counts()
    print(f"{sample.n} joins: answered {sample.n - c['empty']}, empty {c['empty']}; minutes "
          f"{answers.prior_minutes():.2f}; {'COMPLETE' if c['empty'] == 0 else 'NOT complete'}")
    return 0 if c["empty"] == 0 else 1


def cmd_score(a) -> int:
    d = check_out_dir(a.dir)
    sample, answers = open_sample(d)
    if (d / RESULT_NAME).exists():
        raise lm.Refused("the result already exists: the metric is computed once")
    r = jc.score(sample, answers) | {"sample_sha256": sample.sha256, "gate": jc.GATE}
    jc.write_new(d / RESULT_NAME, (json.dumps(r, indent=1, allow_nan=False) + "\n").encode("utf-8"))
    lo, hi = r["wilson95"] or (None, None)
    print(f"verdict: {r['verdict']}")
    print(f"answers S {r['answers']['S']} D {r['answers']['D']} U {r['answers']['U']}; decided {r['decided']}")
    if r["join_error"] is not None:
        m = r["misattribution"]
        print(f"join error {r['join_error']:.3f} [{lo:.3f}, {hi:.3f}] vs p* {r['p_star']:.4f}; implied misattribution "
              f"{m['at_rate']:.3f} [{m['at_lower']:.3f}, {m['at_upper']:.3f}]")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sample", "check", "verify", "score"):
        p = sub.add_parser(name)
        p.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="where the sample and answers live (under results/)")
        if name in ("sample", "check"):
            p.add_argument("--data", type=Path, required=True)
            p.add_argument("--tracks", type=Path, required=True)
        if name == "check":
            p.add_argument("--dry", action="store_true", help="every check and one clip, no window")
    a = ap.parse_args(argv)
    try:
        return {"sample": cmd_sample, "check": cmd_check, "verify": cmd_verify, "score": cmd_score}[a.cmd](a)
    except lm.Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    except lm.SaveFailed as e:
        print(f"NOT SAVED: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
