"""Fill a glove-identity labels CSV (PR #31's gate) one moment at a time, from the BLURRED video.

  python label_moments.py LABELS.csv BLURRED.mp4            label: a window plays each moment; press a key
  python label_moments.py LABELS.csv BLURRED.mp4 --check    every check, one clip decoded, no window
  python label_moments.py LABELS.csv [BLURRED.mp4] --verify counts per value only (L / R / ? / empty)

For each row (t = 0, 5, 10 ... s) the window first holds the frame at t inside a RED frame, then plays
t - 1 s to t + 1 s and holds the red frame again, on a loop. Answer for the red frame:

  L  boxer A is on the left of the picture    R  on the right of the picture    ?  can't tell
  <- or Backspace  previous moment     ->  next moment (no change)     Space  replay
  E  first empty moment                Q or Esc  quit (everything is already saved)

An answer counts only once the moment has been on screen for MIN_DWELL_S and the previous answer's key
was released, so holding a key down never answers more than one moment. Every answer is saved at once:
quitting or a crash loses nothing, and running the same command again resumes at the first empty row. The
session times itself and adds to `# minutes:`. `# A:` (who boxer A is) is asked once, in this terminal, as
glove colours from a fixed list; the tool writes "the boxer in <colour> gloves" itself, so no name can get in.

The tool reads only the labels CSV and the blurred video. It never opens the glove method's output
(.npz / .log), so nothing it shows can lean on the method's guesses.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from boxingvi import labelmoments as lm  # noqa: E402

HOLD_MS = 1200          # how long the red frame is held, before and after the clip
MIN_DWELL_S = 0.4       # an answer key is ignored until the moment has been on screen this long
REPEAT_GAP_MS = 60      # a release followed by a press within this is key auto-repeat, not a new press
POLL_MS = 30
BORDER = 6
RED, GREY = "#e02020", "#303030"


def open_inputs(labels, video, roots=None):
    """Every check, in order: paths, the CSV's format, the video's streams and timing, then the pair.
    Returns the labels file, a VideoGuard holding the video open, and its metadata."""
    roots = roots or {}
    lpath = lm.check_labels_path(labels, roots.get("participant", lm.PARTICIPANT_ROOT))
    guard = lm.VideoGuard(video, roots.get("blurred", lm.BLURRED_ROOT))
    try:
        lf = lm.LabelFile.load(lpath)
        meta = lm.probe(guard.check())
        lm.check_pair(lf, guard.check(), meta)
    except BaseException:
        guard.close()
        raise
    return lf, guard, meta


def summary(lf) -> str:
    c = lf.counts()
    return f"{len(lf.values)} rows: L {c['L']}, R {c['R']}, ? {c['?']}, empty {c['empty']}"


class App:
    """The window. All file writes go through the Session; this class only shows clips and maps keys."""

    def __init__(self, root, session, guard, meta, title, clock=time.monotonic):
        import tkinter as tk
        self.tk = tk
        self.root, self.s, self.guard, self.meta, self.clock = root, session, guard, meta, clock
        self.cache, self.pending = {}, {}
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.clip, self.photos = None, {}
        self.after_id = self.poll_id = self.arm_id = None
        self.shown_at = None
        self.armed = True               # False from an answer until that key is released
        self.answer_code = self._press_code = None
        self.message = ""
        self.closed = False             # True only after a successful final save
        root.title(f"Label moments: {title}")
        root.configure(bg="black")
        self.frame = tk.Frame(root, bg=GREY, padx=BORDER, pady=BORDER)
        self.frame.pack()
        self.image = tk.Label(self.frame, bg="black")
        self.image.pack()
        self.caption = tk.Label(root, font=("Segoe UI", 15, "bold"), fg="white", bg="black", anchor="w")
        self.caption.pack(fill="x", padx=8)
        self.status = tk.Label(root, font=("Segoe UI", 12), fg="#dddddd", bg="black", anchor="w", justify="left")
        self.status.pack(fill="x", padx=8)
        tk.Label(root, font=("Segoe UI", 11), fg="#9a9a9a", bg="black", anchor="w", justify="left",
                 text="L = A on the LEFT of the picture     R = A on the RIGHT of the picture     "
                      "? = can't tell\n"
                      "<- / Backspace = previous     -> = next     Space = replay     "
                      "E = first empty     Q / Esc = quit (saved)").pack(fill="x", padx=8, pady=(0, 8))
        root.bind("<KeyPress>", self.on_key)
        root.bind("<KeyRelease>", self.on_release)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.show_row()

    # -- timers --
    def _cancel(self, name):
        aid = getattr(self, name)
        if aid is not None:
            self.root.after_cancel(aid)
            setattr(self, name, None)

    # -- clips --
    def _request(self, t):
        if t not in self.cache and t not in self.pending:
            self.pending[t] = self.pool.submit(lm.decode_clip, self.guard, self.meta, t)

    def _photo(self, j):
        if j not in self.photos:
            c = self.clip
            ppm = b"P6 %d %d 255\n" % (c.width, c.height) + c.frames[j]
            self.photos[j] = self.tk.PhotoImage(data=ppm, format="PPM")
        return self.photos[j]

    # -- display --
    def show_row(self):
        for name in ("after_id", "poll_id"):
            self._cancel(name)
        self.shown_at, self.clip, self.photos = None, None, {}
        self.frame.config(bg=GREY)
        self.image.config(image="")
        t = self.s.lf.times[self.s.i]
        keep = {t} | ({self.s.lf.times[self.s.i + 1]} if self.s.i + 1 < self.s.n else set())
        for old in [x for x in self.pending if x not in keep]:   # a jump leaves clips nobody will show
            self.pending.pop(old).cancel()
        for old in [x for x in self.cache if x not in keep]:
            del self.cache[old]
        self._request(t)
        self.caption.config(text=f"loading t = {t} s ...", fg="white")
        self._update_status()
        self._wait_for(t)

    def _wait_for(self, t):
        """Poll the decode without blocking the window (keys keep working while it loads)."""
        self.poll_id = None
        if t in self.pending and not self.pending[t].done():
            self.poll_id = self.root.after(POLL_MS, lambda: self._wait_for(t))
            return
        if t in self.pending:
            try:
                self.cache[t] = self.pending.pop(t).result()
            except Exception as e:     # noqa: BLE001 -- Refused, or ffmpeg failing to start: never stall
                why = str(e) if isinstance(e, lm.Refused) else "the video could not be decoded"
                self.caption.config(text=f"cannot show this moment ({why}). Press ? (can't tell) or move on.",
                                    fg="white")
                self.shown_at = self.clock()           # only ? and moves are accepted: see handle()
                return
        self.clip = self.cache[t]
        if self.s.i + 1 < self.s.n:
            self._request(self.s.lf.times[self.s.i + 1])   # decode the next moment while this one is judged
        self._hold()

    def _hold(self):
        c = self.clip
        self.frame.config(bg=RED)
        self.image.config(image=self._photo(c.centre))
        self.caption.config(text=f"RED FRAME = t {c.t} s.  Is A on the picture's L (left) or R (right)?",
                            fg=RED)
        if self.shown_at is None:
            self.shown_at = self.clock()
        self.after_id = self.root.after(HOLD_MS, lambda: self._play(0))

    def _play(self, j):
        c = self.clip
        if j >= len(c.frames):
            self._hold()
            return
        self.frame.config(bg=RED if j == c.centre else GREY)
        self.image.config(image=self._photo(j))
        a, b = c.start_f / c.fps, (c.start_f + len(c.frames) - 1) / c.fps
        self.caption.config(text=f"playing {a:.1f} s to {b:.1f} s  (red = t {c.t} s)", fg="white")
        self.after_id = self.root.after(max(1, int(1000 / c.fps)), lambda: self._play(j + 1))

    def _update_status(self):
        s = self.s
        cur = s.lf.values[s.i] or "-"
        done = s.n - s.lf.counts()["empty"]
        self.status.config(text=f"moment {s.i + 1} of {s.n}    current answer: {cur}    answered {done}/{s.n}"
                                f"    A = {s.lf.head('A')}"
                                + (f"\n{self.message}" if self.message else ""))

    # -- keys --
    def on_key(self, event):
        if self.arm_id is not None:            # a release and a press back to back: auto-repeat
            self._cancel("arm_id")
        self._press_code = getattr(event, "keycode", None)
        return self.handle(lm.action_for(event.keysym))

    def on_release(self, event):
        """Only the release of the key that gave the last answer re-arms (letting go of Shift while still
        holding / does not)."""
        if self.answer_code is not None and getattr(event, "keycode", None) != self.answer_code:
            return
        self._cancel("arm_id")
        self.arm_id = self.root.after(REPEAT_GAP_MS, self._arm)

    def _arm(self):
        self.arm_id = None
        self.armed = True

    def handle(self, action):
        if action is None:
            return None
        if action in lm.VALUES:
            if not self.armed:
                return "held"                           # the last answer's key has not been released
            if self.shown_at is None or self.clock() - self.shown_at < MIN_DWELL_S:
                return "too-soon"                       # the moment has not been on screen yet
            if self.clip is None and action != "?":
                return "not-shown"                      # nothing to judge: only "can't tell" is honest
            self.message = ""
            try:
                self.s.label(action)
            except lm.FileChanged as e:
                self.message = f"NOT SAVED: {e}. Press Q, then run the command again."
                self._update_status()
                return "file-changed"
            except lm.SaveFailed as e:
                self.message = f"NOT SAVED: {e}. Close it there, then press the key again."
                self._update_status()
                return "save-failed"
            self.armed = False
            self.answer_code = self._press_code
            if self.s.done():
                self.message = f"All {self.s.n} moments answered and saved. Q to finish, or <- to review."
        elif action == "quit":
            return self.quit()
        else:
            self.message = ""
            {"back": self.s.back, "next": self.s.forward, "first_empty": self.s.to_first_empty,
             "replay": lambda: None}[action]()
        self.show_row()
        return action

    def quit(self):
        """Save (it records the time since the last answer) and close. If the save fails the window stays
        open, so answers kept only in memory are not thrown away -- unless the file changed underneath this
        window: then its copy is the stale one, and it closes without writing."""
        try:
            self.s.save()
        except lm.FileChanged:
            self._close()
            return "closed-unsaved"
        except lm.SaveFailed as e:
            self.message = f"NOT SAVED: {e}. Close it there, then press Q again."
            self._update_status()
            return "save-failed"
        self.closed = True
        self._close()
        return "quit"

    def _close(self):
        for name in ("after_id", "poll_id", "arm_id"):
            self._cancel(name)
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def ask_description(session, read=input, write=print) -> bool:
    while session.needs_description():
        write("Who is boxer A? Type his glove colour (or two colours for two-tone gloves) from this list:")
        write("  " + ", ".join(lm.GLOVE_COLOURS))
        write("The file will say 'the boxer in <colour> gloves'. A stays the same person for the whole "
              "video. Empty line = quit.")
        try:
            text = read("A's gloves: ")
        except (EOFError, KeyboardInterrupt):
            return False
        if not text.strip():
            return False
        try:
            session.describe(text)
            write(f"  # A: {session.lf.head('A')}")
        except lm.Refused as e:
            write(f"  not accepted: {e}")
    return True


def main(argv=None, roots=None, read=input) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", type=Path, help="a gg1022-v*.labels.csv under ~/comps-participant-data/")
    ap.add_argument("video", type=Path, nargs="?", help="its BLURRED copy under ~/comps-participant-data/blurred/")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="run every check and decode one clip; no window")
    mode.add_argument("--verify", action="store_true", help="counts per value only; the video is optional")
    a = ap.parse_args(argv)
    roots = roots or {}
    guard = None
    try:
        if a.verify and a.video is None:
            lf = lm.LabelFile.load(lm.check_labels_path(a.labels, roots.get("participant", lm.PARTICIPANT_ROOT)))
        else:
            if a.video is None:
                ap.error("give the blurred video too")
            lf, guard, meta = open_inputs(a.labels, a.video, roots)
        if a.verify:
            v = lm.verify(lf)
            print(f"{summary(lf)}; A {'ok' if v['A_ok'] else 'MISSING or not allowed'}; "
                  f"minutes {v['minutes']:.2f}; {'COMPLETE' if v['complete'] else 'NOT complete'}")
            return 0 if v["complete"] else 1
        if a.check:
            k = lf.first_empty() or 0
            clip = lm.decode_clip(guard, meta, lf.times[k])
            print(f"ok: {summary(lf)}; first clip {len(clip.frames)} frames at "
                  f"{clip.width}x{clip.height}; no window opened")
            return 0
        session = lm.Session(lf)
        if not ask_description(session, read=read):
            print("quit before starting; nothing changed.")
            return 1
        import tkinter as tk
        root = tk.Tk()
        app = App(root, session, guard, meta, a.labels.name.split(".")[0])
        root.mainloop()
        if not app.closed:
            print("NOT saved from this window: the file changed underneath it (another copy of the tool?), so "
                  "it was not overwritten. Check it with --verify, then run the command again.", file=sys.stderr)
            return 3
        print(f"saved: {summary(lf)}; minutes so far {session.minutes():.1f}")
        if lf.first_empty() is None:
            print("every row answered. Check it with --verify.")
        else:
            print("run the same command again to carry on from the first empty row.")
        return 0
    except lm.Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    except lm.SaveFailed as e:
        print(f"NOT SAVED: {e}", file=sys.stderr)
        return 3
    except OSError:
        print("refused: a file could not be read or written", file=sys.stderr)
        return 2
    finally:
        if guard is not None:
            guard.close()


if __name__ == "__main__":
    sys.exit(main())
