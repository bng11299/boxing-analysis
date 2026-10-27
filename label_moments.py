"""Fill a glove-identity labels CSV (PR #31's gate) one moment at a time, from the BLURRED video.

  python label_moments.py LABELS.csv BLURRED.mp4            label: a window plays each moment; press a key
  python label_moments.py LABELS.csv BLURRED.mp4 --check    every check, one clip decoded, no window
  python label_moments.py LABELS.csv [BLURRED.mp4] --verify counts per value only (L / R / ? / empty)

For each row (t = 0, 5, 10 ... s) the window first holds the frame at t inside a RED frame, then plays
t - 1 s to t + 1 s and holds the red frame again, on a loop. Answer for the red frame:

  L  boxer A is on the left       R  boxer A is on the right       ?  can't tell
  <- or Backspace  previous moment     ->  next moment (no change)     Space  replay
  E  first empty moment                Q or Esc  quit (everything is already saved)

Every answer is saved at once, so quitting or a crash loses nothing; running the same command again
resumes at the first empty row. The session times itself and adds to `# minutes:`. `# A:` (who boxer A
is) is asked once, in this terminal, in colours and plain words only: never a name.

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
BORDER = 6
RED, GREY = "#e02020", "#303030"


def open_inputs(labels, video, roots=None):
    """Every check, in order: paths, the CSV's format, the video's streams, then the pair."""
    roots = roots or {}
    lpath = lm.check_labels_path(labels, roots.get("participant", lm.PARTICIPANT_ROOT))
    vpath = lm.check_video_path(video, roots.get("blurred", lm.BLURRED_ROOT))
    lf = lm.LabelFile.load(lpath)
    meta = lm.probe(vpath)
    lm.check_pair(lf, vpath, meta)
    return lf, vpath, meta


def summary(lf) -> str:
    c = lf.counts()
    return f"{len(lf.values)} rows: L {c['L']}, R {c['R']}, ? {c['?']}, empty {c['empty']}"


class App:
    """The window. All file writes go through the Session; this class only shows clips and maps keys."""

    def __init__(self, root, session, video, meta, title, clock=time.monotonic):
        import tkinter as tk
        self.tk = tk
        self.root, self.s, self.video, self.meta, self.clock = root, session, video, meta, clock
        self.cache, self.pending = {}, {}
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.clip, self.photos, self.after_id = None, {}, None
        self.shown_at = None
        self.message = ""
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
                 text="L = A on the LEFT     R = A on the RIGHT     ? = can't tell\n"
                      "<- / Backspace = previous     -> = next     Space = replay     "
                      "E = first empty     Q / Esc = quit (saved)").pack(fill="x", padx=8, pady=(0, 8))
        root.bind("<Key>", self.on_key)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.show_row()

    # -- clips --
    def _future(self, k):
        t = self.s.lf.times[k]
        if t not in self.cache and t not in self.pending:
            self.pending[t] = self.pool.submit(lm.decode_clip, self.video, self.meta, t)
        return t

    def _clip(self, k):
        t = self._future(k)
        if t not in self.cache:
            self.cache[t] = self.pending.pop(t).result()
        for old in [x for x in self.cache if x != t][:-2]:     # keep this clip and two others
            del self.cache[old]
        return self.cache[t]

    def _photo(self, j):
        if j not in self.photos:
            c = self.clip
            ppm = b"P6 %d %d 255\n" % (c.width, c.height) + c.frames[j]
            self.photos[j] = self.tk.PhotoImage(data=ppm, format="PPM")
        return self.photos[j]

    # -- display --
    def show_row(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.shown_at = None
        self.caption.config(text=f"loading t = {self.s.lf.times[self.s.i]} s ...")
        self._update_status()
        self.root.update_idletasks()
        try:
            self.clip = self._clip(self.s.i)
        except lm.Refused as e:
            self.clip = None
            self.caption.config(text=f"cannot show this moment: {e}. Use ? or move on.")
            self.shown_at = self.clock()
            return
        self.photos = {}
        if self.s.i + 1 < self.s.n:
            self._future(self.s.i + 1)            # decode the next moment while this one is judged
        self._hold(first=True)

    def _hold(self, first=False):
        c = self.clip
        self.frame.config(bg=RED)
        self.image.config(image=self._photo(c.centre))
        self.caption.config(text=f"RED FRAME = t {c.t} s.  Where is A: L (left) or R (right)?", fg=RED)
        if self.shown_at is None:
            self.shown_at = self.clock()
        self.after_id = self.root.after(HOLD_MS, lambda: self._play(0))

    def _play(self, j):
        c = self.clip
        if j >= len(c.frames):
            self._hold()
            return
        if j == c.centre:
            self.frame.config(bg=RED)
        else:
            self.frame.config(bg=GREY)
        self.image.config(image=self._photo(j))
        self.caption.config(text=f"playing {c.t - 1} s to {c.t + 1} s  (red = t {c.t} s)", fg="white")
        self.after_id = self.root.after(max(1, int(1000 / c.fps)), lambda: self._play(j + 1))

    def _update_status(self):
        s = self.s
        cur = s.lf.values[s.i] or "-"
        c = s.lf.counts()
        done = s.n - c["empty"]
        self.status.config(text=f"moment {s.i + 1} of {s.n}    current answer: {cur}    answered {done}/{s.n}"
                                f"    A = {s.lf.head('A')}"
                                + (f"\n{self.message}" if self.message else ""))

    # -- keys --
    def on_key(self, event):
        return self.handle(lm.action_for(event.keysym))

    def handle(self, action):
        if action is None:
            return None
        self.message = ""
        if action in lm.VALUES:
            if self.shown_at is None or self.clock() - self.shown_at < MIN_DWELL_S:
                return "too-soon"                       # the moment has not been on screen yet
            try:
                self.s.label(action)
            except lm.SaveFailed as e:
                self.message = f"NOT SAVED: {e}. Close it there, then press the key again."
                self._update_status()
                return "save-failed"
            if self.s.done():
                self.message = (f"All {self.s.n} moments answered and saved. Q to finish, "
                                "or <- to review.")
        elif action == "back":
            self.s.back()
        elif action == "next":
            self.s.forward()
        elif action == "first_empty":
            self.s.to_first_empty()
        elif action == "replay":
            pass
        elif action == "quit":
            self.quit()
            return "quit"
        self.show_row()
        return action

    def quit(self):
        try:
            self.s.save()                       # records the time spent since the last answer
        except lm.SaveFailed as e:
            print(f"NOT SAVED on quit: {e}", file=sys.stderr)
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def ask_description(session, read=input, write=print) -> bool:
    while session.needs_description():
        write("Describe boxer A in colours and plain words, e.g. 'the boxer in red gloves'.")
        write("Never a name. A stays the same person for the whole video. Empty line = quit.")
        text = read("A: ")
        if not text.strip():
            return False
        try:
            session.describe(text)
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
    try:
        if a.verify and a.video is None:
            lf = lm.LabelFile.load(lm.check_labels_path(a.labels, roots.get("participant", lm.PARTICIPANT_ROOT)))
        else:
            if a.video is None:
                ap.error("give the blurred video too")
            lf, vpath, meta = open_inputs(a.labels, a.video, roots)
        if a.verify:
            v = lm.verify(lf)
            print(f"{summary(lf)}; A {'ok' if v['A_ok'] else 'MISSING or not allowed'}; "
                  f"minutes {v['minutes']:.2f}; {'COMPLETE' if v['complete'] else 'NOT complete'}")
            return 0 if v["complete"] else 1
        if a.check:
            k = lf.first_empty() or 0
            clip = lm.decode_clip(vpath, meta, lf.times[k])
            print(f"ok: {summary(lf)}; first clip {len(clip.frames)} frames at "
                  f"{clip.width}x{clip.height}; no window opened")
            return 0
        session = lm.Session(lf)
        if not ask_description(session, read=read):
            print("quit before starting; nothing changed.")
            return 1
        import tkinter as tk
        root = tk.Tk()
        App(root, session, vpath, meta, a.labels.name.split(".")[0])
        root.mainloop()
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


if __name__ == "__main__":
    sys.exit(main())
