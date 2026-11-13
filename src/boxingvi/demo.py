"""Logic behind the local demo app (demo_app.py): importable without streamlit.

The app is a thin layer over this module, so everything that decides what is trusted lives
here and is tested without a browser (tests/test_demo.py):

  * which finished clips the gallery may show (all seven analyse_clip.OUTPUTS, regular files
    that really sit inside the results folder);
  * what an upload may be called on disk (never its own name: upload_<time>_<hex>, which
    satisfies analyse_clip._CLIP_NAME) and where (inside the uploads folder, nowhere else);
  * the argument LISTS for the two subprocesses (analyse_clip.py and ffmpeg): never a shell,
    never a string, user text only ever as one list element after a validator has passed it;
  * the participant-folder refusal, reusing boxers.PARTICIPANT_ROOTS and faceblur._under
    rather than a second definition of "participant";
  * reading the files the app shows: summary.txt verbatim, the classified CSV counted.

Own footage only: uploads and outputs live under results/ (gitignored). Nothing here makes a
network call.
"""

from __future__ import annotations

import csv
import os
import re
import secrets
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):   # analyse_clip.py sits at the repo root, the library under src/
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analyse_clip  # noqa: E402  (OUTPUTS, _CLIP_NAME, _DEVICE: one definition, not a copy)
from boxingvi import classify, faceblur, report  # noqa: E402
from boxingvi.boxers import PARTICIPANT_ROOTS  # noqa: E402

DEFAULT_RESULTS = Path("results/analyse")
DEFAULT_UPLOADS = Path("results/demo_uploads")
ANALYSE_SCRIPT = REPO_ROOT / "analyse_clip.py"
OUTPUT_SUFFIXES = tuple(suffix for suffix, _ in analyse_clip.OUTPUTS)
ALLOWED_EXTENSIONS = (".mp4", ".mov", ".m4v")
STANCES = tuple(sorted(classify.STANCES))
TABLE_COLUMNS = ("time_s", "hand", "lead_rear", "punch_type", "confidence", "hands_agree")
UPLOAD_PREFIX = "upload_"
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024   # keep equal to server.maxUploadSize in .streamlit/config.toml
MAX_TEXT_BYTES = 1024 * 1024            # summary.txt is ~2 kB; anything this big is not ours
MAX_CSV_BYTES = 20 * 1024 * 1024
MAX_CSV_ROWS = 200_000
MAX_ROUNDS_TEXT = 200
FFMPEG_TIMEOUT_S = 1800
WEB_SUFFIX = "overlay.web.mp4"
_ROUNDS_CHARS = re.compile(r"[0-9.,\- ]*")


class DemoError(ValueError):
    """A refusal the app shows to the person using it (never a traceback)."""


# ------------------------------------------------------------------ paths ---

def _resolved(p) -> Path:
    # faceblur._under compares text and resolves nothing: resolve both sides here, so a relative
    # path, `..` or a link cannot step around the check.
    return faceblur._strip_verbatim(Path(p).resolve())


def is_inside(path, root) -> bool:
    """True when `path` resolves to `root` or something under it (links and `..` followed)."""
    return faceblur._under(_resolved(path), _resolved(root))


def contained(root, rel) -> Path:
    """`rel` joined under `root`, resolved, and proven to stay strictly inside it. A name must be
    relative text: an absolute path, a drive, a NUL, or enough `..` to climb out is refused."""
    if isinstance(rel, Path):
        rel = str(rel)
    if not isinstance(rel, str) or not rel or "\x00" in rel:
        raise DemoError("not a usable path")
    if os.path.isabs(rel) or Path(rel).is_absolute() or Path(rel).drive or rel.startswith(("/", "\\")):
        raise DemoError("an absolute path is not accepted here")
    root_r = _resolved(root)
    out = _resolved(root_r / rel)
    if out == root_r or not is_inside(out, root_r):
        raise DemoError("that path leaves its folder")
    return out


def refuse_participant(path, roots=PARTICIPANT_ROOTS) -> Path:
    """Participant footage lives only under ~/comps-participant-data (boxers.PARTICIPANT_ROOTS,
    docs/commands/privacy.md); the demo never reads or writes there. Resolved first, so a relative
    path, `..` or a link cannot step around it. Returns the resolved path."""
    rp = _resolved(path)
    for root in roots:
        if is_inside(rp, root):
            raise DemoError("that folder holds participant data; the demo works on your own footage only")
    return rp


def check_folder(path, roots=PARTICIPANT_ROOTS) -> Path:
    """A folder the app is pointed at (--results-dir, --uploads-dir): not a participant folder."""
    return refuse_participant(path, roots)


# ------------------------------------------------------------------ gallery ---

@dataclass(frozen=True)
class Clip:
    name: str
    folder: Path

    def path(self, suffix: str) -> Path:
        if suffix not in OUTPUT_SUFFIXES and suffix != WEB_SUFFIX:
            raise DemoError(f"{suffix!r} is not one of the clip's outputs")
        return self.folder / f"{self.name}.{suffix}"


def is_complete(folder, name: str, roots=PARTICIPANT_ROOTS) -> bool:
    """All seven outputs exist as regular files that resolve inside `folder` (a link out of it
    does not count), and `name` is a name analyse_clip could have written."""
    if not analyse_clip._CLIP_NAME.fullmatch(name):
        return False
    try:
        base = check_folder(folder, roots)
    except DemoError:
        return False
    for suffix in OUTPUT_SUFFIXES:
        p = base / f"{name}.{suffix}"
        try:
            if not p.is_file() or not is_inside(p, base):
                return False
        except OSError:
            return False
    return True


def find_clips(folder, roots=PARTICIPANT_ROOTS) -> "list[Clip]":
    """Finished clips in `folder`, by name. A clip with any of the seven outputs missing (a run
    still going, a --no-overlay run, a half-deleted set) is left out, not shown half-empty.
    The staging folders analyse_clip makes (.<clip>.xxxx) and the cached web previews are not
    outputs and are ignored."""
    base = check_folder(folder, roots)
    if not base.is_dir():
        return []
    names = set()
    try:
        entries = list(base.iterdir())
    except OSError:
        return []
    for e in entries:
        head = e.name.partition(".")[0]
        if analyse_clip._CLIP_NAME.fullmatch(head):
            names.add(head)
    return [Clip(n, base) for n in sorted(names) if is_complete(base, n, roots)]


# ------------------------------------------------------------------ uploads ---

def upload_extension(filename) -> str:
    """The canonical extension for an uploaded file name, or a refusal. Only the final suffix is
    looked at, case-insensitively; the name itself is never kept."""
    if not isinstance(filename, str) or not filename or "\x00" in filename:
        raise DemoError("the file has no usable name")
    ext = Path(filename.replace("\\", "/")).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise DemoError(f"{ext or 'that'} is not accepted: upload a {', '.join(ALLOWED_EXTENSIONS)} video")
    return ext


def new_upload_name(now: "datetime | None" = None) -> str:
    """upload_<YYYYMMDD_HHMMSS>_<6 hex>: letters, digits and _ only, so it always satisfies
    analyse_clip._CLIP_NAME and never carries anything the uploader typed."""
    now = now or datetime.now()
    name = f"{UPLOAD_PREFIX}{now:%Y%m%d_%H%M%S}_{secrets.token_hex(3)}"
    assert analyse_clip._CLIP_NAME.fullmatch(name)
    return name


def is_upload_name(name) -> bool:
    return (isinstance(name, str) and name.startswith(UPLOAD_PREFIX)
            and analyse_clip._CLIP_NAME.fullmatch(name) is not None)


def out_dir_for(uploads_dir) -> Path:
    return Path(uploads_dir) / "out"


def save_upload(uploads_dir, filename, fileobj, max_bytes: int = MAX_UPLOAD_BYTES,
                roots=PARTICIPANT_ROOTS, now: "datetime | None" = None) -> Path:
    """Copy an uploaded file into `uploads_dir` as upload_<...><ext> and return the path.
    The extension is checked first, the size while copying (a file over the cap is deleted, not
    kept), and the file is created exclusively so an existing path is never overwritten."""
    ext = upload_extension(filename)
    folder = check_folder(uploads_dir, roots)
    folder.mkdir(parents=True, exist_ok=True)
    name = new_upload_name(now)
    dest = contained(folder, f"{name}{ext}")
    total = 0
    try:
        with open(dest, "xb") as out:
            while True:
                block = fileobj.read(1 << 20)
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    raise DemoError(f"the file is over the {max_bytes // (1024 * 1024)} MB limit")
                out.write(block)
        if total == 0:
            raise DemoError("the file is empty")
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return dest


def find_upload(uploads_dir, name: str, roots=PARTICIPANT_ROOTS) -> "Path | None":
    """The stored video for an upload name, or None. The name must be one this module made."""
    if not is_upload_name(name):
        return None
    folder = check_folder(uploads_dir, roots)
    for ext in ALLOWED_EXTENSIONS:
        p = folder / f"{name}{ext}"
        if p.is_file() and is_inside(p, folder):
            return p
    return None


def list_uploads(uploads_dir, roots=PARTICIPANT_ROOTS) -> "list[Path]":
    """Videos this app stored in `uploads_dir`, newest name first (names sort by time)."""
    folder = check_folder(uploads_dir, roots)
    if not folder.is_dir():
        return []
    found = [p for p in folder.iterdir()
             if p.is_file() and p.suffix in ALLOWED_EXTENSIONS and is_upload_name(p.stem) and is_inside(p, folder)]
    return sorted(found, key=lambda p: p.name, reverse=True)


def delete_upload(uploads_dir, name: str, roots=PARTICIPANT_ROOTS) -> "list[Path]":
    """Remove one upload and everything made from it: the video, the seven outputs, the web
    preview and any staging folder a failed run kept. Only names this module generated are
    accepted, and every path is built from that name and proven inside its folder. Returns what
    was removed."""
    if not is_upload_name(name):
        raise DemoError("only uploads made by this app can be deleted here")
    folder = check_folder(uploads_dir, roots)
    out = out_dir_for(folder)
    removed: list[Path] = []

    def drop(base: Path, fname: str) -> None:
        p = base / fname
        if not (p.is_file() or p.is_symlink()):
            return
        if not is_inside(p.parent, base):
            raise DemoError("refusing to delete outside the uploads folder")
        p.unlink()   # a link is removed as a link; its target is never touched
        removed.append(p)

    for ext in ALLOWED_EXTENSIONS:
        drop(folder, f"{name}{ext}")
    if out.is_dir() and is_inside(out, folder):
        for suffix in OUTPUT_SUFFIXES + (WEB_SUFFIX,):
            drop(out, f"{name}.{suffix}")
        for e in list(out.iterdir()):
            if e.is_dir() and not e.is_symlink() and e.name.startswith(f".{name}."):
                shutil.rmtree(e, ignore_errors=True)
                removed.append(e)
    return removed


# ------------------------------------------------------------------ analyse_clip argv ---

def validate_rounds(text) -> "str | None":
    """An optional rounds string, in the format analyse_clip takes ('0-180,240-420', seconds).
    Empty means the whole clip (None). The format, order and count are checked by report.parse_rounds
    -- the function analyse_clip calls -- here against an unbounded clip length; analyse_clip checks
    the end against the real duration. Anything outside digits . , - and spaces is refused."""
    if text is None:
        return None
    if not isinstance(text, str):
        raise DemoError("rounds must be text such as 0-180,240-420")
    text = text.strip()
    if not text:
        return None
    if len(text) > MAX_ROUNDS_TEXT or not _ROUNDS_CHARS.fullmatch(text):
        raise DemoError("rounds: use start-end in seconds, comma separated, e.g. 0-180,240-420")
    try:
        report.parse_rounds(text, 1.0e9)
    except ValueError as e:
        raise DemoError(f"rounds: {e}") from None
    return text


def validate_stance(stance) -> str:
    """There is no default: the tool never guesses a stance."""
    if stance not in STANCES:
        raise DemoError("choose the boxer's stance (orthodox or southpaw); the tool never guesses one")
    return stance


def validate_device(device, cuda_available: bool = False) -> str:
    if not isinstance(device, str) or not analyse_clip._DEVICE.fullmatch(device):
        raise DemoError("device must be cpu or cuda:N")
    if device != "cpu" and not cuda_available:
        raise DemoError("no CUDA device is available here; use cpu")
    return device


def available_devices() -> "list[str]":
    """cpu, plus cuda:N for each device torch reports. torch is imported only here, and only
    when the Analyse tab is drawn."""
    devices = ["cpu"]
    try:
        import torch
        if torch.cuda.is_available():
            devices += [f"cuda:{i}" for i in range(min(torch.cuda.device_count(), 99))]
    except Exception:   # no torch, or a driver that fails to load: cpu is always there
        pass
    return devices


def build_analyse_argv(video, stance, device, rounds, out_dir, python=None, cuda_available: bool = False) -> "list[str]":
    """The command line for analyse_clip.py as a LIST (no shell anywhere). `video` must be a file this
    app stored (an upload_ name with an accepted extension); stance, device and rounds go through
    their validators; every path is absolute so none can read as an option."""
    video = Path(video)
    if not is_upload_name(video.stem) or video.suffix not in ALLOWED_EXTENSIONS:
        raise DemoError("only a video stored by this app can be analysed")
    if _resolved(video).parent != _resolved(Path(out_dir)).parent:   # outputs go in <uploads>/out
        raise DemoError("the video is not in the uploads folder")
    validate_stance(stance)
    validate_device(device, cuda_available)
    rounds = validate_rounds(rounds)
    argv = [python or sys.executable, "-u", str(ANALYSE_SCRIPT),
            "--video", str(_resolved(video)),
            "--stance", stance,
            "--device", device,
            "--out-dir", str(_resolved(out_dir))]
    if rounds is not None:
        argv.append(f"--rounds={rounds}")
    return argv


def run_analyse(argv: "list[str]", on_line=None, popen=subprocess.Popen) -> int:
    """Run an argv built by build_analyse_argv and hand each output line (stdout and stderr merged,
    so the order is what a terminal would show) to `on_line`. Returns the exit code. No shell, no
    stdin, and YOLO_OFFLINE=1 so the pose library makes no network call of its own."""
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        raise DemoError("the command must be an argument list")
    env = dict(os.environ, PYTHONUNBUFFERED="1", YOLO_OFFLINE="1")
    proc = popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                 text=True, encoding="utf-8", errors="replace", shell=False, env=env)
    try:
        for line in proc.stdout:
            if on_line is not None:
                on_line(line.rstrip("\r\n"))
        return proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# ------------------------------------------------------------------ web preview (ffmpeg) ---

def web_preview_path(overlay) -> Path:
    overlay = Path(overlay)
    name = overlay.name
    if not name.endswith(".overlay.mp4"):
        raise DemoError("not an overlay video")
    return overlay.with_name(name[: -len("overlay.mp4")] + "overlay.web.mp4")


def ffmpeg_web_argv(ffmpeg, src, dst) -> "list[str]":
    """720p H.264 without audio, faststart so it plays while it loads: the overlays are ~1080x1920
    at 60 fps (hundreds of MB), too big to push through the browser as they are."""
    src, dst = _resolved(src), _resolved(dst)
    return [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(src),
            "-vf", "scale=-2:720", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
            "-movflags", "+faststart", "-an", str(dst)]


def preview_is_fresh(src, web) -> bool:
    try:
        return web.is_file() and web.stat().st_size > 0 and web.stat().st_mtime >= src.stat().st_mtime
    except OSError:
        return False


def find_ffmpeg() -> "str | None":
    return shutil.which("ffmpeg")


def ensure_web_preview(overlay, ffmpeg="auto", run=subprocess.run) -> "Path | None":
    """The cached <clip>.overlay.web.mp4 next to `overlay`, made on first use and again only if the
    overlay is newer. None when ffmpeg is not installed (the caller offers the original instead).
    A failed transcode raises DemoError and leaves no half-written file behind."""
    overlay = Path(overlay)
    web = web_preview_path(overlay)
    if preview_is_fresh(overlay, web):
        return web
    exe = find_ffmpeg() if ffmpeg == "auto" else ffmpeg
    if not exe:
        return None
    tmp = web.with_name(f"{web.stem}.{secrets.token_hex(4)}.tmp.mp4")
    try:
        r = run(ffmpeg_web_argv(exe, overlay, tmp), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=FFMPEG_TIMEOUT_S, shell=False)
        if r.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
            raise DemoError("ffmpeg could not make a web preview of the overlay")
        os.replace(tmp, web)
    except subprocess.TimeoutExpired:
        raise DemoError("making the web preview took too long") from None
    except OSError as e:
        raise DemoError(f"could not write the web preview ({e.strerror or e})") from None
    finally:
        tmp.unlink(missing_ok=True)
    return web


# ------------------------------------------------------------------ reading what is shown ---

def read_summary(path) -> str:
    """summary.txt exactly as written: its caveats are the honest-claims part of the project, so
    it is never shortened, reflowed or paraphrased. Size-capped only so a swapped-in giant file
    cannot fill the page."""
    p = Path(path)
    if p.stat().st_size > MAX_TEXT_BYTES:
        raise DemoError(f"{p.name} is larger than {MAX_TEXT_BYTES // 1024} kB; not shown")
    return p.read_bytes().decode("utf-8", errors="replace")


@dataclass(frozen=True)
class PunchSummary:
    total: int
    counts: "dict[str, int]"          # punch type -> punches, most common first
    rows: "list[dict[str, str]]"      # TABLE_COLUMNS only, in file order


def summarise_punches(csv_path) -> PunchSummary:
    """Count a classified.csv by punch_type and keep the display columns. The file is read with the
    csv module under size and row caps; a missing column is a refusal, not a silent gap."""
    p = Path(csv_path)
    if p.stat().st_size > MAX_CSV_BYTES:
        raise DemoError(f"{p.name} is larger than {MAX_CSV_BYTES // (1024 * 1024)} MB; not read")
    counts: Counter = Counter()
    rows: list[dict[str, str]] = []
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        missing = [c for c in TABLE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise DemoError(f"{p.name} has no {', '.join(missing)} column")
        for row in reader:
            if len(rows) >= MAX_CSV_ROWS:
                raise DemoError(f"{p.name} has more than {MAX_CSV_ROWS} punches; not read")
            rows.append({c: (row.get(c) or "") for c in TABLE_COLUMNS})
            counts[(row.get("punch_type") or "").strip() or "unnamed"] += 1
    ordered = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return PunchSummary(total=len(rows), counts=ordered, rows=rows)
