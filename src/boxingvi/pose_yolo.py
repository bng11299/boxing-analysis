"""Single-person YOLO11-pose extraction: video in, `PoseSequence` out.

Phase 1 is the solo version -- one boxer, one camera -- so each frame keeps ONE
person. Two-boxer tracking (ByteTrack) is Phase 2 and deliberately not here.

Output is COCO-17, the same joint order as BoxingVI's AlphaPose skeletons, so a
`PoseSequence` converts to a `data.Track` and every downstream tool (wrist-speed
rule, features, classifier) accepts it unchanged. The AlphaPose -> YOLO11-pose
accuracy loss is a separate measurement (checklist W7); nothing here assumes it
is zero.

Trust boundaries: the video is untrusted (see `video.py`); the model checkpoint
is a pickle, so it is refused unless its sha256 is pinned and matches, computed on
the exact bytes read. Ultralytics reopens the file afterwards, so a swap between
the check and the load is possible for an attacker who can already write to the
weights directory -- keep that directory user-private. The pose file is untrusted
too: `PoseSequence.load` reads every array header and refuses the file before any
array is allocated (see `_read_pose_npz`). Ultralytics is imported lazily so the
rest of the repo (and the tests) need neither it nor a GPU.
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
import zipfile
import zlib

# The pipeline runs offline. Set at import as well as before loading, in case ultralytics is imported elsewhere first.
os.environ["YOLO_OFFLINE"] = "1"
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import N_JOINTS, Track
from .npyread import read_data, read_header
from .punchfind import L_SHOULDER, L_WRIST, R_SHOULDER, R_WRIST
from .video import VideoInfo, fingerprint, iter_frames, probe

USABLE_MIN_CONF = 0.3   # same default as the overlay, so "drawn" and "usable" agree
USABLE_MIN_JOINTS = 8   # of 17; below this the pose is a guess, not a measurement
# The joints the wrist-speed rule differences. Eight solid joints say nothing if
# the wrists are among the guessed ones: a guessed wrist that jumps reads as a punch.
# Hips are read too (`body_scale`), but only through a median over the whole
# track, which a few guessed hips cannot move, so they are not required per frame.
# Requiring BOTH wrists also drops frames where only the far one is hidden (common
# in a three-quarter view); per-hand masking in the punch finder would recover the
# visible hand's punches, and `Track.conf` is carried so that it can.
REQUIRED_JOINTS = (L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST)
PERSON_IOU_KEEP = 0.3   # a box overlapping last frame's by this much is "the same boxer"
CONTINUITY_BONUS = 2.0  # ... and its area counts double, so a bigger bystander can't steal the track
# ~9 h at 60 fps, the same bound as video.MAX_TIMING_PACKETS: a pose has one row per
# frame. That bounds what a pose file may cost to load: ~410 MB for xy + conf at
# float32 (what `extract` writes), ~820 MB at float64. The longest real one (V4) is
# 35,363 frames.
MAX_POSE_FRAMES = 2_000_000
# A tracks table has one row per (frame, person). The same 2M bound caps what loading
# one costs: ~440 MB for box + xy + conf at float32 (what `extract_tracks` writes),
# ~880 MB at float64; it covers ~2.8 h of 50 fps footage with four people tracked.
MAX_TABLE_ROWS = 2_000_000
MAX_POSE_TEXT = 4096   # characters in `source` / `model`; a 'U' scalar's size is in its header
# A pose file has at most 9 members and a tracks table 12; 32 leaves room without letting a file of
# millions of empty members through. 64 KiB of central directory is ~1,400 entries at most.
MAX_NPZ_MEMBERS = 32
MAX_ZIP_DIRECTORY = 64 * 1024
_EOCD = struct.Struct("<4s4H2LH")   # end-of-central-directory record, 22 bytes


_FP = re.compile(r"[0-9a-f]{64}")


def _check_fp(fp) -> None:
    """A stored video fingerprint is empty (older files, never bound) or a sha256 hex digest."""
    if not isinstance(fp, str) or (fp and not _FP.fullmatch(fp)):
        raise ValueError("video_fp must be empty or a 64-character lowercase sha256 hex digest")


_INT64_MAX = float(2 ** 53)       # beyond this a float cannot hold every whole number exactly


def _field(z, key: str) -> np.ndarray:
    try:
        return z[key]
    except KeyError:
        raise ValueError(f"not a pose file: the field {key!r} is missing") from None


def _whole_array(z, key: str, lo: int = 0) -> np.ndarray:
    """A stored integer array (frame numbers, track ids), VALIDATED BEFORE any conversion: an
    integer dtype, or a float dtype whose every value is finite and a whole number. A stored
    -0.5 or 1.9 is refused, never truncated to 0 or 1 (`astype` would have done exactly that
    and produced a plausible, wrong identity). Values must be >= `lo`."""
    a = _field(z, key)
    if a.ndim != 1:
        raise ValueError(f"{key} must be a 1-D array")
    if a.dtype.kind in "iu":
        if a.dtype.kind == "u" and a.size and int(a.max()) > np.iinfo(np.int64).max:
            raise ValueError(f"{key} holds a value beyond the integer range")
    elif a.dtype.kind == "f":
        if not np.isfinite(a).all():
            raise ValueError(f"{key} must be finite")
        if (a != np.floor(a)).any():
            raise ValueError(f"{key} must be whole numbers, not fractions")
        if a.size and np.abs(a).max() > _INT64_MAX:
            raise ValueError(f"{key} holds a value beyond the integer range")
    else:
        raise ValueError(f"{key} must be an integer array, got dtype {a.dtype}")
    out = a.astype(np.int64)
    if out.size and out.min() < lo:
        raise ValueError("a negative track id: -1 means unidentified and is never a person"
                         if key == "track_id" else f"{key} holds a value below {lo}")
    return out


def _float_array(z, key: str) -> np.ndarray:
    a = _field(z, key)
    if a.dtype.kind != "f":
        raise ValueError(f"{key} must be a floating-point array, got dtype {a.dtype}")
    return a


def _text(z, key: str) -> str:
    """A stored string (source, model, video_fp): a 0-d unicode array, checked BEFORE str()
    (a number or a vector would otherwise become plausible-looking text)."""
    a = _field(z, key)
    if a.shape != () or a.dtype.kind != "U":
        raise ValueError(f"{key} must be a single string, got shape {a.shape} dtype {a.dtype}")
    return str(a.item())


def _bool_vector(z, key: str) -> np.ndarray:
    a = _field(z, key)
    if a.ndim != 1 or a.dtype.kind != "b":
        raise ValueError(f"{key} must be a 1-D bool array, got shape {a.shape} dtype {a.dtype}")
    return a


def _scalar(z, key: str, whole: bool, lo: float = 0.0) -> "int | float":
    """A stored scalar, validated before conversion: one finite number, whole when `whole`
    (a width of 640.7 is refused, not read as 640), and >= `lo`."""
    a = _field(z, key)
    if a.shape != () or a.dtype.kind not in "iuf":
        raise ValueError(f"{key} must be a single number")
    x = a.item()
    if isinstance(x, float) and not np.isfinite(x):
        raise ValueError(f"{key} must be finite")
    if whole and x != int(x):
        raise ValueError(f"{key} must be a whole number, got {x!r}")
    if not x >= lo:
        raise ValueError(f"{key} must be at least {lo:g}, got {x!r}")
    return int(x) if whole else float(x)


def verify_binding(pose, video, check_name: bool = True) -> bool:
    """Is this pose table / sequence the one extracted from THIS video file?

    Size, frame rate and frame count agree between unrelated recordings (a camera's fixed
    settings), so they identify nothing. Two checks do: the file NAME recorded at extraction
    (`source`), and, when the table carries one, the content fingerprint (`video.fingerprint`).
    Raises ValueError on a mismatch of either. Returns True when the content fingerprint was
    checked and matched, False when the table has none (files written before it existed):
    those are accepted on the name alone and the caller should report them as unbound.
    `check_name=False` skips the name (a PoseSequence in a hand-made test fixture)."""
    video = Path(video)
    if check_name and pose.source != video.name:
        raise ValueError(f"the pose table was extracted from {pose.source!r}, not {video.name!r}")
    if not pose.video_fp:
        return False
    if pose.video_fp != fingerprint(video):
        raise ValueError(f"the pose table's video fingerprint does not match {video.name!r}: "
                         "it was extracted from a different recording")
    return True


@dataclass
class PoseSequence:
    """Per-frame pose for EVERY frame of the video; undetected frames are NaN.

    `xy` is (F, 17, 2) in pixels, `conf` (F, 17), `detected` (F,) bool. Keeping
    the gaps (rather than dropping frames) keeps index i == video frame i, which
    the overlay and any label alignment rely on.
    """

    xy: np.ndarray
    conf: np.ndarray
    detected: np.ndarray
    fps: float
    width: int
    height: int
    source: str = ""
    model: str = ""
    video_fp: str = ""     # video.fingerprint of the source file; "" = not recorded

    def __post_init__(self) -> None:
        _check_fp(self.video_fp)
        if self.detected.ndim != 1:
            raise ValueError(f"detected must be a 1-D per-frame mask, got shape {self.detected.shape}")
        f = len(self.detected)
        if self.xy.shape != (f, N_JOINTS, 2) or self.conf.shape != (f, N_JOINTS):
            raise ValueError(f"shape mismatch: xy {self.xy.shape}, conf {self.conf.shape}, "
                             f"detected {self.detected.shape}")
        if self.detected.dtype != bool:
            raise ValueError("detected must be a bool array")
        if not (np.isfinite(self.fps) and self.fps > 0):
            raise ValueError(f"fps must be positive, got {self.fps!r}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if not np.all(np.isfinite(self.xy[self.detected])):
            raise ValueError("a frame marked detected has non-finite coordinates")
        if not np.all((self.conf >= 0) & (self.conf <= 1)):  # a loaded file is untrusted; NaN fails
            raise ValueError("joint confidence must lie in [0, 1]")

    def __len__(self) -> int:
        return len(self.detected)

    def usable(self, min_conf: float = USABLE_MIN_CONF,
               min_joints: int = USABLE_MIN_JOINTS,
               required: "tuple[int, ...]" = REQUIRED_JOINTS) -> np.ndarray:
        """Frames worth trusting: detected, at least `min_joints` joints at or above
        `min_conf`, AND every joint in `required` (both shoulders and wrists) among
        them. `detected` only says a box was found -- on the first real clip 15% of
        detected frames had under 8 solid joints (a blurred lens, a partial body).
        Downstream code should gate on this, not on `detected`."""
        solid = self.conf >= min_conf
        return (self.detected & (solid.sum(axis=1) >= min_joints)
                & solid[:, list(required)].all(axis=1))

    def to_track(self, usable_only: bool = True, min_conf: float = USABLE_MIN_CONF) -> Track:
        """Usable frames by default (deny by default); `usable_only=False` keeps every
        detected frame. Gaps survive as jumps in `frame_ids`; confidence travels along.
        Pass the same `min_conf` the run reported with, or the two sets differ."""
        ids = np.flatnonzero(self.usable(min_conf) if usable_only else self.detected)
        return Track(frame_ids=ids, xy=self.xy[ids].astype(np.float64), fps=self.fps,
                     source=self.source, conf=self.conf[ids].astype(np.float64))

    def save(self, path) -> None:
        np.savez_compressed(path, xy=self.xy, conf=self.conf, detected=self.detected,
                            fps=self.fps, width=self.width, height=self.height,
                            source=self.source, model=self.model, video_fp=self.video_fp)

    @classmethod
    def load(cls, path) -> "PoseSequence":
        """Load a file written by `save`. The file is untrusted: a ~2 KB npz whose
        array headers claim 10^12 frames made `np.load` allocate before any check ran
        (MemoryError; a smaller claim silently took hundreds of MB). Every header is
        now read and checked first -- keys, zip compression method, dtypes, agreeing
        shapes, the frame cap -- through a byte cap, and only then is any array
        allocated. Never unpickles. A file that cannot be opened raises OSError;
        anything wrong inside it, or a read that fails part-way, ValueError."""
        p = Path(path)
        with open(p, "rb") as fh:
            try:
                z = _read_pose_npz(fh, p.name)
            except (zipfile.BadZipFile, zlib.error, EOFError, RuntimeError,
                    OSError) as e:  # zipfile's ways of saying "corrupt"; encrypted is Runtime
                raise ValueError(f"{p.name}: not a readable pose file ({type(e).__name__})") from e
        # The headers were bounded above; these are the same field checks as
        # `PoseTable.load` (kind and shape before conversion), then the constructor's.
        fps = _scalar(z, "fps", whole=False)
        if not fps > 0:
            raise ValueError(f"fps must be positive, got {fps!r}")
        return cls(xy=_float_array(z, "xy"), conf=_float_array(z, "conf"),
                   detected=_bool_vector(z, "detected"), fps=fps,
                   width=_scalar(z, "width", whole=True, lo=1),
                   height=_scalar(z, "height", whole=True, lo=1),
                   source=_text(z, "source"), model=_text(z, "model"),
                   video_fp=_text(z, "video_fp") if "video_fp" in z else "")


# What `save` writes, and the dtype kinds accepted for each: float coordinates and
# confidence, a bool mask, a numeric fps, integer sizes, unicode text. Everything
# else -- object (pickle), structured, bytes, complex -- is refused from the header.
_POSE_KINDS = {"xy": "f", "conf": "f", "detected": "b", "fps": "fiu",
               "width": "iu", "height": "iu", "source": "U", "model": "U",
               "video_fp": "U"}
# Members a pose file may lack: files written before fingerprints have no video_fp,
# and load as "" (never bound to a video).
_POSE_OPTIONAL = frozenset({"video_fp"})


# How `np.savez` (stored) and `np.savez_compressed` (deflate) write members. zipfile
# decompresses deflate in bounded steps, but bz2 and lzma with no output limit: a
# 2 KB bz2 member reached 584 MB just to yield the 8-byte .npy magic.
_POSE_COMPRESSION = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
# Deflate's ceiling (258 bytes from a 2-bit code, with block overhead). A header that
# promises more data than its zip entry could expand to is refused before `np.empty`:
# without this, a 1.3 KB file claiming exactly the frame cap at float64 allocated
# 544 MB and only then found the data missing.
_DEFLATE_MAX_RATIO = 1032


# The same for a tracks table (`PoseTable.save`). Whole-number fields also accept
# floats because `_whole_array` / `_scalar` refuse any fraction before converting.
_TABLE_KINDS = {"frame": "iuf", "track_id": "iuf", "box": "f", "xy": "f", "conf": "f",
                "n_frames": "iuf", "fps": "fiu", "width": "iuf", "height": "iuf",
                "source": "U", "model": "U", "video_fp": "U"}


def _pose_shapes(f: int) -> "dict[str, tuple]":
    return {"xy": (f, N_JOINTS, 2), "conf": (f, N_JOINTS), "detected": (f,)}


def _table_shapes(n: int) -> "dict[str, tuple]":
    return {"frame": (n,), "track_id": (n,), "box": (n, 4), "xy": (n, N_JOINTS, 2),
            "conf": (n, N_JOINTS)}


def _npy_header(member, key: str, name: str, kinds: "dict[str, str]" = _POSE_KINDS):
    """(shape, fortran_order, dtype) from a member's .npy header, read through a cap
    (`npyread.read_header`), with the dtype checked against what `save` writes for
    `key`. Leaves `member` positioned at the first byte of array data."""
    shape, fortran, dtype = read_header(member, f"{name}: {key}")
    max_item = 4 * MAX_POSE_TEXT if dtype.kind == "U" else 8  # 'U' is 4 bytes a character
    if (dtype.kind not in kinds[key] or dtype.hasobject or dtype.fields is not None
            or dtype.subdtype is not None or dtype.itemsize > max_item):
        raise ValueError(f"{name}: {key} has dtype {dtype.str!r}; its {key} "
                         f"must be of kind {kinds[key]!r}")
    return shape, fortran, dtype


def _check_zip_directory(fh, name: str, what: str) -> None:
    """Bound the zip's central directory BEFORE `zipfile.ZipFile` reads it: ZipFile reads
    as many bytes as the end-of-central-directory record declares and builds one object
    per entry, so a file of millions of tiny members costs memory no array cap sees.
    np.savez writes the end record last, with no archive comment, no zip64 directory
    (it needs > 65,535 members or > 2 GiB, far past the row caps) and one disk."""
    size = os.fstat(fh.fileno()).st_size
    if size < _EOCD.size:
        raise zipfile.BadZipFile("too short to be a zip")
    fh.seek(size - _EOCD.size)
    (sig, disk, cd_disk, n_disk, n_total, cd_size, cd_offset,
     comment) = _EOCD.unpack(fh.read(_EOCD.size))
    if sig != b"PK\x05\x06" or comment:
        raise zipfile.BadZipFile("no end-of-directory record at the end of the file")
    zip64 = False
    if size >= _EOCD.size + 20:
        fh.seek(size - _EOCD.size - 20)
        zip64 = fh.read(4) == b"PK\x06\x07"
    if (zip64 or 0xFFFF in (n_disk, n_total) or 0xFFFFFFFF in (cd_size, cd_offset)
            or disk or cd_disk or n_disk != n_total):
        raise ValueError(f"{name}: zip64 or multi-disk directory; a {what} written by "
                         f"np.savez has neither")
    if n_total > MAX_NPZ_MEMBERS or cd_size > MAX_ZIP_DIRECTORY:
        raise ValueError(f"{name}: zip directory of {n_total:,} members in {cd_size:,} bytes; "
                         f"a {what} has at most {MAX_NPZ_MEMBERS} members in "
                         f"{MAX_ZIP_DIRECTORY:,} bytes")
    fh.seek(0)


def _read_pose_npz(fh, name: str, kinds: "dict[str, str]" = _POSE_KINDS,
                   shapes=_pose_shapes, row_key: str = "detected",
                   cap: "int | None" = None, what: str = "pose file",
                   unit: str = "frames") -> "dict[str, np.ndarray]":
    """Validate every member's .npy header, then read the arrays -- in that order,
    from one open file, so what was checked is what is read. `row_key`'s 1-D length
    is the row count; `shapes(rows)` gives each array's expected shape (a key not in
    it is a 0-d scalar), and `cap` (default MAX_POSE_FRAMES, read at call time) bounds
    the rows before anything is allocated."""
    cap = MAX_POSE_FRAMES if cap is None else cap
    _check_zip_directory(fh, name, what)
    with zipfile.ZipFile(fh) as zf:
        listed = zf.namelist()
        if len(listed) > MAX_NPZ_MEMBERS or len(set(listed)) != len(listed):
            raise ValueError(f"{name}: {len(listed)} zip members ({len(set(listed))} distinct names); "
                             f"a {what} has at most {MAX_NPZ_MEMBERS}, each named once")
        names = set(listed)
        missing = [k for k in kinds
                   if k + ".npy" not in names and k not in _POSE_OPTIONAL]
        if missing:
            raise ValueError(f"{name}: not a {what}, missing {', '.join(missing)}")
        keys = [k for k in kinds if k + ".npy" in names]
        for k in keys:  # before any member is opened: opening one decompresses
            if zf.getinfo(k + ".npy").compress_type not in _POSE_COMPRESSION:
                raise ValueError(f"{name}: {k} uses zip compression method "
                                 f"{zf.getinfo(k + '.npy').compress_type}; a {what} is "
                                 f"stored or deflated (np.savez / np.savez_compressed)")
        heads = {}
        for k in keys:
            with zf.open(k + ".npy") as m:
                heads[k] = _npy_header(m, k, name, kinds)
        dshape = heads[row_key][0]
        if len(dshape) != 1 or type(dshape[0]) is not int or dshape[0] < 0:  # (True,) parses
            raise ValueError(f"{name}: {row_key} has shape {dshape}, expected ({unit},)")
        f = dshape[0]
        want = shapes(f)
        for k in keys:
            if heads[k][0] != want.get(k, ()):  # tuple == compares lengths and every dim
                raise ValueError(f"{name}: {k} has shape {heads[k][0]}, expected "
                                 f"{want.get(k, ())} for {f} {unit}")
        if f > cap:  # before anything is allocated
            raise ValueError(f"{name}: claims {f:,} {unit}; the cap is {cap:,}")
        size = os.fstat(fh.fileno()).st_size
        for k in keys:  # what the member could hold at most, before allocating it
            info = zf.getinfo(k + ".npy")
            packed = min(info.compress_size, size)  # a declared size can lie; the file can't
            room = min(info.file_size, packed if info.compress_type == zipfile.ZIP_STORED
                       else packed * _DEFLATE_MAX_RATIO)
            need = heads[k][2].itemsize * int(np.prod(heads[k][0], dtype=np.int64))
            if need > room:
                raise ValueError(f"{name}: {k} promises {need:,} bytes of data; its "
                                 f"{packed:,}-byte zip entry can hold at most {room:,}")
        out = {}
        for k in keys:
            with zf.open(k + ".npy") as m:
                if _npy_header(m, k, name, kinds) != heads[k]:  # rewritten since validated
                    raise ValueError(f"{name}: {k} changed while being read")
                out[k] = read_data(m, *heads[k], f"{name}: {k}")
        return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def select_person(boxes: np.ndarray, prev_box: "np.ndarray | None" = None) -> "int | None":
    """Index of the boxer among this frame's detections, or None.

    Largest box wins, with a bonus for continuity with last frame's pick. The
    gym has bystanders, and the nearest one is often the biggest box; continuity
    is what stops the track hopping to them mid-round. Non-finite or degenerate
    boxes are ignored rather than trusted.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    best, best_score = None, 0.0
    for i, b in enumerate(boxes):
        if not (np.all(np.isfinite(b)) and b[2] > b[0] and b[3] > b[1]):
            continue  # inverted corners multiply to a POSITIVE area, so test the sides
        area = (b[2] - b[0]) * (b[3] - b[1])
        score = area * (CONTINUITY_BONUS if prev_box is not None
                        and _iou(b, prev_box) >= PERSON_IOU_KEEP else 1.0)
        if score > best_score:
            best, best_score = i, score
    return best


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_yolo(weights, sha256: "str | None", device: str = "cpu"):
    """Load a YOLO pose checkpoint, deny-by-default on integrity."""
    p = Path(weights).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"no weights at {p}")
    if not sha256:
        raise ValueError("refusing a checkpoint with no pinned sha256 "
                         "(compute one with: python cli/run_ingest.py --hash-weights <file>)")
    if sha256_file(p) != sha256.lower():
        raise ValueError(f"{p.name} does not match the pinned sha256; not loading it")
    # The pipeline runs offline: without this, ultralytics resolves DNS at import and sends usage events while
    # it predicts. Read once, when ultralytics is first imported, so it is set before that import.
    os.environ["YOLO_OFFLINE"] = "1"
    try:
        from ultralytics import YOLO  # lazy: heavy, and not needed to test this module
    except ImportError as e:
        raise RuntimeError("ultralytics is not installed (pip install -r requirements-pose.txt)") from e
    model = YOLO(str(p))
    model.to(device)
    return model


def yolo_predictor(model, conf: float = 0.25, device: str = "cpu", missing_conf: float = 1.0):
    """Wrap a loaded model as predict(frame_rgb) -> (boxes, kp_xy, kp_conf).

    `missing_conf` is the confidence given to every keypoint when the model reports
    none. The face blur passes 0.0: an unconfident face falls back to more blur."""

    def predict(frame_rgb: np.ndarray):
        # Ultralytics treats a bare ndarray as BGR (OpenCV convention); ours is RGB.
        r = model.predict(frame_rgb[..., ::-1], conf=conf, device=device, verbose=False)[0]
        if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
            return np.zeros((0, 4)), np.zeros((0, N_JOINTS, 2)), np.zeros((0, N_JOINTS))
        boxes = r.boxes.xyxy.cpu().numpy()
        kxy = r.keypoints.xy.cpu().numpy()
        kc = (r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None
              else np.full(kxy.shape[:2], missing_conf))
        return boxes, kxy, kc

    return predict


@dataclass
class PoseTable:
    """EVERY tracked person in every frame, one row per (frame, track id).

    Two-boxer footage holds more people than boxers (a referee in the ring,
    coaches at the side), and ByteTrack can split one person into several ids
    across a clinch. So nothing here decides who the boxers are: that is decided
    later, from labels (Olympic Boxing) or by the caller. Coordinates in pixels.
    """

    frame: np.ndarray       # (N,) int64 video frame index
    track_id: np.ndarray    # (N,) int64 ByteTrack id
    box: np.ndarray         # (N, 4) x1, y1, x2, y2
    xy: np.ndarray          # (N, 17, 2)
    conf: np.ndarray        # (N, 17)
    n_frames: int
    fps: float
    width: int
    height: int
    source: str = ""
    model: str = ""
    video_fp: str = ""      # video.fingerprint of the source file; "" = not recorded

    def __post_init__(self) -> None:
        _check_fp(self.video_fp)
        n = len(self.frame)
        if (self.track_id.shape != (n,) or self.box.shape != (n, 4)
                or self.xy.shape != (n, N_JOINTS, 2) or self.conf.shape != (n, N_JOINTS)):
            raise ValueError("pose table arrays do not have matching shapes")
        if self.n_frames < 1 or not (np.isfinite(self.fps) and self.fps > 0) or self.width <= 0 or self.height <= 0:
            raise ValueError("pose table needs positive frame count, fps and size")
        if n and (self.frame.min() < 0 or self.frame.max() >= self.n_frames):
            raise ValueError("a row's frame is outside the video")
        if not (np.isfinite(self.xy).all() and np.isfinite(self.box).all()):
            raise ValueError("pose table contains non-finite coordinates")
        if not np.all((self.conf >= 0) & (self.conf <= 1)):
            raise ValueError("joint confidence must lie in [0, 1]")
        if n and self.track_id.min() < 0:      # -1 is "no id yet" and is dropped at extraction
            raise ValueError("a negative track id: -1 means unidentified and is never a person")
        if len(set(zip(self.frame.tolist(), self.track_id.tolist()))) != n:
            raise ValueError("one track id appears twice in one frame")

    def ids(self) -> "list[int]":
        """Track ids, most-seen first."""
        u, c = np.unique(self.track_id, return_counts=True)
        return [int(i) for i in u[np.argsort(-c, kind="stable")]]

    def track(self, tid: int) -> Track:
        """One id as a Track (with confidence), in frame order."""
        rows = np.flatnonzero(self.track_id == tid)
        rows = rows[np.argsort(self.frame[rows], kind="stable")]
        return Track(frame_ids=self.frame[rows].astype(np.int64), xy=self.xy[rows].astype(np.float64),
                     fps=self.fps, source=f"{self.source}#{tid}", conf=self.conf[rows].astype(np.float64))

    def save(self, path) -> None:
        np.savez_compressed(path, frame=self.frame, track_id=self.track_id, box=self.box, xy=self.xy,
                            conf=self.conf, n_frames=self.n_frames, fps=self.fps, width=self.width,
                            height=self.height, source=self.source, model=self.model, video_fp=self.video_fp)

    @classmethod
    def load(cls, path) -> "PoseTable":
        """Load a table. A file is untrusted, so each stored field is read through a helper that
        checks its raw dtype and shape BEFORE converting it, and a missing field is a ValueError:
        frame and track_id are 1-D integer (or finite whole-float) arrays in range, box / xy / conf
        are float arrays, n_frames / width / height are 0-d whole numbers >= 1, fps is a 0-d finite
        number > 0, source / model / video_fp are 0-d strings. A fraction is refused, not truncated
        into a different frame or id. Cross-field consistency (matching lengths, frames inside the
        video, finite coordinates, confidence in [0, 1], no repeated id in a frame) is then checked
        by the constructor.

        As in `PoseSequence.load`, every member's .npy header is read and checked first
        (keys, zip compression method, dtype kinds, agreeing shapes, MAX_TABLE_ROWS, size
        against the zip entry) and only then is anything allocated: a few-KB file whose
        headers claim huge shapes is refused instead of allocating gigabytes. Never
        unpickles. A file that cannot be opened raises OSError; anything wrong inside
        it, ValueError."""
        p = Path(path)
        with open(p, "rb") as fh:
            try:
                z = _read_pose_npz(fh, p.name, _TABLE_KINDS, _table_shapes, "frame",
                                   MAX_TABLE_ROWS, "tracks table", "rows")
            except (zipfile.BadZipFile, zlib.error, EOFError, RuntimeError, OSError) as e:
                raise ValueError(f"{p.name}: not a readable tracks table ({type(e).__name__})") from e
        frame, track_id = _whole_array(z, "frame"), _whole_array(z, "track_id")
        fps = _scalar(z, "fps", whole=False)
        if not fps > 0:
            raise ValueError(f"fps must be positive, got {fps!r}")
        return cls(frame=frame, track_id=track_id,
                   box=_float_array(z, "box"), xy=_float_array(z, "xy"), conf=_float_array(z, "conf"),
                   n_frames=_scalar(z, "n_frames", whole=True, lo=1), fps=fps,
                   width=_scalar(z, "width", whole=True, lo=1), height=_scalar(z, "height", whole=True, lo=1),
                   source=_text(z, "source"), model=_text(z, "model"),
                   video_fp=_text(z, "video_fp") if "video_fp" in z else "")


TRACKER_CONFIG = Path(__file__).resolve().parents[2] / "config" / "bytetrack_50fps.yaml"


def yolo_tracker(model, conf: float = 0.1, device: str = "cpu", tracker: "str | Path" = TRACKER_CONFIG):
    """Wrap a loaded model as track(frame_rgb) -> (boxes, kp_xy, kp_conf, ids), keeping
    ByteTrack's state between calls (`persist=True`). That state is never reset, so
    use ONE freshly loaded model per video; run_ingest.py processes one video per run.
    Detections without an id come back as -1 and are dropped by `extract_tracks`."""
    tracker = str(tracker)

    def track(frame_rgb: np.ndarray):
        r = model.track(frame_rgb[..., ::-1], persist=True, tracker=tracker, conf=conf,
                        device=device, verbose=False)[0]
        if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
            return np.zeros((0, 4)), np.zeros((0, N_JOINTS, 2)), np.zeros((0, N_JOINTS)), np.zeros(0, np.int64)
        boxes = r.boxes.xyxy.cpu().numpy()
        kxy = r.keypoints.xy.cpu().numpy()
        kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kxy.shape[:2])
        ids = (r.boxes.id.cpu().numpy().astype(np.int64) if r.boxes.id is not None
               else np.full(len(boxes), -1, np.int64))
        return boxes, kxy, kc, ids

    return track


def extract_tracks(video, track, model_name: str = "", max_frames: "int | None" = None,
                   info: "VideoInfo | None" = None) -> PoseTable:
    """Run `track` over every frame; keep every person who has a track id and a
    complete, finite pose. Nothing is chosen here (see PoseTable)."""
    video = Path(video).resolve()
    info = info or probe(video)
    rows = {"frame": [], "track_id": [], "box": [], "xy": [], "conf": []}
    n = 0
    for i, frame in enumerate(iter_frames(video, info, max_frames)):
        n = i + 1
        boxes, kxy, kc, ids = track(frame)
        boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
        kxy, kc, ids = np.asarray(kxy, np.float64), np.asarray(kc, np.float64), np.asarray(ids)
        if not (len(kxy) == len(kc) == len(ids) == len(boxes)):
            raise ValueError(f"frame {i}: the tracker returned {len(boxes)} boxes but "
                             f"{len(kxy)} poses and {len(ids)} ids")
        seen = set()
        for b, p, c, t in zip(boxes, kxy, kc, ids):
            t = int(t)
            if t < 0 or t in seen or p.shape != (N_JOINTS, 2) or not (np.isfinite(p).all() and np.isfinite(b).all()):
                continue  # no id yet, a duplicate id, or a broken pose: never a guessed row
            seen.add(t)
            rows["frame"].append(i); rows["track_id"].append(t); rows["box"].append(b)
            rows["xy"].append(p); rows["conf"].append(np.clip(np.nan_to_num(c), 0.0, 1.0))
    if n == 0:
        raise ValueError(f"{video.name}: no frames could be decoded")
    k = len(rows["frame"])
    return PoseTable(np.asarray(rows["frame"], np.int64), np.asarray(rows["track_id"], np.int64),
                     np.asarray(rows["box"], np.float32).reshape(k, 4),
                     np.asarray(rows["xy"], np.float32).reshape(k, N_JOINTS, 2),
                     np.asarray(rows["conf"], np.float32).reshape(k, N_JOINTS),
                     n, info.fps, info.width, info.height, source=video.name, model=model_name)


def extract(video, predict, model_name: str = "", max_frames: "int | None" = None,
            info: "VideoInfo | None" = None) -> PoseSequence:
    """Run `predict` over every frame and keep one person per frame. `info` is
    `probe(video)`, passed in to spare a second packet scan."""
    video = Path(video).resolve()
    info = info or probe(video)
    xs, cs, det = [], [], []
    prev = None
    for frame in iter_frames(video, info, max_frames):
        boxes, kxy, kc = predict(frame)
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        k = select_person(boxes, prev)
        if k is None or np.asarray(kxy)[k].shape != (N_JOINTS, 2) \
                or not np.all(np.isfinite(np.asarray(kxy)[k])):
            xs.append(np.full((N_JOINTS, 2), np.nan, np.float32))
            cs.append(np.zeros(N_JOINTS, np.float32))
            det.append(False)
            continue  # prev_box kept: a one-frame dropout shouldn't reset the track
        prev = boxes[k]
        xs.append(np.asarray(kxy[k], np.float32))
        cs.append(np.clip(np.nan_to_num(np.asarray(kc[k], np.float32)), 0.0, 1.0))
        det.append(True)
    if not xs:
        raise ValueError(f"{video.name}: no frames could be decoded")
    return PoseSequence(np.stack(xs), np.stack(cs), np.asarray(det, bool), info.fps,
                        info.width, info.height, source=video.name, model=model_name)
