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
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import N_JOINTS, Track
from .npyread import read_data, read_header
from .punchfind import L_SHOULDER, L_WRIST, R_SHOULDER, R_WRIST
from .video import VideoInfo, iter_frames, probe

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
MAX_POSE_TEXT = 4096    # characters in `source` / `model`; a 'U' scalar's size is in its header


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

    def __post_init__(self) -> None:
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
                            source=self.source, model=self.model)

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
        return cls(xy=z["xy"], conf=z["conf"], detected=z["detected"],
                   fps=float(z["fps"]), width=int(z["width"]), height=int(z["height"]),
                   source=str(z["source"]), model=str(z["model"]))


# What `save` writes, and the dtype kinds accepted for each: float coordinates and
# confidence, a bool mask, a numeric fps, integer sizes, unicode text. Everything
# else -- object (pickle), structured, bytes, complex -- is refused from the header.
_POSE_KINDS = {"xy": "f", "conf": "f", "detected": "b", "fps": "fiu",
               "width": "iu", "height": "iu", "source": "U", "model": "U"}


# How `np.savez` (stored) and `np.savez_compressed` (deflate) write members. zipfile
# decompresses deflate in bounded steps, but bz2 and lzma with no output limit: a
# 2 KB bz2 member reached 584 MB just to yield the 8-byte .npy magic.
_POSE_COMPRESSION = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
# Deflate's ceiling (258 bytes from a 2-bit code, with block overhead). A header that
# promises more data than its zip entry could expand to is refused before `np.empty`:
# without this, a 1.3 KB file claiming exactly the frame cap at float64 allocated
# 544 MB and only then found the data missing.
_DEFLATE_MAX_RATIO = 1032


def _npy_header(member, key: str, name: str):
    """(shape, fortran_order, dtype) from a member's .npy header, read through a cap
    (`npyread.read_header`), with the dtype checked against what `save` writes for
    `key`. Leaves `member` positioned at the first byte of array data."""
    shape, fortran, dtype = read_header(member, f"{name}: {key}")
    max_item = 4 * MAX_POSE_TEXT if dtype.kind == "U" else 8  # 'U' is 4 bytes a character
    if (dtype.kind not in _POSE_KINDS[key] or dtype.hasobject or dtype.fields is not None
            or dtype.subdtype is not None or dtype.itemsize > max_item):
        raise ValueError(f"{name}: {key} has dtype {dtype.str!r}; a pose file's {key} "
                         f"must be of kind {_POSE_KINDS[key]!r}")
    return shape, fortran, dtype


def _read_pose_npz(fh, name: str) -> "dict[str, np.ndarray]":
    """Validate every member's .npy header, then read the arrays -- in that order,
    from one open file, so what was checked is what is read."""
    with zipfile.ZipFile(fh) as zf:
        names = set(zf.namelist())
        missing = [k for k in _POSE_KINDS if k + ".npy" not in names]
        if missing:
            raise ValueError(f"{name}: not a pose file, missing {', '.join(missing)}")
        for k in _POSE_KINDS:  # before any member is opened: opening one decompresses
            if zf.getinfo(k + ".npy").compress_type not in _POSE_COMPRESSION:
                raise ValueError(f"{name}: {k} uses zip compression method "
                                 f"{zf.getinfo(k + '.npy').compress_type}; a pose file is "
                                 f"stored or deflated (np.savez / np.savez_compressed)")
        heads = {}
        for k in _POSE_KINDS:
            with zf.open(k + ".npy") as m:
                heads[k] = _npy_header(m, k, name)
        dshape = heads["detected"][0]
        if len(dshape) != 1 or type(dshape[0]) is not int or dshape[0] < 0:  # (True,) parses
            raise ValueError(f"{name}: detected has shape {dshape}, expected (frames,)")
        f = dshape[0]
        want = {"xy": (f, N_JOINTS, 2), "conf": (f, N_JOINTS), "detected": (f,)}
        for k in _POSE_KINDS:
            if heads[k][0] != want.get(k, ()):  # tuple == compares lengths and every dim
                raise ValueError(f"{name}: {k} has shape {heads[k][0]}, expected "
                                 f"{want.get(k, ())} for a {f}-frame pose")
        if f > MAX_POSE_FRAMES:  # before anything is allocated
            raise ValueError(f"{name}: claims {f:,} frames; the cap is {MAX_POSE_FRAMES:,} "
                             f"(~9 h at 60 fps)")
        size = os.fstat(fh.fileno()).st_size
        for k in _POSE_KINDS:  # what the member could hold at most, before allocating it
            info = zf.getinfo(k + ".npy")
            packed = min(info.compress_size, size)  # a declared size can lie; the file can't
            room = min(info.file_size, packed if info.compress_type == zipfile.ZIP_STORED
                       else packed * _DEFLATE_MAX_RATIO)
            need = heads[k][2].itemsize * int(np.prod(heads[k][0], dtype=np.int64))
            if need > room:
                raise ValueError(f"{name}: {k} promises {need:,} bytes of data; its "
                                 f"{packed:,}-byte zip entry can hold at most {room:,}")
        out = {}
        for k in _POSE_KINDS:
            with zf.open(k + ".npy") as m:
                if _npy_header(m, k, name) != heads[k]:  # rewritten since validated
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
                         "(compute one with: python run_ingest.py --hash-weights <file>)")
    if sha256_file(p) != sha256.lower():
        raise ValueError(f"{p.name} does not match the pinned sha256; not loading it")
    try:
        from ultralytics import YOLO  # lazy: heavy, and not needed to test this module
    except ImportError as e:
        raise RuntimeError("ultralytics is not installed (pip install -r requirements-pose.txt)") from e
    model = YOLO(str(p))
    model.to(device)
    return model


def yolo_predictor(model, conf: float = 0.25, device: str = "cpu"):
    """Wrap a loaded model as predict(frame_rgb) -> (boxes, kp_xy, kp_conf)."""

    def predict(frame_rgb: np.ndarray):
        # Ultralytics treats a bare ndarray as BGR (OpenCV convention); ours is RGB.
        r = model.predict(frame_rgb[..., ::-1], conf=conf, device=device, verbose=False)[0]
        if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
            return np.zeros((0, 4)), np.zeros((0, N_JOINTS, 2)), np.zeros((0, N_JOINTS))
        boxes = r.boxes.xyxy.cpu().numpy()
        kxy = r.keypoints.xy.cpu().numpy()
        kc = (r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None
              else np.ones(kxy.shape[:2]))
        return boxes, kxy, kc

    return predict


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
