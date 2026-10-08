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
weights directory -- keep that directory user-private. Ultralytics is imported
lazily so the rest of the repo (and the tests) need neither it nor a GPU.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import N_JOINTS, Track
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


_FP = re.compile(r"[0-9a-f]{64}")


def _check_fp(fp) -> None:
    """A stored video fingerprint is empty (older files, never bound) or a sha256 hex digest."""
    if not isinstance(fp, str) or (fp and not _FP.fullmatch(fp)):
        raise ValueError("video_fp must be empty or a 64-character lowercase sha256 hex digest")


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
        with np.load(path, allow_pickle=False) as z:
            return cls(xy=z["xy"], conf=z["conf"], detected=z["detected"],
                       fps=float(z["fps"]), width=int(z["width"]), height=int(z["height"]),
                       source=str(z["source"]), model=str(z["model"]),
                       video_fp=str(z["video_fp"]) if "video_fp" in z.files else "")


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
        with np.load(path, allow_pickle=False) as z:
            return cls(frame=z["frame"].astype(np.int64), track_id=z["track_id"].astype(np.int64),
                       box=z["box"], xy=z["xy"], conf=z["conf"], n_frames=int(z["n_frames"]),
                       fps=float(z["fps"]), width=int(z["width"]), height=int(z["height"]),
                       source=str(z["source"]), model=str(z["model"]),
                       video_fp=str(z["video_fp"]) if "video_fp" in z.files else "")


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
