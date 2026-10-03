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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import N_JOINTS, Track
from .video import VideoInfo, iter_frames, probe

PERSON_IOU_KEEP = 0.3   # a box overlapping last frame's by this much is "the same boxer"
CONTINUITY_BONUS = 2.0  # ... and its area counts double, so a bigger bystander can't steal the track


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

    def __len__(self) -> int:
        return len(self.detected)

    def to_track(self) -> Track:
        """Detected frames only; gaps survive as jumps in `frame_ids`."""
        ids = np.flatnonzero(self.detected)
        return Track(frame_ids=ids, xy=self.xy[ids].astype(np.float64), fps=self.fps,
                     source=self.source)

    def save(self, path) -> None:
        np.savez_compressed(path, xy=self.xy, conf=self.conf, detected=self.detected,
                            fps=self.fps, width=self.width, height=self.height,
                            source=self.source, model=self.model)

    @classmethod
    def load(cls, path) -> "PoseSequence":
        with np.load(path, allow_pickle=False) as z:
            return cls(xy=z["xy"], conf=z["conf"], detected=z["detected"],
                       fps=float(z["fps"]), width=int(z["width"]), height=int(z["height"]),
                       source=str(z["source"]), model=str(z["model"]))


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


def extract(video, predict, model_name: str = "", max_frames: "int | None" = None) -> PoseSequence:
    """Run `predict` over every frame and keep one person per frame."""
    video = Path(video).resolve()
    info: VideoInfo = probe(video)
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
