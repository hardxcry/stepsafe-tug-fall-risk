"""Markerless pose extraction from ordinary video with MediaPipe Pose Landmarker."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"

# MediaPipe landmark indices used by the analysis
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24
L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, NOSE = 25, 26, 27, 28, 0
SKELETON = [(11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
            (11, 13), (13, 15), (12, 14), (14, 16), (27, 31), (28, 32)]


@dataclass
class PoseSequence:
    """world: (T, 33, 3) metres, hip-centred; image: (T, 33, 2) normalised; vis: (T, 33)."""
    fps: float
    world: np.ndarray
    image: np.ndarray
    vis: np.ndarray

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.world)) / self.fps

    @property
    def detected(self) -> np.ndarray:
        return ~np.isnan(self.world[:, 0, 0])


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(MODEL_URL, timeout=120)
        r.raise_for_status()
        MODEL_PATH.write_bytes(r.content)
    return MODEL_PATH


def _seatedness(world: np.ndarray) -> float:
    """Mean knee angle in degrees: small when seated, near 180 when standing."""
    out = []
    for hip, knee, ankle in [(L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)]:
        u, v = world[hip] - world[knee], world[ankle] - world[knee]
        cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
        out.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return float(np.mean(out))


def select_subject(candidates: list[list[tuple]], fps: float, start: int = 0,
                   max_jump: float = 0.25) -> "PoseSequence":
    """Follow one person through frames that may contain several (for example a clinician
    walking beside the patient). The subject is whoever is most clearly seated at `start`;
    afterwards the nearest pose to the subject's predicted hip position is kept.

    candidates[f] is a list of (world(33,3), image(33,2), vis(33)) for frame f.
    """
    n = len(candidates)
    world = np.full((n, 33, 3), np.nan)
    image = np.full((n, 33, 2), np.nan)
    vis = np.zeros((n, 33))
    pos = vel = None
    for f in range(start, n):
        cands = candidates[f]
        if not cands:
            continue
        centres = [c[1][[L_HIP, R_HIP]].mean(axis=0) for c in cands]
        if pos is None:
            k = int(np.argmin([_seatedness(c[0]) for c in cands]))
        else:
            pred = pos + (vel if vel is not None else 0)
            dist = [float(np.linalg.norm(c - pred)) for c in centres]
            k = int(np.argmin(dist))
            if dist[k] > max_jump:
                continue                      # nobody near the subject: treat as a missed frame
            vel = 0.7 * (vel if vel is not None else 0) + 0.3 * (centres[k] - pos)
        pos = centres[k]
        world[f], image[f], vis[f] = cands[k]
    return PoseSequence(fps, world, image, vis)


def detect(video_path: str | Path, max_fps: float = 30.0, max_people: int = 3,
           progress=None) -> tuple[list[list[tuple]], float]:
    """Pose candidates for every analysed frame, and the analysis frame rate."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(src_fps / max_fps)))
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.VIDEO, num_poses=max_people,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    frames: list[list[tuple]] = []
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                          int(1000 * i / src_fps))
                frames.append([(np.array([[q.x, q.y, q.z] for q in w]),
                                np.array([[q.x, q.y] for q in im]),
                                np.array([q.visibility for q in im]))
                               for w, im in zip(res.pose_world_landmarks, res.pose_landmarks)])
                if progress and n_frames:
                    progress(min(1.0, i / n_frames))
            i += 1
    cap.release()
    return frames, src_fps / step


def extract(video_path: str | Path, max_fps: float = 30.0, start_s: float = 0.0,
            end_s: float | None = None, progress=None) -> PoseSequence:
    """Pose sequence of the test subject between start_s and end_s of the video."""
    frames, fps = detect(video_path, max_fps=max_fps, progress=progress)
    a = int(start_s * fps)
    b = int(end_s * fps) if end_s else len(frames)
    return select_subject(frames[a:b], fps)


def draw_skeleton(frame, image_landmarks: np.ndarray, color=(80, 220, 120)):
    """Overlay the skeleton on a BGR frame (in place)."""
    import cv2
    h, w = frame.shape[:2]
    if np.isnan(image_landmarks[0, 0]):
        return frame
    pts = (image_landmarks * [w, h]).astype(int)
    for a, b in SKELETON:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 3, cv2.LINE_AA)
    for i in set(sum(SKELETON, ())):
        cv2.circle(frame, tuple(pts[i]), 5, (255, 255, 255), -1, cv2.LINE_AA)
    return frame
