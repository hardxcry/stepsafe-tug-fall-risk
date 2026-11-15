"""Timed Up and Go analysis from a pose sequence.

Segments the test into five phases (sit-to-stand, walk out, turn, walk back, turn-to-sit),
times it, and derives gait features. Risk category follows the CDC STEADI cut-off: an older
adult who takes 12 seconds or longer is at increased risk of falling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .pose import (L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, PoseSequence, R_ANKLE, R_HIP, R_KNEE,
                   R_SHOULDER)

STEADI_CUTOFF_S = 12.0
ONSET = 0.10            # fraction of peak rise/descent rate that marks movement onset/offset
TURN_LO, TURN_HI = 0.12, 0.88   # fractions of 180 degrees marking turn start and end
PHASES = ["sit_to_stand", "walk_out", "turn", "walk_back", "turn_to_sit"]


@dataclass
class TugResult:
    total_time: float
    start: float
    end: float
    phases: dict[str, tuple[float, float]]
    steps: int
    cadence_spm: float
    trunk_lean_mean_deg: float
    trunk_lean_sd_deg: float
    detection_rate: float
    increased_risk: bool
    signals: dict = field(default_factory=dict, repr=False)

    def durations(self) -> dict[str, float]:
        return {k: b - a for k, (a, b) in self.phases.items()}

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("signals")
        return d | {"phase_durations": self.durations()}


class TugError(ValueError):
    """Raised when the recording cannot be analysed (for example no sit-stand-sit pattern)."""


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees for arrays of 3-D points."""
    u, v = a - b, c - b
    cos = np.sum(u * v, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _interp_nan(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    bad = np.isnan(x)
    if bad.all():
        raise TugError("No person detected in the recording.")
    idx = np.arange(len(x))
    x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def _smooth(x: np.ndarray, fps: float, seconds: float) -> np.ndarray:
    win = max(5, int(seconds * fps) | 1)
    if win >= len(x):
        win = (len(x) - 1) | 1
    return savgol_filter(x, win, 2) if win >= 5 else x


def _rate(x: np.ndarray, fps: float, seconds: float) -> np.ndarray:
    """Smoothed first derivative (units per second)."""
    win = min(max(5, int(seconds * fps) | 1), (len(x) - 1) | 1)
    return savgol_filter(x, win, 2, deriv=1, delta=1 / fps)


def _first(cond: np.ndarray, start: int = 0) -> int | None:
    hits = np.flatnonzero(cond[start:])
    return int(hits[0] + start) if len(hits) else None


def _last(cond: np.ndarray, end: int | None = None) -> int | None:
    hits = np.flatnonzero(cond[:end])
    return int(hits[-1]) if len(hits) else None


def posture_signal(w: np.ndarray) -> np.ndarray:
    """Mean of hip and knee joint angles, both sides: about 100 deg seated, 170+ standing."""
    knee = (_angle(w[:, L_HIP], w[:, L_KNEE], w[:, L_ANKLE]) +
            _angle(w[:, R_HIP], w[:, R_KNEE], w[:, R_ANKLE])) / 2
    hip = (_angle(w[:, L_SHOULDER], w[:, L_HIP], w[:, L_KNEE]) +
           _angle(w[:, R_SHOULDER], w[:, R_HIP], w[:, R_KNEE])) / 2
    return (knee + hip) / 2


def yaw_signal(w: np.ndarray) -> np.ndarray:
    """Body heading about the vertical axis (radians, unwrapped) from shoulder and hip lines."""
    v = (w[:, R_SHOULDER] - w[:, L_SHOULDER]) + (w[:, R_HIP] - w[:, L_HIP])
    return np.unwrap(np.arctan2(v[:, 2], v[:, 0]))


def _turns_from_yaw(yaw, fps, i_up, i_down, i_end, n):
    walk0 = slice(i_up, i_up + max(3, int(0.5 * fps)))
    dev = np.abs(yaw - np.median(yaw[walk0]))          # rotation away from the outbound heading
    i_half = _first(dev[i_up:i_down + 1] > np.pi / 2)
    if i_half is None:
        raise TugError("No turn detected; the person must walk away, turn and come back.")
    i_half += i_up
    i_t1s = (_last(dev[i_up:i_half] < TURN_LO * np.pi) or 0) + i_up
    i_t1e = _first(dev > TURN_HI * np.pi, i_half) or i_half
    back = slice(i_t1e, min(n, i_t1e + max(3, int(0.5 * fps))))
    dev2 = np.abs(yaw - np.median(yaw[back]))          # rotation away from the return heading
    i_t2s = _first(dev2 > TURN_LO * np.pi, i_t1e + int(0.3 * fps))
    if i_t2s is None or i_t2s > i_end:
        i_t2s = i_down                                 # no clear second turn: fall back to sit onset
    return i_t1s, i_t1e, i_t2s


def _turns_from_travel(seq, fps, i_start, i_up, i_down):
    """Turns located from how far the person has travelled from the chair in the image:
    sideways hip displacement (side view) or change in apparent body height (front view),
    whichever varies more. The first turn surrounds the farthest point; walking is where
    travel speed exceeds half its median."""
    img = seq.image
    x = _interp_nan((img[:, L_HIP, 0] + img[:, R_HIP, 0]) / 2)
    height = _interp_nan(np.abs((img[:, L_SHOULDER, 1] + img[:, R_SHOULDER, 1]) -
                                (img[:, L_ANKLE, 1] + img[:, R_ANKLE, 1])) / 2)
    ref = slice(max(0, i_start - int(0.5 * fps)), i_start + 1)
    side = np.abs(x - np.median(x[ref]))
    depth = np.abs(np.log(np.maximum(height, 1e-3) / np.median(height[ref])))
    up = slice(i_up, i_down + 1)
    travel = side if np.ptp(side[up]) >= 0.6 * np.ptp(depth[up]) else depth
    travel = _smooth(travel, fps, 0.7)
    if np.ptp(travel[up]) < 0.08:
        raise TugError("The person does not appear to walk away from the chair and back.")
    i_far = i_up + int(np.argmax(travel[up]))
    speed = np.abs(_rate(travel, fps, 0.7))
    moving = speed > 0.5 * np.median(speed[up])
    i_t1s = _last(moving[:i_far], None)
    i_t1s = i_up if i_t1s is None or i_t1s < i_up else i_t1s + 1
    i_t1e = _first(moving, i_far)
    i_t1e = i_far if i_t1e is None or i_t1e > i_down else i_t1e
    i_t2s = _last(moving[:i_down + 1], None)
    i_t2s = i_down if i_t2s is None or i_t2s < i_t1e else min(i_t2s + 1, i_down)
    return i_t1s, i_t1e, i_t2s, travel


def analyse(seq: PoseSequence) -> TugResult:
    fps, n = seq.fps, len(seq.world)
    if n < int(3 * fps):
        raise TugError("Recording is too short for a Timed Up and Go test.")
    det = float(seq.detected.mean())
    if det < 0.6:
        raise TugError(f"Person visible in only {det:.0%} of frames; re-record with the whole body in view.")
    w = np.stack([_interp_nan(seq.world[:, j, k]) for j in range(33) for k in range(3)], 1).reshape(n, 33, 3)
    t = seq.t

    # ---- posture: seated <-> standing --------------------------------------------------
    post = _smooth(posture_signal(w), fps, 0.5)
    sit, stand = np.percentile(post, 5), np.percentile(post, 90)
    if stand - sit < 35:
        raise TugError("No sit-to-stand movement found; the test must start and end seated.")
    lo, mid, hi = sit + 0.15 * (stand - sit), sit + 0.5 * (stand - sit), sit + 0.85 * (stand - sit)
    i_up = _first(post > hi)
    i_down = _last(post > hi)
    if i_up is None or i_down is None or i_down - i_up < fps:
        raise TugError("Could not find a standing period between two seated periods.")
    # Movement onset/offset from the rate of posture change: the test starts when the rise
    # towards standing begins (rate first exceeds ONSET x its peak) and ends when the descent
    # into the chair has stopped. Level thresholds alone lag a smooth movement by ~0.4 s.
    rate = _rate(posture_signal(w), fps, 0.5)
    r0 = _last(post[:i_up] < lo, None)
    r1 = _first(post < lo, i_down)
    i_start = i_end = None
    if r0 is not None and r1 is not None:
        k_up = r0 + int(np.argmax(rate[r0:i_up + 1]))
        i_start = _last(rate[:k_up] < ONSET * rate[k_up], None) or 0
        k_dn = i_down + int(np.argmin(rate[i_down:r1 + 1]))
        i_end = _first(rate > ONSET * rate[k_dn], k_dn)
        i_end = n - 1 if i_end is None else i_end
    if i_start is None:
        raise TugError("The recording must begin with the person seated.")
    if i_end is None:
        raise TugError("The recording must end with the person seated again.")

    # ---- the two turns ---------------------------------------------------------------------
    yaw = _smooth(yaw_signal(w), fps, 0.6)
    walk0 = slice(i_up, i_up + max(3, int(0.5 * fps)))
    has_image = not np.isnan(seq.image[:, L_HIP, 0]).all()
    if has_image:      # real video: use travel in the image, which is robust to camera angle
        i_t1s, i_t1e, i_t2s, travel = _turns_from_travel(seq, fps, i_start, i_up, i_down)
    else:              # pose-only input: use body heading
        i_t1s, i_t1e, i_t2s = _turns_from_yaw(yaw, fps, i_up, i_down, i_end, n)
        travel = None
    idx = {"start": i_start, "up": i_up, "t1s": max(i_t1s, i_up), "t1e": i_t1e,
           "t2s": max(i_t2s, i_t1e), "end": i_end}
    phases = {"sit_to_stand": (t[idx["start"]], t[idx["up"]]),
              "walk_out": (t[idx["up"]], t[idx["t1s"]]),
              "turn": (t[idx["t1s"]], t[idx["t1e"]]),
              "walk_back": (t[idx["t1e"]], t[idx["t2s"]]),
              "turn_to_sit": (t[idx["t2s"]], t[idx["end"]])}

    # ---- gait features during the two walking phases --------------------------------------
    ankle = _smooth(np.linalg.norm(w[:, L_ANKLE] - w[:, R_ANKLE], axis=1), fps, 0.25)
    walking = np.zeros(n, bool)
    walking[idx["up"]:idx["t1s"]] = True
    walking[idx["t1e"]:idx["t2s"]] = True
    steps = 0
    for a, b in [(idx["up"], idx["t1s"]), (idx["t1e"], idx["t2s"])]:
        if b - a > 3:
            pk, _ = find_peaks(ankle[a:b], distance=max(1, int(0.25 * fps)),
                               prominence=0.25 * np.ptp(ankle[a:b]) + 1e-6)
            steps += len(pk)
    walk_time = walking.sum() / fps
    trunk = ((w[:, L_SHOULDER] + w[:, R_SHOULDER]) - (w[:, L_HIP] + w[:, R_HIP])) / 2
    lean = np.degrees(np.arccos(np.clip(-trunk[:, 1] / (np.linalg.norm(trunk, axis=1) + 1e-9), -1, 1)))
    lw = lean[walking] if walking.any() else lean

    total = float(t[i_end] - t[i_start])
    return TugResult(
        total_time=total, start=float(t[i_start]), end=float(t[i_end]),
        phases={k: (float(a), float(b)) for k, (a, b) in phases.items()},
        steps=int(steps), cadence_spm=float(60 * steps / walk_time) if walk_time > 0 else 0.0,
        trunk_lean_mean_deg=float(lw.mean()), trunk_lean_sd_deg=float(lw.std()),
        detection_rate=det, increased_risk=total >= STEADI_CUTOFF_S,
        signals={"t": t, "posture": post, "yaw_deg": np.degrees(yaw - np.median(yaw[walk0])),
                 "ankle_distance": ankle, "travel": travel,
                 "thresholds": (float(lo), float(mid), float(hi))})
