"""Kinematic simulator: a 3-D stick figure performing the Timed Up and Go test with known
ground-truth phase times. Used to measure the timing error of the analysis under controlled
sensor noise, frame rate and tracking drop-out. It is a test bench, not a substitute for
validation on recordings of real patients.

Body frame (matches MediaPipe world landmarks): x lateral, y down, z forward, hip centre at origin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pose import (L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, PoseSequence, R_ANKLE, R_HIP, R_KNEE,
                   R_SHOULDER)

THIGH, SHANK, TRUNK, HIP_W, SHOULDER_W = 0.42, 0.43, 0.50, 0.10, 0.18


@dataclass
class Protocol:
    """Phase durations in seconds."""
    pre_sit: float = 1.5
    sit_to_stand: float = 1.5
    walk_out: float = 3.0
    turn: float = 1.5
    walk_back: float = 3.0
    turn_to_sit: float = 3.0
    post_sit: float = 1.5
    cadence_spm: float = 105.0
    second_turn_same_direction: bool = True

    @property
    def truth(self) -> dict[str, tuple[float, float]]:
        t0 = self.pre_sit
        edges = np.cumsum([t0, self.sit_to_stand, self.walk_out, self.turn, self.walk_back,
                           self.turn_to_sit])
        names = ["sit_to_stand", "walk_out", "turn", "walk_back", "turn_to_sit"]
        return {n: (float(edges[i]), float(edges[i + 1])) for i, n in enumerate(names)}

    @property
    def total(self) -> float:
        return (self.sit_to_stand + self.walk_out + self.turn + self.walk_back + self.turn_to_sit)


def _ramp(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """Smooth 0 -> 1 transition between times a and b (minimum-jerk profile)."""
    s = np.clip((t - a) / max(b - a, 1e-6), 0, 1)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def simulate(p: Protocol, fps: float = 30.0, noise_m: float = 0.0, dropout: float = 0.0,
             seed: int | None = None) -> PoseSequence:
    rng = np.random.default_rng(seed)
    tr = p.truth
    duration = tr["turn_to_sit"][1] + p.post_sit
    t = np.arange(0, duration, 1 / fps)
    n = len(t)

    # standing fraction: rises during sit-to-stand, falls over the last 60% of turn-to-sit
    sit_onset = tr["turn_to_sit"][0] + 0.4 * (tr["turn_to_sit"][1] - tr["turn_to_sit"][0])
    s = _ramp(t, *tr["sit_to_stand"]) - _ramp(t, sit_onset, tr["turn_to_sit"][1])
    # heading: first turn at the far end, second turn over the first 60% of turn-to-sit
    t2_end = tr["turn_to_sit"][0] + 0.6 * (tr["turn_to_sit"][1] - tr["turn_to_sit"][0])
    sign = 1.0 if p.second_turn_same_direction else -1.0
    yaw = np.pi * _ramp(t, *tr["turn"]) + sign * np.pi * _ramp(t, tr["turn_to_sit"][0], t2_end)
    # gait cycle active while upright and walking
    walking = ((t >= tr["walk_out"][0]) & (t < tr["turn"][0])) | \
              ((t >= tr["walk_back"][0]) & (t < tr["turn_to_sit"][0]))
    phase = 2 * np.pi * (p.cadence_spm / 120.0) * t
    swing = np.radians(22) * np.sin(phase) * walking

    thigh_pitch = np.radians(90) * (1 - s)                 # 90 deg forward when seated
    lean = np.radians(8) + np.radians(30) * np.sin(np.pi * np.clip(s, 0, 1)) * (~walking)
    world = np.full((n, 33, 3), 0.0)

    def leg(side: float, sw: np.ndarray):
        hip = np.stack([np.full(n, side * HIP_W), np.zeros(n), np.zeros(n)], 1)
        ang = thigh_pitch + sw
        knee = hip + THIGH * np.stack([np.zeros(n), np.cos(ang), np.sin(ang)], 1)
        flex = np.radians(12) * np.clip(-np.sin(phase) * side, 0, 1) * walking  # swing-knee flexion
        ankle = knee + SHANK * np.stack([np.zeros(n), np.cos(flex), -np.sin(flex)], 1)
        return hip, knee, ankle

    lh, lk, la = leg(+1, swing)
    rh, rk, ra = leg(-1, -swing)
    up = TRUNK * np.stack([np.zeros(n), -np.cos(lean), np.sin(lean)], 1)
    ls = up + np.array([SHOULDER_W, 0, 0])
    rs = up + np.array([-SHOULDER_W, 0, 0])
    for j, pts in {L_HIP: lh, R_HIP: rh, L_KNEE: lk, R_KNEE: rk, L_ANKLE: la, R_ANKLE: ra,
                   L_SHOULDER: ls, R_SHOULDER: rs}.items():
        c, sn = np.cos(yaw), np.sin(yaw)                   # rotate about the vertical axis
        world[:, j, 0] = c * pts[:, 0] + sn * pts[:, 2]
        world[:, j, 1] = pts[:, 1]
        world[:, j, 2] = -sn * pts[:, 0] + c * pts[:, 2]

    if noise_m > 0:
        world += rng.normal(0, noise_m, world.shape)
    vis = np.ones((n, 33))
    if dropout > 0:
        lost = rng.random(n) < dropout
        world[lost] = np.nan
        vis[lost] = 0
    return PoseSequence(fps, world, np.full((n, 33, 2), np.nan), vis)


def random_protocol(rng: np.random.Generator) -> Protocol:
    """Sample a plausible test, from brisk healthy adults to slow, frail performers."""
    slow = rng.uniform(0.0, 1.0)                           # 0 = fast, 1 = very slow
    return Protocol(
        pre_sit=rng.uniform(1.0, 2.5), post_sit=rng.uniform(1.0, 2.5),
        sit_to_stand=rng.uniform(0.9, 1.6) + 2.0 * slow * rng.uniform(0.3, 1),
        walk_out=rng.uniform(1.8, 2.8) + 5.0 * slow * rng.uniform(0.3, 1),
        turn=rng.uniform(0.9, 1.6) + 2.5 * slow * rng.uniform(0.3, 1),
        walk_back=rng.uniform(1.8, 2.8) + 5.0 * slow * rng.uniform(0.3, 1),
        turn_to_sit=rng.uniform(1.8, 2.8) + 3.5 * slow * rng.uniform(0.3, 1),
        cadence_spm=rng.uniform(70, 120) - 25 * slow,
        second_turn_same_direction=bool(rng.random() < 0.5))
