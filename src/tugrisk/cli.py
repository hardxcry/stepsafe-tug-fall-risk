"""Analyse one video from the command line.

Run:  python -m tugrisk.cli path/to/video.mp4 [--start 10 --end 40]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .pose import PoseSequence, extract
from .tug import analyse


def crop(seq: PoseSequence, start: float | None, end: float | None) -> PoseSequence:
    a = int((start or 0) * seq.fps)
    b = int(end * seq.fps) if end else len(seq.world)
    return PoseSequence(seq.fps, seq.world[a:b], seq.image[a:b], seq.vis[a:b])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, help="seconds into the video where the test clip begins")
    ap.add_argument("--end", type=float, help="seconds into the video where the test clip ends")
    ap.add_argument("--save-pose", help="write the pose sequence to this .npz file")
    args = ap.parse_args()
    seq = extract(args.video, progress=lambda f: print(f"\rpose estimation {f:.0%}", end=""))
    print(f"\n{len(seq.world)} frames at {seq.fps:.1f} fps, person detected in {seq.detected.mean():.0%}")
    if args.save_pose:
        np.savez_compressed(args.save_pose, fps=seq.fps, world=seq.world, image=seq.image, vis=seq.vis)
    result = analyse(crop(seq, args.start, args.end))
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
