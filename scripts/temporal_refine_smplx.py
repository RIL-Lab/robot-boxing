#!/usr/bin/env python3
"""Refine existing SMPL-X NPZ rotations and report temporal quality."""
import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from multi_hmr_boxing_video import rotation_quality, temporal_filter_rotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tracks", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.tracks:
        with np.load(path, allow_pickle=False) as source:
            data = {key: source[key] for key in source.files}
        rotations = np.concatenate([
            data["global_orient"][:, None],
            data["body_pose"].reshape(-1, 21, 3),
            data["left_hand_pose"].reshape(-1, 15, 3),
            data["right_hand_pose"].reshape(-1, 15, 3),
            data.get("jaw_pose", np.zeros((len(data["global_orient"]), 3), np.float32))[:, None],
        ], axis=1).astype(np.float32)
        before = rotation_quality(rotations)
        rotations = temporal_filter_rotations(rotations)
        after = rotation_quality(rotations)
        data["global_orient"] = rotations[:, 0]
        data["body_pose"] = rotations[:, 1:22].reshape(-1, 63)
        data["left_hand_pose"] = rotations[:, 22:37].reshape(-1, 45)
        data["right_hand_pose"] = rotations[:, 37:52].reshape(-1, 45)
        data["jaw_pose"] = rotations[:, 52]
        data["temporal_refined"] = np.bool_(True)
        fd, temp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
        os.close(fd)
        try:
            np.savez_compressed(temp_name, **data)
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        print(json.dumps({"path": str(path), "before": before, "after": after}, ensure_ascii=False))


if __name__ == "__main__":
    main()
