#!/usr/bin/env python3
"""Apply explicit Z-up floor offsets to SMPL-X NPZ tracks in place."""
import argparse
import os
import tempfile
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("items", nargs="+", help="PATH=Z_OFFSET_METERS")
    args = parser.parse_args()
    for item in args.items:
        raw_path, raw_offset = item.rsplit("=", 1)
        path = Path(raw_path)
        offset = float(raw_offset)
        with np.load(path, allow_pickle=False) as source:
            data = {key: source[key] for key in source.files}
        transl = np.asarray(data["transl"], dtype=np.float32).copy()
        transl[:, 2] += offset
        data["transl"] = transl
        previous = float(np.asarray(data.get("ground_offset_z", 0.0)).reshape(-1)[0])
        data["ground_offset_z"] = np.float32(previous + offset)
        fd, temp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
        os.close(fd)
        try:
            np.savez_compressed(temp_name, **data)
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        print(f"{path}: Z {offset:+.3f} m")


if __name__ == "__main__":
    main()
