#!/usr/bin/env python3
"""Prepare a short, stable two-boxer AITViewer demonstration."""
import os
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path("/home/ubuntu22/boxing_smplx_teacher_demo")
FIST_SOURCE = Path("/home/ubuntu22/boxing_smplx_duel_sample")
FRAMES = 180

for index in (1, 2):
    path = ROOT / f"person_{index}_smplx.npz"
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    for key, value in list(data.items()):
        value = np.asarray(value)
        if value.ndim and value.shape[0] >= FRAMES and key not in {"betas"}:
            data[key] = value[:FRAMES]
    # Use a robust, fixed fist estimated across another boxing sequence.
    # A zero SMPL-X hand pose is an open palm, not a fist.
    with np.load(FIST_SOURCE / f"person_{index}_smplx.npz", allow_pickle=False) as fist_source:
        for key in ("left_hand_pose", "right_hand_pose"):
            fist = np.median(fist_source[key].reshape(-1, 15, 3), axis=0).reshape(45).astype(np.float32)
            data[key] = np.repeat(fist[None], FRAMES, axis=0)
    data["showcase_trimmed"] = np.bool_(True)
    fd, temp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temp_name, **data)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    print(path, FRAMES)
