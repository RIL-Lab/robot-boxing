#!/usr/bin/env python3
"""Remove implausible root-translation jumps from SMPL-X NPZ tracks."""
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


ROOT = Path("/home/ubuntu22/boxing_smplx_teacher_demo")
MAX_STEP = 0.075  # metres per frame at 30 FPS

for index in (1, 2):
    path = ROOT / f"person_{index}_smplx.npz"
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    original = np.asarray(data["transl"], np.float32)
    before = np.linalg.norm(np.diff(original, axis=0), axis=1)
    local_median = median_filter(original, size=(9, 1), mode="nearest")
    repaired = original.copy()
    outlier = np.linalg.norm(original - local_median, axis=1) > 0.18
    repaired[outlier] = local_median[outlier]
    repaired = savgol_filter(repaired, 11, 2, axis=0, mode="interp").astype(np.float32)
    # Hard speed ceiling prevents multi-frame detector switches from surviving.
    for frame in range(1, len(repaired)):
        delta = repaired[frame] - repaired[frame - 1]
        distance = float(np.linalg.norm(delta))
        if distance > MAX_STEP:
            repaired[frame] = repaired[frame - 1] + delta * (MAX_STEP / distance)
    data["transl"] = repaired
    data["translation_refined"] = np.bool_(True)
    after = np.linalg.norm(np.diff(repaired, axis=0), axis=1)
    fd, temp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temp_name, **data)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    print(
        f"Boxer {index}: max step {before.max():.3f} -> {after.max():.3f} m, "
        f"repaired frames={int(outlier.sum())}"
    )
