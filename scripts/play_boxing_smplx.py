#!/usr/bin/env python3
"""Replay one or two SMPL-X parameter tracks in AITViewer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from aitviewer.configuration import CONFIG as C

C.window_type = os.environ.get("AITVIEWER_WINDOW_TYPE", "glfw")
C.auto_set_camera_target = False
C.auto_set_floor = False
C.shadows_enabled = False
C.vsync = False

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer


MODEL_ROOT = Path("/home/ubuntu22/models/smplx_pkl_only")


def _array(data, names, width, frames=None, required=True):
    for name in names:
        if name in data:
            value = np.asarray(data[name], dtype=np.float32)
            if value.ndim == 3 and value.shape[-1] == 3:
                value = value.reshape(value.shape[0], -1)
            if value.ndim == 1:
                value = value.reshape(1, -1)
            if value.ndim != 2 or value.shape[1] != width:
                raise ValueError(
                    f"{name}: expected (T, {width}), got {value.shape}"
                )
            if frames is not None and value.shape[0] != frames:
                raise ValueError(
                    f"{name}: expected {frames} frames, got {value.shape[0]}"
                )
            return value
    if required:
        raise KeyError(f"missing one of: {', '.join(names)}")
    return np.zeros((frames, width), dtype=np.float32)


def load_track(path: Path):
    with np.load(path, allow_pickle=False) as data:
        root = _array(data, ("global_orient", "root_orient", "poses_root"), 3)
        frames = root.shape[0]
        body = _array(data, ("body_pose", "pose_body", "poses_body"), 63, frames)
        left = _array(
            data,
            ("left_hand_pose", "pose_lhand", "poses_left_hand"),
            45,
            frames,
            required=False,
        )
        right = _array(
            data,
            ("right_hand_pose", "pose_rhand", "poses_right_hand"),
            45,
            frames,
            required=False,
        )
        trans = _array(data, ("transl", "trans", "translation"), 3, frames)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 30.0
        betas = np.asarray(data["betas"], dtype=np.float32) if "betas" in data else np.zeros(10, np.float32)

    betas = betas.reshape(-1, betas.shape[-1] if betas.ndim > 1 else betas.size)
    betas = betas[0, :10]
    if betas.size < 10:
        betas = np.pad(betas, (0, 10 - betas.size))
    betas = np.repeat(betas[None], frames, axis=0).astype(np.float32)

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid fps in {path}: {fps}")
    for name, value in (("root", root), ("body", body), ("left", left),
                        ("right", right), ("trans", trans), ("betas", betas)):
        if not np.isfinite(value).all():
            raise ValueError(f"{path}: {name} contains NaN/Inf")
    return root, body, left, right, trans, betas, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tracks", nargs="+", type=Path,
        help="one output directory, or one/two SMPL-X .npz files",
    )
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--keep-absolute-position", action="store_true")
    args = parser.parse_args()
    if len(args.tracks) == 1 and args.tracks[0].is_dir():
        directory = args.tracks[0]
        direct = [directory / "person_1_smplx.npz", directory / "person_2_smplx.npz"]
        if not all(path.is_file() for path in direct):
            parser.error(
                "请传入单个 clip 目录；日期总目录不支持连续拼接播放。"
            )
        args.tracks = direct
    if len(args.tracks) not in (1, 2):
        parser.error("provide one or two tracks")

    C.smplx_models = str(args.model_root)
    viewer = Viewer(samples=int(os.environ.get("AITVIEWER_SAMPLES", "0")))
    expected_frames = None
    expected_fps = None
    initial_positions = []
    colors = ((0.23, 0.48, 0.95, 1.0), (0.93, 0.25, 0.22, 1.0))

    for index, path in enumerate(args.tracks):
        root, body, left, right, trans, betas, fps = load_track(path)
        if expected_frames is None:
            expected_frames, expected_fps = len(root), fps
        elif len(root) != expected_frames or abs(fps - expected_fps) > 1e-3:
            raise ValueError("the two tracks must have identical frame counts and FPS")
        if not args.keep_absolute_position:
            center = (trans[0] if len(args.tracks) == 1 else np.zeros(3, np.float32))
            trans = trans - center
        initial_positions.append(trans[0].copy())

        layer = SMPLLayer(model_type="smplx", gender="neutral", device=C.device, use_pca=False)
        seq = SMPLSequence(
            poses_body=body,
            poses_root=root,
            poses_left_hand=left,
            poses_right_hand=right,
            trans=trans,
            betas=betas,
            smpl_layer=layer,
            z_up=True,
            name=f"Boxer {index + 1}: {path.stem}",
            color=colors[index],
        )
        # Camera-space regressors estimate the pelvis translation, not the
        # floor plane. Ground each reconstructed mesh using a robust low
        # percentile of its per-frame lowest vertex so legs are not clipped.
        vertices = np.asarray(seq.vertices)
        frame_floor = vertices[:, :, 2].min(axis=1)
        ground_z = float(np.percentile(frame_floor, 10.0))
        lift_z = 0.015 - ground_z
        seq.trans[:, 2] += lift_z
        seq.redraw()
        initial_positions[-1][2] += lift_z
        print(f"Boxer {index + 1}: ground correction Z {lift_z:+.3f} m")
        viewer.scene.add(seq)

    viewer.playback_fps = expected_fps
    viewer.run_animations = True
    # Frame both fighters at startup. The exported tracks and z_up=True use a
    # Z-up world, so look horizontally across the ring at torso height.
    pair_center = np.mean(initial_positions, axis=0).astype(np.float32)
    pair_center[2] = 0.9
    viewer.scene.camera.target = pair_center
    viewer.scene.camera.position = pair_center + np.array([3.0, -4.0, 1.2], np.float32)
    viewer.scene.camera.up = np.array([0.0, 0.0, 1.0], np.float32)
    viewer.scene.camera.near = 0.01
    viewer.scene.camera.far = 100.0
    print(f"Loaded {len(args.tracks)} track(s), {expected_frames} frames @ {expected_fps:.3f} FPS")
    viewer.run()


if __name__ == "__main__":
    main()
