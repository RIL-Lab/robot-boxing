#!/usr/bin/env python3
"""Convert a boxing video to two tracked, AITViewer-ready SMPL-X NPZ files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import smplx
import torch
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import savgol_filter
from ultralytics import YOLO

ELEMENTS_ROOT = next(
    (
        root for root in (
            Path("/media/ubuntu22/Elements"),
            Path("/media/ubuntu22/Elements1"),
        )
        if (root / "multi-hmr").is_dir()
    ),
    Path("/media/ubuntu22/Elements"),
)
MULTI_HMR = ELEMENTS_ROOT / "multi-hmr"
SMPLX_MODEL_ROOT = Path("/home/ubuntu22/下载/models_smplx_v1_1/models")
PIPELINE_VERSION = "boxing_smplx_v4_spatial_contact"
sys.path.insert(0, str(MULTI_HMR))

from demo import forward_model, get_camera_parameters, load_model, open_image


@dataclass
class Track:
    track_id: int
    observations: dict[int, dict] = field(default_factory=dict)

    @property
    def last_frame(self):
        return max(self.observations)

    @property
    def last(self):
        return self.observations[self.last_frame]


def detection_to_numpy(human, image_size, original_width, original_height):
    out = {}
    for key in ("scores", "loc", "transl_pelvis", "rotvec", "shape", "j2d"):
        value = human[key].detach().float().cpu().numpy()
        out[key] = value
    def undo_square_padding(points):
        scale = min(image_size / original_width, image_size / original_height)
        resized = np.array([original_width * scale, original_height * scale])
        padding = (image_size - resized) / 2
        return (points - padding) / resized

    out["loc"] = undo_square_padding(out["loc"].reshape(1, 2))[0]
    joints = undo_square_padding(out["j2d"].reshape(-1, 2))
    good = np.isfinite(joints).all(1)
    out["j2d_norm"] = joints.astype(np.float32)
    if np.any(good):
        visible_joints = joints[good]
        lo, hi = np.percentile(visible_joints, [5, 95], axis=0)
        out["bbox"] = np.r_[lo, hi]
        out["center"] = (lo + hi) / 2
        out["area"] = float(np.prod(np.maximum(hi - lo, 0)))
    else:
        out["j2d_norm"] = np.full_like(joints, np.nan, dtype=np.float32)
        out["bbox"] = np.zeros(4, np.float32)
        out["center"] = out["loc"]
        out["area"] = 0.0
    return out


def box_iou(a, b):
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:], b[2:])
    inter = np.prod(np.maximum(hi - lo, 0))
    area_a = np.prod(np.maximum(a[2:] - a[:2], 0))
    area_b = np.prod(np.maximum(b[2:] - b[:2], 0))
    return float(inter / max(area_a + area_b - inter, 1e-8))


def match_yolo_tracks(track_map, yolo_result, detections, frame, width, height):
    boxes = yolo_result.boxes
    if boxes.id is None or not detections:
        return
    ids = boxes.id.int().cpu().numpy()
    yolo_boxes = boxes.xyxy.float().cpu().numpy() / np.array([width, height, width, height])
    yolo_centers = (yolo_boxes[:, :2] + yolo_boxes[:, 2:]) / 2
    cost = np.empty((len(ids), len(detections)), np.float32)
    for i, (yb, yc) in enumerate(zip(yolo_boxes, yolo_centers)):
        for j, det in enumerate(detections):
            cost[i, j] = np.linalg.norm(yc - det["center"]) - 0.15 * box_iou(yb, det["bbox"])
    rows, cols = linear_sum_assignment(cost)
    for i, j in zip(rows, cols):
        if cost[i, j] > 0.30:
            continue
        track_id = int(ids[i])
        if track_id not in track_map:
            track_map[track_id] = Track(track_id)
        det = detections[j]
        # Use the more stable detector box for tracking/selection statistics.
        det["bbox"] = yolo_boxes[i]
        det["center"] = yolo_centers[i]
        det["area"] = float(np.prod(yolo_boxes[i, 2:] - yolo_boxes[i, :2]))
        track_map[track_id].observations[frame] = det


def associate(tracks, detections, frame, max_gap=12, max_cost=0.28):
    active = [t for t in tracks if frame - t.last_frame <= max_gap]
    if active and detections:
        cost = np.empty((len(active), len(detections)), np.float32)
        for i, track in enumerate(active):
            prev = track.last
            for j, det in enumerate(detections):
                image_dist = np.linalg.norm(prev["center"] - det["center"])
                prev_z = float(prev["transl_pelvis"].reshape(-1)[2])
                det_z = float(det["transl_pelvis"].reshape(-1)[2])
                depth_dist = abs(prev_z - det_z) / max(abs(prev_z), 1.0)
                cost[i, j] = image_dist + 0.08 * depth_dist
        rows, cols = linear_sum_assignment(cost)
        used = set()
        for i, j in zip(rows, cols):
            if cost[i, j] <= max_cost:
                active[i].observations[frame] = detections[j]
                used.add(j)
    else:
        used = set()
    for index, det in enumerate(detections):
        if index not in used:
            track = Track(len(tracks) + 1)
            track.observations[frame] = det
            tracks.append(track)


def track_stats(track, num_frames):
    frames = np.array(sorted(track.observations))
    obs = [track.observations[int(f)] for f in frames]
    centers = np.stack([x["center"] for x in obs])
    motion = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.zeros(1)
    guard_scores = []
    for item in obs:
        joints = item.get("j2d_norm")
        if joints is None or len(joints) <= 21:
            continue
        top, bottom = item["bbox"][1], item["bbox"][3]
        height = max(bottom - top, 1e-5)
        # SMPL-X wrists 20/21. Boxers keep both hands in the upper 62% of
        # their body box much more consistently than referees/spectators.
        wrist_level = (joints[[20, 21], 1] - top) / height
        guard_scores.append(float(np.mean(wrist_level < 0.62)))
    return {
        "track_id": track.track_id,
        "frames": len(frames),
        "coverage": len(frames) / num_frames,
        "first": int(frames[0]),
        "last": int(frames[-1]),
        "median_score": float(np.median([x["scores"] for x in obs])),
        "median_area": float(np.median([x["area"] for x in obs])),
        "median_center_y": float(np.median(centers[:, 1])),
        "motion": float(np.sum(np.clip(motion, 0, 0.15))),
        "boxing_guard": float(np.mean(guard_scores)) if guard_scores else 0.0,
    }


def stitch_tracklets(tracks, max_gap=18, max_overlap=10):
    """Join detector IDs that switch while following the same nearby person."""
    tracks = sorted(tracks, key=lambda t: (min(t.observations), -len(t.observations)))
    changed = True
    while changed:
        changed = False
        for i, first in enumerate(tracks):
            first_start, first_end = min(first.observations), max(first.observations)
            best = None
            for j, second in enumerate(tracks):
                if i == j:
                    continue
                second_start = min(second.observations)
                if second_start <= first_start or second_start - first_end > max_gap:
                    continue
                if first_end - second_start > max_overlap:
                    continue
                fa = first.observations[min(first.observations, key=lambda f: abs(f - second_start))]
                fb = second.observations[min(second.observations, key=lambda f: abs(f - first_end))]
                center_dist = float(np.linalg.norm(fa["center"] - fb["center"]))
                area_ratio = abs(np.log(max(fa["area"], 1e-6) / max(fb["area"], 1e-6)))
                za = float(fa["transl_pelvis"].reshape(-1)[2])
                zb = float(fb["transl_pelvis"].reshape(-1)[2])
                depth_delta = abs(za - zb) / max(abs(za), abs(zb), 1.0)
                cost = center_dist + 0.06 * area_ratio + 0.08 * depth_delta
                if center_dist < 0.24 and area_ratio < 1.5 and depth_delta < 0.60:
                    if best is None or cost < best[0]:
                        best = (cost, j, second)
            if best is None:
                continue
            _, j, second = best
            for frame, obs in second.observations.items():
                old = first.observations.get(frame)
                if old is None or float(obs["scores"].reshape(-1)[0]) > float(old["scores"].reshape(-1)[0]):
                    first.observations[frame] = obs
            tracks.pop(j)
            changed = True
            break
    return tracks


def select_boxers(tracks, num_frames):
    viable = [t for t in tracks if len(t.observations) >= max(5, int(num_frames * 0.20))]
    guarded = [t for t in viable if track_stats(t, num_frames)["boxing_guard"] >= 0.45]
    if len(guarded) >= 2:
        # Fighters are normally the two dominant, lower-in-frame moving
        # bodies. Restricting to the two strongest candidates prevents a
        # persistent central referee from winning on coverage alone.
        def prominence(track):
            stats = track_stats(track, num_frames)
            return (
                stats["median_area"]
                * max(stats["median_center_y"], 0.05) ** 2
                * (1.0 + 0.15 * np.log1p(stats["motion"]))
            )

        ranked = sorted(guarded, key=prominence, reverse=True)
        if len(ranked) >= 3 and prominence(ranked[2]) > 0.82 * prominence(ranked[1]):
            raise RuntimeError(
                "ambiguous third active person (possible referee/spectator); refusing automatic export"
            )
        viable = ranked[:2]
    if len(viable) < 2:
        raise RuntimeError(f"only {len(viable)} persistent person tracks; need two")
    best = None
    for i, left in enumerate(viable):
        sl = track_stats(left, num_frames)
        for right in viable[i + 1:]:
            sr = track_stats(right, num_frames)
            common = sorted(set(left.observations) & set(right.observations))
            overlap = len(common) / num_frames
            if common:
                distances = [np.linalg.norm(left.observations[f]["center"] - right.observations[f]["center"]) for f in common]
                proximity = float(np.mean(np.exp(-np.asarray(distances) / 0.30)))
            else:
                proximity = 0.0
            # Persistent, large, moving people that remain near one another are
            # much more likely to be the two fighters than spectators/referee.
            score = (
                2.4 * (sl["coverage"] + sr["coverage"])
                + 0.8 * overlap
                + 0.5 * proximity
                + 10.0 * (sl["median_area"] + sr["median_area"])
                + 1.5 * (sl["median_center_y"] + sr["median_center_y"])
                + 2.0 * (sl["boxing_guard"] + sr["boxing_guard"])
                + 0.12 * np.log1p(sl["motion"] + sr["motion"])
            )
            candidate = (score, left, right, overlap, proximity)
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best


def stabilize_pair_identities(left, right):
    """Keep boxer identities fixed when detector IDs swap during a turn/crossing."""
    common = sorted(set(left.observations) & set(right.observations))
    if len(common) < 2:
        return 0

    def feature(obs):
        trans = obs["transl_pelvis"].reshape(3).astype(np.float32)
        shape = obs["shape"].reshape(-1).astype(np.float32)
        shape = shape[:10] / 2.5
        return trans, shape

    first = common[0]
    prev_left = feature(left.observations[first])
    prev_right = feature(right.observations[first])
    swaps = 0
    for frame in common[1:]:
        left_obs = left.observations[frame]
        right_obs = right.observations[frame]
        cur_left = feature(left_obs)
        cur_right = feature(right_obs)

        def transition_cost(previous, current):
            position = np.linalg.norm(previous[0] - current[0])
            shape = np.linalg.norm(previous[1] - current[1])
            return float(position + 0.35 * shape)

        same = transition_cost(prev_left, cur_left) + transition_cost(prev_right, cur_right)
        crossed = transition_cost(prev_left, cur_right) + transition_cost(prev_right, cur_left)
        # Require a meaningful margin so noisy HMR translations do not cause
        # identity oscillation while the athletes are close together.
        if crossed + 0.08 < same:
            left.observations[frame], right.observations[frame] = right_obs, left_obs
            cur_left, cur_right = cur_right, cur_left
            swaps += 1
        prev_left, prev_right = cur_left, cur_right
    return swaps


def interpolate_values(frames, values, all_frames):
    result = np.empty((len(all_frames), values.shape[1]), np.float32)
    for d in range(values.shape[1]):
        result[:, d] = np.interp(all_frames, frames, values[:, d])
    return result


def interpolate_joints_2d(frames, observations, all_frames, joint_count=22):
    """Interpolate indexed 2D joints while preserving missing observations."""
    values = np.full((len(frames), joint_count, 2), np.nan, np.float32)
    for index, observation in enumerate(observations):
        joints = np.asarray(observation.get("j2d_norm", []), np.float32).reshape(-1, 2)
        count = min(len(joints), joint_count)
        if count:
            values[index, :count] = joints[:count]
    result = np.full((len(all_frames), joint_count, 2), np.nan, np.float32)
    for joint in range(joint_count):
        for coordinate in range(2):
            valid = np.isfinite(values[:, joint, coordinate])
            if np.count_nonzero(valid) >= 2:
                result[:, joint, coordinate] = np.interp(
                    all_frames, frames[valid], values[valid, joint, coordinate]
                )
            elif np.count_nonzero(valid) == 1:
                result[:, joint, coordinate] = values[valid, joint, coordinate][0]
    return result


def interpolate_rotvec(frames, rotvec, all_frames):
    out = np.empty((len(all_frames), rotvec.shape[1], 3), np.float32)
    for joint in range(rotvec.shape[1]):
        rotations = Rotation.from_rotvec(rotvec[:, joint])
        if len(frames) == 1:
            out[:, joint] = rotvec[0, joint]
            continue
        query = np.clip(all_frames, frames[0], frames[-1])
        out[:, joint] = Slerp(frames, rotations)(query).as_rotvec()
    return out


def smooth(array, window=7):
    if len(array) < 5:
        return array
    window = min(window, len(array) if len(array) % 2 else len(array) - 1)
    return savgol_filter(array, window, 2, axis=0, mode="interp").astype(np.float32)


def repair_translation_jumps(translations, max_step=0.075):
    """Repair detector switches and enforce a plausible per-frame root speed."""
    local_median = median_filter(translations, size=(9, 1), mode="nearest")
    repaired = translations.copy()
    outlier = np.linalg.norm(translations - local_median, axis=1) > 0.18
    repaired[outlier] = local_median[outlier]
    if len(repaired) >= 11:
        repaired = savgol_filter(repaired, 11, 2, axis=0, mode="interp").astype(np.float32)
    for frame in range(1, len(repaired)):
        delta = repaired[frame] - repaired[frame - 1]
        distance = float(np.linalg.norm(delta))
        if distance > max_step:
            repaired[frame] = repaired[frame - 1] + delta * (max_step / distance)
    return repaired, int(outlier.sum())


def compute_smplx_body_joints(archives):
    """Forward SMPL-X archives and return the first 22 body joints."""
    frames = len(archives[0]["transl"])
    body_model = smplx.create(
        str(SMPLX_MODEL_ROOT), model_type="smplx", gender="neutral", ext="npz",
        use_pca=False, num_betas=10, batch_size=frames,
    ).eval()
    result = []
    for archive in archives:
        if len(archive["transl"]) != frames:
            raise RuntimeError("the two boxer tracks have different frame counts")
        zeros3 = torch.zeros((frames, 3), dtype=torch.float32)
        betas = np.asarray(archive["betas"], np.float32).reshape(1, -1)[:, :10]
        betas = np.repeat(betas, frames, axis=0)
        with torch.inference_mode():
            output = body_model(
                global_orient=torch.from_numpy(np.asarray(archive["global_orient"], np.float32)),
                body_pose=torch.from_numpy(np.asarray(archive["body_pose"], np.float32)),
                left_hand_pose=torch.from_numpy(np.asarray(archive["left_hand_pose"], np.float32)),
                right_hand_pose=torch.from_numpy(np.asarray(archive["right_hand_pose"], np.float32)),
                jaw_pose=torch.from_numpy(np.asarray(archive["jaw_pose"], np.float32)),
                leye_pose=zeros3, reye_pose=zeros3,
                expression=torch.zeros((frames, 10), dtype=torch.float32),
                betas=torch.from_numpy(betas),
                transl=torch.from_numpy(np.asarray(archive["transl"], np.float32)),
                return_verts=False,
            )
        result.append(output.joints[:, :22].cpu().numpy())
    return result


def align_projected_punch_contacts(paths, max_ground_correction=2.0):
    """Align confident 2D punch contacts along camera depth in 3D.

    The monocular network estimates each person's depth independently.  A
    punch can therefore overlap the target in the image yet miss in a top
    view.  Detect only extended-arm image contacts and adjust the pair's
    relative depth smoothly around those instants while preserving its center.
    """
    archives = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            archives.append({key: data[key].copy() for key in data.files})
    if not all("source_joints_2d" in archive for archive in archives):
        return {
            "enabled": True, "accepted": False, "contact_events": 0,
            "reason": "missing indexed source 2D joints; regenerate with current pipeline",
        }

    frames = len(archives[0]["transl"])
    joints_2d = [np.asarray(a["source_joints_2d"], np.float32) for a in archives]
    joints_3d = compute_smplx_body_joints(archives)
    # Head/neck/upper torso/arms/fists are valid boxing targets.  Arms and
    # fists may participate in contact here, but remain excluded from the body
    # penetration rejection test below.
    target_joints = np.array([9, 12, 15, 16, 17, 18, 19, 20, 21])
    arm_chains = ((16, 18, 20), (17, 19, 21))
    candidates = []
    for attacker in (0, 1):
        defender = 1 - attacker
        a2, d2 = joints_2d[attacker], joints_2d[defender]
        for shoulder, elbow, wrist in arm_chains:
            for frame in range(frames):
                required = np.r_[a2[frame, [shoulder, elbow, wrist]].reshape(-1),
                                 d2[frame, target_joints].reshape(-1)]
                if not np.isfinite(required).all():
                    continue
                defender_height = float(
                    np.percentile(d2[frame, :22, 1], 95)
                    - np.percentile(d2[frame, :22, 1], 5)
                )
                if defender_height <= 1e-4:
                    continue
                upper = a2[frame, shoulder] - a2[frame, elbow]
                lower = a2[frame, wrist] - a2[frame, elbow]
                cosine = float(np.dot(upper, lower) / max(
                    np.linalg.norm(upper) * np.linalg.norm(lower), 1e-8
                ))
                extension = float(
                    np.linalg.norm(a2[frame, wrist] - a2[frame, shoulder])
                    / defender_height
                )
                target_distances = np.linalg.norm(
                    d2[frame, target_joints] - a2[frame, wrist], axis=1
                ) / defender_height
                nearest = int(np.argmin(target_distances))
                projected_distance = float(target_distances[nearest])
                # >120-degree elbow plus close projected target: conservative
                # evidence of a strike/contact, rather than hands in guard.
                if cosine < -0.35 and extension > 0.18 and projected_distance < 0.18:
                    candidates.append({
                        "frame": frame, "attacker": attacker, "wrist": wrist,
                        "target": int(target_joints[nearest]),
                        "projected_distance": projected_distance,
                    })

    # Non-maximum suppression: retain the closest instant of each contact and
    # prevent one punch from producing many adjacent anchors.
    events = []
    for item in sorted(candidates, key=lambda x: x["projected_distance"]):
        if any(
            old["attacker"] == item["attacker"]
            and old["wrist"] == item["wrist"]
            and abs(old["frame"] - item["frame"]) <= 6
            for old in events
        ):
            continue
        events.append(item)
    events.sort(key=lambda x: x["frame"])
    root_ground_distance = np.linalg.norm(
        (archives[1]["transl"] - archives[0]["transl"])[:, :2], axis=1
    )
    ignored_body_close_events = [
        event for event in events if root_ground_distance[event["frame"]] < 0.50
    ]
    ignored_body_close_frames = sorted({
        int(event["frame"]) for event in ignored_body_close_events
    })
    events = [
        event for event in events if root_ground_distance[event["frame"]] >= 0.50
    ]

    correction_sum = np.zeros((frames, 2), np.float32)
    correction_weight = np.zeros(frames, np.float32)
    contact_before = []
    for event in events:
        frame, attacker, wrist, target = (
            event["frame"], event["attacker"], event["wrist"], event["target"]
        )
        defender = 1 - attacker
        wrist_position = joints_3d[attacker][frame, wrist]
        target_position = joints_3d[defender][frame, target]
        before = float(np.linalg.norm(wrist_position - target_position))
        contact_before.append(before)
        # Desired joint-center distance approximates glove-to-target *surface*
        # contact.  Torso/head targets need a larger radius than wrists; aiming
        # at the torso center would pull the athletes into each other.
        if target in (9, 12):
            contact_radius = 0.26
        elif target == 15:
            contact_radius = 0.20
        elif target in (16, 17):
            contact_radius = 0.18
        else:
            contact_radius = 0.10
        vertical_gap = float(target_position[2] - wrist_position[2])
        desired_ground_length = np.sqrt(max(contact_radius ** 2 - vertical_gap ** 2, 0.0))
        current_ground_gap = target_position[:2] - wrist_position[:2]
        current_ground_length = float(np.linalg.norm(current_ground_gap))
        direction = (
            current_ground_gap / current_ground_length
            if current_ground_length > 1e-6
            else np.array([0.0, 1.0], np.float32)
        )
        desired_ground_gap = direction * desired_ground_length
        # Convert target-wrist correction to change in root2_xy-root1_xy.
        change = desired_ground_gap - current_ground_gap
        if attacker == 1:
            change = -change
        change_length = float(np.linalg.norm(change))
        if change_length > max_ground_correction:
            change *= max_ground_correction / change_length
        event["distance_3d_before_m"] = before
        event["relative_ground_correction_xy_m"] = change.tolist()
        # Give a large monocular-depth correction enough lead-in/lead-out time
        # to remain below the per-frame motion ceiling.
        radius = max(7, int(np.ceil(np.linalg.norm(change) / 0.035)) + 4)
        for sample in range(max(0, frame - radius), min(frames, frame + radius + 1)):
            weight = float(np.exp(-0.5 * ((sample - frame) / 3.0) ** 2))
            correction_sum[sample] += change * weight
            correction_weight[sample] += weight

    correction = np.divide(
        correction_sum, correction_weight[:, None],
        out=np.zeros_like(correction_sum), where=correction_weight[:, None] > 0,
    )
    if frames >= 5:
        correction = smooth(correction, window=7)
    # Avoid introducing a ground-plane jump even when contact anchors disagree.
    # This is pair-relative motion, so each athlete receives only half.
    max_pair_step = 0.060
    for frame in range(1, frames):
        step = correction[frame] - correction[frame - 1]
        length = float(np.linalg.norm(step))
        if length > max_pair_step:
            correction[frame] = correction[frame - 1] + step * (max_pair_step / length)
    for frame in range(frames - 2, -1, -1):
        step = correction[frame] - correction[frame + 1]
        length = float(np.linalg.norm(step))
        if length > max_pair_step:
            correction[frame] = correction[frame + 1] + step * (max_pair_step / length)
    archives[0]["transl"][:, :2] -= correction * 0.5
    archives[1]["transl"][:, :2] += correction * 0.5

    # Re-evaluate the constrained contacts.  A remaining large 3D miss means
    # image evidence and reconstructed geometry disagree; reject rather than
    # silently exporting a misleading top view.
    aligned_joints = compute_smplx_body_joints(archives)
    contact_after = []
    for event in events:
        defender = 1 - event["attacker"]
        separation = (
            aligned_joints[event["attacker"]][event["frame"], event["wrist"]]
            - aligned_joints[defender][event["frame"], event["target"]]
        )
        distance = float(np.linalg.norm(separation))
        ground_distance = float(np.linalg.norm(separation[:2]))
        event["distance_3d_after_m"] = distance
        event["ground_distance_after_m"] = ground_distance
        event["contact_tolerance_m"] = (
            0.32 if event["target"] in (9, 12, 15, 16, 17) else 0.24
        )
        event["ground_contact_tolerance_m"] = (
            0.30 if event["target"] in (9, 12) else
            0.24 if event["target"] in (15, 16, 17) else 0.18
        )
        contact_after.append(distance)
    bad_contact_frames = [
        int(event["frame"]) for event in events
        if event["ground_distance_after_m"] > event["ground_contact_tolerance_m"]
        or event["distance_3d_after_m"] > event["contact_tolerance_m"]
    ]
    bad_contact_frames = sorted(set(bad_contact_frames + ignored_body_close_frames))
    for archive, path in zip(archives, paths):
        archive["contact_ground_plane_aligned"] = np.bool_(bool(events))
        archive["contact_event_count"] = np.int32(len(events))
        np.savez_compressed(path, **archive)
    return {
        "enabled": True,
        "method": "2D extended-arm contact anchors with symmetric 3D ground-plane refinement",
        "contact_events": len(events),
        "ignored_body_close_contact_events": len(ignored_body_close_events),
        "ignored_body_close_frames": ignored_body_close_frames,
        "events": events,
        "max_ground_correction_m": float(np.linalg.norm(correction, axis=1).max()) if frames else 0.0,
        "mean_contact_distance_before_m": float(np.mean(contact_before)) if contact_before else None,
        "mean_contact_distance_after_m": float(np.mean(contact_after)) if contact_after else None,
        "p90_contact_distance_after_m": float(np.percentile(contact_after, 90)) if contact_after else None,
        "bad_contact_frames": bad_contact_frames,
        "accepted": True,
        "reason": "ok" if not bad_contact_frames else
                  "unresolved 3D misses will be removed by the segment filter",
    }


def trim_pair_collisions(
    paths, fps, penetration_ratio=0.70, padding_frames=2,
    min_clean_seconds=1.0, extra_reject_frames=(),
):
    """Remove gross body/body intersections without fabricating motion.

    SMPL-X core joints form capsules for torso, head and legs.  Arms, wrists
    and hands are deliberately excluded, so glove/glove and glove/body contact
    remain valid boxing data.  Only deep core/core penetration is rejected.
    Every sufficiently long contiguous clean run is saved separately.
    """
    archives = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            archives.append({key: data[key].copy() for key in data.files})

    first = archives[0]["transl"].astype(np.float32, copy=True)
    second = archives[1]["transl"].astype(np.float32, copy=True)
    if len(first) != len(second):
        raise RuntimeError("the two boxer tracks have different frame counts")

    frames = len(first)
    joints = compute_smplx_body_joints(archives)
    # (joint A, joint B, radius metres).  No shoulder-to-wrist or hand
    # capsules appear here: punches can touch or enter the opponent surface
    # slightly without causing the boxing exchange to be deleted.
    core_segments = (
        (0, 3, 0.16), (3, 6, 0.17), (6, 9, 0.18),
        (9, 12, 0.14), (12, 15, 0.12),
        (1, 2, 0.13), (16, 17, 0.13),
        (1, 4, 0.11), (4, 7, 0.09),
        (2, 5, 0.11), (5, 8, 0.09),
    )

    def sampled_capsules(person_joints):
        weights = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        points, radii = [], []
        for joint_a, joint_b, radius in core_segments:
            points.append(
                person_joints[:, joint_a, None] * (1.0 - weights[None, :, None])
                + person_joints[:, joint_b, None] * weights[None, :, None]
            )
            radii.append(radius)
        return np.stack(points, axis=1), np.asarray(radii, np.float32)

    capsules_a, radii_a = sampled_capsules(joints[0])
    capsules_b, radii_b = sampled_capsules(joints[1])
    centerline_distance = np.linalg.norm(
        capsules_a[:, :, None, :, None] - capsules_b[:, None, :, None, :],
        axis=-1,
    ).min(axis=(3, 4))
    clearance_ratio = centerline_distance / (
        radii_a[None, :, None] + radii_b[None, None, :]
    )
    deepest_ratio = clearance_ratio.min(axis=(1, 2))
    penetrating_pairs = np.count_nonzero(clearance_ratio < penetration_ratio, axis=(1, 2))
    # One very deep intersection, or two simultaneous core-capsule overlaps,
    # indicates body-through-body failure.  Require persistence across two
    # frames so one noisy fit does not remove a valid punch.
    collision = (deepest_ratio < penetration_ratio * 0.65) | (penetrating_pairs >= 2)
    if len(collision) > 1:
        persistent = np.convolve(collision.astype(np.int16), np.ones(3, np.int16), mode="same") >= 2
        collision &= persistent
    body_collision_frames = int(np.count_nonzero(collision))
    geometry_reject = np.zeros(frames, dtype=bool)
    for frame in extra_reject_frames:
        if 0 <= int(frame) < frames:
            geometry_reject[int(frame)] = True
    collision |= geometry_reject
    raw_collision_frames = int(np.count_nonzero(collision))
    if padding_frames > 0 and np.any(collision):
        kernel = np.ones(2 * padding_frames + 1, dtype=np.int16)
        collision = np.convolve(collision.astype(np.int16), kernel, mode="same") > 0

    clean = ~collision
    runs = []
    start = None
    for frame, is_clean in enumerate(np.r_[clean, False]):
        if is_clean and start is None:
            start = frame
        elif not is_clean and start is not None:
            runs.append((start, frame))
            start = None
    required_frames = max(2, int(np.ceil(min_clean_seconds * fps)))
    eligible = [run for run in runs if run[1] - run[0] >= required_frames]

    # SMPL-X body-pose joints 16..21 are shoulders, elbows and wrists.  Use
    # their robust frame-to-frame rotation as an action cue.  Engagement
    # proximity downweights a long segment where both athletes merely reset.
    def segment_score(run):
        begin, end = run
        arm_motion = []
        for archive in archives:
            pose = archive["body_pose"].reshape(len(first), 21, 3)[begin:end, 15:21]
            if len(pose) < 2:
                arm_motion.append(0.0)
                continue
            relative_rotation = (
                Rotation.from_rotvec(pose[:-1].reshape(-1, 3)).inv()
                * Rotation.from_rotvec(pose[1:].reshape(-1, 3))
            )
            step_deg = np.rad2deg(relative_rotation.magnitude())
            arm_motion.append(float(np.mean(np.clip(step_deg, 0.0, 25.0))))
        root_distance = np.linalg.norm((second - first)[begin:end, :2], axis=1)
        proximity = float(np.mean(np.exp(-root_distance / 1.2)))
        mean_arm_motion = float(np.mean(arm_motion))
        score = mean_arm_motion * (0.5 + proximity) + 0.015 * (end - begin)
        return score, mean_arm_motion, proximity

    candidate_segments = []
    for run in eligible:
        score, arm_motion, proximity = segment_score(run)
        candidate_segments.append({
            "frame_range": [int(run[0]), int(run[1])],
            "frames": int(run[1] - run[0]),
            "action_score": float(score),
            "mean_arm_step_deg": float(arm_motion),
            "engagement_proximity": float(proximity),
        })
    if candidate_segments:
        selected = max(candidate_segments, key=lambda item: item["action_score"])
        selected_range = selected["frame_range"]
        chosen = (selected_range[0], selected_range[1])
    else:
        chosen = max(runs, key=lambda item: item[1] - item[0]) if runs else (0, 0)
    kept_frames = chosen[1] - chosen[0]
    accepted = bool(candidate_segments)

    temporal_keys = {
        "global_orient", "body_pose", "left_hand_pose", "right_hand_pose",
        "jaw_pose", "transl", "source_joints_2d",
    }
    segments_root = paths[0].parent / "segments"
    if segments_root.is_dir():
        shutil.rmtree(segments_root)

    def write_segment(begin, end, output_paths):
        for archive, path in zip(archives, output_paths):
            sliced = {key: value.copy() for key, value in archive.items()}
            for key in temporal_keys:
                if key in sliced:
                    sliced[key] = sliced[key][begin:end]
            sliced["collision_filter_applied"] = np.bool_(raw_collision_frames > 0)
            sliced["source_frame_start"] = np.int32(begin)
            sliced["source_frame_end_exclusive"] = np.int32(end)
            sliced["body_penetration_ratio"] = np.float32(penetration_ratio)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **sliced)

    if accepted:
        # Keep the highest-action clean segment at the legacy paths so the
        # existing AITViewer command continues to work.
        write_segment(chosen[0], chosen[1], paths)

        # Preserve every other clean boxing interval as its own continuous
        # clip.  We never concatenate across a removed body-intersection gap.
        for index, candidate in enumerate(candidate_segments, 1):
            begin, end = candidate["frame_range"]
            segment_dir = segments_root / f"segment_{index:03d}"
            segment_paths = [
                segment_dir / "person_1_smplx.npz",
                segment_dir / "person_2_smplx.npz",
            ]
            write_segment(begin, end, segment_paths)
            candidate["output_subdir"] = str(segment_dir.relative_to(paths[0].parent))
    else:
        # A rejected clip must not leave replayable contaminated tracks in the
        # clean dataset.  The report and conversion log remain for auditing.
        for path in paths:
            path.unlink(missing_ok=True)

    return {
        "enabled": True,
        "method": "SMPL-X torso/head/leg capsules; arms and fists excluded",
        "body_penetration_ratio": float(penetration_ratio),
        "padding_frames": int(padding_frames),
        "minimum_clean_frames": int(required_frames),
        "body_collision_frames": body_collision_frames,
        "raw_collision_frames": raw_collision_frames,
        "unresolved_contact_frames": int(np.count_nonzero(geometry_reject)),
        "padded_rejected_frames": int(np.count_nonzero(collision)),
        "deepest_core_clearance_ratio": float(deepest_ratio.min()) if len(first) else 0.0,
        "selection_policy": "boxing arm action plus engagement proximity",
        "candidate_segments": candidate_segments,
        "kept_source_frame_range": [int(chosen[0]), int(chosen[1])],
        "kept_frames": int(kept_frames),
        "trimmed": bool(raw_collision_frames > 0),
        "accepted": bool(accepted),
    }


def temporal_filter_rotations(rotvec):
    """Suppress monocular pose flips in quaternion space without averaging axes."""
    frames, joints, _ = rotvec.shape
    if frames < 3:
        return rotvec
    quats = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_quat().reshape(frames, joints, 4)
    # q and -q encode the same rotation. Unwrap signs before filtering.
    for t in range(1, frames):
        flip = np.sum(quats[t] * quats[t - 1], axis=1) < 0
        quats[t, flip] *= -1

    candidate = median_filter(quats, size=(5, 1, 1), mode="nearest")
    candidate /= np.maximum(np.linalg.norm(candidate, axis=2, keepdims=True), 1e-8)
    delta = (
        Rotation.from_quat(candidate.reshape(-1, 4)).inv()
        * Rotation.from_quat(quats.reshape(-1, 4))
    ).magnitude().reshape(frames, joints)
    # Root flips are implausible at video rate; limbs retain a looser limit
    # so fast punches are not flattened.
    thresholds = np.full(joints, np.deg2rad(48.0))
    thresholds[0] = np.deg2rad(25.0)
    replace = delta > thresholds[None]
    quats[replace] = candidate[replace]

    if frames >= 5:
        window = min(5, frames if frames % 2 else frames - 1)
        quats = savgol_filter(quats, window, 2, axis=0, mode="interp")
        quats /= np.maximum(np.linalg.norm(quats, axis=2, keepdims=True), 1e-8)

    # Enforce a hard angular-velocity ceiling along the shortest SO(3) path.
    # At 30 FPS these still permit very fast punches while preventing a limb
    # or torso from teleporting between poses.
    filtered = Rotation.from_quat(quats.reshape(-1, 4)).as_rotvec().reshape(frames, joints, 3)
    limits = np.full(joints, np.deg2rad(30.0))
    limits[0] = np.deg2rad(15.0)
    if joints > 22:
        limits[22:] = np.deg2rad(40.0)
    for t in range(1, frames):
        previous = Rotation.from_rotvec(filtered[t - 1])
        current = Rotation.from_rotvec(filtered[t])
        delta_rotation = previous.inv() * current
        delta = delta_rotation.as_rotvec()
        angle = np.linalg.norm(delta, axis=1)
        scale = np.minimum(1.0, limits / np.maximum(angle, 1e-8))
        filtered[t] = (previous * Rotation.from_rotvec(delta * scale[:, None])).as_rotvec()
    return filtered.astype(np.float32)


def rotation_quality(rotvec):
    if len(rotvec) < 2:
        return {"p95_step_deg": 0.0, "max_step_deg": 0.0, "steps_over_45_deg": 0}
    rel = (
        Rotation.from_rotvec(rotvec[:-1].reshape(-1, 3)).inv()
        * Rotation.from_rotvec(rotvec[1:].reshape(-1, 3))
    )
    per_frame = np.rad2deg(rel.magnitude()).reshape(len(rotvec) - 1, -1).max(axis=1)
    return {
        "p95_step_deg": float(np.percentile(per_frame, 95)),
        "max_step_deg": float(per_frame.max()),
        "steps_over_45_deg": int(np.count_nonzero(per_frame > 45.0)),
    }


def export_track(track, num_frames, fps, path, common_origin, fist_hands=None):
    frames = np.array(sorted(track.observations), dtype=np.float64)
    obs = [track.observations[int(f)] for f in frames]
    all_frames = np.arange(num_frames, dtype=np.float64)
    rotations = np.stack([x["rotvec"] for x in obs]).astype(np.float32)
    rotations = interpolate_rotvec(frames, rotations, all_frames)
    rotations = temporal_filter_rotations(rotations)
    if fist_hands is not None:
        rotations[:, 22:37] = fist_hands[0][None]
        rotations[:, 37:52] = fist_hands[1][None]
    translations = np.stack([x["transl_pelvis"].reshape(3) for x in obs]).astype(np.float32)
    translations = smooth(interpolate_values(frames, translations, all_frames))
    joints_2d = interpolate_joints_2d(frames, obs, all_frames)

    # Camera (x right, y down, z forward) -> Z-up world (x right, y forward, z up).
    basis = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float32)
    root_mats = Rotation.from_rotvec(rotations[:, 0]).as_matrix()
    rotations[:, 0] = Rotation.from_matrix(basis[None] @ root_mats).as_rotvec()
    translations = translations @ basis.T - common_origin
    translations, translation_repairs = repair_translation_jumps(translations)
    betas = np.median(np.stack([x["shape"] for x in obs]), axis=0).astype(np.float32)

    np.savez_compressed(
        path,
        global_orient=rotations[:, 0],
        body_pose=rotations[:, 1:22].reshape(num_frames, 63),
        left_hand_pose=rotations[:, 22:37].reshape(num_frames, 45),
        right_hand_pose=rotations[:, 37:52].reshape(num_frames, 45),
        jaw_pose=rotations[:, 52],
        transl=translations,
        source_joints_2d=joints_2d,
        betas=betas[:10],
        fps=np.float32(fps),
        source_track_id=np.int32(track.track_id),
        fixed_fists=np.bool_(fist_hands is not None),
        translation_repairs=np.int32(translation_repairs),
    )
    quality = rotation_quality(rotations)
    quality["translation_repairs"] = translation_repairs
    quality["max_translation_step_m"] = float(
        np.linalg.norm(np.diff(translations, axis=0), axis=1).max() if num_frames > 1 else 0.0
    )
    return quality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", default="multiHMR_672_S")
    parser.add_argument("--yolo-model", type=Path, default=ELEMENTS_ROOT / "yolo11s.pt")
    parser.add_argument("--det-thresh", type=float, default=0.25)
    parser.add_argument("--nms-kernel", type=int, default=3)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--fist-template-dir", type=Path,
        default=Path("/home/ubuntu22/boxing_smplx_duel_sample"),
        help="directory containing two stable boxing SMPL-X tracks used to form fixed fists",
    )
    parser.add_argument("--animated-hands", action="store_true", help="keep estimated per-frame finger poses")
    parser.add_argument(
        "--body-penetration-ratio", type=float, default=0.70,
        help="core-capsule penetration threshold; arms and fists are excluded",
    )
    parser.add_argument(
        "--collision-padding", type=int, default=2,
        help="also reject this many frames on each side of a collision",
    )
    parser.add_argument(
        "--min-clean-seconds", type=float, default=1.0,
        help="reject the clip unless a contiguous clean segment this long remains",
    )
    parser.add_argument(
        "--max-contact-ground-correction", type=float, default=2.0,
        help="maximum pair ground-plane correction at a punch contact (metres)",
    )
    args = parser.parse_args()

    # The upstream demo resolves checkpoints and mean parameters relative to
    # its repository rather than relative to this script.
    os.chdir(MULTI_HMR)

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; fix NVIDIA driver/device nodes first")
    cap = cv2.VideoCapture(str(args.input_video))
    if not cap.isOpened():
        raise FileNotFoundError(args.input_video)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, args.max_frames) if args.max_frames else total
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model, device=device).eval()
    yolo = YOLO(str(args.yolo_model))
    image_size = int(model.img_size)
    camera = get_camera_parameters(image_size, device=device)
    track_map = {}
    frame_index = 0
    processed = 0
    temp = args.output_dir / "_frame.jpg"
    while frame_index < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % args.stride == 0:
            cv2.imwrite(str(temp), frame)
            tensor, _ = open_image(temp, image_size, device=device)
            yolo_result = yolo.track(
                frame, classes=[0], persist=True, tracker="bytetrack.yaml",
                verbose=False, conf=0.25,
            )[0]
            humans = forward_model(model, tensor, camera, args.det_thresh, args.nms_kernel)
            detections = [
                detection_to_numpy(h, image_size, frame.shape[1], frame.shape[0])
                for h in humans
            ]
            match_yolo_tracks(
                track_map, yolo_result, detections, processed,
                frame.shape[1], frame.shape[0],
            )
            processed += 1
            if processed % 50 == 0:
                print(f"frames={processed} tracks={len(track_map)} detections={len(detections)}", flush=True)
        frame_index += 1
    cap.release()
    temp.unlink(missing_ok=True)
    if processed == 0:
        raise RuntimeError("no frames processed")

    tracks = stitch_tracklets(list(track_map.values()))
    score, boxer1, boxer2, overlap, proximity = select_boxers(tracks, processed)
    identity_swaps = stabilize_pair_identities(boxer1, boxer2)
    first_common = sorted(set(boxer1.observations) & set(boxer2.observations))
    origin_frame = first_common[0] if first_common else 0
    pair_trans = []
    for track in (boxer1, boxer2):
        nearest = min(track.observations, key=lambda f: abs(f - origin_frame))
        t = track.observations[nearest]["transl_pelvis"].reshape(3)
        pair_trans.append(np.array([t[0], t[2], -t[1]], np.float32))
    common_origin = np.mean(pair_trans, axis=0)
    out_fps = source_fps / args.stride
    fist_hands = None
    if not args.animated_hands:
        hand_samples = {"left_hand_pose": [], "right_hand_pose": []}
        for index in (1, 2):
            template_path = args.fist_template_dir / f"person_{index}_smplx.npz"
            with np.load(template_path, allow_pickle=False) as template:
                for key in hand_samples:
                    hand_samples[key].append(template[key].reshape(-1, 15, 3))
        fist_hands = (
            np.median(np.concatenate(hand_samples["left_hand_pose"], axis=0), axis=0).astype(np.float32),
            np.median(np.concatenate(hand_samples["right_hand_pose"], axis=0), axis=0).astype(np.float32),
        )
    output_paths = [
        args.output_dir / "person_1_smplx.npz",
        args.output_dir / "person_2_smplx.npz",
    ]
    temporal_quality = [
        export_track(boxer1, processed, out_fps, output_paths[0], common_origin, fist_hands),
        export_track(boxer2, processed, out_fps, output_paths[1], common_origin, fist_hands),
    ]
    contact_alignment = align_projected_punch_contacts(
        output_paths,
        max_ground_correction=max(float(args.max_contact_ground_correction), 0.0),
    )
    collision_filter = trim_pair_collisions(
        output_paths,
        fps=out_fps,
        penetration_ratio=max(float(args.body_penetration_ratio), 0.0),
        padding_frames=max(int(args.collision_padding), 0),
        min_clean_seconds=max(float(args.min_clean_seconds), 0.0),
        extra_reject_frames=contact_alignment.get("bad_contact_frames", ()),
    )

    selected_stats = [track_stats(boxer1, processed), track_stats(boxer2, processed)]
    coverage_ok = all(x["coverage"] >= 0.70 for x in selected_stats)
    temporal_ok = all(x["p95_step_deg"] <= 45.0 and x["max_step_deg"] <= 90.0 for x in temporal_quality)
    geometry_ok = contact_alignment["accepted"]
    collision_ok = collision_filter["accepted"]
    qc_pass = coverage_ok and temporal_ok and geometry_ok and collision_ok
    if not geometry_ok:
        for path in output_paths:
            path.unlink(missing_ok=True)
        segments_root = args.output_dir / "segments"
        if segments_root.is_dir():
            shutil.rmtree(segments_root)

    report = {
        "pipeline_version": PIPELINE_VERSION,
        "source": str(args.input_video), "source_fps": source_fps,
        "output_fps": out_fps, "frames": collision_filter["kept_frames"],
        "source_processed_frames": processed, "stride": args.stride,
        "selected_tracks": [boxer1.track_id, boxer2.track_id],
        "selected_stats": selected_stats,
        "identity_swaps_corrected": identity_swaps,
        "contact_alignment": contact_alignment,
        "collision_filter": collision_filter,
        "temporal_quality": temporal_quality,
        "pair_score": score, "overlap": overlap, "proximity": proximity,
        "qc_pass": qc_pass,
        "qc_reason": "ok" if qc_pass else (
            "selected boxer coverage below 70%" if not coverage_ok else
            "implausible temporal rotation jumps remain" if not temporal_ok else
            "projected punch contacts do not agree with 3D geometry" if not geometry_ok else
            "no sufficiently long collision-free contiguous segment"
        ),
        "tracks": [track_stats(t, processed) for t in tracks],
    }
    (args.output_dir / "selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
