#!/usr/bin/env python3
"""Batch-convert every clip listed by boxing altview manifests to two SMPL-X tracks."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


CONVERTER = Path(__file__).resolve().with_name("multi_hmr_boxing_video.py")
REQUIRED_PIPELINE_VERSION = "boxing_smplx_v4_spatial_contact"
DEFAULT_ELEMENTS_ROOT = next(
    (
        root for root in (
            Path("/media/ubuntu22/Elements"),
            Path("/media/ubuntu22/Elements1"),
        )
        if (root / "处理后的boxing视频").is_dir()
    ),
    Path("/media/ubuntu22/Elements"),
)


def complete(output_dir: Path) -> bool:
    report = output_dir / "selection_report.json"
    tracks = [output_dir / "person_1_smplx.npz", output_dir / "person_2_smplx.npz"]
    if not report.is_file() or not all(path.is_file() for path in tracks):
        return False
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
        collision = data.get("collision_filter", {})
        contact = data.get("contact_alignment", {})
        # Old outputs did not run the pair-collision cleanliness gate and must
        # be regenerated even if their earlier QC flag happened to be true.
        return bool(
            data.get("pipeline_version") == REQUIRED_PIPELINE_VERSION
            and data.get("qc_pass")
            and collision.get("accepted")
            and str(collision.get("method", "")).startswith("SMPL-X torso/head/leg capsules")
            and contact.get("accepted")
            and str(contact.get("method", "")).startswith("2D extended-arm contact anchors")
        )
    except Exception:
        return False


def discover(root: Path):
    for manifest in sorted(root.rglob("altview/manifest.json")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for clip in data.get("clips", []):
            filename = clip.get("file_name")
            if not filename:
                continue
            video = manifest.parent.parent / filename
            if video.is_file():
                yield manifest, clip, video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", nargs="?", type=Path,
        default=DEFAULT_ELEMENTS_ROOT / "处理后的boxing视频",
    )
    parser.add_argument("--limit", type=int, help="process at most N clips (useful for a trial run)")
    parser.add_argument("--force", action="store_true", help="reprocess outputs that already passed QC")
    parser.add_argument("--stride", type=int, default=1, choices=(1, 2))
    parser.add_argument(
        "--output-root", type=Path,
        help="write all SMPL-X results here while preserving source subdirectories",
    )
    args = parser.parse_args()

    jobs = list(discover(args.root))
    if args.limit is not None:
        jobs = jobs[: args.limit]
    print(f"Found {len(jobs)} clip(s)", flush=True)
    summary = []
    for number, (manifest, clip, video) in enumerate(jobs, 1):
        if args.output_root is None:
            output_dir = manifest.parent / "smplx" / video.stem
        else:
            relative_group = manifest.parent.parent.relative_to(args.root)
            output_dir = args.output_root / relative_group / video.stem
        if not args.force and complete(output_dir):
            print(f"[{number}/{len(jobs)}] SKIP passed: {video}", flush=True)
            summary.append({"video": str(video), "output": str(output_dir), "status": "skipped_passed"})
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "conversion.log"
        print(f"[{number}/{len(jobs)}] RUN {video}", flush=True)
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    sys.executable, str(CONVERTER), str(video), str(output_dir),
                    "--stride", str(args.stride),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        status = "failed"
        qc_pass = False
        report_path = output_dir / "selection_report.json"
        if result.returncode == 0 and report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            qc_pass = bool(report.get("qc_pass"))
            status = "passed" if qc_pass else "rejected_qc"
        entry = {
            "video": str(video), "output": str(output_dir), "status": status,
            "qc_pass": qc_pass, "seconds": round(time.time() - started, 2),
            "log": str(log_path),
        }
        summary.append(entry)
        print(f"[{number}/{len(jobs)}] {status.upper()} -> {output_dir}", flush=True)
        summary_path = (args.output_root or args.root) / "smplx_batch_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    passed = sum(x["status"] in {"passed", "skipped_passed"} for x in summary)
    summary_path = (args.output_root or args.root) / "smplx_batch_summary.json"
    print(f"Done: {passed}/{len(summary)} passed. Summary: {summary_path}")


if __name__ == "__main__":
    main()
