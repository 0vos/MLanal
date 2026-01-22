from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from spatial_capture import get_intrinsics_for_saved_image, read_frames_jsonl


_FRAME_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class ExportResult:
    out_dir: Path
    images_dir: Path
    transforms_path: Path
    num_frames: int


def _parse_frame_number(image_file: str) -> int:
    m = _FRAME_RE.findall(image_file)
    if not m:
        raise ValueError(f"Could not parse frame number from: {image_file}")
    return int(m[-1])


def _scale_intrinsics(k: np.ndarray, sx: float, sy: float) -> tuple[float, float, float, float]:
    fx = float(k[0, 0]) * sx
    fy = float(k[1, 1]) * sy
    cx = float(k[0, 2]) * sx
    cy = float(k[1, 2]) * sy
    return fx, fy, cx, cy


def export_nerfstudio_dataset(
    capture_dir: Path,
    out_dir: Path,
    every_n: int,
    max_frames: Optional[int],
    image_max_size: int,
    normalize_poses: bool,
    translate_thresh_m: float,
    rotate_thresh_deg: float,
) -> ExportResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = list(read_frames_jsonl(capture_dir))
    if not records:
        raise RuntimeError(f"No frames found in {capture_dir}")

    def rot_deg_from_rel(rel: np.ndarray) -> float:
        r = rel[:3, :3].astype(np.float64)
        # Clamp for numeric stability.
        c = (np.trace(r) - 1.0) * 0.5
        c = float(np.clip(c, -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    def trans_m_from_rel(rel: np.ndarray) -> float:
        t = rel[:3, 3].astype(np.float64)
        return float(np.linalg.norm(t))

    selected = []
    last = None
    for i, r in enumerate(records):
        if every_n > 1 and (i % every_n) != 0:
            continue

        if last is None:
            selected.append(r)
            last = r
        else:
            # Frame selection based on pose delta (helps when you capture many frames).
            rel = np.linalg.inv(last.camera_transform_cw_4x4) @ r.camera_transform_cw_4x4
            dt = trans_m_from_rel(rel)
            drot = rot_deg_from_rel(rel)
            if (translate_thresh_m > 0 and dt >= translate_thresh_m) or (rotate_thresh_deg > 0 and drot >= rotate_thresh_deg):
                selected.append(r)
                last = r

        if max_frames is not None and len(selected) >= int(max_frames):
            break

    if not selected:
        raise RuntimeError("No frames selected; check --every-n / --max-frames")

    frames_json = []

    # Collect poses for optional normalization.
    c2w_list = []

    for r in selected:
        src = capture_dir / r.image_file
        if not src.exists():
            raise FileNotFoundError(f"Missing image: {src}")

        frame_idx = _parse_frame_number(r.image_file)
        out_name = f"{frame_idx:06d}.jpg"
        dst = images_dir / out_name

        with Image.open(src) as im:
            im = im.convert("RGB")
            w0, h0 = im.size
            w, h = w0, h0

            if image_max_size and max(w0, h0) > int(image_max_size):
                s = float(image_max_size) / float(max(w0, h0))
                w = max(int(round(w0 * s)), 1)
                h = max(int(round(h0 * s)), 1)
                im = im.resize((w, h), resample=Image.BILINEAR)

            im.save(dst, quality=95)

        k_saved, kw, kh = get_intrinsics_for_saved_image(r, capture_dir)
        # k_saved is defined in the saved JPEG coordinate system (kw,kh). If we resized, scale it.
        sx = float(w) / float(kw)
        sy = float(h) / float(kh)
        fl_x, fl_y, cx, cy = _scale_intrinsics(k_saved, sx=sx, sy=sy)

        c2w = np.asarray(r.camera_transform_cw_4x4, dtype=np.float32)
        if c2w.shape != (4, 4):
            raise ValueError("camera_transform_cw_4x4 must be 4x4")
        c2w_list.append(c2w)

        frames_json.append(
            {
                "file_path": f"images/{out_name}",
                "transform_matrix": c2w.tolist(),
                "w": int(w),
                "h": int(h),
                "fl_x": float(fl_x),
                "fl_y": float(fl_y),
                "cx": float(cx),
                "cy": float(cy),
                "camera_model": "PINHOLE",
            }
        )

    # Normalize translations so the scene sits near origin and has a reasonable scale.
    norm = {
        "enabled": bool(normalize_poses),
        "center": [0.0, 0.0, 0.0],
        "scale": 1.0,
    }

    if normalize_poses:
        t = np.stack([m[:3, 3] for m in c2w_list], axis=0)
        center = t.mean(axis=0)
        rad = np.linalg.norm(t - center[None, :], axis=1)
        scale = float(np.percentile(rad, 90))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0

        norm["center"] = center.astype(float).tolist()
        norm["scale"] = float(scale)

        for f in frames_json:
            m = np.asarray(f["transform_matrix"], dtype=np.float32)
            m[:3, 3] = (m[:3, 3] - center) / scale
            f["transform_matrix"] = m.tolist()

    transforms = {
        "fl_x": float(frames_json[0]["fl_x"]),
        "fl_y": float(frames_json[0]["fl_y"]),
        "cx": float(frames_json[0]["cx"]),
        "cy": float(frames_json[0]["cy"]),
        "w": int(frames_json[0]["w"]),
        "h": int(frames_json[0]["h"]),
        "camera_model": "PINHOLE",
        "frames": frames_json,
        "patrune": {
            "source_capture_dir": str(capture_dir),
            "every_n": int(every_n),
            "max_frames": int(max_frames) if max_frames is not None else None,
            "image_max_size": int(image_max_size),
            "normalize": norm,
        },
    }

    transforms_path = out_dir / "transforms.json"
    transforms_path.write_text(json.dumps(transforms, indent=2), encoding="utf-8")

    return ExportResult(
        out_dir=out_dir,
        images_dir=images_dir,
        transforms_path=transforms_path,
        num_frames=len(frames_json),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Export SpatialCapture_* to Nerfstudio 'transforms.json' dataset.")
    ap.add_argument("--input", required=True, type=Path, help="Path to SpatialCapture_* folder")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dataset dir (default: <capture>/nerfstudio_export)",
    )
    ap.add_argument("--every-n", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument(
        "--translate-thresh-m",
        type=float,
        default=0.0,
        help="Select next frame only if camera moved at least this many meters since last selected (0 disables)",
    )
    ap.add_argument(
        "--rotate-thresh-deg",
        type=float,
        default=0.0,
        help="Select next frame only if camera rotated at least this many degrees since last selected (0 disables)",
    )
    ap.add_argument(
        "--image-max-size",
        type=int,
        default=1280,
        help="Downscale so max(width,height) <= this (0 disables)",
    )
    ap.add_argument(
        "--normalize-poses",
        action="store_true",
        help="Center and scale camera translations for easier NeRF/3DGS optimization",
    )

    args = ap.parse_args()
    out_dir = args.out
    if out_dir is None:
        out_dir = args.input / "nerfstudio_export"

    res = export_nerfstudio_dataset(
        capture_dir=args.input,
        out_dir=out_dir,
        every_n=args.every_n,
        max_frames=args.max_frames,
        image_max_size=args.image_max_size,
        normalize_poses=bool(args.normalize_poses),
        translate_thresh_m=float(args.translate_thresh_m),
        rotate_thresh_deg=float(args.rotate_thresh_deg),
    )

    print("Exported Nerfstudio dataset:")
    print(f"- out_dir: {res.out_dir}")
    print(f"- images: {res.images_dir}")
    print(f"- transforms: {res.transforms_path}")
    print(f"- frames: {res.num_frames}")


if __name__ == "__main__":
    main()
