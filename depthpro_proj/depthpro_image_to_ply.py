from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

import step1_3d_photo


def _load_rgb_u8_path(path: Path, image_max_size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        w0, h0 = im.size
        if image_max_size > 0 and max(w0, h0) > int(image_max_size):
            scale = float(image_max_size) / float(max(w0, h0))
            w1 = int(round(w0 * scale))
            h1 = int(round(h0 * scale))
            im = im.resize((w1, h1), resample=Image.Resampling.LANCZOS)
        return np.array(im, dtype=np.uint8)


def _intrinsics_from_args(
    w: int,
    h: int,
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
    focal_px: Optional[float],
    focal_px_pred: Optional[float],
) -> Tuple[np.ndarray, dict]:
    if focal_px is None:
        focal_px = focal_px_pred

    if focal_px is not None and (fx is None and fy is None):
        fx = float(focal_px)
        fy = float(focal_px)

    if fx is None:
        fx = 0.9 * float(max(w, h))
    if fy is None:
        fy = float(fx)
    if cx is None:
        cx = (float(w) - 1.0) * 0.5
    if cy is None:
        cy = (float(h) - 1.0) * 0.5

    k = np.array(
        [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    meta = {
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
        "focal_px_arg": (float(focal_px) if focal_px is not None else None),
        "focal_px_pred": (float(focal_px_pred) if focal_px_pred is not None else None),
    }
    return k, meta


def _points_from_depth(
    depth_m: np.ndarray,
    rgb_u8: np.ndarray,
    k: np.ndarray,
    grid_step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if depth_m.shape[:2] != rgb_u8.shape[:2]:
        raise ValueError("depth and rgb must have matching H,W")

    h, w = depth_m.shape
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    step = int(grid_step)
    if step <= 0:
        step = 1

    ys = np.arange(0, h, step, dtype=np.int32)
    xs = np.arange(0, w, step, dtype=np.int32)
    xx, yy = np.meshgrid(xs, ys)
    z = depth_m[yy, xx].astype(np.float32)

    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    xxv = xx[valid].astype(np.float32)
    yyv = yy[valid].astype(np.float32)
    zv = z[valid].astype(np.float32)

    x = (xxv - cx) * zv / max(fx, 1e-6)
    y = (yyv - cy) * zv / max(fy, 1e-6)

    pts = np.stack([x, y, zv], axis=1).astype(np.float32)
    cols = rgb_u8[yy, xx][valid].astype(np.uint8)
    return pts, cols


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "DepthPro (single image) -> colored point cloud PLY.\n"
            "Designed for easy comparison with SHARP's gaussian point clouds: both output .ply." 
        )
    )
    ap.add_argument("--image", type=Path, required=True, help="Input JPG/PNG")
    ap.add_argument("--out", type=Path, default=None, help="Output directory (default: pcd_<stem> next to image)")

    ap.add_argument(
        "--image-max-size",
        type=int,
        default=0,
        help="Downscale longest side before DepthPro (0 keeps original resolution)",
    )
    ap.add_argument(
        "--grid-step",
        type=int,
        default=1,
        help="Sample every N pixels for point cloud (higher=fewer points, does NOT change depth prediction)",
    )
    ap.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Optional cap on points (0 disables). Uses deterministic subsampling.",
    )

    ap.add_argument("--depth-device", default="auto", help="DepthPro device: auto/cpu/mps/cuda")

    ap.add_argument("--focal-px", type=float, default=None, help="Optional focal length in pixels")
    ap.add_argument("--fx", type=float, default=None, help="Override fx")
    ap.add_argument("--fy", type=float, default=None, help="Override fy")
    ap.add_argument("--cx", type=float, default=None, help="Override cx")
    ap.add_argument("--cy", type=float, default=None, help="Override cy")

    args = ap.parse_args()

    image_path: Path = args.image
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    out_dir = args.out
    if out_dir is None:
        out_dir = image_path.parent / f"pcd_{image_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_u8 = _load_rgb_u8_path(image_path, image_max_size=int(args.image_max_size))

    # Predict DepthPro metric depth
    depth_m, focal_px_pred = step1_3d_photo._predict_depth_depthpro(  # type: ignore[attr-defined]
        rgb=rgb_u8,
        device=str(args.depth_device),
        f_px=(float(args.focal_px) if args.focal_px is not None else None),
    )

    h, w = depth_m.shape
    k, k_meta = _intrinsics_from_args(
        w=w,
        h=h,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        focal_px=args.focal_px,
        focal_px_pred=focal_px_pred,
    )

    # Persist inputs for debugging/reuse
    Image.fromarray(rgb_u8, mode="RGB").save(out_dir / "scene.png")
    np.save(out_dir / "depth.npy", depth_m.astype(np.float32))

    pts, cols = _points_from_depth(depth_m=depth_m, rgb_u8=rgb_u8, k=k, grid_step=int(args.grid_step))

    max_points = int(args.max_points)
    if max_points > 0 and pts.shape[0] > max_points:
        keep = np.linspace(0, pts.shape[0] - 1, num=max_points).astype(np.int64)
        pts = pts[keep]
        cols = cols[keep]

    ply_path = out_dir / "scene_points_camera.ply"
    step1_3d_photo._write_ply_xyzrgb(ply_path, pts.astype(np.float32), cols.astype(np.uint8))  # type: ignore[attr-defined]

    meta = {
        "image": str(image_path),
        "image_max_size": int(args.image_max_size),
        "grid_step": int(args.grid_step),
        "max_points": int(args.max_points),
        "depth": {"method": "depthpro", "device": str(args.depth_device), "focal_px_pred": focal_px_pred},
        "intrinsics": k_meta,
        "outputs": {
            "scene_png": str(out_dir / "scene.png"),
            "depth_npy": str(out_dir / "depth.npy"),
            "ply_camera": str(ply_path),
        },
        "points": int(pts.shape[0]),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {ply_path} with {pts.shape[0]} points")


if __name__ == "__main__":
    main()
