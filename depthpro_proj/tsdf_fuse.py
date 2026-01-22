from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

from spatial_capture import (
    FrameRecord,
    arkit_cw_to_open3d_cw,
    get_intrinsics_for_saved_image,
    read_frames_jsonl,
)


def _read_depth_f32(path: Path, width: int, height: int) -> np.ndarray:
    raw = path.read_bytes()
    depth = np.frombuffer(raw, dtype=np.float32)
    if depth.size != width * height:
        raise ValueError(f"Depth size mismatch: got {depth.size}, expected {width*height}")
    return depth.reshape((height, width))


def _intrinsic_from_k(k: np.ndarray, w: int, h: int) -> o3d.camera.PinholeCameraIntrinsic:
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    return intrinsic


def fuse_tsdf(
    capture_dir: Path,
    out_mesh: Path,
    voxel_length: float = 0.02,
    sdf_trunc: float = 0.06,
    depth_trunc: float = 6.0,
    every_n: int = 1,
    max_frames: Optional[int] = None,
) -> None:
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_length),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    used = 0
    for idx, rec in enumerate(read_frames_jsonl(capture_dir)):
        if max_frames is not None and used >= max_frames:
            break
        if (idx % every_n) != 0:
            continue
        if not rec.depth_file:
            continue
        if rec.depth_width is None or rec.depth_height is None:
            continue

        depth_path = capture_dir / rec.depth_file
        if not depth_path.exists():
            continue

        k, img_w, img_h = get_intrinsics_for_saved_image(rec, capture_dir)
        intrinsic = _intrinsic_from_k(k, img_w, img_h)

        # Depth map is recorded from ARFrame.sceneDepth.depthMap.
        # It is typically aligned to the camera image but can have different resolution.
        # For now we assume the depth is already in the saved image space; if not, you'll need a resize + intrinsics scale.
        depth = _read_depth_f32(depth_path, rec.depth_width, rec.depth_height)

        # Convert to Open3D images
        color = o3d.io.read_image(str(capture_dir / rec.image_file))
        depth_o3d = o3d.geometry.Image(depth)

        # If depth resolution differs, resize depth to match color (simple nearest / linear).
        if depth.shape[1] != img_w or depth.shape[0] != img_h:
            depth_img = o3d.t.geometry.Image(depth.astype(np.float32)).resize(img_w, img_h, o3d.t.geometry.InterpType.Linear)
            depth_o3d = o3d.geometry.Image(np.asarray(depth_img))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
        )

        cw = arkit_cw_to_open3d_cw(rec.camera_transform_cw_4x4)
        volume.integrate(rgbd, intrinsic, cw)
        used += 1

    if used == 0:
        raise RuntimeError(
            "No depth frames were found/usable. On non-LiDAR devices, sceneDepth is often unavailable. "
            "Next step: run a monocular depth model on your recorded JPEGs, then fuse those predicted depth maps."
        )

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    out_mesh.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_mesh), mesh)
    print(f"Wrote mesh: {out_mesh} (frames integrated: {used})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuse Patrune SpatialCapture_* into a TSDF mesh (requires per-frame depth).")
    ap.add_argument("--input", required=True, type=Path, help="Path to SpatialCapture_* folder")
    ap.add_argument("--output", required=True, type=Path, help="Output mesh path (e.g. out/scene.ply or out/scene.obj)")
    ap.add_argument("--voxel", type=float, default=0.02, help="TSDF voxel length in meters")
    ap.add_argument("--trunc", type=float, default=0.06, help="TSDF truncation distance in meters")
    ap.add_argument("--depth-trunc", type=float, default=6.0, help="Ignore depth beyond this (meters)")
    ap.add_argument("--every-n", type=int, default=1, help="Use every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to fuse")
    args = ap.parse_args()

    fuse_tsdf(
        capture_dir=args.input,
        out_mesh=args.output,
        voxel_length=args.voxel,
        sdf_trunc=args.trunc,
        depth_trunc=args.depth_trunc,
        every_n=args.every_n,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
