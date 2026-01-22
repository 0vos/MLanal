from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

from spatial_capture import arkit_cw_to_open3d_cw, read_frames_jsonl
from step1_3d_photo import run_step1


@dataclass(frozen=True)
class FuseOutputs:
    fused_ply_path: Path
    meta_json_path: Path


def _recolor_points_from_frames(
    points_world: np.ndarray,
    frames_dir: Path,
    frame_indices: list[int],
    occlusion_thresh_m: float,
    base_colors_u8: Optional[np.ndarray] = None,
    fuse_meta: Optional[dict] = None,
) -> np.ndarray:
    """Recolor world points by projecting into per-frame inpainted images.

    Uses a simple z-consistency check with the per-frame predicted depth to avoid painting
    occluded points with background colors.

    Returns uint8 colors (N,3).
    """
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError("points_world must be (N,3)")

    n = points_world.shape[0]
    if base_colors_u8 is None:
        colors = np.zeros((n, 3), dtype=np.uint8)
    else:
        bc = np.asarray(base_colors_u8, dtype=np.uint8)
        if bc.shape != (n, 3):
            raise ValueError("base_colors_u8 must be (N,3)")
        colors = bc.copy()

    best_err = np.full((n,), np.inf, dtype=np.float32)

    pts_h = np.concatenate([points_world.astype(np.float32), np.ones((n, 1), dtype=np.float32)], axis=1)

    if fuse_meta is None:
        fuse_meta_path = frames_dir / "fuse_meta.json"
        if fuse_meta_path.exists():
            fuse_meta = json.loads(fuse_meta_path.read_text(encoding="utf-8"))
        else:
            fuse_meta = {}

    pose_map = {
        int(p["frame_index"]): np.array(p["cameraTransform_cw_4x4"], dtype=np.float32).reshape((4, 4), order="F")
        for p in fuse_meta.get("poses", [])
    }

    for idx in frame_indices:
        per = frames_dir / f"step1_{idx:06d}"
        meta_path = per / "step1_meta.json"
        img_path = per / "scene_image.png"
        depth_path = per / "depth_pred.npy"

        if not meta_path.exists() or not img_path.exists() or not depth_path.exists():
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_w, meta_h = meta["saved_image_wh"]
        k = np.array(meta["intrinsics_saved"], dtype=np.float32)
        fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])

        # Load frame-specific pose
        # We store the original ARKit cw in frames.jsonl; convert to Open3D-friendly pinhole cw.
        # Note: read_frames_jsonl is cheap for small N, but we avoid it here by reading cached pose from meta is not present.
        # The fusion meta always contains per-frame cameraTransform_cw_4x4.
        # For recoloring we read it from fuse_meta.json later, so here we rely on per-frame meta only for K/size.

        # Image + depth
        img = np.asarray(o3d.io.read_image(str(img_path)))
        depth = np.load(str(depth_path)).astype(np.float32)
        h, w = int(img.shape[0]), int(img.shape[1])

        # If the actual image size differs from the meta, scale intrinsics accordingly.
        if int(meta_w) > 0 and int(meta_h) > 0 and (w != int(meta_w) or h != int(meta_h)):
            sx = float(w) / float(meta_w)
            sy = float(h) / float(meta_h)
            fx *= sx
            cx *= sx
            fy *= sy
            cy *= sy

        if depth.shape[0] != int(h) or depth.shape[1] != int(w):
            depth_t = o3d.t.geometry.Image(depth).resize(int(w), int(h), o3d.t.geometry.InterpType.Linear)
            depth = np.asarray(depth_t).astype(np.float32)

        cw_arkit = pose_map.get(int(idx))
        if cw_arkit is None:
            continue
        cw = arkit_cw_to_open3d_cw(cw_arkit)
        wc = np.linalg.inv(cw).astype(np.float32)

        pc = (wc @ pts_h.T).T
        z = pc[:, 2]
        valid = z > 1e-6
        x = pc[:, 0]
        y = pc[:, 1]

        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
        ui = u.astype(np.int32)
        vi = v.astype(np.int32)
        inb = (ui >= 0) & (ui < int(w)) & (vi >= 0) & (vi < int(h))
        valid &= inb

        idxs = np.nonzero(valid)[0]
        if idxs.size == 0:
            continue

        d_samp = depth[vi[idxs], ui[idxs]]
        err = np.abs(d_samp - z[idxs])
        ok = np.isfinite(d_samp) & np.isfinite(err) & (err < float(occlusion_thresh_m))
        if not np.any(ok):
            continue

        idxs_ok = idxs[ok]
        err_ok = err[ok]
        better = err_ok < best_err[idxs_ok]
        if not np.any(better):
            continue

        chosen = idxs_ok[better]
        best_err[chosen] = err_ok[better].astype(np.float32)
        colors[chosen] = img[vi[chosen], ui[chosen], :3].astype(np.uint8)

    return colors


def _mat4_to_list_col_major(m: np.ndarray) -> list[float]:
    m = np.asarray(m, dtype=np.float32)
    if m.shape != (4, 4):
        raise ValueError("expected 4x4")
    return m.reshape((16,), order="F").astype(float).tolist()


def fuse_keyframes(
    capture_dir: Path,
    out_ply: Path,
    out_dir: Path,
    every_n: int,
    max_frames: Optional[int],
    voxel_size: float,
    max_points: int,
    method: str,
    write_ascii: bool,
    tsdf_voxel_length: float,
    tsdf_sdf_trunc: float,
    tsdf_depth_trunc: float,
    recolor: bool,
    recolor_occlusion_thresh: float,
    # Step1 settings
    mask_method: str,
    mask_dilate: int,
    mask_prompt: str,
    langsam_sam_type: str,
    langsam_box_threshold: float,
    langsam_text_threshold: float,
    mask_shape: str,
    inpaint_method: str,
    inpaint_radius: int,
    inpaint_device: str,
    inpaint_model: str,
    inpaint_steps: int,
    inpaint_guidance: float,
    inpaint_seed: Optional[int],
    depth_method: str,
    depth_model: str,
    depth_device: str,
    min_depth: float,
    max_depth: float,
) -> FuseOutputs:
    frames = list(read_frames_jsonl(capture_dir))
    if not frames:
        raise RuntimeError(f"No frames found in {capture_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = out_ply if out_ply.is_absolute() else (out_dir / out_ply)

    fused = o3d.geometry.PointCloud()
    volume = None
    if method == "tsdf":
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(tsdf_voxel_length),
            sdf_trunc=float(tsdf_sdf_trunc),
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    used_indices: list[int] = []
    used_cw: list[np.ndarray] = []

    for idx in range(len(frames)):
        if (idx % int(every_n)) != 0:
            continue
        if max_frames is not None and len(used_indices) >= int(max_frames):
            break

        per_frame_out = out_dir / f"step1_{idx:06d}"

        outputs = run_step1(
            capture_dir=capture_dir,
            frame_index=idx,
            mask_method=mask_method,
            mask_dilate=mask_dilate,
            mask_prompt=mask_prompt,
            langsam_sam_type=langsam_sam_type,
            langsam_box_threshold=float(langsam_box_threshold),
            langsam_text_threshold=float(langsam_text_threshold),
            mask_shape=mask_shape,
            inpaint_method=inpaint_method,
            inpaint_radius=inpaint_radius,
            inpaint_device=inpaint_device,
            inpaint_model=inpaint_model,
            inpaint_steps=inpaint_steps,
            inpaint_guidance=inpaint_guidance,
            inpaint_seed=inpaint_seed,
            depth_method=depth_method,
            depth_model=depth_model,
            depth_device=depth_device,
            min_depth=min_depth,
            max_depth=max_depth,
            out_dir=per_frame_out,
            max_points=max_points,
            drop_person_points=False,
        )

        used_indices.append(idx)
        used_cw.append(frames[idx].camera_transform_cw_4x4.astype(np.float32))

        if method == "pcd":
            pcd_i = o3d.io.read_point_cloud(str(outputs.ply_path))
            if pcd_i.is_empty():
                print(f"[fuse] skipping empty pcd for frame {idx}")
                continue
            fused += pcd_i
            print(f"[fuse] added frame {idx}: points={len(pcd_i.points)} total={len(fused.points)}")
        elif method == "tsdf":
            assert volume is not None
            # Load Step1 meta to get the intrinsics and image size actually used for depth.
            meta = json.loads((per_frame_out / "step1_meta.json").read_text(encoding="utf-8"))
            w, h = meta["saved_image_wh"]
            k = np.array(meta["intrinsics_saved"], dtype=np.float32)

            fx = float(k[0, 0])
            fy = float(k[1, 1])
            cx = float(k[0, 2])
            cy = float(k[1, 2])
            intrinsic = o3d.camera.PinholeCameraIntrinsic(int(w), int(h), fx, fy, cx, cy)

            color = o3d.io.read_image(str(per_frame_out / "scene_image.png"))
            depth = np.load(str(per_frame_out / "depth_pred.npy")).astype(np.float32)
            depth_o3d = o3d.geometry.Image(depth)

            # Ensure depth resolution matches color.
            if depth.shape[1] != int(w) or depth.shape[0] != int(h):
                depth_img = o3d.t.geometry.Image(depth).resize(int(w), int(h), o3d.t.geometry.InterpType.Linear)
                depth_o3d = o3d.geometry.Image(np.asarray(depth_img))

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth_o3d,
                depth_scale=1.0,
                depth_trunc=float(tsdf_depth_trunc),
                convert_rgb_to_intensity=False,
            )

            cw = arkit_cw_to_open3d_cw(frames[idx].camera_transform_cw_4x4)
            volume.integrate(rgbd, intrinsic, cw)
            print(f"[fuse] integrated frame {idx} into TSDF")
        else:
            raise ValueError("method must be one of: pcd, tsdf")

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    if method == "pcd":
        if fused.is_empty():
            raise RuntimeError("No points accumulated; check Step1 outputs")

        if voxel_size > 0:
            fused = fused.voxel_down_sample(voxel_size=float(voxel_size))

        # Deterministic downsample if still too large
        if max_points is not None and len(fused.points) > int(max_points):
            pts = np.asarray(fused.points)
            cols = np.asarray(fused.colors) if fused.has_colors() else None
            keep = np.linspace(0, pts.shape[0] - 1, num=int(max_points)).astype(np.int64)
            pts = pts[keep]
            fused.points = o3d.utility.Vector3dVector(pts)
            if cols is not None and cols.shape[0] == keep.shape[0]:
                fused.colors = o3d.utility.Vector3dVector(cols)

        o3d.io.write_point_cloud(str(out_ply), fused, write_ascii=bool(write_ascii))
    else:
        assert volume is not None
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()

        # Use --out as mesh path. Also write an extracted point cloud next to it.
        pcd = volume.extract_point_cloud()

        # Optional post-process recoloring from the per-frame inpainted images.
        if recolor:
            fuse_meta = {
                "poses": [
                    {
                        "frame_index": int(i),
                        "cameraTransform_cw_4x4": _mat4_to_list_col_major(cw),
                    }
                    for i, cw in zip(used_indices, used_cw)
                ]
            }

            pts = np.asarray(pcd.points)
            base_cols = None
            if pcd.has_colors() and len(pcd.colors) == len(pcd.points):
                base_cols = (np.asarray(pcd.colors) * 255.0).clip(0, 255).astype(np.uint8)

            if pts.shape[0] > 0:
                cols_u8 = _recolor_points_from_frames(
                    pts,
                    frames_dir=out_dir,
                    frame_indices=used_indices,
                    occlusion_thresh_m=float(recolor_occlusion_thresh),
                    base_colors_u8=base_cols,
                    fuse_meta=fuse_meta,
                )
                pcd.colors = o3d.utility.Vector3dVector((cols_u8.astype(np.float32) / 255.0).clip(0.0, 1.0))

                v = np.asarray(mesh.vertices)
                if v.shape[0] > 0:
                    base_vcols = None
                    if mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(mesh.vertices):
                        base_vcols = (np.asarray(mesh.vertex_colors) * 255.0).clip(0, 255).astype(np.uint8)

                    vcols_u8 = _recolor_points_from_frames(
                        v,
                        frames_dir=out_dir,
                        frame_indices=used_indices,
                        occlusion_thresh_m=float(recolor_occlusion_thresh),
                        base_colors_u8=base_vcols,
                        fuse_meta=fuse_meta,
                    )
                    mesh.vertex_colors = o3d.utility.Vector3dVector((vcols_u8.astype(np.float32) / 255.0).clip(0.0, 1.0))

        o3d.io.write_triangle_mesh(str(out_ply), mesh, write_ascii=bool(write_ascii))
        pcd_path = out_ply.with_name(out_ply.stem + "_pcd.ply")
        o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=bool(write_ascii))

    # Relative pose metadata for next stage
    meta_path = out_dir / "fuse_meta.json"
    t0_inv = None
    if used_cw:
        t0_inv = np.linalg.inv(used_cw[0])

    meta = {
        "capture_dir": str(capture_dir),
        "used_frame_indices": used_indices,
        "every_n": int(every_n),
        "max_frames": None if max_frames is None else int(max_frames),
        "method": method,
        "voxel_size": float(voxel_size),
        "write_ascii": bool(write_ascii),
        "tsdf_voxel_length": float(tsdf_voxel_length),
        "tsdf_sdf_trunc": float(tsdf_sdf_trunc),
        "tsdf_depth_trunc": float(tsdf_depth_trunc),
        "recolor": bool(recolor),
        "recolor_occlusion_thresh": float(recolor_occlusion_thresh),
        "max_points": int(max_points),
        "poses": [
            {
                "frame_index": int(i),
                "cameraTransform_cw_4x4": _mat4_to_list_col_major(cw),
                "relative_to_first_cw_4x4": _mat4_to_list_col_major((t0_inv @ cw) if t0_inv is not None else cw),
            }
            for i, cw in zip(used_indices, used_cw)
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return FuseOutputs(fused_ply_path=out_ply, meta_json_path=meta_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fuse multiple SpatialCapture_* keyframes by running Step1 per frame and merging world point clouds."
    )
    ap.add_argument("--input", required=True, type=Path, help="Path to SpatialCapture_* folder")
    ap.add_argument("--out", type=Path, default=Path("fused_world.ply"), help="Output fused point cloud path")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for per-frame Step1 outputs and fusion meta (default: <capture>/fuse)",
    )
    ap.add_argument("--every-n", type=int, default=1, help="Use every Nth frame")
    ap.add_argument("--max-frames", type=int, default=3, help="Max number of frames to fuse (default: 3 for quick test)")
    ap.add_argument("--method", choices=["pcd", "tsdf"], default="pcd", help="Fusion method: pcd (merge points) or tsdf (integrate predicted depth)")
    ap.add_argument("--voxel", type=float, default=0.02, help="Voxel size in meters for point-cloud downsampling (pcd mode only)")
    ap.add_argument("--max-points", type=int, default=600_000, help="Clamp fused point count")
    ap.add_argument("--write-ascii", action="store_true", help="Write outputs as ASCII PLY (larger files, easier to inspect)")

    ap.add_argument("--tsdf-voxel-length", type=float, default=0.01, help="TSDF voxel length in meters (tsdf mode). Smaller = sharper but slower")
    ap.add_argument("--tsdf-trunc", type=float, default=0.04, help="TSDF truncation distance in meters (tsdf mode)")
    ap.add_argument("--tsdf-depth-trunc", type=float, default=6.0, help="Ignore depth beyond this (meters) (tsdf mode)")

    ap.add_argument(
        "--recolor",
        action="store_true",
        help="Recolor fused outputs by projecting into per-frame inpainted images with a depth consistency check (improves color sharpness).",
    )
    ap.add_argument(
        "--recolor-occlusion-thresh",
        type=float,
        default=0.15,
        help="Meters: allowable |z - depth(u,v)| when recoloring (smaller = stricter visibility).",
    )

    # Step1 passthrough
    ap.add_argument("--mask", choices=["rembg", "centerbox", "langsam"], default="langsam")
    ap.add_argument("--mask-dilate", type=int, default=4)
    ap.add_argument("--mask-prompt", default="person.")
    ap.add_argument("--langsam-sam-type", default="sam2.1_hiera_large")
    ap.add_argument("--langsam-box-threshold", type=float, default=0.15)
    ap.add_argument("--langsam-text-threshold", type=float, default=0.15)
    ap.add_argument("--mask-shape", choices=["original", "rectangle", "square"], default="original")

    ap.add_argument("--inpaint", choices=["opencv", "diffusers", "none"], default="diffusers")
    ap.add_argument("--inpaint-radius", type=int, default=3)
    ap.add_argument("--inpaint-device", default="mps")
    ap.add_argument("--inpaint-model", default="runwayml/stable-diffusion-inpainting")
    ap.add_argument("--inpaint-steps", type=int, default=25)
    ap.add_argument("--inpaint-guidance", type=float, default=7.5)
    ap.add_argument("--inpaint-seed", type=int, default=None)

    ap.add_argument("--depth", choices=["midas", "depthpro"], default="depthpro")
    ap.add_argument("--depth-model", default="MiDaS_small")
    ap.add_argument("--depth-device", default="mps")
    ap.add_argument("--min-depth", type=float, default=0.5)
    ap.add_argument("--max-depth", type=float, default=10.0)

    args = ap.parse_args()

    out_dir = args.outdir
    if out_dir is None:
        out_dir = args.input / "fuse"

    outputs = fuse_keyframes(
        capture_dir=args.input,
        out_ply=args.out,
        out_dir=out_dir,
        every_n=args.every_n,
        max_frames=args.max_frames,
        voxel_size=args.voxel,
        max_points=args.max_points,
        method=args.method,
        write_ascii=bool(args.write_ascii),
        tsdf_voxel_length=args.tsdf_voxel_length,
        tsdf_sdf_trunc=args.tsdf_trunc,
        tsdf_depth_trunc=args.tsdf_depth_trunc,
        recolor=bool(args.recolor),
        recolor_occlusion_thresh=args.recolor_occlusion_thresh,
        mask_method=args.mask,
        mask_dilate=args.mask_dilate,
        mask_prompt=args.mask_prompt,
        langsam_sam_type=args.langsam_sam_type,
        langsam_box_threshold=args.langsam_box_threshold,
        langsam_text_threshold=args.langsam_text_threshold,
        mask_shape=args.mask_shape,
        inpaint_method=args.inpaint,
        inpaint_radius=args.inpaint_radius,
        inpaint_device=args.inpaint_device,
        inpaint_model=args.inpaint_model,
        inpaint_steps=args.inpaint_steps,
        inpaint_guidance=args.inpaint_guidance,
        inpaint_seed=args.inpaint_seed,
        depth_method=args.depth,
        depth_model=args.depth_model,
        depth_device=args.depth_device,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    print("Wrote:")
    print(f"- {outputs.fused_ply_path}")
    print(f"- {outputs.meta_json_path}")


if __name__ == "__main__":
    main()
