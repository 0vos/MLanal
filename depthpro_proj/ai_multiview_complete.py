from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from PIL import Image

import depth_to_mesh as dtm  # type: ignore
from step1_3d_photo import _inpaint_diffusers, _inpaint_opencv  # type: ignore


@dataclass(frozen=True)
class ViewSpec:
    name: str
    yaw_deg: float


def _load_mesh_dir(mesh_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    scene_path = mesh_dir / "scene.png"
    depth_path = mesh_dir / "depth.npy"
    meta_path = mesh_dir / "meta.json"

    if not scene_path.exists() or not depth_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"mesh_dir must contain scene.png, depth.npy, meta.json: {mesh_dir}")

    rgb = np.array(Image.open(scene_path).convert("RGB"), dtype=np.uint8)
    depth = np.load(depth_path).astype(np.float32)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if depth.shape[:2] != rgb.shape[:2]:
        d_im = Image.fromarray(depth.astype(np.float32))
        d_im = d_im.resize((rgb.shape[1], rgb.shape[0]), resample=Image.Resampling.BILINEAR)
        depth = np.array(d_im, dtype=np.float32)
    return rgb, depth, meta


def _intrinsics_from_meta(meta: Dict[str, Any], w: int, h: int) -> np.ndarray:
    intr = meta.get("intrinsics")
    if not isinstance(intr, dict):
        raise ValueError("meta.json missing intrinsics")
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return dtm._scale_intrinsics(k, scale_x=1.0, scale_y=1.0)  # type: ignore[attr-defined]


def _rot_y(yaw_rad: float) -> np.ndarray:
    c = float(math.cos(yaw_rad))
    s = float(math.sin(yaw_rad))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _backproject(depth: np.ndarray, k: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth.shape[:2]
    ys = np.arange(0, h, int(step), dtype=np.int32)
    xs = np.arange(0, w, int(step), dtype=np.int32)
    uu, vv = np.meshgrid(xs, ys)
    z = depth[vv, uu].astype(np.float32)

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    x = (uu.astype(np.float32) - cx) * (z / max(fx, 1e-6))
    y = (vv.astype(np.float32) - cy) * (z / max(fy, 1e-6))

    pts = np.stack([x, y, z], axis=2).reshape(-1, 3)
    pix = np.stack([uu, vv], axis=2).reshape(-1, 2)

    valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 1e-6)
    pts = pts[valid]
    pix = pix[valid]

    return pts.astype(np.float32), pix[:, 0].astype(np.int32), pix[:, 1].astype(np.int32)


def _render_rotated_view(
    rgb: np.ndarray,
    depth: np.ndarray,
    k: np.ndarray,
    center_cam: np.ndarray,
    rot: np.ndarray,
    step: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate the scene around `center_cam` by `rot` and render to the original camera."""
    h, w = depth.shape[:2]

    pts, px, py = _backproject(depth, k=k, step=int(step))

    # Rotate around center
    c = center_cam.reshape(1, 3).astype(np.float32)
    pts2 = (c + (rot @ (pts - c).T).T).astype(np.float32)

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    z = pts2[:, 2]
    valid = z > 1e-6
    x = pts2[:, 0]
    y = pts2[:, 1]

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    inb = (ui >= 0) & (ui < int(w)) & (vi >= 0) & (vi < int(h))
    valid &= inb

    ui = ui[valid]
    vi = vi[valid]
    z = z[valid]

    src_x = px[valid]
    src_y = py[valid]
    src_rgb = rgb[src_y, src_x, :]

    lin = vi.astype(np.int64) * int(w) + ui.astype(np.int64)
    order = np.argsort(z, kind="mergesort")
    lin_s = lin[order]

    _, first = np.unique(lin_s, return_index=True)
    keep = order[first]

    ui_k = ui[keep]
    vi_k = vi[keep]
    z_k = z[keep]
    rgb_k = src_rgb[keep]

    out_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    out_depth = np.full((h, w), np.inf, dtype=np.float32)

    out_rgb[vi_k, ui_k, :] = rgb_k
    out_depth[vi_k, ui_k] = z_k.astype(np.float32)

    vis = np.isfinite(out_depth)
    return out_rgb, out_depth, vis


def _resize_rgb_u8(rgb: np.ndarray, max_size: int) -> Tuple[np.ndarray, float, float]:
    """Resize RGB while keeping aspect ratio. Returns (rgb_resized, sx, sy)."""
    if int(max_size) <= 0:
        return rgb, 1.0, 1.0
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= int(max_size):
        return rgb, 1.0, 1.0
    scale = float(max_size) / float(m)
    w1 = int(round(w * scale))
    h1 = int(round(h * scale))
    im = Image.fromarray(rgb, mode="RGB").resize((w1, h1), resample=Image.Resampling.LANCZOS)
    rgb1 = np.array(im, dtype=np.uint8)
    return rgb1, float(w1) / float(w), float(h1) / float(h)


def _mask_iou(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    a = a_u8 > 0
    b = b_u8 > 0
    inter = int(np.count_nonzero(a & b))
    if inter <= 0:
        return 0.0
    uni = int(np.count_nonzero(a | b))
    return float(inter) / float(max(uni, 1))


def _nms_masks(masks_u8: list[np.ndarray], scores: list[float], iou_thresh: float) -> list[int]:
    if not masks_u8:
        return []
    order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1]
    keep: list[int] = []
    for idx in order.tolist():
        m = masks_u8[int(idx)]
        ok = True
        for j in keep:
            if _mask_iou(m, masks_u8[int(j)]) >= float(iou_thresh):
                ok = False
                break
        if ok:
            keep.append(int(idx))
    return keep


def _instances_from_sam2(
    rgb: np.ndarray,
    min_area: int,
    max_area_frac: float,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_thresh: float,
    min_mask_region_area: int,
    keep_topk: int,
    nms_iou: float,
    sam2_model_id: str,
) -> list[dict[str, Any]]:
    h, w = rgb.shape[:2]
    masks = dtm._sam2_auto_masks(  # type: ignore[attr-defined]
        rgb,
        model_id=str(sam2_model_id),
        device=dtm._torch_auto_device(),  # type: ignore[attr-defined]
        points_per_side=int(points_per_side),
        pred_iou_thresh=float(pred_iou_thresh),
        stability_score_thresh=float(stability_thresh),
        min_mask_region_area=int(min_mask_region_area),
    )

    inst_id, inst_meta = dtm._instance_id_map_from_sam2_masks(  # type: ignore[attr-defined]
        masks,
        h=int(h),
        w=int(w),
        min_area=int(min_area),
        max_area_frac=float(max_area_frac),
    )

    cand: list[dict[str, Any]] = []
    unique_ids = np.unique(inst_id)
    unique_ids = unique_ids[unique_ids >= 0]
    for iid in unique_ids.tolist():
        iid = int(iid)
        mask = (inst_id == iid).astype(np.uint8) * 255
        area = int(np.count_nonzero(mask))
        if area < int(min_area):
            continue
        cand.append({"instance_id": iid, "mask": mask, "mask_area_px": area})

    if cand:
        keep_idx = _nms_masks(
            masks_u8=[c["mask"] for c in cand],
            scores=[float(c["mask_area_px"]) for c in cand],
            iou_thresh=float(nms_iou),
        )
        cand = [cand[i] for i in keep_idx]

    cand.sort(key=lambda x: int(x["mask_area_px"]), reverse=True)
    if int(keep_topk) > 0:
        cand = cand[: int(keep_topk)]

    # Keep meta for debugging
    for c in cand:
        c["meta"] = next((m for m in inst_meta if int(m.get("instance_id", -1)) == int(c["instance_id"])), None)

    return cand


def _masked_center_3d(depth: np.ndarray, k: np.ndarray, mask_u8: np.ndarray, step: int) -> np.ndarray:
    h, w = depth.shape[:2]
    mask = mask_u8 > 0

    ys = np.arange(0, h, int(step), dtype=np.int32)
    xs = np.arange(0, w, int(step), dtype=np.int32)
    uu, vv = np.meshgrid(xs, ys)

    m = mask[vv, uu]
    if not np.any(m):
        # fall back to global median
        pts, _, _ = _backproject(depth, k=k, step=max(int(step), 2))
        if pts.shape[0] == 0:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.nanmedian(pts, axis=0).astype(np.float32)

    z = depth[vv, uu].astype(np.float32)
    z = z[m]
    uu = uu[m].astype(np.float32)
    vv = vv[m].astype(np.float32)

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    x = (uu - cx) * (z / max(fx, 1e-6))
    y = (vv - cy) * (z / max(fy, 1e-6))

    pts = np.stack([x, y, z], axis=1)
    pts = pts[np.isfinite(pts).all(axis=1) & (pts[:, 2] > 1e-6)]
    if pts.shape[0] == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    return np.nanmedian(pts, axis=0).astype(np.float32)


def _pointcloud_from_depth_rgb(depth: np.ndarray, rgb: np.ndarray, k: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
    pts, px, py = _backproject(depth, k=k, step=int(step))
    cols = rgb[py, px, :].astype(np.uint8)
    return pts, cols


def complete_object_multiview(
    rgb0: np.ndarray,
    depth0: np.ndarray,
    k: np.ndarray,
    mask_u8: np.ndarray,
    out_dir: Path,
    views: list[ViewSpec],
    render_step: int,
    points_step: int,
    inpaint: str,
    inpaint_model: str,
    inpaint_device: str,
    inpaint_steps: int,
    inpaint_guidance: float,
    inpaint_seed: Optional[int],
    depth_device: str,
    view_max_size: int,
    skip_depthpro_views: bool,
    poisson_depth: int,
    voxel_down: float,
    skip_mesh: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "views").mkdir(parents=True, exist_ok=True)

    center = _masked_center_3d(depth0, k=k, mask_u8=mask_u8, step=max(int(render_step), 2))

    # Save mask
    Image.fromarray(mask_u8, mode="L").save(out_dir / "mask.png")

    # Base point cloud from original view (masked)
    base_pts, base_cols = _pointcloud_from_depth_rgb(depth0, rgb0, k=k, step=int(points_step))
    # Apply mask filtering using UV lookup
    h, w = depth0.shape[:2]
    u = np.clip(np.round((base_pts[:, 0] / np.maximum(base_pts[:, 2], 1e-6)) * float(k[0, 0]) + float(k[0, 2])).astype(np.int32), 0, w - 1)
    v = np.clip(np.round((base_pts[:, 1] / np.maximum(base_pts[:, 2], 1e-6)) * float(k[1, 1]) + float(k[1, 2])).astype(np.int32), 0, h - 1)
    keep = (mask_u8[v, u] > 0)
    base_pts = base_pts[keep]
    base_cols = base_cols[keep]

    all_pts = [base_pts]
    all_cols = [base_cols]

    view_meta: list[dict[str, Any]] = []

    fx = float(k[0, 0])

    for vi, spec in enumerate(views):
        yaw = float(spec.yaw_deg)
        rot = _rot_y(math.radians(yaw))

        rgb_r, depth_r, vis = _render_rotated_view(rgb0, depth0, k=k, center_cam=center, rot=rot, step=int(render_step))

        hole = (~vis).astype(np.uint8) * 255
        Image.fromarray(rgb_r, mode="RGB").save(out_dir / "views" / f"{spec.name}_warp.png")
        Image.fromarray(hole, mode="L").save(out_dir / "views" / f"{spec.name}_hole.png")

        if inpaint == "none":
            rgb_i = rgb_r
        elif inpaint == "opencv":
            rgb_i = _inpaint_opencv(rgb_r, hole, radius=6)
        elif inpaint == "diffusers":
            rgb_i = _inpaint_diffusers(
                rgb=rgb_r,
                mask_u8=hole,
                model_id=str(inpaint_model),
                device=str(inpaint_device),
                num_inference_steps=int(inpaint_steps),
                guidance_scale=float(inpaint_guidance),
                seed=inpaint_seed,
            )
        else:
            raise ValueError("inpaint must be one of: none, opencv, diffusers")

        Image.fromarray(rgb_i, mode="RGB").save(out_dir / "views" / f"{spec.name}_inpaint.png")

        if bool(skip_depthpro_views):
            # Fast path: trust the warp-rendered metric depth.
            depth_i = depth_r.astype(np.float32)
        else:
            # DepthPro on inpainted view (use fx as f_px to keep scale consistent).
            # Downscale the view to keep per-view inference manageable.
            rgb_small, sx, sy = _resize_rgb_u8(rgb_i, max_size=int(view_max_size))
            depth_small, _ = dtm._predict_depthpro_depth(rgb_small, device=str(depth_device), f_px=float(fx) * float(sx))  # type: ignore[attr-defined]
            if depth_small.shape[:2] != rgb_small.shape[:2]:
                d_im = Image.fromarray(depth_small.astype(np.float32))
                d_im = d_im.resize((rgb_small.shape[1], rgb_small.shape[0]), resample=Image.Resampling.BILINEAR)
                depth_small = np.array(d_im, dtype=np.float32)

            # Resize depth back to full view
            d_im = Image.fromarray(depth_small.astype(np.float32))
            d_im = d_im.resize((depth0.shape[1], depth0.shape[0]), resample=Image.Resampling.BILINEAR)
            depth_i = np.array(d_im, dtype=np.float32)

        np.save(out_dir / "views" / f"{spec.name}_depth.npy", depth_i.astype(np.float32))

        # Backproject to points in the *rotated-scene* frame, then un-rotate back
        pts_i, cols_i = _pointcloud_from_depth_rgb(depth_i.astype(np.float32), rgb_i, k=k, step=int(points_step))
        c = center.reshape(1, 3).astype(np.float32)
        pts_u = (c + (rot.T @ (pts_i - c).T).T).astype(np.float32)

        all_pts.append(pts_u)
        all_cols.append(cols_i)

        view_meta.append({"name": spec.name, "yaw_deg": yaw})

    (out_dir / "views.json").write_text(json.dumps(view_meta, indent=2), encoding="utf-8")

    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((cols.astype(np.float32) / 255.0).astype(np.float64))

    if voxel_down > 0:
        pcd = pcd.voxel_down_sample(float(voxel_down))

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

    # Poisson requires normals.
    # Estimate in world/camera space; orient roughly towards the camera origin.
    radius = float(max(voxel_down * 4.0, 0.05))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    try:
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0], dtype=np.float64))
    except Exception:
        pass
    pcd.normalize_normals()

    o3d.io.write_point_cloud(str(out_dir / "points.ply"), pcd, write_ascii=False)

    if bool(skip_mesh):
        return

    if len(pcd.points) < 1000:
        raise RuntimeError("Too few points for Poisson reconstruction")

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(poisson_depth))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    # Crop to the point cloud AABB to remove Poisson 'fog'
    aabb = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(aabb)

    o3d.io.write_triangle_mesh(str(out_dir / "mesh_poisson.ply"), mesh, write_ascii=False)
    o3d.io.write_triangle_mesh(str(out_dir / "mesh_poisson.obj"), mesh, write_ascii=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="True-AI-ish completion: SAM2 instance mask + depth-warp novel views + diffusion inpaint + DepthPro + Poisson")
    ap.add_argument("--mesh-dir", type=Path, help="Existing depth_to_mesh output dir (contains scene.png/depth.npy/meta.json)")
    ap.add_argument("--image", type=Path, help="Input image; if set, runs DepthPro first")
    ap.add_argument("--out", required=True, type=Path, help="Output directory")

    # Segmentation
    ap.add_argument("--sam2-model-id", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--sam2-points-per-side", type=int, default=32)
    ap.add_argument("--sam2-pred-iou-thresh", type=float, default=0.85)
    ap.add_argument("--sam2-stability-thresh", type=float, default=0.95)
    ap.add_argument("--sam2-min-mask-region-area", type=int, default=200)
    ap.add_argument("--sam2-min-area", type=int, default=800)
    ap.add_argument("--sam2-max-area-frac", type=float, default=0.50)
    ap.add_argument("--keep-topk", type=int, default=5)
    ap.add_argument("--nms-iou", type=float, default=0.60)
    ap.add_argument("--instance-id", type=int, default=-1, help="Pick a specific SAM2 instance id; default selects the largest kept")

    # Novel view generation
    ap.add_argument("--n-views", type=int, default=6)
    ap.add_argument("--yaw-max-deg", type=float, default=50.0)
    ap.add_argument("--render-step", type=int, default=2)
    ap.add_argument("--points-step", type=int, default=2)

    # Inpainting
    ap.add_argument("--inpaint", default="diffusers", choices=["diffusers", "opencv", "none"])
    ap.add_argument("--inpaint-model", default="runwayml/stable-diffusion-inpainting")
    ap.add_argument("--inpaint-device", default="auto")
    ap.add_argument("--inpaint-steps", type=int, default=25)
    ap.add_argument("--inpaint-guidance", type=float, default=7.5)
    ap.add_argument("--inpaint-seed", type=int, default=0)

    # DepthPro
    ap.add_argument("--depth-device", default="auto")
    ap.add_argument(
        "--view-max-size",
        type=int,
        default=768,
        help="Max size for per-view DepthPro inference (keeps runtime reasonable). 0 disables resizing.",
    )
    ap.add_argument(
        "--skip-depthpro-views",
        action="store_true",
        help="Do not run DepthPro on novel views; use warp-rendered depth only (fast but less 'AI completion').",
    )

    # Reconstruction
    ap.add_argument("--poisson-depth", type=int, default=9)
    ap.add_argument("--voxel-down", type=float, default=0.01)
    ap.add_argument("--skip-mesh", action="store_true", help="Only write fused point cloud; skip Poisson mesh")

    args = ap.parse_args()

    if (args.mesh_dir is None) == (args.image is None):
        raise SystemExit("Provide exactly one of --mesh-dir or --image")

    if args.mesh_dir is not None:
        rgb0, depth0, meta = _load_mesh_dir(Path(args.mesh_dir))
        k = _intrinsics_from_meta(meta, w=rgb0.shape[1], h=rgb0.shape[0])
    else:
        img = dtm._load_rgb_u8_path(Path(args.image), image_max_size=0)  # type: ignore[attr-defined]
        depth0, foc_px = dtm._predict_depthpro_depth(img, device=str(args.depth_device), f_px=None)  # type: ignore[attr-defined]
        k = dtm._intrinsics_from_args(
            w=int(img.shape[1]),
            h=int(img.shape[0]),
            fx=None,
            fy=None,
            cx=None,
            cy=None,
            focal_px=(float(foc_px) if foc_px is not None else None),
        )  # type: ignore[attr-defined]
        rgb0 = img
        meta = {"intrinsics": {"fx": float(k[0, 0]), "fy": float(k[1, 1]), "cx": float(k[0, 2]), "cy": float(k[1, 2])}}

    # Segment
    instances = _instances_from_sam2(
        rgb0,
        min_area=int(args.sam2_min_area),
        max_area_frac=float(args.sam2_max_area_frac),
        points_per_side=int(args.sam2_points_per_side),
        pred_iou_thresh=float(args.sam2_pred_iou_thresh),
        stability_thresh=float(args.sam2_stability_thresh),
        min_mask_region_area=int(args.sam2_min_mask_region_area),
        keep_topk=int(args.keep_topk),
        nms_iou=float(args.nms_iou),
        sam2_model_id=str(args.sam2_model_id),
    )

    if not instances:
        raise RuntimeError("No SAM2 instances found (try lowering --sam2-min-area or --sam2-pred-iou-thresh)")

    # Save instance preview
    inst_out = args.out / "sam2"
    inst_out.mkdir(parents=True, exist_ok=True)
    (inst_out / "instances.json").write_text(json.dumps([{k: v for k, v in c.items() if k != "mask"} for c in instances], indent=2), encoding="utf-8")
    for c in instances:
        Image.fromarray(c["mask"], mode="L").save(inst_out / f"mask_{int(c['instance_id']):03d}.png")

    if int(args.instance_id) >= 0:
        chosen = next((c for c in instances if int(c["instance_id"]) == int(args.instance_id)), None)
        if chosen is None:
            raise RuntimeError(f"Requested instance_id {args.instance_id} not found in kept set")
    else:
        chosen = instances[0]

    mask = chosen["mask"]
    iid = int(chosen["instance_id"])

    # View specs
    n = int(args.n_views)
    yaw_max = float(args.yaw_max_deg)
    if n <= 1:
        yaws = [0.0]
    else:
        yaws = np.linspace(-yaw_max, yaw_max, n, dtype=np.float32).tolist()
    views = [ViewSpec(name=f"view_{i:03d}", yaw_deg=float(y)) for i, y in enumerate(yaws)]

    obj_out = args.out / f"obj_{iid:03d}"

    # If seed=0 treat as deterministic, else allow None
    seed = int(args.inpaint_seed)

    complete_object_multiview(
        rgb0=rgb0,
        depth0=depth0.astype(np.float32),
        k=k.astype(np.float32),
        mask_u8=mask.astype(np.uint8),
        out_dir=obj_out,
        views=views,
        render_step=int(args.render_step),
        points_step=int(args.points_step),
        inpaint=str(args.inpaint),
        inpaint_model=str(args.inpaint_model),
        inpaint_device=str(args.inpaint_device),
        inpaint_steps=int(args.inpaint_steps),
        inpaint_guidance=float(args.inpaint_guidance),
        inpaint_seed=seed,
        depth_device=str(args.depth_device),
        view_max_size=int(args.view_max_size),
        skip_depthpro_views=bool(args.skip_depthpro_views),
        poisson_depth=int(args.poisson_depth),
        voxel_down=float(args.voxel_down),
        skip_mesh=bool(args.skip_mesh),
    )


if __name__ == "__main__":
    main()
