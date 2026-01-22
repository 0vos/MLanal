from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

import depth_to_displacement


@dataclass(frozen=True)
class MeshOutputs:
    out_dir: Path
    obj_path: Path
    mtl_path: Path


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
) -> np.ndarray:
    """Build a simple pinhole intrinsics matrix.

    For single-image mode, we often only know (or can estimate) a focal length in pixels.
    We default principal point to the image center.
    """
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
    return k


def _predict_depthpro_depth(rgb_u8: np.ndarray, device: str, f_px: Optional[float]) -> Tuple[np.ndarray, Optional[float]]:
    """Predict metric depth for a single RGB image using DepthPro.

    This reuses DepthPro wiring already implemented in step1_3d_photo.
    Returns (depth_m, focal_px_or_None).
    """
    # We intentionally call the internal helper so we don't duplicate model-loading logic.
    return depth_to_displacement.step1_3d_photo._predict_depth_depthpro(  # type: ignore[attr-defined]
        rgb=rgb_u8,
        device=str(device),
        f_px=f_px,
    )


def _smooth_depth(depth_m: np.ndarray, method: str, ksize: int, bilateral_sigma_space: float) -> np.ndarray:
    """Denoise metric depth while preserving edges.

    This is meant for "no background deletion" workflows where small depth noise
    makes large planar regions (like floors) look pitted.

    method:
      - none
      - median (robust, CPU, good default)
      - bilateral (edge-preserving; requires opencv-python)
    """
    method = str(method).strip().lower()
    if method in ("none", "off", "0", "false"):
        return depth_m.astype(np.float32)

    k = int(ksize)
    if k <= 1:
        return depth_m.astype(np.float32)
    if (k % 2) == 0:
        k += 1

    d = depth_m.astype(np.float32, copy=False)

    if method == "median":
        try:
            from scipy import ndimage

            return ndimage.median_filter(d, size=int(k)).astype(np.float32)
        except Exception:
            # SciPy should exist in this repo; if not, just return unchanged.
            return d.astype(np.float32)

    if method == "bilateral":
        try:
            import cv2  # type: ignore

            # Depth scale varies per scene; pick sigmaColor relative to median depth.
            valid = d > 0
            if not np.any(valid):
                return d
            z_med = float(np.median(d[valid]))
            sigma_color = max(0.01 * z_med, 0.01)
            sigma_space = float(bilateral_sigma_space) if float(bilateral_sigma_space) > 0 else float(k)
            # OpenCV bilateral expects float32.
            return cv2.bilateralFilter(d, d=int(k), sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space)).astype(np.float32)
        except Exception:
            # If OpenCV isn't installed, fall back to median.
            return _smooth_depth(d, method="median", ksize=int(k), bilateral_sigma_space=float(bilateral_sigma_space))

    raise ValueError("depth_smooth must be one of: none, median, bilateral")


def run_image(
    image_path: Path,
    out_dir: Optional[Path],
    image_max_size: int,
    grid_step: int,
    mode: str,
    plane_width: float,
    plane_height: float,
    relief_strength_m: float,
    # intrinsics overrides
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
    focal_px: Optional[float],
    depth_device: str,
    # depth smoothing
    depth_smooth: str,
    depth_smooth_ksize: int,
    depth_bilateral_sigma_space: float,
    # plane-growth controls
    fit_bottom_frac: float,
    plane_distance_thresh_m: float,
    plane_ransac_iters: int,
    max_height_m: float,
    tri_max_height_step_m: float,
    suppress_masks: Optional[list[Path]],
    suppress_mode: str,
    # geometry-based growth controls
    enable_growth_filter: bool,
    keep_normal_dot: float,
    seed_normal_dot: float,
    seed_band_m: float,
    neighbor_max_height_step_m: float,
    neighbor_max_depth_step_m: float,
    neighbor_max_3d_step_m: float,
    support_depth_pctl: float,
    support_depth_margin_m: float,
    seed_depth_pctl: float,
    seed_depth_margin_m: float,
    keep_nearest_seed_component: bool,
    suppress_vertical_planes: bool,
    vertical_plane_max_dot: float,
    vertical_plane_min_inliers: int,
    vertical_plane_max_planes: int,
    # instance-aware growth (SAM2)
    sam2_instances: bool,
    sam2_model_id: str,
    sam2_points_per_side: int,
    sam2_pred_iou_thresh: float,
    sam2_stability_thresh: float,
    sam2_min_mask_region_area: int,
    sam2_min_area: int,
    sam2_max_area_frac: float,
    stack_contact_dist_m: float,
    stack_min_height_delta_m: float,
    sam2_hard_filter: bool,
    salvage_supported_instances: bool,
    salvage_contact_band_m: float,
    salvage_min_grid_area: int,
    salvage_depth_margin_m: float,
    detect_table_plane: bool,
    table_seed_normal_dot: float,
    table_min_height_m: float,
    table_max_height_m: float,
    table_min_inliers: int,
    tri_max_edge_m: float,
    drop_floating: bool,
) -> MeshOutputs:
    if out_dir is None:
        out_dir = image_path.parent / f"mesh_{image_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_img = _load_rgb_u8_path(image_path, image_max_size=int(image_max_size))
    h, w = scene_img.shape[:2]

    depth, foc_px_pred = _predict_depthpro_depth(scene_img, device=str(depth_device), f_px=focal_px)
    if depth.shape != (h, w):
        d_im = Image.fromarray(depth.astype(np.float32))
        d_im = d_im.resize((w, h), resample=Image.Resampling.BILINEAR)
        depth = np.array(d_im, dtype=np.float32)

    if str(depth_smooth).strip().lower() not in ("none", "off", "0", "false"):
        depth = _smooth_depth(
            depth_m=depth,
            method=str(depth_smooth),
            ksize=int(depth_smooth_ksize),
            bilateral_sigma_space=float(depth_bilateral_sigma_space),
        )

    k = _intrinsics_from_args(w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy, focal_px=(focal_px if focal_px is not None else foc_px_pred))

    # Persist inputs for debugging/reuse.
    Image.fromarray(scene_img, mode="RGB").save(out_dir / "scene.png")
    np.save(out_dir / "depth.npy", depth.astype(np.float32))
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "image": str(image_path),
                "image_max_size": int(image_max_size),
                "mode": str(mode),
                "intrinsics": {"fx": float(k[0, 0]), "fy": float(k[1, 1]), "cx": float(k[0, 2]), "cy": float(k[1, 2])},
                "depth": {"method": "depthpro", "device": str(depth_device), "focal_px_pred": (float(foc_px_pred) if foc_px_pred is not None else None)},
                "depth_smooth": {
                    "method": str(depth_smooth),
                    "ksize": int(depth_smooth_ksize),
                    "bilateral_sigma_space": float(depth_bilateral_sigma_space),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Create mesh topology
    _, uv, faces = _grid_mesh(h, w, step=int(grid_step))

    # Sample positions
    ys = np.arange(0, h, int(grid_step), dtype=np.int32)
    xs = np.arange(0, w, int(grid_step), dtype=np.int32)
    uu, vv = np.meshgrid(xs, ys)

    if mode == "camera":
        vertices = _camera_vertices_from_depth(depth, k=k, xs=uu, ys=vv)
        center = np.nanmedian(vertices, axis=0)
        vertices = (vertices - center).astype(np.float32)
    elif mode == "relief":
        disp01 = depth_to_displacement._displacement_from_depth(  # type: ignore[attr-defined]
            depth,
            mode="disparity",
            clip_percentile=(1.0, 99.0),
            invert=False,
        )
        x = (uu.astype(np.float32) / max(float(w - 1), 1.0) - 0.5) * float(plane_width)
        y = (0.5 - vv.astype(np.float32) / max(float(h - 1), 1.0)) * float(plane_height)
        z = disp01[vv, uu] * float(relief_strength_m)
        vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    elif mode == "plane":
        # Mirror the capture-dir plane path, just with our K+depth.
        fit_step = max(1, int(grid_step))
        y0 = int(max(0.0, 1.0 - float(fit_bottom_frac)) * float(h))
        ys_fit = np.arange(y0, h, fit_step, dtype=np.int32)
        xs_fit = np.arange(0, w, fit_step, dtype=np.int32)
        uu_fit, vv_fit = np.meshgrid(xs_fit, ys_fit)
        pts_fit = _camera_vertices_from_depth(depth, k=k, xs=uu_fit, ys=vv_fit)
        valid_fit = np.isfinite(pts_fit).all(axis=1)
        pts_fit = pts_fit[valid_fit]
        if pts_fit.shape[0] < 500:
            raise RuntimeError("Not enough valid points to fit plane; try lower --grid-step or increase image resolution")

        n, d_arr = _fit_plane_open3d(
            pts_fit,
            distance_threshold=float(plane_distance_thresh_m),
            ransac_n=3,
            num_iterations=int(plane_ransac_iters),
        )
        d = float(d_arr[0])
        if float(n[1]) > 0.0:
            n = -n
            d = -d

        p = _camera_vertices_from_depth(depth, k=k, xs=uu, ys=vv)
        valid = np.isfinite(p).all(axis=1)

        gh = int(ys.shape[0])
        gw = int(xs.shape[0])
        p_grid = p.reshape(gh, gw, 3)
        valid_grid = valid.reshape(gh, gw)

        n_surf = _normals_from_grid_points(p_grid)
        dot_surf = np.abs(np.sum(n_surf * n.reshape(1, 1, 3), axis=-1))

        dist = (p @ n.reshape(3, 1)).reshape(-1) + d
        height = np.clip(dist, 0.0, float(max_height_m)).astype(np.float32)
        height[~valid] = np.nan

        # Suppress (flatten) specific regions (e.g. wall/ceiling) provided as masks.
        if suppress_masks:
            sup = np.zeros((h, w), dtype=np.bool_)
            for mp in suppress_masks:
                with Image.open(mp) as im:
                    im = im.convert("L")
                    if im.size != (w, h):
                        im = im.resize((w, h), resample=Image.Resampling.NEAREST)
                    m = (np.array(im, dtype=np.uint8) > 0)
                sup |= m
            sup_flat = sup[vv, uu].reshape(-1)
            if suppress_mode == "nan":
                height[sup_flat] = np.nan
            elif suppress_mode == "zero":
                height[sup_flat] = 0.0
            else:
                raise ValueError("suppress_mode must be 'nan' or 'zero'")

        height_grid = height.reshape(gh, gw)

        # Default: keep only near-plane content.
        allow_grid = np.isfinite(height_grid) & valid_grid

        # Optional geometry-based growth filtering (reuses existing helpers).
        table_plane = None
        support_vert_mask_flat: Optional[np.ndarray] = None
        if enable_growth_filter:
            # Compute seeds from near-plane band.
            seeds_grid = allow_grid.copy()
            if float(seed_normal_dot) > 0.0:
                seeds_grid &= (dot_surf >= float(seed_normal_dot))
            seeds_grid &= (height_grid <= float(seed_band_m))
            # Depth gates
            z_grid = p_grid[..., 2].astype(np.float32)
            z_allowed = z_grid[np.isfinite(z_grid) & allow_grid]
            if z_allowed.size > 0:
                z_support_p = float(np.percentile(z_allowed, float(support_depth_pctl)))
                allow_grid &= (z_grid <= z_support_p + float(support_depth_margin_m))
                z_seed_p = float(np.percentile(z_allowed, float(seed_depth_pctl)))
                seeds_grid &= (z_grid <= z_seed_p + float(seed_depth_margin_m))

            # Save support vertices for later mesh-component filtering (drop_floating).
            support_vert_mask_flat = seeds_grid.reshape(-1).astype(np.bool_)

            # BFS growth with barriers
            visited = _bfs_connected_mask_barrier(
                seeds=seeds_grid,
                allowed=allow_grid,
                height=height_grid,
                depth_z=z_grid,
                max_height_step_m=float(neighbor_max_height_step_m),
                max_depth_step_m=float(neighbor_max_depth_step_m),
                p_grid=p_grid,
                max_3d_step_m=float(neighbor_max_3d_step_m),
            )

            keep = visited

            # Optional SAM2 filtering (same semantics as capture-dir path: soft by default).
            if sam2_instances:
                try:
                    inst_grid, _ = _sam2_instance_grid(
                        scene_img=scene_img,
                        gh=gh,
                        gw=gw,
                        model_id=str(sam2_model_id),
                        points_per_side=int(sam2_points_per_side),
                        pred_iou_thresh=float(sam2_pred_iou_thresh),
                        stability_thresh=float(sam2_stability_thresh),
                        min_mask_region_area=int(sam2_min_mask_region_area),
                        min_area=int(sam2_min_area),
                        max_area_frac=float(sam2_max_area_frac),
                        out_dir=out_dir,
                    )
                    keep_inst = _supported_instances_keep_mask(
                        inst_grid=inst_grid,
                        p_grid=p_grid,
                        valid_grid=valid_grid,
                        height=height_grid,
                        seeds=seeds_grid,
                        stack_contact_dist_m=float(stack_contact_dist_m),
                        stack_min_height_delta_m=float(stack_min_height_delta_m),
                    )
                    if sam2_hard_filter:
                        keep &= keep_inst
                    else:
                        # soft assist: do not delete geometry solely due to SAM miss
                        keep |= keep_inst
                except Exception:
                    pass

            allow_grid = keep

        # Turn heights into vertices. Plane mesh uses (u,v,height) with a stable basis.
        u_axis, v_axis = _plane_basis(n)
        p_proj = p - (dist.reshape(-1, 1) * n.reshape(1, 3))
        pu = (p_proj @ u_axis.reshape(3, 1)).reshape(-1)
        pv = (p_proj @ v_axis.reshape(3, 1)).reshape(-1)
        verts = np.stack([pu, -pv, height], axis=-1).astype(np.float32)
        # Mask out dropped vertices by setting them to NaN so downstream face dropping works.
        allow_flat = allow_grid.reshape(-1)
        verts[~allow_flat] = np.nan
        vertices = verts

        # Drop overly steep triangles and long-edge bridge triangles.
        faces = _drop_faces_with_nan_vertices(vertices, faces)
        faces = _drop_faces_with_large_height_step(vertices, faces, max_step=float(tri_max_height_step_m))
        if float(tri_max_edge_m) > 0.0:
            faces = _drop_faces_with_long_edges(vertices, faces, max_edge=float(tri_max_edge_m))

        # Finally: after bridge-triangle culling, remove any disconnected mesh components
        # that do not touch the support plane seeds.
        if drop_floating and support_vert_mask_flat is not None and faces.size > 0:
            faces = _keep_face_components_touching_support(
                faces=faces,
                n_vertices=int(vertices.shape[0]),
                support_vertices=support_vert_mask_flat.astype(np.bool_),
            )

    else:
        raise ValueError("mode must be 'camera', 'relief', or 'plane'")

    # Write material + mesh
    material_name = "mat0"
    obj_path = out_dir / "mesh.obj"
    mtl_path = out_dir / "mesh.mtl"

    tex_path = out_dir / "albedo.png"
    Image.fromarray(scene_img, mode="RGB").save(tex_path)

    vertices, uv, faces = _compact_mesh(vertices=vertices, uvs=uv, faces=faces)

    _write_mtl(mtl_path, material_name=material_name, diffuse_tex_rel=tex_path.name)
    _write_obj(
        obj_path,
        mtl_name=mtl_path.name,
        material_name=material_name,
        vertices=vertices,
        uvs=uv,
        faces=faces,
    )

    return MeshOutputs(out_dir=out_dir, obj_path=obj_path, mtl_path=mtl_path)


def _load_u16_png(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("I")
        arr = np.array(im)
    if arr.ndim != 2:
        raise ValueError("Expected a single-channel displacement image")
    if arr.dtype != np.uint16:
        # Pillow sometimes yields int32; convert safely.
        arr = np.clip(arr, 0, 65535).astype(np.uint16)
    return arr


def _write_obj(
    obj_path: Path,
    mtl_name: str,
    material_name: str,
    vertices: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be (N,3)")
    if uvs.ndim != 2 or uvs.shape[1] != 2:
        raise ValueError("uvs must be (N,2)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be (M,3)")

    with obj_path.open("w", encoding="utf-8") as f:
        f.write("# depth_to_mesh generated\n")
        f.write(f"mtllib {mtl_name}\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for vt in uvs:
            f.write(f"vt {vt[0]} {vt[1]}\n")
        f.write(f"usemtl {material_name}\n")
        # OBJ is 1-indexed; we write v/vt pairs (same indexing)
        for tri in faces:
            a, b, c = (int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1)
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")


def _compact_mesh(vertices: np.ndarray, uvs: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove unused/non-finite vertices and remap faces.

    Our OBJ writer assumes a 1:1 mapping between vertex index and UV index.
    This compaction preserves that by compacting both arrays identically.
    """
    if faces.size == 0:
        return vertices, uvs, faces

    n = int(vertices.shape[0])
    used = np.unique(faces.reshape(-1).astype(np.int32))
    used = used[(used >= 0) & (used < n)]
    if used.size == 0:
        return vertices[:0], uvs[:0], faces[:0]

    # Faces should already avoid NaN vertices, but be defensive.
    finite_mask = np.isfinite(vertices).all(axis=1)
    used = used[finite_mask[used]]
    if used.size == 0:
        return vertices[:0], uvs[:0], faces[:0]

    remap = np.full((n,), -1, dtype=np.int32)
    remap[used] = np.arange(int(used.size), dtype=np.int32)
    faces_new = remap[faces]
    keep = (faces_new >= 0).all(axis=1)
    faces_new = faces_new[keep]

    v_new = vertices[used]
    uv_new = uvs[used] if uvs.shape[0] == n else uvs[: v_new.shape[0]]
    return v_new.astype(np.float32), uv_new.astype(np.float32), faces_new.astype(np.int32)


def _write_mtl(mtl_path: Path, material_name: str, diffuse_tex_rel: str) -> None:
    with mtl_path.open("w", encoding="utf-8") as f:
        f.write("# depth_to_mesh material\n")
        f.write(f"newmtl {material_name}\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write("d 1.0\n")
        f.write("illum 1\n")
        f.write(f"map_Kd {diffuse_tex_rel}\n")


def _drop_faces_with_nan_vertices(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return faces
    valid_v = np.isfinite(vertices).all(axis=1)
    keep = valid_v[faces].all(axis=1)
    return faces[keep]


def _drop_faces_with_large_height_step(vertices: np.ndarray, faces: np.ndarray, max_step: float) -> np.ndarray:
    if faces.size == 0:
        return faces
    max_step = float(max_step)
    if max_step <= 0:
        return faces
    hz = vertices[:, 2].astype(np.float32)
    tri_h = hz[faces]  # (M,3)
    keep = (np.nanmax(tri_h, axis=1) - np.nanmin(tri_h, axis=1)) <= max_step
    keep = np.isfinite(keep) & keep
    return faces[keep]


def _drop_faces_with_long_edges(vertices: np.ndarray, faces: np.ndarray, max_edge: float) -> np.ndarray:
    if faces.size == 0:
        return faces
    max_edge = float(max_edge)
    if max_edge <= 0:
        return faces
    tri = vertices[faces].astype(np.float32)  # (M,3,3)
    a = tri[:, 0, :]
    b = tri[:, 1, :]
    c = tri[:, 2, :]
    dab = np.linalg.norm(a - b, axis=1)
    dbc = np.linalg.norm(b - c, axis=1)
    dca = np.linalg.norm(c - a, axis=1)
    keep = np.maximum(np.maximum(dab, dbc), dca) <= max_edge
    keep = np.isfinite(keep) & keep
    return faces[keep]


def _grid_mesh(h: int, w: int, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (vertex_indices, uv, faces) for a regular grid.

    vertex_indices: (gh, gw) int indices into flattened vertex array.
    uv: (N,2)
    faces: (M,3)
    """
    if step <= 0:
        raise ValueError("step must be > 0")

    ys = np.arange(0, h, step, dtype=np.int32)
    xs = np.arange(0, w, step, dtype=np.int32)
    gh = int(ys.shape[0])
    gw = int(xs.shape[0])

    # Flattened vertex order: row-major over (y,x)
    grid = np.arange(gh * gw, dtype=np.int32).reshape(gh, gw)

    # UVs in [0,1], v flipped for typical image coordinates
    uu, vv = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    u = uu.reshape(-1) / max(float(w - 1), 1.0)
    v = 1.0 - (vv.reshape(-1) / max(float(h - 1), 1.0))
    uv = np.stack([u, v], axis=-1).astype(np.float32)

    # Faces
    faces = []
    for iy in range(gh - 1):
        for ix in range(gw - 1):
            a = int(grid[iy, ix])
            b = int(grid[iy, ix + 1])
            c = int(grid[iy + 1, ix])
            d = int(grid[iy + 1, ix + 1])
            # two triangles: (a,b,c) and (b,d,c)
            faces.append((a, b, c))
            faces.append((b, d, c))
    faces_np = np.array(faces, dtype=np.int32)
    return grid, uv, faces_np


def _scale_intrinsics(k: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    kk = k.copy().astype(np.float32)
    kk[0, 0] *= float(scale_x)
    kk[1, 1] *= float(scale_y)
    kk[0, 2] *= float(scale_x)
    kk[1, 2] *= float(scale_y)
    return kk


def _camera_vertices_from_depth(depth_m: np.ndarray, k: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Create vertices in camera coordinates from metric depth and intrinsics.

    xs, ys are 2D grids of pixel indices into depth_m.
    Returns (N,3) float32.
    """
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    z = depth_m[ys, xs].astype(np.float32)
    u = xs.astype(np.float32)
    v = ys.astype(np.float32)

    x = (u - cx) * z / max(fx, 1e-6)
    y = (v - cy) * z / max(fy, 1e-6)
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    return verts


def _fit_plane_open3d(points: np.ndarray, distance_threshold: float, ransac_n: int, num_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fit plane ax+by+cz+d=0 via Open3D RANSAC. Returns (n(3,), d)."""
    try:
        import open3d as o3d  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("open3d is required for plane fitting. Install: pip install open3d") from e

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=float(distance_threshold),
        ransac_n=int(ransac_n),
        num_iterations=int(num_iterations),
    )
    a, b, c, d = [float(x) for x in plane_model]
    n = np.array([a, b, c], dtype=np.float32)
    norm = float(np.linalg.norm(n))
    if norm <= 1e-6:
        raise RuntimeError("Degenerate plane normal")
    n = n / norm
    d = float(d) / norm
    return n, np.array([d], dtype=np.float32)


def _plane_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return orthonormal basis (u,v) spanning the plane with normal n."""
    n = n.astype(np.float32)
    # Pick a helper vector not parallel to n
    a = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(a, n))) > 0.9:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u = np.cross(a, n)
    u = u / max(float(np.linalg.norm(u)), 1e-6)
    v = np.cross(n, u)
    v = v / max(float(np.linalg.norm(v)), 1e-6)
    return u.astype(np.float32), v.astype(np.float32)


def _normals_from_grid_points(p_grid: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals from a grid of 3D points using central differences.

    p_grid: (gh, gw, 3)
    returns: (gh, gw, 3)
    """
    gh, gw = p_grid.shape[:2]
    dpdx = np.zeros_like(p_grid, dtype=np.float32)
    dpdy = np.zeros_like(p_grid, dtype=np.float32)

    if gw >= 3:
        dpdx[:, 1:-1, :] = (p_grid[:, 2:, :] - p_grid[:, :-2, :]) * 0.5
    if gh >= 3:
        dpdy[1:-1, :, :] = (p_grid[2:, :, :] - p_grid[:-2, :, :]) * 0.5

    n = np.cross(dpdx, dpdy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(norm, 1e-6)
    # Replace NaNs/Infs with a default
    n = np.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)
    return n.astype(np.float32)


def _bfs_connected_mask(seeds: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """4-neighbor BFS over a grid.

    seeds, allowed: (gh, gw) bool
    returns: visited (gh, gw) bool
    """
    gh, gw = seeds.shape
    visited = np.zeros((gh, gw), dtype=np.bool_)
    if not np.any(seeds):
        return visited

    from collections import deque

    q = deque()
    ys, xs = np.nonzero(seeds & allowed)
    for y, x in zip(ys.tolist(), xs.tolist()):
        visited[y, x] = True
        q.append((y, x))

    while q:
        y, x = q.popleft()
        if y > 0 and (not visited[y - 1, x]) and allowed[y - 1, x]:
            visited[y - 1, x] = True
            q.append((y - 1, x))
        if y + 1 < gh and (not visited[y + 1, x]) and allowed[y + 1, x]:
            visited[y + 1, x] = True
            q.append((y + 1, x))
        if x > 0 and (not visited[y, x - 1]) and allowed[y, x - 1]:
            visited[y, x - 1] = True
            q.append((y, x - 1))
        if x + 1 < gw and (not visited[y, x + 1]) and allowed[y, x + 1]:
            visited[y, x + 1] = True
            q.append((y, x + 1))

    return visited


def _bfs_connected_mask_barrier(
    seeds: np.ndarray,
    allowed: np.ndarray,
    height: np.ndarray,
    depth_z: np.ndarray,
    max_height_step_m: float,
    max_depth_step_m: float,
    p_grid: Optional[np.ndarray] = None,
    max_3d_step_m: float = 0.0,
) -> np.ndarray:
    """4-neighbor BFS with edge barriers.

    This is intentionally conservative: it avoids "leaking" into far background
    across depth discontinuities (occlusion edges) and avoids growing across
    large height jumps.

    seeds, allowed, height, depth_z: (gh, gw)
    returns: visited (gh, gw)
    """
    gh, gw = seeds.shape
    visited = np.zeros((gh, gw), dtype=np.bool_)
    if not np.any(seeds):
        return visited

    from collections import deque

    q = deque()
    ys, xs = np.nonzero(seeds & allowed)
    for y, x in zip(ys.tolist(), xs.tolist()):
        visited[y, x] = True
        q.append((y, x))

    max_h = float(max_height_step_m)
    max_d = float(max_depth_step_m)
    max_3d = float(max_3d_step_m)

    def ok_edge(y0: int, x0: int, y1: int, x1: int) -> bool:
        if not allowed[y1, x1]:
            return False
        h0 = float(height[y0, x0])
        h1 = float(height[y1, x1])
        z0 = float(depth_z[y0, x0])
        z1 = float(depth_z[y1, x1])
        if not (np.isfinite(h0) and np.isfinite(h1) and np.isfinite(z0) and np.isfinite(z1)):
            return False
        if max_h > 0.0 and abs(h1 - h0) > max_h:
            return False
        if max_d > 0.0 and abs(z1 - z0) > max_d:
            return False
        # Density barrier: if local 3D spacing suddenly becomes huge, it's usually an occlusion gap
        # and any connectivity across it tends to become "stringing".
        if max_3d > 0.0 and p_grid is not None:
            p0 = p_grid[y0, x0]
            p1 = p_grid[y1, x1]
            if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                return False
            if float(np.linalg.norm(p1 - p0)) > max_3d:
                return False
        return True

    while q:
        y, x = q.popleft()
        if y > 0 and (not visited[y - 1, x]) and ok_edge(y, x, y - 1, x):
            visited[y - 1, x] = True
            q.append((y - 1, x))
        if y + 1 < gh and (not visited[y + 1, x]) and ok_edge(y, x, y + 1, x):
            visited[y + 1, x] = True
            q.append((y + 1, x))
        if x > 0 and (not visited[y, x - 1]) and ok_edge(y, x, y, x - 1):
            visited[y, x - 1] = True
            q.append((y, x - 1))
        if x + 1 < gw and (not visited[y, x + 1]) and ok_edge(y, x, y, x + 1):
            visited[y, x + 1] = True
            q.append((y, x + 1))

    return visited


def _segment_planes_open3d(
    points: np.ndarray,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
    max_planes: int,
    min_inliers: int,
) -> list[tuple[np.ndarray, float]]:
    """Iteratively segment multiple planes with Open3D.

    Returns list of planes as (unit_normal(3,), d) for ax+by+cz+d=0.
    """
    try:
        import open3d as o3d  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("open3d is required for plane fitting. Install: pip install open3d") from e

    if points.shape[0] < int(min_inliers):
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    planes: list[tuple[np.ndarray, float]] = []
    remaining = pcd
    for _ in range(int(max_planes)):
        if np.asarray(remaining.points).shape[0] < int(min_inliers):
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=float(distance_threshold),
            ransac_n=int(ransac_n),
            num_iterations=int(num_iterations),
        )
        if len(inliers) < int(min_inliers):
            break
        a, b, c, d = [float(x) for x in plane_model]
        n = np.array([a, b, c], dtype=np.float32)
        norm = float(np.linalg.norm(n))
        if norm <= 1e-6:
            break
        n = n / norm
        d = float(d) / norm
        planes.append((n.astype(np.float32), float(d)))
        remaining = remaining.select_by_index(inliers, invert=True)

    return planes


def _connected_components_4(mask: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return 4-neighbor connected components for a boolean grid.

    Returns list of (size, ys, xs).
    """
    gh, gw = mask.shape
    visited = np.zeros_like(mask, dtype=np.bool_)
    comps: list[tuple[int, np.ndarray, np.ndarray]] = []
    if not np.any(mask):
        return comps

    from collections import deque

    for sy in range(gh):
        for sx in range(gw):
            if not mask[sy, sx] or visited[sy, sx]:
                continue
            q = deque([(sy, sx)])
            visited[sy, sx] = True
            ys: list[int] = []
            xs: list[int] = []
            while q:
                y, x = q.popleft()
                ys.append(y)
                xs.append(x)
                if y > 0 and mask[y - 1, x] and (not visited[y - 1, x]):
                    visited[y - 1, x] = True
                    q.append((y - 1, x))
                if y + 1 < gh and mask[y + 1, x] and (not visited[y + 1, x]):
                    visited[y + 1, x] = True
                    q.append((y + 1, x))
                if x > 0 and mask[y, x - 1] and (not visited[y, x - 1]):
                    visited[y, x - 1] = True
                    q.append((y, x - 1))
                if x + 1 < gw and mask[y, x + 1] and (not visited[y, x + 1]):
                    visited[y, x + 1] = True
                    q.append((y, x + 1))

            ys_a = np.array(ys, dtype=np.int32)
            xs_a = np.array(xs, dtype=np.int32)
            comps.append((int(len(ys)), ys_a, xs_a))

    comps.sort(key=lambda t: t[0], reverse=True)
    return comps


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        p = int(self.parent[x])
        if p != x:
            self.parent[x] = self.find(p)
        return int(self.parent[x])

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if int(self.rank[ra]) < int(self.rank[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        if int(self.rank[ra]) == int(self.rank[rb]):
            self.rank[ra] = np.int8(int(self.rank[ra]) + 1)


def _keep_face_components_touching_support(
    faces: np.ndarray,
    n_vertices: int,
    support_vertices: np.ndarray,
) -> np.ndarray:
    """Keep only face components (connected via shared vertices) that touch support vertices."""
    if faces.size == 0:
        return faces
    if support_vertices.shape[0] != int(n_vertices):
        raise ValueError("support_vertices must be length n_vertices")

    m = int(faces.shape[0])
    dsu = _DSU(m)

    # Build (vertex, face_idx) pairs for all face corners.
    v = faces.reshape(-1).astype(np.int32)
    f = np.repeat(np.arange(m, dtype=np.int32), 3)
    order = np.argsort(v, kind="mergesort")
    v = v[order]
    f = f[order]

    # Union all faces that share a vertex.
    start = 0
    while start < v.shape[0]:
        vv = int(v[start])
        end = start + 1
        while end < v.shape[0] and int(v[end]) == vv:
            end += 1
        if end - start >= 2:
            base = int(f[start])
            for i in range(start + 1, end):
                dsu.union(base, int(f[i]))
        start = end

    # Determine which components touch support.
    comp_support: dict[int, bool] = {}
    for i in range(m):
        r = dsu.find(i)
        if r in comp_support:
            continue
        comp_support[r] = False

    for i in range(m):
        r = dsu.find(i)
        if comp_support[r]:
            continue
        a, b, c = int(faces[i, 0]), int(faces[i, 1]), int(faces[i, 2])
        if support_vertices[a] or support_vertices[b] or support_vertices[c]:
            comp_support[r] = True

    keep = np.zeros((m,), dtype=np.bool_)
    for i in range(m):
        keep[i] = bool(comp_support[dsu.find(i)])

    return faces[keep]


def _torch_auto_device() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _sam2_auto_masks(
    image_rgb_u8: np.ndarray,
    model_id: str,
    device: str,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    min_mask_region_area: int,
) -> list[dict]:
    """Generate prompt-free instance masks using SAM2 AMG."""
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
        from sam2.build_sam import build_sam2_hf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("sam2 is required for --sam2-instances. Install it in the venv.") from e

    model = build_sam2_hf(model_id=model_id, device=device)
    amg = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=int(points_per_side),
        pred_iou_thresh=float(pred_iou_thresh),
        stability_score_thresh=float(stability_score_thresh),
        min_mask_region_area=int(min_mask_region_area),
        output_mode="binary_mask",
    )
    return amg.generate(image_rgb_u8)


def _instance_id_map_from_sam2_masks(
    masks: list[dict],
    h: int,
    w: int,
    min_area: int,
    max_area_frac: float,
) -> tuple[np.ndarray, list[dict]]:
    """Create a single per-pixel instance-id map (non-overlapping).

    We sort by SAM2 scores and assign pixels to the best mask first.
    Returns (inst_id[h,w] int32, kept_meta).
    """
    inst = np.full((h, w), -1, dtype=np.int32)
    kept: list[dict] = []

    total = float(h * w)
    # Prefer stable / high iou / large area (but cap huge masks later)
    def score(m: dict) -> tuple:
        return (
            float(m.get("stability_score", 0.0)),
            float(m.get("predicted_iou", 0.0)),
            float(m.get("area", 0.0)),
        )

    for m in sorted(masks, key=score, reverse=True):
        seg = m.get("segmentation")
        if seg is None:
            continue
        seg = np.asarray(seg).astype(bool)
        if seg.shape != (h, w):
            continue
        area = int(m.get("area", int(seg.sum())))
        if area < int(min_area):
            continue
        if float(area) / max(total, 1.0) > float(max_area_frac):
            continue

        # Assign pixels not yet claimed
        claim = seg & (inst < 0)
        if not np.any(claim):
            continue
        new_id = len(kept)
        inst[claim] = int(new_id)

        meta = {
            "id": int(new_id),
            "area": int(area),
            "predicted_iou": float(m.get("predicted_iou", 0.0)),
            "stability_score": float(m.get("stability_score", 0.0)),
            "bbox": [int(x) for x in (m.get("bbox") or [])],
        }
        kept.append(meta)

    return inst, kept


def _supported_instances_via_stacking(
    inst_grid: np.ndarray,
    valid_grid: np.ndarray,
    support_seed_grid: np.ndarray,
    p_grid: np.ndarray,
    height_grid: np.ndarray,
    contact_dist_m: float,
    stack_min_height_delta_m: float,
) -> np.ndarray:
    """Return keep mask (gh,gw) based on support + stacking relations.

    - Base supported: any instance that touches support_seed_grid.
    - Stacked: instance that has close 3D contact to a supported instance and is above it.
    """
    gh, gw = inst_grid.shape
    n_inst = int(inst_grid.max()) + 1 if inst_grid.size > 0 else 0
    if n_inst <= 0:
        return np.zeros((gh, gw), dtype=np.bool_)

    # Median height per instance
    med_h = np.full((n_inst,), np.nan, dtype=np.float32)
    for i in range(n_inst):
        m = (inst_grid == i) & valid_grid & np.isfinite(height_grid)
        if np.any(m):
            med_h[i] = float(np.nanmedian(height_grid[m]))

    # Base supported
    supported = set(int(i) for i in np.unique(inst_grid[support_seed_grid & (inst_grid >= 0)]) if i >= 0)
    if not supported:
        return np.zeros((gh, gw), dtype=np.bool_)

    # Build adjacency edges from 4-neighbor transitions.
    # We keep a single boolean adjacency and a minimal contact height relation.
    adj: list[tuple[int, int]] = []
    max_d = float(contact_dist_m)
    for y in range(gh):
        for x in range(gw):
            if not valid_grid[y, x]:
                continue
            a = int(inst_grid[y, x])
            if a < 0:
                continue
            pa = p_grid[y, x]
            if not np.isfinite(pa).all():
                continue
            # right
            if x + 1 < gw and valid_grid[y, x + 1]:
                b = int(inst_grid[y, x + 1])
                if b >= 0 and b != a:
                    pb = p_grid[y, x + 1]
                    if np.isfinite(pb).all() and float(np.linalg.norm(pa - pb)) <= max_d:
                        adj.append((a, b))
            # down
            if y + 1 < gh and valid_grid[y + 1, x]:
                b = int(inst_grid[y + 1, x])
                if b >= 0 and b != a:
                    pb = p_grid[y + 1, x]
                    if np.isfinite(pb).all() and float(np.linalg.norm(pa - pb)) <= max_d:
                        adj.append((a, b))

    # Propagate support upward in height.
    delta = float(stack_min_height_delta_m)
    changed = True
    # Limit iterations to avoid worst-case cycles
    for _ in range(32):
        if not changed:
            break
        changed = False
        for a, b in adj:
            ha = float(med_h[a]) if np.isfinite(med_h[a]) else None
            hb = float(med_h[b]) if np.isfinite(med_h[b]) else None
            if ha is None or hb is None:
                continue
            # If a is supported and b is above a, keep b.
            if a in supported and b not in supported and hb >= ha + delta:
                supported.add(b)
                changed = True
            # Symmetric
            if b in supported and a not in supported and ha >= hb + delta:
                supported.add(a)
                changed = True

    keep = (inst_grid >= 0) & np.isin(inst_grid, np.array(sorted(supported), dtype=np.int32))
    return keep.astype(np.bool_)


def run(
    capture_dir: Path,
    frame_index: int,
    out_dir: Optional[Path],
    image_max_size: int,
    grid_step: int,
    mode: str,
    plane_width: float,
    plane_height: float,
    relief_strength_m: float,
    # plane-growth controls
    fit_bottom_frac: float,
    plane_distance_thresh_m: float,
    plane_ransac_iters: int,
    max_height_m: float,
    tri_max_height_step_m: float,
    suppress_masks: Optional[list[Path]],
    suppress_mode: str,
    # geometry-based growth controls
    enable_growth_filter: bool,
    keep_normal_dot: float,
    seed_normal_dot: float,
    seed_band_m: float,
    neighbor_max_height_step_m: float,
    neighbor_max_depth_step_m: float,
    neighbor_max_3d_step_m: float,
    support_depth_pctl: float,
    support_depth_margin_m: float,
    seed_depth_pctl: float,
    seed_depth_margin_m: float,
    keep_nearest_seed_component: bool,
    suppress_vertical_planes: bool,
    vertical_plane_max_dot: float,
    vertical_plane_min_inliers: int,
    vertical_plane_max_planes: int,
    # instance-aware growth (SAM2)
    sam2_instances: bool,
    sam2_model_id: str,
    sam2_points_per_side: int,
    sam2_pred_iou_thresh: float,
    sam2_stability_thresh: float,
    sam2_min_mask_region_area: int,
    sam2_min_area: int,
    sam2_max_area_frac: float,
    stack_contact_dist_m: float,
    stack_min_height_delta_m: float,
    sam2_hard_filter: bool,
    salvage_supported_instances: bool,
    salvage_contact_band_m: float,
    salvage_min_grid_area: int,
    salvage_depth_margin_m: float,
    detect_table_plane: bool,
    table_seed_normal_dot: float,
    table_min_height_m: float,
    table_max_height_m: float,
    table_min_inliers: int,
    tri_max_edge_m: float,
    drop_floating: bool,
) -> MeshOutputs:
    if out_dir is None:
        out_dir = capture_dir / f"mesh_{int(frame_index):06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fast path: disable segmentation + inpaint, just DepthPro.
    disp_out = depth_to_displacement.run(
        capture_dir=capture_dir,
        frame_index=frame_index,
        mask_method="none",
        mask_dilate=0,
        mask_prompt="person.",
        langsam_sam_type="sam2.1_hiera_large",
        langsam_box_threshold=0.15,
        langsam_text_threshold=0.15,
        mask_shape="original",
        inpaint_method="none",
        inpaint_radius=0,
        inpaint_device="auto",
        inpaint_model="runwayml/stable-diffusion-inpainting",
        inpaint_steps=25,
        inpaint_guidance=7.5,
        inpaint_seed=None,
        depth_method="depthpro",
        depth_model="MiDaS_small",
        depth_device="auto",
        min_depth=0.5,
        max_depth=10.0,
        out_dir=out_dir,
        displacement_mode="disparity",
        invert=False,
        clip_lo=1.0,
        clip_hi=99.0,
        normal_strength=1.0,
        normal_method="depth",
        normal_y_flip=False,
        detail_sigma=0.0,
        far_detail_suppression=0.0,
        guided_radius=0,
        guided_eps=1e-3,
        bilateral_d=0,
        bilateral_sigma_color=0.05,
        bilateral_sigma_space=9.0,
    )

    # Optionally downscale to keep the mesh reasonable
    with Image.open(disp_out.scene_image_path) as im:
        im = im.convert("RGB")
        w0, h0 = im.size
        if image_max_size > 0 and max(w0, h0) > int(image_max_size):
            scale = float(image_max_size) / float(max(w0, h0))
            w1 = int(round(w0 * scale))
            h1 = int(round(h0 * scale))
            im = im.resize((w1, h1), resample=Image.Resampling.LANCZOS)
        scene_img = np.array(im, dtype=np.uint8)

    depth = np.load(disp_out.depth_refined_npy_path).astype(np.float32)
    if depth.shape[:2] != scene_img.shape[:2]:
        # Resize depth to match (if we downscaled scene)
        d_im = Image.fromarray(depth.astype(np.float32))
        d_im = d_im.resize((scene_img.shape[1], scene_img.shape[0]), resample=Image.Resampling.BILINEAR)
        depth = np.array(d_im, dtype=np.float32)

    h, w = depth.shape

    # Create mesh topology
    _, uv, faces = _grid_mesh(h, w, step=int(grid_step))

    # Sample positions
    ys = np.arange(0, h, int(grid_step), dtype=np.int32)
    xs = np.arange(0, w, int(grid_step), dtype=np.int32)
    uu, vv = np.meshgrid(xs, ys)

    if mode == "camera":
        # Compute intrinsics for this frame and scale to the resized raster.
        frames = list(depth_to_displacement.step1_3d_photo.read_frames_jsonl(capture_dir))
        if not frames:
            raise RuntimeError(f"No frames in {capture_dir}")
        idx = int(frame_index)
        if idx < 0:
            idx = len(frames) + idx
        if idx < 0 or idx >= len(frames):
            raise IndexError(f"frame_index {frame_index} out of range (0..{len(frames)-1})")
        rec = frames[idx]
        k, w_saved, h_saved = depth_to_displacement.step1_3d_photo.get_intrinsics_for_saved_image(rec, capture_dir)

        scale_x = float(w) / float(w_saved)
        scale_y = float(h) / float(h_saved)
        k = _scale_intrinsics(k, scale_x=scale_x, scale_y=scale_y)

        vertices = _camera_vertices_from_depth(depth, k=k, xs=uu, ys=vv)

        # Center mesh around origin for nicer viewing.
        center = np.nanmedian(vertices, axis=0)
        vertices = (vertices - center).astype(np.float32)

    elif mode == "relief":
        # Relief mode: generate a flat plane and displace along +Z.
        # (Not real-scale unless you tune relief_strength_m.)
        disp01 = depth_to_displacement._displacement_from_depth(  # type: ignore[attr-defined]
            depth,
            mode="disparity",
            clip_percentile=(1.0, 99.0),
            invert=False,
        )
        x = (uu.astype(np.float32) / max(float(w - 1), 1.0) - 0.5) * float(plane_width)
        y = (0.5 - vv.astype(np.float32) / max(float(h - 1), 1.0)) * float(plane_height)
        z = disp01[vv, uu] * float(relief_strength_m)
        vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    elif mode == "plane":
        # Fit a dominant plane from the bottom part of the image (likely floor/table).
        frames = list(depth_to_displacement.step1_3d_photo.read_frames_jsonl(capture_dir))
        if not frames:
            raise RuntimeError(f"No frames in {capture_dir}")
        idx = int(frame_index)
        if idx < 0:
            idx = len(frames) + idx
        if idx < 0 or idx >= len(frames):
            raise IndexError(f"frame_index {frame_index} out of range (0..{len(frames)-1})")
        rec = frames[idx]
        k, w_saved, h_saved = depth_to_displacement.step1_3d_photo.get_intrinsics_for_saved_image(rec, capture_dir)

        scale_x = float(w) / float(w_saved)
        scale_y = float(h) / float(h_saved)
        k = _scale_intrinsics(k, scale_x=scale_x, scale_y=scale_y)

        # Points for fitting: sample a denser grid in the bottom fraction.
        fit_step = max(1, int(grid_step))
        y0 = int(max(0.0, 1.0 - float(fit_bottom_frac)) * float(h))
        ys_fit = np.arange(y0, h, fit_step, dtype=np.int32)
        xs_fit = np.arange(0, w, fit_step, dtype=np.int32)
        uu_fit, vv_fit = np.meshgrid(xs_fit, ys_fit)

        pts_fit = _camera_vertices_from_depth(depth, k=k, xs=uu_fit, ys=vv_fit)
        valid_fit = np.isfinite(pts_fit).all(axis=1)
        pts_fit = pts_fit[valid_fit]
        if pts_fit.shape[0] < 500:
            raise RuntimeError("Not enough valid points to fit plane; try lower --grid-step or increase image resolution")

        n, d_arr = _fit_plane_open3d(
            pts_fit,
            distance_threshold=float(plane_distance_thresh_m),
            ransac_n=3,
            num_iterations=int(plane_ransac_iters),
        )
        d = float(d_arr[0])

        # Orient normal so that +height roughly means "above" the plane.
        # In our camera coords (x right, y down, z forward), floor normal often points up => ny < 0.
        if float(n[1]) > 0.0:
            n = -n
            d = -d

        # Compute per-vertex camera points, project to plane, and represent as (u,v,height).
        p = _camera_vertices_from_depth(depth, k=k, xs=uu, ys=vv)
        valid = np.isfinite(p).all(axis=1)

        gh = int(ys.shape[0])
        gw = int(xs.shape[0])
        p_grid = p.reshape(gh, gw, 3)
        valid_grid = valid.reshape(gh, gw)

        n_surf = _normals_from_grid_points(p_grid)
        # Use absolute dot; some normals can flip
        dot_surf = np.abs(np.sum(n_surf * n.reshape(1, 1, 3), axis=-1))

        dist = (p @ n.reshape(3, 1)).reshape(-1) + d
        height = np.clip(dist, 0.0, float(max_height_m)).astype(np.float32)
        height[~valid] = np.nan

        # Suppress (flatten) specific semantic regions (e.g. wall/ceiling) provided as masks.
        if suppress_masks:
            sup = np.zeros((h, w), dtype=np.bool_)
            for mp in suppress_masks:
                with Image.open(mp) as im:
                    im = im.convert("L")
                    if im.size != (w, h):
                        im = im.resize((w, h), resample=Image.Resampling.NEAREST)
                    m = (np.array(im, dtype=np.uint8) > 0)
                sup |= m
            # height is defined per grid vertex (uu/vv sampling), not per pixel.
            sup_flat = sup[vv, uu].reshape(-1)
            if suppress_mode == "nan":
                height[sup_flat] = np.nan
            elif suppress_mode == "zero":
                height[sup_flat] = 0.0
            else:
                raise ValueError("suppress_mode must be 'zero' or 'nan'")

        # Geometry-based growth filter:
        # Keep only regions that can plausibly be supported by floor/table and remain connected
        # without leaking across large depth discontinuities.
        table_plane = None
        if enable_growth_filter and detect_table_plane:
            # Candidate points: horizontal-ish surfaces within a height band.
            dist_grid = dist.reshape(gh, gw)
            cand = valid_grid & (dist_grid >= float(table_min_height_m)) & (dist_grid <= float(table_max_height_m))
            # Normals from depth can be noisy; allow disabling this constraint for table detection.
            if float(table_seed_normal_dot) > 0.0:
                cand &= (dot_surf >= float(table_seed_normal_dot))
            pts_cand = p_grid[cand].reshape(-1, 3)
            if pts_cand.shape[0] >= int(table_min_inliers):
                try:
                    n2, d2_arr = _fit_plane_open3d(
                        pts_cand,
                        distance_threshold=float(plane_distance_thresh_m),
                        ransac_n=3,
                        num_iterations=int(plane_ransac_iters),
                    )
                    d2 = float(d2_arr[0])
                    # Require roughly parallel to the floor plane
                    if float(np.dot(n2, n)) < 0:
                        n2 = -n2
                        d2 = -d2
                    if float(np.dot(n2, n)) >= 0.90:
                        table_plane = (n2.astype(np.float32), float(d2))
                except Exception:
                    table_plane = None

        vertical_plane_mask = np.zeros((gh, gw), dtype=np.bool_)
        if enable_growth_filter and suppress_vertical_planes:
            # Find dominant vertical-ish planes (walls) and suppress them.
            # This is geometric, not semantic: it reduces background walls without
            # depending on wall/ceiling classifiers.
            pts_all = p[valid].astype(np.float32)
            if pts_all.shape[0] >= int(vertical_plane_min_inliers):
                try:
                    planes = _segment_planes_open3d(
                        pts_all,
                        distance_threshold=float(plane_distance_thresh_m),
                        ransac_n=3,
                        num_iterations=int(plane_ransac_iters),
                        max_planes=int(vertical_plane_max_planes),
                        min_inliers=int(vertical_plane_min_inliers),
                    )
                    for (pn, pd) in planes:
                        # vertical-ish relative to support plane normal
                        if abs(float(np.dot(pn, n))) > float(vertical_plane_max_dot):
                            continue
                        # avoid suppressing detected tabletop plane
                        if table_plane is not None and abs(float(np.dot(pn, table_plane[0]))) > 0.9:
                            continue
                        distp = (p @ pn.reshape(3, 1)).reshape(-1) + float(pd)
                        distp_grid = distp.reshape(gh, gw)
                        vertical_plane_mask |= (valid_grid & (np.abs(distp_grid) <= float(plane_distance_thresh_m)))
                except Exception:
                    vertical_plane_mask = np.zeros((gh, gw), dtype=np.bool_)

            # Save for debugging
            if np.any(vertical_plane_mask):
                dbg = (vertical_plane_mask.astype(np.uint8) * 255)
                Image.fromarray(dbg, mode="L").save(out_dir / "mask_vertical_planes.png")

        support_vert_mask_flat: Optional[np.ndarray] = None
        if enable_growth_filter:
            dist_grid = dist.reshape(gh, gw)
            support_dist_floor = np.abs(dist_grid)
            support_dist_table = None
            if table_plane is not None:
                n2, d2 = table_plane
                dist2 = (p @ n2.reshape(3, 1)).reshape(-1) + float(d2)
                dist2_grid = dist2.reshape(gh, gw)
                support_dist_table = np.abs(dist2_grid)
                support_dist = np.minimum(support_dist_floor, support_dist_table)
            else:
                support_dist = support_dist_floor

            # Seeds: near a support plane. Do NOT require a horizontal surface normal here;
            # otherwise we miss contact regions on vertical sides and table edges.
            seeds_floor = valid_grid & (support_dist_floor <= float(seed_band_m))
            seeds_table = (
                (valid_grid & (support_dist_table <= float(seed_band_m)))
                if support_dist_table is not None
                else np.zeros_like(seeds_floor)
            )
            seeds = seeds_floor | seeds_table

            # Allowed: valid vertices, optionally filtered by normal alignment.
            # NOTE: keep_normal_dot is a *soft* constraint; default is permissive.
            allowed = valid_grid.copy()
            if float(keep_normal_dot) > 0.0:
                allowed &= (dot_surf >= float(keep_normal_dot))

            # Suppress dominant vertical planes (walls) from growth.
            if np.any(vertical_plane_mask):
                allowed &= ~vertical_plane_mask

            # Distance gating: cap how far we allow the grown region to go.
            # We derive this from the support seeds depth distribution; anything much farther is
            # typically background (and is a common source of stringing).
            depth_z = p_grid[:, :, 2].astype(np.float32)

            # 0) Filter seeds by depth to avoid far-away floor/ground creating their own seed islands.
            seeds0 = seeds.copy()
            z0 = depth_z[seeds0]
            z0 = z0[np.isfinite(z0)]
            if z0.size > 0 and float(seed_depth_pctl) > 0.0:
                pctl0 = float(np.clip(seed_depth_pctl, 50.0, 100.0))
                z_seed_base = float(np.percentile(z0, pctl0))
                z_seed_max = z_seed_base + float(seed_depth_margin_m)
                seeds &= (depth_z <= z_seed_max)
                Image.fromarray(((depth_z <= z_seed_max).astype(np.uint8) * 255), mode="L").save(out_dir / "mask_seed_depth_keep.png")

            # 1) Keep only the nearest seed connected component (removes disconnected background islands).
            seeds_main = seeds.copy()
            if keep_nearest_seed_component:
                # Keep nearest component per support plane (floor and table separately), then union.
                def pick_nearest_component(seed_mask: np.ndarray) -> np.ndarray:
                    comps = _connected_components_4(seed_mask)
                    if not comps:
                        return np.zeros_like(seed_mask)
                    best = None
                    best_z = None
                    for _, cy, cx in comps[:10]:
                        zc = depth_z[cy, cx]
                        zc = zc[np.isfinite(zc)]
                        if zc.size == 0:
                            continue
                        mz = float(np.median(zc))
                        if best is None or mz < float(best_z):
                            best = (cy, cx)
                            best_z = mz
                    outm = np.zeros_like(seed_mask)
                    if best is not None:
                        cy, cx = best
                        outm[cy, cx] = True
                    return outm

                main_floor = pick_nearest_component(seeds_floor)
                main_table = pick_nearest_component(seeds_table) if support_dist_table is not None else np.zeros_like(seeds_floor)
                seeds_main = main_floor | main_table

            Image.fromarray((seeds.astype(np.uint8) * 255), mode="L").save(out_dir / "mask_seeds.png")
            Image.fromarray((seeds_main.astype(np.uint8) * 255), mode="L").save(out_dir / "mask_seeds_main.png")

            # 2) Distance gating: cap how far we allow the grown region to go.
            z_seed = depth_z[seeds_main]
            z_seed = z_seed[np.isfinite(z_seed)]
            if z_seed.size > 0 and float(support_depth_pctl) > 0.0:
                pctl = float(np.clip(support_depth_pctl, 50.0, 100.0))
                z_base = float(np.percentile(z_seed, pctl))
                z_max = z_base + float(support_depth_margin_m)
                allowed &= (depth_z <= z_max)
                Image.fromarray(((depth_z <= z_max).astype(np.uint8) * 255), mode="L").save(out_dir / "mask_depth_keep.png")

            # Optional: instance-aware support filtering using SAM2 prompt-free masks.
            seeds_extra = np.zeros((gh, gw), dtype=np.bool_)
            if sam2_instances:
                inst_id, inst_meta = _instance_id_map_from_sam2_masks(
                    _sam2_auto_masks(
                        scene_img,
                        model_id=str(sam2_model_id),
                        device=_torch_auto_device(),
                        points_per_side=int(sam2_points_per_side),
                        pred_iou_thresh=float(sam2_pred_iou_thresh),
                        stability_score_thresh=float(sam2_stability_thresh),
                        min_mask_region_area=int(sam2_min_mask_region_area),
                    ),
                    h=h,
                    w=w,
                    min_area=int(sam2_min_area),
                    max_area_frac=float(sam2_max_area_frac),
                )
                # Save meta for debugging
                (out_dir / "sam2_instances.json").write_text(
                    __import__("json").dumps({"model_id": str(sam2_model_id), "instances": inst_meta}, indent=2),
                    encoding="utf-8",
                )
                # Sample to the mesh grid
                inst_grid = inst_id[vv, uu].reshape(gh, gw)

                # Keep instances that touch support and those stacked above them.
                keep_inst_grid = _supported_instances_via_stacking(
                    inst_grid=inst_grid,
                    valid_grid=valid_grid,
                    support_seed_grid=seeds_main,
                    p_grid=p_grid,
                    height_grid=height.reshape(gh, gw),
                    contact_dist_m=float(stack_contact_dist_m),
                    stack_min_height_delta_m=float(stack_min_height_delta_m),
                )

                # SAM2 is useful, but it can miss objects. By default we treat it as a helper
                # (for seeding/salvage + debugging), not a hard gate.
                if sam2_hard_filter:
                    allowed &= keep_inst_grid

                # Save pixel-space keep mask for quick visual inspection
                supported_ids = np.unique(inst_grid[keep_inst_grid])
                supported_ids = supported_ids[supported_ids >= 0]
                keep_pix = (inst_id >= 0) & np.isin(inst_id, supported_ids)
                Image.fromarray((keep_pix.astype(np.uint8) * 255), mode="L").save(out_dir / "mask_sam2_keep.png")

                # 2b) Salvage: if a nearby, meaningful instance touches the support plane (floor/table)
                # but is disconnected due to view angle / occlusion, add its contact band as extra seeds.
                if salvage_supported_instances:
                    # Determine a "near" depth from the main seeds.
                    z_ref = float(np.median(z_seed)) if z_seed.size > 0 else float(np.nanmedian(depth_z[valid_grid]))
                    z_salvage_max = z_ref + float(salvage_depth_margin_m)
                    inst_ids = np.unique(inst_grid[(inst_grid >= 0) & valid_grid])
                    for iid in inst_ids.tolist():
                        iid = int(iid)
                        m = (inst_grid == iid) & valid_grid
                        if int(np.count_nonzero(m)) < int(salvage_min_grid_area):
                            continue
                        # Must be near-ish
                        z_m = depth_z[m]
                        z_m = z_m[np.isfinite(z_m)]
                        if z_m.size == 0 or float(np.median(z_m)) > float(z_salvage_max):
                            continue
                        # Must have some pixels close to the support plane.
                        sd = support_dist[m]
                        sd = sd[np.isfinite(sd)]
                        if sd.size == 0:
                            continue
                        if float(np.percentile(sd, 10.0)) > float(salvage_contact_band_m):
                            continue
                        seeds_extra |= (m & (support_dist <= float(salvage_contact_band_m)))

                    if np.any(seeds_extra):
                        Image.fromarray((seeds_extra.astype(np.uint8) * 255), mode="L").save(out_dir / "mask_seeds_extra.png")

            # Prevent growth across occlusion/depth discontinuities.
            height_grid = height.reshape(gh, gw)
            grown = _bfs_connected_mask_barrier(
                seeds=(seeds_main | seeds_extra),
                allowed=allowed,
                height=height_grid,
                depth_z=depth_z,
                max_height_step_m=float(neighbor_max_height_step_m),
                max_depth_step_m=float(neighbor_max_depth_step_m),
                p_grid=p_grid,
                max_3d_step_m=float(neighbor_max_3d_step_m),
            )

            # After removing bridge triangles, we also drop any floating grown regions
            # that are not connected to the support plane (floor/table) via seeds.
            if drop_floating:
                grown_supported = _bfs_connected_mask_barrier(
                    seeds=(seeds | seeds_extra),
                    allowed=(grown & allowed),
                    height=height_grid,
                    depth_z=depth_z,
                    max_height_step_m=float(neighbor_max_height_step_m),
                    max_depth_step_m=float(neighbor_max_depth_step_m),
                    p_grid=p_grid,
                    max_3d_step_m=float(neighbor_max_3d_step_m),
                )
                grown = grown_supported

            grown_flat = grown.reshape(-1)

            # Save for debugging
            dbg = (grown.astype(np.uint8) * 255)
            Image.fromarray(dbg, mode="L").save(out_dir / "mask_grown.png")

            # Anything not in grown gets flattened.
            height[~grown_flat] = 0.0

            # Support vertices for mesh-level component filtering later.
            # Note: grown is on the grid; vertices are flattened row-major in the same order.
            support_vert_mask_flat = (seeds_main | seeds_extra).reshape(-1)

        p_proj = p - dist.reshape(-1, 1).astype(np.float32) * n.reshape(1, 3)
        u_axis, v_axis = _plane_basis(n)
        u_coord = (p_proj @ u_axis.reshape(3, 1)).reshape(-1).astype(np.float32)
        v_coord = (p_proj @ v_axis.reshape(3, 1)).reshape(-1).astype(np.float32)

        vertices = np.stack([u_coord, v_coord, height], axis=-1).astype(np.float32)

        # Drop triangles that span a large height discontinuity (reduces "stringing").
        if tri_max_height_step_m > 0:
            keep = []
            hvals = height
            max_step = float(tri_max_height_step_m)
            for tri in faces:
                a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                ha, hb, hc = hvals[a], hvals[b], hvals[c]
                if not (np.isfinite(ha) and np.isfinite(hb) and np.isfinite(hc)):
                    continue
                if max(ha, hb, hc) - min(ha, hb, hc) > max_step:
                    continue
                keep.append((a, b, c))
            faces = np.array(keep, dtype=np.int32)

        # Drop triangles that span a large 3D edge length in camera space.
        # This directly removes sparse "bridge" triangles that cause stringing to background.
        if tri_max_edge_m > 0:
            keep = []
            p_flat = p.astype(np.float32)
            max_e = float(tri_max_edge_m)
            for tri in faces:
                a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                pa, pb, pc = p_flat[a], p_flat[b], p_flat[c]
                if not (np.isfinite(pa).all() and np.isfinite(pb).all() and np.isfinite(pc).all()):
                    continue
                dab = float(np.linalg.norm(pa - pb))
                dbc = float(np.linalg.norm(pb - pc))
                dca = float(np.linalg.norm(pc - pa))
                if max(dab, dbc, dca) > max_e:
                    continue
                keep.append((a, b, c))
            faces = np.array(keep, dtype=np.int32)

        # Finally: after bridge-triangle culling, remove any disconnected mesh components
        # that do not touch the support plane seeds. This drops dense-but-far background slabs.
        if drop_floating and support_vert_mask_flat is not None and faces.size > 0:
            faces = _keep_face_components_touching_support(
                faces=faces,
                n_vertices=int(vertices.shape[0]),
                support_vertices=support_vert_mask_flat.astype(np.bool_),
            )

        # Center for nicer viewing.
        center = np.nanmedian(vertices, axis=0)
        vertices = (vertices - center).astype(np.float32)

        # Save plane metadata
        (out_dir / "plane_fit.json").write_text(
            __import__("json").dumps(
                {
                    "mode": "plane",
                    "normal_camera": [float(x) for x in n.tolist()],
                    "d": float(d),
                    "fit_bottom_frac": float(fit_bottom_frac),
                    "distance_thresh_m": float(plane_distance_thresh_m),
                    "plane_ransac_iters": int(plane_ransac_iters),
                    "max_height_m": float(max_height_m),
                    "tri_max_height_step_m": float(tri_max_height_step_m),
                    "suppress_mode": str(suppress_mode),
                    "suppress_masks": [str(p) for p in (suppress_masks or [])],
                    "growth_filter": {
                        "enabled": bool(enable_growth_filter),
                        "keep_normal_dot": float(keep_normal_dot),
                        "seed_normal_dot": float(seed_normal_dot),
                        "seed_band_m": float(seed_band_m),
                        "neighbor_max_height_step_m": float(neighbor_max_height_step_m),
                        "neighbor_max_depth_step_m": float(neighbor_max_depth_step_m),
                        "neighbor_max_3d_step_m": float(neighbor_max_3d_step_m),
                        "support_depth_pctl": float(support_depth_pctl),
                        "support_depth_margin_m": float(support_depth_margin_m),
                        "seed_depth_pctl": float(seed_depth_pctl),
                        "seed_depth_margin_m": float(seed_depth_margin_m),
                        "keep_nearest_seed_component": bool(keep_nearest_seed_component),
                        "suppress_vertical_planes": bool(suppress_vertical_planes),
                        "vertical_plane_max_dot": float(vertical_plane_max_dot),
                        "vertical_plane_min_inliers": int(vertical_plane_min_inliers),
                        "vertical_plane_max_planes": int(vertical_plane_max_planes),
                        "sam2_instances": bool(sam2_instances),
                        "sam2_model_id": str(sam2_model_id),
                        "sam2_hard_filter": bool(sam2_hard_filter),
                        "salvage_supported_instances": bool(salvage_supported_instances),
                        "salvage_contact_band_m": float(salvage_contact_band_m),
                        "salvage_min_grid_area": int(salvage_min_grid_area),
                        "salvage_depth_margin_m": float(salvage_depth_margin_m),
                        "detect_table_plane": bool(detect_table_plane),
                        "table_seed_normal_dot": float(table_seed_normal_dot),
                        "table_min_height_m": float(table_min_height_m),
                        "table_max_height_m": float(table_max_height_m),
                        "table_min_inliers": int(table_min_inliers),
                        "tri_max_edge_m": float(tri_max_edge_m),
                        "drop_floating": bool(drop_floating),
                        "table_plane": (
                            {
                                "normal_camera": [float(x) for x in table_plane[0].tolist()],
                                "d": float(table_plane[1]),
                            }
                            if table_plane is not None
                            else None
                        ),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        raise ValueError("mode must be 'camera', 'relief', or 'plane'")

    # Write material + mesh
    material_name = "mat0"
    obj_path = out_dir / "mesh.obj"
    mtl_path = out_dir / "mesh.mtl"

    # Save a copy of the texture next to OBJ for convenience
    tex_path = out_dir / "albedo.png"
    Image.fromarray(scene_img, mode="RGB").save(tex_path)

    vertices, uv, faces = _compact_mesh(vertices=vertices, uvs=uv, faces=faces)

    _write_mtl(mtl_path, material_name=material_name, diffuse_tex_rel=tex_path.name)
    _write_obj(
        obj_path,
        mtl_name=mtl_path.name,
        material_name=material_name,
        vertices=vertices,
        uvs=uv,
        faces=faces,
    )

    return MeshOutputs(out_dir=out_dir, obj_path=obj_path, mtl_path=mtl_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Step2 (fast): DepthPro-only -> displacement -> build an image-sized grid mesh and export OBJ.\n"
            "This disables segmentation/inpaint for speed (mask=none, inpaint=none)."
        )
    )

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="Path to SpatialCapture_* folder")
    src.add_argument("--image", type=Path, help="Path to a single RGB image (jpg/png) for one-off meshing")

    ap.add_argument("--frame-index", type=int, default=0, help="Only used with --input")
    ap.add_argument("--out", type=Path, default=None)

    ap.add_argument("--image-max-size", type=int, default=1024, help="Downscale before meshing to keep OBJ manageable")
    ap.add_argument("--grid-step", type=int, default=2, help="Sample every N pixels for mesh vertices (higher = fewer verts)")

    ap.add_argument(
        "--mode",
        choices=["camera", "plane", "relief"],
        default="camera",
        help="'camera' exports a real-scale camera-space surface. 'plane' grows geometry from a fitted ground/table plane. 'relief' is a simple plane displacement.",
    )

    ap.add_argument("--depth-device", default="auto", help="Only used with --image (DepthPro device: auto/cpu/mps/cuda)")
    ap.add_argument(
        "--depth-smooth",
        choices=["none", "median", "bilateral"],
        default="none",
        help="Only used with --image: smooth the predicted metric depth to reduce floor/plane pitting (recommended: median)",
    )
    ap.add_argument(
        "--depth-smooth-ksize",
        type=int,
        default=5,
        help="Only used with --image: smoothing kernel size (odd). Larger = smoother but more detail loss.",
    )
    ap.add_argument(
        "--depth-bilateral-sigma-space",
        type=float,
        default=7.0,
        help="Only used with --image + --depth-smooth bilateral: sigmaSpace in pixels (edge-preserving).",
    )
    ap.add_argument("--focal-px", type=float, default=None, help="Only used with --image: optional focal length in pixels")
    ap.add_argument("--fx", type=float, default=None, help="Only used with --image: override fx")
    ap.add_argument("--fy", type=float, default=None, help="Only used with --image: override fy")
    ap.add_argument("--cx", type=float, default=None, help="Only used with --image: override cx")
    ap.add_argument("--cy", type=float, default=None, help="Only used with --image: override cy")

    ap.add_argument("--plane-width", type=float, default=2.0)
    ap.add_argument("--plane-height", type=float, default=2.0)
    ap.add_argument("--relief-strength-m", type=float, default=0.15, help="Relief mode displacement strength in world units")

    ap.add_argument("--fit-bottom-frac", type=float, default=0.60, help="Plane mode: fit plane using the bottom fraction of the image")
    ap.add_argument("--plane-distance-thresh-m", type=float, default=0.02, help="Plane mode: RANSAC inlier threshold in meters")
    ap.add_argument("--plane-ransac-iters", type=int, default=2000, help="Plane mode: RANSAC iterations")
    ap.add_argument("--max-height-m", type=float, default=1.50, help="Plane mode: clamp growth height in meters")
    ap.add_argument("--tri-max-height-step-m", type=float, default=0.25, help="Plane mode: drop triangles spanning bigger height jumps (reduces stringing)")

    ap.add_argument(
        "--suppress-mask",
        action="append",
        default=None,
        help="Plane mode: path to a mask PNG to suppress (flatten) region. Can be passed multiple times (e.g. wall + ceiling masks).",
    )
    ap.add_argument(
        "--suppress-mode",
        choices=["zero", "nan"],
        default="zero",
        help="How to apply suppression masks: 'zero' flattens to plane; 'nan' removes geometry (may create holes).",
    )

    ap.add_argument(
        "--growth-filter",
        action="store_true",
        help=(
            "Plane mode: enable geometry-based filtering (keep only regions that are supported by floor/table and "
            "connected without leaking across big depth discontinuities). Writes debug masks into the output folder."
        ),
    )
    ap.add_argument(
        "--keep-normal-dot",
        type=float,
        default=0.0,
        help=(
            "Plane mode + growth-filter: optional surface normal dot threshold (relative to support plane normal). "
            "0 disables the normal filter (recommended if you want to keep vertical sides of objects)."
        ),
    )
    ap.add_argument(
        "--seed-normal-dot",
        type=float,
        default=0.85,
        help="Plane mode + growth-filter: used only for tabletop plane detection candidates (horizontal-ish surfaces)",
    )
    ap.add_argument("--seed-band-m", type=float, default=0.03, help="Plane mode + growth-filter: distance-to-support band for seeding")

    ap.add_argument(
        "--neighbor-max-height-step-m",
        type=float,
        default=0.05,
        help="Plane mode + growth-filter: BFS barrier; disallow growth across height jumps larger than this (meters)",
    )
    ap.add_argument(
        "--neighbor-max-depth-step-m",
        type=float,
        default=0.25,
        help="Plane mode + growth-filter: BFS barrier; disallow growth across depth jumps larger than this (meters)",
    )
    ap.add_argument(
        "--neighbor-max-3d-step-m",
        type=float,
        default=0.12,
        help="Plane mode + growth-filter: density barrier; disallow growth across 3D neighbor gaps larger than this (meters)",
    )
    ap.add_argument(
        "--support-depth-pctl",
        type=float,
        default=92.0,
        help="Plane mode + growth-filter: compute a max allowed depth from support seeds (percentile, 50..100). Set 0 to disable.",
    )
    ap.add_argument(
        "--support-depth-margin-m",
        type=float,
        default=0.5,
        help="Plane mode + growth-filter: extra depth margin beyond the seed depth percentile (meters)",
    )
    ap.add_argument(
        "--seed-depth-pctl",
        type=float,
        default=75.0,
        help="Plane mode + growth-filter: filter support seeds by depth percentile (50..100). Lower removes far-away floor seeds. 0 disables.",
    )
    ap.add_argument(
        "--seed-depth-margin-m",
        type=float,
        default=0.25,
        help="Plane mode + growth-filter: extra depth margin beyond seed-depth percentile (meters)",
    )
    ap.add_argument(
        "--keep-nearest-seed-component",
        action="store_true",
        help="Plane mode + growth-filter: keep only the nearest connected component of support seeds (removes disconnected background islands).",
    )

    ap.add_argument(
        "--suppress-vertical-planes",
        action="store_true",
        help="Plane mode + growth-filter: detect dominant vertical-ish planes (walls) and suppress them geometrically.",
    )
    ap.add_argument(
        "--vertical-plane-max-dot",
        type=float,
        default=0.25,
        help="Plane mode + suppress-vertical-planes: abs(dot(plane_normal, support_normal)) must be <= this to count as vertical",
    )
    ap.add_argument(
        "--vertical-plane-min-inliers",
        type=int,
        default=12000,
        help="Plane mode + suppress-vertical-planes: minimum inliers to accept a dominant wall plane",
    )
    ap.add_argument(
        "--vertical-plane-max-planes",
        type=int,
        default=2,
        help="Plane mode + suppress-vertical-planes: try to suppress up to this many vertical planes",
    )

    ap.add_argument(
        "--sam2-instances",
        action="store_true",
        help=(
            "Plane mode + growth-filter: use SAM2 automatic instance masks (no text prompt) to keep supported/stacked objects "
            "and suppress unrelated background. Writes mask_sam2_keep.png + sam2_instances.json."
        ),
    )
    ap.add_argument("--sam2-model-id", type=str, default="facebook/sam2.1-hiera-large")
    ap.add_argument("--sam2-points-per-side", type=int, default=32)
    ap.add_argument("--sam2-pred-iou-thresh", type=float, default=0.85)
    ap.add_argument("--sam2-stability-thresh", type=float, default=0.95)
    ap.add_argument("--sam2-min-mask-region-area", type=int, default=0)
    ap.add_argument("--sam2-min-area", type=int, default=800)
    ap.add_argument(
        "--sam2-max-area-frac",
        type=float,
        default=0.35,
        help="Drop extremely large masks (e.g., whole wall/ceiling) larger than this fraction of the image.",
    )
    ap.add_argument(
        "--stack-contact-dist-m",
        type=float,
        default=0.05,
        help="SAM2 stacking: 3D distance threshold to consider two instance boundaries in contact.",
    )
    ap.add_argument(
        "--stack-min-height-delta-m",
        type=float,
        default=0.03,
        help="SAM2 stacking: instance must be this much higher (median height) to be considered stacked above a supported one.",
    )
    ap.add_argument(
        "--sam2-hard-filter",
        action="store_true",
        help="SAM2: if set, restrict growth strictly inside supported/stacked SAM2 instances (can drop objects SAM misses).",
    )
    ap.add_argument(
        "--salvage-supported-instances",
        action="store_true",
        help="SAM2: salvage nearby supported instances even if disconnected by adding their support-contact band as extra seeds.",
    )
    ap.add_argument(
        "--salvage-contact-band-m",
        type=float,
        default=0.04,
        help="SAM2 salvage: how close an instance must get to support plane to be considered supported (meters)",
    )
    ap.add_argument(
        "--salvage-min-grid-area",
        type=int,
        default=30,
        help="SAM2 salvage: minimum instance area in grid-cells to salvage (filters tiny noise instances)",
    )
    ap.add_argument(
        "--salvage-depth-margin-m",
        type=float,
        default=0.8,
        help="SAM2 salvage: allow salvaged instances up to (median seed depth + margin) in meters",
    )

    ap.add_argument(
        "--detect-table-plane",
        action="store_true",
        help="Plane mode + growth-filter: try to detect a secondary horizontal support plane (e.g., tabletop) so objects on it are kept.",
    )
    ap.add_argument(
        "--table-seed-normal-dot",
        type=float,
        default=0.4,
        help="Table plane detection: optional normal dot threshold. Set 0 to ignore normals (more robust under noisy depth).",
    )
    ap.add_argument("--table-min-height-m", type=float, default=0.35)
    ap.add_argument("--table-max-height-m", type=float, default=1.60)
    ap.add_argument("--table-min-inliers", type=int, default=1200)

    ap.add_argument(
        "--tri-max-edge-m",
        type=float,
        default=0.25,
        help="Plane mode: drop triangles whose camera-space edges exceed this length (removes low-density stringing)",
    )

    ap.add_argument(
        "--drop-floating",
        action="store_true",
        help="Plane mode + growth-filter: drop any grown regions not connected to the support plane via seeds.",
    )

    args = ap.parse_args()

    common_kwargs = dict(
        out_dir=args.out,
        image_max_size=int(args.image_max_size),
        grid_step=int(args.grid_step),
        mode=str(args.mode),
        plane_width=float(args.plane_width),
        plane_height=float(args.plane_height),
        relief_strength_m=float(args.relief_strength_m),
        fit_bottom_frac=float(args.fit_bottom_frac),
        plane_distance_thresh_m=float(args.plane_distance_thresh_m),
        plane_ransac_iters=int(args.plane_ransac_iters),
        max_height_m=float(args.max_height_m),
        tri_max_height_step_m=float(args.tri_max_height_step_m),
        suppress_masks=[Path(p) for p in (args.suppress_mask or [])],
        suppress_mode=str(args.suppress_mode),
        enable_growth_filter=bool(args.growth_filter),
        keep_normal_dot=float(args.keep_normal_dot),
        seed_normal_dot=float(args.seed_normal_dot),
        seed_band_m=float(args.seed_band_m),
        neighbor_max_height_step_m=float(args.neighbor_max_height_step_m),
        neighbor_max_depth_step_m=float(args.neighbor_max_depth_step_m),
        neighbor_max_3d_step_m=float(args.neighbor_max_3d_step_m),
        support_depth_pctl=float(args.support_depth_pctl),
        support_depth_margin_m=float(args.support_depth_margin_m),
        seed_depth_pctl=float(args.seed_depth_pctl),
        seed_depth_margin_m=float(args.seed_depth_margin_m),
        keep_nearest_seed_component=bool(args.keep_nearest_seed_component),
        suppress_vertical_planes=bool(args.suppress_vertical_planes),
        vertical_plane_max_dot=float(args.vertical_plane_max_dot),
        vertical_plane_min_inliers=int(args.vertical_plane_min_inliers),
        vertical_plane_max_planes=int(args.vertical_plane_max_planes),
        sam2_instances=bool(args.sam2_instances),
        sam2_model_id=str(args.sam2_model_id),
        sam2_points_per_side=int(args.sam2_points_per_side),
        sam2_pred_iou_thresh=float(args.sam2_pred_iou_thresh),
        sam2_stability_thresh=float(args.sam2_stability_thresh),
        sam2_min_mask_region_area=int(args.sam2_min_mask_region_area),
        sam2_min_area=int(args.sam2_min_area),
        sam2_max_area_frac=float(args.sam2_max_area_frac),
        stack_contact_dist_m=float(args.stack_contact_dist_m),
        stack_min_height_delta_m=float(args.stack_min_height_delta_m),
        sam2_hard_filter=bool(args.sam2_hard_filter),
        salvage_supported_instances=bool(args.salvage_supported_instances),
        salvage_contact_band_m=float(args.salvage_contact_band_m),
        salvage_min_grid_area=int(args.salvage_min_grid_area),
        salvage_depth_margin_m=float(args.salvage_depth_margin_m),
        detect_table_plane=bool(args.detect_table_plane),
        table_seed_normal_dot=float(args.table_seed_normal_dot),
        table_min_height_m=float(args.table_min_height_m),
        table_max_height_m=float(args.table_max_height_m),
        table_min_inliers=int(args.table_min_inliers),
        tri_max_edge_m=float(args.tri_max_edge_m),
        drop_floating=bool(args.drop_floating),
    )

    if args.input is not None:
        out = run(
            capture_dir=args.input,
            frame_index=int(args.frame_index),
            **common_kwargs,
        )
    else:
        out = run_image(
            image_path=args.image,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            focal_px=args.focal_px,
            depth_device=str(args.depth_device),
            depth_smooth=str(args.depth_smooth),
            depth_smooth_ksize=int(args.depth_smooth_ksize),
            depth_bilateral_sigma_space=float(args.depth_bilateral_sigma_space),
            **common_kwargs,
        )

    print("Wrote:")
    print(f"- {out.obj_path}")
    print(f"- {out.mtl_path}")


if __name__ == "__main__":
    main()
