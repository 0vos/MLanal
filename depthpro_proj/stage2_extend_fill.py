from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

from spatial_capture import arkit_cw_to_open3d_cw


@dataclass(frozen=True)
class Stage2Outputs:
    cleaned_pcd_path: Path
    labeled_pcd_path: Path
    extend_only_path: Path
    fill_only_path: Path
    meta_path: Path


def _load_u8_image(path: Path) -> np.ndarray:
    img = np.asarray(o3d.io.read_image(str(path)))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return img.astype(np.uint8)


def _load_u8_mask(path: Path) -> np.ndarray:
    m = _load_u8_image(path)
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(np.uint8)


def _resize_nearest_u8(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[0] == h and img.shape[1] == w:
        return img
    t = o3d.t.geometry.Image(img).resize(int(w), int(h), o3d.t.geometry.InterpType.Nearest)
    return np.asarray(t).astype(img.dtype)


def _resize_linear_f32(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[0] == h and img.shape[1] == w:
        return img
    t = o3d.t.geometry.Image(img.astype(np.float32)).resize(int(w), int(h), o3d.t.geometry.InterpType.Linear)
    return np.asarray(t).astype(np.float32)


def _scale_intrinsics_if_needed(k: np.ndarray, meta_w: int, meta_h: int, w: int, h: int) -> tuple[float, float, float, float]:
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    if meta_w > 0 and meta_h > 0 and (w != meta_w or h != meta_h):
        sx = float(w) / float(meta_w)
        sy = float(h) / float(meta_h)
        fx *= sx
        cx *= sx
        fy *= sy
        cy *= sy
    return fx, fy, cx, cy


def clean_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    stat_nb_neighbors: int,
    stat_std_ratio: float,
    radius_nb_points: int,
    radius: float,
    dbscan_eps: float,
    dbscan_min_points: int,
    min_cluster_size: int,
) -> o3d.geometry.PointCloud:
    def apply_step(name: str, before: o3d.geometry.PointCloud, fn) -> o3d.geometry.PointCloud:
        if before.is_empty():
            return before
        after = fn(before)
        if after.is_empty():
            print(f"[stage2][clean] step '{name}' removed all points; skipping.")
            return before
        # If a step removes *too much*, treat it as mis-tuned for this dataset and skip.
        before_n = len(before.points)
        after_n = len(after.points)
        min_keep = max(500, int(0.01 * before_n))
        if after_n < min_keep:
            print(
                f"[stage2][clean] step '{name}' kept {after_n}/{before_n} points (<{min_keep}); skipping."
            )
            return before
        return after

    out = pcd

    if voxel_size and voxel_size > 0:
        out = apply_step("voxel_down_sample", out, lambda x: x.voxel_down_sample(voxel_size=float(voxel_size)))

    if stat_nb_neighbors > 0 and stat_std_ratio > 0:
        out = apply_step(
            "statistical_outlier",
            out,
            lambda x: x.remove_statistical_outlier(nb_neighbors=int(stat_nb_neighbors), std_ratio=float(stat_std_ratio))[0],
        )

    if radius_nb_points > 0 and radius > 0:
        out = apply_step(
            "radius_outlier",
            out,
            lambda x: x.remove_radius_outlier(nb_points=int(radius_nb_points), radius=float(radius))[0],
        )

    if dbscan_eps > 0 and dbscan_min_points > 0 and min_cluster_size > 0:
        def _dbscan_prune(x: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
            labels = np.array(
                x.cluster_dbscan(eps=float(dbscan_eps), min_points=int(dbscan_min_points), print_progress=False)
            )
            if labels.size == 0 or labels.max() < 0:
                return x

            kept = np.zeros_like(labels, dtype=bool)
            for lab in range(int(labels.max()) + 1):
                idxs = np.where(labels == lab)[0]
                if idxs.size >= int(min_cluster_size):
                    kept[idxs] = True
            if not np.any(kept):
                return o3d.geometry.PointCloud()
            return x.select_by_index(np.where(kept)[0])

        out = apply_step("dbscan_prune_small_clusters", out, _dbscan_prune)

    return out


def label_extend_fill(
    points_world: np.ndarray,
    fuse_dir: Path,
    fuse_meta: dict,
    depth_consistency_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Label each world point as extend(1) vs fill(0) using multi-frame depth+mask consistency.

    A point is considered "observed" in a frame if:
    - it projects into the frame bounds
    - it is NOT inside that frame's person_mask (mask==0)
    - predicted depth at (u,v) is finite and |depth - z_cam| < threshold

    extend: observed in >=2 frames
    fill: otherwise

    Returns (label_u8, support_count_u8)
    """
    n = points_world.shape[0]
    pts_h = np.concatenate([points_world.astype(np.float32), np.ones((n, 1), dtype=np.float32)], axis=1)

    pose_map = {
        int(p["frame_index"]): np.array(p["cameraTransform_cw_4x4"], dtype=np.float32).reshape((4, 4), order="F")
        for p in fuse_meta.get("poses", [])
    }

    used = [int(i) for i in fuse_meta.get("used_frame_indices", [])]
    if not used:
        used = sorted(pose_map.keys())

    support = np.zeros((n,), dtype=np.uint8)

    # Heuristic: depending on how the upstream TSDF integration was configured,
    # some fused outputs behave as if the stored pose should be used as world->camera.
    # We'll auto-pick the projection mode per-frame (cw direct vs inverse(cw)) based
    # on which yields more in-bounds, in-front projections.
    def _project(points_h: np.ndarray, cw_open3d: np.ndarray, fx: float, fy: float, cx: float, cy: float, w: int, h: int):
        def _count_in_front_in_bounds(pc: np.ndarray):
            z = pc[:, 2]
            valid = z > 1e-6
            x = pc[:, 0]
            y = pc[:, 1]
            u = fx * (x / z) + cx
            v = fy * (y / z) + cy
            ui = u.astype(np.int32)
            vi = v.astype(np.int32)
            inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            valid &= inb
            return valid, ui, vi, z

        # Mode A: treat cw as camera->world (so use wc to get camera coords)
        wc = np.linalg.inv(cw_open3d).astype(np.float32)
        pc_a = (wc @ points_h.T).T
        valid_a, ui_a, vi_a, z_a = _count_in_front_in_bounds(pc_a)
        count_a = int(valid_a.sum())

        # Mode B: treat cw as world->camera (use cw directly)
        pc_b = (cw_open3d @ points_h.T).T
        valid_b, ui_b, vi_b, z_b = _count_in_front_in_bounds(pc_b)
        count_b = int(valid_b.sum())

        if count_b > count_a:
            return valid_b, ui_b, vi_b, z_b, "cw_direct"
        return valid_a, ui_a, vi_a, z_a, "inverse_cw"

    # Use a subset to decide modes fast.
    if n > 50000:
        sample_idx = np.linspace(0, n - 1, num=50000).astype(np.int64)
        pts_h_sample = pts_h[sample_idx]
    else:
        sample_idx = None
        pts_h_sample = pts_h

    for idx in used:
        per = fuse_dir / f"step1_{idx:06d}"
        meta_path = per / "step1_meta.json"
        if not meta_path.exists():
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_w, meta_h = int(meta["saved_image_wh"][0]), int(meta["saved_image_wh"][1])
        k = np.array(meta["intrinsics_saved"], dtype=np.float32)

        img_path = per / "scene_image.png"
        depth_path = per / "depth_pred.npy"
        mask_path = per / "person_mask.png"
        if not img_path.exists() or not depth_path.exists() or not mask_path.exists():
            continue

        img = _load_u8_image(img_path)
        h, w = int(img.shape[0]), int(img.shape[1])

        depth = np.load(str(depth_path)).astype(np.float32)
        depth = _resize_linear_f32(depth, w=w, h=h)

        mask = _load_u8_mask(mask_path)
        mask = _resize_nearest_u8(mask, w=w, h=h)

        fx, fy, cx, cy = _scale_intrinsics_if_needed(k, meta_w=meta_w, meta_h=meta_h, w=w, h=h)

        cw_arkit = pose_map.get(idx)
        if cw_arkit is None:
            continue
        cw = arkit_cw_to_open3d_cw(cw_arkit)

        valid_s, ui_s, vi_s, z_s, mode = _project(pts_h_sample, cw, fx, fy, cx, cy, w, h)
        # Re-project full set with the chosen mode.
        if mode == "cw_direct":
            pc = (cw.astype(np.float32) @ pts_h.T).T
        else:
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
        inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        valid &= inb

        idxs = np.nonzero(valid)[0]
        if idxs.size == 0:
            continue

        # Skip pixels that were originally the person in this frame.
        not_person = mask[vi[idxs], ui[idxs]] == 0
        if not np.any(not_person):
            continue
        idxs = idxs[not_person]
        if idxs.size == 0:
            continue

        d = depth[vi[idxs], ui[idxs]]
        err = np.abs(d - z[idxs])
        ok = np.isfinite(d) & np.isfinite(err) & (err < float(depth_consistency_m))
        if not np.any(ok):
            continue

        support[idxs[ok]] = np.minimum(support[idxs[ok]] + 1, 255).astype(np.uint8)

    label = (support >= 2).astype(np.uint8)  # 1=extend, 0=fill
    return label, support


def tint_colors(base_rgb_u8: np.ndarray, label_u8: np.ndarray) -> np.ndarray:
    """Tint colors for visualization: extend keeps color, fill becomes reddish and darker."""
    c = base_rgb_u8.astype(np.float32)
    out = c.copy()

    fill = label_u8 == 0
    if np.any(fill):
        # darken + red tint
        out[fill] *= 0.45
        out[fill, 0] = np.clip(out[fill, 0] + 90.0, 0, 255)

    return out.clip(0, 255).astype(np.uint8)


def run_stage2(
    capture_dir: Path,
    fuse_dir: Optional[Path],
    input_pcd: Path,
    out_dir: Optional[Path],
    depth_consistency_m: float,
    # noise removal
    clean_voxel: float,
    stat_nb_neighbors: int,
    stat_std_ratio: float,
    radius_nb_points: int,
    radius: float,
    dbscan_eps: float,
    dbscan_min_points: int,
    min_cluster_size: int,
) -> Stage2Outputs:
    if fuse_dir is None:
        fuse_dir = capture_dir / "fuse"
    if out_dir is None:
        out_dir = fuse_dir / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)

    fuse_meta_path = fuse_dir / "fuse_meta.json"
    if not fuse_meta_path.exists():
        raise FileNotFoundError(f"Missing {fuse_meta_path}. Run fuse_keyframes.py first.")
    fuse_meta = json.loads(fuse_meta_path.read_text(encoding="utf-8"))

    pcd = o3d.io.read_point_cloud(str(input_pcd))
    if pcd.is_empty():
        raise RuntimeError(f"Empty input point cloud: {input_pcd}")

    pcd_clean = clean_point_cloud(
        pcd,
        voxel_size=clean_voxel,
        stat_nb_neighbors=stat_nb_neighbors,
        stat_std_ratio=stat_std_ratio,
        radius_nb_points=radius_nb_points,
        radius=radius,
        dbscan_eps=dbscan_eps,
        dbscan_min_points=dbscan_min_points,
        min_cluster_size=min_cluster_size,
    )

    if pcd_clean.is_empty():
        # Shouldn't happen due to fail-safe cleaning, but be defensive.
        print("[stage2][clean] cleaning produced empty cloud; falling back to original input.")
        pcd_clean = pcd

    pts = np.asarray(pcd_clean.points)
    base_cols = None
    if pcd_clean.has_colors() and len(pcd_clean.colors) == len(pcd_clean.points):
        base_cols = (np.asarray(pcd_clean.colors) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        base_cols = np.full((pts.shape[0], 3), 200, dtype=np.uint8)

    label_u8, support_u8 = label_extend_fill(
        pts,
        fuse_dir=fuse_dir,
        fuse_meta=fuse_meta,
        depth_consistency_m=depth_consistency_m,
    )

    # Outputs
    cleaned_path = out_dir / "stage2_points_clean.ply"
    labeled_path = out_dir / "stage2_points_labeled.ply"
    extend_path = out_dir / "stage2_extend_only.ply"
    fill_path = out_dir / "stage2_fill_only.ply"
    meta_path = out_dir / "stage2_meta.json"

    o3d.io.write_point_cloud(str(cleaned_path), pcd_clean)

    tinted = tint_colors(base_cols, label_u8)
    pcd_labeled = o3d.geometry.PointCloud()
    pcd_labeled.points = o3d.utility.Vector3dVector(pts)
    pcd_labeled.colors = o3d.utility.Vector3dVector((tinted.astype(np.float32) / 255.0).clip(0, 1))
    o3d.io.write_point_cloud(str(labeled_path), pcd_labeled)

    extend_idxs = np.where(label_u8 == 1)[0]
    fill_idxs = np.where(label_u8 == 0)[0]

    if extend_idxs.size:
        o3d.io.write_point_cloud(str(extend_path), pcd_labeled.select_by_index(extend_idxs))
    else:
        print("[stage2] extend-only is empty; skipping write")

    if fill_idxs.size:
        o3d.io.write_point_cloud(str(fill_path), pcd_labeled.select_by_index(fill_idxs))
    else:
        print("[stage2] fill-only is empty; skipping write")

    meta = {
        "capture_dir": str(capture_dir),
        "fuse_dir": str(fuse_dir),
        "input_pcd": str(input_pcd),
        "depth_consistency_m": float(depth_consistency_m),
        "cleaning": {
            "clean_voxel": float(clean_voxel),
            "stat_nb_neighbors": int(stat_nb_neighbors),
            "stat_std_ratio": float(stat_std_ratio),
            "radius_nb_points": int(radius_nb_points),
            "radius": float(radius),
            "dbscan_eps": float(dbscan_eps),
            "dbscan_min_points": int(dbscan_min_points),
            "min_cluster_size": int(min_cluster_size),
        },
        "counts": {
            "points_in": int(len(pcd.points)),
            "points_clean": int(len(pcd_clean.points)),
            "extend": int(extend_idxs.size),
            "fill": int(fill_idxs.size),
            "support_mean": float(support_u8.astype(np.float32).mean()) if support_u8.size else 0.0,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return Stage2Outputs(
        cleaned_pcd_path=cleaned_path,
        labeled_pcd_path=labeled_path,
        extend_only_path=extend_path,
        fill_only_path=fill_path,
        meta_path=meta_path,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage2 (baseline): clean fused point cloud noise and label extend vs fill using multi-frame depth consistency."
    )
    ap.add_argument("--input", required=True, type=Path, help="Path to SpatialCapture_* folder")
    ap.add_argument("--fuse-dir", type=Path, default=None, help="Fusion dir containing fuse_meta.json and step1_*/ (default: <capture>/fuse)")
    ap.add_argument(
        "--pcd",
        type=Path,
        default=None,
        help="Input point cloud to label (default: <fuse>/fused_tsdf_mesh_3frames_recolor_fix_pcd.ply if exists)",
    )
    ap.add_argument("--outdir", type=Path, default=None, help="Output directory (default: <fuse>/stage2)")

    ap.add_argument("--depth-consistency", type=float, default=0.20, help="Meters: depth consistency threshold")

    # Noise removal knobs
    ap.add_argument("--clean-voxel", type=float, default=0.0, help="Optional voxel downsample before cleaning (meters)")
    ap.add_argument("--stat-nn", type=int, default=30, help="Statistical outlier: nb_neighbors")
    ap.add_argument("--stat-std", type=float, default=2.0, help="Statistical outlier: std_ratio")
    ap.add_argument("--radius-n", type=int, default=8, help="Radius outlier: min neighbors")
    ap.add_argument("--radius", type=float, default=0.06, help="Radius outlier: radius (meters)")
    ap.add_argument("--dbscan-eps", type=float, default=0.06, help="DBSCAN eps (meters)")
    ap.add_argument("--dbscan-min", type=int, default=30, help="DBSCAN min_points")
    ap.add_argument("--min-cluster", type=int, default=200, help="Drop clusters smaller than this")

    args = ap.parse_args()

    fuse_dir = args.fuse_dir
    if fuse_dir is None:
        fuse_dir = args.input / "fuse"

    pcd = args.pcd
    if pcd is None:
        # try common outputs
        cand = fuse_dir / "fused_tsdf_mesh_3frames_recolor_fix_pcd.ply"
        if cand.exists():
            pcd = cand
        else:
            # last resort
            cand2 = fuse_dir / "fused_tsdf_mesh_3frames_pcd.ply"
            pcd = cand2

    outputs = run_stage2(
        capture_dir=args.input,
        fuse_dir=fuse_dir,
        input_pcd=pcd,
        out_dir=args.outdir,
        depth_consistency_m=args.depth_consistency,
        clean_voxel=args.clean_voxel,
        stat_nb_neighbors=args.stat_nn,
        stat_std_ratio=args.stat_std,
        radius_nb_points=args.radius_n,
        radius=args.radius,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_points=args.dbscan_min,
        min_cluster_size=args.min_cluster,
    )

    print("Wrote:")
    print(f"- {outputs.cleaned_pcd_path}")
    print(f"- {outputs.labeled_pcd_path}")
    print(f"- {outputs.extend_only_path}")
    print(f"- {outputs.fill_only_path}")
    print(f"- {outputs.meta_path}")


if __name__ == "__main__":
    main()
