from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

from spatial_capture import arkit_cw_to_open3d_cw, read_frames_jsonl


@dataclass(frozen=True)
class AnchorEstimate:
    prompt: str
    world_xyz_median: list[float]
    world_xyz_mean: list[float]
    n_obs: int


def _load_u8_image(path: Path) -> np.ndarray:
    img = np.asarray(o3d.io.read_image(str(path)))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return img.astype(np.uint8)


def _load_u8_mask(path: Path) -> np.ndarray:
    m = np.asarray(o3d.io.read_image(str(path)))
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(np.uint8)


def _mask_from_langsam(image_rgb_u8: np.ndarray, prompt: str, box_threshold: float, text_threshold: float) -> np.ndarray:
    try:
        from lang_sam import LangSAM
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "LangSAM not available. Install optional ML deps (see README) or use --use-person-mask-only."
        ) from e

    model = LangSAM()

    # LangSAM expects PIL image
    from PIL import Image

    pil = Image.fromarray(image_rgb_u8)
    masks, _, _, _ = model.predict(
        pil,
        prompt,
        box_threshold=float(box_threshold),
        text_threshold=float(text_threshold),
    )

    if masks is None:
        return np.zeros((image_rgb_u8.shape[0], image_rgb_u8.shape[1]), dtype=np.uint8)

    masks_np = np.asarray(masks)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]

    # masks may be float probabilities or bool
    if masks_np.dtype != np.bool_:
        masks_bin = masks_np > 0.5
    else:
        masks_bin = masks_np

    union = np.any(masks_bin, axis=0)
    return (union.astype(np.uint8) * 255)


def _point_from_mask(
    depth_m: np.ndarray,
    mask_u8: np.ndarray,
    depth_quantile: float,
) -> Optional[tuple[int, int, float]]:
    ys, xs = np.nonzero(mask_u8 > 0)
    if ys.size == 0:
        return None

    d = depth_m[ys, xs].astype(np.float32)
    good = np.isfinite(d) & (d > 1e-4)
    if not np.any(good):
        return None

    xs = xs[good]
    ys = ys[good]
    d = d[good]

    q = float(depth_quantile)
    q = float(np.clip(q, 0.0, 1.0))
    # Bias toward the closest surface inside the mask (helps avoid "background bleeding"
    # when depth is imperfect on large masks).
    d_thr = float(np.quantile(d, q))
    keep = d <= d_thr
    if not np.any(keep):
        keep = np.ones_like(d, dtype=bool)

    xs_k = xs[keep]
    ys_k = ys[keep]
    d_k = d[keep]

    x_med = int(np.median(xs_k))
    y_med = int(np.median(ys_k))
    d_med = float(np.median(d_k))
    return x_med, y_med, d_med


def _robust_center(points: np.ndarray, mad_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (center, inlier_mask) using median + MAD distance filtering."""
    if points.shape[0] == 0:
        return np.zeros((3,), dtype=np.float32), np.zeros((0,), dtype=bool)
    if points.shape[0] <= 2:
        return np.median(points, axis=0), np.ones((points.shape[0],), dtype=bool)

    med = np.median(points, axis=0)
    dist = np.linalg.norm(points - med[None, :], axis=1)
    mad = np.median(np.abs(dist - np.median(dist)))
    if not np.isfinite(mad) or mad < 1e-6:
        return med, np.ones((points.shape[0],), dtype=bool)

    # Keep within mad_sigma * 1.4826 * MAD (approx std).
    thr = float(mad_sigma) * 1.4826 * float(mad)
    inl = dist <= thr
    # Be conservative: if we drop almost everything, keep all.
    if inl.sum() < max(2, points.shape[0] // 3):
        inl[:] = True
    return med, inl


def _backproject_pixel(u: float, v: float, depth: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return np.array([x, y, z, 1.0], dtype=np.float32)


def estimate_object_anchors(
    capture_dir: Path,
    fuse_dir: Optional[Path],
    prompts: list[str],
    use_person_mask_only: bool,
    box_threshold: float,
    text_threshold: float,
    depth_quantile: float,
    reject_mad_sigma: float,
    out_path: Optional[Path],
) -> dict:
    if fuse_dir is None:
        fuse_dir = capture_dir / "fuse"

    frames = list(read_frames_jsonl(capture_dir))
    frames_by_idx = {}
    for f in frames:
        try:
            idx = int(Path(f.image_file).stem.split("_")[-1])
        except Exception:
            continue
        frames_by_idx[idx] = f

    fuse_meta_path = fuse_dir / "fuse_meta.json"
    used = None
    if fuse_meta_path.exists():
        fuse_meta = json.loads(fuse_meta_path.read_text(encoding="utf-8"))
        used = [int(x) for x in fuse_meta.get("used_frame_indices", [])]

    if not used:
        # Fall back to all available step1 dirs
        used = sorted(
            int(p.name.split("_")[-1])
            for p in fuse_dir.glob("step1_*")
            if p.is_dir() and p.name.split("_")[-1].isdigit()
        )

    if not used:
        raise RuntimeError(f"No frames to process in {fuse_dir}. Run fuse_keyframes.py first.")

    out = {
        "capture_dir": str(capture_dir),
        "fuse_dir": str(fuse_dir),
        "used_frame_indices": used,
        "anchors": [],
    }

    for prompt in prompts:
        world_pts = []
        per_frame = []

        for idx in used:
            step_dir = fuse_dir / f"step1_{idx:06d}"
            meta_path = step_dir / "step1_meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            w, h = int(meta["saved_image_wh"][0]), int(meta["saved_image_wh"][1])
            k = np.array(meta["intrinsics_saved"], dtype=np.float32)
            fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])

            depth_path = step_dir / "depth_pred.npy"
            if not depth_path.exists():
                continue
            depth = np.load(str(depth_path)).astype(np.float32)
            if depth.shape != (h, w):
                depth = np.asarray(
                    o3d.t.geometry.Image(depth).resize(w, h, o3d.t.geometry.InterpType.Linear)
                ).astype(np.float32)

            mask = None
            if prompt.strip().lower() in ("person", "person."):
                pm = step_dir / "person_mask.png"
                if pm.exists():
                    mask = _load_u8_mask(pm)
            elif not use_person_mask_only:
                img_path = step_dir / "scene_hole.png"
                if not img_path.exists():
                    img_path = step_dir / "scene_image.png"
                if img_path.exists():
                    img = _load_u8_image(img_path)
                    mask = _mask_from_langsam(img, prompt=prompt, box_threshold=box_threshold, text_threshold=text_threshold)

            if mask is None:
                continue
            if mask.shape != (h, w):
                mask = np.asarray(
                    o3d.t.geometry.Image(mask).resize(w, h, o3d.t.geometry.InterpType.Nearest)
                ).astype(np.uint8)

            mpt = _point_from_mask(depth, mask, depth_quantile=depth_quantile)
            if mpt is None:
                continue
            u, v, d = mpt

            fr = frames_by_idx.get(idx)
            if fr is None:
                continue
            cw = arkit_cw_to_open3d_cw(fr.camera_transform_cw_4x4)

            p_cam = _backproject_pixel(float(u), float(v), float(d), fx=fx, fy=fy, cx=cx, cy=cy)
            p_w = (cw @ p_cam)[:3]

            world_pts.append(p_w)
            per_frame.append(
                {"frame_index": idx, "u": int(u), "v": int(v), "depth": float(d), "world_xyz": p_w.astype(float).tolist()}
            )

        if not world_pts:
            out["anchors"].append({"prompt": prompt, "n_obs": 0, "per_frame": per_frame})
            continue

        pts = np.stack(world_pts, axis=0)
        med, inl = _robust_center(pts, mad_sigma=float(reject_mad_sigma))
        pts_inl = pts[inl]
        mean = pts_inl.mean(axis=0)

        out["anchors"].append(
            {
                "prompt": prompt,
                "n_obs": int(pts.shape[0]),
                "n_inliers": int(pts_inl.shape[0]),
                "world_xyz_mean": mean.astype(float).tolist(),
                "world_xyz_median": med.astype(float).tolist(),
                "per_frame": per_frame,
            }
        )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Estimate world-space anchor positions for objects using per-frame depth_pred.npy + poses. "
            "This is an MVP building block for a hybrid 'skybox + true 3D objects' world." 
        )
    )
    ap.add_argument("--input", required=True, type=Path, help="Path to SpatialCapture_* folder")
    ap.add_argument("--fuse-dir", type=Path, default=None, help="Default: <capture>/fuse (expects step1_*/ outputs)")
    ap.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Object prompt(s) for LangSAM. Can be repeated. Special-case: 'person' uses person_mask.png.",
    )
    ap.add_argument(
        "--use-person-mask-only",
        action="store_true",
        help="Do not run LangSAM; only estimate anchor for prompt 'person' from person_mask.png.",
    )
    ap.add_argument("--langsam-box-threshold", type=float, default=0.15)
    ap.add_argument("--langsam-text-threshold", type=float, default=0.15)
    ap.add_argument(
        "--depth-quantile",
        type=float,
        default=0.2,
        help="Pick a representative depth from the closest q-quantile inside the mask (0.2 biases toward foreground)",
    )
    ap.add_argument(
        "--reject-mad-sigma",
        type=float,
        default=3.0,
        help="Robust outlier rejection threshold in MAD-sigma units (higher keeps more points)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON (default: <fuse>/anchors.json)",
    )

    args = ap.parse_args()
    fuse_dir = args.fuse_dir
    if fuse_dir is None:
        fuse_dir = args.input / "fuse"

    out_path = args.out
    if out_path is None:
        out_path = fuse_dir / "anchors.json"

    prompts = args.prompt
    if not prompts:
        prompts = ["person"]

    estimate_object_anchors(
        capture_dir=args.input,
        fuse_dir=fuse_dir,
        prompts=prompts,
        use_person_mask_only=bool(args.use_person_mask_only),
        box_threshold=float(args.langsam_box_threshold),
        text_threshold=float(args.langsam_text_threshold),
        depth_quantile=float(args.depth_quantile),
        reject_mad_sigma=float(args.reject_mad_sigma),
        out_path=out_path,
    )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
