from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

import step1_3d_photo


def _load_rgb_u8(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.array(im, dtype=np.uint8)


def _to_gray_f32(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32) / 255.0
    # Rec.709 luma
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


@dataclass(frozen=True)
class DisplacementOutputs:
    out_dir: Path
    scene_image_path: Path
    person_mask_path: Path
    depth_npy_path: Path
    depth_refined_npy_path: Path
    displacement_u16_path: Path
    displacement_preview_path: Path
    normal_map_path: Path
    meta_path: Path


def _save_u16_png(path: Path, img_u16: np.ndarray) -> None:
    if img_u16.dtype != np.uint16:
        raise ValueError("img_u16 must be uint16")
    if img_u16.ndim != 2:
        raise ValueError("img_u16 must be (H,W)")
    # Let Pillow infer the correct 16-bit mode from dtype.
    Image.fromarray(img_u16).save(path)


def _save_u8_png(path: Path, img_u8: np.ndarray) -> None:
    Image.fromarray(img_u8).save(path)


def _gaussian_blur_f32(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    try:
        import cv2  # type: ignore

        # ksize=0 lets OpenCV derive it from sigma
        return cv2.GaussianBlur(img.astype(np.float32), (0, 0), float(sigma))
    except Exception:
        # Very small fallback: separable blur via PIL at lower quality.
        pil = Image.fromarray(img.astype(np.float32))
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
        return np.array(pil, dtype=np.float32)


def _box_filter_f32(img: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return img.astype(np.float32)
    try:
        import cv2  # type: ignore

        k = 2 * int(radius) + 1
        return cv2.boxFilter(img.astype(np.float32), ddepth=-1, ksize=(k, k), normalize=True)
    except Exception:
        # Fallback: approximate with Gaussian
        return _gaussian_blur_f32(img.astype(np.float32), sigma=float(max(radius, 1)))


def _guided_filter_gray(guide_gray01: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Edge-preserving guided filter for a single-channel guide + single-channel src.

    This is a numpy/cv2 implementation of the classic guided filter (He et al.).
    guide_gray01: (H,W) float32 in [0,1]
    src:          (H,W) float32
    """
    I = guide_gray01.astype(np.float32)
    p = src.astype(np.float32)

    mean_I = _box_filter_f32(I, radius)
    mean_p = _box_filter_f32(p, radius)
    mean_Ip = _box_filter_f32(I * p, radius)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = _box_filter_f32(I * I, radius)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + float(eps))
    b = mean_p - a * mean_I

    mean_a = _box_filter_f32(a, radius)
    mean_b = _box_filter_f32(b, radius)

    q = mean_a * I + mean_b
    return q.astype(np.float32)


def _bilateral_filter_f32(img: np.ndarray, diameter: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    if diameter <= 0:
        return img
    try:
        import cv2  # type: ignore

        return cv2.bilateralFilter(
            img.astype(np.float32),
            int(diameter),
            float(sigma_color),
            float(sigma_space),
        ).astype(np.float32)
    except Exception:
        return img


def _refine_depth(
    depth_m: np.ndarray,
    guide_rgb_u8: Optional[np.ndarray],
    detail_sigma: float,
    far_detail_suppression: float,
    guided_radius: int,
    guided_eps: float,
    bilateral_d: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
) -> np.ndarray:
    """Make depth more stable for displacement.

    1) Work in disparity to reduce far noise.
    2) Suppress high-frequency detail as distance increases.
    3) Optional bilateral smoothing in disparity.
    """
    depth = depth_m.astype(np.float32)
    depth = np.nan_to_num(depth, nan=np.nan, posinf=np.nan, neginf=np.nan)

    valid = np.isfinite(depth)
    if not np.any(valid):
        raise RuntimeError("Depth contains no finite values")

    z = depth.copy()
    z[~valid] = np.nan

    z_min = float(np.nanpercentile(z, 1))
    z_max = float(np.nanpercentile(z, 99))
    z = np.clip(z, z_min, z_max)

    # Disparity (larger when closer)
    disp = 1.0 / np.maximum(z, 1e-3)

    # Edge-aware base layer
    disp_base = disp
    if guide_rgb_u8 is not None and guided_radius > 0:
        gray = _to_gray_f32(guide_rgb_u8)
        disp_base = _guided_filter_gray(gray, disp, radius=int(guided_radius), eps=float(guided_eps))
    elif detail_sigma > 0:
        disp_base = _gaussian_blur_f32(disp, sigma=float(detail_sigma))

    detail = disp - disp_base

    # Suppress detail more as z increases (farther -> less high-frequency)
    if far_detail_suppression > 0:
        t = (z - z_min) / max(z_max - z_min, 1e-6)
        w = np.exp(-float(far_detail_suppression) * t).astype(np.float32)
        disp = disp_base + detail * w
    else:
        disp = disp_base + detail

    disp = _bilateral_filter_f32(
        disp,
        diameter=int(bilateral_d),
        sigma_color=float(bilateral_sigma_color),
        sigma_space=float(bilateral_sigma_space),
    )

    # Back to depth
    z_ref = 1.0 / np.maximum(disp, 1e-6)
    z_ref[~valid] = np.nan
    return z_ref.astype(np.float32)


def _displacement_from_depth(
    depth_m: np.ndarray,
    mode: str,
    clip_percentile: Tuple[float, float],
    invert: bool,
) -> np.ndarray:
    """Return displacement in [0,1] float32.

    mode:
      - 'disparity': uses 1/z (better for far noise)
      - 'depth': uses z directly
    """
    z = depth_m.astype(np.float32)
    z = np.nan_to_num(z, nan=np.nan, posinf=np.nan, neginf=np.nan)
    valid = np.isfinite(z)
    if not np.any(valid):
        raise RuntimeError("Depth contains no finite values")

    if mode == "disparity":
        s = 1.0 / np.maximum(z, 1e-3)
    elif mode == "depth":
        s = z.copy()
    else:
        raise ValueError("mode must be 'disparity' or 'depth'")

    lo_p, hi_p = clip_percentile
    lo = float(np.nanpercentile(s, lo_p))
    hi = float(np.nanpercentile(s, hi_p))
    s = np.clip(s, lo, hi)

    out = (s - lo) / max(hi - lo, 1e-6)
    out[~valid] = 0.0

    if invert:
        out = 1.0 - out

    return out.astype(np.float32)


def _normal_map_from_height(height01: np.ndarray, strength: float) -> np.ndarray:
    """Generate a tangent-space normal map (RGB uint8) from a height map in [0,1]."""
    h = height01.astype(np.float32)

    # Sobel-like finite differences (simple, stable)
    dx = np.zeros_like(h)
    dy = np.zeros_like(h)

    dx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) * 0.5
    dy[1:-1, :] = (h[2:, :] - h[:-2, :]) * 0.5

    sx = float(strength)
    sy = float(strength)

    nx = -dx * sx
    ny = -dy * sy
    nz = np.ones_like(h, dtype=np.float32)

    n = np.stack([nx, ny, nz], axis=-1)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(norm, 1e-6)

    # Map from [-1,1] to [0,255]
    rgb = (n * 0.5 + 0.5) * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _normal_map_from_depth(
    depth_m: np.ndarray,
    k: np.ndarray,
    strength: float,
    y_flip: bool,
) -> np.ndarray:
    """Compute a view-space normal map from metric depth + intrinsics.

    This tends to look less "flat" than height-gradient normals because perspective is respected.
    """
    z = depth_m.astype(np.float32)
    valid = np.isfinite(z)
    if not np.any(valid):
        raise RuntimeError("Depth contains no finite values")

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    h, w = z.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    x = (uu - cx) * z / max(fx, 1e-6)
    y = (vv - cy) * z / max(fy, 1e-6)
    p = np.stack([x, y, z], axis=-1).astype(np.float32)

    # Central differences in 3D
    dpdx = np.zeros_like(p)
    dpdy = np.zeros_like(p)
    dpdx[:, 1:-1, :] = (p[:, 2:, :] - p[:, :-2, :]) * 0.5
    dpdy[1:-1, :, :] = (p[2:, :, :] - p[:-2, :, :]) * 0.5

    # Cross product gives surface normal
    n = np.cross(dpdx, dpdy)
    n_norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(n_norm, 1e-6)

    # Ensure nz points "out" consistently (positive Z)
    flip = n[..., 2] < 0
    n[flip] *= -1.0

    # Apply user strength as additional tilt scaling
    # (scale x/y components, then renormalize)
    s = float(strength)
    n[..., 0] *= s
    n[..., 1] *= s
    n_norm2 = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(n_norm2, 1e-6)

    if y_flip:
        n[..., 1] *= -1.0

    n[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    rgb = (n * 0.5 + 0.5) * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def run(
    capture_dir: Path,
    frame_index: int,
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
    out_dir: Optional[Path],
    displacement_mode: str,
    invert: bool,
    clip_lo: float,
    clip_hi: float,
    normal_strength: float,
    normal_method: str,
    normal_y_flip: bool,
    detail_sigma: float,
    far_detail_suppression: float,
    guided_radius: int,
    guided_eps: float,
    bilateral_d: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
) -> DisplacementOutputs:
    if out_dir is None:
        out_dir = capture_dir / f"displacement_{int(frame_index):06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse step1 for: (mask -> inpaint -> depth). Skip PLY.
    step1_out = step1_3d_photo.run_step1(
        capture_dir=capture_dir,
        frame_index=frame_index,
        mask_method=mask_method,
        mask_dilate=mask_dilate,
        mask_prompt=mask_prompt,
        langsam_sam_type=langsam_sam_type,
        langsam_box_threshold=langsam_box_threshold,
        langsam_text_threshold=langsam_text_threshold,
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
        out_dir=out_dir,
        max_points=1,
        drop_person_points=False,
        export_ply=False,
    )

    guide_rgb = _load_rgb_u8(step1_out.scene_image_path)

    depth = np.load(step1_out.depth_npy_path).astype(np.float32)
    depth_ref = _refine_depth(
        depth,
        guide_rgb_u8=guide_rgb,
        detail_sigma=detail_sigma,
        far_detail_suppression=far_detail_suppression,
        guided_radius=guided_radius,
        guided_eps=guided_eps,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
    )

    depth_ref_path = out_dir / "depth_refined.npy"
    np.save(depth_ref_path, depth_ref)

    disp01 = _displacement_from_depth(
        depth_ref,
        mode=displacement_mode,
        clip_percentile=(clip_lo, clip_hi),
        invert=bool(invert),
    )

    disp_u16 = np.clip(disp01 * 65535.0, 0, 65535).astype(np.uint16)
    disp_u16_path = out_dir / "displacement_u16.png"
    _save_u16_png(disp_u16_path, disp_u16)

    disp_preview = (np.clip(disp01, 0.0, 1.0) * 255.0).astype(np.uint8)
    disp_preview_path = out_dir / "displacement_preview.png"
    _save_u8_png(disp_preview_path, disp_preview)

    if normal_method == "height":
        normal_rgb = _normal_map_from_height(disp01, strength=float(normal_strength))
    elif normal_method == "depth":
        frames = list(step1_3d_photo.read_frames_jsonl(capture_dir))
        if not frames:
            raise RuntimeError(f"No frames in {capture_dir}")
        idx = int(frame_index)
        if idx < 0:
            idx = len(frames) + idx
        if idx < 0 or idx >= len(frames):
            raise IndexError(f"frame_index {frame_index} out of range (0..{len(frames)-1})")
        rec = frames[idx]
        k, w_saved, h_saved = step1_3d_photo.get_intrinsics_for_saved_image(rec, capture_dir)

        # Match intrinsics to the raster we used (inpaint may resize)
        h, w = guide_rgb.shape[:2]
        if (w_saved, h_saved) != (w, h):
            scale_x = float(w) / float(w_saved)
            scale_y = float(h) / float(h_saved)
            k = step1_3d_photo._scale_intrinsics(k, scale_x=scale_x, scale_y=scale_y)  # type: ignore[attr-defined]

        normal_rgb = _normal_map_from_depth(
            depth_ref,
            k=k,
            strength=float(normal_strength),
            y_flip=bool(normal_y_flip),
        )
    else:
        raise ValueError("normal_method must be 'height' or 'depth'")
    normal_path = out_dir / "normal_map.png"
    _save_u8_png(normal_path, normal_rgb)

    meta = {
        "capture_dir": str(capture_dir),
        "frame_index": int(frame_index),
        "mask_method": mask_method,
        "mask_dilate": int(mask_dilate),
        "inpaint_method": inpaint_method,
        "depth_method": depth_method,
        "depth_model": depth_model,
        "depth_device": depth_device,
        "min_depth": float(min_depth),
        "max_depth": float(max_depth),
        "refine": {
            "detail_sigma": float(detail_sigma),
            "far_detail_suppression": float(far_detail_suppression),
            "guided_radius": int(guided_radius),
            "guided_eps": float(guided_eps),
            "bilateral_d": int(bilateral_d),
            "bilateral_sigma_color": float(bilateral_sigma_color),
            "bilateral_sigma_space": float(bilateral_sigma_space),
        },
        "displacement": {
            "mode": displacement_mode,
            "invert": bool(invert),
            "clip_percentile": [float(clip_lo), float(clip_hi)],
        },
        "normal": {"strength": float(normal_strength), "method": normal_method, "y_flip": bool(normal_y_flip)},
        "outputs": {
            "scene_image": str(step1_out.scene_image_path.name),
            "person_mask": str(step1_out.person_mask_path.name),
            "depth": str(step1_out.depth_npy_path.name),
            "depth_refined": str(depth_ref_path.name),
            "displacement_u16": str(disp_u16_path.name),
            "displacement_preview": str(disp_preview_path.name),
            "normal_map": str(normal_path.name),
        },
    }

    meta_path = out_dir / "displacement_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return DisplacementOutputs(
        out_dir=out_dir,
        scene_image_path=step1_out.scene_image_path,
        person_mask_path=step1_out.person_mask_path,
        depth_npy_path=step1_out.depth_npy_path,
        depth_refined_npy_path=depth_ref_path,
        displacement_u16_path=disp_u16_path,
        displacement_preview_path=disp_preview_path,
        normal_map_path=normal_path,
        meta_path=meta_path,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Single-image displacement pipeline: mask(optional) -> inpaint -> DepthPro/MiDaS -> "
            "depth refinement -> displacement(16-bit) + normal map.\n"
            "Designed for Blender/Unity-style plane displacement rather than noisy point clouds."
        )
    )
    ap.add_argument("--input", required=True, type=Path, help="Path to SpatialCapture_* folder")
    ap.add_argument("--frame-index", type=int, default=0)

    ap.add_argument(
        "--mask",
        choices=["auto", "langsam", "rembg", "centerbox", "none"],
        default="auto",
        help="Masking strategy. 'auto' prefers LangSAM segmentation then falls back to rembg then centerbox.",
    )
    ap.add_argument("--mask-dilate", type=int, default=4)
    ap.add_argument("--mask-prompt", default="person.")
    ap.add_argument("--langsam-sam-type", default="sam2.1_hiera_large")
    ap.add_argument("--langsam-box-threshold", type=float, default=0.15)
    ap.add_argument("--langsam-text-threshold", type=float, default=0.15)
    ap.add_argument("--mask-shape", choices=["original", "rectangle", "square"], default="original")

    ap.add_argument("--inpaint", choices=["opencv", "diffusers", "none"], default="opencv")
    ap.add_argument("--inpaint-radius", type=int, default=3)
    ap.add_argument("--inpaint-device", default="auto")
    ap.add_argument("--inpaint-model", default="runwayml/stable-diffusion-inpainting")
    ap.add_argument("--inpaint-steps", type=int, default=25)
    ap.add_argument("--inpaint-guidance", type=float, default=7.5)
    ap.add_argument("--inpaint-seed", type=int, default=None)

    ap.add_argument("--depth", choices=["midas", "depthpro"], default="depthpro")
    ap.add_argument("--depth-model", default="MiDaS_small")
    ap.add_argument("--depth-device", default="auto")
    ap.add_argument("--min-depth", type=float, default=0.5)
    ap.add_argument("--max-depth", type=float, default=10.0)

    ap.add_argument("--out", type=Path, default=None)

    ap.add_argument("--disp-mode", choices=["disparity", "depth"], default="disparity")
    ap.add_argument("--invert", action="store_true", help="Invert displacement (useful for 'dent' look)")
    ap.add_argument("--clip-lo", type=float, default=1.0, help="Percentile low clip for displacement")
    ap.add_argument("--clip-hi", type=float, default=99.0, help="Percentile high clip for displacement")

    ap.add_argument("--normal-strength", type=float, default=1.0)
    ap.add_argument(
        "--normal-method",
        choices=["height", "depth"],
        default="depth",
        help="Normal generation: 'depth' uses DepthPro depth + intrinsics (more realistic); 'height' uses displacement gradient.",
    )
    ap.add_argument(
        "--normal-y-flip",
        action="store_true",
        help="Flip Y (green) channel for normal map conventions (useful for Unity/DirectX pipelines).",
    )

    ap.add_argument("--detail-sigma", type=float, default=3.0, help="Gaussian sigma for separating low/high freq in disparity")
    ap.add_argument(
        "--far-detail-suppression",
        type=float,
        default=3.0,
        help="Larger suppresses high-frequency detail more strongly for far depth",
    )
    ap.add_argument("--bilateral-d", type=int, default=9)
    ap.add_argument("--bilateral-sigma-color", type=float, default=0.05)
    ap.add_argument("--bilateral-sigma-space", type=float, default=9.0)

    ap.add_argument(
        "--guided-radius",
        type=int,
        default=16,
        help="Guided filter radius for edge-aware disparity smoothing (uses inpainted RGB as guidance).",
    )
    ap.add_argument(
        "--guided-eps",
        type=float,
        default=1e-3,
        help="Guided filter epsilon (larger = smoother, less edge-following).",
    )

    args = ap.parse_args()

    outputs = run(
        capture_dir=args.input,
        frame_index=args.frame_index,
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
        out_dir=args.out,
        displacement_mode=args.disp_mode,
        invert=bool(args.invert),
        clip_lo=float(args.clip_lo),
        clip_hi=float(args.clip_hi),
        normal_strength=float(args.normal_strength),
        normal_method=str(args.normal_method),
        normal_y_flip=bool(args.normal_y_flip),
        detail_sigma=float(args.detail_sigma),
        far_detail_suppression=float(args.far_detail_suppression),
        guided_radius=int(args.guided_radius),
        guided_eps=float(args.guided_eps),
        bilateral_d=int(args.bilateral_d),
        bilateral_sigma_color=float(args.bilateral_sigma_color),
        bilateral_sigma_space=float(args.bilateral_sigma_space),
    )

    print("Wrote:")
    print(f"- {outputs.scene_image_path}")
    print(f"- {outputs.person_mask_path}")
    print(f"- {outputs.depth_npy_path}")
    print(f"- {outputs.depth_refined_npy_path}")
    print(f"- {outputs.displacement_u16_path}")
    print(f"- {outputs.displacement_preview_path}")
    print(f"- {outputs.normal_map_path}")
    print(f"- {outputs.meta_path}")


if __name__ == "__main__":
    main()
