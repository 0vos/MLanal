from __future__ import annotations

import argparse
import json
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Literal

import numpy as np
from PIL import Image

from spatial_capture import read_frames_jsonl, get_intrinsics_for_saved_image, arkit_cw_to_open3d_cw


@dataclass(frozen=True)
class Step1Outputs:
    scene_image_path: Path
    person_mask_path: Path
    depth_npy_path: Path
    ply_path: Optional[Path]


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.array(im, dtype=np.uint8)
    return arr


def _save_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def _dilate_mask(mask_u8: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask_u8

    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "OpenCV is required for mask dilation. Install: pip install opencv-python"
        ) from e

    kernel = np.ones((9, 9), np.uint8)
    return cv2.dilate(mask_u8, kernel, iterations=int(iterations))


def _mask_to_rectangle_mask(mask_u8: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask_u8 > 0)
    if coords.size == 0:
        return np.zeros_like(mask_u8, dtype=np.uint8)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    rect = np.zeros_like(mask_u8, dtype=np.uint8)
    rect[int(y_min) : int(y_max) + 1, int(x_min) : int(x_max) + 1] = 255
    return rect


def _mask_to_square_mask(mask_u8: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask_u8 > 0)
    if coords.size == 0:
        return np.zeros_like(mask_u8, dtype=np.uint8)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    height = int(y_max - y_min + 1)
    width = int(x_max - x_min + 1)
    side = int(max(height, width))
    cy = int((y_min + y_max) // 2)
    cx = int((x_min + x_max) // 2)
    half = side // 2

    y1 = max(cy - half, 0)
    x1 = max(cx - half, 0)
    y2 = min(y1 + side, mask_u8.shape[0])
    x2 = min(x1 + side, mask_u8.shape[1])

    square = np.zeros_like(mask_u8, dtype=np.uint8)
    square[y1:y2, x1:x2] = 255
    return square


def _person_mask_rembg(rgb: np.ndarray) -> np.ndarray:
    """Returns mask (H,W) uint8 0/255 where 255 means person/foreground."""
    try:
        from rembg import remove  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "rembg is not installed. Install optional deps: pip install -r requirements-ml.txt"
        ) from e

    # rembg returns an RGBA image when given bytes or PIL
    pil = Image.fromarray(rgb, mode="RGB")
    out = remove(pil)
    if isinstance(out, Image.Image):
        rgba = out
    elif isinstance(out, (bytes, bytearray)):
        rgba = Image.open(io.BytesIO(out))
    else:
        # Some rembg versions may return a file-like object.
        rgba = Image.open(out)  # type: ignore[arg-type]

    rgba = rgba.convert("RGBA")
    alpha = np.array(rgba, dtype=np.uint8)[..., 3]
    mask = (alpha > 128).astype(np.uint8) * 255
    return mask


_LANGSAM_MODEL = None
_LANGSAM_SAM_TYPE: Optional[str] = None


def _person_mask_langsam(
    rgb: np.ndarray,
    text_prompt: str,
    sam_type: str,
    box_threshold: float,
    text_threshold: float,
    mask_shape: Literal["original", "rectangle", "square"],
    dilate_iterations: int,
) -> np.ndarray:
    """PhySIC-style GSAM mask via LangSAM.

    Returns uint8 mask (H,W) in {0,255} where 255 denotes the person.
    """
    global _LANGSAM_MODEL, _LANGSAM_SAM_TYPE

    try:
        from lang_sam import LangSAM  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "lang_sam is not installed (or failed to import).\n"
            "Install it from Git (and its SAM2 dependency) with:\n"
            "  pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git@918043ed4666eea04da88aa179eb8d27ef4b1a1d\n"
            "If you cannot reach GitHub reliably, use --mask rembg or --mask centerbox instead."
        ) from e

    if text_prompt and not text_prompt.endswith("."):
        text_prompt = text_prompt + "."

    if _LANGSAM_MODEL is None or _LANGSAM_SAM_TYPE != sam_type:
        _LANGSAM_MODEL = LangSAM(sam_type=sam_type)
        _LANGSAM_SAM_TYPE = sam_type

    image = Image.fromarray(rgb, mode="RGB")
    results = _LANGSAM_MODEL.predict(
        [image],
        [text_prompt],
        box_threshold=float(box_threshold),
        text_threshold=float(text_threshold),
    )
    if not results:
        return np.zeros(rgb.shape[:2], dtype=np.uint8)

    result = results[0]
    masks = result.get("masks")
    if masks is None:
        return np.zeros(rgb.shape[:2], dtype=np.uint8)

    def _to_bool_mask(m) -> Optional[np.ndarray]:
        if m is None:
            return None
        try:
            import torch  # type: ignore

            if isinstance(m, torch.Tensor):
                m = m.detach().cpu().numpy()
        except Exception:
            pass

        arr = np.asarray(m)
        if arr.size == 0:
            return None

        # Common shapes:
        # - (H,W)
        # - (N,H,W)
        # - occasionally with singleton dims that can be squeezed
        if arr.ndim >= 4:
            arr = np.squeeze(arr)

        if arr.ndim == 3:
            if arr.dtype == np.bool_:
                return np.any(arr, axis=0)
            if np.issubdtype(arr.dtype, np.floating):
                return np.any(arr > 0.5, axis=0)
            return np.any(arr > 0, axis=0)

        if arr.ndim != 2:
            return None

        if arr.dtype == np.bool_:
            return arr
        if np.issubdtype(arr.dtype, np.floating):
            # LangSAM may output per-pixel probabilities in [0,1]
            return arr > 0.5
        return arr > 0

    union: Optional[np.ndarray] = None
    if isinstance(masks, (list, tuple)):
        for m in masks:
            bm = _to_bool_mask(m)
            if bm is None:
                continue
            union = bm if union is None else (union | bm)
    else:
        bm = _to_bool_mask(masks)
        if bm is not None:
            union = bm

    if union is None:
        return np.zeros(rgb.shape[:2], dtype=np.uint8)

    mask_u8 = union.astype(np.uint8) * 255
    mask_u8 = _dilate_mask(mask_u8, dilate_iterations)

    if mask_shape == "rectangle":
        mask_u8 = _mask_to_rectangle_mask(mask_u8)
    elif mask_shape == "square":
        mask_u8 = _mask_to_square_mask(mask_u8)

    return mask_u8


def _person_mask_center_box(rgb: np.ndarray, frac_w: float = 0.45, frac_h: float = 0.7) -> np.ndarray:
    h, w = rgb.shape[:2]
    bw = int(w * frac_w)
    bh = int(h * frac_h)
    x0 = max((w - bw) // 2, 0)
    y0 = max((h - bh) // 2, 0)
    x1 = min(x0 + bw, w)
    y1 = min(y0 + bh, h)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def _inpaint_opencv(rgb: np.ndarray, mask_u8: np.ndarray, radius: int) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "OpenCV is required for inpainting. Install: pip install opencv-python"
        ) from e

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, mask_u8, float(radius), cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


_DIFFUSERS_PIPE = None
_DIFFUSERS_MODEL_ID: Optional[str] = None


def _inpaint_diffusers(
    rgb: np.ndarray,
    mask_u8: np.ndarray,
    model_id: str,
    device: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: Optional[int],
) -> np.ndarray:
    """Optional AI inpainting via Diffusers.

    Notes:
    - Some inpainting models require accepting a license on HuggingFace.
    - For private/gated models, set `HUGGINGFACE_HUB_TOKEN`.
    """
    try:
        import torch
        from diffusers import AutoPipelineForInpainting  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "diffusers/torch is not installed. Install optional deps: pip install -r requirements-ml.txt"
        ) from e

    if device == "auto":
        if torch.backends.mps.is_available():
            device_t = torch.device("mps")
        elif torch.cuda.is_available():
            device_t = torch.device("cuda")
        else:
            device_t = torch.device("cpu")
    else:
        device_t = torch.device(device)

    global _DIFFUSERS_PIPE, _DIFFUSERS_MODEL_ID
    if _DIFFUSERS_PIPE is None or _DIFFUSERS_MODEL_ID != model_id:
        dtype = torch.float16 if device_t.type in ("cuda", "mps") else torch.float32
        _DIFFUSERS_PIPE = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=dtype)
        _DIFFUSERS_PIPE = _DIFFUSERS_PIPE.to(device_t)
        _DIFFUSERS_MODEL_ID = model_id

    image = Image.fromarray(rgb, mode="RGB")
    mask = Image.fromarray(mask_u8, mode="L")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device_t).manual_seed(int(seed))

    out = _DIFFUSERS_PIPE(
        prompt="",
        image=image,
        mask_image=mask,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    )

    result = out.images[0]
    return np.array(result.convert("RGB"), dtype=np.uint8)


def _apply_hole(rgb: np.ndarray, mask_u8: np.ndarray, fill_rgb: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Return an RGB image with the masked region replaced by a solid color."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be (H,W,3)")
    if mask_u8.shape[:2] != rgb.shape[:2]:
        raise ValueError("mask_u8 must match rgb spatial dimensions")

    out = rgb.copy()
    hole = mask_u8 > 0
    out[hole] = np.array(fill_rgb, dtype=np.uint8)
    return out


def _predict_depth_midas(rgb: np.ndarray, device: str, model_type: str, min_depth: float, max_depth: float) -> np.ndarray:
    """Predict a pseudo-metric depth map in meters-like scale.

    MiDaS outputs relative depth (typically closer -> larger values). We convert to a bounded Z by
    normalizing, inverting, and scaling into [min_depth, max_depth]. This is NOT metric.
    """
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for depth prediction. Install optional deps: pip install -r requirements-ml.txt"
        ) from e

    if device == "auto":
        if torch.backends.mps.is_available():
            device_t = torch.device("mps")
        elif torch.cuda.is_available():
            device_t = torch.device("cuda")
        else:
            device_t = torch.device("cpu")
    else:
        device_t = torch.device(device)

    # torch.hub.load() may hit GitHub API rate limits during repo validation.
    # Worse: some torch versions throw KeyError('Authorization') on that error path.
    # MiDaS internally calls torch.hub again (e.g. EfficientNet backbones), so we
    # temporarily disable the validation hook globally for the duration of loading.
    validate_fn = getattr(torch.hub, "_validate_not_a_forked_repo", None)
    if validate_fn is not None:
        torch.hub._validate_not_a_forked_repo = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    try:
        try:
            midas = torch.hub.load(
                "intel-isl/MiDaS",
                model_type,
                trust_repo=True,
                skip_validation=True,
            )
        except TypeError:
            midas = torch.hub.load("intel-isl/MiDaS", model_type)
    finally:
        if validate_fn is not None:
            torch.hub._validate_not_a_forked_repo = validate_fn  # type: ignore[attr-defined]
    midas.to(device_t)
    midas.eval()

    validate_fn = getattr(torch.hub, "_validate_not_a_forked_repo", None)
    if validate_fn is not None:
        torch.hub._validate_not_a_forked_repo = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    try:
        try:
            transforms = torch.hub.load(
                "intel-isl/MiDaS",
                "transforms",
                trust_repo=True,
                skip_validation=True,
            )
        except TypeError:
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    finally:
        if validate_fn is not None:
            torch.hub._validate_not_a_forked_repo = validate_fn  # type: ignore[attr-defined]
    if "small" in model_type or "DPT" not in model_type:
        transform = transforms.small_transform
    else:
        transform = transforms.dpt_transform

    inp = transform(rgb).to(device_t)
    with torch.no_grad():
        pred = midas(inp)
        pred = F.interpolate(
            pred.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

    rel = pred.squeeze(0).detach().float().cpu().numpy()
    rel = np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize to [0,1]
    lo = float(np.percentile(rel, 1))
    hi = float(np.percentile(rel, 99))
    rel = np.clip((rel - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    # Convert to depth-like Z: nearer=smaller Z
    inv = rel
    z = 1.0 / (0.01 + 0.99 * inv)

    # Normalize and map to [min_depth, max_depth]
    z = (z - z.min()) / max(z.max() - z.min(), 1e-6)
    z = z * (max_depth - min_depth) + min_depth
    return z.astype(np.float32)


_DEPTHPRO_MODEL = None
_DEPTHPRO_TRANSFORM = None


def _predict_depth_depthpro(rgb: np.ndarray, device: str, f_px: Optional[float]) -> Tuple[np.ndarray, Optional[float]]:
    """Metric depth via Apple's DepthPro.

    Returns (depth_meters, focal_length_px_or_None).
    """
    try:
        import torch
        import depth_pro  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "depth_pro is not installed. Install optional deps: see backend/README.md (DepthPro section)."
        ) from e

    # DepthPro's default config loads weights from `./checkpoints/depth_pro.pt` (relative to CWD).
    # Make this robust by locating the checkpoint relative to this repo.
    from dataclasses import replace
    from pathlib import Path

    if device == "auto":
        if torch.backends.mps.is_available():
            device_t = torch.device("mps")
        elif torch.cuda.is_available():
            device_t = torch.device("cuda")
        else:
            device_t = torch.device("cpu")
    else:
        device_t = torch.device(device)

    global _DEPTHPRO_MODEL, _DEPTHPRO_TRANSFORM
    if _DEPTHPRO_MODEL is None or _DEPTHPRO_TRANSFORM is None:
        ckpt_candidates = [
            # If you cloned apple/ml-depth-pro into backend/ml-depth-pro
            Path(__file__).resolve().parent / "ml-depth-pro" / "checkpoints" / "depth_pro.pt",
            # If you run from within the DepthPro repo root
            Path.cwd() / "checkpoints" / "depth_pro.pt",
        ]
        ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)

        if ckpt_path is None:
            raise RuntimeError(
                "DepthPro checkpoint not found. Expected one of:\n"
                + "\n".join(f"- {p}" for p in ckpt_candidates)
                + "\nDownload it by running: backend/ml-depth-pro/get_pretrained_models.sh"
            )

        # Use DepthPro's default architecture, just override where to load weights from.
        try:
            from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Unexpected DepthPro package layout; cannot import DepthPro config.") from e

        cfg = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(ckpt_path))
        _DEPTHPRO_MODEL, _DEPTHPRO_TRANSFORM = depth_pro.create_model_and_transforms(config=cfg)
        _DEPTHPRO_MODEL.eval()

    _DEPTHPRO_MODEL.to(device_t)

    image = Image.fromarray(rgb, mode="RGB")
    inp = _DEPTHPRO_TRANSFORM(image)
    if inp.ndim == 3:
        inp = inp.unsqueeze(0)
    inp = inp.to(device_t)

    f_px_arg = None
    if f_px is not None:
        # DepthPro expects a tensor-like `f_px` because it calls `.squeeze()`.
        f_px_arg = torch.tensor([float(f_px)], device=device_t, dtype=torch.float32)

    with torch.no_grad():
        pred = _DEPTHPRO_MODEL.infer(inp, f_px=f_px_arg)

    depth = pred["depth"]
    if isinstance(depth, torch.Tensor):
        depth_np = depth.detach().cpu().numpy()
    else:
        depth_np = np.asarray(depth)

    # DepthPro returns depth as (H,W) (see depth_pro.DepthPro.infer: depth.squeeze()).
    # Be tolerant in case a different version returns (B,H,W).
    if depth_np.ndim == 3:
        depth_np = depth_np[0]
    elif depth_np.ndim != 2:
        raise RuntimeError(f"Unexpected DepthPro depth shape: {depth_np.shape}")
    depth_np = depth_np.astype(np.float32)
    foc = pred.get("focallength_px")
    foc_out: Optional[float] = None
    if foc is not None:
        if isinstance(foc, torch.Tensor):
            foc = foc.detach().cpu().numpy()
        try:
            foc_out = float(np.asarray(foc).reshape(-1)[0])
        except Exception:
            foc_out = None

    return depth_np, foc_out


def _depth_to_points(depth: np.ndarray, k: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    z = depth.astype(np.float32)
    x = (uu - cx) * z / max(fx, 1e-6)
    y = (vv - cy) * z / max(fy, 1e-6)

    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return pts


def _write_ply_xyzrgb(path: Path, points: np.ndarray, colors_u8: np.ndarray) -> None:
    assert points.shape[0] == colors_u8.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        for p, c in zip(points, colors_u8):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def _scale_intrinsics(k: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    kk = k.copy().astype(np.float32)
    kk[0, 0] *= float(scale_x)
    kk[1, 1] *= float(scale_y)
    kk[0, 2] *= float(scale_x)
    kk[1, 2] *= float(scale_y)
    return kk


def _resize_mask_nearest(mask_u8: np.ndarray, w: int, h: int) -> np.ndarray:
    if mask_u8.shape[:2] == (h, w):
        return mask_u8
    try:
        import cv2  # type: ignore

        return cv2.resize(mask_u8, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
    except Exception:
        # Pillow fallback
        m = Image.fromarray(mask_u8, mode="L")
        m = m.resize((int(w), int(h)), resample=Image.Resampling.NEAREST)
        return np.array(m, dtype=np.uint8)


def run_step1(
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
    max_points: int,
    drop_person_points: bool,
    export_ply: bool = True,
) -> Step1Outputs:
    frames = list(read_frames_jsonl(capture_dir))
    if not frames:
        raise RuntimeError(f"No frames in {capture_dir}")

    idx = int(frame_index)
    if idx < 0:
        idx = len(frames) + idx
    if idx < 0 or idx >= len(frames):
        raise IndexError(f"frame_index {frame_index} out of range (0..{len(frames)-1})")

    rec = frames[idx]
    img_path = capture_dir / rec.image_file
    rgb = _load_rgb(img_path)

    k, w, h = get_intrinsics_for_saved_image(rec, capture_dir)
    if rgb.shape[1] != w or rgb.shape[0] != h:
        # Use JPEG dimensions as truth for raster ops.
        h, w = rgb.shape[0], rgb.shape[1]

    if out_dir is None:
        out_dir = capture_dir / f"step1_{idx:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) person mask
    # Prefer segmentation-based methods; centerbox is a last-resort fallback.
    if mask_method == "none":
        person_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    elif mask_method == "rembg":
        person_mask = _person_mask_rembg(rgb)
        person_mask = _dilate_mask(person_mask, mask_dilate)
    elif mask_method == "centerbox":
        person_mask = _person_mask_center_box(rgb)
        person_mask = _dilate_mask(person_mask, mask_dilate)
    elif mask_method == "langsam":
        if mask_shape not in ("original", "rectangle", "square"):
            raise ValueError("mask_shape must be one of: original, rectangle, square")
        person_mask = _person_mask_langsam(
            rgb,
            text_prompt=mask_prompt,
            sam_type=langsam_sam_type,
            box_threshold=langsam_box_threshold,
            text_threshold=langsam_text_threshold,
            mask_shape=mask_shape,  # type: ignore[arg-type]
            dilate_iterations=mask_dilate,
        )
    elif mask_method == "auto":
        person_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        # 1) Try LangSAM first (true segmentation).
        try:
            if mask_shape not in ("original", "rectangle", "square"):
                raise ValueError("mask_shape must be one of: original, rectangle, square")
            m = _person_mask_langsam(
                rgb,
                text_prompt=mask_prompt,
                sam_type=langsam_sam_type,
                box_threshold=langsam_box_threshold,
                text_threshold=langsam_text_threshold,
                mask_shape=mask_shape,  # type: ignore[arg-type]
                dilate_iterations=mask_dilate,
            )
            if int(np.count_nonzero(m)) > 64:
                person_mask = m
        except Exception:
            pass

        # 2) Fallback to rembg.
        if int(np.count_nonzero(person_mask)) <= 64:
            try:
                m = _person_mask_rembg(rgb)
                m = _dilate_mask(m, mask_dilate)
                if int(np.count_nonzero(m)) > 64:
                    person_mask = m
            except Exception:
                pass

        # 3) Last resort: center box.
        if int(np.count_nonzero(person_mask)) <= 64:
            person_mask = _person_mask_center_box(rgb)
            person_mask = _dilate_mask(person_mask, mask_dilate)
    else:
        raise ValueError("mask_method must be one of: none, rembg, centerbox, langsam, auto")

    # Create an explicit hole image (person region blanked).
    hole_rgb = _apply_hole(rgb, person_mask, fill_rgb=(255, 255, 255))

    # 2) inpaint
    if inpaint_method == "opencv":
        scene_rgb = _inpaint_opencv(rgb, person_mask, inpaint_radius)
    elif inpaint_method == "diffusers":
        scene_rgb = _inpaint_diffusers(
            hole_rgb,
            person_mask,
            model_id=inpaint_model,
            device=inpaint_device,
            num_inference_steps=inpaint_steps,
            guidance_scale=inpaint_guidance,
            seed=inpaint_seed,
        )
    elif inpaint_method == "none":
        scene_rgb = rgb
    else:
        raise ValueError("inpaint_method must be one of: opencv, diffusers, none")

    # Diffusers inpainting pipelines may internally resize/crop.
    # Keep downstream stages consistent by adapting mask + intrinsics to the inpainted resolution.
    if scene_rgb.shape[:2] != rgb.shape[:2]:
        orig_h, orig_w = rgb.shape[:2]
        scene_h, scene_w = scene_rgb.shape[:2]
        scale_x = float(scene_w) / float(orig_w)
        scale_y = float(scene_h) / float(orig_h)
        person_mask = _resize_mask_nearest(person_mask, w=scene_w, h=scene_h)
        k = _scale_intrinsics(k, scale_x=scale_x, scale_y=scale_y)
        h, w = scene_h, scene_w

    # 3) depth
    depthpro_f_px: Optional[float] = None
    depthpro_f_out: Optional[float] = None
    if depth_method == "midas":
        depth = _predict_depth_midas(
            scene_rgb,
            device=depth_device,
            model_type=depth_model,
            min_depth=min_depth,
            max_depth=max_depth,
        )
    elif depth_method == "depthpro":
        depthpro_f_px = float(k[0, 0])
        depth, depthpro_f_out = _predict_depth_depthpro(
            scene_rgb,
            device=depth_device,
            f_px=depthpro_f_px,
        )
    else:
        raise ValueError("depth_method must be one of: midas, depthpro")

    ply_world_path: Optional[Path] = None
    if export_ply:
        # 4) point cloud sampling policy
        # If you want a *complete* filled scene, keep all pixels (including the inpainted region).
        # Optionally you can drop the original person region to leave a geometric hole.
        if drop_person_points:
            depth_masked = depth.copy()
            depth_masked[person_mask > 0] = np.nan
        else:
            depth_masked = depth

        # 5) points + colors
        pts = _depth_to_points(depth_masked, k)
        cols = scene_rgb.reshape(-1, 3)

        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        cols = cols[valid]

        if pts.shape[0] == 0:
            raise RuntimeError("No valid points after masking; check mask/inpaint/depth settings")

        # Downsample deterministically
        if pts.shape[0] > max_points:
            keep = np.linspace(0, pts.shape[0] - 1, num=max_points).astype(np.int64)
            pts = pts[keep]
            cols = cols[keep]

        # Optionally transform to world (Open3D-friendly pose)
        cw_open3d = arkit_cw_to_open3d_cw(rec.camera_transform_cw_4x4)
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
        pts_world = (cw_open3d @ pts_h.T).T[:, :3]

    # Outputs
    hole_path = out_dir / "scene_hole.png"
    scene_path = out_dir / "scene_image.png"
    mask_path = out_dir / "person_mask.png"
    depth_path = out_dir / "depth_pred.npy"
    ply_cam_path = out_dir / "scene_points_camera.ply"
    ply_world_path_tmp = out_dir / "scene_points_world.ply"

    _save_png(hole_path, hole_rgb)
    _save_png(scene_path, scene_rgb)
    _save_png(mask_path, person_mask)
    np.save(depth_path, depth)
    if export_ply:
        _write_ply_xyzrgb(ply_cam_path, pts.astype(np.float32), cols.astype(np.uint8))
        _write_ply_xyzrgb(ply_world_path_tmp, pts_world.astype(np.float32), cols.astype(np.uint8))
        ply_world_path = ply_world_path_tmp

    meta = {
        "capture_dir": str(capture_dir),
        "frame_index": idx,
        "image_file": rec.image_file,
        "mask_method": mask_method,
        "mask_dilate": int(mask_dilate),
        "inpaint_method": inpaint_method,
        "depth_method": depth_method,
        "depth_model": depth_model,
        "depth_device": depth_device,
        "min_depth": float(min_depth),
        "max_depth": float(max_depth),
        "depthpro_f_px_in": depthpro_f_px,
        "depthpro_f_px_out": depthpro_f_out,
        "intrinsics_saved": k.astype(float).tolist(),
        "saved_image_wh": [int(w), int(h)],
        "points": int(pts_world.shape[0]) if export_ply else 0,
        "export_ply": bool(export_ply),
    }
    (out_dir / "step1_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return Step1Outputs(
        scene_image_path=scene_path,
        person_mask_path=mask_path,
        depth_npy_path=depth_path,
        ply_path=ply_world_path,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Step1: remove human -> inpaint -> depth -> point cloud (single keyframe)")
    ap.add_argument("--input", required=True, type=Path, help="Path to SpatialCapture_* folder")
    ap.add_argument("--frame-index", type=int, default=0, help="Keyframe index to process (supports negative)")

    ap.add_argument(
        "--mask",
        choices=["auto", "langsam", "rembg", "centerbox", "none"],
        default="auto",
        help="Masking strategy. 'auto' prefers LangSAM segmentation then falls back to rembg then centerbox.",
    )
    ap.add_argument("--mask-dilate", type=int, default=4)
    ap.add_argument("--mask-prompt", default="person.", help="LangSAM text prompt (used when --mask langsam)")
    ap.add_argument(
        "--langsam-sam-type",
        default="sam2.1_hiera_large",
        help="LangSAM SAM type (used when --mask langsam). Example: sam2.1_hiera_large",
    )
    ap.add_argument(
        "--langsam-box-threshold",
        type=float,
        default=0.15,
        help="GroundingDINO box threshold for LangSAM (lower finds more, but may add false positives)",
    )
    ap.add_argument(
        "--langsam-text-threshold",
        type=float,
        default=0.15,
        help="GroundingDINO text threshold for LangSAM (lower finds more, but may add false positives)",
    )
    ap.add_argument(
        "--mask-shape",
        choices=["original", "rectangle", "square"],
        default="original",
        help="Optional post-process shape for the person mask (used when --mask langsam)",
    )

    ap.add_argument("--inpaint", choices=["opencv", "diffusers", "none"], default="opencv")
    ap.add_argument("--inpaint-radius", type=int, default=3)
    ap.add_argument("--inpaint-device", default="auto", help="auto|cpu|mps|cuda (used for --inpaint diffusers)")
    ap.add_argument(
        "--inpaint-model",
        default="runwayml/stable-diffusion-inpainting",
        help="HuggingFace model id for diffusers inpainting (used for --inpaint diffusers)",
    )
    ap.add_argument("--inpaint-steps", type=int, default=25)
    ap.add_argument("--inpaint-guidance", type=float, default=7.5)
    ap.add_argument("--inpaint-seed", type=int, default=None)

    ap.add_argument("--depth", choices=["midas", "depthpro"], default="midas", help="Depth backend")

    ap.add_argument("--depth-model", default="MiDaS_small", help="MiDaS model name (e.g., MiDaS_small, DPT_Hybrid)")
    ap.add_argument("--depth-device", default="auto", help="auto|cpu|mps|cuda")
    ap.add_argument("--min-depth", type=float, default=0.5)
    ap.add_argument("--max-depth", type=float, default=10.0)

    ap.add_argument("--out", type=Path, default=None, help="Output directory (default: <capture>/step1_XXXXXX)")
    ap.add_argument("--max-points", type=int, default=400_000)
    ap.add_argument(
        "--drop-person-points",
        action="store_true",
        help="If set, removes points in the original person mask region (creates a geometric hole).",
    )
    ap.add_argument(
        "--no-ply",
        action="store_true",
        help="If set, skips point cloud / PLY export (still writes mask/inpaint/depth).",
    )

    args = ap.parse_args()

    outputs = run_step1(
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
        max_points=args.max_points,
        drop_person_points=bool(args.drop_person_points),
        export_ply=not bool(args.no_ply),
    )

    print("Wrote:")
    print(f"- {outputs.scene_image_path}")
    print(f"- {outputs.person_mask_path}")
    print(f"- {outputs.depth_npy_path}")
    if outputs.ply_path is not None:
        print(f"- {outputs.ply_path}")


if __name__ == "__main__":
    main()
