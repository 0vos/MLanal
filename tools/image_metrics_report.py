#!/usr/bin/env python3
"""Generate a per-image quality/metadata table for images under a folder.

Metrics included (per file):
- brightness (mean luminance), contrast (luminance std)
- sharpness (variance of Laplacian on luminance)
- noise level estimate (robust sigma from Laplacian MAD)
- color distribution summary (RGB mean/std, entropies, and 16-bin histograms)
- resolution and size

Example:
  /Users/rufuslee/Documents/GitHub/MLanal/mlanalenv/bin/python backend/image_metrics_report.py \
    --input images --output images/image_metrics_report.csv --recursive
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ImageRow:
    path: str
    filename: str
    ext: str
    format: str
    file_size_bytes: int

    width_px: int
    height_px: int
    megapixels: float
    aspect_ratio: float

    brightness_mean: float
    contrast_std: float
    sharpness_laplacian_var: float
    noise_sigma_est: float

    r_mean: float
    g_mean: float
    b_mean: float
    r_std: float
    g_std: float
    b_std: float

    r_entropy: float
    g_entropy: float
    b_entropy: float

    rgb_hist_16: str


def _iter_image_files(input_dir: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        it = input_dir.rglob("*")
    else:
        it = input_dir.glob("*")

    for path in it:
        if not path.is_file():
            continue
        if path.suffix.lower() in IMAGE_EXTS:
            yield path


def _rgb_to_luminance(rgb_u8: np.ndarray) -> np.ndarray:
    # rgb_u8: HxWx3 uint8
    rgb = rgb_u8.astype(np.float32)
    # Rec. 709 luminance
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _laplacian_4n(img: np.ndarray) -> np.ndarray:
    """4-neighbor Laplacian with reflect padding."""
    img = img.astype(np.float32)
    p = np.pad(img, ((1, 1), (1, 1)), mode="reflect")
    center = p[1:-1, 1:-1]
    up = p[:-2, 1:-1]
    down = p[2:, 1:-1]
    left = p[1:-1, :-2]
    right = p[1:-1, 2:]
    return (up + down + left + right) - 4.0 * center


def _entropy_from_u8(channel_u8: np.ndarray) -> float:
    hist = np.bincount(channel_u8.reshape(-1), minlength=256).astype(np.float64)
    p = hist / hist.sum() if hist.sum() > 0 else hist
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _hist_16_from_u8(channel_u8: np.ndarray) -> list[float]:
    hist, _ = np.histogram(channel_u8, bins=16, range=(0, 256), density=False)
    total = float(hist.sum())
    if total <= 0:
        return [0.0] * 16
    return [float(x / total) for x in hist]


def _robust_sigma_from_laplacian(lap: np.ndarray) -> float:
    # Robust sigma estimate via MAD. Interpretable as a relative noise proxy.
    med = float(np.median(lap))
    mad = float(np.median(np.abs(lap - med)))
    if mad == 0.0:
        return 0.0
    return mad / 0.6745


def _analyze_one(path: Path, base_dir: Path) -> Optional[ImageRow]:
    try:
        file_size = path.stat().st_size
        with Image.open(path) as im0:
            im = ImageOps.exif_transpose(im0)
            fmt = (im.format or "").strip() or "unknown"
            im_rgb = im.convert("RGB")

        rgb = np.array(im_rgb, dtype=np.uint8)
        h, w = rgb.shape[0], rgb.shape[1]

        lum = _rgb_to_luminance(rgb)
        brightness_mean = float(lum.mean())
        contrast_std = float(lum.std())

        lap = _laplacian_4n(lum)
        sharpness_var = float(lap.var())
        noise_sigma = float(_robust_sigma_from_laplacian(lap))

        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]

        r_mean = float(r.mean())
        g_mean = float(g.mean())
        b_mean = float(b.mean())
        r_std = float(r.std())
        g_std = float(g.std())
        b_std = float(b.std())

        r_ent = _entropy_from_u8(r)
        g_ent = _entropy_from_u8(g)
        b_ent = _entropy_from_u8(b)

        hist_payload = {
            "r": _hist_16_from_u8(r),
            "g": _hist_16_from_u8(g),
            "b": _hist_16_from_u8(b),
        }

        rel = str(path.resolve())
        try:
            rel = str(path.resolve().relative_to(base_dir.resolve()))
        except Exception:
            pass

        megapixels = float((w * h) / 1_000_000.0)
        aspect_ratio = float(w / h) if h != 0 else float("nan")

        return ImageRow(
            path=rel,
            filename=path.name,
            ext=path.suffix.lower(),
            format=fmt,
            file_size_bytes=int(file_size),
            width_px=int(w),
            height_px=int(h),
            megapixels=megapixels,
            aspect_ratio=aspect_ratio,
            brightness_mean=brightness_mean,
            contrast_std=contrast_std,
            sharpness_laplacian_var=sharpness_var,
            noise_sigma_est=noise_sigma,
            r_mean=r_mean,
            g_mean=g_mean,
            b_mean=b_mean,
            r_std=r_std,
            g_std=g_std,
            b_std=b_std,
            r_entropy=float(r_ent),
            g_entropy=float(g_ent),
            b_entropy=float(b_ent),
            rgb_hist_16=json.dumps(hist_payload, ensure_ascii=False),
        )
    except Exception as e:
        print(f"[WARN] Failed to analyze: {path} ({e})")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze images and output a metrics table.")
    parser.add_argument("--input", "-i", type=str, default="images", help="Input folder (default: images)")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="images/image_metrics_report.csv",
        help="Output path (.csv or .xlsx). Default: images/image_metrics_report.csv",
    )
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of images (0 means no limit)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")

    rows: list[dict] = []
    n_seen = 0

    for img_path in _iter_image_files(input_dir, args.recursive):
        n_seen += 1
        row = _analyze_one(img_path, base_dir=Path.cwd())
        if row is not None:
            rows.append(asdict(row))

        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        raise SystemExit(f"No images found under: {input_dir}")

    df = pd.DataFrame(rows)

    # A stable, readable column order
    column_order = [
        "path",
        "filename",
        "ext",
        "format",
        "file_size_bytes",
        "width_px",
        "height_px",
        "megapixels",
        "aspect_ratio",
        "brightness_mean",
        "contrast_std",
        "sharpness_laplacian_var",
        "noise_sigma_est",
        "r_mean",
        "g_mean",
        "b_mean",
        "r_std",
        "g_std",
        "b_std",
        "r_entropy",
        "g_entropy",
        "b_entropy",
        "rgb_hist_16",
    ]

    df = df[[c for c in column_order if c in df.columns]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(output_path, index=False, encoding="utf-8")
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False)
    else:
        raise SystemExit("Output must end with .csv or .xlsx")

    print(f"Wrote {len(df)} rows to: {output_path}")
    print(f"Scanned files (including failures): {n_seen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
