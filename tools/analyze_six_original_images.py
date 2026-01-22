#!/usr/bin/env python3
"""Analyze the six original images (IMG_9546/9568/9569/9579/9581/9582) and write a small report.

Outputs (default under images/original_six_analysis/):
- six_images_metrics.csv : numeric metrics per image
- six_images_report.md   : human-readable analysis (Chinese)
- per-image plots         : luminance histogram + saturation histogram

Run:
  mlanalenv/bin/python tools/analyze_six_original_images.py \
    --images-dir images \
    --outdir images/original_six_analysis
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


STEMS = ["IMG_9546", "IMG_9568", "IMG_9569", "IMG_9579", "IMG_9581", "IMG_9582"]

SCENE_HINTS = {
    "IMG_9546": "复杂工作间环境（物体零碎且不常见，远处有房子）",
    "IMG_9568": "卧室角落（封闭、堆叠、多光线）",
    "IMG_9569": "工位桌椅（结构简单、平面直角多）",
    "IMG_9579": "走廊（长而直的地面、强透视）",
    "IMG_9581": "玻璃罩船模（反光/高光导致几何歧义）",
    "IMG_9582": "黑色电梯间（低照度/暗部信息少）",
}


def rgb_to_luminance(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def laplacian_4n(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    p = np.pad(img, ((1, 1), (1, 1)), mode="reflect")
    c = p[1:-1, 1:-1]
    up = p[:-2, 1:-1]
    down = p[2:, 1:-1]
    left = p[1:-1, :-2]
    right = p[1:-1, 2:]
    return (up + down + left + right) - 4.0 * c


def robust_sigma_from_laplacian(lap: np.ndarray) -> float:
    med = float(np.median(lap))
    mad = float(np.median(np.abs(lap - med)))
    if mad == 0.0:
        return 0.0
    return mad / 0.6745


def rgb_to_hsv_saturation(rgb_u8: np.ndarray) -> np.ndarray:
    """Return saturation in [0,1]. Lightweight HSV conversion."""
    rgb = rgb_u8.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    # S = delta / cmax, define S=0 when cmax==0
    s = np.zeros_like(cmax)
    mask = cmax > 1e-8
    s[mask] = delta[mask] / cmax[mask]
    return s


@dataclass
class Metrics:
    stem: str
    scene: str
    path: str
    file_size_bytes: int
    width_px: int
    height_px: int
    megapixels: float

    brightness_mean: float
    contrast_std: float
    sharpness_laplacian_var: float
    noise_sigma_est: float

    # Exposure / clipping proxies
    shadow_clip_frac: float  # luminance <= 5
    highlight_clip_frac: float  # luminance >= 250
    midtone_frac: float  # 40<=luma<=200

    # Color proxies
    saturation_mean: float
    saturation_p95: float
    color_cast_rg: float  # r_mean - g_mean
    color_cast_bg: float  # b_mean - g_mean


def find_image(images_dir: Path, stem: str) -> Path:
    # Prefer jpeg/jpeg then anything else
    candidates = [
        images_dir / f"{stem}.jpeg",
        images_dir / f"{stem}.jpg",
        images_dir / f"{stem}.png",
        images_dir / f"{stem}.webp",
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback: any match
    matches = list(images_dir.glob(f"{stem}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Image not found for stem={stem} under {images_dir}")


def analyze_one(path: Path, stem: str, base_dir: Path) -> Metrics:
    file_size = path.stat().st_size
    with Image.open(path) as im0:
        im = ImageOps.exif_transpose(im0)
        im_rgb = im.convert("RGB")

    rgb = np.array(im_rgb, dtype=np.uint8)
    h, w = rgb.shape[0], rgb.shape[1]

    lum = rgb_to_luminance(rgb)
    brightness_mean = float(lum.mean())
    contrast_std = float(lum.std())

    lap = laplacian_4n(lum)
    sharpness_var = float(lap.var())
    noise_sigma = float(robust_sigma_from_laplacian(lap))

    shadow_clip = float(np.mean(lum <= 5.0))
    highlight_clip = float(np.mean(lum >= 250.0))
    midtone = float(np.mean((lum >= 40.0) & (lum <= 200.0)))

    sat = rgb_to_hsv_saturation(rgb)
    sat_mean = float(np.mean(sat))
    sat_p95 = float(np.quantile(sat, 0.95))

    r_mean = float(rgb[..., 0].mean())
    g_mean = float(rgb[..., 1].mean())
    b_mean = float(rgb[..., 2].mean())

    try:
        rel = str(path.resolve().relative_to(base_dir.resolve()))
    except Exception:
        rel = str(path.resolve())

    return Metrics(
        stem=stem,
        scene=SCENE_HINTS.get(stem, ""),
        path=rel,
        file_size_bytes=int(file_size),
        width_px=int(w),
        height_px=int(h),
        megapixels=float((w * h) / 1_000_000.0),
        brightness_mean=brightness_mean,
        contrast_std=contrast_std,
        sharpness_laplacian_var=sharpness_var,
        noise_sigma_est=noise_sigma,
        shadow_clip_frac=shadow_clip,
        highlight_clip_frac=highlight_clip,
        midtone_frac=midtone,
        saturation_mean=sat_mean,
        saturation_p95=sat_p95,
        color_cast_rg=float(r_mean - g_mean),
        color_cast_bg=float(b_mean - g_mean),
    )


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def _qual_exposure(m: Metrics) -> str:
    # Very simple heuristics. Thresholds are empirical & report-friendly.
    if m.brightness_mean < 70 and m.shadow_clip_frac > 0.02:
        return "偏暗且暗部裁切明显"
    if m.brightness_mean < 80:
        return "偏暗"
    if m.highlight_clip_frac > 0.01 and m.brightness_mean > 140:
        return "偏亮且高光裁切明显"
    if m.highlight_clip_frac > 0.01:
        return "高光裁切明显"
    if 90 <= m.brightness_mean <= 140:
        return "曝光较均衡"
    return "曝光可能偏亮"


def _qual_sharpness(m: Metrics, sharp_ref: float) -> str:
    # Use relative ranking to avoid scene-dependent absolute thresholds.
    if m.sharpness_laplacian_var >= sharp_ref * 1.25:
        return "细节清晰（相对更锐）"
    if m.sharpness_laplacian_var <= sharp_ref * 0.75:
        return "细节偏软/可能有运动模糊"
    return "清晰度中等"


def _qual_noise(m: Metrics, noise_ref: float) -> str:
    if m.noise_sigma_est >= noise_ref * 1.25:
        return "噪声偏高（纹理/暗部可能更粗糙）"
    if m.noise_sigma_est <= noise_ref * 0.75:
        return "噪声较低"
    return "噪声中等"


def _qual_color(m: Metrics) -> str:
    cast = []
    if m.color_cast_rg > 6:
        cast.append("偏红/偏暖")
    elif m.color_cast_rg < -6:
        cast.append("偏绿/偏冷")

    if m.color_cast_bg > 6:
        cast.append("偏蓝")
    elif m.color_cast_bg < -6:
        cast.append("偏黄")

    if not cast:
        cast_str = "色彩中性"
    else:
        cast_str = "、".join(cast)

    if m.saturation_mean < 0.20:
        sat_str = "饱和度偏低（偏灰/低彩度）"
    elif m.saturation_mean > 0.35:
        sat_str = "饱和度偏高（颜色更浓）"
    else:
        sat_str = "饱和度中等"

    return f"{cast_str}；{sat_str}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze the six original images and write a small report.")
    ap.add_argument("--images-dir", type=str, default="images", help="Directory containing original images")
    ap.add_argument("--outdir", type=str, default="images/original_six_analysis", help="Output directory")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics: list[Metrics] = []
    for stem in STEMS:
        img_path = find_image(images_dir, stem)
        metrics.append(analyze_one(img_path, stem=stem, base_dir=Path.cwd()))

    df = pd.DataFrame([asdict(m) for m in metrics])

    # Reference levels for relative qualitative labels
    sharp_ref = float(df["sharpness_laplacian_var"].median())
    noise_ref = float(df["noise_sigma_est"].median())

    # Round for readability
    csv_path = outdir / "six_images_metrics.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Render markdown report
    # Keep a small, report-friendly table
    view_cols = [
        "stem",
        "scene",
        "width_px",
        "height_px",
        "megapixels",
        "brightness_mean",
        "contrast_std",
        "sharpness_laplacian_var",
        "noise_sigma_est",
        "shadow_clip_frac",
        "highlight_clip_frac",
        "saturation_mean",
    ]
    view = df[view_cols].copy()
    view["megapixels"] = view["megapixels"].map(lambda x: f"{x:.2f}")
    for c in ["brightness_mean", "contrast_std", "sharpness_laplacian_var", "noise_sigma_est", "saturation_mean"]:
        view[c] = view[c].map(lambda x: f"{x:.3f}")
    view["shadow_clip_frac"] = view["shadow_clip_frac"].map(_fmt_pct)
    view["highlight_clip_frac"] = view["highlight_clip_frac"].map(_fmt_pct)

    # Use DataFrame.to_markdown when available (tabulate installed earlier)
    table_md = view.to_markdown(index=False)

    lines: list[str] = []
    lines.append("# 6张原图信息分析（质量/曝光/色彩）")
    lines.append("")
    lines.append("说明：以下数值是从原图直接计算的统计量（亮度=Rec.709 灰度均值；对比度=灰度标准差；锐度=灰度拉普拉斯方差；噪声=拉普拉斯 MAD 估计）。")
    lines.append("另外加入了‘暗部/高光裁切比例’和‘饱和度’作为更直观的曝光/色彩代理指标。")
    lines.append("")
    lines.append("## 汇总表")
    lines.append("")
    lines.append(table_md)
    lines.append("")

    # Per-image narrative
    lines.append("## 分图结论（可直接引用）")
    lines.append("")
    for m in metrics:
        exposure = _qual_exposure(m)
        sharp = _qual_sharpness(m, sharp_ref=sharp_ref)
        noise = _qual_noise(m, noise_ref=noise_ref)
        color = _qual_color(m)

        lines.append(f"### {m.stem}")
        lines.append(f"- 场景：{m.scene}")
        lines.append(f"- 分辨率：{m.width_px}×{m.height_px}（{m.megapixels:.2f}MP）")
        lines.append(f"- 曝光/动态范围：{exposure}（暗部裁切={_fmt_pct(m.shadow_clip_frac)}；高光裁切={_fmt_pct(m.highlight_clip_frac)}；中间调占比={_fmt_pct(m.midtone_frac)}）")
        lines.append(f"- 细节/噪声：{sharp}；{noise}（锐度={m.sharpness_laplacian_var:.1f}；噪声估计={m.noise_sigma_est:.3f}）")
        lines.append(f"- 色彩：{color}（平均饱和度={m.saturation_mean:.3f}；S_p95={m.saturation_p95:.3f}）")
        lines.append("")

    # Cross-image highlights
    # Rankings for quick insights
    df_rank = df.copy()
    df_rank["stem"] = df_rank["stem"].astype(str)
    best_bright = df_rank.sort_values("brightness_mean", ascending=False).iloc[0]["stem"]
    darkest = df_rank.sort_values("brightness_mean", ascending=True).iloc[0]["stem"]
    sharpest = df_rank.sort_values("sharpness_laplacian_var", ascending=False).iloc[0]["stem"]
    softest = df_rank.sort_values("sharpness_laplacian_var", ascending=True).iloc[0]["stem"]
    noisiest = df_rank.sort_values("noise_sigma_est", ascending=False).iloc[0]["stem"]
    cleanest = df_rank.sort_values("noise_sigma_est", ascending=True).iloc[0]["stem"]

    lines.append("## 跨图对比要点")
    lines.append("")
    lines.append(f"- 最亮：{best_bright}；最暗：{darkest}")
    lines.append(f"- 最清晰：{sharpest}；最软：{softest}")
    lines.append(f"- 噪声最高：{noisiest}；噪声最低：{cleanest}")
    lines.append("- 建议：如果要做后续 3D/深度重建的稳定性讨论，可把‘暗部裁切+噪声’与 Depth/点云误差一起作为解释变量（暗部信息少/噪声高更容易导致几何不稳定）。")
    lines.append("")

    report_path = outdir / "six_images_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
