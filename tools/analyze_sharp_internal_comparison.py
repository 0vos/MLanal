#!/usr/bin/env python3
"""Create an internal SHARP-result comparison report for SHARP-generated PLY depthmaps.

Inputs (defaults match this repo's outputs):
- images/sharp_depthmaps/sharp_ply_depthmaps_summary.csv
- images/image_metrics_report.csv (optional; used to correlate photo stats vs depth stats)
- images/sharp_depthmaps/<IMG_xxxx>/{depth_vis.png,density.png,color_mean.png,hist_depth.png}
- images/<IMG_xxxx>.(jpg|jpeg|png|...)

Outputs:
- images/sharp_depthmaps/analysis/ (plots + montages)
- images/sharp_depthmaps/analysis/sharp_internal_comparison.md (report)

Run:
  /Users/rufuslee/Documents/GitHub/MLanal/mlanalenv/bin/python tools/analyze_sharp_internal_comparison.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageOps


SCENE_DESCRIPTIONS: Dict[str, str] = {
    "IMG_9546": "复杂工作间环境：物体零碎且不常见，能看到远处房子（大景深/远距离结构）",
    "IMG_9568": "卧室角落：较封闭，有高低物体堆叠，多种光线（遮挡与多光源）",
    "IMG_9569": "桌椅工位：结构相对简单（平面/直角多）",
    "IMG_9579": "走廊：漫长笔直地面（强透视/长平面）",
    "IMG_9581": "玻璃罩内的船模：玻璃反光（高光/反射导致深度歧义）",
    "IMG_9582": "黑色电梯间：整体偏暗（低照度/噪声/动态范围挑战）",
}


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"]


def _save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _stem_from_row(ply_path: str) -> str:
    return Path(ply_path).stem


def _find_original_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        p1 = images_dir / f"{stem}{ext}"
        p2 = images_dir / f"{stem}{ext.upper()}"
        if p1.exists():
            return p1
        if p2.exists():
            return p2
    return None


def _open_rgb(path: Path, max_side: int = 900) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")
    w, h = im.size
    scale = min(1.0, float(max_side) / max(w, h))
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return im


def _make_montage(
    stem: str,
    images_dir: Path,
    sharp_dir: Path,
    out_path: Path,
) -> bool:
    orig = _find_original_image(images_dir, stem)
    tiles: list[tuple[str, Optional[Path]]] = [
        ("original", orig),
        ("depth_vis", sharp_dir / stem / "depth_vis.png"),
        ("density", sharp_dir / stem / "density.png"),
        ("color_mean", sharp_dir / stem / "color_mean.png"),
        ("hist_depth", sharp_dir / stem / "hist_depth.png"),
        ("mask", sharp_dir / stem / "mask.png"),
    ]

    # Load existing images
    loaded: list[tuple[str, Image.Image]] = []
    for title, p in tiles:
        if p is None or not p.exists():
            continue
        try:
            im = _open_rgb(p, max_side=900)
            loaded.append((title, im))
        except Exception:
            continue

    if not loaded:
        return False

    # Grid: 3 columns
    cols = 3
    rows = int(np.ceil(len(loaded) / cols))

    # Normalize tile size
    tile_w = max(im.size[0] for _, im in loaded)
    tile_h = max(im.size[1] for _, im in loaded)

    pad = 10
    label_h = 26

    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = rows * (tile_h + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))

    # Simple labels without extra font deps (draw rectangle + omit text if font missing)
    try:
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for idx, (title, im) in enumerate(loaded):
            r = idx // cols
            c = idx % cols
            x0 = pad + c * (tile_w + pad)
            y0 = pad + r * (tile_h + label_h + pad)

            # Paste centered
            ox, oy = im.size
            px = x0 + (tile_w - ox) // 2
            py = y0 + label_h + (tile_h - oy) // 2
            canvas.paste(im, (px, py))

            # Label bar
            draw.rectangle([x0, y0, x0 + tile_w, y0 + label_h], fill=(45, 45, 45))
            if font is not None:
                draw.text((x0 + 8, y0 + 6), title, fill=(230, 230, 230), font=font)

        canvas.save(out_path)
        return True
    except Exception:
        # Fallback: no labels
        x = pad
        y = pad
        for _, im in loaded:
            canvas.paste(im, (x, y))
            x += tile_w + pad
        canvas.save(out_path)
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate SHARP internal comparison report.")
    ap.add_argument(
        "--sharp-summary",
        type=str,
        default="images/sharp_depthmaps/sharp_ply_depthmaps_summary.csv",
        help="SHARP depthmaps summary CSV",
    )
    ap.add_argument(
        "--image-metrics",
        type=str,
        default="images/image_metrics_report.csv",
        help="Image metrics CSV (optional; join by IMG_xxxx)",
    )
    ap.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Folder with original images",
    )
    ap.add_argument(
        "--sharp-dir",
        type=str,
        default="images/sharp_depthmaps",
        help="Folder with per-stem outputs",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="images/sharp_depthmaps/analysis",
        help="Output folder",
    )

    args = ap.parse_args()

    sharp_summary = Path(args.sharp_summary)
    img_metrics = Path(args.image_metrics)
    images_dir = Path(args.images_dir)
    sharp_dir = Path(args.sharp_dir)
    outdir = Path(args.outdir)

    if not sharp_summary.exists():
        raise SystemExit(f"Missing SHARP summary CSV: {sharp_summary}")
    outdir.mkdir(parents=True, exist_ok=True)

    s = pd.read_csv(sharp_summary)
    s["stem"] = s["ply_path"].map(_stem_from_row)
    s["scene"] = s["stem"].map(lambda x: SCENE_DESCRIPTIONS.get(x, ""))

    # Optional join with image metrics
    combined = s.copy()
    if img_metrics.exists():
        m = pd.read_csv(img_metrics)
        m["stem"] = m["filename"].map(lambda x: Path(str(x)).stem)
        combined = combined.merge(m, on="stem", how="left", suffixes=("_sharp", "_img"))

    # Save combined table (useful for later DepthPro comparison too)
    combined.to_csv(outdir / "sharp_internal_combined.csv", index=False, encoding="utf-8")

    # Plot: valid coverage
    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=combined.sort_values("valid_ratio"), x="stem", y="valid_ratio")
    plt.xticks(rotation=30, ha="right")
    plt.title("SHARP projected depth valid_ratio (coverage)")
    _save_fig(outdir / "bar_valid_ratio.png")

    # Plot: depth range (p99)
    plt.figure(figsize=(9, 4.5))
    order = combined.sort_values("depth_p99")["stem"].tolist()
    sns.barplot(data=combined, x="stem", y="depth_p99", order=order)
    plt.xticks(rotation=30, ha="right")
    plt.title("SHARP depth_p99 (far tail, robust)")
    plt.ylabel("depth")
    _save_fig(outdir / "bar_depth_p99.png")

    # Plot: depth median
    plt.figure(figsize=(9, 4.5))
    order = combined.sort_values("depth_median")["stem"].tolist()
    sns.barplot(data=combined, x="stem", y="depth_median", order=order)
    plt.xticks(rotation=30, ha="right")
    plt.title("SHARP depth_median")
    plt.ylabel("depth")
    _save_fig(outdir / "bar_depth_median.png")

    # Plot: density_mean
    plt.figure(figsize=(9, 4.5))
    order = combined.sort_values("density_mean")["stem"].tolist()
    sns.barplot(data=combined, x="stem", y="density_mean", order=order)
    plt.xticks(rotation=30, ha="right")
    plt.title("SHARP projected density_mean (points per pixel)")
    _save_fig(outdir / "bar_density_mean.png")

    # Correlations: select numeric columns from both sides
    numeric_cols = [c for c in combined.columns if combined[c].dtype.kind in "if"]
    if numeric_cols:
        corr = combined[numeric_cols].corr(numeric_only=True)
        corr.to_csv(outdir / "correlation_matrix.csv", encoding="utf-8")

        plt.figure(figsize=(min(28, 1.1 * len(corr.columns) + 3), min(28, 1.0 * len(corr.columns) + 3)))
        sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.25)
        plt.title("Correlation across SHARP depth stats + image metrics")
        _save_fig(outdir / "correlation_heatmap.png")

    # Scatter: brightness vs depth_mean (if available)
    if "brightness_mean" in combined.columns:
        plt.figure(figsize=(6.5, 4.5))
        sns.scatterplot(data=combined, x="brightness_mean", y="depth_mean")
        sns.regplot(data=combined, x="brightness_mean", y="depth_mean", scatter=False, color="orange")
        plt.title("brightness_mean vs SHARP depth_mean")
        _save_fig(outdir / "scatter_brightness_vs_depth_mean.png")

        plt.figure(figsize=(6.5, 4.5))
        sns.scatterplot(data=combined, x="brightness_mean", y="valid_ratio")
        sns.regplot(data=combined, x="brightness_mean", y="valid_ratio", scatter=False, color="orange")
        plt.title("brightness_mean vs SHARP valid_ratio")
        _save_fig(outdir / "scatter_brightness_vs_valid_ratio.png")

    # Create montages per scene
    montage_dir = outdir / "montages"
    montage_dir.mkdir(parents=True, exist_ok=True)
    montage_ok: set[str] = set()
    montage_rows = []
    for stem in combined["stem"].tolist():
        out_path = montage_dir / f"{stem}_montage.png"
        ok = _make_montage(stem, images_dir=images_dir, sharp_dir=sharp_dir, out_path=out_path)
        if ok:
            montage_ok.add(stem)
        montage_rows.append({"stem": stem, "montage": str(out_path) if ok else ""})

    pd.DataFrame(montage_rows).to_csv(outdir / "montages_index.csv", index=False, encoding="utf-8")

    # Build Markdown report
    report = outdir / "sharp_internal_comparison.md"

    # Lightweight summary table
    table_cols = [
        "stem",
        "scene",
        "valid_ratio",
        "depth_median",
        "depth_p99",
        "depth_max",
        "density_mean",
        "density_max",
        "camera_source",
    ]
    existing_cols = [c for c in table_cols if c in combined.columns]
    table = combined[existing_cols].copy()
    for c in ["valid_ratio", "depth_median", "depth_p99", "depth_max", "density_mean", "density_max"]:
        if c in table.columns:
            table[c] = pd.to_numeric(table[c], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.4f}")

    md_lines = []
    md_lines.append("# SHARP 内部结果对比分析（PLY → 深度投影）")
    md_lines.append("")
    md_lines.append("本报告基于 SHARP 输出的点云 PLY，使用 PLY 内置相机参数（或兜底参数）投影生成深度图与辅助图，并对 6 个场景做对比。")
    md_lines.append("")

    md_lines.append("## 总览表")
    md_lines.append("")
    md_lines.append(table.to_markdown(index=False))
    md_lines.append("")

    # Auto interpretation highlights (keep it short and scene-focused)
    md_lines.append("## 解读要点（仅基于 SHARP 投影统计）")
    md_lines.append("")

    def _rank(col: str, ascending: bool) -> list[str]:
        if col not in combined.columns:
            return []
        tmp = combined[["stem", col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna().sort_values(col, ascending=ascending)
        return tmp["stem"].tolist()

    farthest = _rank("depth_p99", ascending=False)[:2]
    closest = _rank("depth_p99", ascending=True)[:2]
    md_lines.append(f"- 远距离结构最明显（depth_p99 最大）：{', '.join(farthest) if farthest else 'N/A'}")
    md_lines.append(f"- 场景更封闭/近距离为主（depth_p99 最小）：{', '.join(closest) if closest else 'N/A'}")

    corridor_like = _rank("depth_median", ascending=False)[:2]
    md_lines.append(f"- 中位深度更大（更像长走廊/远景）：{', '.join(corridor_like) if corridor_like else 'N/A'}")

    md_lines.append(
        "- 注意：6 张图的 valid_ratio 都约 4.48%~4.55%，说明这是“稀疏点云→像素投影（近似 z-buffer）”的覆盖率，并非稠密深度。做 DepthPro 对比时建议先统一 mask 再算误差。"
    )
    md_lines.append("")

    md_lines.append("## 对比图")
    md_lines.append("")
    md_lines.append("- 覆盖率（valid_ratio）：![](bar_valid_ratio.png)")
    md_lines.append("- 远处尾部深度（depth_p99）：![](bar_depth_p99.png)")
    md_lines.append("- 中位深度（depth_median）：![](bar_depth_median.png)")
    md_lines.append("- 像素点密度（density_mean）：![](bar_density_mean.png)")
    if (outdir / "correlation_heatmap.png").exists():
        md_lines.append("- 相关性热力图（含照片指标）：![](correlation_heatmap.png)")
    if (outdir / "scatter_brightness_vs_depth_mean.png").exists():
        md_lines.append("- 亮度 vs 深度均值：![](scatter_brightness_vs_depth_mean.png)")
        md_lines.append("- 亮度 vs 覆盖率：![](scatter_brightness_vs_valid_ratio.png)")
    md_lines.append("")

    md_lines.append("## 场景可视化（原图 / 深度 / 密度 / 颜色投影 / 直方图 / mask）")
    md_lines.append("")
    for stem in combined["stem"].tolist():
        desc = SCENE_DESCRIPTIONS.get(stem, "")
        montage_abs = montage_dir / f"{stem}_montage.png"
        montage_rel = Path("montages") / f"{stem}_montage.png"
        md_lines.append(f"### {stem}")
        if desc:
            md_lines.append(f"- 场景：{desc}")
        if stem in montage_ok and montage_abs.exists():
            md_lines.append(f"![]({montage_rel.as_posix()})")
        md_lines.append("")

    report.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote report: {report}")
    print(f"Wrote plots to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
