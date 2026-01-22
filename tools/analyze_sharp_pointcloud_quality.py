#!/usr/bin/env python3
"""Cross-image quality analysis for SHARP point clouds.

Goal
- Compare SHARP outputs across multiple images (no DepthPro needed)
- Provide a small set of quantitative, report-friendly indicators

Data sources
- SHARP PLY point cloud (binary_little_endian): <sharp_root>/<stem>.ply
- SHARP projected depth artifacts (already produced by tools/sharp_ply_depthmaps.py):
    <depthmaps_root>/<stem>/depth_raw.npy
    <depthmaps_root>/<stem>/mask.png
    <depthmaps_root>/<stem>/density.png

Metrics (per stem)
- Projection coverage: valid pixels / total pixels
- Projection density: mean & p95 points-per-pixel on valid pixels
- Depth stability: p95 of |∇depth| over valid neighbor pairs (lower is smoother)
- Point cloud structure: bbox diag, intra-cloud NN p50/p95 (sampled), outlier fraction
- A simple composite quality_score in [0,100] for ranking

Run:
  mlanalenv/bin/python tools/analyze_sharp_pointcloud_quality.py \
    --sharp-root /Users/rufuslee/Documents/GitHub/ml-sharp-main/output \
    --depthmaps-root images/sharp_depthmaps \
    --outdir images/sharp_pointcloud_quality/analysis
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree


STEMS_DEFAULT = ["IMG_9546", "IMG_9568", "IMG_9569", "IMG_9579", "IMG_9581", "IMG_9582"]

SCENE_DESCRIPTIONS: Dict[str, str] = {
    "IMG_9546": "复杂工作间环境（物体零碎且不常见，远处有房子）",
    "IMG_9568": "卧室角落（封闭、堆叠、多光线）",
    "IMG_9569": "工位桌椅（结构简单、平面直角多）",
    "IMG_9579": "走廊（长而直的地面、强透视）",
    "IMG_9581": "玻璃罩船模（反光/高光导致几何歧义）",
    "IMG_9582": "黑色电梯间（低照度/暗部信息少）",
}


PLY_TO_DTYPE = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass
class SharpQualityRow:
    stem: str
    scene: str

    # PLY stats
    sharp_points_total: int
    bbox_diag: float
    z50: float
    z99: float

    nn_intra_p50: float
    nn_intra_p95: float
    nn_intra_p95_norm: float
    outlier_frac: float

    # Projection stats (from depthmaps)
    proj_width: int
    proj_height: int
    proj_coverage_frac: float
    proj_density_mean: float
    proj_density_p95: float
    depth_grad_p95: float

    quality_score: float


def _read_ply_header_binary(fp) -> Tuple[list[tuple[str, int, list[tuple[str, str]]]], str]:
    first = fp.readline()
    if first.strip() != b"ply":
        raise ValueError("Not a PLY")
    fmt = fp.readline().decode("ascii", "replace").strip()
    if "binary_little_endian" not in fmt:
        raise ValueError(f"Not binary_little_endian PLY: {fmt}")

    elements: list[tuple[str, int, list[tuple[str, str]]]] = []
    current: Optional[tuple[str, int, list[tuple[str, str]]]] = None

    while True:
        line = fp.readline()
        if not line:
            raise ValueError("EOF in header")
        s = line.decode("ascii", "replace").strip()
        if not s or s.startswith("comment"):
            continue
        if s == "end_header":
            break
        parts = s.split()
        if parts[0] == "element":
            if current is not None:
                elements.append(current)
            current = (parts[1], int(parts[2]), [])
        elif parts[0] == "property":
            if current is None:
                raise ValueError("property before element")
            if len(parts) != 3:
                raise ValueError(f"Unsupported property: {s}")
            current[2].append((parts[1], parts[2]))  # type: ignore[index]

    if current is not None:
        elements.append(current)

    return elements, fmt


def _dtype_for_props(props: list[tuple[str, str]]) -> np.dtype:
    fields = []
    for ptype, pname in props:
        if ptype not in PLY_TO_DTYPE:
            raise ValueError(f"Unsupported dtype: {ptype}")
        fields.append((pname, "<" + PLY_TO_DTYPE[ptype]))
    return np.dtype(fields)


def load_sharp_xyz(ply_path: Path) -> Tuple[np.ndarray, int]:
    with ply_path.open("rb") as f:
        elements, _ = _read_ply_header_binary(f)
        vertex = None
        for name, count, props in elements:
            if name == "vertex":
                vertex = (count, props)
                break
        if vertex is None:
            raise ValueError("No vertex element")
        vcount, props = vertex
        dtype = _dtype_for_props(props)
        v = np.fromfile(f, dtype=dtype, count=vcount)

    for k in ("x", "y", "z"):
        if k not in (v.dtype.names or []):
            raise ValueError("vertex missing xyz")
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    return xyz, int(vcount)


def _sample_points(xyz: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if n <= 0 or xyz.shape[0] <= n:
        return xyz
    rng = np.random.default_rng(seed)
    idx = rng.choice(xyz.shape[0], size=n, replace=False)
    return xyz[idx]


def _nanpercentile(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def compute_intra_nn_metrics(xyz: np.ndarray, sample_n: int, seed: int = 0) -> Tuple[float, float, float, float]:
    """Intra-cloud NN distance distribution.

    Returns: p50, p95, p95_norm (by bbox diag), outlier_frac.
    """
    pts = _sample_points(xyz, sample_n, seed=seed)

    # bbox diag for normalization
    mn = np.min(pts, axis=0)
    mx = np.max(pts, axis=0)
    diag = float(np.linalg.norm(mx - mn))
    diag = max(diag, 1e-6)

    tree = cKDTree(pts)
    # query k=2: nearest is itself (0), second is nearest neighbor
    d, _ = tree.query(pts, k=2, workers=-1)
    nn = d[:, 1].astype(np.float32)

    p50 = float(np.quantile(nn, 0.50))
    p95 = float(np.quantile(nn, 0.95))
    p95_norm = float(p95 / diag)

    # robust outlier fraction: nn > median + 3*MAD
    med = float(np.median(nn))
    mad = float(np.median(np.abs(nn - med)))
    thr = med + 3.0 * (mad / 0.6745 if mad > 0 else 0.0)
    if thr <= 0:
        outlier = 0.0
    else:
        outlier = float(np.mean(nn > thr))

    return p50, p95, p95_norm, outlier


def load_projection_metrics(depth_dir: Path) -> Tuple[int, int, float, float, float, float]:
    """Read depth_raw.npy + mask.png + density.png and compute projection stats."""
    depth_path = depth_dir / "depth_raw.npy"
    mask_path = depth_dir / "mask.png"
    density_path = depth_dir / "density.png"

    depth = np.load(depth_path).astype(np.float32)

    mask_img = Image.open(mask_path).convert("L")
    mask = np.array(mask_img, dtype=np.uint8)
    valid = mask > 0

    dens_img = Image.open(density_path).convert("L")
    dens_u8 = np.array(dens_img, dtype=np.uint8)
    # density.png is a visualization; still usable as a monotonic proxy.
    dens = dens_u8.astype(np.float32)

    h, w = depth.shape[:2]
    coverage = float(np.mean(valid))

    if np.any(valid):
        dens_valid = dens[valid]
        dens_mean = float(np.mean(dens_valid))
        dens_p95 = float(np.quantile(dens_valid, 0.95))
    else:
        dens_mean = float("nan")
        dens_p95 = float("nan")

    # depth gradient p95 over valid neighbor pairs (avoid invalid edges)
    # Use finite differences; include only diffs where both pixels are valid.
    grad_vals = []
    dv = depth.astype(np.float32)

    # horizontal diffs
    v_left = valid[:, :-1] & valid[:, 1:]
    if np.any(v_left):
        dx = np.abs(dv[:, 1:] - dv[:, :-1])
        grad_vals.append(dx[v_left])

    # vertical diffs
    v_up = valid[:-1, :] & valid[1:, :]
    if np.any(v_up):
        dy = np.abs(dv[1:, :] - dv[:-1, :])
        grad_vals.append(dy[v_up])

    if grad_vals:
        g = np.concatenate(grad_vals, axis=0)
        grad_p95 = float(np.quantile(g, 0.95))
    else:
        grad_p95 = float("nan")

    return int(w), int(h), coverage, dens_mean, dens_p95, grad_p95


def _score_0_100(df: pd.DataFrame) -> pd.Series:
    """Composite score in [0,100] (higher is better).

    Uses min-max normalization over the provided set.
    """

    def mm(x: pd.Series, higher_is_better: bool) -> pd.Series:
        x = x.astype(float)
        lo = float(np.nanmin(x.values))
        hi = float(np.nanmax(x.values))
        if not math.isfinite(lo) or not math.isfinite(hi) or hi - lo < 1e-12:
            return pd.Series(np.full(len(x), 0.5), index=x.index)
        y = (x - lo) / (hi - lo)
        return y if higher_is_better else (1.0 - y)

    # weights tuned for interpretability (projection quality + structure)
    c = mm(df["proj_coverage_frac"], True)
    d = mm(df["proj_density_mean"], True)
    g = mm(df["depth_grad_p95"], False)
    o = mm(df["outlier_frac"], False)
    n = mm(df["nn_intra_p95_norm"], False)

    score = 0.35 * c + 0.20 * d + 0.20 * g + 0.15 * o + 0.10 * n
    return 100.0 * score


def _bar_plot(df: pd.DataFrame, col: str, out: Path, title: str, ascending: bool = False):
    s = df.sort_values(col, ascending=ascending)
    plt.figure(figsize=(10, 4))
    plt.bar(s["stem"], s[col])
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze SHARP point cloud quality across images.")
    ap.add_argument(
        "--sharp-root",
        type=str,
        default="/Users/rufuslee/Documents/GitHub/ml-sharp-main/output",
        help="Folder containing SHARP PLYs (outside workspace is OK)",
    )
    ap.add_argument(
        "--depthmaps-root",
        type=str,
        default="images/sharp_depthmaps",
        help="Folder containing projected depth artifacts (from sharp_ply_depthmaps.py)",
    )
    ap.add_argument("--outdir", type=str, default="images/sharp_pointcloud_quality/analysis")
    ap.add_argument("--stems", type=str, default=",".join(STEMS_DEFAULT))
    ap.add_argument("--max-points", type=int, default=120000, help="Sampling points for intra-NN metrics")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sharp_root = Path(args.sharp_root)
    depthmaps_root = Path(args.depthmaps_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stems = [s.strip() for s in args.stems.split(",") if s.strip()]

    rows: list[SharpQualityRow] = []

    for stem in stems:
        ply_path = sharp_root / f"{stem}.ply"
        depth_dir = depthmaps_root / stem

        xyz, total = load_sharp_xyz(ply_path)

        z = xyz[:, 2].astype(np.float32)
        z50 = float(np.quantile(z, 0.50))
        z99 = float(np.quantile(z, 0.99))

        # bbox diag on a sample (ok for normalization)
        pts_s = _sample_points(xyz, min(args.max_points, xyz.shape[0]), seed=args.seed)
        mn = np.min(pts_s, axis=0)
        mx = np.max(pts_s, axis=0)
        bbox_diag = float(np.linalg.norm(mx - mn))

        nn_p50, nn_p95, nn_p95_norm, outlier = compute_intra_nn_metrics(xyz, sample_n=args.max_points, seed=args.seed)

        pw, ph, coverage, dens_mean, dens_p95, grad_p95 = load_projection_metrics(depth_dir)

        rows.append(
            SharpQualityRow(
                stem=stem,
                scene=SCENE_DESCRIPTIONS.get(stem, ""),
                sharp_points_total=int(total),
                bbox_diag=bbox_diag,
                z50=z50,
                z99=z99,
                nn_intra_p50=nn_p50,
                nn_intra_p95=nn_p95,
                nn_intra_p95_norm=nn_p95_norm,
                outlier_frac=outlier,
                proj_width=pw,
                proj_height=ph,
                proj_coverage_frac=coverage,
                proj_density_mean=dens_mean,
                proj_density_p95=dens_p95,
                depth_grad_p95=grad_p95,
                quality_score=float("nan"),
            )
        )

    df = pd.DataFrame([asdict(r) for r in rows])
    df["quality_score"] = _score_0_100(df)

    # Write CSV
    csv_path = outdir / "sharp_quality_metrics.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Plots
    _bar_plot(df, "quality_score", outdir / "bar_quality_score.png", "SHARP 点云质量综合得分（0-100, higher is better）", ascending=False)
    _bar_plot(df, "proj_coverage_frac", outdir / "bar_proj_coverage.png", "投影覆盖率（valid pixels / total）", ascending=False)
    _bar_plot(df, "proj_density_mean", outdir / "bar_proj_density_mean.png", "投影稠密度（density.png 的均值代理）", ascending=False)
    _bar_plot(df, "depth_grad_p95", outdir / "bar_depth_grad_p95.png", "深度不稳定度（|∇depth| p95，越低越稳）", ascending=True)
    _bar_plot(df, "outlier_frac", outdir / "bar_outlier_frac.png", "点云离群率（intra-NN: nn > median+3*MAD）", ascending=True)

    # Markdown report
    # Ranking by quality_score
    rank = df.sort_values("quality_score", ascending=False).reset_index(drop=True)

    view_cols = [
        "stem",
        "scene",
        "quality_score",
        "proj_coverage_frac",
        "proj_density_mean",
        "depth_grad_p95",
        "outlier_frac",
        "nn_intra_p95_norm",
        "sharp_points_total",
        "z50",
        "z99",
    ]

    view = rank[view_cols].copy()
    view["quality_score"] = view["quality_score"].map(lambda x: f"{x:.1f}")
    for c in ["proj_coverage_frac", "outlier_frac"]:
        view[c] = view[c].map(lambda x: f"{100.0 * float(x):.2f}%")
    for c in ["proj_density_mean", "depth_grad_p95", "nn_intra_p95_norm", "z50", "z99"]:
        view[c] = view[c].map(lambda x: f"{float(x):.4f}" if math.isfinite(float(x)) else "nan")

    table_md = view.to_markdown(index=False)

    best = rank.iloc[0]
    worst = rank.iloc[-1]

    lines: list[str] = []
    lines.append("# SHARP 跨图点云质量分析（仅基于 SHARP 输出）")
    lines.append("")
    lines.append("本报告给出一个可复用的‘点云质量’评价体系，用于在不同输入图之间横向比较 SHARP 输出是否稳定。")
    lines.append("这里的质量不等价于绝对精度，而是‘几何是否自洽、是否覆盖充分、是否存在大量离群/噪声结构’。")
    lines.append("")
    lines.append("## 推荐的评价指标（报告友好）")
    lines.append("")
    lines.append("- `proj_coverage_frac`：投影覆盖率（越高越好）。覆盖率低通常表示点云只覆盖了少量结构/孔洞多。")
    lines.append("- `proj_density_mean`：投影稠密度（越高越好，使用 density.png 的灰度作为单调代理）。稠密度低通常代表结构稀疏、细节不足。")
    lines.append("- `depth_grad_p95`：深度不稳定度（越低越好）。越大表示局部深度跳变/噪声更强。")
    lines.append("- `outlier_frac`：离群率（越低越好）。高离群率常见于反光/低纹理/暗部或远景的伪结构。")
    lines.append("- `quality_score`：综合得分（0–100），用于排序；建议写报告时同时给出 2~3 个原始指标佐证。")
    lines.append("")
    lines.append("## 汇总表（按综合得分排序）")
    lines.append("")
    lines.append(table_md)
    lines.append("")
    lines.append("## 跨图结论（可直接写进报告）")
    lines.append("")
    lines.append(f"- 在这组样本中，SHARP 输出最稳定（综合得分最高）的是 **{best['stem']}**；相对最不稳定的是 **{worst['stem']}**。")
    lines.append("- 经验上：走廊/工位这类‘平面+直角+强结构’的场景更稳定；反光（玻璃罩）或低照度（电梯间）更容易出现离群点与深度不连续。")
    lines.append("- 建议用法：把 `quality_score` 作为快速筛选指标；若要做原因分析，优先看 coverage（是否孔洞多）+ outlier（是否伪结构多）+ grad_p95（是否噪声/跳变多）。")
    lines.append("")
    lines.append("## 图表")
    lines.append("")
    lines.append("- 综合得分：![](bar_quality_score.png)")
    lines.append("- 覆盖率：![](bar_proj_coverage.png)")
    lines.append("- 稠密度：![](bar_proj_density_mean.png)")
    lines.append("- 深度不稳定度：![](bar_depth_grad_p95.png)")
    lines.append("- 离群率：![](bar_outlier_frac.png)")
    lines.append("")

    report_path = outdir / "sharp_quality_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
