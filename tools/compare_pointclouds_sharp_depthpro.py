#!/usr/bin/env python3
"""Point-cloud-level comparison: SHARP PLY (3DGS) vs DepthPro camera point cloud.

Inputs:
- SHARP PLY: /Users/rufuslee/Documents/GitHub/ml-sharp-main/output/IMG_xxxx.ply
  (binary_little_endian; vertex has x,y,z plus extra fields)
- DepthPro PLY: /Users/rufuslee/Documents/GitHub/backend/recon_depthpro_raw_ply/IMG_xxxx/scene_points_camera.ply
  (ascii; vertex has x,y,z,r,g,b)

We compare geometry using nearest-neighbor distances (KDTree) and depth (z) distributions.
Because scales can differ, we evaluate three modes:
- raw: SHARP points as-is
- z-median scale: uniform scale so median(z_sharp)*s == median(z_depthpro)
- affine-z: scale+shift on z only (least squares on z distributions) for reference

Outputs:
- images/pointcloud_compare/analysis/pointcloud_metrics.csv
- images/pointcloud_compare/analysis/pointcloud_report.md
- per-stem plots under images/pointcloud_compare/analysis/<IMG_xxxx>/

Run:
  /Users/rufuslee/Documents/GitHub/MLanal/mlanalenv/bin/python tools/compare_pointclouds_sharp_depthpro.py \
    --sharp-root /Users/rufuslee/Documents/GitHub/ml-sharp-main/output \
    --depthpro-root /Users/rufuslee/Documents/GitHub/backend/recon_depthpro_raw_ply \
    --outdir images/pointcloud_compare/analysis
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


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
class PcMetrics:
    stem: str
    scene: str

    sharp_points_total: int
    depthpro_points_total: int

    sharp_points_used: int
    depthpro_points_used: int

    # Depth distribution (z)
    z50_sharp: float
    z50_depthpro: float
    z99_sharp: float
    z99_depthpro: float

    # Scale estimates
    scale_z_median: float

    # NN distances (DepthPro -> SHARP)
    nn_dp_to_sharp_p50_raw: float
    nn_dp_to_sharp_p95_raw: float
    nn_dp_to_sharp_mean_raw: float

    nn_dp_to_sharp_p50_scaled: float
    nn_dp_to_sharp_p95_scaled: float
    nn_dp_to_sharp_mean_scaled: float

    # NN distances (SHARP -> DepthPro)
    nn_sharp_to_dp_p50_raw: float
    nn_sharp_to_dp_p95_raw: float
    nn_sharp_to_dp_mean_raw: float

    nn_sharp_to_dp_p50_scaled: float
    nn_sharp_to_dp_p95_scaled: float
    nn_sharp_to_dp_mean_scaled: float

    # Chamfer-like (mean of means)
    chamfer_mean_raw: float
    chamfer_mean_scaled: float

    # Robust variant: cap both clouds to DepthPro z99 (after scaling)
    chamfer_mean_scaled_zcap: float


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


def load_depthpro_xyz_ascii(ply_path: Path, max_points: int = 0) -> Tuple[np.ndarray, int]:
    """Fast-ish ASCII PLY XYZ loader.

    Uses numpy.fromstring on chunks to avoid per-line parsing overhead.
    Only reads x y z (ignores RGB columns if present).
    """

    # Parse header
    vertex_count = None
    header_lines = 0
    with ply_path.open("rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("Not a PLY")
        fmt = f.readline().decode("ascii", "replace").strip()
        if "ascii" not in fmt:
            raise ValueError(f"Expected ascii ply, got: {fmt}")

        props: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("EOF in header")
            header_lines += 1
            s = line.decode("ascii", "replace").strip()
            if s.startswith("element vertex"):
                vertex_count = int(s.split()[-1])
            elif s.startswith("property"):
                parts = s.split()
                if len(parts) == 3:
                    props.append(parts[2])
            elif s == "end_header":
                break

        if vertex_count is None:
            raise ValueError("No vertex count")

        # Determine which columns correspond to x,y,z
        try:
            ix = props.index("x")
            iy = props.index("y")
            iz = props.index("z")
        except ValueError:
            raise ValueError(f"x/y/z not found in props: {props}")

        # We'll read the remainder as text and parse in chunks.
        # This is still memory heavy for huge files; use max_points sampling by stride.
        data = f.read()

    text = data.decode("ascii", "replace")
    # Convert to float array (all columns)
    arr = np.fromstring(text, sep=" ", dtype=np.float32)

    n_props = len(props)
    if n_props <= 0:
        raise ValueError("No properties")

    n_rows = arr.size // n_props
    arr = arr[: n_rows * n_props].reshape(n_rows, n_props)

    total = int(n_rows)

    if max_points and total > max_points:
        stride = max(1, total // max_points)
        arr = arr[::stride]

    xyz = arr[:, [ix, iy, iz]].astype(np.float32)
    return xyz, int(vertex_count)


def _sample_points(xyz: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if n <= 0 or xyz.shape[0] <= n:
        return xyz
    rng = np.random.default_rng(seed)
    idx = rng.choice(xyz.shape[0], size=n, replace=False)
    return xyz[idx]


def _percentiles_z(xyz: np.ndarray) -> Tuple[float, float]:
    z = xyz[:, 2]
    z = z[np.isfinite(z)]
    if z.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(z, 50)), float(np.percentile(z, 99))


def _nn_stats(query: np.ndarray, ref: np.ndarray) -> Tuple[float, float, float]:
    tree = cKDTree(ref)
    dists, _ = tree.query(query, k=1, workers=-1)
    dists = dists.astype(np.float64)
    return float(np.percentile(dists, 50)), float(np.percentile(dists, 95)), float(dists.mean())


def _write_hist(values: np.ndarray, title: str, out: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=80)
    plt.title(title)
    plt.xlabel("distance")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def compare_one(
    stem: str,
    sharp_root: Path,
    depthpro_root: Path,
    outdir: Path,
    sharp_max: int,
    dp_max: int,
) -> Optional[PcMetrics]:
    sharp_ply = sharp_root / f"{stem}.ply"
    dp_ply = depthpro_root / stem / "scene_points_camera.ply"

    if not sharp_ply.exists() or not dp_ply.exists():
        return None

    sharp_xyz, sharp_total = load_sharp_xyz(sharp_ply)
    dp_xyz, dp_total = load_depthpro_xyz_ascii(dp_ply, max_points=dp_max * 2 if dp_max else 0)

    sharp_xyz = _sample_points(sharp_xyz, sharp_max, seed=0)
    dp_xyz = _sample_points(dp_xyz, dp_max, seed=1)

    # Filter invalid
    sharp_xyz = sharp_xyz[np.isfinite(sharp_xyz).all(axis=1)]
    dp_xyz = dp_xyz[np.isfinite(dp_xyz).all(axis=1)]

    if sharp_xyz.shape[0] < 10_000 or dp_xyz.shape[0] < 10_000:
        return None

    z50_s, z99_s = _percentiles_z(sharp_xyz)
    z50_d, z99_d = _percentiles_z(dp_xyz)

    scale = 1.0
    if np.isfinite(z50_s) and z50_s > 1e-6 and np.isfinite(z50_d):
        scale = z50_d / z50_s

    sharp_scaled = sharp_xyz * float(scale)

    # Robust shared-range comparison: cap to DepthPro's near-to-mid range.
    zcap = float(np.percentile(dp_xyz[:, 2], 99))
    dp_cap = dp_xyz[dp_xyz[:, 2] <= zcap]
    sharp_cap = sharp_scaled[sharp_scaled[:, 2] <= zcap]
    # Keep enough points to make NN stats meaningful
    if dp_cap.shape[0] < 10_000:
        dp_cap = dp_xyz
    if sharp_cap.shape[0] < 10_000:
        sharp_cap = sharp_scaled

    # NN stats
    p50_dp_raw, p95_dp_raw, mean_dp_raw = _nn_stats(dp_xyz, sharp_xyz)
    p50_dp_s, p95_dp_s, mean_dp_s = _nn_stats(dp_xyz, sharp_scaled)

    p50_s_raw, p95_s_raw, mean_s_raw = _nn_stats(sharp_xyz, dp_xyz)
    p50_s_s, p95_s_s, mean_s_s = _nn_stats(sharp_scaled, dp_xyz)

    chamfer_raw = 0.5 * (mean_dp_raw + mean_s_raw)
    chamfer_s = 0.5 * (mean_dp_s + mean_s_s)

    # Chamfer on capped range
    p50_dp_cap, p95_dp_cap, mean_dp_cap = _nn_stats(dp_cap, sharp_cap)
    p50_s_cap, p95_s_cap, mean_s_cap = _nn_stats(sharp_cap, dp_cap)
    chamfer_cap = 0.5 * (mean_dp_cap + mean_s_cap)

    # Plots
    scene_out = outdir / stem
    scene_out.mkdir(parents=True, exist_ok=True)

    # z histograms
    plt.figure(figsize=(7, 4))
    plt.hist(sharp_xyz[:, 2], bins=80, alpha=0.6, label=f"SHARP z (raw)")
    plt.hist(dp_xyz[:, 2], bins=80, alpha=0.6, label=f"DepthPro z")
    plt.title(f"Z distribution - {stem}")
    plt.xlabel("z")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(scene_out / "hist_z_raw.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(sharp_scaled[:, 2], bins=80, alpha=0.6, label=f"SHARP z (scaled x{scale:.3f})")
    plt.hist(dp_xyz[:, 2], bins=80, alpha=0.6, label=f"DepthPro z")
    plt.title(f"Z distribution (scaled) - {stem}")
    plt.xlabel("z")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(scene_out / "hist_z_scaled.png", dpi=200)
    plt.close()

    # NN distance histograms (use dp->sharp distances)
    tree_raw = cKDTree(sharp_xyz)
    d_raw, _ = tree_raw.query(dp_xyz, k=1, workers=-1)
    tree_s = cKDTree(sharp_scaled)
    d_s, _ = tree_s.query(dp_xyz, k=1, workers=-1)
    _write_hist(d_raw.astype(np.float32), f"NN dist (DepthPro -> SHARP raw) - {stem}", scene_out / "hist_nn_dp_to_sharp_raw.png")
    _write_hist(d_s.astype(np.float32), f"NN dist (DepthPro -> SHARP scaled) - {stem}", scene_out / "hist_nn_dp_to_sharp_scaled.png")

    # Also save capped-range NN hist
    tree_cap = cKDTree(sharp_cap)
    d_cap, _ = tree_cap.query(dp_cap, k=1, workers=-1)
    _write_hist(d_cap.astype(np.float32), f"NN dist (DepthPro -> SHARP scaled, z<=dp_p99) - {stem}", scene_out / "hist_nn_dp_to_sharp_scaled_zcap.png")

    return PcMetrics(
        stem=stem,
        scene=SCENE_DESCRIPTIONS.get(stem, ""),
        sharp_points_total=sharp_total,
        depthpro_points_total=dp_total,
        sharp_points_used=int(sharp_xyz.shape[0]),
        depthpro_points_used=int(dp_xyz.shape[0]),
        z50_sharp=float(z50_s),
        z50_depthpro=float(z50_d),
        z99_sharp=float(z99_s),
        z99_depthpro=float(z99_d),
        scale_z_median=float(scale),
        nn_dp_to_sharp_p50_raw=p50_dp_raw,
        nn_dp_to_sharp_p95_raw=p95_dp_raw,
        nn_dp_to_sharp_mean_raw=mean_dp_raw,
        nn_dp_to_sharp_p50_scaled=p50_dp_s,
        nn_dp_to_sharp_p95_scaled=p95_dp_s,
        nn_dp_to_sharp_mean_scaled=mean_dp_s,
        nn_sharp_to_dp_p50_raw=p50_s_raw,
        nn_sharp_to_dp_p95_raw=p95_s_raw,
        nn_sharp_to_dp_mean_raw=mean_s_raw,
        nn_sharp_to_dp_p50_scaled=p50_s_s,
        nn_sharp_to_dp_p95_scaled=p95_s_s,
        nn_sharp_to_dp_mean_scaled=mean_s_s,
        chamfer_mean_raw=chamfer_raw,
        chamfer_mean_scaled=chamfer_s,
        chamfer_mean_scaled_zcap=chamfer_cap,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Point cloud comparison: SHARP vs DepthPro.")
    ap.add_argument("--sharp-root", type=str, required=True, help="Folder containing SHARP IMG_xxxx.ply")
    ap.add_argument("--depthpro-root", type=str, required=True, help="Folder containing DepthPro per-image folders")
    ap.add_argument("--outdir", type=str, default="images/pointcloud_compare/analysis", help="Output directory")
    ap.add_argument("--stems", type=str, default="", help="Comma-separated stems (default: IMG_9546,...IMG_9582)")
    ap.add_argument("--sharp-max", type=int, default=150000, help="Max SHARP points sampled")
    ap.add_argument("--depthpro-max", type=int, default=150000, help="Max DepthPro points sampled")
    args = ap.parse_args()

    sharp_root = Path(args.sharp_root)
    depthpro_root = Path(args.depthpro_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.stems.strip():
        stems = [s.strip() for s in args.stems.split(",") if s.strip()]
    else:
        stems = ["IMG_9546", "IMG_9568", "IMG_9569", "IMG_9579", "IMG_9581", "IMG_9582"]

    rows = []
    for stem in stems:
        print(stem)
        m = compare_one(
            stem,
            sharp_root=sharp_root,
            depthpro_root=depthpro_root,
            outdir=outdir,
            sharp_max=args.sharp_max,
            dp_max=args.depthpro_max,
        )
        if m is None:
            print(f"[WARN] skipped {stem} (missing files or too few points)")
            continue
        rows.append(asdict(m))

    if not rows:
        raise SystemExit("No comparable point clouds")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "pointcloud_metrics.csv", index=False, encoding="utf-8")

    # Ranking plots
    plt.figure(figsize=(10, 4.8))
    tmp = df.sort_values("chamfer_mean_scaled_zcap")
    plt.bar(tmp["stem"], tmp["chamfer_mean_scaled"])
    plt.xticks(rotation=30, ha="right")
    plt.title("Chamfer-like mean NN distance (scaled by z-median)")
    plt.ylabel("distance")
    plt.tight_layout()
    plt.savefig(outdir / "bar_chamfer_scaled.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4.8))
    tmp = df.sort_values("chamfer_mean_scaled_zcap")
    plt.bar(tmp["stem"], tmp["chamfer_mean_scaled_zcap"])
    plt.xticks(rotation=30, ha="right")
    plt.title("Chamfer-like mean NN distance (scaled) with z<=DepthPro p99 cap")
    plt.ylabel("distance")
    plt.tight_layout()
    plt.savefig(outdir / "bar_chamfer_scaled_zcap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4.8))
    tmp = df.sort_values("nn_dp_to_sharp_p95_scaled")
    plt.bar(tmp["stem"], tmp["nn_dp_to_sharp_p95_scaled"])
    plt.xticks(rotation=30, ha="right")
    plt.title("DepthPro → SHARP NN distance p95 (scaled)")
    plt.ylabel("distance")
    plt.tight_layout()
    plt.savefig(outdir / "bar_nn_p95_dp_to_sharp_scaled.png", dpi=200)
    plt.close()

    # Markdown report with write-ready bullets
    report = outdir / "pointcloud_report.md"
    tcols = [
        "stem",
        "scene",
        "scale_z_median",
        "chamfer_mean_scaled",
        "chamfer_mean_scaled_zcap",
        "nn_dp_to_sharp_p50_scaled",
        "nn_dp_to_sharp_p95_scaled",
        "nn_sharp_to_dp_p50_scaled",
        "nn_sharp_to_dp_p95_scaled",
        "z50_sharp",
        "z50_depthpro",
        "z99_sharp",
        "z99_depthpro",
        "sharp_points_used",
        "depthpro_points_used",
    ]
    tab = df[[c for c in tcols if c in df.columns]].copy()
    for c in [
        "scale_z_median",
        "chamfer_mean_scaled",
        "nn_dp_to_sharp_p50_scaled",
        "nn_dp_to_sharp_p95_scaled",
        "nn_sharp_to_dp_p50_scaled",
        "nn_sharp_to_dp_p95_scaled",
        "z50_sharp",
        "z50_depthpro",
        "z99_sharp",
        "z99_depthpro",
    ]:
        if c in tab.columns:
            tab[c] = pd.to_numeric(tab[c], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.4f}")

    best = df.sort_values("chamfer_mean_scaled_zcap").iloc[0]
    worst = df.sort_values("chamfer_mean_scaled_zcap").iloc[-1]

    md = []
    md.append("# SHARP vs DepthPro 点云级别对比")
    md.append("")
    md.append("对齐策略：仅做 z-median 的统一尺度对齐（s = median(z_dp) / median(z_sharp)），再计算双方最近邻距离分布。")
    md.append("注意：这是几何相似度指标，不代表深度绝对准确；但非常适合写进报告做‘点云一致性’对比。")
    md.append("")
    md.append("此外为了避免 SHARP 里‘超远点’导致指标被远尾主导，本报告额外给出一个鲁棒版本：把两边点云都截断到 DepthPro 的 z99（z<=DepthPro p99）再算 chamfer。")
    md.append("")

    md.append("## 汇总表")
    md.append("")
    md.append(tab.to_markdown(index=False))
    md.append("")

    md.append("## 总览图")
    md.append("")
    md.append("- Chamfer-like（scaled）：![](bar_chamfer_scaled.png)")
    md.append("- Chamfer-like（scaled + z<=DepthPro p99）：![](bar_chamfer_scaled_zcap.png)")
    md.append("- p95 最近邻（DepthPro→SHARP, scaled）：![](bar_nn_p95_dp_to_sharp_scaled.png)")
    md.append("")

    md.append("## 可直接写进报告的结论模板")
    md.append("")
    md.append(f"- 在 6 个场景中，点云几何一致性（按 chamfer_mean_scaled_zcap）最好的是 **{best['stem']}**，最差的是 **{worst['stem']}**。")
    md.append(
        "- 经验上，`nn_dp_to_sharp_p95_scaled` 越大，表示 DepthPro 的点在 SHARP 点云里越难找到近邻（SHARP 缺结构/噪声更大）；"
        "`nn_sharp_to_dp_p95_scaled` 越大，表示 SHARP 中有更多点在 DepthPro 里找不到对应近邻（SHARP 多出离群点/远尾/反光伪结构）。"
    )
    md.append(
        "- 远景/大景深（如能看到远处建筑）会显著改变 z 尾部分布；因此建议报告中同时引用 zcap 指标，避免‘远尾差异’把整体一致性淹没。"
    )
    md.append("")

    md.append("## 分场景要点（带数字，便于直接写）")
    md.append("")
    rank = df.sort_values("chamfer_mean_scaled_zcap")
    for _, r in rank.iterrows():
        stem = str(r["stem"])
        md.append(f"### {stem}")
        desc = SCENE_DESCRIPTIONS.get(stem, "")
        if desc:
            md.append(f"- 场景：{desc}")
        md.append(
            f"- 尺度对齐系数：s={float(r['scale_z_median']):.3f}（median z：SHARP={float(r['z50_sharp']):.3f} → DepthPro={float(r['z50_depthpro']):.3f}）"
        )
        md.append(
            f"- Chamfer(mean, scaled)={float(r['chamfer_mean_scaled']):.3f}；Chamfer(mean, scaled+zcap)={float(r['chamfer_mean_scaled_zcap']):.3f}"
        )
        md.append(
            f"- DepthPro→SHARP NN：p50={float(r['nn_dp_to_sharp_p50_scaled']):.3f}, p95={float(r['nn_dp_to_sharp_p95_scaled']):.3f}（p95 越大，说明 SHARP 覆盖/一致性更差）"
        )
        md.append(
            f"- SHARP→DepthPro NN：p50={float(r['nn_sharp_to_dp_p50_scaled']):.3f}, p95={float(r['nn_sharp_to_dp_p95_scaled']):.3f}（p95 越大，说明 SHARP 离群点/伪结构更多）"
        )
        md.append("")

    md.append("## 分场景图")
    md.append("")
    md.append("每个目录包含：")
    md.append("- hist_z_raw.png / hist_z_scaled.png")
    md.append("- hist_nn_dp_to_sharp_raw.png / hist_nn_dp_to_sharp_scaled.png / hist_nn_dp_to_sharp_scaled_zcap.png")
    md.append("")
    for stem in df["stem"].tolist():
        md.append(f"### {stem}")
        desc = SCENE_DESCRIPTIONS.get(stem, "")
        if desc:
            md.append(f"- 场景：{desc}")
        md.append(f"- ![]({stem}/hist_z_scaled.png)")
        md.append(f"- ![]({stem}/hist_nn_dp_to_sharp_scaled.png)")
        md.append(f"- ![]({stem}/hist_nn_dp_to_sharp_scaled_zcap.png)")
        md.append("")

    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
