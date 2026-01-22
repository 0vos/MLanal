#!/usr/bin/env python3
"""Convert SHARP/3DGS-style .ply files into depth maps and auxiliary images.

This is designed for the PLYs produced in `ml-sharp-main/output/*.ply`, which
embed:
- element vertex: x,y,z plus gaussian attributes (f_dc_0..2, opacity, scale, rot...)
- element extrinsic (16 floats), intrinsic (9 floats), image_size (2 uints)

We project vertices into the embedded camera and produce:
- depth_raw.npy: float32 depth in the PLY's units (0 = missing)
- depth_vis.png: depth visualization (percentile-normalized, viridis)
- mask.png: valid depth mask
- density.png: point density per pixel (log scaled)
- color_mean.png: mean projected color per pixel (sigmoid(f_dc) -> RGB)
- hist_depth.png: depth histogram

Also writes a CSV summary across all processed PLYs.

Example:
  /Users/rufuslee/Documents/GitHub/MLanal/mlanalenv/bin/python tools/sharp_ply_depthmaps.py \
    --input /Users/rufuslee/Documents/GitHub/ml-sharp-main/output \
    --outdir images/sharp_depthmaps \
    --pattern "*.ply"
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PLY_TO_DTYPE: Dict[str, str] = {
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
class PlySummary:
    ply_path: str
    vertex_count: int
    width: int
    height: int

    valid_ratio: float
    depth_min: float
    depth_p01: float
    depth_median: float
    depth_mean: float
    depth_p99: float
    depth_max: float

    density_mean: float
    density_max: float

    camera_source: str


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def _read_ply_header(fp) -> Tuple[List[Tuple[str, int, List[Tuple[str, str]]]], int]:
    """Return (elements, header_bytes). elements = [(name,count,[(type,name),...])]."""
    elements: List[Tuple[str, int, List[Tuple[str, str]]]] = []
    current = None
    header_bytes = 0

    first = fp.readline()
    header_bytes += len(first)
    if first.strip() != b"ply":
        raise ValueError("Not a PLY file")

    fmt = fp.readline()
    header_bytes += len(fmt)
    if b"binary_little_endian" not in fmt:
        raise ValueError(f"Unsupported PLY format line: {fmt!r}")

    while True:
        line = fp.readline()
        if not line:
            raise ValueError("Unexpected EOF in header")
        header_bytes += len(line)
        s = line.decode("ascii", "replace").strip()
        if not s or s.startswith("comment"):
            continue
        if s == "end_header":
            break
        parts = s.split()
        if parts[0] == "element":
            if current is not None:
                elements.append(current)
            name = parts[1]
            count = int(parts[2])
            current = (name, count, [])
        elif parts[0] == "property":
            if current is None:
                raise ValueError("property before element")
            # We only support scalar properties: property <type> <name>
            if len(parts) != 3:
                raise ValueError(f"Unsupported property line: {s}")
            ptype, pname = parts[1], parts[2]
            current[2].append((ptype, pname))  # type: ignore[index]
        else:
            # Ignore other header directives if present
            continue

    if current is not None:
        elements.append(current)

    return elements, header_bytes


def _element_dtype(props: List[Tuple[str, str]]) -> np.dtype:
    fields = []
    for ptype, pname in props:
        if ptype not in PLY_TO_DTYPE:
            raise ValueError(f"Unsupported PLY property type: {ptype}")
        fields.append((pname, "<" + PLY_TO_DTYPE[ptype]))
    return np.dtype(fields)


def _read_ply_binary(path: Path) -> Dict[str, np.ndarray]:
    """Reads all elements into arrays keyed by element name."""
    out: Dict[str, np.ndarray] = {}
    with path.open("rb") as f:
        elements, _ = _read_ply_header(f)
        for name, count, props in elements:
            dtype = _element_dtype(props)
            arr = np.fromfile(f, dtype=dtype, count=count)
            out[name] = arr
    return out


def _guess_image_size_from_images_dir(ply_path: Path, images_dir: Path) -> tuple[int, int] | None:
    """Try to infer image size from a matching image in images_dir (by stem)."""
    if not images_dir.exists() or not images_dir.is_dir():
        return None

    candidates = []
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"):
        candidates.append(images_dir / f"{ply_path.stem}{ext}")
        candidates.append(images_dir / f"{ply_path.stem}{ext.upper()}")

    for c in candidates:
        if not c.exists():
            continue
        try:
            with Image.open(c) as im:
                w, h = im.size
            return int(w), int(h)
        except Exception:
            continue
    return None


def _default_intrinsics(width: int, height: int) -> np.ndarray:
    """Reasonable pinhole intrinsics when none are available.

    SHARP PLYs we saw use cx=width/2, cy=height/2 and fx~0.75*width.
    """
    fx = 0.75 * float(width)
    fy = 0.75 * float(width)
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def _project_to_depth(
    xyz: np.ndarray,
    K: np.ndarray,
    extrinsic: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (depth(H,W) with inf for missing, mask(H,W), density(H,W))."""

    # world->camera
    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([xyz.astype(np.float32), ones], axis=1)  # Nx4
    cam_h = (extrinsic.astype(np.float32) @ pts_h.T).T
    cam = cam_h[:, :3]

    z = cam[:, 2]
    valid = z > 1e-6
    cam = cam[valid]
    z = z[valid]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    u = (fx * (cam[:, 0] / z) + cx).astype(np.int32)
    v = (fy * (cam[:, 1] / z) + cy).astype(np.int32)

    inb = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[inb]
    v = v[inb]
    z = z[inb].astype(np.float32)

    depth = np.full((height, width), np.inf, dtype=np.float32)
    density = np.zeros((height, width), dtype=np.int32)

    flat_depth = depth.ravel()
    flat_density = density.ravel()
    idx = (v.astype(np.int64) * int(width) + u.astype(np.int64)).astype(np.int64)

    np.minimum.at(flat_depth, idx, z)
    np.add.at(flat_density, idx, 1)

    mask = np.isfinite(depth)
    return depth, mask, density


def _project_color_mean(
    xyz: np.ndarray,
    rgb01: np.ndarray,
    K: np.ndarray,
    extrinsic: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Mean color per pixel (not z-buffered). Returns uint8 HxWx3."""
    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([xyz.astype(np.float32), ones], axis=1)
    cam_h = (extrinsic.astype(np.float32) @ pts_h.T).T
    cam = cam_h[:, :3]
    z = cam[:, 2]

    valid = z > 1e-6
    cam = cam[valid]
    z = z[valid]
    rgb = rgb01[valid]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    u = (fx * (cam[:, 0] / z) + cx).astype(np.int32)
    v = (fy * (cam[:, 1] / z) + cy).astype(np.int32)

    inb = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[inb]
    v = v[inb]
    rgb = rgb[inb]

    idx = (v.astype(np.int64) * int(width) + u.astype(np.int64)).astype(np.int64)

    sums = np.zeros((height * width, 3), dtype=np.float64)
    counts = np.zeros((height * width,), dtype=np.int32)

    np.add.at(sums, idx, rgb.astype(np.float64))
    np.add.at(counts, idx, 1)

    counts_safe = np.maximum(counts, 1)[:, None]
    mean = sums / counts_safe
    mean[counts == 0] = 0.0

    img = (np.clip(mean.reshape(height, width, 3), 0.0, 1.0) * 255.0).astype(np.uint8)
    return img


def _save_depth_vis(depth: np.ndarray, mask: np.ndarray, out_path: Path) -> Tuple[float, float, float, float, float, float]:
    valid = depth[mask]
    if valid.size == 0:
        Image.fromarray(np.zeros(depth.shape, dtype=np.uint8)).save(out_path)
        return 0, 0, 0, 0, 0, 0

    dmin = float(valid.min())
    dmax = float(valid.max())
    p01 = float(np.percentile(valid, 1))
    p99 = float(np.percentile(valid, 99))
    dmed = float(np.median(valid))
    dmean = float(valid.mean())

    lo, hi = p01, p99
    if hi <= lo:
        lo, hi = dmin, dmax
    if hi <= lo:
        lo, hi = dmin, dmin + 1.0

    norm = (depth.copy() - lo) / (hi - lo)
    norm[~mask] = 0.0
    norm = np.clip(norm, 0.0, 1.0)

    cmap = plt.get_cmap("viridis")
    rgba = (cmap(norm) * 255.0).astype(np.uint8)
    rgb = rgba[..., :3]
    Image.fromarray(rgb).save(out_path)

    return dmin, p01, dmed, dmean, p99, dmax


def _save_density_vis(density: np.ndarray, out_path: Path) -> Tuple[float, float]:
    den = density.astype(np.float32)
    dmax = float(den.max()) if den.size else 0.0
    dmean = float(den.mean()) if den.size else 0.0

    if dmax <= 0:
        Image.fromarray(np.zeros(density.shape, dtype=np.uint8)).save(out_path)
        return dmean, dmax

    vis = np.log1p(den) / math.log1p(dmax)
    vis = (np.clip(vis, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(vis).save(out_path)
    return dmean, dmax


def _plot_depth_hist(depth: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    valid = depth[mask]
    plt.figure(figsize=(7, 4))
    if valid.size:
        plt.hist(valid, bins=60)
    plt.title("Depth histogram")
    plt.xlabel("depth")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def process_ply(ply_path: Path, out_root: Path, images_dir: Path) -> PlySummary:
    data = _read_ply_binary(ply_path)

    if "vertex" not in data:
        raise ValueError("PLY has no vertex element")

    v = data["vertex"]
    required = {"x", "y", "z"}
    if not required.issubset(set(v.dtype.names or [])):
        raise ValueError(f"vertex element missing xyz: {v.dtype.names}")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    # Camera blocks
    # NOTE: Some PLYs have headers declaring these elements but the binary payload ends
    # right after vertices (no extrinsic/intrinsic/image_size data). Handle gracefully.
    camera_source_parts: list[str] = []

    extr = np.eye(4, dtype=np.float32)
    K: np.ndarray | None = None
    width: int | None = None
    height: int | None = None

    if "extrinsic" in data and data["extrinsic"].size:
        extr_f = data["extrinsic"]
        field = extr_f.dtype.names[0]
        vals = extr_f[field].astype(np.float32).reshape(-1)
        if vals.size >= 16:
            extr = vals[:16].reshape(4, 4)
            camera_source_parts.append("extrinsic:ply")
        else:
            camera_source_parts.append("extrinsic:fallback(identity)")
    else:
        camera_source_parts.append("extrinsic:fallback(identity)")

    if "intrinsic" in data and data["intrinsic"].size:
        intr_f = data["intrinsic"]
        field = intr_f.dtype.names[0]
        vals = intr_f[field].astype(np.float32).reshape(-1)
        if vals.size >= 9:
            K = vals[:9].reshape(3, 3)
            camera_source_parts.append("intrinsic:ply")

    if "image_size" in data and data["image_size"].size:
        isz = data["image_size"]
        field = isz.dtype.names[0]
        vals = isz[field].astype(np.int64).reshape(-1)
        if vals.size >= 2:
            width, height = int(vals[0]), int(vals[1])
            camera_source_parts.append("image_size:ply")

    if width is None or height is None:
        guessed = _guess_image_size_from_images_dir(ply_path, images_dir)
        if guessed is not None:
            width, height = guessed
            camera_source_parts.append("image_size:images_dir")

    if width is None or height is None:
        width, height = 1024, 768
        camera_source_parts.append("image_size:fallback(1024x768)")

    if K is None:
        K = _default_intrinsics(width, height)
        camera_source_parts.append("intrinsic:fallback(default)")

    # Color from SH DC, if present
    rgb01 = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(set(v.dtype.names or [])):
        fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
        rgb01 = _sigmoid(fdc)

    out_dir = out_root / ply_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    depth, mask, density = _project_to_depth(xyz, K, extr, width, height)

    # Save raw depth (0 = missing)
    depth_raw = depth.copy()
    depth_raw[~mask] = 0.0
    np.save(out_dir / "depth_raw.npy", depth_raw.astype(np.float32))

    # Save mask
    Image.fromarray((mask.astype(np.uint8) * 255)).save(out_dir / "mask.png")

    # Save visualizations
    dmin, p01, dmed, dmean, p99, dmax = _save_depth_vis(depth, mask, out_dir / "depth_vis.png")
    den_mean, den_max = _save_density_vis(density, out_dir / "density.png")

    color_img = _project_color_mean(xyz, rgb01, K, extr, width, height)
    Image.fromarray(color_img).save(out_dir / "color_mean.png")

    _plot_depth_hist(depth, mask, out_dir / "hist_depth.png")

    valid_ratio = float(mask.mean())

    return PlySummary(
        ply_path=str(ply_path),
        vertex_count=int(xyz.shape[0]),
        width=int(width),
        height=int(height),
        valid_ratio=valid_ratio,
        depth_min=dmin,
        depth_p01=p01,
        depth_median=dmed,
        depth_mean=dmean,
        depth_p99=p99,
        depth_max=dmax,
        density_mean=float(den_mean),
        density_max=float(den_max),

        camera_source=";".join(camera_source_parts),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Render depth maps from SHARP PLY point clouds.")
    ap.add_argument("--input", "-i", type=str, required=True, help="Input folder containing .ply files")
    ap.add_argument("--outdir", "-o", type=str, default="images/sharp_depthmaps", help="Output root directory")
    ap.add_argument("--pattern", type=str, default="*.ply", help="Glob pattern inside input (default: *.ply)")
    ap.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Folder containing the original images (used to infer width/height if missing in PLY payload)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Limit number of PLYs (0 = no limit)")
    args = ap.parse_args()

    in_dir = Path(args.input)
    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"Input dir not found: {in_dir}")

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images_dir)

    ply_files = sorted(in_dir.glob(args.pattern))
    if not ply_files:
        raise SystemExit(f"No files matched {args.pattern} under {in_dir}")

    summaries: List[dict] = []
    failures: List[dict] = []
    for i, ply_path in enumerate(ply_files, start=1):
        print(f"[{i}/{len(ply_files)}] {ply_path.name}")
        try:
            s = process_ply(ply_path, out_root, images_dir=images_dir)
            summaries.append(asdict(s))
        except Exception as e:
            print(f"[WARN] Failed: {ply_path} ({e})")
            failures.append({"ply_path": str(ply_path), "error": str(e)})
        if args.limit and len(summaries) >= args.limit:
            break

    df = pd.DataFrame(summaries)
    df.to_csv(out_root / "sharp_ply_depthmaps_summary.csv", index=False, encoding="utf-8")
    print(f"Wrote summary: {out_root / 'sharp_ply_depthmaps_summary.csv'}")

    if failures:
        pd.DataFrame(failures).to_csv(out_root / "sharp_ply_depthmaps_failures.csv", index=False, encoding="utf-8")
        print(f"Wrote failures: {out_root / 'sharp_ply_depthmaps_failures.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
