from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def _read_obj_simple(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a very simple OBJ: v, vt, f v/vt (triangle faces).

    Assumes (as produced by this repo) that v and vt indices match.
    """
    vertices = []
    uvs = []
    faces = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vt "):
                parts = line.split()
                if len(parts) >= 3:
                    uvs.append([float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) != 3:
                    continue
                tri = []
                ok = True
                for p in parts:
                    # formats: v/vt, v/vt/vn, v//vn
                    if "/" in p:
                        v_str = p.split("/")[0]
                    else:
                        v_str = p
                    if not v_str:
                        ok = False
                        break
                    idx = int(v_str)
                    if idx < 0:
                        ok = False
                        break
                    tri.append(idx - 1)  # OBJ is 1-indexed
                if ok:
                    faces.append(tri)

    v = np.asarray(vertices, dtype=np.float32)
    vt = np.asarray(uvs, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)

    if vt.shape[0] == 0:
        # If no UVs, make dummy UVs so we can still write v/vt in output
        vt = np.zeros((v.shape[0], 2), dtype=np.float32)
    return v, vt, f


def _compact_mesh_same_index(vertices: np.ndarray, uvs: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop non-finite vertices and unused vertices; remap faces.

    Assumes faces index into vertices and that UVs follow the same indexing.
    """
    if faces.size == 0 or vertices.size == 0:
        return vertices[:0].astype(np.float32), uvs[:0].astype(np.float32), faces[:0].astype(np.int32)

    finite_v = np.isfinite(vertices).all(axis=1)
    keep_face = finite_v[faces].all(axis=1)
    faces2 = faces[keep_face]
    if faces2.size == 0:
        return vertices[:0].astype(np.float32), uvs[:0].astype(np.float32), faces[:0].astype(np.int32)

    used = np.unique(faces2.reshape(-1))
    remap = np.full((vertices.shape[0],), -1, dtype=np.int32)
    remap[used] = np.arange(int(used.size), dtype=np.int32)

    v2 = vertices[used]
    uv2 = uvs[used] if (uvs.shape[0] == vertices.shape[0]) else np.zeros((v2.shape[0], 2), dtype=np.float32)
    f2 = remap[faces2]
    return v2.astype(np.float32), uv2.astype(np.float32), f2.astype(np.int32)


def _write_obj_same_index(path: Path, mtl_name: str, material_name: str, vertices: np.ndarray, uvs: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# close_mesh generated\n")
        f.write(f"mtllib {mtl_name}\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for vt in uvs:
            f.write(f"vt {vt[0]} {vt[1]}\n")
        f.write(f"usemtl {material_name}\n")
        for tri in faces:
            a, b, c = (int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1)
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")


def _write_mtl(path: Path, material_name: str, tex_name: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# close_mesh material\n")
        f.write(f"newmtl {material_name}\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write("d 1.0\n")
        f.write("illum 1\n")
        f.write(f"map_Kd {tex_name}\n")


def _boundary_directed_edges(faces: np.ndarray) -> np.ndarray:
    """Return directed boundary edges (u->v) based on triangle winding."""
    # Count undirected edges
    a = faces[:, 0]
    b = faces[:, 1]
    c = faces[:, 2]
    directed = np.stack([
        np.stack([a, b], axis=1),
        np.stack([b, c], axis=1),
        np.stack([c, a], axis=1),
    ], axis=0).reshape(-1, 2)

    und = np.sort(directed, axis=1)
    # hash undirected edges
    key = und[:, 0].astype(np.int64) << 32 | und[:, 1].astype(np.int64)
    order = np.argsort(key, kind="mergesort")
    key_s = key[order]
    dir_s = directed[order]

    boundary = []
    i = 0
    while i < key_s.shape[0]:
        j = i + 1
        while j < key_s.shape[0] and key_s[j] == key_s[i]:
            j += 1
        if (j - i) == 1:
            boundary.append(dir_s[i])
        i = j

    if not boundary:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(boundary, dtype=np.int32)


def _extract_loops_from_directed_edges(edges: np.ndarray) -> list[list[int]]:
    """Extract vertex loops from directed boundary edges u->v.

    Assumes (approximately) manifold boundaries where each vertex has out-degree ~1.
    """
    if edges.size == 0:
        return []
    nxt: dict[int, int] = {}
    for u, v in edges.tolist():
        u = int(u)
        v = int(v)
        if u not in nxt:
            nxt[u] = v

    visited: set[int] = set()
    loops: list[list[int]] = []

    for start in list(nxt.keys()):
        if start in visited:
            continue
        cur = start
        loop: list[int] = []
        for _ in range(20000):
            if cur in visited:
                break
            visited.add(cur)
            loop.append(cur)
            if cur not in nxt:
                break
            cur = nxt[cur]
            if cur == start:
                break
        if len(loop) >= 3 and cur == start:
            loops.append(loop)

    return loops


def _poly_area_xy(vertices: np.ndarray, loop: list[int]) -> float:
    pts = vertices[np.asarray(loop, dtype=np.int32), :]
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    # Shoelace
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _cap_hole_loops_front(
    vertices: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
    bedges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cap internal holes on the *front* surface.

    We find all boundary loops, treat the largest-area loop as the outer silhouette,
    and cap the rest using a simple centroid fan. This is conservative and intended
    mainly for patching floor/table holes.
    """
    loops = _extract_loops_from_directed_edges(bedges)
    if len(loops) <= 1:
        return vertices, uvs, faces

    areas = [abs(_poly_area_xy(vertices, loop)) for loop in loops]
    outer_idx = int(np.argmax(np.asarray(areas, dtype=np.float64)))

    new_v = [vertices]
    new_uv = [uvs]
    new_f = [faces]

    for i, loop in enumerate(loops):
        if i == outer_idx:
            continue
        idx = np.asarray(loop, dtype=np.int32)
        pts = vertices[idx]
        if pts.shape[0] < 3:
            continue
        c = np.mean(pts, axis=0).astype(np.float32)
        # UV: average of boundary UVs
        cuv = np.mean(uvs[idx], axis=0).astype(np.float32)

        c_idx = int(vertices.shape[0] + sum(x.shape[0] for x in new_v[1:]))
        new_v.append(c.reshape(1, 3))
        new_uv.append(cuv.reshape(1, 2))

        tris = []
        for j in range(len(loop)):
            a = int(loop[j])
            b = int(loop[(j + 1) % len(loop)])
            tris.append((c_idx, a, b))
        new_f.append(np.asarray(tris, dtype=np.int32))

    v2 = np.concatenate(new_v, axis=0)
    uv2 = np.concatenate(new_uv, axis=0)
    f2 = np.concatenate(new_f, axis=0)
    return v2, uv2, f2


def close_heightfield(
    obj_in: Path,
    out_dir: Path,
    thickness: float,
    tex_name: str = "albedo.png",
    cap_holes: bool = False,
) -> Tuple[Path, Path]:
    v, vt, f = _read_obj_simple(obj_in)
    if f.size == 0:
        raise RuntimeError("No faces found in input mesh")

    # Ensure UV count matches vertices for our writer
    if vt.shape[0] != v.shape[0]:
        if vt.shape[0] == 0:
            vt = np.zeros((v.shape[0], 2), dtype=np.float32)
        else:
            # fallback: pad or truncate
            vv = np.zeros((v.shape[0], 2), dtype=np.float32)
            n = min(vt.shape[0], v.shape[0])
            vv[:n] = vt[:n]
            vt = vv

    # IMPORTANT: depth_to_mesh outputs may contain NaN placeholder vertices.
    # Compact here so downstream (closing/capping) doesn't propagate NaNs.
    v, vt, f = _compact_mesh_same_index(v, vt, f)
    if f.size == 0:
        raise RuntimeError("No valid faces after compaction (all faces referenced non-finite vertices)")

    z = v[:, 2]
    z0 = float(np.nanmin(z[np.isfinite(z)]))
    back_z = z0 - float(thickness)

    n_front = int(v.shape[0])
    v_back = v.copy()
    v_back[:, 2] = back_z

    # Duplicate UVs for back vertices (simple but keeps indexing consistent)
    vt_back = vt.copy()

    v2 = np.concatenate([v, v_back], axis=0)
    vt2 = np.concatenate([vt, vt_back], axis=0)

    # Back faces: reverse winding
    f_back = f[:, ::-1] + n_front

    # Side walls from boundary
    bedges = _boundary_directed_edges(f)
    side_faces = []
    for e in bedges:
        a, b = int(e[0]), int(e[1])
        a2 = a + n_front
        b2 = b + n_front
        # two triangles forming a quad
        side_faces.append((a, b, b2))
        side_faces.append((a, b2, a2))

    # Optionally cap internal holes on the front surface
    if cap_holes:
        v2_front, vt2_front, f_front = _cap_hole_loops_front(vertices=v, uvs=vt, faces=f, bedges=bedges)
        # Update front arrays before duplicating to back.
        v = v2_front
        vt = vt2_front
        f = f_front

        z = v[:, 2]
        z0 = float(np.nanmin(z[np.isfinite(z)]))
        back_z = z0 - float(thickness)
        n_front = int(v.shape[0])
        v_back = v.copy()
        v_back[:, 2] = back_z
        vt_back = vt.copy()
        v2 = np.concatenate([v, v_back], axis=0)
        vt2 = np.concatenate([vt, vt_back], axis=0)
        f_back = f[:, ::-1] + n_front

        bedges = _boundary_directed_edges(f)
        side_faces = []
        for e in bedges:
            a, b = int(e[0]), int(e[1])
            a2 = a + n_front
            b2 = b + n_front
            side_faces.append((a, b, b2))
            side_faces.append((a, b2, a2))

    if side_faces:
        f_side = np.asarray(side_faces, dtype=np.int32)
        f2 = np.concatenate([f, f_back, f_side], axis=0)
    else:
        f2 = np.concatenate([f, f_back], axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    mtl_path = out_dir / "mesh_closed.mtl"
    obj_path = out_dir / "mesh_closed.obj"

    _write_mtl(mtl_path, material_name="mat0", tex_name=tex_name)
    _write_obj_same_index(obj_path, mtl_name=mtl_path.name, material_name="mat0", vertices=v2, uvs=vt2, faces=f2)
    return obj_path, mtl_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Make a depth_to_mesh OBJ watertight (closed) by adding a back surface + side walls.")
    ap.add_argument("--input", type=Path, required=True, help="Input mesh.obj from depth_to_mesh")
    ap.add_argument("--out", type=Path, default=None, help="Output directory (default: sibling folder mesh_closed)")
    ap.add_argument("--thickness", type=float, default=0.03, help="Back shell thickness in mesh units (plane mode is meters)")
    ap.add_argument("--tex", type=str, default="albedo.png", help="Texture filename to reference in the MTL")
    ap.add_argument(
        "--cap-holes",
        action="store_true",
        help="Cap internal holes on the front surface (useful to patch floor/table holes).",
    )

    args = ap.parse_args()
    out_dir = args.out
    if out_dir is None:
        out_dir = args.input.parent / "mesh_closed"

    obj_path, mtl_path = close_heightfield(
        obj_in=args.input,
        out_dir=out_dir,
        thickness=float(args.thickness),
        tex_name=str(args.tex),
        cap_holes=bool(args.cap_holes),
    )
    print("Wrote:")
    print(f"- {obj_path}")
    print(f"- {mtl_path}")


if __name__ == "__main__":
    main()
