from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d


def main() -> None:
    ap = argparse.ArgumentParser(description="Quick viewer for PLY/OBJ meshes or point clouds")
    ap.add_argument("path", type=Path)
    args = ap.parse_args()

    p = args.path
    if not p.exists():
        raise FileNotFoundError(p)

    geom = None
    if p.suffix.lower() in {".ply", ".pcd"}:
        geom = o3d.io.read_point_cloud(str(p))
        if geom.is_empty():
            mesh = o3d.io.read_triangle_mesh(str(p))
            if not mesh.is_empty():
                mesh.compute_vertex_normals()
                geom = mesh
    else:
        mesh = o3d.io.read_triangle_mesh(str(p))
        if mesh.is_empty():
            raise RuntimeError(f"Unsupported or unreadable file: {p}")
        mesh.compute_vertex_normals()
        geom = mesh

    o3d.visualization.draw_geometries([geom], window_name=str(p))


if __name__ == "__main__":
    main()
