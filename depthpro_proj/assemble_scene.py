from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class ObjMesh:
    vertices: np.ndarray  # (N,3) float32
    uvs: np.ndarray  # (Nt,2) float32
    faces_v: np.ndarray  # (F,3) int32
    faces_vt: np.ndarray  # (F,3) int32
    face_mtl: List[str]  # len F
    mtllibs: List[str]


def _parse_mtl_first_map_kd(mtl_path: Path) -> Optional[str]:
    if not mtl_path.exists():
        return None
    tex: Optional[str] = None
    with mtl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("map_kd"):
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    tex = parts[1].strip()
                    break
    return tex


def _resolve_path(path_from_json: Path, anchors: List[Path]) -> Path:
    """Resolve a possibly-relative path against a few plausible anchors."""
    if path_from_json.is_absolute():
        return path_from_json
    for a in anchors:
        try_path = (a / path_from_json).resolve()
        if try_path.exists():
            return try_path
    # Fall back to the first anchor for a stable (even if non-existent) path.
    return (anchors[0] / path_from_json).resolve()


def _read_obj(path: Path) -> ObjMesh:
    vertices: List[List[float]] = []
    uvs: List[List[float]] = []
    faces_v: List[List[int]] = []
    faces_vt: List[List[int]] = []
    face_mtl: List[str] = []
    mtllibs: List[str] = []

    cur_mtl = "default"

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("mtllib "):
                mtllibs.append(line.split(maxsplit=1)[1].strip())
            elif line.startswith("usemtl "):
                cur_mtl = line.split(maxsplit=1)[1].strip() or "default"
            elif line.startswith("v "):
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
                tri_v: List[int] = []
                tri_vt: List[int] = []
                ok = True
                for p in parts:
                    # formats: v, v/vt, v/vt/vn, v//vn
                    toks = p.split("/")
                    if len(toks) >= 1 and toks[0]:
                        vi = int(toks[0])
                    else:
                        ok = False
                        break
                    vti: Optional[int] = None
                    if len(toks) >= 2 and toks[1]:
                        vti = int(toks[1])

                    if vi <= 0 or (vti is not None and vti <= 0):
                        ok = False
                        break

                    tri_v.append(vi - 1)
                    tri_vt.append((vti - 1) if vti is not None else (vi - 1))

                if ok:
                    faces_v.append(tri_v)
                    faces_vt.append(tri_vt)
                    face_mtl.append(cur_mtl)

    v = np.asarray(vertices, dtype=np.float32)
    vt = np.asarray(uvs, dtype=np.float32)
    fv = np.asarray(faces_v, dtype=np.int32)
    fvt = np.asarray(faces_vt, dtype=np.int32)

    if vt.shape[0] == 0:
        vt = np.zeros((max(v.shape[0], 0), 2), dtype=np.float32)

    return ObjMesh(vertices=v, uvs=vt, faces_v=fv, faces_vt=fvt, face_mtl=face_mtl, mtllibs=mtllibs)


def _compact_mesh(mesh: ObjMesh) -> ObjMesh:
    if mesh.faces_v.size == 0 or mesh.vertices.size == 0:
        return mesh

    finite_v = np.isfinite(mesh.vertices).all(axis=1)
    keep_face = finite_v[mesh.faces_v].all(axis=1)
    fv = mesh.faces_v[keep_face]
    fvt = mesh.faces_vt[keep_face]
    face_mtl = [m for m, k in zip(mesh.face_mtl, keep_face.tolist()) if k]

    if fv.size == 0:
        return ObjMesh(
            vertices=mesh.vertices[:0],
            uvs=mesh.uvs[:0],
            faces_v=mesh.faces_v[:0],
            faces_vt=mesh.faces_vt[:0],
            face_mtl=[],
            mtllibs=mesh.mtllibs,
        )

    used_v = np.unique(fv.reshape(-1))
    remap_v = np.full((mesh.vertices.shape[0],), -1, dtype=np.int32)
    remap_v[used_v] = np.arange(int(used_v.size), dtype=np.int32)

    v2 = mesh.vertices[used_v]
    fv2 = remap_v[fv]

    # UV compaction (keep only referenced vt indices; if invalid, fall back to v indexing)
    if mesh.uvs.size == 0:
        uv2 = np.zeros((v2.shape[0], 2), dtype=np.float32)
        fvt2 = fv2.copy()
    else:
        valid_vt = (fvt >= 0) & (fvt < mesh.uvs.shape[0])
        if not bool(np.all(valid_vt)):
            # Some faces reference missing UVs; degrade to "same index" UVs when possible.
            uv2 = mesh.uvs[: mesh.vertices.shape[0]]
            if uv2.shape[0] != mesh.vertices.shape[0]:
                uv2 = np.zeros((mesh.vertices.shape[0], 2), dtype=np.float32)
            uv2 = uv2[used_v]
            fvt2 = fv2.copy()
        else:
            used_vt = np.unique(fvt.reshape(-1))
            remap_vt = np.full((mesh.uvs.shape[0],), -1, dtype=np.int32)
            remap_vt[used_vt] = np.arange(int(used_vt.size), dtype=np.int32)
            uv2 = mesh.uvs[used_vt]
            fvt2 = remap_vt[fvt]

    return ObjMesh(vertices=v2.astype(np.float32), uvs=uv2.astype(np.float32), faces_v=fv2.astype(np.int32), faces_vt=fvt2.astype(np.int32), face_mtl=face_mtl, mtllibs=mesh.mtllibs)


def _remove_base_faces_by_masks(base: ObjMesh, masks_u8: List[np.ndarray], image_size: Tuple[int, int], votes: int) -> ObjMesh:
    if base.faces_v.size == 0 or not masks_u8:
        return base

    h, w = int(image_size[0]), int(image_size[1])
    vt = base.uvs
    if vt.size == 0:
        return base

    fvt = base.faces_vt
    uv = vt[fvt]  # (F,3,2)
    u = np.clip(uv[:, :, 0], 0.0, 1.0)
    v = np.clip(uv[:, :, 1], 0.0, 1.0)
    x = np.clip(np.round(u * max(float(w - 1), 1.0)).astype(np.int32), 0, w - 1)
    y = np.clip(np.round((1.0 - v) * max(float(h - 1), 1.0)).astype(np.int32), 0, h - 1)

    remove = np.zeros((base.faces_v.shape[0],), dtype=bool)
    for m in masks_u8:
        if m.shape[0] != h or m.shape[1] != w:
            # Defensive resize: keep nearest-neighbor semantics
            m = np.array(Image.fromarray(m).resize((w, h), resample=Image.NEAREST), dtype=np.uint8)
        inside = (m[y, x] > 0)
        remove |= (inside.sum(axis=1) >= int(votes))

    keep_face = ~remove
    fv = base.faces_v[keep_face]
    fvt = base.faces_vt[keep_face]
    face_mtl = [m for m, k in zip(base.face_mtl, keep_face.tolist()) if k]

    # Compact again so we don't keep unused vertices
    out = ObjMesh(vertices=base.vertices, uvs=base.uvs, faces_v=fv, faces_vt=fvt, face_mtl=face_mtl, mtllibs=base.mtllibs)
    return _compact_mesh(out)


def _write_mtl_block(material_name: str, tex_name: Optional[str]) -> str:
    lines = [
        f"newmtl {material_name}\n",
        "Ka 1.0 1.0 1.0\n",
        "Kd 1.0 1.0 1.0\n",
        "Ks 0.0 0.0 0.0\n",
        "d 1.0\n",
        "illum 1\n",
    ]
    if tex_name:
        lines.append(f"map_Kd {tex_name}\n")
    lines.append("\n")
    return "".join(lines)


def assemble_scene(
    base_obj: Path,
    objects_summary: Path,
    out_dir: Path,
    remove_from_base: bool,
    face_votes: int,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(objects_summary.read_text(encoding="utf-8"))
    anchors = [
        objects_summary.parent,
        objects_summary.parent.parent,
        base_obj.parent,
        base_obj.parent.parent,
        Path.cwd(),
    ]

    image_path = _resolve_path(Path(summary.get("image", "")), anchors)

    base = _compact_mesh(_read_obj(base_obj))
    if base.faces_v.size == 0:
        raise RuntimeError(f"Base mesh has no valid faces: {base_obj}")

    scene_img = Image.open(image_path).convert("RGB")
    h, w = scene_img.size[1], scene_img.size[0]

    # Load object meshes + masks
    objects = summary.get("objects", [])
    obj_meshes: List[Tuple[str, ObjMesh, Path]] = []  # (label, mesh, obj_dir)
    masks: List[np.ndarray] = []

    for o in objects:
        iid = int(o.get("instance_id"))
        obj_path = _resolve_path(Path(o.get("obj")), anchors)
        obj_dir = obj_path.parent
        label = f"obj_{iid:03d}"

        mesh = _compact_mesh(_read_obj(obj_path))
        if mesh.faces_v.size == 0:
            continue
        obj_meshes.append((label, mesh, obj_dir))

        mask_path = obj_dir / "mask.png"
        if mask_path.exists():
            masks.append(np.array(Image.open(mask_path).convert("L"), dtype=np.uint8))

    if remove_from_base and masks:
        base = _remove_base_faces_by_masks(base, masks_u8=masks, image_size=(h, w), votes=int(face_votes))

    # Build output textures + mtl
    out_obj = out_dir / "scene.obj"
    out_mtl = out_dir / "scene.mtl"

    tex_dir = out_dir / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)

    mtl_blocks: List[str] = ["# assemble_scene material\n\n"]

    def add_material_from_source(label: str, src_obj: Path, src_dir: Path) -> str:
        # Find referenced MTL and map_Kd
        src_mesh = _read_obj(src_obj)
        tex_name: Optional[str] = None
        if src_mesh.mtllibs:
            mtl_path = (src_dir / src_mesh.mtllibs[0]).resolve()
            tex = _parse_mtl_first_map_kd(mtl_path)
            if tex:
                # Try a few likely locations (close_mesh mtl often references albedo.png
                # living one directory above the output folder).
                candidates = [
                    (mtl_path.parent / tex).resolve(),
                    (src_dir / tex).resolve(),
                    (src_dir.parent / tex).resolve(),
                    (src_dir.parent.parent / tex).resolve(),
                ]
                tex_path = next((p for p in candidates if p.exists()), None)
                if tex_path is not None:
                    dst_tex_name = f"{label}_{tex_path.name}"
                    shutil.copyfile(tex_path, tex_dir / dst_tex_name)
                    tex_name = f"textures/{dst_tex_name}"

        mat_name = f"mat_{label}"
        mtl_blocks.append(_write_mtl_block(mat_name, tex_name))
        return mat_name

    # Base material
    base_mat = add_material_from_source("base", base_obj, base_obj.parent)

    # Object materials
    obj_mats: Dict[str, str] = {}
    for label, _, obj_dir in obj_meshes:
        obj_path = obj_dir / "mesh_closed.obj"
        if not obj_path.exists():
            # Fall back to whatever summary points to
            obj_path = obj_dir / "mesh.obj"
        obj_mats[label] = add_material_from_source(label, obj_path, obj_dir)

    out_mtl.write_text("".join(mtl_blocks), encoding="utf-8")

    # Write combined OBJ
    with out_obj.open("w", encoding="utf-8") as f:
        f.write("# assemble_scene generated\n")
        f.write(f"mtllib {out_mtl.name}\n")

        v_offset = 0
        vt_offset = 0

        def write_chunk(name: str, mesh: ObjMesh, material: str) -> None:
            nonlocal v_offset, vt_offset
            f.write(f"g {name}\n")
            for v in mesh.vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for vt in mesh.uvs:
                f.write(f"vt {vt[0]} {vt[1]}\n")
            f.write(f"usemtl {material}\n")
            for fv, fvt in zip(mesh.faces_v, mesh.faces_vt):
                a = int(fv[0]) + 1 + v_offset
                b = int(fv[1]) + 1 + v_offset
                c = int(fv[2]) + 1 + v_offset
                ta = int(fvt[0]) + 1 + vt_offset
                tb = int(fvt[1]) + 1 + vt_offset
                tc = int(fvt[2]) + 1 + vt_offset
                f.write(f"f {a}/{ta} {b}/{tb} {c}/{tc}\n")

            v_offset += int(mesh.vertices.shape[0])
            vt_offset += int(mesh.uvs.shape[0])

        write_chunk("base", base, base_mat)
        for label, mesh, _obj_dir in obj_meshes:
            write_chunk(label, mesh, obj_mats.get(label, f"mat_{label}"))

    return out_obj, out_mtl


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge base scene mesh + completed object meshes into one OBJ/MTL bundle")
    ap.add_argument("--base", required=True, type=Path, help="Base scene OBJ (usually a closed/capped mesh)")
    ap.add_argument("--objects-summary", required=True, type=Path, help="complete_objects_summary.json path")
    ap.add_argument("--out", required=True, type=Path, help="Output directory")
    ap.add_argument(
        "--remove-from-base",
        action="store_true",
        help="Remove base faces covered by any object mask (reduces double surfaces)",
    )
    ap.add_argument(
        "--face-votes",
        type=int,
        default=2,
        help="How many of a triangle's 3 UV vertices must land inside a mask to remove it (default: 2)",
    )

    args = ap.parse_args()
    assemble_scene(
        base_obj=args.base,
        objects_summary=args.objects_summary,
        out_dir=args.out,
        remove_from_base=bool(args.remove_from_base),
        face_votes=int(args.face_votes),
    )


if __name__ == "__main__":
    main()
