from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import depth_to_mesh as dtm  # type: ignore


@dataclass
class ObjMesh:
    vertices: np.ndarray  # (N,3)
    uvs: np.ndarray  # (Nt,2)
    faces_v: np.ndarray  # (F,3)
    faces_vt: np.ndarray  # (F,3)


def _read_obj(path: Path) -> ObjMesh:
    vertices: List[List[float]] = []
    uvs: List[List[float]] = []
    faces_v: List[List[int]] = []
    faces_vt: List[List[int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
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
                tri_v: List[int] = []
                tri_vt: List[int] = []
                ok = True
                for p in parts:
                    toks = p.split("/")
                    if not toks[0]:
                        ok = False
                        break
                    vi = int(toks[0])
                    vti = int(toks[1]) if (len(toks) >= 2 and toks[1]) else vi
                    if vi <= 0 or vti <= 0:
                        ok = False
                        break
                    tri_v.append(vi - 1)
                    tri_vt.append(vti - 1)
                if ok:
                    faces_v.append(tri_v)
                    faces_vt.append(tri_vt)

    v = np.asarray(vertices, dtype=np.float32)
    vt = np.asarray(uvs, dtype=np.float32)
    fv = np.asarray(faces_v, dtype=np.int32)
    fvt = np.asarray(faces_vt, dtype=np.int32)

    if vt.shape[0] == 0:
        vt = np.zeros((v.shape[0], 2), dtype=np.float32)
    if vt.shape[0] != v.shape[0] and vt.shape[0] > 0:
        # Keep the repo convention: same indexing. Pad/truncate.
        vv = np.zeros((v.shape[0], 2), dtype=np.float32)
        n = min(vt.shape[0], v.shape[0])
        vv[:n] = vt[:n]
        vt = vv

    return ObjMesh(vertices=v, uvs=vt, faces_v=fv, faces_vt=fvt)


def _compact(mesh: ObjMesh) -> ObjMesh:
    if mesh.faces_v.size == 0 or mesh.vertices.size == 0:
        return mesh

    finite_v = np.isfinite(mesh.vertices).all(axis=1)
    keep_face = finite_v[mesh.faces_v].all(axis=1)
    fv = mesh.faces_v[keep_face]
    fvt = mesh.faces_vt[keep_face]

    if fv.size == 0:
        return ObjMesh(mesh.vertices[:0], mesh.uvs[:0], fv, fvt)

    used_v = np.unique(fv.reshape(-1))
    remap_v = np.full((mesh.vertices.shape[0],), -1, dtype=np.int32)
    remap_v[used_v] = np.arange(int(used_v.size), dtype=np.int32)

    v2 = mesh.vertices[used_v]
    uv2 = mesh.uvs[used_v] if mesh.uvs.shape[0] == mesh.vertices.shape[0] else np.zeros((v2.shape[0], 2), dtype=np.float32)
    fv2 = remap_v[fv]
    fvt2 = fv2.copy()  # same-index convention

    return ObjMesh(v2.astype(np.float32), uv2.astype(np.float32), fv2.astype(np.int32), fvt2.astype(np.int32))


def _infer_grid_dims_from_uv(uv: np.ndarray) -> Tuple[int, int]:
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must be (N,2)")

    n = int(uv.shape[0])
    if n <= 0:
        raise ValueError("empty uv")

    v0 = float(uv[0, 1])
    tol = 1e-6
    gw = 0
    for i in range(n):
        if abs(float(uv[i, 1]) - v0) < tol:
            gw += 1
        else:
            break
    if gw <= 1:
        raise RuntimeError("Failed to infer grid width from UVs")
    if (n % gw) != 0:
        raise RuntimeError(f"UV count {n} not divisible by inferred gw={gw}")
    gh = n // gw
    return int(gh), int(gw)


def _mask_iou(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    a = a_u8 > 0
    b = b_u8 > 0
    inter = int(np.count_nonzero(a & b))
    if inter <= 0:
        return 0.0
    uni = int(np.count_nonzero(a | b))
    return float(inter) / float(max(uni, 1))


def _nms_masks(masks_u8: list[np.ndarray], scores: list[float], iou_thresh: float) -> list[int]:
    if not masks_u8:
        return []
    order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1]
    keep: list[int] = []
    for idx in order.tolist():
        m = masks_u8[int(idx)]
        ok = True
        for j in keep:
            if _mask_iou(m, masks_u8[int(j)]) >= float(iou_thresh):
                ok = False
                break
        if ok:
            keep.append(int(idx))
    return keep


def _union_object_mask(
    rgb_u8: np.ndarray,
    sam2_model_id: str,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_thresh: float,
    min_mask_region_area: int,
    min_area: int,
    max_area_frac: float,
    keep_topk: int,
    nms_iou: float,
) -> Tuple[np.ndarray, list[dict[str, Any]]]:
    h, w = rgb_u8.shape[:2]
    masks = dtm._sam2_auto_masks(  # type: ignore[attr-defined]
        rgb_u8,
        model_id=str(sam2_model_id),
        device=dtm._torch_auto_device(),  # type: ignore[attr-defined]
        points_per_side=int(points_per_side),
        pred_iou_thresh=float(pred_iou_thresh),
        stability_score_thresh=float(stability_thresh),
        min_mask_region_area=int(min_mask_region_area),
    )

    inst_id, inst_meta = dtm._instance_id_map_from_sam2_masks(  # type: ignore[attr-defined]
        masks,
        h=int(h),
        w=int(w),
        min_area=int(min_area),
        max_area_frac=float(max_area_frac),
    )

    cand: list[dict[str, Any]] = []
    unique_ids = np.unique(inst_id)
    unique_ids = unique_ids[unique_ids >= 0]
    for iid in unique_ids.tolist():
        iid = int(iid)
        mask = (inst_id == iid).astype(np.uint8) * 255
        area = int(np.count_nonzero(mask))
        if area < int(min_area):
            continue
        cand.append({"instance_id": iid, "mask": mask, "mask_area_px": area})

    # NMS to suppress redundant fragments
    if cand:
        keep_idx = _nms_masks(
            masks_u8=[c["mask"] for c in cand],
            scores=[float(c["mask_area_px"]) for c in cand],
            iou_thresh=float(nms_iou),
        )
        cand = [cand[i] for i in keep_idx]

    cand.sort(key=lambda x: int(x["mask_area_px"]), reverse=True)
    if int(keep_topk) > 0:
        cand = cand[: int(keep_topk)]

    union = np.zeros((h, w), dtype=np.uint8)
    for c in cand:
        union = np.maximum(union, c["mask"].astype(np.uint8))

    # Attach meta for debugging
    kept_meta = []
    for c in cand:
        iid = int(c["instance_id"])
        meta = next((m for m in inst_meta if int(m.get("instance_id", -1)) == iid), None)
        kept_meta.append({"instance_id": iid, "mask_area_px": int(c["mask_area_px"]), "meta": meta})

    return union, kept_meta


def _faces_from_grid_mask(vertices: np.ndarray, uv: np.ndarray, union_mask_u8: np.ndarray) -> np.ndarray:
    # Generate triangles for cells where vertices are finite and inside union mask.
    h_img, w_img = union_mask_u8.shape[:2]
    gh, gw = _infer_grid_dims_from_uv(uv)

    if int(gh) * int(gw) != int(vertices.shape[0]):
        raise RuntimeError("Grid dims do not match vertex count")

    finite = np.isfinite(vertices).all(axis=1)

    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)
    x = np.clip(np.round(u * max(float(w_img - 1), 1.0)).astype(np.int32), 0, w_img - 1)
    y = np.clip(np.round((1.0 - v) * max(float(h_img - 1), 1.0)).astype(np.int32), 0, h_img - 1)
    inside = (union_mask_u8[y, x] > 0)

    faces: List[Tuple[int, int, int]] = []

    for iy in range(gh - 1):
        row = iy * gw
        row2 = (iy + 1) * gw
        for ix in range(gw - 1):
            a = row + ix
            b = a + 1
            c = row2 + ix
            d = c + 1

            if not (finite[a] and finite[b] and finite[c] and finite[d]):
                continue

            # Vote: require at least 2/3 vertices of triangle inside mask to keep
            tri1_inside = int(inside[a]) + int(inside[c]) + int(inside[b])
            tri2_inside = int(inside[b]) + int(inside[c]) + int(inside[d])
            if tri1_inside >= 2:
                faces.append((a, c, b))
            if tri2_inside >= 2:
                faces.append((b, c, d))

    if not faces:
        return np.zeros((0, 3), dtype=np.int32)
    return np.asarray(faces, dtype=np.int32)


def _remove_faces_by_union_mask(mesh: ObjMesh, union_mask_u8: np.ndarray, face_votes: int) -> ObjMesh:
    if mesh.faces_v.size == 0:
        return mesh

    h, w = union_mask_u8.shape[:2]
    vt = mesh.uvs
    fvt = mesh.faces_vt
    uv = vt[fvt]  # (F,3,2)
    u = np.clip(uv[:, :, 0], 0.0, 1.0)
    v = np.clip(uv[:, :, 1], 0.0, 1.0)

    x = np.clip(np.round(u * max(float(w - 1), 1.0)).astype(np.int32), 0, w - 1)
    y = np.clip(np.round((1.0 - v) * max(float(h - 1), 1.0)).astype(np.int32), 0, h - 1)

    inside = (union_mask_u8[y, x] > 0)
    remove = (inside.sum(axis=1) >= int(face_votes))

    keep = ~remove
    fv = mesh.faces_v[keep]
    fvt2 = mesh.faces_vt[keep]

    out = ObjMesh(mesh.vertices, mesh.uvs, fv, fvt2)
    return _compact(out)


def _write_obj(path: Path, mtl_name: str, material_name: str, mesh: ObjMesh, group_name: Optional[str] = None, v_offset: int = 0, vt_offset: int = 0, append: bool = False) -> Tuple[int, int]:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        if not append:
            f.write("# mesh_first_patch generated\n")
            f.write(f"mtllib {mtl_name}\n")
        if group_name is not None:
            f.write(f"g {group_name}\n")
        for v in mesh.vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for vt in mesh.uvs:
            f.write(f"vt {vt[0]} {vt[1]}\n")
        f.write(f"usemtl {material_name}\n")
        for fv, fvt in zip(mesh.faces_v, mesh.faces_vt):
            a = int(fv[0]) + 1 + int(v_offset)
            b = int(fv[1]) + 1 + int(v_offset)
            c = int(fv[2]) + 1 + int(v_offset)
            ta = int(fvt[0]) + 1 + int(vt_offset)
            tb = int(fvt[1]) + 1 + int(vt_offset)
            tc = int(fvt[2]) + 1 + int(vt_offset)
            f.write(f"f {a}/{ta} {b}/{tb} {c}/{tc}\n")

    return v_offset + int(mesh.vertices.shape[0]), vt_offset + int(mesh.uvs.shape[0])


def _write_mtl(path: Path, material_name: str, tex_name: str) -> None:
    path.write_text(
        "".join(
            [
                "# mesh_first_patch material\n",
                f"newmtl {material_name}\n",
                "Ka 1.0 1.0 1.0\n",
                "Kd 1.0 1.0 1.0\n",
                "Ks 0.0 0.0 0.0\n",
                "d 1.0\n",
                "illum 1\n",
                f"map_Kd {tex_name}\n",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Mesh-first patch: union SAM2 object masks, then re-add front faces from the depth_to_mesh grid and merge into the base mesh")
    ap.add_argument("--mesh-dir", required=True, type=Path, help="depth_to_mesh output directory (expects scene.png, mesh.obj, albedo.png)")
    ap.add_argument("--base", type=Path, default=None, help="Base mesh OBJ to patch (default: mesh_closed_cap/mesh_closed.obj if exists)")
    ap.add_argument("--out", required=True, type=Path, help="Output directory")

    # SAM2 parameters (defaults favor keeping more objects)
    ap.add_argument("--sam2-model-id", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--sam2-points-per-side", type=int, default=32)
    ap.add_argument("--sam2-pred-iou-thresh", type=float, default=0.85)
    ap.add_argument("--sam2-stability-thresh", type=float, default=0.95)
    ap.add_argument("--sam2-min-mask-region-area", type=int, default=200)
    ap.add_argument("--sam2-min-area", type=int, default=600)
    ap.add_argument("--sam2-max-area-frac", type=float, default=0.60)
    ap.add_argument("--keep-topk", type=int, default=30)
    ap.add_argument("--nms-iou", type=float, default=0.60)

    ap.add_argument("--remove-from-base", action="store_true", help="Remove base faces covered by the union object mask (reduces z-fighting)")
    ap.add_argument("--face-votes", type=int, default=2)

    args = ap.parse_args()

    mesh_dir = Path(args.mesh_dir)
    scene_path = mesh_dir / "scene.png"
    albedo_path = mesh_dir / "albedo.png"
    mesh_obj_path = mesh_dir / "mesh.obj"

    if not scene_path.exists() or not albedo_path.exists() or not mesh_obj_path.exists():
        raise FileNotFoundError("mesh_dir must contain scene.png, albedo.png, mesh.obj")

    base_obj: Path
    if args.base is not None:
        base_obj = Path(args.base)
    else:
        cand = mesh_dir / "mesh_closed_cap" / "mesh_closed.obj"
        if cand.exists():
            base_obj = cand
        else:
            base_obj = mesh_obj_path

    if not base_obj.exists():
        raise FileNotFoundError(f"Base OBJ not found: {base_obj}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.array(Image.open(scene_path).convert("RGB"), dtype=np.uint8)

    # Union mask over many instances
    union_mask, kept_meta = _union_object_mask(
        rgb_u8=rgb,
        sam2_model_id=str(args.sam2_model_id),
        points_per_side=int(args.sam2_points_per_side),
        pred_iou_thresh=float(args.sam2_pred_iou_thresh),
        stability_thresh=float(args.sam2_stability_thresh),
        min_mask_region_area=int(args.sam2_min_mask_region_area),
        min_area=int(args.sam2_min_area),
        max_area_frac=float(args.sam2_max_area_frac),
        keep_topk=int(args.keep_topk),
        nms_iou=float(args.nms_iou),
    )

    Image.fromarray(union_mask, mode="L").save(out_dir / "union_mask.png")
    (out_dir / "sam2_kept.json").write_text(json.dumps({"kept": kept_meta}, indent=2), encoding="utf-8")

    # Build patch faces from the ORIGINAL grid vertices (mesh.obj contains NaN placeholders)
    raw_mesh = _read_obj(mesh_obj_path)
    uv = raw_mesh.uvs
    verts = raw_mesh.vertices

    faces_patch = _faces_from_grid_mask(verts, uv, union_mask_u8=union_mask)
    patch = ObjMesh(vertices=verts, uvs=uv, faces_v=faces_patch, faces_vt=faces_patch.copy())
    patch = _compact(patch)

    if patch.faces_v.size == 0:
        raise RuntimeError("Patch mesh has no faces. Try relaxing SAM2 thresholds or increasing keep_topk.")

    # Load base mesh
    base = _compact(_read_obj(base_obj))
    if base.faces_v.size == 0:
        raise RuntimeError("Base mesh has no valid faces")

    if bool(args.remove_from_base):
        base = _remove_faces_by_union_mask(base, union_mask_u8=union_mask, face_votes=int(args.face_votes))

    # Write output texture + mtl
    shutil.copyfile(albedo_path, out_dir / "albedo.png")
    mtl_path = out_dir / "mesh_patched.mtl"
    obj_path = out_dir / "mesh_patched.obj"
    material = "mat0"
    _write_mtl(mtl_path, material_name=material, tex_name="albedo.png")

    # Write combined OBJ with 2 groups, one shared material
    v_off, vt_off = _write_obj(obj_path, mtl_name=mtl_path.name, material_name=material, mesh=base, group_name="base", append=False)
    _write_obj(obj_path, mtl_name=mtl_path.name, material_name=material, mesh=patch, group_name="patch", v_offset=v_off, vt_offset=vt_off, append=True)

    # Summary
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "mesh_dir": str(mesh_dir),
                "base_obj": str(base_obj),
                "out_obj": str(obj_path),
                "out_mtl": str(mtl_path),
                "n_base_faces": int(base.faces_v.shape[0]),
                "n_patch_faces": int(patch.faces_v.shape[0]),
                "n_kept_instances": int(len(kept_meta)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
