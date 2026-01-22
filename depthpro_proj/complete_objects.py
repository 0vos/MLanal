from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

# Reuse SAM2 + helper logic from depth_to_mesh (already in this repo).
import depth_to_mesh as dtm  # type: ignore


def _mask_iou(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    a = a_u8 > 0
    b = b_u8 > 0
    inter = int(np.count_nonzero(a & b))
    if inter <= 0:
        return 0.0
    uni = int(np.count_nonzero(a | b))
    return float(inter) / float(max(uni, 1))


def _nms_masks(masks_u8: list[np.ndarray], scores: list[float], iou_thresh: float) -> list[int]:
    """Return indices to keep after greedy NMS on binary masks."""
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


def _read_obj_simple(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a very simple OBJ: v, vt, f v/vt (triangle faces).

    Assumes (as produced by this repo) that v and vt indices match.
    """
    vertices: List[List[float]] = []
    uvs: List[List[float]] = []
    faces: List[List[int]] = []

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
                tri: List[int] = []
                ok = True
                for p in parts:
                    if "/" in p:
                        v_str = p.split("/")[0]
                    else:
                        v_str = p
                    if not v_str:
                        ok = False
                        break
                    idx = int(v_str)
                    if idx <= 0:
                        ok = False
                        break
                    tri.append(idx - 1)
                if ok:
                    faces.append(tri)

    v = np.asarray(vertices, dtype=np.float32)
    vt = np.asarray(uvs, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)

    if vt.shape[0] == 0:
        vt = np.zeros((v.shape[0], 2), dtype=np.float32)
    elif vt.shape[0] != v.shape[0]:
        # Pad/truncate to keep our "same index" convention.
        vv = np.zeros((v.shape[0], 2), dtype=np.float32)
        n = min(vt.shape[0], v.shape[0])
        vv[:n] = vt[:n]
        vt = vv

    return v, vt, f


def _compact_mesh_same_index(vertices: np.ndarray, uvs: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop non-finite vertices and unused vertices; remap faces.

    Assumes faces index into vertices and UVs follow the same indexing.
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
        f.write("# complete_objects generated\n")
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
        f.write("# complete_objects material\n")
        f.write(f"newmtl {material_name}\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write("d 1.0\n")
        f.write("illum 1\n")
        f.write(f"map_Kd {tex_name}\n")


def _mask_from_uv(vertices_uv: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Return a per-vertex bool mask based on UV->pixel lookup."""
    h, w = mask_u8.shape[:2]
    u = np.clip(vertices_uv[:, 0], 0.0, 1.0)
    v = np.clip(vertices_uv[:, 1], 0.0, 1.0)
    x = np.clip(np.round(u * max(float(w - 1), 1.0)).astype(np.int32), 0, w - 1)
    y = np.clip(np.round((1.0 - v) * max(float(h - 1), 1.0)).astype(np.int32), 0, h - 1)
    return (mask_u8[y, x] > 0)


def _submesh_by_vertex_mask(vertices: np.ndarray, uvs: np.ndarray, faces: np.ndarray, vmask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if faces.size == 0:
        return vertices[:0], uvs[:0], faces[:0]

    keep_face = vmask[faces].all(axis=1)
    faces2 = faces[keep_face]
    if faces2.size == 0:
        return vertices[:0], uvs[:0], faces[:0]

    used = np.unique(faces2.reshape(-1))
    remap = np.full((vertices.shape[0],), -1, dtype=np.int32)
    remap[used] = np.arange(int(used.size), dtype=np.int32)
    v2 = vertices[used]
    uv2 = uvs[used]
    f2 = remap[faces2]
    return v2.astype(np.float32), uv2.astype(np.float32), f2.astype(np.int32)


def _boundary_directed_edges(faces: np.ndarray) -> np.ndarray:
    a = faces[:, 0]
    b = faces[:, 1]
    c = faces[:, 2]
    directed = np.stack(
        [
            np.stack([a, b], axis=1),
            np.stack([b, c], axis=1),
            np.stack([c, a], axis=1),
        ],
        axis=0,
    ).reshape(-1, 2)

    und = np.sort(directed, axis=1)
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


def _extract_loops(edges: np.ndarray) -> list[list[int]]:
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


def _close_submesh(vertices: np.ndarray, uvs: np.ndarray, faces: np.ndarray, thickness: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Close a (front) submesh into a watertight solid by adding back + side walls."""
    if faces.size == 0:
        return vertices, uvs, faces

    z = vertices[:, 2]
    z0 = float(np.nanmin(z[np.isfinite(z)]))
    back_z = z0 - float(thickness)

    n_front = int(vertices.shape[0])
    v_back = vertices.copy()
    v_back[:, 2] = back_z

    uv_back = uvs.copy()

    v2 = np.concatenate([vertices, v_back], axis=0)
    uv2 = np.concatenate([uvs, uv_back], axis=0)

    f_back = faces[:, ::-1] + n_front

    bedges = _boundary_directed_edges(faces)
    side_faces = []
    for e in bedges:
        a, b = int(e[0]), int(e[1])
        a2 = a + n_front
        b2 = b + n_front
        side_faces.append((a, b, b2))
        side_faces.append((a, b2, a2))

    if side_faces:
        f_side = np.asarray(side_faces, dtype=np.int32)
        f2 = np.concatenate([faces, f_back, f_side], axis=0)
    else:
        f2 = np.concatenate([faces, f_back], axis=0)

    return v2.astype(np.float32), uv2.astype(np.float32), f2.astype(np.int32)


def _classify_crop_imagenet(pil_rgb: Image.Image) -> Dict[str, Any]:
    """Cheap local classification (ImageNet) to get a coarse object guess.

    This is not perfect for indoor objects, but helps drive heuristics/logging.
    """
    try:
        import torch
        import torchvision
        from torchvision.transforms import v2 as T
    except Exception:
        return {"ok": False, "error": "torch/torchvision not available"}

    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"

    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights).to(device)
    model.eval()

    preprocess = weights.transforms()
    x = preprocess(pil_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        topk = torch.topk(prob, k=5)

    cats = weights.meta.get("categories", [])
    out = []
    for score, idx in zip(topk.values.detach().cpu().numpy().tolist(), topk.indices.detach().cpu().numpy().tolist()):
        label = cats[int(idx)] if int(idx) < len(cats) else str(int(idx))
        out.append({"label": label, "p": float(score)})

    return {"ok": True, "device": device, "top5": out}


def complete_objects(
    image_path: Path,
    mesh_obj: Path,
    out_dir: Path,
    sam2_model_id: str,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_thresh: float,
    min_mask_region_area: int,
    min_area: int,
    max_area_frac: float,
    thickness_m: float,
    classify: bool,
    keep_topk: int,
    nms_iou: float,
    min_mesh_faces: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = Image.open(image_path).convert("RGB")
    scene_np = np.array(scene, dtype=np.uint8)
    h, w = scene_np.shape[:2]

    v, vt, f = _read_obj_simple(mesh_obj)
    if f.size == 0:
        raise RuntimeError("Input mesh has no faces")

    # depth_to_mesh meshes can contain NaN placeholder vertices; compact to avoid
    # propagating NaNs into submeshes/exports.
    v, vt, f = _compact_mesh_same_index(v, vt, f)
    if f.size == 0:
        raise RuntimeError("Input mesh has no valid faces after compaction")

    # Run SAM2 auto masks and build an instance id map
    masks = dtm._sam2_auto_masks(  # type: ignore[attr-defined]
        scene_np,
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

    # Save instances meta
    (out_dir / "sam2_instances.json").write_text(
        json.dumps({"model_id": str(sam2_model_id), "instances": inst_meta}, indent=2),
        encoding="utf-8",
    )

    # Build candidate masks and apply NMS to remove redundant/overlapping fragments.
    cand: list[dict[str, Any]] = []
    unique_ids = np.unique(inst_id)
    unique_ids = unique_ids[unique_ids >= 0]
    for iid in unique_ids.tolist():
        iid = int(iid)
        mask = (inst_id == iid).astype(np.uint8) * 255
        area = int(np.count_nonzero(mask))
        if area < int(min_area):
            continue
        cand.append({"iid": iid, "mask": mask, "area": area})

    if cand:
        keep_idx = _nms_masks(
            masks_u8=[c["mask"] for c in cand],
            scores=[float(c["area"]) for c in cand],
            iou_thresh=float(nms_iou),
        )
        cand = [cand[i] for i in keep_idx]

    # Keep top-K largest objects (helps ignore many tiny irrelevant segments)
    cand.sort(key=lambda x: int(x["area"]), reverse=True)
    if int(keep_topk) > 0:
        cand = cand[: int(keep_topk)]

    # For each instance, extract and close a submesh.
    results: List[Dict[str, Any]] = []

    for c in cand:
        iid = int(c["iid"])
        mask = c["mask"]
        area = int(c["area"])

        vmask = _mask_from_uv(vt, mask)
        v_sub, uv_sub, f_sub = _submesh_by_vertex_mask(v, vt, f, vmask)
        if f_sub.size == 0 or v_sub.shape[0] < 50 or int(f_sub.shape[0]) < int(min_mesh_faces):
            continue

        # Close into a watertight solid
        v_closed, uv_closed, f_closed = _close_submesh(v_sub, uv_sub, f_sub, thickness=float(thickness_m))

        obj_dir = out_dir / f"obj_{iid:03d}"
        obj_dir.mkdir(parents=True, exist_ok=True)

        # Save instance mask for debugging
        Image.fromarray(mask, mode="L").save(obj_dir / "mask.png")

        # Reuse scene texture (front will look correct; sides/back are best-effort)
        tex_path = obj_dir / "albedo.png"
        scene.save(tex_path)

        mtl_path = obj_dir / "mesh_closed.mtl"
        obj_path = obj_dir / "mesh_closed.obj"
        _write_mtl(mtl_path, material_name="mat0", tex_name=tex_path.name)
        _write_obj_same_index(obj_path, mtl_name=mtl_path.name, material_name="mat0", vertices=v_closed, uvs=uv_closed, faces=f_closed)

        # Optional coarse classification
        cls: Dict[str, Any] = {}
        if classify:
            ys, xs = np.nonzero(mask > 0)
            if ys.size > 0:
                y0, y1 = int(ys.min()), int(ys.max())
                x0, x1 = int(xs.min()), int(xs.max())
                pad = 8
                y0 = max(y0 - pad, 0)
                x0 = max(x0 - pad, 0)
                y1 = min(y1 + pad, h - 1)
                x1 = min(x1 + pad, w - 1)
                crop = scene.crop((x0, y0, x1 + 1, y1 + 1))
                cls = _classify_crop_imagenet(crop)
                (obj_dir / "classification.json").write_text(json.dumps(cls, indent=2), encoding="utf-8")

        results.append(
            {
                "instance_id": iid,
                "mask_area_px": area,
                "out_dir": str(obj_dir),
                "obj": str(obj_path),
                "mtl": str(mtl_path),
                "classification": cls,
            }
        )

    summary_path = out_dir / "complete_objects_summary.json"
    summary_path.write_text(json.dumps({"mesh": str(mesh_obj), "image": str(image_path), "objects": results}, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Object-first completion (local): SAM2 instance masks -> extract per-object submesh from the generated mesh -> "
            "close each object into a watertight solid.\n\n"
            "This is a practical local baseline: it makes each object closed (no holes).\n"
            "Later we can upgrade geometry completion (hallucinate hidden backs) with heavier generative models."
        )
    )
    ap.add_argument("--image", type=Path, required=True, help="Scene RGB image (e.g. mesh_*/scene.png or albedo.png)")
    ap.add_argument("--mesh", type=Path, required=True, help="Input mesh.obj (from depth_to_mesh.py)")
    ap.add_argument("--out", type=Path, required=True, help="Output directory")

    ap.add_argument("--sam2-model-id", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--sam2-points-per-side", type=int, default=32)
    ap.add_argument("--sam2-pred-iou-thresh", type=float, default=0.85)
    ap.add_argument("--sam2-stability-thresh", type=float, default=0.95)
    ap.add_argument("--sam2-min-mask-region-area", type=int, default=100)
    ap.add_argument("--sam2-min-area", type=int, default=4000)
    ap.add_argument("--sam2-max-area-frac", type=float, default=0.35)

    ap.add_argument("--thickness-m", type=float, default=0.03, help="Extrusion thickness for closed solids (plane mode units are meters)")
    ap.add_argument("--classify", action="store_true", help="Run a coarse ImageNet classifier per object crop (local ResNet50)")

    ap.add_argument("--keep-topk", type=int, default=12, help="Keep only the top-K largest SAM2 instances (0 = keep all)")
    ap.add_argument("--nms-iou", type=float, default=0.65, help="Mask NMS IoU threshold for suppressing redundant overlapping instances")
    ap.add_argument("--min-mesh-faces", type=int, default=300, help="Drop extracted submeshes with fewer than this many faces")

    args = ap.parse_args()

    summary_path = complete_objects(
        image_path=args.image,
        mesh_obj=args.mesh,
        out_dir=args.out,
        sam2_model_id=str(args.sam2_model_id),
        points_per_side=int(args.sam2_points_per_side),
        pred_iou_thresh=float(args.sam2_pred_iou_thresh),
        stability_thresh=float(args.sam2_stability_thresh),
        min_mask_region_area=int(args.sam2_min_mask_region_area),
        min_area=int(args.sam2_min_area),
        max_area_frac=float(args.sam2_max_area_frac),
        thickness_m=float(args.thickness_m),
        classify=bool(args.classify),
        keep_topk=int(args.keep_topk),
        nms_iou=float(args.nms_iou),
        min_mesh_faces=int(args.min_mesh_faces),
    )

    print("Wrote:")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
