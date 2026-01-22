from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from PIL import Image

import depth_to_mesh as dtm  # type: ignore


@dataclass(frozen=True)
class View:
    idx: int
    yaw_deg: float


def _torch_device(device: str):
    import torch

    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _canonical_device(device: str) -> str:
    device = str(device).strip().lower()
    if device in ("auto", "cpu", "cuda", "mps"):
        return device
    return "auto"


def _resize_rgb(rgb_u8: np.ndarray, max_size: int) -> Tuple[np.ndarray, float]:
    if int(max_size) <= 0:
        return rgb_u8, 1.0
    h, w = rgb_u8.shape[:2]
    m = max(h, w)
    if m <= int(max_size):
        return rgb_u8, 1.0
    scale = float(max_size) / float(m)
    w1 = int(round(w * scale))
    h1 = int(round(h * scale))
    im = Image.fromarray(rgb_u8, mode="RGB").resize((w1, h1), resample=Image.Resampling.LANCZOS)
    return np.array(im, dtype=np.uint8), scale


def _intrinsics_from_meta(meta: Dict[str, Any]) -> np.ndarray:
    intr = meta.get("intrinsics")
    if not isinstance(intr, dict):
        raise ValueError("meta.json missing intrinsics")
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


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


def _sam2_instances(
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
) -> list[dict[str, Any]]:
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

    # Attach meta for debug
    for c in cand:
        iid = int(c["instance_id"])
        c["meta"] = next((m for m in inst_meta if int(m.get("instance_id", -1)) == iid), None)

    return cand


def _union_masks(instances: list[dict[str, Any]], h: int, w: int, max_objects: int) -> np.ndarray:
    u = np.zeros((h, w), dtype=np.uint8)
    for i, inst in enumerate(instances[: int(max_objects)]):
        m = inst.get("mask")
        if m is None:
            continue
        mu = np.asarray(m, dtype=np.uint8)
        if mu.shape[:2] != (h, w):
            mu = np.array(Image.fromarray(mu).resize((w, h), resample=Image.Resampling.NEAREST), dtype=np.uint8)
        u = np.maximum(u, mu)
    return u


def _composite_object(rgb_u8: np.ndarray, mask_u8: np.ndarray, bg: Tuple[int, int, int]) -> np.ndarray:
    out = np.full_like(rgb_u8, np.array(bg, dtype=np.uint8))
    keep = mask_u8 > 0
    out[keep] = rgb_u8[keep]
    return out


def _predict_depth(rgb_u8: np.ndarray, depth_device: str, f_px: Optional[float]) -> np.ndarray:
    depth, _ = dtm._predict_depthpro_depth(rgb_u8, device=_canonical_device(str(depth_device)), f_px=f_px)  # type: ignore[attr-defined]
    return depth.astype(np.float32)


def _to_o3d_intrinsics(k: np.ndarray, w: int, h: int) -> o3d.camera.PinholeCameraIntrinsic:
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    intr = o3d.camera.PinholeCameraIntrinsic(int(w), int(h), fx, fy, cx, cy)
    return intr


def _pcd_from_depth_rgb(depth_m: np.ndarray, rgb_u8: np.ndarray, intr: o3d.camera.PinholeCameraIntrinsic, depth_trunc: float) -> o3d.geometry.PointCloud:
    color = o3d.geometry.Image(rgb_u8)
    depth = o3d.geometry.Image(depth_m.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=1.0,
        depth_trunc=float(depth_trunc),
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
    return pcd


def _rot_y(yaw_rad: float) -> np.ndarray:
    c = float(math.cos(yaw_rad))
    s = float(math.sin(yaw_rad))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _register_to_ref(
    pcd: o3d.geometry.PointCloud,
    ref: o3d.geometry.PointCloud,
    init: np.ndarray,
    voxel: float,
) -> np.ndarray:
    # Downsample
    p = pcd.voxel_down_sample(float(voxel))
    r = ref.voxel_down_sample(float(voxel))
    p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(voxel * 2.5), max_nn=30))
    r.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(voxel * 2.5), max_nn=30))

    reg = o3d.pipelines.registration.registration_icp(
        p,
        r,
        max_correspondence_distance=float(voxel * 4.0),
        init=init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )
    return np.asarray(reg.transformation, dtype=np.float64)


def _svd_generate_frames(
    rgb_u8: np.ndarray,
    num_frames: int,
    model_id: str,
    device: str,
    seed: int,
) -> list[np.ndarray]:
    import torch
    from diffusers import StableVideoDiffusionPipeline

    device_t = _torch_device(_canonical_device(device))
    dtype = torch.float16 if device_t.type in ("cuda", "mps") else torch.float32

    pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device_t)

    # SVD expects PIL image
    im = Image.fromarray(rgb_u8, mode="RGB")
    generator = torch.Generator(device=device_t)
    generator.manual_seed(int(seed))

    out = pipe(image=im, num_frames=int(num_frames), generator=generator)
    frames = out.frames[0]
    return [np.array(f.convert("RGB"), dtype=np.uint8) for f in frames]


def run(
    mesh_dir: Path,
    out_dir: Path,
    mode: str,
    max_objects: int,
    mask_path: Optional[Path],
    # SAM2
    sam2_model_id: str,
    sam2_points_per_side: int,
    sam2_pred_iou_thresh: float,
    sam2_stability_thresh: float,
    sam2_min_mask_region_area: int,
    sam2_min_area: int,
    sam2_max_area_frac: float,
    keep_topk: int,
    nms_iou: float,
    # diffusion
    svd_model_id: str,
    svd_frames: int,
    svd_seed: int,
    svd_device: str,
    bg_r: int,
    bg_g: int,
    bg_b: int,
    # depth
    depth_device: str,
    depth_trunc: float,
    view_max_size: int,
    # fusion
    yaw_max_deg: float,
    icp_voxel: float,
    tsdf_voxel: float,
    tsdf_trunc: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_path = mesh_dir / "scene.png"
    meta_path = mesh_dir / "meta.json"
    if not scene_path.exists() or not meta_path.exists():
        raise FileNotFoundError("mesh_dir must contain scene.png and meta.json")

    rgb0 = np.array(Image.open(scene_path).convert("RGB"), dtype=np.uint8)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    k0 = _intrinsics_from_meta(meta)

    h0, w0 = rgb0.shape[:2]

    if mode == "union":
        if mask_path is not None:
            mp = Path(mask_path)
            m = np.array(Image.open(mp).convert("L"), dtype=np.uint8)
            if m.shape[:2] != (h0, w0):
                m = np.array(Image.fromarray(m, mode="L").resize((w0, h0), resample=Image.Resampling.NEAREST), dtype=np.uint8)
            mask = (m > 0).astype(np.uint8) * 255
            Image.fromarray(mask, mode="L").save(out_dir / "mask_union.png")
        else:
            # segmentation (SAM2)
            inst = _sam2_instances(
                rgb_u8=rgb0,
                sam2_model_id=str(sam2_model_id),
                points_per_side=int(sam2_points_per_side),
                pred_iou_thresh=float(sam2_pred_iou_thresh),
                stability_thresh=float(sam2_stability_thresh),
                min_mask_region_area=int(sam2_min_mask_region_area),
                min_area=int(sam2_min_area),
                max_area_frac=float(sam2_max_area_frac),
                keep_topk=int(keep_topk),
                nms_iou=float(nms_iou),
            )

            (out_dir / "sam2").mkdir(parents=True, exist_ok=True)
            (out_dir / "sam2" / "instances.json").write_text(
                json.dumps([{k: v for k, v in c.items() if k != "mask"} for c in inst], indent=2),
                encoding="utf-8",
            )
            for c in inst:
                Image.fromarray(np.asarray(c["mask"], dtype=np.uint8), mode="L").save(
                    out_dir / "sam2" / f"mask_{int(c['instance_id']):03d}.png"
                )

            if not inst:
                raise RuntimeError(
                    "No SAM2 instances. Try increasing --sam2-points-per-side or lowering thresholds, or pass --mask."
                )

            mask = _union_masks(inst, h=h0, w=w0, max_objects=int(max_objects))
            Image.fromarray(mask, mode="L").save(out_dir / "mask_union.png")

        obj_rgb = _composite_object(rgb0, mask_u8=mask, bg=(int(bg_r), int(bg_g), int(bg_b)))
        Image.fromarray(obj_rgb, mode="RGB").save(out_dir / "object_rgb.png")
    else:
        raise ValueError("mode must be: union")

    # diffusion novel views
    frames = _svd_generate_frames(
        obj_rgb,
        num_frames=int(svd_frames),
        model_id=str(svd_model_id),
        device=str(svd_device),
        seed=int(svd_seed),
    )

    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr, mode="RGB").save(out_dir / "frames" / f"frame_{i:03d}.png")

    # build reference point cloud from frame 0 depth
    fr0 = frames[0]
    fr0_small, scale0 = _resize_rgb(fr0, max_size=int(view_max_size))
    k_small = dtm._scale_intrinsics(k0, scale_x=float(scale0), scale_y=float(scale0))  # type: ignore[attr-defined]
    depth0 = _predict_depth(fr0_small, depth_device=str(depth_device), f_px=float(k_small[0, 0]))

    if depth0.shape[:2] != fr0_small.shape[:2]:
        depth0 = np.array(Image.fromarray(depth0).resize((fr0_small.shape[1], fr0_small.shape[0]), resample=Image.Resampling.BILINEAR), dtype=np.float32)

    intr0 = _to_o3d_intrinsics(k_small, w=int(fr0_small.shape[1]), h=int(fr0_small.shape[0]))
    ref = _pcd_from_depth_rgb(depth0, fr0_small, intr0, depth_trunc=float(depth_trunc))

    # TSDF fusion
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(tsdf_voxel),
        sdf_trunc=float(tsdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    def integrate_view(rgb_u8: np.ndarray, depth_m: np.ndarray, intr: o3d.camera.PinholeCameraIntrinsic, T_cw: np.ndarray) -> None:
        color = o3d.geometry.Image(rgb_u8)
        depth = o3d.geometry.Image(depth_m.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1.0,
            depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intr, T_cw)

    # frame0 pose
    T0 = np.eye(4, dtype=np.float64)
    integrate_view(fr0_small, depth0, intr0, T0)

    n = len(frames)
    yaw_max = float(yaw_max_deg)

    poses: list[dict[str, Any]] = [{"frame": 0, "yaw_deg": 0.0, "T_cw": T0.reshape((16,), order="F").tolist()}]

    for i in range(1, n):
        fr = frames[i]
        fr_small, scale = _resize_rgb(fr, max_size=int(view_max_size))
        k_i = dtm._scale_intrinsics(k0, scale_x=float(scale), scale_y=float(scale))  # type: ignore[attr-defined]
        depth_i = _predict_depth(fr_small, depth_device=str(depth_device), f_px=float(k_i[0, 0]))
        if depth_i.shape[:2] != fr_small.shape[:2]:
            depth_i = np.array(Image.fromarray(depth_i).resize((fr_small.shape[1], fr_small.shape[0]), resample=Image.Resampling.BILINEAR), dtype=np.float32)

        intr_i = _to_o3d_intrinsics(k_i, w=int(fr_small.shape[1]), h=int(fr_small.shape[0]))
        pcd_i = _pcd_from_depth_rgb(depth_i, fr_small, intr_i, depth_trunc=float(depth_trunc))

        # initial guess rotation around Y
        yaw = (-yaw_max + (2.0 * yaw_max) * (float(i) / float(max(n - 1, 1))))
        R = _rot_y(math.radians(float(yaw)))
        init = np.eye(4, dtype=np.float64)
        init[:3, :3] = R

        T = _register_to_ref(pcd_i, ref=ref, init=init, voxel=float(icp_voxel))

        integrate_view(fr_small, depth_i, intr_i, T)
        poses.append({"frame": i, "yaw_deg": float(yaw), "T_cw": T.reshape((16,), order="F").tolist()})

        np.save(out_dir / "frames" / f"frame_{i:03d}_depth.npy", depth_i.astype(np.float32))

    (out_dir / "poses.json").write_text(json.dumps(poses, indent=2), encoding="utf-8")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(str(out_dir / "mesh_tsdf.ply"), mesh, write_ascii=False)
    o3d.io.write_triangle_mesh(str(out_dir / "mesh_tsdf.obj"), mesh, write_ascii=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diffusion novel-view + DepthPro + TSDF fusion (Rodin-ish direction, local baseline)")
    ap.add_argument("--mesh-dir", required=True, type=Path, help="depth_to_mesh output directory")
    ap.add_argument("--out", required=True, type=Path)

    ap.add_argument("--mode", default="union", choices=["union"], help="Current baseline processes union of many instance masks")
    ap.add_argument("--max-objects", type=int, default=30)
    ap.add_argument("--mask", type=Path, default=None, help="Optional binary mask image; if set, bypasses SAM2")

    # SAM2
    ap.add_argument("--sam2-model-id", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--sam2-points-per-side", type=int, default=64)
    ap.add_argument("--sam2-pred-iou-thresh", type=float, default=0.75)
    ap.add_argument("--sam2-stability-thresh", type=float, default=0.90)
    ap.add_argument("--sam2-min-mask-region-area", type=int, default=80)
    ap.add_argument("--sam2-min-area", type=int, default=300)
    ap.add_argument("--sam2-max-area-frac", type=float, default=0.80)
    ap.add_argument("--keep-topk", type=int, default=60)
    ap.add_argument("--nms-iou", type=float, default=0.70)

    # diffusion (novel views)
    ap.add_argument("--svd-model-id", default="stabilityai/stable-video-diffusion-img2vid")
    ap.add_argument("--svd-frames", type=int, default=14)
    ap.add_argument("--svd-seed", type=int, default=0)
    ap.add_argument("--svd-device", default="auto", help="auto|cpu|mps|cuda")
    ap.add_argument("--bg-r", type=int, default=255)
    ap.add_argument("--bg-g", type=int, default=255)
    ap.add_argument("--bg-b", type=int, default=255)

    # depth
    ap.add_argument("--depth-device", default="auto")
    ap.add_argument("--depth-trunc", type=float, default=10.0)
    ap.add_argument("--view-max-size", type=int, default=768)

    # fusion
    ap.add_argument("--yaw-max-deg", type=float, default=50.0)
    ap.add_argument("--icp-voxel", type=float, default=0.03)
    ap.add_argument("--tsdf-voxel", type=float, default=0.01)
    ap.add_argument("--tsdf-trunc", type=float, default=0.04)

    args = ap.parse_args()

    run(
        mesh_dir=Path(args.mesh_dir),
        out_dir=Path(args.out),
        mode=str(args.mode),
        max_objects=int(args.max_objects),
        mask_path=Path(args.mask) if args.mask is not None else None,
        sam2_model_id=str(args.sam2_model_id),
        sam2_points_per_side=int(args.sam2_points_per_side),
        sam2_pred_iou_thresh=float(args.sam2_pred_iou_thresh),
        sam2_stability_thresh=float(args.sam2_stability_thresh),
        sam2_min_mask_region_area=int(args.sam2_min_mask_region_area),
        sam2_min_area=int(args.sam2_min_area),
        sam2_max_area_frac=float(args.sam2_max_area_frac),
        keep_topk=int(args.keep_topk),
        nms_iou=float(args.nms_iou),
        svd_model_id=str(args.svd_model_id),
        svd_frames=int(args.svd_frames),
        svd_seed=int(args.svd_seed),
        svd_device=str(args.svd_device),
        bg_r=int(args.bg_r),
        bg_g=int(args.bg_g),
        bg_b=int(args.bg_b),
        depth_device=str(args.depth_device),
        depth_trunc=float(args.depth_trunc),
        view_max_size=int(args.view_max_size),
        yaw_max_deg=float(args.yaw_max_deg),
        icp_voxel=float(args.icp_voxel),
        tsdf_voxel=float(args.tsdf_voxel),
        tsdf_trunc=float(args.tsdf_trunc),
    )


if __name__ == "__main__":
    main()
