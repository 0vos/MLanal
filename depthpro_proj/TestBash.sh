python step1_3d_photo.py \
       --input SpatialCapture_2026-01-15T02-30-04.724Z \
       --frame-index 0 \
       --mask centerbox \
       --inpaint opencv \
       --depth-device auto

python view_ply.py SpatialCapture_2026-01-15T02-30-04.724Z/step1_000000/scene_points_world.ply

python fuse_keyframes.py \
       --input SpatialCapture_2026-01-15T02-30-04.724Z \
       --method tsdf \
       --max-frames 3 \
       --every-n 1 \
       --tsdf-voxel-length 0.01 \
       --tsdf-trunc 0.04 \
       --recolor \
       --recolor-occlusion-thresh 0.25 \
       --out fused_tsdf_mesh_3frames_recolor_fix2.ply

# Step2 (fast): DepthPro only -> fit floor plane -> keep only growable regions
# (Uses geometry-based growth filtering; avoids semantic wall/ceiling masks)
python depth_to_mesh.py \
       --input SpatialCapture_2026-01-15T02-30-04.724Z \
       --frame-index 0 \
       --mode plane \
       --image-max-size 768 \
       --grid-step 3 \
       --fit-bottom-frac 0.60 \
       --plane-distance-thresh-m 0.02 \
       --max-height-m 1.50 \
       --tri-max-height-step-m 0.25 \
       --tri-max-edge-m 0.28 \
       --growth-filter \
       --drop-floating \
       --detect-table-plane \
       --table-seed-normal-dot 0 \
       --seed-band-m 0.08 \
       --neighbor-max-height-step-m 0.10 \
       --neighbor-max-depth-step-m 0.25 \
       --neighbor-max-3d-step-m 0.10 \
       --support-depth-pctl 92 --support-depth-margin-m 0.6 \
       --seed-depth-pctl 65 --seed-depth-margin-m 0.30 \
       --keep-nearest-seed-component \
       --suppress-vertical-planes --vertical-plane-min-inliers 8000 \
       --sam2-instances \
       --sam2-max-area-frac 0.30 \
       --stack-contact-dist-m 0.07 --stack-min-height-delta-m 0.02 \
       --salvage-supported-instances \
       --salvage-contact-band-m 0.06 --salvage-min-grid-area 25 --salvage-depth-margin-m 1.4

# Step2 (single image): 任意 JPG/PNG（任意大小，横版/竖版都支持；会自动读取 EXIF 旋转）
# - 建议先用 --image-max-size 控制面数（例如 1024/1280）
# - 如果你想保留原始分辨率：--image-max-size 0

# Step2 (single image, RAW): 纯 DepthPro（不缩放、不做 plane 生长/分割/去背景）
# - 目标：尽量“原汁原味”地把 DepthPro 深度变成 mesh / point cloud
# - 注意：原图 + grid-step=1 会非常大（OBJ 可能几百万三角面）；如果太慢/太大，把 --grid-step 调到 2/3

# (RAW-PLY) 只生成点云（更方便跟 SHARP 的 gaussian .ply 直接对比）
python depthpro_image_to_ply.py \
       --image images/IMG_9546.jpeg \
       --out recon_depthpro_raw_ply/IMG_9546 \
       --image-max-size 0 \
       --grid-step 2 \
       --depth-device auto

# 查看点云：
# python view_ply.py recon_depthpro_raw_ply/IMG_9546/scene_points_camera.ply

# (RAW-MESH) 直接导出相机空间表面 mesh.obj（不缩放）
python depth_to_mesh.py \
                      --image images/IMG_9546.jpeg \
                      --out recon_depthpro_raw/IMG_9546 \
                      --mode camera \
                      --image-max-size 0 \
                      --grid-step 1 \
                      --depth-smooth none \
                      --depth-device auto

# (RAW-POINTS) 用上一步输出的 depth.npy + scene.png + meta.json 反投影成彩色点云 PLY
OUT_DIR=recon_depthpro_raw/IMG_9546 python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

import open3d as o3d

out_dir = Path(os.environ["OUT_DIR"])

depth = np.load(out_dir / "depth.npy").astype(np.float32)
rgb = np.array(Image.open(out_dir / "scene.png").convert("RGB"), dtype=np.uint8)
meta = json.loads((out_dir / "meta.json").read_text())
k = meta["intrinsics"]
fx, fy, cx, cy = float(k["fx"]), float(k["fy"]), float(k["cx"]), float(k["cy"])

h, w = depth.shape
step = 1  # 想要更稀疏就改成 2/3/4

ys = np.arange(0, h, step, dtype=np.int32)
xs = np.arange(0, w, step, dtype=np.int32)
xx, yy = np.meshgrid(xs, ys)
z = depth[yy, xx]

valid = np.isfinite(z) & (z > 0)
xxv = xx[valid].astype(np.float32)
yyv = yy[valid].astype(np.float32)
zv = z[valid].astype(np.float32)

x = (xxv - cx) * zv / fx
y = (yyv - cy) * zv / fy
pts = np.stack([x, y, zv], axis=1)
cols = rgb[yy, xx][valid].astype(np.float32) / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

ply_path = out_dir / "scene_points_camera_depthpro.ply"
o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False)
print(f"Wrote {ply_path} with {len(pts)} points")
PY

# (BATCH) 对 images/ 下所有图片做 RAW-MESH + RAW-POINTS（输出到 recon_depthpro_raw/<basename>/）
# bash/zsh 都能跑；如果 images/ 里混了非图片文件，记得自己改一下 glob
mkdir -p recon_depthpro_raw
for img in images/*; do
       base="$(basename "$img")"
       stem="${base%.*}"
       out="recon_depthpro_raw/$stem"
       python depth_to_mesh.py \
              --image "$img" \
              --out "$out" \
              --mode camera \
              --image-max-size 0 \
              --grid-step 1 \
              --depth-smooth none \
              --depth-device auto

       OUT_DIR="$out" python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

import open3d as o3d

out_dir = Path(os.environ["OUT_DIR"])
depth = np.load(out_dir / "depth.npy").astype(np.float32)
rgb = np.array(Image.open(out_dir / "scene.png").convert("RGB"), dtype=np.uint8)
meta = json.loads((out_dir / "meta.json").read_text())
k = meta["intrinsics"]
fx, fy, cx, cy = float(k["fx"]), float(k["fy"]), float(k["cx"]), float(k["cy"])

h, w = depth.shape
step = 1
ys = np.arange(0, h, step, dtype=np.int32)
xs = np.arange(0, w, step, dtype=np.int32)
xx, yy = np.meshgrid(xs, ys)
z = depth[yy, xx]

valid = np.isfinite(z) & (z > 0)
xxv = xx[valid].astype(np.float32)
yyv = yy[valid].astype(np.float32)
zv = z[valid].astype(np.float32)

x = (xxv - cx) * zv / fx
y = (yyv - cy) * zv / fy
pts = np.stack([x, y, zv], axis=1)
cols = rgb[yy, xx][valid].astype(np.float32) / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
ply_path = out_dir / "scene_points_camera_depthpro.ply"
o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False)
print(f"Wrote {ply_path} with {len(pts)} points")
PY
done

# (BATCH-PLY) 对 images/ 下所有图片仅生成点云 PLY（方便与 SHARP 对比）
mkdir -p recon_depthpro_raw_ply
for img in images/*; do
       base="$(basename "$img")"
       stem="${base%.*}"
       python depthpro_image_to_ply.py \
              --image "$img" \
              --out "recon_depthpro_raw_ply/$stem" \
              --image-max-size 0 \
              --grid-step 2 \
              --depth-device auto
done

# (A) 推荐：plane 模式，从地面/桌面支撑平面生长 + 强力去背景
python depth_to_mesh.py \
       --image IMG_9546.jpeg \
       --out mesh_IMG_9546 \
       --mode plane \
       --image-max-size 1024 \
       --grid-step 3 \
       --fit-bottom-frac 0.60 \
       --plane-distance-thresh-m 0.02 \
       --max-height-m 1.50 \
       --tri-max-height-step-m 0.25 \
       --tri-max-edge-m 0.28 \
       --growth-filter \
       --drop-floating \
       --detect-table-plane \
       --table-seed-normal-dot 0 \
       --seed-band-m 0.08 \
       --neighbor-max-height-step-m 0.10 \
       --neighbor-max-depth-step-m 0.25 \
       --neighbor-max-3d-step-m 0.10 \
       --support-depth-pctl 92 --support-depth-margin-m 0.6 \
       --seed-depth-pctl 65 --seed-depth-margin-m 0.30 \
       --keep-nearest-seed-component \
       --suppress-vertical-planes --vertical-plane-min-inliers 8000 \
       --sam2-instances \
       --sam2-max-area-frac 0.30 \
       --stack-contact-dist-m 0.07 --stack-min-height-delta-m 0.02 \
       --salvage-supported-instances \
       --salvage-contact-band-m 0.06 --salvage-min-grid-area 25 --salvage-depth-margin-m 1.4

# Step2 (single image): 把生成的 2.5D mesh 变成“闭合/可打印”的 watertight 网格（加背面+侧壁）
python close_mesh.py \
       --input mesh_IMG_9546/mesh.obj \
       --out mesh_IMG_9546/mesh_closed \
       --thickness 0.03

# Step2 (single image): 进一步把地板/桌面等“洞”补上（只封内部洞，不会破坏外轮廓）
python close_mesh.py \
       --input mesh_IMG_9546/mesh.obj \
       --out mesh_IMG_9546/mesh_closed_cap \
       --thickness 0.03 \
       --cap-holes

# Step3 (object-first): SAM2 分割每个物体 -> 从 mesh 中抽取该物体部分 -> 各自做闭合 solid
# 输出：mesh_IMG_9546/objects_completed/obj_*/mesh_closed.obj
python complete_objects.py \
       --image mesh_IMG_9546/scene.png \
       --mesh mesh_IMG_9546/mesh.obj \
       --out mesh_IMG_9546/objects_completed \
       --thickness-m 0.03 \
       --classify

# Step4 (assemble): 把“闭合+补洞”的 base 场景 + 补全后的物体，汇聚回一个 scene.obj
# - 默认会 compact 掉 NaN 顶点（老的 mesh.obj 里可能有占位 NaN）
# - 加 --remove-from-base 会把 base 里被物体 mask 覆盖的三角面删掉，减少重叠表面
python assemble_scene.py \
       --base mesh_IMG_9546/mesh_closed_cap/mesh_closed.obj \
       --objects-summary mesh_IMG_9546/objects_completed_v2/complete_objects_summary.json \
       --out mesh_IMG_9546/scene_assembled_v2 \
       --remove-from-base

# Step5 (true-AI-ish completion, single image):
# “真正的分割”=SAM2 自动实例；“虚拟视角”=用原始 DepthPro 深度做重投影生成新视角，再用 Diffusers inpaint 补洞，
# 然后对每个视角跑 DepthPro，并把多视角点云反变换回原坐标，最后 Poisson 重建得到更完整的几何。
#
# 注意：Diffusers inpaint 首次会下载模型（需要联网/可能需要 HuggingFace token）。
# 如果你只想离线跑通：把 --inpaint diffusers 改成 --inpaint opencv。

# (A) 直接用 depth_to_mesh 的输出目录（你目前只跑了 depth_to_mesh，最推荐用这个）
python ai_multiview_complete.py \
       --mesh-dir mesh_IMG_9546 \
       --out mesh_IMG_9546/ai_multiview \
       --keep-topk 5 --nms-iou 0.60 \
       --n-views 7 --yaw-max-deg 55 \
       --render-step 2 --points-step 2 \
       --inpaint diffusers \
       --view-max-size 768 \
       --depth-device auto \
       --poisson-depth 9 --voxel-down 0.01

# (A2) 离线/更快：OpenCV inpaint + 不跑“每个新视角的 DepthPro”（几何补全会弱一些，但可快速验证流程）
# python ai_multiview_complete.py --mesh-dir mesh_IMG_9546 --out mesh_IMG_9546/ai_multiview_fast --keep-topk 5 --n-views 5 --yaw-max-deg 45 --render-step 3 --points-step 3 --inpaint opencv --skip-depthpro-views --poisson-depth 8 --voxel-down 0.02

# (B) 指定一个实例 id（先看 mesh_IMG_9546/ai_multiview/sam2/instances.json 再填）
# python ai_multiview_complete.py --mesh-dir mesh_IMG_9546 --out mesh_IMG_9546/ai_multiview --instance-id 7 --n-views 9 --yaw-max-deg 70 --inpaint diffusers

# Step6 (mesh-first patch):
# 目标：不做“点云重建整个物体”，而是直接基于 depth_to_mesh 生成的网格顶点（含 NaN 占位）
# 用“原图上的 SAM2 多物体 union mask”把正面该有的面补回来，然后和 closed/cap base 合并。
# 输出：mesh_IMG_9546/mesh_first_patch/mesh_patched.obj（包含 base + patch 两个 group）

python mesh_first_patch.py \
       --mesh-dir mesh_IMG_9546 \
       --out mesh_IMG_9546/mesh_first_patch \
       --keep-topk 30 --nms-iou 0.60 \
       --sam2-min-area 600 --sam2-max-area-frac 0.60 \
       --remove-from-base --face-votes 2

# Step7 (Rodin-ish direction, experimental): diffusion novel views (SVD) -> DepthPro -> TSDF fusion
# 注意：首次运行会下载 SVD 模型（需要联网/可能需要 HuggingFace token）。
python diffusion_multiview_tsdf.py \
       --mesh-dir mesh_IMG_9546 \
       --out mesh_IMG_9546/diffusion_tsdf \
       --svd-device auto \
       --svd-frames 14 \
       --view-max-size 768 \
       --tsdf-voxel 0.01 \
       --tsdf-trunc 0.04

# Optional (Unity path): SHARP (3DGS) -> triangle mesh
# 1) Run SHARP in its own env to get a gaussian .ply (example):
#    sharp predict -i /path/to/images -o /path/to/out/gaussians
# 2) Convert the resulting gaussian .ply into a mesh you can import to Unity:
#    python sharp_to_mesh.py --gs-ply /path/to/out/gaussians/*.ply --out /path/to/out/mesh

# Optional (Unity path, relief-style): treat SHARP gaussians as a depth map (z-buffer) and triangulate like depth_to_mesh.
# This tends to look closer to the original photo than Poisson meshing the 3DGS cloud.
# Example (uses your IMG_9546 paths):
python sharp_ply_to_depth_mesh.py \
       --gs-ply /Users/rufuslee/Downloads/ml-sharp-main/output/gaussians/IMG_9546.ply \
       --image IMG_9546.jpeg \
       --out /Users/rufuslee/Downloads/ml-sharp-main/output/sharp_depth_mesh \
       --image-max-size 1280 --grid-step 3 \
       --opacity-thresh 0.02 --max-gaussians 400000 --fill-holes \
       --depth-smooth median --depth-smooth-ksize 7 \
       --tri-max-edge-m 0.28 --tri-max-depth-step-m 0.25

# (B) camera 模式：直接导出相机空间表面（背景会更多，适合看原始深度形状）
python depth_to_mesh.py \
       --image /path/to/iphone_photo.jpg \
       --mode camera \
       --image-max-size 0 \
       --grid-step 1 \
       --depth-smooth none

# (C) 如果单图内参不准（透视不对/尺度怪），可以手动覆盖：
# python depth_to_mesh.py --image /path/to/iphone_photo.jpg --mode camera --image-max-size 1024 --grid-step 3 --depth-smooth median --depth-smooth-ksize 7 --focal-px 1800
# 或者更细：--fx 1800 --fy 1800 --cx (w-1)/2 --cy (h-1)/2

# Optional semantic suppression (less reliable for your scene):
# python segment_scene.py \
#        --image SpatialCapture_2026-01-15T02-30-04.724Z/mesh_000000/scene_image.png \
#        --out SpatialCapture_2026-01-15T02-30-04.724Z/mesh_000000/seg \
#        --labels wall,ceiling
# python depth_to_mesh.py \
#        --input SpatialCapture_2026-01-15T02-30-04.724Z \
#        --frame-index 0 \
#        --mode plane \
#        --image-max-size 1024 \
#        --grid-step 2 \
#        --fit-bottom-frac 0.60 \
#        --plane-distance-thresh-m 0.02 \
#        --max-height-m 1.50 \
#        --tri-max-height-step-m 0.25 \
#        --suppress-mask SpatialCapture_2026-01-15T02-30-04.724Z/mesh_000000/seg/mask_wall.png \
#        --suppress-mask SpatialCapture_2026-01-15T02-30-04.724Z/mesh_000000/seg/mask_ceiling.png \
#        --plane-width 2.0 --plane-height 2.0 \
#        --relief-strength-m 0.15

# Stage2 (baseline): 清理“空气连线/噪声点”，并按多视角深度一致性标注 extend vs fill
python stage2_extend_fill.py \
       --input SpatialCapture_2026-01-15T02-30-04.724Z \
       --pcd SpatialCapture_2026-01-15T02-30-04.724Z/fuse/fused_tsdf_mesh_3frames_recolor_fix_pcd.ply \
       --depth-consistency 0.20 \
       --radius 0.06 --radius-n 8 \
       --stat-nn 30 --stat-std 2.0 \
       --dbscan-eps 0.06 --dbscan-min 30 --min-cluster 200

/Users/rufuslee/Documents/GitHub/backend/giyubackendenv/bin/python /Users/rufuslee/Documents/GitHub/backend/depth_to_mesh.py --input /Users/rufuslee/Documents/GitHub/backend/SpatialCapture_2026-01-15T02-30-04.724Z --frame-index 0 --mode plane --image-max-size 768 --grid-step 3 --fit-bottom-frac 0.60 --plane-distance-thresh-m 0.02 --max-height-m 1.5 --tri-max-height-step-m 0.25 --tri-max-edge-m 0.22 --growth-filter --detect-table-plane --table-seed-normal-dot 0 --seed-band-m 0.01 --neighbor-max-height-step-m 0.10 --neighbor-max-depth-step-m 0.25 --neighbor-max-3d-step-m 0.10 --support-depth-pctl 92 --support-depth-margin-m 0.6 --seed-depth-pctl 65 --seed-depth-margin-m 0.30 --keep-nearest-seed-component --suppress-vertical-planes --vertical-plane-min-inliers 8000 --sam2-instances --sam2-points-per-side 32 --sam2-pred-iou-thresh 0.85 --sam2-stability-thresh 0.95 --sam2-max-area-frac 0.30 --stack-contact-dist-m 0.07 --stack-min-height-delta-m 0.02 --salvage-supported-instances --salvage-contact-band-m 0.06 --salvage-min-grid-area 25 --salvage-depth-margin-m 1.4

python sharp_to_mesh.py --gs-ply /Users/rufuslee/Downloads/ml-sharp-main/output/gaussians/IMG_9546.ply --out /Users/rufuslee/Downloads/ml-sharp-main/output/mesh_unity

/Users/rufuslee/Documents/GitHub/backend/giyubackendenv/bin/python /Users/rufuslee/Documents/GitHub/backend/sharp_ply_to_depth_mesh.py --gs-ply /Users/rufuslee/Downloads/ml-sharp-main/output/gaussians/IMG_9546.ply --image /Users/rufuslee/Documents/GitHub/backend/IMG_9546.jpeg --out /Users/rufuslee/Downloads/ml-sharp-main/output/sharp_depth_mesh --image-max-size 1280 --grid-step 3 --opacity-thresh 0.02 --max-gaussians 400000 --fill-holes --tri-max-edge-m 0.28 --tri-max-depth-step-m 0.25
