# 虚拟视觉：场景体重建
## 演示视频
由于当时主摄像头的20米视觉裁剪，这个场景体只显示了一半即被截断，但仍可以进行视觉观察。左上角为人物的第一人称视角，和手机摄像头完全不相关。现在调试阶段AR环境中还会显示场景体，而在应用阶段场景体只供人物交互而在用户视角中隐藏


https://github.com/user-attachments/assets/fd90bb3f-1686-4aca-bf10-bbc60a1f4cc1


## 快速验证
### Depth Pro路径
- 在终端cd depthpro_proj进入Depth Pro的文件夹
- git clone https://github.com/apple/ml-depth-pro 拉取depth pro仓库
- 完成DepthPro相关配置，并下载模型；新开一个环境配置depthpro_proj的requirements.txt
- 使用以下命令快速验证点云生成结果
```bash
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
```
### SHARP路径
- cd ml-sharp-main进入sharp文件夹
- 配置环境并推理生成
## 其他
iOS代码因为使用了Unity as a Library，涉及内容比较复杂，在此不展示了


