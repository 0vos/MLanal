# Virtual Vision: Scene Volume Reconstruction
## Demo Video
Due to the 20-meter visual clipping of the main camera at the time, this scene volume is only half displayed and truncated, but visual observation is still possible. The top left corner is the character's first-person perspective, completely unrelated to the phone camera. Currently, in the debugging phase, the scene volume is still displayed in the AR environment, while in the application phase, the scene volume is only for character interaction and is hidden from the user's perspective.


https://github.com/user-attachments/assets/fd90bb3f-1686-4aca-bf10-bbc60a1f4cc1


## Quick Start
### Depth Pro Pipeline
- In the terminal, `cd depthpro_proj` to enter the Depth Pro folder.
- `git clone https://github.com/apple/ml-depth-pro` to pull the Depth Pro repository.
- Complete the Depth Pro configuration and download the model; create a new environment and configure `requirements.txt` of `depthpro_proj`.
- Use the following commands to quickly verify the point cloud generation results:
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
### SHARP Pipeline
- `cd ml-sharp-main` to enter the SHARP folder.
- Configure the environment and run inference to generate.
## Other
The iOS code uses Unity as a Library and involves complex content, so it is not shown here.


