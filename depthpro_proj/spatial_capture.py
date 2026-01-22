from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FrameRecord:
    timestamp: float
    image_file: str
    depth_file: Optional[str]
    depth_width: Optional[int]
    depth_height: Optional[int]
    image_resolution_w: int
    image_resolution_h: int
    intrinsics3x3: np.ndarray  # (3,3) float32, source
    saved_image_w: Optional[int]
    saved_image_h: Optional[int]
    saved_intrinsics3x3: Optional[np.ndarray]  # (3,3) float32
    camera_transform_cw_4x4: np.ndarray  # (4,4) float32
    orientation: str


def _mat3_from_list(vals: list[float]) -> np.ndarray:
    arr = np.array(vals, dtype=np.float32)
    if arr.size != 9:
        raise ValueError(f"Expected 9 floats for 3x3, got {arr.size}")
    return arr.reshape((3, 3), order="F")  # Swift flattens column-major


def _mat4_from_list(vals: list[float]) -> np.ndarray:
    arr = np.array(vals, dtype=np.float32)
    if arr.size != 16:
        raise ValueError(f"Expected 16 floats for 4x4, got {arr.size}")
    return arr.reshape((4, 4), order="F")  # Swift flattens column-major


def read_frames_jsonl(capture_dir: Path) -> Iterator[FrameRecord]:
    manifest = capture_dir / "frames.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"frames.jsonl not found in: {capture_dir}")

    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            intr = _mat3_from_list(obj["intrinsics3x3"])
            cw = _mat4_from_list(obj["cameraTransform_cw_4x4"])

            saved_w = obj.get("savedImageW")
            saved_h = obj.get("savedImageH")
            saved_k = obj.get("savedIntrinsics3x3")
            saved_intr = _mat3_from_list(saved_k) if saved_k is not None else None

            yield FrameRecord(
                timestamp=float(obj["timestamp"]),
                image_file=str(obj["imageFile"]),
                depth_file=obj.get("depthFile"),
                depth_width=obj.get("depthWidth"),
                depth_height=obj.get("depthHeight"),
                image_resolution_w=int(obj["imageResolutionW"]),
                image_resolution_h=int(obj["imageResolutionH"]),
                intrinsics3x3=intr,
                saved_image_w=int(saved_w) if saved_w is not None else None,
                saved_image_h=int(saved_h) if saved_h is not None else None,
                saved_intrinsics3x3=saved_intr,
                camera_transform_cw_4x4=cw,
                orientation=str(obj.get("orientation", "unknown")),
            )


def load_jpeg_size(capture_dir: Path, image_file: str) -> Tuple[int, int]:
    p = capture_dir / image_file
    with Image.open(p) as im:
        w, h = im.size
    return int(w), int(h)


def infer_saved_intrinsics_from_source(
    source_k: np.ndarray,
    source_w: int,
    source_h: int,
    saved_w: int,
    saved_h: int,
) -> np.ndarray:
    """Compute intrinsics for the saved JPEG when the app saved CIImage.oriented(.right) then uniformly scaled.

    Matches the Swift logic:
    - rotate right: fx' = fy, fy' = fx, cx' = cy, cy' = (w0 - 1) - cx
    - scale uniformly to match saved dimensions

    Returns K_saved in the saved JPEG pixel coordinate system.
    """
    fx0 = float(source_k[0, 0])
    fy0 = float(source_k[1, 1])
    cx0 = float(source_k[0, 2])
    cy0 = float(source_k[1, 2])

    fx1 = fy0
    fy1 = fx0
    cx1 = cy0
    cy1 = float(max(source_w - 1, 1)) - cx0

    # After rotate right: width becomes source_h, height becomes source_w
    w1 = max(int(source_h), 1)
    h1 = max(int(source_w), 1)

    sx = float(saved_w) / float(w1)
    sy = float(saved_h) / float(h1)
    s = 0.5 * (sx + sy)

    k = np.array(
        [[fx1 * s, 0.0, cx1 * s], [0.0, fy1 * s, cy1 * s], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return k


def get_intrinsics_for_saved_image(record: FrameRecord, capture_dir: Path) -> Tuple[np.ndarray, int, int]:
    """Returns (K, width, height) for the saved JPEG space."""
    if record.saved_intrinsics3x3 is not None and record.saved_image_w and record.saved_image_h:
        return record.saved_intrinsics3x3, record.saved_image_w, record.saved_image_h

    w, h = load_jpeg_size(capture_dir, record.image_file)
    k = infer_saved_intrinsics_from_source(
        record.intrinsics3x3,
        record.image_resolution_w,
        record.image_resolution_h,
        w,
        h,
    )
    return k, w, h


def arkit_cw_to_open3d_cw(cw_arkit: np.ndarray) -> np.ndarray:
    """Convert ARKit camera-to-world pose to an Open3D-friendly camera-to-world pose.

    ARKit camera convention (camera space): +X right, +Y up, camera looks along -Z.
    Our depth backprojection uses the common pinhole convention: +X right, +Y down, +Z forward.

    To map backprojected points into ARKit camera space, we must flip BOTH Y and Z:
        p_arkit = T * p_pinhole,  T = diag(1, -1, -1, 1)

    Flipping only one axis (e.g. Z) is a reflection (det=-1) and causes mirroring.
    Thus we return: cw_open3d = cw_arkit @ T.
    """
    t = np.eye(4, dtype=np.float32)
    t[1, 1] = -1.0
    t[2, 2] = -1.0
    return (cw_arkit @ t).astype(np.float32)
