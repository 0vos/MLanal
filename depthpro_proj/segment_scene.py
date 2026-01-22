from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SegmentOutputs:
    out_dir: Path
    masks: Dict[str, Path]
    meta_path: Path


def _load_rgb_u8(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.array(im, dtype=np.uint8)


def _save_png(path: Path, arr_u8: np.ndarray) -> None:
    Image.fromarray(arr_u8).save(path)


def _to_bool_mask(m) -> Optional[np.ndarray]:
    if m is None:
        return None
    try:
        import torch  # type: ignore

        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
    except Exception:
        pass

    arr = np.asarray(m)
    if arr.size == 0:
        return None

    if arr.ndim >= 4:
        arr = np.squeeze(arr)

    if arr.ndim == 3:
        if arr.dtype == np.bool_:
            return np.any(arr, axis=0)
        if np.issubdtype(arr.dtype, np.floating):
            return np.any(arr > 0.5, axis=0)
        return np.any(arr > 0, axis=0)

    if arr.ndim != 2:
        return None

    if arr.dtype == np.bool_:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        return arr > 0.5
    return arr > 0


def _union_masks(masks) -> np.ndarray:
    union: Optional[np.ndarray] = None
    if isinstance(masks, (list, tuple)):
        for m in masks:
            bm = _to_bool_mask(m)
            if bm is None:
                continue
            union = bm if union is None else (union | bm)
    else:
        bm = _to_bool_mask(masks)
        if bm is not None:
            union = bm
    if union is None:
        return np.zeros((1, 1), dtype=np.bool_)
    return union


_LANGSAM_MODEL = None
_LANGSAM_SAM_TYPE: Optional[str] = None


def _segment_langsam(
    rgb_u8: np.ndarray,
    prompts: List[str],
    sam_type: str,
    box_threshold: float,
    text_threshold: float,
) -> Dict[str, np.ndarray]:
    global _LANGSAM_MODEL, _LANGSAM_SAM_TYPE

    try:
        from lang_sam import LangSAM  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "lang_sam failed to import. Ensure backend venv has LangSAM+SAM2 installed."
        ) from e

    if _LANGSAM_MODEL is None or _LANGSAM_SAM_TYPE != sam_type:
        _LANGSAM_MODEL = LangSAM(sam_type=sam_type)
        _LANGSAM_SAM_TYPE = sam_type

    image = Image.fromarray(rgb_u8, mode="RGB")

    out: Dict[str, np.ndarray] = {}
    for prompt in prompts:
        p = prompt.strip()
        if not p:
            continue
        if not p.endswith("."):
            p = p + "."

        results = _LANGSAM_MODEL.predict(
            [image],
            [p],
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
        )
        if not results:
            out[prompt] = np.zeros(rgb_u8.shape[:2], dtype=np.uint8)
            continue
        result = results[0]
        masks = result.get("masks")
        union = _union_masks(masks)
        if union.shape != rgb_u8.shape[:2]:
            # If LangSAM returns different size, resize with nearest.
            m = Image.fromarray((union.astype(np.uint8) * 255), mode="L")
            m = m.resize((rgb_u8.shape[1], rgb_u8.shape[0]), resample=Image.Resampling.NEAREST)
            out[prompt] = np.array(m, dtype=np.uint8)
        else:
            out[prompt] = (union.astype(np.uint8) * 255)

    return out


def run(
    image_path: Path,
    out_dir: Path,
    labels: List[str],
    sam_type: str,
    box_threshold: float,
    text_threshold: float,
) -> SegmentOutputs:
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = _load_rgb_u8(image_path)

    masks = _segment_langsam(
        rgb,
        prompts=labels,
        sam_type=sam_type,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    out_paths: Dict[str, Path] = {}
    for label, mask_u8 in masks.items():
        safe = label.strip().replace(" ", "_").replace("/", "_").replace(".", "")
        p = out_dir / f"mask_{safe}.png"
        _save_png(p, mask_u8.astype(np.uint8))
        out_paths[label] = p

    # Combined mask
    if out_paths:
        union = np.zeros(rgb.shape[:2], dtype=np.uint8)
        for p in out_paths.values():
            union |= (np.array(Image.open(p).convert("L"), dtype=np.uint8) > 0).astype(np.uint8) * 255
        union_path = out_dir / "mask_union.png"
        _save_png(union_path, union)
        out_paths["__union__"] = union_path

    meta = {
        "image": str(image_path),
        "labels": labels,
        "sam_type": sam_type,
        "box_threshold": float(box_threshold),
        "text_threshold": float(text_threshold),
        "outputs": {k: str(v.name) for k, v in out_paths.items()},
    }
    meta_path = out_dir / "segment_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return SegmentOutputs(out_dir=out_dir, masks=out_paths, meta_path=meta_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Segment scene regions (e.g. wall/ceiling) using LangSAM (GroundingDINO + SAM2).\n"
            "Outputs per-label masks and a union mask."
        )
    )
    ap.add_argument("--image", required=True, type=Path, help="Input RGB image path")
    ap.add_argument("--out", required=True, type=Path, help="Output directory")
    ap.add_argument(
        "--labels",
        default="wall,ceiling",
        help="Comma-separated prompts. Example: wall,ceiling,floor,table",
    )
    ap.add_argument(
        "--sam-type",
        default="sam2.1_hiera_large",
        help="LangSAM SAM type, e.g. sam2.1_hiera_large",
    )
    ap.add_argument("--box-threshold", type=float, default=0.15)
    ap.add_argument("--text-threshold", type=float, default=0.15)

    args = ap.parse_args()
    labels = [s.strip() for s in str(args.labels).split(",") if s.strip()]

    outputs = run(
        image_path=args.image,
        out_dir=args.out,
        labels=labels,
        sam_type=str(args.sam_type),
        box_threshold=float(args.box_threshold),
        text_threshold=float(args.text_threshold),
    )

    print("Wrote:")
    for k, v in outputs.masks.items():
        print(f"- {k}: {v}")
    print(f"- meta: {outputs.meta_path}")


if __name__ == "__main__":
    main()
