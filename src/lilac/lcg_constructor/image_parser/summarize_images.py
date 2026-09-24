#!/usr/bin/env python3
"""
Step 0 — generate VLM-based page-level summaries for every image in a
benchmark's `image_components/dev/`.

Input:  datasets/<DS>/image_components/dev/*.{jpg,jpeg,png,bmp,webp}
        (FLAT glob, no recursion — F-010 fix: previously a recursive glob
         inflated MMCoQA from 704 unique images to 5879 cached duplicates.)
Output: artifacts/<DS>/image_summaries/dev/<stem>.txt

Multi-GPU + resumable via the shared helper `_caption_via_qwen_vl.caption_images`.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import List

from src.lilac.lcg_constructor.preprocessing._caption_via_qwen_vl import caption_images
from src.utils.utils import REPO_ROOT

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROMPT_TXT = "Generate a summary of the given image. Include all the texts within the image."


def collect_images(img_dir: pathlib.Path) -> List[pathlib.Path]:
    """Flat (non-recursive) glob; F-010 fix."""
    imgs: List[pathlib.Path] = []
    for p in sorted(img_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_data", required=True,
                    help="Benchmark name, e.g. MP-DocVQA, MMCoQA, …")
    ap.add_argument("--start_idx", type=int, default=0,
                    help="Slice start (for chunked runs)")
    ap.add_argument("--end_idx", type=int, default=0,
                    help="Slice end (0 = no upper bound)")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--num_gpus", type=int, default=0,
                    help="Cap on parallel GPU workers (0 = use all visible)")
    args = ap.parse_args()

    img_dir = pathlib.Path(f"{REPO_ROOT}/datasets/{args.target_data}/image_components/test")
    out_dir = pathlib.Path(f"{REPO_ROOT}/artifacts/{args.target_data}/image_summaries/test")

    if not img_dir.is_dir():
        print(f"❌  Image dir not found: {img_dir}")
        return

    imgs = collect_images(img_dir)
    if not imgs:
        print(f"❌  No images found in {img_dir}")
        return
    print(f"📂  Found {len(imgs):,} images in {img_dir} (flat glob)")

    s, e = args.start_idx, (args.end_idx if args.end_idx > 0 else len(imgs))
    imgs = imgs[s:e]
    if not imgs:
        print(f"⚠️  Slice [{s}:{e}] is empty")
        return
    print(f"🔪  Processing slice [{s}:{e}] = {len(imgs):,} images")

    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [str(p) for p in imgs]
    output_paths = [str(out_dir / f"{p.stem}.txt") for p in imgs]

    caption_images(
        image_paths=image_paths,
        output_paths=output_paths,
        prompt=PROMPT_TXT,
        max_tokens=args.max_tokens,
        num_gpus=(args.num_gpus or None),
    )

    written = sum(1 for p in output_paths if pathlib.Path(p).exists())
    print(f"✅  {written}/{len(output_paths)} outputs present in {out_dir}")


if __name__ == "__main__":
    main()
