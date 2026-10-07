"""
Shared multi-GPU, resumable Qwen2.5-VL-7B captioning helper.

Used by:
  - step 0 (`image_parser/summarize_images.py`): page-level summaries
  - step 5 (Qwen bbox crop pass): per-crop caption.text fill
  - layout-analyzer adapters (`doclayout_yolo_to_lilac.py`, `mineru_to_lilac.py`):
    per-region caption.text fill for vqa-type datasets

Design:
  - One worker process per visible GPU (CUDA_VISIBLE_DEVICES isolation).
  - Each worker writes per-image .txt incrementally → resumable on SIGTERM.
  - Skips images whose output .txt already exists (resumability + dedupe).
  - Caller passes parallel lists of input image paths and output txt paths.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path
from typing import List, Tuple


def _visible_gpu_ids() -> List[int]:
    """CUDA ordinals exposed via CUDA_VISIBLE_DEVICES, else all reported by torch."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if env.strip():
        ids = [int(t) for t in env.split(",") if t.strip().isdigit()]
        if ids:
            return ids
    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count())) or [0]
    except Exception:
        pass
    return [0]


def _worker(
    gpu_id: int,
    slice_: List[Tuple[str | List[str], str, str]],
    prompt: str,
    max_tokens: int,
    show_progress: bool,
) -> None:
    """Load Qwen2.5-VL on the assigned GPU, caption each (img, out) one at a time."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Imports happen AFTER setting CUDA_VISIBLE_DEVICES so torch sees only this GPU.
    import gc
    import torch
    from src.models.mllm.qwen2_5_vl_7b import Qwen2_5_VL
    from tqdm import tqdm

    model = Qwen2_5_VL()
    iterator = tqdm(slice_, desc=f"qwen-vl gpu={gpu_id}", disable=not show_progress)
    
    # Ép PyTorch không lưu Gradient (Tiết kiệm VRAM tối đa)
    with torch.no_grad():
        for i, (img_path, out_path, item_prompt) in enumerate(iterator):
            out_p = Path(out_path)
            if out_p.exists():
                continue
            try:
                images = img_path if isinstance(img_path, list) else [img_path]
                obj = {"text": item_prompt, "images": [str(path) for path in images]}
                result = model.infer([obj], batch_size=4, max_tokens=max_tokens)
                text = (result[0] if result else "").strip()
                out_p.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = out_p.with_suffix(out_p.suffix + ".tmp")
                tmp_path.write_text(text, encoding="utf-8")
                tmp_path.replace(out_p)
            except Exception as e:
                print(f"[gpu{gpu_id}] error on {img_path}: {e}", file=sys.stderr)
                traceback.print_exc(limit=2)
                # Giải phóng VRAM ngay lập tức nếu bức ảnh này bị lỗi/tràn bộ nhớ
                torch.cuda.empty_cache()
                continue
            finally:
                # Cứ sau 20 ảnh, dọn sạch rác VRAM & RAM đệm 1 lần
                if i % 20 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

def caption_images(
    image_paths: List[str | List[str]],
    output_paths: List[str],
    *,
    prompt: str = "Generate a summary of the given image. Include all the texts within the image.",
    max_tokens: int = 2048,
    num_gpus: int | None = None,
    show_progress: bool = True,
    prompts: List[str] | None = None,
) -> None:
    """Caption a list of images with Qwen2.5-VL, sharded across visible GPUs.

    Args
    ----
    image_paths   : paths, or ordered image-path lists, parallel to output_paths
    output_paths  : where to write each image's .txt result
    prompt        : Qwen-VL text prompt; default is the page-summary prompt
    max_tokens    : per-output token cap
    num_gpus      : cap the worker count (None => use all visible GPUs)
    show_progress : per-worker tqdm bar (only the gpu=first one shows by default
                    if you want quieter logs — set False to silence all)

    Idempotent: skips outputs that already exist. Writes atomically via .tmp swap
    so SIGTERM mid-write leaves either the old content or nothing (never half).
    """
    assert len(image_paths) == len(output_paths), "parallel lists required"

    if prompts is not None and len(prompts) != len(image_paths):
        raise ValueError("prompts must be parallel to image_paths")
    prompt_values = prompts or [prompt] * len(image_paths)
    todo = [
        (img, out, item_prompt)
        for img, out, item_prompt in zip(image_paths, output_paths, prompt_values)
        if not Path(out).exists()
    ]
    if not todo:
        print(f"[caption_images] all {len(image_paths)} outputs already exist — nothing to do.")
        return
    print(f"[caption_images] {len(todo)}/{len(image_paths)} to caption")

    gpus = _visible_gpu_ids()
    if num_gpus is not None:
        gpus = gpus[:num_gpus]
    if not gpus:
        gpus = [0]
    print(f"[caption_images] using GPUs {gpus}")

    if len(gpus) == 1:
        # Single-process path: avoid mp overhead for small jobs.
        _worker(gpus[0], todo, prompt, max_tokens, show_progress)
        return

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    per_gpu = math.ceil(len(todo) / len(gpus))
    procs: List[mp.Process] = []
    for rank, gpu in enumerate(gpus):
        s, e = rank * per_gpu, min((rank + 1) * per_gpu, len(todo))
        if s >= e:
            continue
        slice_ = todo[s:e]
        p = mp.Process(
            target=_worker,
            args=(gpu, slice_, prompt, max_tokens, show_progress and rank == 0),
        )
        p.start()
        procs.append(p)

    failures = 0
    for p in procs:
        p.join()
        if p.exitcode != 0:
            print(f"[caption_images] worker pid={p.pid} exit={p.exitcode}", file=sys.stderr)
            failures += 1
    if failures:
        print(f"[caption_images] {failures}/{len(procs)} workers exited non-zero — "
              f"resumable: re-run to retry.", file=sys.stderr)
