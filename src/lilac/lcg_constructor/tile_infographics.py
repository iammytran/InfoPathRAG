#!/usr/bin/env python3
"""Build a tile-based InfoVQA corpus.

This pipeline intentionally does not read or create ``parsed_documents``.
It writes a manifest, Qwen-VL sidecars, per-tile serializations, and the
standard MM-Embed ``.pt``/``.json`` pair used by the retriever.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import math

from PIL import Image

from src.lilac.lcg_constructor.preprocessing._caption_via_qwen_vl import (
    caption_images,
)
from src.models.embedder.mmembed import MMEmbed
from src.models.embedder.sharded_runner import encode_one_corpus
from src.utils.utils import REPO_ROOT

DEFAULT_INPUT = Path(REPO_ROOT) / "datasets/InfoVQA/image_components/test"
DEFAULT_ROOT = Path(REPO_ROOT) / "datasets/InfoVQA/tiles"
DEFAULT_ESTIMATES = Path(REPO_ROOT) / "debug/estimated_components.json"
TILE_PROMPT = (
    "The first image is a tile from an infographic and the second image is "
    "the complete original infographic. Describe only the tile, but use the "
    "complete infographic as context. Transcribe all visible text and explain "
    "the tile's spatial role. Be factual and concise."
)

def prepare_tiled_inputs(
    images_dir: Path,
    staging_dir: Path,
    manifest_path: Path,
    *,
    tile_width: int | None = None,
    tile_height: int | None = None,
    overlap: int = 0,
    max_tiles: int = 4,
    estimated_components_path: Path = DEFAULT_ESTIMATES,
) -> dict[str, Any]:
    """Tile every infographic and persist coordinates in a manifest.

    ``tile_width`` and ``tile_height`` are independent knobs. Set either to
    ``None`` to derive that dimension per infographic from ``max_tiles`` while
    preserving the image aspect ratio.
    """
    if tile_width is not None and tile_width <= 0:
        raise ValueError("tile_width must be positive or None")
    if tile_height is not None and tile_height <= 0:
        raise ValueError("tile_height must be positive or None")
    if tile_width is None and tile_height is None and max_tiles < 1:
        raise ValueError("max_tiles must be positive when dimensions are automatic")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    estimates = _load_estimated_components(estimated_components_path)
    staging_dir.mkdir(parents=True, exist_ok=True)
    infographics: list[dict[str, Any]] = []
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    for source in sorted(images_dir.iterdir()):
        if not source.is_file() or source.suffix.lower() not in extensions:
            continue
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            estimated = _estimated_for_source(source, estimates)
            current_width, current_height = _resolve_tile_size(
                width, height, tile_width, tile_height,
                max_tiles=max_tiles, estimated_components=estimated,
            )
            if overlap >= min(current_width, current_height):
                raise ValueError("overlap must be smaller than the effective tile size")
            xs = _positions(width, current_width, current_width - overlap)
            ys = _positions(height, current_height, current_height - overlap)
            infographic_dir = staging_dir / source.stem
            infographic_dir.mkdir(parents=True, exist_ok=True)
            tiles = []
            for row, y in enumerate(ys):
                for col, x in enumerate(xs):
                    w = min(current_width, width - x)
                    h = min(current_height, height - y)
                    tile_name = (
                        f"{source.stem}__r{row:03d}_c{col:03d}"
                        f"__x{x}_y{y}_w{w}_h{h}.png"
                    )
                    tile_path = infographic_dir / tile_name
                    image.crop((x, y, x + w, y + h)).save(tile_path)
                    tiles.append({
                        "component_id": f"i_1_t{len(tiles):04d}",
                        "filename": str(tile_path),
                        "path": str(tile_path),
                        "row": row, "column": col,
                        "x": x, "y": y, "width": w, "height": h,
                    })
            infographics.append({
                "id": source.stem,
                "estimated_components": estimated,
                "original": {"filename": str(source), "path": str(source),
                             "width": width, "height": height},
                "tiles": tiles,
            })
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "infographics": infographics}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

def _positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    result = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if result[-1] != last:
        result.append(last)
    return result


def _resolve_tile_size(
    width: int,
    height: int,
    tile_width: int | None,
    tile_height: int | None,
    max_tiles: int,
    estimated_components: int | None = None,
) -> tuple[int, int]:
    if tile_width is not None and tile_height is not None:
        return tile_width, tile_height
    if tile_width is not None:
        return tile_width, max(1, round(tile_width * height / width))
    if tile_height is not None:
        return max(1, round(tile_height * width / height)), tile_height

    requested_tiles = estimated_components or max_tiles
    requested_tiles = max(1, min(requested_tiles, max_tiles))
    columns = max(1, math.ceil(math.sqrt(requested_tiles * width / height)))
    rows = max(1, math.ceil(requested_tiles / columns))
    return math.ceil(width / columns), math.ceil(height / rows)


def _load_estimated_components(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing estimated-components JSON: {path}. "
            "Run estimate_components.py first."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected an object in {path}")
    estimates = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Invalid estimated_components for {key!r}: {value!r}")
        estimates[str(key)] = value
    return estimates


def _estimated_for_source(source: Path, estimates: dict[str, int]) -> int:
    candidates = (str(source), str(source.resolve()), source.name, source.stem)
    for candidate in candidates:
        if candidate in estimates:
            return estimates[candidate]
    raise KeyError(
        f"No estimated_components entry for {source}. "
        "Expected its absolute path, relative path, filename, or stem in the JSON."
    )


def tile_infographics(
    input_dir: Path,
    output_dir: Path,
    *,
    tile_width: int | None = None,
    tile_height: int | None = None,
    overlap: int = 0,
    max_tiles: int = 4,
    estimated_components_path: Path = DEFAULT_ESTIMATES,
) -> dict[str, Any]:
    """Tile every image and return/write a stable location manifest."""
    return prepare_tiled_inputs(
        input_dir,
        output_dir,
        output_dir / "manifest.json",
        tile_width=tile_width,
        tile_height=tile_height,
        overlap=overlap,
        max_tiles=max_tiles,
        estimated_components_path=estimated_components_path,
    )


def caption_tiles(manifest: dict[str, Any], output_dir: Path, num_gpus: int | None) -> None:
    jobs = []
    for infographic in manifest["infographics"]:
        original = infographic["original"]["path"]
        for tile in infographic["tiles"]:
            output = output_dir / "summaries" / Path(tile["filename"]).name.replace(
                ".png", ".txt"
            )
            jobs.append((tile["path"], original, str(output), tile))

    if not jobs:
        return
    caption_images(
        [[tile, original] for tile, original, _, _ in jobs],
        [out for _, _, out, _ in jobs],
        prompt=TILE_PROMPT,
        max_tokens=1024,
        num_gpus=num_gpus,
    )
    summary_by_filename = {}
    for _, _, out, tile in jobs:
        summary_by_filename[tile["filename"]] = Path(out).read_text(
            encoding="utf-8"
        ) if Path(out).exists() else ""
    for infographic in manifest["infographics"]:
        for tile in infographic["tiles"]:
            tile["caption"] = summary_by_filename.get(tile["filename"], "")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def serialize_tiles(manifest: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    serialization_dir = output_dir / "serializations"
    serialization_dir.mkdir(parents=True, exist_ok=True)
    top = []
    low = []
    for infographic in manifest["infographics"]:
        original = infographic["original"]
        original_path = original["path"]
        original_name = Path(original["filename"]).name
        top.append({
            "id": [original_name, "i_1"],
            "target": {"text": f"{Path(original_name).stem} [SEP] original infographic",
                       "images": [original_path]},
        })
        for tile in infographic["tiles"]:
            location = (
                f"row={tile['row']} column={tile['column']} "
                f"x={tile['x']} y={tile['y']} "
                f"width={tile['width']} height={tile['height']}"
            )
            low.append({
                "id": [original_name, tile["component_id"]],
                "target": {
                    "text": (
                        f"{Path(original_name).stem} [SEP] tile location: {location}"
                        f" [SEP] {tile.get('caption', '')}"
                    ),
                    "images": [tile["path"], original_path],
                },
            })
    top_path = serialization_dir / "image.json"
    low_path = serialization_dir / "subimage.json"
    top_path.write_text(json.dumps(top, indent=2), encoding="utf-8")
    low_path.write_text(json.dumps(low, indent=2), encoding="utf-8")
    return top_path, low_path


def embed_serializations(top_path: Path, low_path: Path, output_dir: Path, num_gpus: int) -> None:
    embedding_dir = output_dir / "embeddings" / "MM-Embed"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    for corpus_path, name, batch_size in (
        (top_path, "image", 1),
        (low_path, "subimage", 4),
    ):
        encode_one_corpus(
            embedder_cls=MMEmbed,
            tmp_prefix=f"tile_{name}_",
            corpus_in_filepath=str(corpus_path),
            out_embeddings_path=str(embedding_dir / f"{name}.pt"),
            out_index_path=str(embedding_dir / f"{name}.json"),
            num_gpus=num_gpus,
            max_length=4096,
            batch_size=batch_size,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tile-width", type=int, default=None)
    parser.add_argument("--tile-height", type=int, default=None)
    parser.add_argument("--estimated-components", type=Path,
                        default=DEFAULT_ESTIMATES)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--max-tiles", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    args = parser.parse_args()

    manifest = tile_infographics(
        args.input_dir, args.output_dir,
        tile_width=args.tile_width, tile_height=args.tile_height,
        overlap=args.overlap, max_tiles=args.max_tiles,
        estimated_components_path=args.estimated_components,
    )
    if not args.skip_qwen:
        caption_tiles(manifest, args.output_dir, args.num_gpus)
    top_path, low_path = serialize_tiles(manifest, args.output_dir)
    if not args.skip_embed:
        embed_serializations(top_path, low_path, args.output_dir, args.num_gpus)


if __name__ == "__main__":
    main()
