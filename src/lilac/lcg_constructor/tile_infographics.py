#!/usr/bin/env python3
"""Build a tile-based InfoVQA corpus using aspect-aware infographic grid tiling."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image

from src.lilac.lcg_constructor.preprocessing._caption_via_qwen_vl import (
    caption_images,
)
from src.models.embedder.mmembed import MMEmbed
from src.models.embedder.sharded_runner import encode_one_corpus
from src.utils.utils import REPO_ROOT

DEFAULT_INPUT = Path(REPO_ROOT) / "datasets/InfoVQA/image_components/test"
ARTIFACTS_ROOT = Path(REPO_ROOT) / "artifacts"
DEFAULT_ESTIMATES = Path(REPO_ROOT) / "debug/estimated_components.json"
DEFAULT_IMAGE_SUMMARIES = ARTIFACTS_ROOT / "image_summaries"
DEFAULT_PROCESS_TILES = ARTIFACTS_ROOT / "boundaries"
DEFAULT_TILES_AFTER_PROCESS = ARTIFACTS_ROOT / "processed_tiles"
DEFAULT_FACTS_EACH_TILE = ARTIFACTS_ROOT / "facts"
DEFAULT_OCR_EACH_TILE = ARTIFACTS_ROOT / "ocrs"
DEFAULT_TILES = ARTIFACTS_ROOT / "tiles"
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def get_boundary_prompt(tile_location: str) -> str:
    """Tạo boundary prompt hoàn chỉnh dựa trên vị trí tile được truyền vào."""
    return (
        f"Analyze this infographic tile (Location: {tile_location}) to determine if any text lines, words, "
        "table rows, or chart lines are cut off or abruptly truncated at its outer borders.\n\n"
        "Rules:\n"
        "- Return JSON only: {\"is_cut\": true|false, \"sides\": [\"left\"|\"right\"|\"top\"|\"bottom\"], \"reason\": \"...\"}\n"
        "- 'sides' must be [] if 'is_cut' is false.\n"
        "- Column 0 tiles CANNOT be cut on the left.\n"
        "- IGNORE normal multi-line text wrapping and whitespace/margins.\n"
        "- CRITICAL: If any words or sentences at the extreme right or left edge look abruptly sliced in half, incomplete, or cut off mid-word (e.g., ending with 'ope...' instead of 'open'), you MUST mark it as cut for that side.\n"
        "Output JSON:"
    )

OCR_PROMPT = (
    "Extract every readable text item from this infographic tile. Return JSON only "
    'as {"ocr":["text item 1","text item 2"]}. Preserve numbers, units, labels, '
    "and table cell text. Do not add explanations."
)
FACTS_FROM_OCR_PROMPT = (
    "For each OCR item below, generate a clear, natural, and standalone factual sentence "
    "that explains or expands upon its meaning based on the tile and original infographic. "
    "Do not just copy the OCR text verbatim; instead, rewrite and contextualize it into a proper sentence. "
    "Return JSON only in the form "
    '{"facts":[{"ocr":"<original_ocr_item>","fact":"<natural_factual_sentence>"}]}. '
    "Keep the exact same order and count as the OCR items. Do not invent information.\nOCR items:\n"
)


def resolve_tile_grid(
    width: int,
    height: int,
    max_tiles: int = 4,
    estimated: int | None = None,
    min_subcomponents: int = 4,
    min_aspect_ratio: float = 2.0,
) -> tuple[int, int]:
    """Calculate (rows, cols) layout tailored for infographics."""
    aspect = height / max(width, 1)

    # 1. Trần động: nới trần x2 cho infographic siêu dài/rộng (>= 4:1 hoặc <= 1:4)
    effective_max = max_tiles * 2 if (aspect >= 4.0 or aspect <= 0.25) else max_tiles

    # 2. Số lượng tile yêu cầu dựa trên mật độ thành phần
    if estimated:
        requested = max(1, math.ceil(estimated / max(min_subcomponents, 1)))
    else:
        requested = max_tiles

    # 3. Chia nhánh theo tỷ lệ aspect
    if aspect >= min_aspect_ratio:
        # Infographic dài dọc: ép thành 1 cột
        return min(effective_max, requested), 1
    if aspect <= 1.0 / min_aspect_ratio:
        # Infographic trải ngang: ép thành 1 hàng
        return 1, min(effective_max, requested)

    # Infographic cân đối / gần vuông
    total = min(effective_max, requested)
    if total <= 2:
        return 2, 1
    return 2, 2


def _compute_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """Calculate slice start positions ensuring full coverage of length."""
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def load_estimates(path: Path) -> dict[str, int]:
    """Load and validate estimated component counts."""
    if not path.exists():
        raise FileNotFoundError(f"Missing estimated-components file: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {
        str(k): v for k, v in raw.items()
        if isinstance(v, int) and not isinstance(v, bool) and v >= 1
    }


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
    """Tile infographics according to aspect ratio and component density."""
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    estimates = load_estimates(estimated_components_path)
    staging_dir.mkdir(parents=True, exist_ok=True)
    infographics: list[dict[str, Any]] = []

    for source in sorted(images_dir.iterdir()):
        if not source.is_file() or source.suffix.lower() not in VALID_IMAGE_EXTS:
            continue

        estimated = next(
            (estimates[k] for k in (str(source), str(source.resolve()), source.name, source.stem) if k in estimates),
            None,
        )

        with Image.open(source) as img:
            image = img.convert("RGB")
            width, height = image.size

            # Tính toán kích thước tile dựa trên lưới tile_grid hoặc kích thước chỉ định
            if tile_width is not None and tile_height is not None:
                cur_w, cur_h = tile_width, tile_height
            else:
                rows, cols = resolve_tile_grid(width, height, max_tiles=max_tiles, estimated=estimated)
                cur_w = tile_width or math.ceil(width / cols)
                cur_h = tile_height or math.ceil(height / rows)

            if overlap >= min(cur_w, cur_h):
                raise ValueError(f"Overlap ({overlap}) must be smaller than tile dimensions ({cur_w}x{cur_h})")

            xs = _compute_positions(width, cur_w, cur_w - overlap)
            ys = _compute_positions(height, cur_h, cur_h - overlap)

            infographic_dir = staging_dir / source.stem
            infographic_dir.mkdir(parents=True, exist_ok=True)
            tiles = []

            for row, y in enumerate(ys):
                for col, x in enumerate(xs):
                    w = min(cur_w, width - x)
                    h = min(cur_h, height - y)
                    tile_name = f"{source.stem}__r{row:03d}_c{col:03d}__x{x}_y{y}_w{w}_h{h}.png"
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
                "original": {"filename": str(source), "path": str(source), "width": width, "height": height},
                "tiles": tiles,
            })

    manifest = {"version": 1, "infographics": infographics}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _repair_json_text(raw: str) -> str:
    """Recover common Qwen JSON mistakes, handling truncated JSON arrays safely."""
    raw = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()

    start = next((index for index, char in enumerate(raw) if char in "[{"), None)
    if start is None:
        return raw
    raw = raw[start:]

    # Cải tiến: Nếu chuỗi kết thúc lửng lơ (bị cắt cụt giữa chừng),
    # hãy tìm vị trí kết thúc của object hoàn chỉnh cuối cùng (dấu '}')
    # và cắt bỏ phần đuôi lỗi phía sau đi.
    if not raw.endswith("]") and not raw.endswith("}"):
        last_brace = raw.rfind("}")
        if last_brace != -1:
            raw = raw[:last_brace + 1]

    stack: list[str] = []
    in_string = False
    escaped = False
    for char in raw:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            stack.append("]" if char == "[" else "}")
        elif char in "]}":
            if stack and char == stack[-1]:
                stack.pop()

    raw = re.sub(r",\s*$", "", raw)
    return raw + "".join(reversed(stack))


def _json_from_qwen(path: Path, default: dict[str, Any]) -> dict[str, Any] | list[Any]:
    if not path.exists():
        return default
    raw = _repair_json_text(path.read_text(encoding="utf-8"))
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return default
    return value if isinstance(value, (dict, list)) else default


def extract_ocr_directly(raw: str) -> list[str]:
    """Extract OCR strings from valid or malformed Qwen JSON text."""
    match = re.search(r'"ocr"\s*:\s*\[(.*?)(?:\]|$)', raw, flags=re.DOTALL)
    if not match:
        return []
    return [
        value.replace(r"\"", '"').strip()
        for value in re.findall(r'"((?:\\.|[^"\\])*)"', match.group(1))
        if value.replace(r"\"", '"').strip()
    ]


def extract_facts_directly(raw: str) -> list[dict[str, str]]:
    """Extract fact/evidence pairs from valid or malformed Qwen JSON text."""
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    pattern = r'"ocr"\s*:\s*"(.*?)"\s*,\s*"fact"\s*:\s*"(.*?)"'
    return [
        {"ocr": ocr.replace(r"\"", '"'), "fact": fact.replace(r"\"", '"')}
        for ocr, fact in re.findall(pattern, raw, flags=re.DOTALL)
        if fact.strip()
    ]


def _tile_location(tile: dict[str, Any]) -> str:
    return (
        f"row {tile['row']}, column {tile['column']} of the original; "
        # f"pixel rectangle x={tile['x']}, y={tile['y']}, "
        # f"width={tile['width']}, height={tile['height']}"
    )

def detect_tile_boundaries(
    manifest: dict[str, Any],
    output_dir: Path = DEFAULT_PROCESS_TILES,
    num_gpus: int | None = None,
) -> None:
    """Ask Qwen whether each tile cuts content at one or more boundaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for info in manifest["infographics"]:
        for tile in info["tiles"]:
            out = output_dir / f"{Path(tile['filename']).stem}.json"
            prompt = get_boundary_prompt(_tile_location(tile))
            jobs.append((tile, out, prompt))
    caption_images(
        [tile["path"] for tile, _, _ in jobs],
        [str(out) for _, out, _ in jobs],
        prompts=[prompt for _, _, prompt in jobs],
        max_tokens=256,
        num_gpus=num_gpus,
    )
    for tile, out, _ in jobs:
        result = _json_from_qwen(out, {"is_cut": False, "sides": [], "reason": ""})
        sides = result.get("sides", [])
        boundary = {
            "is_cut": bool(result.get("is_cut", False)),
            "sides": [side for side in sides if side in {"left", "right", "top", "bottom"}],
            "reason": str(result.get("reason", "")),
        }
        tile["boundary"] = boundary
        with open(out, 'w', encoding="utf-8") as file:
            json.dump(boundary, file, indent=4, ensure_ascii=False)
        # out.write_text(
        #      + "\n",
        #     encoding="utf-8",
        # )
    (output_dir / "boundary_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _neighbor(tiles: list[dict[str, Any]], tile: dict[str, Any], side: str) -> dict[str, Any] | None:
    row, col = tile["row"], tile["column"]
    candidates = {
        "left": (row, col - 1),
        "right": (row, col + 1),
        "top": (row - 1, col),
        "bottom": (row + 1, col),
    }
    wanted = candidates[side]
    return next((other for other in tiles if (other["row"], other["column"]) == wanted), None)


def merge_processed_tiles(
    manifest: dict[str, Any],
    output_dir: Path = DEFAULT_TILES_AFTER_PROCESS,
    process_dir: Path = DEFAULT_PROCESS_TILES,
) -> dict[str, Any]:
    """Read Qwen boundary JSON files and merge indicated adjacent tiles."""
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = {"version": 1, "infographics": []}
    for info in manifest["infographics"]:
        tiles = info["tiles"]
        consumed: set[str] = set()
        new_tiles = []
        for tile in tiles:
            if tile["component_id"] in consumed:
                continue
            process_path = process_dir / f"{Path(tile['filename']).stem}.json"
            boundary = _json_from_qwen(
                process_path,
                {"is_cut": False, "sides": [], "reason": ""},
            )
            sides = boundary.get("sides", []) if boundary.get("is_cut", False) else []
            sides = [side for side in sides if side in {"left", "right", "top", "bottom"}]
            partner = next((_neighbor(tiles, tile, side) for side in sides
                            if _neighbor(tiles, tile, side) is not None), None)
            members = [tile]
            if partner is not None and partner["component_id"] not in consumed:
                members.append(partner)
            for member in members:
                consumed.add(member["component_id"])
            x1 = min(member["x"] for member in members)
            y1 = min(member["y"] for member in members)
            x2 = max(member["x"] + member["width"] for member in members)
            y2 = max(member["y"] + member["height"] for member in members)
            source = Path(info["original"]["path"])
            with Image.open(source) as original:
                merged = original.convert("RGB").crop((x1, y1, x2, y2))
                target_dir = output_dir / info["id"]
                target_dir.mkdir(parents=True, exist_ok=True)
                suffix = "_merged_" + "_".join(member["component_id"] for member in members)
                target = target_dir / f"{info['id']}{suffix}.png"
                merged.save(target)
            new_tiles.append({
                "component_id": tile["component_id"],
                "source_component_ids": [member["component_id"] for member in members],
                "filename": str(target),
                "path": str(target),
                "row": tile["row"], "column": tile["column"],
                "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
                "boundary": boundary,
            })
        processed["infographics"].append({
            "id": info["id"], "original": info["original"], "tiles": new_tiles,
        })
    (output_dir / "manifest.json").write_text(json.dumps(processed, indent=2), encoding="utf-8")
    return processed


def extract_processed_ocr(
    manifest: dict[str, Any],
    output_dir: Path,
    raw_output_dir: Path | None = None,
    num_gpus: int | None = None,
) -> None:
    """Extract OCR per tile and write raw and cleaned Qwen responses."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir = raw_output_dir or output_dir / "raw"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for info in manifest["infographics"]:
        for tile in info["tiles"]:
            raw_out = raw_output_dir / f"{Path(tile['filename']).stem}.json"
            out = output_dir / raw_out.name
            jobs.append((tile, raw_out, out))
    caption_images(
        [tile["path"] for tile, _, _ in jobs],
        [str(raw_out) for _, raw_out, _ in jobs],
        prompt=OCR_PROMPT,
        max_tokens=1024,
        num_gpus=num_gpus,
    )
    for tile, raw_out, out in jobs:
        ocr = extract_ocr_directly(raw_out.read_text(encoding="utf-8")) if raw_out.exists() else []
        tile["ocr"] = ocr
        out.write_text(json.dumps({"ocr": ocr}, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def extract_facts_each_tile(
    manifest: dict[str, Any],
    output_dir: Path = DEFAULT_FACTS_EACH_TILE,
    ocr_dir: Path = DEFAULT_OCR_EACH_TILE,
    raw_output_dir: Path | None = None,
    num_gpus: int | None = None,
) -> None:
    """Generate facts from cleaned OCR files and save raw and clean JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir = raw_output_dir or output_dir / "raw"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for info in manifest["infographics"]:
        for tile in info["tiles"]:
            ocr_path = ocr_dir / f"{Path(tile['filename']).stem}.json"
            ocr_raw = json.loads(ocr_path.read_text(encoding="utf-8")) if ocr_path.exists() else {}
            ocr_items = ocr_raw.get("ocr", []) if isinstance(ocr_raw, dict) else []
            ocr_items = [str(item).strip() for item in ocr_items if str(item).strip()]
            tile["ocr"] = ocr_items
            prompt = FACTS_FROM_OCR_PROMPT + "\n".join(
                f"{index + 1}. {text}" for index, text in enumerate(ocr_items)
            )
            raw_out = raw_output_dir / f"{Path(tile['filename']).stem}.json"
            out = output_dir / f"{Path(tile['filename']).stem}.json"
            jobs.append((tile, info["original"]["path"], raw_out, out, prompt))
    caption_images(
        [[tile["path"], original] for tile, original, _, _, _ in jobs],
        [str(raw_out) for _, _, raw_out, _, _ in jobs],
        prompts=[prompt for _, _, _, _, prompt in jobs],
        max_tokens=1024,
        num_gpus=num_gpus,
    )
    for tile, _, raw_out, out, _ in jobs:
        valid_items = (
            extract_facts_directly(raw_out.read_text(encoding="utf-8"))
            if raw_out.exists()
            else []
        )
        tile["facts"] = [item["fact"] for item in valid_items]
        out.write_text(
            json.dumps(valid_items, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def embed_serializations(top_path: Path, low_path: Path, facts_path: Path, output_dir: Path, num_gpus: int) -> None:
    """Encode serialized inputs using MM-Embed."""
    embedding_dir = output_dir / "MM-Embed"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    for path, name, bsize in ((top_path, "image", 1), (low_path, "subimage", 4), (facts_path, "tile_fact", 4)):
        encode_one_corpus(
            embedder_cls=MMEmbed,
            tmp_prefix=f"tile_{name}_",
            corpus_in_filepath=str(path),
            out_embeddings_path=str(embedding_dir / f"{name}.pt"),
            out_index_path=str(embedding_dir / f"{name}.json"),
            num_gpus=num_gpus,
            max_length=4096,
            batch_size=bsize,
        )


def generate_processed_tile_summaries(
    tile_records: list[tuple[dict[str, Any], dict[str, Any], Path]],
    summary_root: Path,
    *,
    use_qwen_summaries: bool = False,
    num_gpus: int = 1,
) -> None:
    """Generate Qwen summaries for processed tiles when enabled."""
    summary_jobs: list[tuple[list[str], Path, str]] = []
    for _, original, processed_path in tile_records:
        summary_jobs.append((
            [str(processed_path), str(original["path"])],
            summary_root / "tile_summaries" / f"{processed_path.stem}.txt",
            (
                "The first image is an infographic tile and the second image is the "
                "complete original infographic. Summarize only the tile, using the "
                "original image as context. Include visible text, numbers, entities, "
                "and the tile's main visual meaning. Be concise and factual."
            ),
        ))
    if use_qwen_summaries:
        caption_images(
            [paths for paths, _, _ in summary_jobs],
            [str(out) for _, out, _ in summary_jobs],
            prompts=[prompt for _, _, prompt in summary_jobs],
            max_tokens=1024,
            num_gpus=num_gpus,
        )


def generate_image_summaries(
    manifest: dict[str, Any],
    output_dir: Path = DEFAULT_IMAGE_SUMMARIES,
    num_gpus: int = 1,
) -> None:
    """Generate one Qwen summary for each original infographic."""
    output_dir.mkdir(parents=True, exist_ok=True)
    infos = manifest.get("infographics", [])
    if not infos:
        return
    caption_images(
        [info["original"]["path"] for info in infos],
        [
            str(output_dir / f"{Path(info['original']['filename']).stem}.txt")
            for info in infos
        ],
        prompt=(
            "Summarize this infographic. Include its visible text, main topic, "
            "important numbers, entities, and overall visual meaning. Be concise "
            "and factual."
        ),
        max_tokens=1024,
        num_gpus=num_gpus,
    )


def serialize_processed_assets(
    manifest: dict[str, Any],
    tiles_after_process_dir: Path = DEFAULT_TILES_AFTER_PROCESS,
    facts_dir: Path = DEFAULT_FACTS_EACH_TILE,
    output_dir: Path = DEFAULT_TILES_AFTER_PROCESS.parent,
    *,
    use_qwen_summaries: bool = False,
    summaries_dir: Path | None = None,
    num_gpus: int = 1,
) -> tuple[Path, Path, Path]:
    """Serialize processed tiles, originals, and facts for MM-Embed.

    Processed tile images are discovered recursively below
    ``tiles_after_process_dir``. Fact files are matched by the processed tile
    stem. When ``use_qwen_summaries`` is enabled, Qwen creates summaries for
    processed tiles only; original infographic summaries are read from their
    existing summary files.
    """
    if not tiles_after_process_dir.is_dir():
        raise FileNotFoundError(tiles_after_process_dir)
    if not facts_dir.is_dir():
        raise FileNotFoundError(facts_dir)
    if num_gpus < 1:
        raise ValueError("num_gpus must be at least 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    serialization_dir = output_dir / "serializations"
    serialization_dir.mkdir(parents=True, exist_ok=True)
    summary_root = summaries_dir or ARTIFACTS_ROOT

    image_paths = [
        path for path in sorted(tiles_after_process_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS
    ]
    if not image_paths:
        raise ValueError(f"No processed tile images found under {tiles_after_process_dir}")

    tile_by_stem = {
        path.stem: path
        for path in image_paths
    }
    tile_records: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    top_records: list[tuple[str, Path, str]] = []
    for info in manifest.get("infographics", []):
        original = info["original"]
        original_path = Path(original.get("path", original["filename"]))
        if not original_path.is_file():
            raise FileNotFoundError(original_path)
        original_name = Path(original["filename"]).name
        top_records.append((original_name, original_path, Path(original_name).stem))
        for tile in info.get("tiles", []):
            original_tile_path = Path(tile.get("path", tile["filename"]))
            processed_path = tile_by_stem.get(Path(tile["filename"]).stem)
            if processed_path is None:
                processed_path = tile_by_stem.get(Path(original_tile_path).stem)
            if processed_path is None:
                continue
            tile_records.append((tile, original, processed_path))

    if not tile_records:
        raise ValueError("No manifest tiles matched processed tile images")

    generate_processed_tile_summaries(
        tile_records,
        summary_root,
        use_qwen_summaries=use_qwen_summaries,
        num_gpus=num_gpus,
    )

    top, low, facts = [], [], []
    for original_name, original_path, stem in top_records:
        summary_path = summary_root / "image_summaries" / f"{stem}.txt"
        summary = summary_path.read_text(encoding="utf-8").strip() if summary_path.exists() else ""
        top.append({
            "id": [original_name, "i_1"],
            "target": {
                "text": f"{stem} [SEP] {summary}".strip(),
                "images": [str(original_path)],
            },
        })

    for tile, original, processed_path in tile_records:
        original_name = Path(original["filename"]).name
        tile_id = tile["component_id"]
        location = (
            f"row={tile.get('row', '?')} column={tile.get('column', '?')} "
            f"x={tile.get('x', '?')} y={tile.get('y', '?')} "
            # f"width={tile.get('width', '?')} height={tile.get('height', '?')}"
        )
        summary_path = summary_root / "tile_summaries" / f"{processed_path.stem}.txt"
        summary = summary_path.read_text(encoding="utf-8").strip() if summary_path.exists() else ""
        low.append({
            "id": [original_name, f"{processed_path.name}", f"{tile_id}"],
            "target": {
                "text": f"{Path(original_name).stem} [SEP] a tile at location {location} of infographic {Path(original_name).stem}.jpeg [SEP] {summary}".strip(),
                "images": [str(processed_path)],
            },
        })
        fact_path = facts_dir / f"{processed_path.stem}.json"
        if not fact_path.exists():
            continue
        raw_facts = json.loads(fact_path.read_text(encoding="utf-8"))
        fact_items = raw_facts.get("facts", []) if isinstance(raw_facts, dict) else raw_facts
        if not isinstance(fact_items, list):
            continue
        for index, fact in enumerate(fact_items):
            if isinstance(fact, dict):
                fact_text = str(fact.get("fact", fact.get("text", ""))).strip()
                evidence = str(fact.get("ocr", fact.get("evidence", ""))).strip()
            else:
                fact_text, evidence = str(fact).strip(), ""
            if not fact_text:
                continue
            facts.append({
                "id": [original_name, f"{processed_path.name}", f"{tile_id}_f{index:04d}"],
                "target": {
                    "text": f"{Path(original_name).stem} [SEP] {fact_text} [SEP] OCR [SEP] {evidence}",
                    "images": [],
                },
            })

    top_path = serialization_dir / "image.json"
    low_path = serialization_dir / "subimage.json"
    facts_path = serialization_dir / "tile_fact.json"
    for path, data in ((top_path, top), (low_path, low), (facts_path, facts)):
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return top_path, low_path, facts_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiled InfoVQA Corpus Builder")
    parser.add_argument("--estimated-components", type=Path, default=DEFAULT_ESTIMATES)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--max-tiles", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--generate-image-summaries", action="store_true")
    parser.add_argument("--generate-tile-summaries", action="store_true")
    args = parser.parse_args()

    print("Running prepare_tiled_inputs...", flush=True)
    manifest = prepare_tiled_inputs(
        DEFAULT_INPUT,
        DEFAULT_TILES,
        DEFAULT_TILES / "manifest.json",
        overlap=args.overlap,
        max_tiles=args.max_tiles,
        estimated_components_path=args.estimated_components,
    )

    print("Running detect_tile_boundaries...", flush=True)
    detect_tile_boundaries(manifest, DEFAULT_PROCESS_TILES, args.num_gpus)
    print("Running merge_processed_tiles...", flush=True)
    processed_manifest = merge_processed_tiles(
        manifest, DEFAULT_TILES_AFTER_PROCESS, DEFAULT_PROCESS_TILES
    )

    print("Running extract_processed_ocr...", flush=True)
    extract_processed_ocr(
        processed_manifest,
        DEFAULT_OCR_EACH_TILE,
        DEFAULT_OCR_EACH_TILE / "raw",
        args.num_gpus,
    )
    print("Running extract_facts_each_tile...", flush=True)
    extract_facts_each_tile(
        processed_manifest,
        DEFAULT_FACTS_EACH_TILE,
        DEFAULT_OCR_EACH_TILE,
        DEFAULT_FACTS_EACH_TILE / "raw",
        args.num_gpus,
    )

    if args.generate_image_summaries:
        print("Running generate_image_summaries...", flush=True)
        generate_image_summaries(processed_manifest, DEFAULT_IMAGE_SUMMARIES, args.num_gpus)

    if args.generate_tile_summaries:
        print("Running generate_processed_tile_summaries...", flush=True)
        tile_records = [
            (tile, info["original"], Path(tile["path"]))
            for info in processed_manifest["infographics"]
            for tile in info["tiles"]
        ]
        generate_processed_tile_summaries(
            tile_records,
            ARTIFACTS_ROOT,
            use_qwen_summaries=True,
            num_gpus=args.num_gpus,
        )

    print("Running serialize_processed_assets...", flush=True)
    top_path, low_path, facts_path = serialize_processed_assets(
        processed_manifest,
        DEFAULT_TILES_AFTER_PROCESS,
        DEFAULT_FACTS_EACH_TILE,
        ARTIFACTS_ROOT / "embeddings",
        summaries_dir=ARTIFACTS_ROOT,
        num_gpus=args.num_gpus,
    )

    print("Running embed_serializations...", flush=True)
    embed_serializations(
        top_path,
        low_path,
        facts_path,
        ARTIFACTS_ROOT / "embeddings",
        args.num_gpus,
    )

    # # artifacts_folder = args.process_tiles_dir.parent
    # # print("Running serialize_tiles...", flush=True)
    # # top_path, low_path, facts_path = serialize_tiles(processed_manifest, artifacts_folder)

    # # if not args.skip_embed:
    # #     print("Running embed_serializations...", flush=True)
    # #     embed_serializations(top_path, low_path, facts_path, artifacts_folder, args.num_gpus)


if __name__ == "__main__":
    main()
    # path = Path("/workspace/LILaC/artifacts/InfoVQA/facts_each_tile/raw/70574_merged_i_1_t0003.json")
    # value = _json_from_qwen(path, {"facts": []})
    # print(value)
    