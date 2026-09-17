#!/usr/bin/env python3
"""Build a tile-based InfoVQA corpus using aspect-aware infographic grid tiling."""
from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image
import torch

from src.lilac.lcg_constructor.preprocessing._caption_via_qwen_vl import (
    caption_images,
)
from src.models.embedder.mmembed import MMEmbed
from src.models.embedder.sharded_runner import encode_one_corpus
from src.utils.utils import REPO_ROOT

DEFAULT_INPUT = Path(REPO_ROOT) / "datasets/InfoVQA/image_components/test"
DEFAULT_ROOT = Path(REPO_ROOT) / "datasets/InfoVQA/tiles"
DEFAULT_ESTIMATES = Path(REPO_ROOT) / "debug/estimated_components.json"
DEFAULT_IMAGE_SUMMARIES = Path(REPO_ROOT) / "artifacts/InfoVQA/image_summaries/test"
DEFAULT_MINERU_TILES = Path(REPO_ROOT) / "datasets/InfoVQA/tiles_mineru"
DEFAULT_PROCESS_TILES = Path(REPO_ROOT) / "artifacts/InfoVQA/process_tiles"
DEFAULT_TILES_AFTER_PROCESS = Path(REPO_ROOT) / "artifacts/InfoVQA/tiles_after_process"
DEFAULT_FACTS_EACH_TILE = Path(REPO_ROOT) / "artifacts/InfoVQA/facts_each_tile"
DEFAULT_OCR_EACH_TILE = Path(REPO_ROOT) / "artifacts/InfoVQA/ocr_each_tile"
TILE_PROMPT = (
    "The first image is a tile from an infographic and the second image is "
    "the complete original infographic. Describe only the tile, but use the "
    "complete infographic as context. Transcribe all visible text and explain "
    "the tile's spatial role. Be factual and concise."
)
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
FACT_PROMPT = (
    "You are an expert at extracting and synthesizing verifiable facts from infographic tiles. "
    "The first image is the tile and the second is the original infographic. "
    "Use the OCR items below as primary evidence.\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. Do NOT just blindly copy or repeat fragmented OCR items into the 'text' field. "
    "Instead, synthesize and rewrite the fragmented pieces into a **complete, concise, and coherent factual sentence**.\n"
    "2. In the 'evidence' field, quote the exact original OCR text used as your source.\n"
    "3. Do not invent facts, do not extrapolate beyond what the image shows, and do not include markdown.\n\n"
    "Example Output Format:\n"
    '{"facts":[\n'
    '  {"text": "There were 907K total spam messages related to COVID-19.", "evidence": "907K Total spam messages related to COVID-19"},\n'
    '  {"text": "The United States is the top location for spam and malware detections.", "evidence": "United States Top location for spam and malware detections"}\n'
    "]}\n\n"
    "OCR items:\n"
)

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


def caption_tiles(manifest: dict[str, Any], output_dir: Path, num_gpus: int | None) -> None:
    """Generate sidecar summaries for tiles via Qwen-VL."""
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for info in manifest["infographics"]:
        orig = info["original"]["path"]
        for tile in info["tiles"]:
            out_file = summaries_dir / f"{Path(tile['filename']).stem}.txt"
            jobs.append((tile, orig, out_file))

    if not jobs:
        return

    caption_images(
        image_paths=[[tile["path"], orig] for tile, orig, _ in jobs],
        output_paths=[str(out) for _, _, out in jobs],
        prompt=TILE_PROMPT,
        max_tokens=1024,
        num_gpus=num_gpus,
    )

    # Đọc trực tiếp caption vào object manifest
    for tile, _, out in jobs:
        tile["caption"] = out.read_text(encoding="utf-8").strip() if out.exists() else ""

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _repair_json_text(raw: str) -> str:
    """Recover common Qwen JSON mistakes, such as truncated closing brackets."""
    raw = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()

    start = next((index for index, char in enumerate(raw) if char in "[{"), None)
    if start is None:
        return raw
    raw = raw[start:]

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

    # A response may end with a comma immediately before truncation.
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
    num_gpus: int | None = None,
) -> None:
    """Extract OCR per tile while preserving the raw Qwen response."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir = output_dir / "raw"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for info in manifest["infographics"]:
        for tile in info["tiles"]:
            out = output_dir / f"{Path(tile['filename']).stem}.json"
            raw_out = raw_output_dir / out.name
            jobs.append((tile, raw_out, out))
    caption_images(
        [tile["path"] for tile, _, _ in jobs],
        [str(raw_out) for _, raw_out, _ in jobs],
        prompt=OCR_PROMPT,
        max_tokens=1024,
        num_gpus=num_gpus,
    )
    for tile, raw_out, out in jobs:
        result = _json_from_qwen(raw_out, {"ocr": []})
        ocr_items = result.get("ocr", []) if isinstance(result, dict) else result
        ocr = [str(item).strip() for item in ocr_items
               if str(item).strip()] if isinstance(ocr_items, list) else []
        tile["ocr"] = ocr
        out.write_text(json.dumps({"ocr": ocr}, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def extract_facts_each_tile(
    manifest: dict[str, Any],
    output_dir: Path = DEFAULT_FACTS_EACH_TILE,
    num_gpus: int | None = None,
) -> None:
    """Generate one fact per OCR item and save one JSON file per tile."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir = output_dir / "raw"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for info in manifest["infographics"]:
        for tile in info["tiles"]:
            ocr_items = tile.get("ocr", [])
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
        result = _json_from_qwen(raw_out, {"facts": []})
        if isinstance(result, dict):
            items = result.get("facts", [])
        else:
            # Qwen sometimes omits the requested {"facts": ...} wrapper.
            items = result
        facts = []
        valid_items = []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                fact = str(item.get("fact", "")).strip()
                if not fact:
                    continue
                facts.append(fact)
                valid_items.append({
                    "ocr": str(item.get("ocr", "")).strip(),
                    "fact": fact,
                })
        tile["facts"] = facts
        out.write_text(json.dumps(valid_items, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def cluster_facts_each_tile(
    facts_dir: Path = DEFAULT_FACTS_EACH_TILE,
    output_dir: Path | None = None,
    *,
    similarity_threshold: float = 0.9,
    num_gpus: int = 1,
) -> dict[str, int]:
    """Embed facts and remove near-duplicate facts independently per tile.

    The first fact in each similarity cluster is retained. Clustering is
    greedy against retained representatives, so a long chain of moderately
    similar facts cannot merge unrelated facts transitively.
    """
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if num_gpus < 1:
        raise ValueError("num_gpus must be at least 1")
    if output_dir is None:
        output_dir = facts_dir.parent / f"{facts_dir.name}_clustered"
    output_dir.mkdir(parents=True, exist_ok=True)

    fact_files = sorted(
        path for path in facts_dir.glob("*.json")
        if path.name != "manifest.json"
    )
    records: list[tuple[Path, list[dict[str, Any]]]] = []
    corpus: list[dict[str, Any]] = []
    for path in fact_files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.get("facts", []) if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            raise ValueError(f"Expected a fact list in {path}")
        valid_items = [
            item for item in items
            if isinstance(item, dict) and str(item.get("fact", "")).strip()
        ]
        records.append((path, valid_items))
        corpus.extend(
            {"id": [path.name, str(index)], "target": {"text": item["fact"]}}
            for index, item in enumerate(valid_items)
        )

    if not corpus:
        return {path.name: 0 for path, _ in records}

    with tempfile.TemporaryDirectory(prefix="fact_cluster_") as work_dir:
        work_path = Path(work_dir)
        corpus_path = work_path / "facts.json"
        embedding_path = work_path / "facts.pt"
        index_path = work_path / "facts.jsonl"
        corpus_path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
        encode_one_corpus(
            embedder_cls=MMEmbed,
            tmp_prefix="fact_cluster_embed_",
            corpus_in_filepath=str(corpus_path),
            out_embeddings_path=str(embedding_path),
            out_index_path=str(index_path),
            num_gpus=num_gpus,
            max_length=4096,
            batch_size=16,
        )
        embeddings = torch.load(embedding_path, map_location="cpu")
        embeddings = torch.nn.functional.normalize(embeddings.float(), dim=1)

    offset = 0
    summary: dict[str, int] = {}
    cluster_metadata: dict[str, list[dict[str, Any]]] = {}
    for path, items in records:
        retained: list[dict[str, Any]] = []
        representatives: list[torch.Tensor] = []
        clusters: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            vector = embeddings[offset + index]
            if representatives:
                similarities = torch.stack(
                    [torch.dot(vector, representative) for representative in representatives]
                )
                best_cluster = int(torch.argmax(similarities).item())
                best_similarity = float(similarities[best_cluster].item())
            else:
                best_cluster = -1
                best_similarity = -1.0

            if best_similarity >= similarity_threshold:
                clusters[best_cluster]["member_indices"].append(index)
                clusters[best_cluster]["similarities"].append(best_similarity)
            else:
                retained.append(item)
                representatives.append(vector)
                clusters.append({
                    "representative_index": index,
                    "member_indices": [index],
                    "similarities": [1.0],
                })
        offset += len(items)
        out_path = output_dir / path.name
        out_path.write_text(
            json.dumps(retained, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        cluster_metadata[path.name] = clusters
        summary[path.name] = len(retained)

    (output_dir / "clusters.json").write_text(
        json.dumps(cluster_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _content_items(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    blocks = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
    if not isinstance(blocks, list):
        return []
    items = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        content = block.get("content", {})
        if not isinstance(content, dict):
            continue
        typed = content.get(f"{block.get('type', '')}_content", [])
        if not isinstance(typed, list):
            continue
        for item in typed:
            if isinstance(item, dict):
                text = str(item.get("content", "")).strip()
                if text and text != "[No text]":
                    items.append(text)
    return items


def _find_mineru_content(mineru_dir: Path, tile: dict[str, Any]) -> Path | None:
    stem = Path(tile["filename"]).stem
    candidates = list(mineru_dir.rglob(f"{stem}_content_list*.json"))
    if candidates:
        return candidates[0]
    candidates = list(mineru_dir.rglob("*content_list*.json"))
    tile_dir = Path(tile["filename"]).parent.name
    for candidate in candidates:
        if stem in candidate.name or stem in str(candidate.parent):
            return candidate
        if tile_dir in candidate.name or tile_dir in str(candidate.parent):
            return candidate
    return None


def extract_tile_facts(
    manifest: dict[str, Any],
    output_dir: Path,
    mineru_dir: Path = DEFAULT_MINERU_TILES,
    num_gpus: int | None = None,
) -> None:
    """Use OCR content_list items plus both images to produce per-tile JSON."""
    facts_dir = output_dir / "facts"
    facts_dir.mkdir(parents=True, exist_ok=True)
    jobs, ocr_by_tile = [], {}
    for info in manifest["infographics"]:
        for tile in info["tiles"]:
            content_path = _find_mineru_content(mineru_dir, tile)
            if content_path is None:
                raise FileNotFoundError(
                    f"No content_list JSON found for tile {tile['filename']} "
                    f"under {mineru_dir}"
                )
            ocr = _content_items(content_path)
            ocr_by_tile[tile["filename"]] = ocr
            out = facts_dir / f"{Path(tile['filename']).stem}.json"
            prompt = FACT_PROMPT + "\n".join(f"- {item}" for item in ocr)
            jobs.append((tile, info["original"]["path"], out, prompt))

    caption_images(
        [[tile["path"], original] for tile, original, _, _ in jobs],
        [str(out) for _, _, out, _ in jobs],
        prompts=[prompt for _, _, _, prompt in jobs],
        max_tokens=1024,
        num_gpus=num_gpus,
    )
    for tile, _, out, _ in jobs:
        try:
            raw_text = out.read_text(encoding="utf-8") if out.exists() else ""
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw_text) if raw_text else {}
        except json.JSONDecodeError:
            parsed = {"facts": [{"text": out.read_text(encoding="utf-8").strip(),
                                 "evidence": ""}]} if out.exists() else {}
        facts = parsed.get("facts", []) if isinstance(parsed, dict) else []
        tile["facts"] = facts if isinstance(facts, list) else []
        tile["ocr_items"] = ocr_by_tile[tile["filename"]]
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def serialize_tiles(
    manifest: dict[str, Any],
    output_dir: Path,
    summaries_dir: Path = DEFAULT_IMAGE_SUMMARIES,
) -> tuple[Path, Path, Path]:
    """Format top-level infographics and tile representations for retriever embedding."""
    serialization_dir = output_dir / "serializations"
    serialization_dir.mkdir(parents=True, exist_ok=True)
    top, low, facts = [], [], []

    for info in manifest["infographics"]:
        orig_name = Path(info["original"]["filename"]).name
        orig_path = info["original"]["path"]
        summary_path = summaries_dir / f"{Path(orig_name).stem}.txt"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing image summary: {summary_path}")
        summary = summary_path.read_text(encoding="utf-8").strip()
        top.append({
            "id": [orig_name, "i_1"],
            "target": {
                "text": f"{Path(orig_name).stem} [SEP] {summary}",
                "images": [orig_path],
            },
        })
        for tile in info["tiles"]:
            loc = f"row={tile['row']} column={tile['column']} x={tile['x']} y={tile['y']} width={tile['width']} height={tile['height']}"
            low.append({
                "id": [orig_name, tile["component_id"]],
                "target": {
                    "text": f"{Path(orig_name).stem} [SEP] a tile at location: {loc} [SEP] {tile.get('caption', '')}",
                    "images": [tile["path"]],
                },
            })
            for fact_index, fact in enumerate(tile.get("facts", [])):
                fact_id = f"{tile['component_id']}_f{fact_index:04d}"
                facts.append({
                    "id": [orig_name, fact_id],
                    "target": {
                        "text": f"{Path(orig_name).stem} [SEP] tile fact [SEP] "
                                     f"{fact.get('fact', fact.get('text', ''))} "
                                     f"[SEP] OCR [SEP] {fact.get('ocr', fact.get('evidence', ''))}",
                        "images": [tile["path"], orig_path],
                    },
                })

    top_path = serialization_dir / "image.json"
    low_path = serialization_dir / "subimage.json"
    facts_path = serialization_dir / "tile_fact.json"
    top_path.write_text(json.dumps(top, indent=2), encoding="utf-8")
    low_path.write_text(json.dumps(low, indent=2), encoding="utf-8")
    facts_path.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    return top_path, low_path, facts_path


def embed_serializations(top_path: Path, low_path: Path, facts_path: Path, output_dir: Path, num_gpus: int) -> None:
    """Encode serialized inputs using MM-Embed."""
    embedding_dir = output_dir / "embeddings" / "MM-Embed"
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
    both originals and processed tiles; otherwise existing summaries are used
    when available and a location-based text description is emitted.
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
    summary_root = summaries_dir or output_dir / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)

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

    summary_jobs: list[tuple[Path, Path, str]] = []
    for original_name, original_path, stem in top_records:
        summary_jobs.append((
            original_path,
            summary_root / f"{stem}.txt",
            (
                "Summarize this infographic for multimodal retrieval. Include its main "
                "topic, important labels, numbers, and relationships. Be concise and factual."
            ),
        ))
    for tile, original, processed_path in tile_records:
        summary_jobs.append((
            processed_path,
            summary_root / f"{processed_path.stem}.txt",
            (
                "Summarize this infographic tile for retrieval. Include visible text, "
                "numbers, entities, and the tile's main visual meaning. Be concise and factual."
            ),
        ))
    if use_qwen_summaries:
        caption_images(
            [str(path) for path, _, _ in summary_jobs],
            [str(out) for _, out, _ in summary_jobs],
            prompts=[prompt for _, _, prompt in summary_jobs],
            max_tokens=1024,
            num_gpus=num_gpus,
        )

    top, low, facts = [], [], []
    for original_name, original_path, stem in top_records:
        summary_path = summary_root / f"{stem}.txt"
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
            f"width={tile.get('width', '?')} height={tile.get('height', '?')}"
        )
        summary_path = summary_root / f"{processed_path.stem}.txt"
        summary = summary_path.read_text(encoding="utf-8").strip() if summary_path.exists() else ""
        low.append({
            "id": [original_name, tile_id],
            "target": {
                "text": f"{Path(original_name).stem} [SEP] {location} [SEP] {summary}".strip(),
                "images": [str(processed_path), str(original.get("path", original["filename"]))],
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
                "id": [original_name, f"{tile_id}_f{index:04d}"],
                "target": {
                    "text": f"{Path(original_name).stem} [SEP] {fact_text} [SEP] OCR [SEP] {evidence}",
                    "images": [str(processed_path), str(original.get("path", original["filename"]))],
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
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tile-width", type=int, default=None)
    parser.add_argument("--tile-height", type=int, default=None)
    parser.add_argument("--estimated-components", type=Path, default=DEFAULT_ESTIMATES)
    parser.add_argument("--mineru-tiles-dir", type=Path, default=DEFAULT_MINERU_TILES)
    parser.add_argument("--process-tiles-dir", type=Path, default=DEFAULT_PROCESS_TILES)
    parser.add_argument("--tiles-after-process-dir", type=Path, default=DEFAULT_TILES_AFTER_PROCESS)
    parser.add_argument("--facts-dir", type=Path, default=DEFAULT_FACTS_EACH_TILE)
    parser.add_argument("--ocr-dir", type=Path, default=DEFAULT_OCR_EACH_TILE)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--max-tiles", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--cluster-facts", action="store_true")
    parser.add_argument("--fact-similarity-threshold", type=float, default=0.9)
    parser.add_argument("--serialize-processed", action="store_true")
    parser.add_argument("--use-qwen-summaries", action="store_true")
    parser.add_argument("--processed-manifest", type=Path, default=None)
    args = parser.parse_args()

    # manifest = {}
    # manifest_file = "/workspace/LILaC/datasets/InfoVQA/tiles/manifest.json"
    # with open(manifest_file, encoding="utf-8") as file:
    #     manifest = json.load(file)
    # print("Running prepare_tiled_inputs...", flush=True)
    # manifest = prepare_tiled_inputs(
    #     args.input_dir,
    #     args.output_dir,
    #     args.output_dir / "manifest.json",
    #     tile_width=args.tile_width,
    #     tile_height=args.tile_height,
    #     overlap=args.overlap,
    #     max_tiles=args.max_tiles,
    #     estimated_components_path=args.estimated_components,
    # )

    if not args.skip_qwen:
        # print("Running detect_tile_boundaries...", flush=True)
        # detect_tile_boundaries(manifest, args.process_tiles_dir, args.num_gpus)
        # print("Running merge_processed_tiles...", flush=True)
        # processed_manifest = merge_processed_tiles(
        #     manifest, args.tiles_after_process_dir, args.process_tiles_dir
        # )
        processed_manifest = {}
        processed_manifest_file = "/workspace/LILaC/artifacts/InfoVQA/ocr_each_tile/manifest_test.json"
        with open(processed_manifest_file, 'r') as file:
            processed_manifest = json.load(file)
        # print("Running extract_processed_ocr...", flush=True)
        # extract_processed_ocr(
        #     processed_manifest, args.ocr_dir, args.num_gpus
        # )
        print("Running extract_facts_each_tile...", flush=True)
        extract_facts_each_tile(
            processed_manifest, args.facts_dir, args.num_gpus
        )
        # manifest = processed_manifest

    if args.serialize_processed or args.use_qwen_summaries:
        manifest_path = args.processed_manifest or (
            args.tiles_after_process_dir / "manifest.json"
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Processed manifest is required for serialization: {manifest_path}"
            )
        processed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print("Running serialize_processed_assets...", flush=True)
        top_path, low_path, facts_path = serialize_processed_assets(
            processed_manifest,
            args.tiles_after_process_dir,
            args.facts_dir,
            args.process_tiles_dir.parent,
            use_qwen_summaries=args.use_qwen_summaries,
            num_gpus=args.num_gpus,
        )
        if not args.skip_embed:
            print("Running embed_serializations...", flush=True)
            embed_serializations(
                top_path,
                low_path,
                facts_path,
                args.process_tiles_dir.parent,
                args.num_gpus,
            )

    # artifacts_folder = args.process_tiles_dir.parent
    # print("Running serialize_tiles...", flush=True)
    # top_path, low_path, facts_path = serialize_tiles(processed_manifest, artifacts_folder)

    # if not args.skip_embed:
    #     print("Running embed_serializations...", flush=True)
    #     embed_serializations(top_path, low_path, facts_path, artifacts_folder, args.num_gpus)


if __name__ == "__main__":
    main()