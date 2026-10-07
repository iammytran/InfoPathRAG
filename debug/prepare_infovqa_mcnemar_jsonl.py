"""Convert InfoVQA retrieval output into per-query McNemar evaluation records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _document_id(node: list[str]) -> str:
    name = str(node[0])
    for suffix in (".json", ".jpeg", ".jpg", ".png"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _component_id(node: list[str]) -> str:
    document = _document_id(node)
    component = str(node[1])
    component_number = component.removeprefix("i_")
    return f"{document}_component_{component_number}"


def _gold_by_qid(qa_path: Path) -> dict[str, dict]:
    gold = {}
    for item in json.loads(qa_path.read_text()):
        images = {
            _document_id([evidence["gold_image"], "i_1"])
            for evidence in item.get("evidences", [])
            if evidence.get("gold_image") is not None
        }
        gold[str(item["qid"])] = {
            "query": item.get("question"),
            "gold_infographic_ids": sorted(images),
        }
    return gold


def convert(input_path: Path, qa_path: Path) -> list[dict]:
    gold = _gold_by_qid(qa_path)
    records = []
    for raw in input_path.read_text().splitlines():
        if not raw.strip():
            continue
        result = json.loads(raw)
        qid = str(result["qid"])
        if qid not in gold:
            raise ValueError(f"Missing gold label for qid {qid}")

        components = []
        ranked_ids = []
        ranked_scores = []
        seen_ids = set()
        if "retrieved_units" in result:
            units = result["retrieved_units"]
            for unit in units:
                nodes = unit.get("nodes") or []
                if not nodes:
                    continue
                node = nodes[0]
                infographic_id = _document_id(node)
                score = float(unit["score"])
                components.append({
                    "component_id": _component_id(node),
                    "infographic_id": infographic_id,
                    "score": score,
                })
                if infographic_id not in seen_ids:
                    seen_ids.add(infographic_id)
                    ranked_ids.append(infographic_id)
                    ranked_scores.append(score)
        else:
            selected_paths = result.get("selected_paths", [])
            score_by_infographic = {}
            for path in selected_paths:
                root = path.get("root_id") or path.get("root")
                if not root:
                    continue
                infographic_id = _document_id(root)
                score = float(path.get("final_score", path.get("score", 0.0)))
                score_by_infographic.setdefault(infographic_id, score)
                components.append({
                    "component_id": "_".join(
                        str(part) for part in (path.get("fact_id") or [])
                    ),
                    "infographic_id": infographic_id,
                    "score": score,
                })
            for infographic_id, score in sorted(
                score_by_infographic.items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                ranked_ids.append(infographic_id)
                ranked_scores.append(score)

        gold_ids = set(gold[qid]["gold_infographic_ids"])
        gold_rank = next(
            (rank for rank, infographic_id in enumerate(ranked_ids, start=1)
             if infographic_id in gold_ids),
            None,
        )
        records.append({
            "query_id": qid,
            "query": gold[qid]["query"],
            "gold_infographic_ids": gold[qid]["gold_infographic_ids"],
            "ranked_infographic_ids": ranked_ids,
            "ranked_scores": ranked_scores,
            "gold_rank": gold_rank,
            "hit_at_3": gold_rank is not None and gold_rank <= 3,
            "reciprocal_rank_at_10": (
                1 / gold_rank if gold_rank is not None and gold_rank <= 10 else 0.0
            ),
            "retrieved_components": components[:10],
            "retrieval_time_ms": (
                result.get("time", {}).get("retrieval_time(ms)")
                or result.get("retrieval_time_ms")
            ),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--qa-path", type=Path, default=Path("datasets/InfoVQA/QAs_test.json")
    )
    args = parser.parse_args()
    records = convert(args.input, args.qa_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
