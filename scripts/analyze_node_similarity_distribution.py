#!/usr/bin/env python3
"""Analyze query-to-node cosine-similarity distributions for InfoVQA.

The three node embedding paths below are intentionally placeholders. Replace
them, or pass the corresponding CLI arguments, before running the script.

The script reports the sampled cosine-similarity distribution separately for
every sampled query and node type. It also emits heuristic diagnostics for
per-query z-score normalization. These diagnostics are evidence for deciding
whether normalization is stable; they are not a statistical hypothesis test.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QA_PATH = REPO_ROOT / "datasets/InfoVQA/QAs_test.json"

# Replace these placeholders with the node embedding artifacts.
FACT_EMBEDDINGS_PT = Path("/path/to/fact_nodes.pt")
FACT_INDEX_JSON = Path("/path/to/fact_nodes.json")
ROOT_EMBEDDINGS_PT = Path("/path/to/root_nodes.pt")
ROOT_INDEX_JSON = Path("/path/to/root_nodes.json")
TILE_EMBEDDINGS_PT = Path("/path/to/tile_nodes.pt")
TILE_INDEX_JSON = Path("/path/to/tile_nodes.json")

# Query embeddings must be generated with the same embedding model as nodes.
QUERY_EMBEDDINGS_PT = Path("/path/to/infovqa_query_embeddings.pt")
QUERY_INDEX_JSON: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--query-embeddings", type=Path, default=QUERY_EMBEDDINGS_PT)
    parser.add_argument("--query-index", type=Path, default=QUERY_INDEX_JSON)
    parser.add_argument("--fact-embeddings", type=Path, default=FACT_EMBEDDINGS_PT)
    parser.add_argument("--fact-index", type=Path, default=FACT_INDEX_JSON)
    parser.add_argument("--root-embeddings", type=Path, default=ROOT_EMBEDDINGS_PT)
    parser.add_argument("--root-index", type=Path, default=ROOT_INDEX_JSON)
    parser.add_argument("--tile-embeddings", type=Path, default=TILE_EMBEDDINGS_PT)
    parser.add_argument("--tile-index", type=Path, default=TILE_INDEX_JSON)
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pair-sample-per-query",
        type=int,
        default=1000,
        help="Number of random nodes per query used to approximate its distribution.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Number of node vectors processed per similarity chunk.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "debug/infovqa_similarity_distributions.json",
    )
    return parser.parse_args()


def _load_tensor(path: Path, label: str) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {label} embedding file: {path}. "
            "Replace the placeholder path or pass the corresponding CLI argument."
        )
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{label} embeddings must be a rank-2 torch.Tensor: {path}")
    return F.normalize(value.float(), p=2, dim=1)


def _load_index(path: Path, expected_rows: int, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} index JSON: {path}")
    with path.open(encoding="utf-8") as file:
        index = json.load(file)
    if not isinstance(index, dict):
        raise ValueError(f"{label} index must be a JSON object: {path}")
    if len(index) != expected_rows:
        raise ValueError(
            f"{label} index has {len(index)} entries but embeddings have "
            f"{expected_rows} rows: {path}"
        )
    return index


def _load_questions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list of QA records: {path}")
    questions = []
    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("qid"), str) and isinstance(item.get("question"), str):
            questions.append({"qid": item["qid"], "question": item["question"]})
    if not questions:
        raise ValueError(f"No valid qid/question records found in: {path}")
    return questions


def _query_rows(
    questions: list[dict[str, str]],
    query_embeddings: torch.Tensor,
    query_index: Path | None,
) -> list[tuple[str, int]]:
    if query_index is None:
        if len(questions) != query_embeddings.shape[0]:
            raise ValueError(
                "Without --query-index, query embedding rows must be in the same "
                "order and count as QAs_test.json."
            )
        return [(item["qid"], row) for row, item in enumerate(questions)]

    index = _load_index(query_index, query_embeddings.shape[0], "query")
    qid_to_row = {}
    for row, value in index.items():
        if isinstance(value, str):
            qid_to_row[value] = int(row)
        elif isinstance(value, dict) and isinstance(value.get("qid"), str):
            qid_to_row[value["qid"]] = int(row)
        elif isinstance(value, list) and value and isinstance(value[0], str):
            qid_to_row[value[0]] = int(row)
    missing = [item["qid"] for item in questions if item["qid"] not in qid_to_row]
    if missing:
        raise ValueError(f"Query index is missing {len(missing)} QA ids, e.g. {missing[0]}")
    return [(item["qid"], qid_to_row[item["qid"]]) for item in questions]


def _stats(values: torch.Tensor) -> dict[str, float | int]:
    values = values.float().flatten()
    if values.numel() == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
        "p01": float(torch.quantile(values, 0.01)),
        "p05": float(torch.quantile(values, 0.05)),
        "p25": float(torch.quantile(values, 0.25)),
        "median": float(torch.quantile(values, 0.50)),
        "p75": float(torch.quantile(values, 0.75)),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
    }


def _zscore_diagnostics(per_query_stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Assess whether per-query z-score inputs are numerically stable."""
    stds = torch.tensor([item["std"] for item in per_query_stats])
    skewness = torch.tensor([item["skewness"] for item in per_query_stats])
    degenerate = stds <= 1e-6
    valid_stds = stds[~degenerate]
    mean_std = float(valid_stds.mean()) if valid_stds.numel() else 0.0
    std_of_stds = float(valid_stds.std(unbiased=False)) if valid_stds.numel() else 0.0
    std_cv = std_of_stds / mean_std if mean_std > 1e-8 else float("inf")
    mean_abs_skew = float(skewness.abs().mean()) if skewness.numel() else 0.0
    degenerate_rate = float(degenerate.float().mean()) if stds.numel() else 1.0

    warnings = []
    if degenerate_rate > 0.05:
        warnings.append("more than 5% of queries have near-zero score variance")
    if std_cv > 1.0:
        warnings.append("per-query score scales vary substantially")
    if mean_abs_skew > 2.0:
        warnings.append("per-query score distributions are strongly skewed")

    return {
        "queries": len(per_query_stats),
        "near_zero_std_rate": degenerate_rate,
        "mean_within_query_std": mean_std,
        "std_of_within_query_std": std_of_stds,
        "coefficient_of_variation_of_query_std": std_cv,
        "mean_absolute_skewness": mean_abs_skew,
        "recommendation": (
            "Per-query z-score is numerically reasonable for relative ranking."
            if not warnings
            else "Use caution with per-query z-score; inspect the warnings."
        ),
        "warnings": warnings,
        "note": (
            "Z-score does not require a normal distribution, but it is unstable "
            "when a query has near-zero variance and sensitive to strong outliers."
        ),
    }


def _analyze_type(
    qids: list[str],
    queries: torch.Tensor,
    nodes: torch.Tensor,
    rng: random.Random,
    pair_sample_per_query: int,
    chunk_size: int,
) -> dict[str, Any]:
    pair_scores: list[torch.Tensor] = []
    per_query: list[dict[str, Any]] = []
    node_count = nodes.shape[0]

    for qid, query in zip(qids, queries):
        best = torch.tensor(float("-inf"))
        sampled_indices = rng.sample(
            range(node_count), min(pair_sample_per_query, node_count)
        )
        query_scores = nodes[sampled_indices] @ query
        pair_scores.append(query_scores)
        query_stats = _stats(query_scores)
        query_mean = query_scores.mean()
        query_std = query_scores.std(unbiased=False)
        centered = query_scores - query_mean
        skewness = (
            float((centered.pow(3).mean() / query_std.pow(3)))
            if query_std > 1e-8
            else 0.0
        )
        for start in range(0, node_count, chunk_size):
            scores = nodes[start : start + chunk_size] @ query
            best = torch.maximum(best, scores.max())
        per_query.append({
            "qid": qid,
            "sampled_node_count": len(sampled_indices),
            "sampled_scores": [float(value) for value in query_scores],
            "stats": query_stats,
            "mean": float(query_mean),
            "std": float(query_std),
            "skewness": skewness,
            "max_similarity": float(best),
        })

    pairwise = torch.cat(pair_scores)
    return {
        "pairwise_sample": _stats(pairwise),
        "per_query": per_query,
        "zscore_diagnostics": _zscore_diagnostics(per_query),
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1].")
    if args.pair_sample_per_query < 1 or args.chunk_size < 1:
        raise ValueError("--pair-sample-per-query and --chunk-size must be positive.")

    questions = _load_questions(args.qa_path)
    query_embeddings = _load_tensor(args.query_embeddings, "query")
    query_rows = _query_rows(questions, query_embeddings, args.query_index)

    rng = random.Random(args.seed)
    sample_size = max(1, round(len(query_rows) * args.fraction))
    sampled = rng.sample(query_rows, min(sample_size, len(query_rows)))
    sampled_qids = [qid for qid, _ in sampled]
    sampled_queries = query_embeddings[[row for _, row in sampled]]

    node_specs = {
        "fact": (args.fact_embeddings, args.fact_index),
        "root": (args.root_embeddings, args.root_index),
        "tile": (args.tile_embeddings, args.tile_index),
    }
    distributions = {}
    for label, (embedding_path, index_path) in node_specs.items():
        nodes = _load_tensor(embedding_path, f"{label} node")
        _load_index(index_path, nodes.shape[0], f"{label} node")
        if nodes.shape[1] != sampled_queries.shape[1]:
            raise ValueError(
                f"Dimension mismatch: query={sampled_queries.shape[1]}, "
                f"{label}={nodes.shape[1]}"
            )
        distributions[label] = _analyze_type(
            sampled_qids,
            sampled_queries,
            nodes,
            rng,
            args.pair_sample_per_query,
            args.chunk_size,
        )

    result = {
        "qa_path": str(args.qa_path),
        "query_embeddings_path": str(args.query_embeddings),
        "fraction": args.fraction,
        "seed": args.seed,
        "num_questions": len(questions),
        "num_sampled_queries": len(sampled_qids),
        "sampled_qids": sampled_qids,
        "distributions": distributions,
        "interpretation": (
            "Use per_query entries to inspect each query's cosine-score "
            "distribution. The zscore_diagnostics fields are heuristic checks "
            "for near-zero variance, unstable scale, and strong skewness."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for label, values in distributions.items():
        print(f"{label}:")
        print(f"  pairwise sample: {values['pairwise_sample']}")
        print(f"  z-score check:   {values['zscore_diagnostics']}")
    print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
