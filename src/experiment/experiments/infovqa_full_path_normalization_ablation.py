"""Compare raw-score and per-query z-score full-path reranking on InfoVQA.

Both configurations use the same ``root_tile_facts`` candidate set and equal
path weights in (fact, tile, root) order:

    (1/3, 1/3, 1/3)

The first run builds and caches the candidate paths. The second run reuses
those cached candidates, so the comparison changes only score normalization.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.experiment.experiments.infovqa_ablation import (
    _configure_logging,
    _run,
)
from src.lilac.retriever.infovqa_retriever import InfoVQARetriever
from src.utils.utils import REPO_ROOT


FULL_PATH_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
NORMALIZATIONS = ("raw", "query_zscore")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw and per-query z-score reranking for "
            "InfoVQA root_tile_facts paths."
        )
    )
    parser.add_argument(
        "--qa-path",
        default=os.path.join(REPO_ROOT, "datasets", "InfoVQA", "QAs_test.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            REPO_ROOT,
            "algorithm_results",
            "LILaC",
            "InfoVQA",
            "full_path_normalization_ablation",
        ),
    )
    parser.add_argument("--root-k", type=int, default=100)
    parser.add_argument("--tile-k", type=int, default=100)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument(
        "--missing-path-policy",
        choices=("error", "skip"),
        default="error",
    )
    return parser


def _retriever(args: argparse.Namespace) -> InfoVQARetriever:
    return InfoVQARetriever(
        cli_args=[
            "--run_mode",
            "infovqa_ablation",
            "--target_dataset",
            "InfoVQA",
            "--run_name",
            f"infovqa_full_path_root{args.root_k}_tile{args.tile_k}_final{args.final_k}",
            "--force_overwrite",
            "True",
        ]
    )


def main() -> None:
    _configure_logging()
    args = _build_parser().parse_args()
    args.path_reranking = True
    args.root_tile_only = True
    args.tree_only = False
    args.weights_file = None
    args.variant = ["root_tile_facts"]
    args.variants = ("root_tile_facts",)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    retriever = _retriever(args)

    summaries = []
    base_results = None
    per_configuration = {}
    for normalization in NORMALIZATIONS:
        args.normalization = normalization
        args.output_dir = str(output / normalization)
        current_summaries, logs, base_results = _run(
            args,
            weights=FULL_PATH_WEIGHTS,
            retriever=retriever,
            base_results=base_results,
            variants=("root_tile_facts",),
        )
        summary = dict(current_summaries[0])
        summary["configuration"] = normalization
        summary["weights"] = list(FULL_PATH_WEIGHTS)
        summaries.append(summary)
        per_configuration[normalization] = {
            "summary": summary,
            "num_queries": len(logs["root_tile_facts"]),
            "per_query_file": str(
                Path(args.output_dir) / "path_reranking_per_query_root_tile_facts.jsonl"
            ),
        }

    comparison = {
        "variant": "root_tile_facts",
        "weights_order": ["fact", "tile", "root"],
        "weights": list(FULL_PATH_WEIGHTS),
        "candidate_set_reused": True,
        "normalizations": list(NORMALIZATIONS),
        "configurations": per_configuration,
        "summaries": summaries,
    }
    summary_path = output / "full_path_normalization_ablation_summary.json"
    summary_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
