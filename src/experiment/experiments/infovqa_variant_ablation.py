"""Run InfoVQA ablations for the four candidate-set variants.

The effective path weights are:
flat_facts=(1, 0, 0), root_facts=(0.5, 0, 0.5),
tile_facts=(0.5, 0.5, 0), and root_tile_facts=(1/3, 1/3, 1/3).
Weights are ordered as (fact, tile, root).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.experiment.experiments.infovqa_ablation import (
    VARIANTS,
    _configure_logging,
    _run,
)
from src.utils.utils import REPO_ROOT


DEFAULT_VARIANT_WEIGHTS = {
    "flat_facts": (1.0, 0.0, 0.0),
    "root_facts": (0.5, 0.0, 0.5),
    "tile_facts": (0.5, 0.5, 0.0),
    "root_tile_facts": (1 / 3, 1 / 3, 1 / 3),
}


def main():
    _configure_logging()
    parser = argparse.ArgumentParser(
        description="Compare InfoVQA flat, root, tile, and root-tile fact variants."
    )
    parser.add_argument(
        "--qa-path",
        default=os.path.join(REPO_ROOT, "datasets", "InfoVQA", "QAs_test.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            REPO_ROOT, "algorithm_results", "LILaC", "InfoVQA", "variant_ablation"
        ),
    )
    parser.add_argument(
        "--variant",
        dest="variants",
        choices=VARIANTS,
        nargs="+",
        default=list(VARIANTS),
        help="Variant(s) to evaluate. Defaults to all four variants.",
    )
    parser.add_argument(
        "--root-tile-only",
        action="store_true",
        help="Evaluate only root_tile_facts (kept for compatibility with the "
             "original ablation command).",
    )
    parser.add_argument("--root-k", type=int, default=100)
    parser.add_argument("--tile-k", type=int, default=100)
    parser.add_argument("--final-k", type=int, default=70)
    parser.add_argument("--normalization", choices=("raw", "query_zscore"), default="query_zscore")
    parser.add_argument("--missing-path-policy", choices=("error", "skip"), default="error")
    parser.add_argument(
        "--path-reranking",
        action="store_true",
        help="Enable path reranking. It is enabled for this study by default.",
    )
    parser.add_argument(
        "--tree-only",
        action="store_true",
        help="Use candidate-tree filtering followed by fact top-k selection "
             "without root/tile/fact score fusion.",
    )
    parser.add_argument(
        "--weights-file",
        help="JSON file containing a [fact, tile, root] 'weights' array.",
    )
    args = parser.parse_args()
    args.path_reranking = True

    variants = ("root_tile_facts",) if args.root_tile_only else tuple(args.variants)
    weights_by_variant = dict(DEFAULT_VARIANT_WEIGHTS)
    if args.weights_file:
        weights_data = json.loads(Path(args.weights_file).read_text())
        try:
            configured_weights = tuple(weights_data["weights"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "weights-file must contain a JSON object with a 'weights' array"
            ) from exc
        if len(configured_weights) != 3:
            raise ValueError("weights-file 'weights' must contain exactly three values")
        weights_by_variant["root_tile_facts"] = configured_weights

    summaries, _, _ = _run(
        args,
        retriever=None,
        variants=variants,
        weights_by_variant=weights_by_variant,
    )
    output = Path(args.output_dir)
    (output / "variant_ablation_summary.json").write_text(
        json.dumps(summaries, indent=2)
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
