"""Compare full-path reranking with alternative path-weight settings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.experiment.experiments.infovqa_ablation import (
    _configure_logging,
    _run,
)
from src.utils.utils import REPO_ROOT


DEFAULT_WEIGHT_CONFIGS = {
    "fact_only": (1.0, 0.0, 0.0),
    "fact_tile": (0.8, 0.2, 0.0),
    "fact_root": (0.8, 0.0, 0.2),
    "full_path": (1 / 3, 1 / 3, 1 / 3),
}


def main():
    _configure_logging()
    parser = argparse.ArgumentParser(
        description="Compare full-path and alternative InfoVQA path weights."
    )
    parser.add_argument(
        "--qa-path",
        default=os.path.join(REPO_ROOT, "datasets", "InfoVQA", "QAs_test.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            REPO_ROOT, "algorithm_results", "LILaC", "InfoVQA", "weight_ablation"
        ),
    )
    parser.add_argument("--root-k", type=int, default=100)
    parser.add_argument("--tile-k", type=int, default=100)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--normalization", choices=("raw", "query_zscore"), default="raw")
    parser.add_argument("--missing-path-policy", choices=("error", "skip"), default="error")
    parser.add_argument(
        "--weights-file",
        help="JSON object mapping configuration names to [fact, tile, root] weights.",
    )
    args = parser.parse_args()
    args.path_reranking = True

    configs = DEFAULT_WEIGHT_CONFIGS
    if args.weights_file:
        configs = {
            name: tuple(values)
            for name, values in json.loads(Path(args.weights_file).read_text()).items()
        }

    summaries = []
    base_results = None
    for name, weights in configs.items():
        current, _, base_results = _run(
            args,
            weights=weights,
            retriever=None,
            base_results=base_results,
            variants=("root_tile_facts",),
        )
        summary = current[0]
        summary["configuration"] = name
        summary["weights"] = list(weights)
        summaries.append(summary)

    output = Path(args.output_dir)
    (output / "weight_ablation_summary.json").write_text(
        json.dumps(summaries, indent=2)
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
