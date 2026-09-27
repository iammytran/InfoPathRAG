"""Run InfoPathRAG InfoVQA weight/top-k sensitivity experiments.

This is separate from the legacy LILaC visualization script. It evaluates
only the ``root_tile_facts`` path variant, stores one summary JSON per
configuration, and creates plots from those saved summaries.

Weights are ordered as (fact, tile, root).
Top-k tuples are ordered as (root_k, tile_k, final_k).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from src.experiment.experiments.infovqa_ablation import _configure_logging, _run
from src.lilac.retriever.infovqa_retriever import InfoVQARetriever
from src.utils.utils import REPO_ROOT


DEFAULT_WEIGHTS = {
    "fact_only": (1.0, 0.0, 0.0),
    "fact_heavy": (0.7, 0.2, 0.1),
    "fact_balanced": (0.5, 0.25, 0.25),
    "equal": (1 / 3, 1 / 3, 1 / 3),
    "tile_heavy": (0.2, 0.6, 0.2),
    "root_heavy": (0.2, 0.2, 0.6),
}

DEFAULT_TOP_KS = {
    "small": (50, 50, 10),
    "default": (100, 100, 10),
    "current": (100, 100, 70),
    "large": (200, 200, 70),
    "root_heavy": (200, 50, 70),
    "tile_heavy": (50, 200, 70),
}

NORMALIZATIONS = ("raw", "query_zscore")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("weights", "topk", "both"), default="both"
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
            "InfoPathRAG",
            "InfoVQA",
            "hyperparameter_sensitivity",
        ),
    )
    parser.add_argument("--root-k", type=int, default=100)
    parser.add_argument("--tile-k", type=int, default=100)
    parser.add_argument("--final-k", type=int, default=70)
    parser.add_argument("--missing-path-policy", choices=("error", "skip"), default="error")
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only run experiments and write JSON summaries.",
    )
    return parser


def _validate_weights(weights: tuple[float, float, float]) -> None:
    if len(weights) != 3 or any(value < 0 for value in weights):
        raise ValueError(f"Invalid weights: {weights}")
    if abs(sum(weights) - 1.0) > 1e-8:
        raise ValueError(f"Weights must sum to 1: {weights}")


def _make_args(base: argparse.Namespace, output_dir: Path, root_k: int, tile_k: int, final_k: int):
    args = SimpleNamespace(**vars(base))
    args.output_dir = str(output_dir)
    args.root_k = root_k
    args.tile_k = tile_k
    args.final_k = final_k
    args.path_reranking = True
    args.tree_only = False
    args.root_tile_only = True
    args.variants = ("root_tile_facts",)
    args.weights_file = None
    return args


def _make_retriever(root_k: int, tile_k: int, final_k: int) -> InfoVQARetriever:
    return InfoVQARetriever(
        cli_args=[
            "--run_mode", "infovqa_ablation",
            "--target_dataset", "InfoVQA",
            "--run_name", f"infovqa_sensitivity_root{root_k}_tile{tile_k}_final{final_k}",
            "--force_overwrite", "True",
        ]
    )


def _run_one(
    base_args: argparse.Namespace,
    output_dir: Path,
    name: str,
    weights: tuple[float, float, float],
    normalization: str,
    root_k: int,
    tile_k: int,
    final_k: int,
    retriever: InfoVQARetriever,
    base_results: dict | None,
) -> tuple[dict, dict | None]:
    _validate_weights(weights)
    args = _make_args(base_args, output_dir, root_k, tile_k, final_k)
    args.normalization = normalization
    reused = base_results is not None
    summaries, logs, base_results = _run(
        args,
        weights=weights,
        retriever=retriever,
        base_results=base_results,
        variants=("root_tile_facts",),
    )
    summary = dict(summaries[0])
    summary.update({
        "experiment": "InfoPathRAG_InfoVQA_hyperparameter_sensitivity",
        "configuration": name,
        "weights_order": ["fact", "tile", "root"],
        "weights": list(weights),
        "normalization": normalization,
        "top_k_order": ["root_k", "tile_k", "final_k"],
        "top_k": [root_k, tile_k, final_k],
        "candidate_set_reused_for_configuration": reused,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary, base_results


def _plot_results(results: list[dict], output_dir: Path, experiment: str) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("Recall@3", "MRR@10"):
        fig, ax = plt.subplots(figsize=(10, 5))
        for normalization in NORMALIZATIONS:
            rows = [
                row for row in results
                if row["normalization"] == normalization
            ]
            rows.sort(key=lambda row: row["configuration"])
            labels = [row["configuration"] for row in rows]
            values = [row[metric] for row in rows]
            ax.plot(labels, values, marker="o", label=normalization)
        ax.set_title(f"InfoPathRAG InfoVQA {experiment}: {metric}")
        ax.set_ylabel(metric)
        ax.set_xlabel("Configuration")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{experiment.lower()}_{metric.replace('@', '_at_').replace('/', '_')}.png", dpi=200)
        plt.close(fig)


def _write_index(results: list[dict], path: Path) -> None:
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> None:
    _configure_logging()
    base_args = _parser().parse_args()
    root_output = Path(base_args.output_dir)
    root_output.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[dict]] = {}

    if base_args.mode in ("weights", "both"):
        results = []
        retriever = _make_retriever(base_args.root_k, base_args.tile_k, base_args.final_k)
        base_results = None
        for name, weights in DEFAULT_WEIGHTS.items():
            for normalization in NORMALIZATIONS:
                summary, base_results = _run_one(
                    base_args,
                    root_output / "weights" / normalization,
                    name,
                    weights,
                    normalization,
                    base_args.root_k,
                    base_args.tile_k,
                    base_args.final_k,
                    retriever,
                    base_results,
                )
                results.append(summary)
        _write_index(results, root_output / "weights" / "summary.json")
        all_results["weights"] = results
        if not base_args.skip_plots:
            _plot_results(results, root_output / "weights", "weights")

    if base_args.mode in ("topk", "both"):
        results = []
        for name, (root_k, tile_k, final_k) in DEFAULT_TOP_KS.items():
            retriever = _make_retriever(root_k, tile_k, final_k)
            base_results = None
            for normalization in NORMALIZATIONS:
                summary, base_results = _run_one(
                    base_args,
                    root_output / "topk" / normalization,
                    name,
                    (1 / 3, 1 / 3, 1 / 3),
                    normalization,
                    root_k,
                    tile_k,
                    final_k,
                    retriever,
                    base_results,
                )
                results.append(summary)
        _write_index(results, root_output / "topk" / "summary.json")
        all_results["topk"] = results
        if not base_args.skip_plots:
            _plot_results(results, root_output / "topk", "topk")

    (root_output / "summary.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
