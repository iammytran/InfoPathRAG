"""Run InfoVQA hyperparameter sensitivity using raw path scores only.

This keeps the same weight and top-k configurations as
``infopathrag_hyperparameter_sensitivity`` but skips query z-score reranking.
"""

from __future__ import annotations

from src.experiment.experiments.infopathrag_hyperparameter_sensitivity import (
    main as run_sensitivity,
)
from src.utils.utils import REPO_ROOT


def main() -> None:
    run_sensitivity(
        normalizations=("raw",),
        default_output_dir=(
            f"{REPO_ROOT}/algorithm_results/InfoPathRAG/InfoVQA/"
            "hyperparameter_sensitivity_raw"
        ),
    )


if __name__ == "__main__":
    main()
