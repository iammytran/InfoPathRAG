"""Run paired bootstrap and sign-flip permutation tests for per-query MRR."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["query_id"])] = row
    return rows


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = int(position), int(position + 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def run_tests(
    lilac: dict[str, dict],
    baseline: dict[str, dict],
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict:
    query_ids = sorted(lilac.keys() & baseline.keys())
    if not query_ids:
        raise ValueError("No common query IDs found.")
    differences = [
        float(lilac[qid]["reciprocal_rank_at_10"])
        - float(baseline[qid]["reciprocal_rank_at_10"])
        for qid in query_ids
    ]
    observed = sum(differences) / len(differences)

    bootstrap_rng = random.Random(seed)
    bootstrap_means = []
    for _ in range(bootstrap_samples):
        sample = [
            differences[bootstrap_rng.randrange(len(differences))]
            for _ in differences
        ]
        bootstrap_means.append(sum(sample) / len(sample))

    permutation_rng = random.Random(seed + 1)
    exceedances = 0
    for _ in range(permutation_samples):
        signed_sum = sum(
            difference if permutation_rng.getrandbits(1) else -difference
            for difference in differences
        )
        if abs(signed_sum / len(differences)) >= abs(observed):
            exceedances += 1

    return {
        "num_common_queries": len(query_ids),
        "lilac_mrr_at_10": sum(
            float(lilac[qid]["reciprocal_rank_at_10"]) for qid in query_ids
        ) / len(query_ids),
        "baseline_mrr_at_10": sum(
            float(baseline[qid]["reciprocal_rank_at_10"]) for qid in query_ids
        ) / len(query_ids),
        "observed_mrr_delta_lilac_minus_baseline": observed,
        "paired_bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "confidence_level": 0.95,
            "mean_delta": observed,
            "ci_lower": _percentile(bootstrap_means, 0.025),
            "ci_upper": _percentile(bootstrap_means, 0.975),
        },
        "paired_permutation_sign_flip": {
            "samples": permutation_samples,
            "seed": seed + 1,
            "alternative": "two-sided",
            "exceedances": exceedances,
            "p_value": (exceedances + 1) / (permutation_samples + 1),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lilac", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--permutation-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    result = run_tests(
        _load(args.lilac),
        _load(args.baseline),
        args.bootstrap_samples,
        args.permutation_samples,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
