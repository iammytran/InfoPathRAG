"""Compare two per-query InfoVQA JSONL files with an exact McNemar test."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["query_id"])] = row
    return rows


def _p_value(b: int, c: int) -> float:
    discordant = b + c
    if not discordant:
        return 1.0
    tail = min(b, c)
    return min(
        1.0,
        2 * sum(math.comb(discordant, i) for i in range(tail + 1))
        / (2 ** discordant),
    )


def compare(lilac: dict[str, dict], baseline: dict[str, dict]) -> dict:
    common = sorted(lilac.keys() & baseline.keys())
    lilac_hits = [bool(lilac[qid]["hit_at_3"]) for qid in common]
    baseline_hits = [bool(baseline[qid]["hit_at_3"]) for qid in common]
    lilac_correct_baseline_wrong = sum(a and not b for a, b in zip(lilac_hits, baseline_hits))
    lilac_wrong_baseline_correct = sum(not a and b for a, b in zip(lilac_hits, baseline_hits))
    both_correct = sum(a and b for a, b in zip(lilac_hits, baseline_hits))
    both_wrong = sum(not a and not b for a, b in zip(lilac_hits, baseline_hits))
    lilac_mrr = sum(float(lilac[qid]["reciprocal_rank_at_10"]) for qid in common) / len(common)
    baseline_mrr = sum(
        float(baseline[qid]["reciprocal_rank_at_10"]) for qid in common
    ) / len(common)
    return {
        "num_common_queries": len(common),
        "lilac_hit_at_3": sum(lilac_hits) / len(common),
        "baseline_hit_at_3": sum(baseline_hits) / len(common),
        "lilac_mrr_at_10": lilac_mrr,
        "baseline_mrr_at_10": baseline_mrr,
        "lilac_correct_baseline_wrong": lilac_correct_baseline_wrong,
        "lilac_wrong_baseline_correct": lilac_wrong_baseline_correct,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "mcnemar_exact_two_sided_p": _p_value(
            lilac_correct_baseline_wrong, lilac_wrong_baseline_correct
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lilac", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(_load(args.lilac), _load(args.baseline))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
