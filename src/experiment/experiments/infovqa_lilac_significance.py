"""Test significance between LILaC and full-path InfoVQA retrieval.

Recall@3 uses McNemar's exact test. MRR@10 uses a paired Wilcoxon signed-rank
test on per-query reciprocal ranks. Results are paired by qid.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

from src.utils.utils import REPO_ROOT


DEFAULT_FULL_PATH = os.path.join(
    REPO_ROOT,
    "algorithm_results",
    "LILaC",
    "InfoVQA",
    "variant_ablation",
    "path_reranking_per_query_root_tile_facts.jsonl",
)
DEFAULT_LILAC = os.path.join(
    REPO_ROOT,
    "algorithm_results_lilac",
    "LILaC",
    "InfoVQA",
    "retrieval",
    "mmembed_22081405",
    "mmembed_22081405.jsonl",
)


def _normalise_doc(value):
    name = Path(str(value)).name
    for suffix in (".jpeg", ".jpg", ".png", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _gold_doc(qid):
    return _normalise_doc(str(qid).split("-", 1)[0])


def _read_records(path):
    text = Path(path).read_text()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        return records
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError(f"Expected a JSON object, array, or JSONL file: {path}")


def _retrieved_docs(record):
    if "retrieved_infographics" in record:
        return [_normalise_doc(value) for value in record["retrieved_infographics"]]

    units = record.get("retrieved_units", [])
    docs = []
    for unit in units:
        nodes = unit.get("nodes", []) if isinstance(unit, dict) else []
        if not nodes:
            continue
        node = nodes[0]
        if isinstance(node, (list, tuple)) and node:
            docs.append(_normalise_doc(node[0]))
    return docs


def _results_by_qid(path):
    results = {}
    for record in _read_records(path):
        qid = record.get("qid", record.get("questionId"))
        if qid is None:
            continue
        qid = str(qid)
        docs = _retrieved_docs(record)
        gold = _gold_doc(qid)
        rank = docs.index(gold) + 1 if gold in docs else None
        results[qid] = {
            "recall@3": rank is not None and rank <= 3,
            "mrr@10": 1 / rank if rank is not None and rank <= 10 else 0.0,
            "mrr@10_hit": rank is not None and rank <= 10,
        }
    if not results:
        raise ValueError(f"No records with qid were found in {path}")
    return results


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _compare_metric(full_path, lilac, metric):
    qids = sorted(full_path.keys() & lilac.keys())
    full_values = np.asarray([full_path[qid][metric] for qid in qids], dtype=float)
    lilac_values = np.asarray([lilac[qid][metric] for qid in qids], dtype=float)
    if metric == "recall@3":
        full_outcomes, lilac_outcomes = full_values.astype(bool), lilac_values.astype(bool)
    else:
        full_outcomes, lilac_outcomes = None, None
    table = None
    if metric == "recall@3":
        table = [
            [
                int(sum(not lilac_hit and not full_hit
                    for lilac_hit, full_hit in zip(lilac_outcomes, full_outcomes))),
                int(sum(not lilac_hit and full_hit
                    for lilac_hit, full_hit in zip(lilac_outcomes, full_outcomes))),
            ],
            [
                int(sum(lilac_hit and not full_hit
                    for lilac_hit, full_hit in zip(lilac_outcomes, full_outcomes))),
                int(sum(lilac_hit and full_hit
                    for lilac_hit, full_hit in zip(lilac_outcomes, full_outcomes))),
            ],
        ]
    result = {
        "metric": metric,
        "full_path_mean": float(np.mean(full_values)),
        "lilac_mean": float(np.mean(lilac_values)),
        "mean_delta_full_path_minus_lilac": (
            float(np.mean(full_values) - np.mean(lilac_values))
        ),
    }
    if metric == "recall@3":
        test = mcnemar(table, exact=True, correction=False)
        result.update({
            "test": "mcnemar_exact",
            "contingency_table": table,
            "statistic": float(test.statistic),
            "pvalue": float(test.pvalue),
        })
    else:
        deltas = full_values - lilac_values
        test = wilcoxon(deltas, alternative="two-sided", method="auto") if np.any(deltas) else None
        result.update({
            "test": "wilcoxon_signed_rank_paired",
            "statistic": None if test is None else float(test.statistic),
            "pvalue": 1.0 if test is None else float(test.pvalue),
            "num_nonzero_deltas": int(np.count_nonzero(deltas)),
        })
    return result


def compare(full_path_file, lilac_file, metric="recall@3"):
    full_path = _results_by_qid(full_path_file)
    lilac = _results_by_qid(lilac_file)
    common_qids = sorted(full_path.keys() & lilac.keys())
    if not common_qids:
        raise ValueError("The two result files have no common qids.")
    if metric == "both":
        metrics = {
            name: _compare_metric(full_path, lilac, name)
            for name in ("recall@3", "mrr@10")
        }
    else:
        if metric not in ("recall@3", "mrr@10"):
            raise ValueError(f"Unsupported metric: {metric}")
        metrics = _compare_metric(full_path, lilac, metric)
    result = {
        "full_path_weights": [1 / 3, 1 / 3, 1 / 3],
        "num_full_path_queries": len(full_path),
        "num_lilac_queries": len(lilac),
        "num_paired_queries": len(common_qids),
        "results": metrics,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run an exact McNemar test for LILaC versus full-path retrieval."
    )
    parser.add_argument("--full-path-file", default=DEFAULT_FULL_PATH)
    parser.add_argument("--lilac-file", default=DEFAULT_LILAC)
    parser.add_argument(
        "--metric",
        choices=("recall@3", "mrr@10", "both"),
        default="recall@3",
        help="Metric to compare. MRR uses paired Wilcoxon; Recall@3 uses "
             "exact McNemar.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            REPO_ROOT,
            "algorithm_results",
            "LILaC",
            "InfoVQA",
            "variant_ablation",
            "lilac_full_path_significance.json",
        ),
    )
    args = parser.parse_args()
    result = compare(args.full_path_file, args.lilac_file, args.metric)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
