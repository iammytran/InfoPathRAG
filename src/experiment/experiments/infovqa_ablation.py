"""InfoVQA MEHR candidate ablation and path-aware reranking study.

The retriever creates each candidate set once. Weight search only reranks the
cached MEHR paths, so every weight tuple sees exactly the same candidates.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import time
from pathlib import Path

from src.utils.utils import REPO_ROOT, read_json_or_jsonl


VARIANTS = ("flat_facts", "root_facts", "tile_facts", "root_tile_facts")
DEFAULT_PATH_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
LOGGER = logging.getLogger("infovqa_ablation")


def _configure_logging():
    log_path = Path(REPO_ROOT) / "debug" / "infovqa_ablation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        "%Y-%m-%dT%H:%M:%S%z",
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
    return log_path


def _doc(value):
    value = str(value)
    for suffix in (".jpeg", ".jpg", ".png", ".json"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _gold(qas):
    result = {}
    for item in qas:
        evidences = item.get("evidences") or []
        image = next(
            (evidence.get("gold_image") for evidence in evidences
             if evidence.get("gold_image") is not None),
            item.get("image_local_name"),
        )
        qid = item.get("qid", item.get("questionId"))
        if qid is not None and image is not None:
            result[str(qid)] = _doc(image)
    return result


def _query_text(qas):
    return {
        str(item.get("qid", item.get("questionId"))): item.get("question")
        for item in qas
        if item.get("qid", item.get("questionId")) is not None
    }


def _metrics(logs):
    labeled = [row for row in logs if row.get("infographic_correct") is not None]
    if not labeled:
        return 0.0, 0.0
    recall = sum(row["infographic_correct"] in row["retrieved_infographics"][:3]
                 for row in labeled) / len(labeled)
    rr = []
    for row in labeled:
        docs = row["retrieved_infographics"][:10]
        gold = row["infographic_correct"]
        rr.append(1 / (docs.index(gold) + 1) if gold in docs else 0.0)
    return recall, sum(rr) / len(rr)


def _score_distribution(rows):
    values = [float(value) for value in rows if value is not None]
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(values), "min": min(values), "max": max(values),
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p05": percentile(0.05), "p50": percentile(0.50), "p95": percentile(0.95),
    }


def _stats_from_paths(results):
    return {
        "fact": _score_distribution(
            [path["fact_score"] for result in results for path in result.get("candidate_paths", [])]
        ),
        "tile": _score_distribution(
            [path["tile_score"] for result in results for path in result.get("candidate_paths", [])]
        ),
        "root": _score_distribution(
            [path["root_score"] for result in results for path in result.get("candidate_paths", [])]
        ),
    }


def _docs(result):
    docs = []
    for path in result.get("retrieved_paths", []):
        document = _doc(path["nodes"][0][0])
        if document not in docs:
            docs.append(document)
    return docs


def _log_row(qid, question, gold, result, variant, final_k, elapsed_ms):
    docs = _docs(result)
    roots = {_doc(root[0]) for root in result.get("candidate_roots", [])}
    tiles = {_doc(tile[0]) for tile in result.get("candidate_tiles", [])}
    selected = sorted(roots | tiles)
    correct = gold.get(str(qid))
    hit = correct is not None and correct in docs[:3]
    error = None
    if correct is not None and not hit:
        error = "ranking_miss" if variant == "flat_facts" or correct in selected else "candidate_miss"
    return {
        "qid": str(qid), "query_text": question, "infographic_correct": correct,
        "candidate_infographics": None if variant == "flat_facts" else selected,
        "retrieved_infographics": docs,
        "correct_infographic_rank": docs.index(correct) + 1 if correct in docs else None,
        "num_candidate_facts": result.get("num_candidate_facts"),
        "retrieval_time_ms": result.get("time", {}).get("retrieval_time(ms)", elapsed_ms),
        "reranking_time_ms": elapsed_ms,
        "correct": hit, "error_type": error,
        "path_weights": result.get("path_weights", [1.0, 0.0, 0.0]),
        "normalization": result.get("normalization", "raw"),
        "selected_paths": result.get("selected_paths", []),
    }


def _summary(name, logs, weights=None, normalization="raw", k_values=None):
    recall, mrr = _metrics(logs)
    num_labeled_queries = sum(
        row.get("infographic_correct") is not None for row in logs
    )
    return {
        "variant": name, "Recall@3": recall, "MRR@10": mrr,
        "avg_candidate_facts": statistics.mean(
            row["num_candidate_facts"] for row in logs
        ) if logs else 0.0,
        "avg_retrieval_time_ms": statistics.mean(
            row["retrieval_time_ms"] for row in logs
        ) if logs else 0.0,
        "avg_reranking_time_ms": statistics.mean(
            row["reranking_time_ms"] for row in logs
        ) if logs else 0.0,
        "num_queries": len(logs), "num_labeled_queries": num_labeled_queries,
        "weights": list(weights) if weights else None,
        "k": dict(k_values or {}),
        "normalization": normalization,
    }


def _paired_analysis(fact_logs, full_logs, output):
    fact = {row["qid"]: row for row in fact_logs}
    full = {row["qid"]: row for row in full_logs}
    pairs = [
        (fact[qid]["correct"], full[qid]["correct"])
        for qid in sorted(fact.keys() & full.keys())
    ]
    both = sum(a and b for a, b in pairs)
    neither = sum(not a and not b for a, b in pairs)
    full_only = sum(not a and b for a, b in pairs)
    fact_only = sum(a and not b for a, b in pairs)
    discordant = full_only + fact_only
    mcnemar_p = 1.0 if discordant == 0 else min(
        1.0, 2 * sum(
            math.comb(discordant, i) for i in range(0, min(full_only, fact_only) + 1)
        ) / (2 ** discordant)
    )
    deltas = []
    rescued, harmed = [], []
    for qid in sorted(fact.keys() & full.keys()):
        fact_row, full_row = fact[qid], full[qid]
        def reciprocal(row):
            rank = row.get("correct_infographic_rank")
            return 1 / rank if rank and rank <= 10 else 0.0
        delta = reciprocal(full_row) - reciprocal(fact_row)
        deltas.append(delta)
        if not fact_row["correct"] and full_row["correct"]:
            rescued.append(full_row)
        if fact_row["correct"] and not full_row["correct"]:
            harmed.append(full_row)
    bootstrap = []
    seed = 1729
    for _ in range(2000):
        seed = (1103515245 * seed + 12345) % (2 ** 31)
        sample = [
            deltas[(seed + index * 7919) % len(deltas)]
            for index in range(len(deltas))
        ] if deltas else [0.0]
        bootstrap.append(sum(sample) / len(sample))
    bootstrap.sort()
    analysis = {
        "both_correct": both, "both_wrong": neither,
        "full_path_correct_fact_only_wrong": full_only,
        "fact_only_correct_full_path_wrong": fact_only,
        "mcnemar_exact_two_sided_p": mcnemar_p,
        "mrr_delta_mean": statistics.mean(deltas) if deltas else 0.0,
        "mrr_delta_bootstrap_95ci": [
            bootstrap[int(0.025 * (len(bootstrap) - 1))],
            bootstrap[int(0.975 * (len(bootstrap) - 1))],
        ],
        "rescued_queries": rescued, "harmed_queries": harmed,
    }
    (output / "path_reranking_paired_analysis.json").write_text(
        json.dumps(analysis, indent=2)
    )
    return analysis


def _run(
    args,
    weights=None,
    retriever=None,
    base_results=None,
    variants=VARIANTS,
    weights_by_variant=None,
):
    from src.lilac.retriever.infovqa_retriever import InfoVQARetriever

    qas = read_json_or_jsonl(args.qa_path)
    gold, questions = _gold(qas), _query_text(qas)
    if retriever is None:
        retriever = InfoVQARetriever(cli_args=[
            "--run_mode", "infovqa_ablation", "--target_dataset", "InfoVQA",
            "--run_name", f"infovqa_path_test_root{args.root_k}_tile{args.tile_k}_final{args.final_k}",
            "--force_overwrite", "True",
        ])
    qids = [
        qid
        for qid in retriever._questions_manager.get_qid_list()
        if str(qid) in questions
    ]
    if not qids:
        raise ValueError(
            "No InfoVQA query IDs were found in both the QA file and loaded "
            "question embeddings."
        )
    LOGGER.info("Running retrieval for all matching InfoVQA queries total=%d", len(qids))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    variant_final_k = {
        variant: 50 if variant == "flat_facts" else args.final_k
        for variant in variants
    }
    if base_results is None:
        base_results = {}
        for variant in variants:
            variant_weights = (
                weights_by_variant.get(variant)
                if weights_by_variant is not None
                else weights
            ) or (1.0, 0.0, 0.0)
            LOGGER.info("Starting retrieval variant=%s total=%d", variant, len(qids))
            for index, qid in enumerate(qids, start=1):
                query_started = time.perf_counter()
                question = retriever._questions_manager.get_question_instance_by_qid(qid)
                base_results[(variant, str(qid))] = retriever.retrieve_infovqa_ablation(
                    str(qid), question.get_embedding(), variant, args.root_k,
                    args.tile_k, variant_final_k[variant],
                    path_reranking=args.path_reranking,
                    path_weights=variant_weights,
                    normalization=args.normalization,
                    missing_path_policy=args.missing_path_policy,
                )
                LOGGER.info(
                    "Retrieved variant=%s qid=%s progress=%d/%d elapsed_ms=%.1f",
                    variant, qid, index, len(qids),
                    (time.perf_counter() - query_started) * 1000,
                )
            LOGGER.info("Finished retrieval variant=%s", variant)
    if weights is None:
        weights = (1.0, 0.0, 0.0)
    logs = {}
    all_rows = []
    for variant in variants:
        variant_weights = (
            weights_by_variant.get(variant)
            if weights_by_variant is not None
            else weights
        ) or (1.0, 0.0, 0.0)
        rows = []
        LOGGER.info(
            "Starting reranking variant=%s weights=%s total=%d",
            variant, variant_weights, len(qids),
        )
        for qid in qids:
            base = base_results[(variant, str(qid))]
            started = time.perf_counter()
            result = (
                retriever.rerank_infovqa_paths(
                    base, variant_weights, args.normalization,
                    args.missing_path_policy
                ) if variant == "root_tile_facts" and args.path_reranking else base
            )
            rows.append(_log_row(
                qid, questions[str(qid)], gold, result, variant,
                variant_final_k[variant],
                (time.perf_counter() - started) * 1000,
            ))
            if len(rows) == 1 or len(rows) % 25 == 0 or len(rows) == len(qids):
                LOGGER.info(
                    "Reranked variant=%s progress=%d/%d",
                    variant, len(rows), len(qids),
                )
        logs[variant] = rows
        all_rows.extend(rows)
        with (output / f"path_reranking_per_query_{variant}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        LOGGER.info("Finished reranking variant=%s output_rows=%d", variant, len(rows))
    with (output / "path_reranking_per_query.jsonl").open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")
    summaries = [
        _summary(
            variant,
            rows,
            variant_weights if args.path_reranking else None,
            args.normalization,
            {
                "root_k": args.root_k,
                "tile_k": args.tile_k,
                "final_k": variant_final_k[variant],
            },
        )
        for variant, rows in logs.items()
        for variant_weights in [
            (
                weights_by_variant.get(variant)
                if weights_by_variant is not None
                else weights
            ) or (1.0, 0.0, 0.0)
        ]
    ]
    (output / "path_reranking_test_summary.json").write_text(
        json.dumps(summaries, indent=2)
    )
    (output / "path_score_distributions.json").write_text(
        json.dumps(_stats_from_paths(list(base_results.values())), indent=2)
    )
    return summaries, logs, base_results


def main():
    log_path = _configure_logging()
    parser = argparse.ArgumentParser(description="InfoVQA MEHR path-aware ablation")
    parser.add_argument(
        "--qa-path", default=os.path.join(REPO_ROOT, "datasets", "InfoVQA", "QAs_test.json")
    )
    parser.add_argument("--output-dir", default=os.path.join(
        REPO_ROOT, "algorithm_results", "LILaC", "InfoVQA", "ablation"
    ))
    parser.add_argument("--root-k", type=int, default=100)
    parser.add_argument("--tile-k", type=int, default=100)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--normalization", choices=("raw", "query_zscore"), default="raw")
    parser.add_argument("--missing-path-policy", choices=("error", "skip"), default="error")
    parser.add_argument("--path-reranking", action="store_true")
    parser.add_argument(
        "--root-tile-only",
        action="store_true",
        help="Only retrieve and evaluate root_tile_facts with the supplied weights.",
    )
    parser.add_argument(
        "--weights-file",
        help="Optional JSON file containing the full-path weights; "
             "defaults to equal fact/tile/root weights.",
    )
    args = parser.parse_args()
    LOGGER.info("Starting InfoVQA ablation qa_path=%s log=%s", args.qa_path, log_path)
    output = Path(args.output_dir)
    weights_data = (
        json.loads(Path(args.weights_file).read_text())
        if args.weights_file else {"weights": DEFAULT_PATH_WEIGHTS}
    )
    args.path_reranking = True
    variants = ("root_tile_facts",) if args.root_tile_only else VARIANTS
    args.normalization = weights_data.get("normalization", args.normalization)
    requested = (
        {
            "MEHR + fact-only": (1.0, 0.0, 0.0),
            "MEHR + full-path": tuple(weights_data["weights"]),
        }
        if args.root_tile_only else
        {
            "MEHR + fact-only": (1.0, 0.0, 0.0),
            "MEHR + fact+tile": (0.8, 0.2, 0.0),
            "MEHR + fact+root": (0.8, 0.0, 0.2),
            "MEHR + full-path": tuple(weights_data["weights"]),
        }
    )
    from src.lilac.retriever.infovqa_retriever import InfoVQARetriever
    retriever = InfoVQARetriever(cli_args=[
        "--run_mode", "infovqa_ablation", "--target_dataset", "InfoVQA",
        "--run_name", f"infovqa_path_test_root{args.root_k}_tile{args.tile_k}_final{args.final_k}",
        "--force_overwrite", "True",
    ])
    LOGGER.info("Retriever initialized; beginning cached retrieval")
    base_results = None
    run_data = {}
    for name, weights in requested.items():
        summaries, logs, base_results = _run(
            args, weights, retriever=retriever, base_results=base_results,
            variants=variants,
        )
        root_summary = next(
            item for item in summaries if item["variant"] == "root_tile_facts"
        )
        root_summary["variant"] = name
        root_summary["weights"] = list(weights)
        run_data[name] = (root_summary, logs["root_tile_facts"])
    if args.root_tile_only:
        final_summaries = [
            run_data["MEHR + fact-only"][0],
            run_data["MEHR + full-path"][0],
        ]
    else:
        flat_summary = next(
            item for item in summaries if item["variant"] == "flat_facts"
        )
        final_summaries = [flat_summary] + [
            run_data[name][0] for name in requested
        ]
    (output / "path_reranking_test_summary.json").write_text(
        json.dumps(final_summaries, indent=2)
    )
    analysis = _paired_analysis(
        run_data["MEHR + fact-only"][1],
        run_data["MEHR + full-path"][1],
        output,
    )
    print(json.dumps({"summaries": final_summaries, "paired_analysis": analysis}, indent=2))
    LOGGER.info("Completed InfoVQA ablation output_dir=%s", output)


if __name__ == "__main__":
    main()
