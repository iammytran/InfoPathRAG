"""InfoPathRAG candidate ablation and path-aware reranking study.

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
LOGGER = logging.getLogger("infopathrag_ablation")


def _configure_logging():
    log_path = Path(REPO_ROOT) / "debug" / "infopathrag_ablation.log"
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


def _metrics_at_k(logs, recall_k=1, mrr_k=3):
    labeled = [row for row in logs if row.get("infographic_correct") is not None]
    if not labeled:
        return 0.0, 0.0

    recall = sum(
        row["infographic_correct"] in row["retrieved_infographics"][:recall_k]
        for row in labeled
    ) / len(labeled)
    reciprocal_ranks = []
    for row in labeled:
        docs = row["retrieved_infographics"][:mrr_k]
        gold = row["infographic_correct"]
        reciprocal_ranks.append(
            1 / (docs.index(gold) + 1) if gold in docs else 0.0
        )
    return recall, sum(reciprocal_ranks) / len(reciprocal_ranks)


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


def _log_row(qid, question, gold, result, variant, elapsed_ms):
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


def _summary(
    name, logs, weights=None, normalization="raw", k_values=None,
    ranking_mode="score_fusion",
):
    recall, mrr = _metrics(logs)
    recall_at_1, mrr_at_3 = _metrics_at_k(logs)
    num_labeled_queries = sum(
        row.get("infographic_correct") is not None for row in logs
    )
    return {
        "variant": name, "Recall@3": recall, "MRR@10": mrr,
        "Recall@1": recall_at_1, "MRR@3": mrr_at_3,
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
        "ranking_mode": ranking_mode,
    }


def _run(
    args,
    weights=None,
    retriever=None,
    base_results=None,
    variants=VARIANTS,
    weights_by_variant=None,
):
    from src.lilac.retriever.my_retriever import MyRetriever

    qas = read_json_or_jsonl(args.qa_path)
    gold, questions = _gold(qas), _query_text(qas)
    if retriever is None:
        retriever = MyRetriever(cli_args=[
            "--run_mode", "infopathrag", "--target_dataset", "InfoVQA",
            "--run_name", f"infopathrag_path_test_root{args.root_k}_tile{args.tile_k}_final{args.final_k}",
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
        variant: args.final_k
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
                candidate_strategy = {
                    "flat_facts": "flat",
                    "root_facts": "root",
                    "tile_facts": "tile",
                    "root_tile_facts": "root_tile",
                }[variant]
                base_results[(variant, str(qid))] = retriever.retrieve_infopathrag(
                    qid=str(qid), query_vec=question.get_embedding(),
                    candidate_strategy=candidate_strategy,
                    ranking_strategy=(
                        getattr(args, "ranking_strategy", "full_path")
                        if args.path_reranking and not getattr(args, "tree_only", False)
                        else "fact_only"
                    ),
                    root_k=args.root_k, tile_k=args.tile_k,
                    top_fact_k=variant_final_k[variant],
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
                retriever.rerank_infovqa_paths_tree_only(
                    base, variant_final_k[variant]
                ) if getattr(args, "tree_only", False) else
                retriever.rerank_infovqa_paths(
                    base, variant_weights, args.normalization,
                    args.missing_path_policy
                ) if variant == "root_tile_facts" and args.path_reranking else base
            )
            rows.append(_log_row(
                qid, questions[str(qid)], gold, result, variant,
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
            variant_weights if args.path_reranking and not getattr(args, "tree_only", False) else None,
            args.normalization if not getattr(args, "tree_only", False) else None,
            {
                "root_k": args.root_k,
                "tile_k": args.tile_k,
                "final_k": variant_final_k[variant],
            },
            "tree_only" if getattr(args, "tree_only", False) else "score_fusion",
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
    parser = argparse.ArgumentParser(description="InfoPathRAG ablation experiments")
    parser.add_argument(
        "--ablation-type",
        choices=("candidate", "reranking"),
        default="candidate",
        help=(
            "candidate compares candidate strategies with fact-only ranking; "
            "reranking fixes root_tile candidates and compares ranking strategies."
        ),
    )
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
    parser.add_argument(
        "--weights-file",
        help="Optional JSON file containing the full-path weights; "
             "defaults to equal fact/tile/root weights.",
    )
    args = parser.parse_args()
    LOGGER.info(
        "Starting InfoPathRAG %s ablation qa_path=%s log=%s",
        args.ablation_type, args.qa_path, log_path,
    )
    output = Path(args.output_dir)
    weights_data = (
        json.loads(Path(args.weights_file).read_text())
        if args.weights_file else {}
    )
    args.normalization = weights_data.get("normalization", args.normalization)
    from src.lilac.retriever.my_retriever import MyRetriever
    retriever = MyRetriever(cli_args=[
        "--run_mode", "infopathrag", "--target_dataset", "InfoVQA",
        "--run_name", f"infopathrag_path_test_root{args.root_k}_tile{args.tile_k}_final{args.final_k}",
        "--force_overwrite", "True",
    ])
    LOGGER.info("Retriever initialized; beginning cached retrieval")
    if args.ablation_type == "candidate":
        args.path_reranking = False
        args.ranking_strategy = "fact_only"
        summaries, _, _ = _run(
            args,
            weights=(1.0, 0.0, 0.0),
            retriever=retriever,
            variants=VARIANTS,
        )
        (output / "path_reranking_test_summary.json").write_text(
            json.dumps(summaries, indent=2)
        )
        print(json.dumps({"ablation_type": "candidate", "summaries": summaries}, indent=2))
        return

    args.path_reranking = True
    ranking_configs = {
        "fact_only": ("fact_only", (1.0, 0.0, 0.0)),
        "fact_tile": ("fact_tile", (0.8, 0.2, 0.0)),
        "fact_root": ("fact_root", (0.8, 0.0, 0.2)),
        "full_path": (
            "full_path",
            tuple(weights_data.get("weights", DEFAULT_PATH_WEIGHTS)),
        ),
    }
    base_results = None
    run_data = {}
    for name, (_, weights) in ranking_configs.items():
        args.ranking_strategy = ranking_configs[name][0]
        summaries, logs, base_results = _run(
            args,
            weights,
            retriever=retriever,
            base_results=base_results,
            variants=("root_tile_facts",),
        )
        summary = summaries[0]
        summary["variant"] = name
        summary["ranking_strategy"] = name
        summary["weights"] = list(weights)
        run_data[name] = (summary, logs["root_tile_facts"])
    (output / "path_reranking_test_summary.json").write_text(
        json.dumps([summary for summary, _ in run_data.values()], indent=2)
    )
    print(json.dumps({
        "ablation_type": "reranking",
        "summaries": [summary for summary, _ in run_data.values()],
    }, indent=2))
    LOGGER.info("Completed InfoPathRAG ablation output_dir=%s", output)


if __name__ == "__main__":
    main()
