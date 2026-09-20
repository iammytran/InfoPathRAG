"""InfoVQA fact-candidate ablation.

Validation selects the tuple (root-k, tile-k, final-k). Test refuses to run
without the selection file, preventing accidental test-set tuning.
"""

import argparse
import json
import os
import statistics
from pathlib import Path

from src.lilac.retriever.retriever import Retriever
from src.utils.utils import REPO_ROOT, read_json_or_jsonl


VARIANTS = ("flat_facts", "root_facts", "tile_facts", "root_tile_facts")


def _doc(value):
    value = str(value)
    for suffix in (".jpeg", ".jpg", ".png", ".json"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _gold(qas):
    return {
        str(item["qid"]): _doc(item["evidences"][0]["gold_image"])
        for item in qas
        if item.get("evidences")
    }


def _metrics(logs):
    recalls, reciprocal = [], []
    for item in logs:
        docs = item["retrieved_infographics"]
        gold = item["infographic_correct"]
        recalls.append(float(gold in docs[:3]))
        try:
            reciprocal.append(1.0 / (docs[:10].index(gold) + 1))
        except ValueError:
            reciprocal.append(0.0)
    return sum(recalls) / len(recalls), sum(reciprocal) / len(reciprocal)


def _run(args):
    qas = read_json_or_jsonl(args.qa_path)
    gold = _gold(qas)
    retriever = Retriever(
        cli_args=[
            "--run_mode", "tree_traversal",
            "--target_dataset", "InfoVQA",
            "--run_name", f"infovqa_ablation_{args.split}",
            "--force_overwrite", "True",
        ]
    )
    qids = set(gold)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logs_by_variant = {}
    for variant in VARIANTS:
        # Keep raw retrieval results separate; each call appends one JSONL row.
        retriever._run_name = f"infovqa_ablation_{args.split}_{variant}"
        logs = []
        for qid in retriever._questions_manager.get_qid_list():
            if str(qid) not in qids:
                continue
            question = retriever._questions_manager.get_question_instance_by_qid(qid)
            result = retriever.retrieve_infovqa_ablation(
                qid=str(qid), query_vec=question.get_embedding(),
                variant=variant, root_k=args.root_k, tile_k=args.tile_k, k_ret=args.final_k,
            )
            paths = result["retrieved_paths"]
            docs = []
            for path in paths:
                doc = _doc(path["nodes"][0][0])
                if doc not in docs:
                    docs.append(doc)
            correct = gold[str(qid)]
            candidate_roots = {_doc(root[0]) for root in result["candidate_roots"]}
            selected = candidate_roots if variant == "root_facts" else (
                {_doc(tile[0]) for tile in result["candidate_tiles"]}
                if variant == "tile_facts" else candidate_roots | {_doc(tile[0]) for tile in result["candidate_tiles"]}
            )
            rank = docs.index(correct) + 1 if correct in docs else None
            logs.append({
                "qid": str(qid), "infographic_correct": correct,
                "candidate_infographics": sorted(selected) if variant != "flat_facts" else None,
                "correct_infographic_rank": rank,
                "correct_infographic_score": next(
                    (path["score"] for path in paths if _doc(path["nodes"][0][0]) == correct), None
                ),
                "num_candidate_facts": result["num_candidate_facts"],
                "retrieval_time_ms": result["time"]["retrieval_time(ms)"],
                "correct": correct in docs[:args.final_k],
                "error_type": (
                    None if correct in docs[:args.final_k]
                    else ("ranking_miss" if variant == "flat_facts" or correct in selected else "candidate_miss")
                ),
                "candidate_selection_not_applicable": variant == "flat_facts",
                "retrieved_infographics": docs,
            })
        logs_by_variant[variant] = logs

    summary = []
    for variant, logs in logs_by_variant.items():
        recall, mrr = _metrics(logs)
        summary.append({
            "variant": variant, "Recall@3": recall, "MRR@10": mrr,
            "avg_candidate_facts": statistics.mean(x["num_candidate_facts"] for x in logs),
            "avg_retrieval_time_ms": statistics.mean(x["retrieval_time_ms"] for x in logs),
            "num_queries": len(logs),
        })
        with (output / f"{args.split}_{variant}_queries.jsonl").open("w") as handle:
            for row in logs:
                handle.write(json.dumps(row) + "\n")
    with (output / f"{args.split}_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    if args.split == "validation":
        best = max(summary, key=lambda row: (row["MRR@10"], row["Recall@3"]))
        selection = {
            "root_k": args.root_k, "tile_k": args.tile_k, "final_k": args.final_k,
            "selected_variant": best["variant"], "validation_summary": best,
        }
        with (output / "selected_top_k.json").open("w") as handle:
            json.dump(selection, handle, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="InfoVQA candidate-set ablation")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--qa-path", required=True)
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "algorithm_results", "LILaC", "InfoVQA", "ablation"))
    parser.add_argument("--root-k", type=int, default=100)
    parser.add_argument("--tile-k", type=int, default=100)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--selection-file")
    args = parser.parse_args()
    if args.split == "test":
        if not args.selection_file:
            parser.error("--selection-file is required for test (run validation first)")
        with open(args.selection_file) as handle:
            selection = json.load(handle)
        args.root_k, args.tile_k, args.final_k = (
            selection["root_k"], selection["tile_k"], selection["final_k"]
        )
    summary = _run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
