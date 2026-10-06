import os
import time

import torch

from src.lilac.retriever.indexer import Indexer
from src.utils.utils import append_to_jsonl_file, artifact_subpath
from .retriever import RUN_CONFIG_PATH, Retriever
class MyRetriever(Retriever):
    """Retriever for hierarchical infographic root/tile/fact paths."""

    def _output_algorithm_name(self) -> str:
        return "InfoPathRAG"

    def __init__(self, config_path: str = RUN_CONFIG_PATH, cli_args=None):
        self._infovqa_fact_to_parent = None
        self._infovqa_indexed_fact_targets = None
        super().__init__(config_path=config_path, cli_args=cli_args)
        self._init_infovqa_indexers()

    def _init_infovqa_indexers(self):
        """Prepare root, tile, and fact indexes for infographic retrieval."""
        emb_dir = artifact_subpath(
            self._metadata_config,
            self._target_dataset,
            "embeddings_dirname",
            self._target_embedder,
        )
        gpu_num = -1

        root_paths = (
            os.path.join(emb_dir, "image.pt"),
            os.path.join(emb_dir, "image.json"),
        )
        tile_paths = (
            os.path.join(emb_dir, "subimage.pt"),
            os.path.join(emb_dir, "subimage.json"),
        )
        fact_paths = [
            os.path.join(emb_dir, "tile_fact.pt"),
            os.path.join(emb_dir, "tile_fact.json"),
        ]

        missing_paths = [
            path
            for paths in (root_paths, tile_paths, fact_paths)
            for path in paths
            if not os.path.exists(path)
        ]
        if missing_paths:
            raise FileNotFoundError(
                "InfoPathRAG embeddings are required: "
                + ", ".join(missing_paths)
            )

        root_index = Indexer()
        root_index.load_embeddings([root_paths], show_progress=True)
        root_index.create_index(gpu_id=gpu_num)

        tile_index = Indexer()
        tile_index.load_embeddings([tile_paths], show_progress=True)
        tile_index.create_index(gpu_id=gpu_num)

        fact_index = Indexer()
        fact_index.load_embeddings([tuple(fact_paths)], show_progress=True)
        fact_index.create_index(gpu_id=gpu_num)
        self.info_level_to_indexer = {
            "root": root_index,
            "tile": tile_index,
            "fact": fact_index,
        }

    def _retrieve_for_run(self, qid, question_embedding, subquery_embeddings):
        if self._run_function_mode == "infopathrag":
            return self.retrieve_infopathrag(
                qid=qid,
                query_vec=question_embedding,
                candidate_strategy="root_tile",
                ranking_strategy="full_path",
                path_weights=(1 / 3, 1 / 3, 1 / 3),
            )
        return super()._retrieve_for_run(qid, question_embedding, subquery_embeddings)

    def _get_infopathrag_maps(self, fact_index):
        """Build graph/index lookup maps once for infographic path retrieval."""
        if (
            self._infovqa_fact_to_parent is not None
            and self._infovqa_indexed_fact_targets is not None
        ):
            return (
                self._infovqa_fact_to_parent,
                self._infovqa_indexed_fact_targets,
            )

        # mapping all the paths from fact's ancestors (root, tile) to that fact
        fact_to_parent = {}
        for filename, edges in self.graph.intra_document_edges.items():
            for parent_id, children in edges.items():
                parent = (filename, str(parent_id))
                for child in children:
                    child_target = self._target_parts(child.get_gcid())
                    if not self._is_tile_target(child_target):
                        continue
                    for fact in self.graph.intra_document_edges.get(
                        filename, {}
                    ).get(child_target[1], []):
                        fact_target = self._target_parts(fact.get_gcid())
                        if self._is_fact_target(fact_target):
                            fact_to_parent.setdefault(fact_target, []).append(
                                (parent, child_target)
                            )

        # mapping the new fact index to the old index that used in tile_fact.json
        indexed_fact_targets = {}
        for indexed_target in (fact_index._idx2target or {}).values():
            normalized = self._target_parts(indexed_target)
            if self._is_fact_target(normalized):
                indexed_fact_targets[normalized] = tuple(indexed_target)

        self._infovqa_fact_to_parent = fact_to_parent
        self._infovqa_indexed_fact_targets = indexed_fact_targets
        print(
            "[Retriever] Prepared InfoPathRAG maps: "
            f"{len(fact_to_parent):,} facts, {len(indexed_fact_targets):,} indexed facts",
            flush=True,
        )

        # print(f"fact_to_parent: {fact_to_parent}")
        # print(f"indexed_fact_targets: {indexed_fact_targets}")
        # print(f"indexed_fact_targets: {indexed_fact_targets}")
        return fact_to_parent, indexed_fact_targets
        
        
    
    
    
    
    
    
        
    @staticmethod
    def _target_parts(target):
        if not isinstance(target, (list, tuple)) or len(target) not in (2, 3):
            raise ValueError(f"Unexpected target format: {target!r}")
        if len(target) == 2:
            return str(target[0]), str(target[1])
        return str(target[0]), str(target[2])

    @staticmethod
    def _is_tile_target(target):
        try:
            _, component_id = MyRetriever._target_parts(target)
        except ValueError:
            return False
        return (
            component_id.startswith("i_1_t")
            and "_f" not in component_id
        )


    @staticmethod
    def _is_fact_target(target):
        try:
            _, component_id = MyRetriever._target_parts(target)
        except ValueError:
            return False
        return (
            "_f" in component_id
        )

    @staticmethod
    def _is_infographic_target(target):
        return (
            isinstance(target, (list, tuple))
            and len(target) == 2
            and str(target[1]) == "i_1"
        )

    def retrieve_infopathrag(
        self,
        qid,
        query_vec,
        candidate_strategy="root_tile",
        ranking_strategy="fact_only",
        root_k=None,
        tile_k=None,
        top_entry_k=70,
        top_fact_k=10,
        path_weights=(1.0, 0.0, 0.0),
        normalization="raw",
        missing_path_policy="error",
    ):
        """Retrieve infographic paths using independent candidate and ranking strategies."""
        candidate_modes = {
            "flat": ("flat", None, None),
            "root": ("root", "root", None),
            "tile": ("tile", None, "tile"),
            "root_tile": ("root_tile", "root_tile", "root_tile"),
        }
        try:
            mode, selected_root_mode, selected_tile_mode = candidate_modes[
                candidate_strategy
            ]
        except KeyError as exc:
            raise ValueError(
                f"Unknown candidate strategy: {candidate_strategy}"
            ) from exc

        ranking_weights = {
            "fact_only": (1.0, 0.0, 0.0),
            "fact_tile": path_weights,
            "fact_root": path_weights,
            "full_path": path_weights,
        }
        if ranking_strategy not in ranking_weights:
            raise ValueError(f"Unknown ranking strategy: {ranking_strategy}")
        rerank = ranking_strategy != "fact_only"
        weights = ranking_weights[ranking_strategy]
        if ranking_strategy == "fact_tile" and weights[2] != 0:
            raise ValueError("fact_tile ranking requires root weight to be zero.")
        if ranking_strategy == "fact_root" and weights[1] != 0:
            raise ValueError("fact_root ranking requires tile weight to be zero.")

        root_k, tile_k, top_entry_k, top_fact_k = self._infovqa_k_values(
            root_k, tile_k, top_entry_k, top_fact_k
        )
        if mode == "flat":
            fact_index = self.info_level_to_indexer["fact"]
            facts = {
                self._target_parts(target)
                for target in (fact_index._idx2target or {}).values()
                if self._is_fact_target(target)
            }
            return self._retrieve_infovqa_candidate_facts(
                qid, query_vec, facts, [], [], candidate_strategy, top_fact_k,
                {}, {}, rerank, weights, normalization, missing_path_policy,
            )

        roots, tiles, ranked_nodes, root_scores, tile_scores = self._infovqa_selection(
            query_vec, root_k, tile_k, top_entry_k
        )
        fact_to_parent, _ = self._get_infopathrag_maps(
            self.info_level_to_indexer["fact"]
        )
        selected_entries = {target for target, _ in ranked_nodes}
        selected_roots = {
            target for target in selected_entries
            if selected_root_mode and target in dict(roots)
        }
        selected_tiles = {
            target for target in selected_entries
            if selected_tile_mode and target in dict(tiles)
        }
        candidate_facts = {
            fact for fact, paths in fact_to_parent.items()
            if any(
                (root in selected_roots if candidate_strategy == "root" else
                 tile in selected_tiles if candidate_strategy == "tile" else
                 root in selected_roots or tile in selected_tiles)
                for root, tile in paths
            )
        }
        return self._retrieve_infovqa_candidate_facts(
            qid, query_vec, candidate_facts,
            list(selected_roots), list(selected_tiles), candidate_strategy, top_fact_k,
            root_scores, tile_scores, rerank, weights, normalization,
            missing_path_policy,
        )

    def _retrieve_infovqa_candidate_facts(
        self, qid, query_vec, candidate_facts, selected_roots, selected_tiles,
        candidate_strategy, k_ret, root_scores=None, tile_scores=None,
        path_reranking=False, path_weights=(1.0, 0.0, 0.0),
        normalization="raw", missing_path_policy="error",
    ):
        """Score, rank, and serialize a common infographic fact candidate set."""
        started = time.perf_counter()
        fact_index = self.info_level_to_indexer.get("fact")
        if fact_index is None or fact_index.get_embeddings() is None:
            raise RuntimeError("InfoPathRAG requires tile_fact embeddings.")
        fact_to_parent, indexed_fact_targets = self._get_infopathrag_maps(
            fact_index
        )
        facts = sorted(fact for fact in candidate_facts if fact in indexed_fact_targets)
        if facts:
            embeddings = fact_index.get_embeddings().float()
            rows = [
                fact_index.get_vector_idx_for_target(indexed_fact_targets[fact])
                for fact in facts
            ]
            query = query_vec.to(embeddings.device, dtype=embeddings.dtype)
            scores = torch.nn.functional.cosine_similarity(
                embeddings[rows], query.unsqueeze(0), dim=1
            ).tolist()
        else:
            scores = []
        scored = []
        path_options = {}
        missing_path_count = 0
        root_scores = root_scores or {}
        tile_scores = tile_scores or {}
        required_roots = set(root_scores)
        required_tiles = set(tile_scores)
        for fact in facts:
            for root, tile in fact_to_parent.get(fact, []):
                required_roots.add(root)
                required_tiles.add(tile)
        root_scores.update(
            self._score_index_targets(
                self.info_level_to_indexer["root"], required_roots, query_vec
            )
        )
        tile_scores.update(
            self._score_index_targets(
                self.info_level_to_indexer["tile"], required_tiles, query_vec
            )
        )
        alpha, beta, gamma = self._validate_path_weights(path_weights)
        if path_reranking and normalization != "raw":
            fact_values = [score for score in scores]
            root_scores = self._normalize_score_map(root_scores, normalization)
            tile_scores = self._normalize_score_map(tile_scores, normalization)
            fact_values = self._normalize_values(fact_values, normalization)
        else:
            fact_values = scores
        for fact, score in zip(facts, scores):
            fact_score = fact_values[facts.index(fact)]
            paths = fact_to_parent.get(fact, [])
            if not paths:
                missing_path_count += 1
                if path_reranking and missing_path_policy == "error":
                    raise RuntimeError(
                        f"Missing graph path metadata for fact {fact!r}"
                    )
                paths = [((fact[0], "i_1"), fact)]
            path_scores = []
            for root, tile in paths:
                if path_reranking and (
                    root not in root_scores or tile not in tile_scores
                ):
                    missing_path_count += 1
                    if missing_path_policy == "error":
                        raise RuntimeError(
                            f"Missing root/tile score for fact path "
                            f"{fact!r}: root={root!r}, tile={tile!r}"
                        )
                    continue
                tile_score = tile_scores.get(tile)
                root_score = root_scores.get(root)
                final_score = (
                    alpha * fact_score + beta * tile_score + gamma * root_score
                    if path_reranking else score
                )
                option = {
                    "fact_id": list(fact),
                    "tile_id": list(tile),
                    "root_id": list(root),
                    "fact_score": score,
                    "tile_score": tile_score,
                    "root_score": root_score,
                }
                path_options.setdefault(fact, []).append(option)
                path_scores.append((final_score, root, tile, tile_score, root_score))
            if not path_scores:
                continue
            final_score, root, tile, tile_score, root_score = max(
                path_scores, key=lambda item: (item[0], item[1], item[2])
            )
            scored.append({
                "fact": fact, "root": root, "tile": tile,
                "fact_score": score, "tile_score": tile_score,
                "root_score": root_score, "score": final_score,
            })
        scored.sort(key=lambda item: item["score"], reverse=True)
        paths = [{
            "nodes": [list(item["root"]), list(item["tile"]), list(item["fact"])],
            "edges": [[item["root"][1], item["tile"][1]],
                      [item["tile"][1], item["fact"][1]]],
            "score": item["score"], "specific_scores": {}, "type": "path",
        } for item in scored[:k_ret]]
        result = {
            "qid": qid, "retrieved_paths": paths,
            "candidate_strategy": candidate_strategy,
            "candidate_facts": [list(fact) for fact in sorted(candidate_facts)],
            "candidate_roots": [list(root) for root in selected_roots],
            "candidate_tiles": [list(tile) for tile in selected_tiles],
            "num_candidate_facts": len(candidate_facts),
            "path_reranking": path_reranking,
            "path_weights": [alpha, beta, gamma],
            "normalization": normalization,
            "selected_paths": [
                {
                    "fact_id": list(item["fact"]),
                    "tile_id": list(item["tile"]),
                    "root_id": list(item["root"]),
                    "fact_score": item["fact_score"],
                    "tile_score": item["tile_score"],
                    "root_score": item["root_score"],
                    "final_score": item["score"],
                }
                for item in scored[:k_ret]
            ],
            "candidate_paths": [
                option
                for item in scored
                for option in path_options[item["fact"]]
            ],
            "missing_path_count": missing_path_count,
            "time": {"retrieval_time(ms)": (time.perf_counter() - started) * 1000},
        }
        append_to_jsonl_file(
            result, os.path.join(self._output_dir, self._run_name + ".jsonl")
        )
        return result

    def rerank_infovqa_paths(
        self, retrieval_result, path_weights, normalization="raw",
        missing_path_policy="error",
    ):
        """Rerank a cached MEHR candidate result without changing candidates."""
        alpha, beta, gamma = self._validate_path_weights(path_weights)
        candidates = retrieval_result.get("candidate_paths", [])
        if not candidates:
            raise RuntimeError("Cached retrieval result has no candidate_paths.")
        fact_values = [float(item["fact_score"]) for item in candidates]
        tile_values = [
            None if item["tile_score"] is None else float(item["tile_score"])
            for item in candidates
        ]
        root_values = [
            None if item["root_score"] is None else float(item["root_score"])
            for item in candidates
        ]
        if normalization != "raw":
            fact_values = self._normalize_values(fact_values, normalization)
            valid_tile = [value for value in tile_values if value is not None]
            valid_root = [value for value in root_values if value is not None]
            tile_norm = self._normalize_values(valid_tile, normalization)
            root_norm = self._normalize_values(valid_root, normalization)
            tile_iter = iter(tile_norm)
            root_iter = iter(root_norm)
            tile_values = [
                next(tile_iter) if value is not None else None
                for value in tile_values
            ]
            root_values = [
                next(root_iter) if value is not None else None
                for value in root_values
            ]
        scored = []
        for item, fact_score, tile_score, root_score in zip(
            candidates, fact_values, tile_values, root_values
        ):
            if any(value is None for value in (fact_score, tile_score, root_score)):
                if missing_path_policy == "error":
                    raise RuntimeError("Missing component score in cached path.")
                continue
            row = dict(item)
            row["final_score"] = (
                alpha * fact_score + beta * tile_score + gamma * root_score
            )
            scored.append(row)
        scored.sort(
            key=lambda item: (
                item["final_score"], tuple(item["root_id"]),
                tuple(item["tile_id"]), tuple(item["fact_id"]),
            ),
            reverse=True,
        )
        best_by_fact = {}
        for item in scored:
            best_by_fact.setdefault(tuple(item["fact_id"]), item)
        scored = list(best_by_fact.values())
        scored.sort(
            key=lambda item: (
                item["final_score"], tuple(item["root_id"]),
                tuple(item["tile_id"]), tuple(item["fact_id"]),
            ),
            reverse=True,
        )
        k_ret = len(retrieval_result["retrieved_paths"])
        paths = [{
            "nodes": [item["root_id"], item["tile_id"], item["fact_id"]],
            "edges": [
                [item["root_id"][1], item["tile_id"][1]],
                [item["tile_id"][1], item["fact_id"][1]],
            ],
            "score": item["final_score"],
            "specific_scores": {},
            "type": "path",
        } for item in scored[:k_ret]]
        result = dict(retrieval_result)
        result["retrieved_paths"] = paths
        result["path_reranking"] = True
        result["path_weights"] = [alpha, beta, gamma]
        result["normalization"] = normalization
        result["selected_paths"] = scored[:k_ret]
        return result

    def rerank_infovqa_paths_tree_only(self, retrieval_result, k_ret):
        """Keep fact retrieval order and select top-k candidate facts.

        The tree is used only to construct the candidate fact set. No
        root/tile/fact score fusion is performed at this stage.
        """
        candidates = retrieval_result.get("candidate_paths", [])
        if not candidates:
            raise RuntimeError("Cached retrieval result has no candidate_paths.")

        ordered = []
        seen_facts = set()
        for item in candidates:
            fact = tuple(item["fact_id"])
            if fact in seen_facts:
                continue
            seen_facts.add(fact)
            row = dict(item)
            row["final_score"] = None
            ordered.append(row)
        selected = ordered[:k_ret]
        paths = [{
            "nodes": [item["root_id"], item["tile_id"], item["fact_id"]],
            "edges": [
                [item["root_id"][1], item["tile_id"][1]],
                [item["tile_id"][1], item["fact_id"][1]],
            ],
            "score": None,
            "specific_scores": {},
            "type": "path",
        } for item in selected]
        result = dict(retrieval_result)
        result["retrieved_paths"] = paths
        result["path_reranking"] = False
        result["tree_only"] = True
        result["path_weights"] = None
        result["normalization"] = None
        result["selected_paths"] = selected
        return result

    @staticmethod
    def _validate_path_weights(weights):
        if len(weights) != 3:
            raise ValueError("Path weights must contain alpha, beta, gamma.")
        values = tuple(float(value) for value in weights)
        if any(value < 0 for value in values):
            raise ValueError("Path weights must be non-negative.")
        if abs(sum(values) - 1.0) > 1e-8:
            raise ValueError("Path weights must sum to 1.")
        return values

    @staticmethod
    def _normalize_values(values, mode):
        if mode == "raw":
            return list(values)
        if mode != "query_zscore":
            raise ValueError(f"Unknown path score normalization: {mode}")
        if not values:
            return []
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = max(variance ** 0.5, 1e-8)
        return [(value - mean) / std for value in values]

    @classmethod
    def _normalize_score_map(cls, scores, mode):
        keys = list(scores)
        values = cls._normalize_values([scores[key] for key in keys], mode)
        return dict(zip(keys, values))

    def _score_index_targets(self, indexer, targets, query_vec):
        """Score only the indexed targets required by candidate paths."""
        indexed_targets = {}
        for target in (indexer._idx2target or {}).values():
            normalized = self._target_parts(target)
            indexed_targets[normalized] = tuple(target)

        targets = [target for target in targets if target in indexed_targets]
        if not targets:
            return {}

        original_targets = [indexed_targets[target] for target in targets]
        rows = [
            indexer.get_vector_idx_for_target(target)
            for target in original_targets
        ]
        embeddings = indexer.get_embeddings().float()[rows]
        query = query_vec.to(embeddings.device, dtype=embeddings.dtype)
        scores = torch.nn.functional.cosine_similarity(
            embeddings, query.unsqueeze(0), dim=1
        ).tolist()
        return dict(zip(targets, scores))

    def _infovqa_selection(self, query_vec, root_k, tile_k, entry_k):
        top_index = self.info_level_to_indexer["root"]
        low_index = self.info_level_to_indexer["tile"]
        all_roots = [
            (self._target_parts(item["target"]), float(item["score"]))
            for item in top_index.knn_search(
                query_vec,
                top_k=root_k,
            ) if self._is_infographic_target(item["target"])
        ]

        # print(f"all_roots: {all_roots}")
        all_tiles = [
            (self._target_parts(item["target"]), float(item["score"]))
            for item in low_index.knn_search(
                query_vec,
                top_k=tile_k,
            ) if self._is_tile_target(item["target"])
        ]
        # print(f"all_tiles: {all_tiles}")
        roots = all_roots
        tiles = all_tiles
        ranked_nodes = sorted(
            dict(roots + tiles).items(), key=lambda item: item[1], reverse=True
        )[:entry_k]
        return roots, tiles, ranked_nodes, dict(all_roots), dict(all_tiles)

    def _infovqa_k_values(self, root_k, tile_k, top_entry_k, top_fact_k):
        params = self._run_config["parameters"]
        root_k = root_k or int(params.get("root_k", params.get("ablation_root_k", 100)))
        tile_k = tile_k or int(params.get("tile_k", params.get("ablation_tile_k", root_k)))
        top_entry_k = top_entry_k or int(params.get("top_entry_k", 70))
        top_fact_k = top_fact_k or int(params.get("top_fact_k", params.get("top_k", 10)))
        return root_k, tile_k, top_entry_k, top_fact_k

    def get_graph(self) -> dict[str, dict[str, dict]]:
        """Return the loaded graph as infographic -> tiles -> facts."""
        graph_view = {}
        for filename, edges in self.graph.intra_document_edges.items():
            infographic = {"tiles": {}}
            for parent_id, children in edges.items():
                if str(parent_id) != "i_1":
                    continue
                for child in children:
                    tile_target = self._target_parts(child.get_gcid())
                    if not self._is_tile_target(tile_target):
                        continue
                    tile_id = tile_target[1]
                    facts = []
                    for fact in edges.get(tile_id, []):
                        fact_target = self._target_parts(fact.get_gcid())
                        if self._is_fact_target(fact_target):
                            facts.append({"id": fact_target[1], "target": list(fact_target)})
                    infographic["tiles"][tile_id] = {
                        "target": list(tile_target),
                        "facts": facts,
                    }
            graph_view[str(filename)] = infographic
        return graph_view

    def get_infographic_hierarchy(self, infographic: str) -> dict[str, object]:
        """Return the hierarchy for one infographic."""
        graph = self.get_graph()
        for name, hierarchy in graph.items():
            if name == infographic or os.path.basename(name) == infographic:
                return {"infographic": name, **hierarchy}
        raise KeyError(f"Infographic not found in graph: {infographic}")


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run InfoPathRAG retrieval.")
    parser.add_argument("--run-name", default="infopathrag")
    parser.add_argument("--candidate-strategy", choices=("flat", "root", "tile", "root_tile"), default="root_tile")
    parser.add_argument("--ranking-strategy", choices=("fact_only", "full_path"), default="full_path")
    parser.add_argument("--top-root-k", type=int, default=100)
    parser.add_argument("--top-tile-k", type=int, default=100)
    parser.add_argument("--top-entry-k", type=int, default=70)
    parser.add_argument("--top-fact-k", type=int, default=10)
    parser.add_argument("--weights", type=float, nargs=3, default=(1 / 3, 1 / 3, 1 / 3))
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--infographic", default=None)
    args = parser.parse_args()

    retriever = MyRetriever(cli_args=[
        "--run_mode", "infopathrag",
        "--target_dataset", "InfoVQA",
        "--run_name", args.run_name,
        "--force_overwrite", "True",
    ])
    if args.graph:
        value = (
            retriever.get_infographic_hierarchy(args.infographic)
            if args.infographic else retriever.get_graph()
        )
        print(json.dumps(value, indent=2, ensure_ascii=False))
        return

    for qid in retriever._questions_manager.get_qid_list():
        question = retriever._questions_manager.get_question_instance_by_qid(qid)
        retriever.retrieve_infopathrag(
            qid=qid,
            query_vec=question.get_embedding(),
            candidate_strategy=args.candidate_strategy,
            ranking_strategy=args.ranking_strategy,
            path_weights=tuple(args.weights),
            root_k=args.top_root_k,
            tile_k=args.top_tile_k,
            top_entry_k=args.top_entry_k,
            top_fact_k=args.top_fact_k,
        )


if __name__ == "__main__":
    main()
    
    
