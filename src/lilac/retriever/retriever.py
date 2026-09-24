import os
import json
import yaml 
import time
import faiss
import torch
import pickle
import logging
import math

import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple, Union, Any

from src.lilac.basic_class.graph import Graph, Subgraph, top_level_gcid_by_low_level_gcid
from src.lilac.retriever.indexer import Indexer, Subindexer
from src.lilac.retriever.query import Question, QuestionsManager
from src.utils.utils import (
    read_json_or_jsonl, write_json_file, append_to_jsonl_file, read_yaml,
    generate_directory, error_if_directory_exists, check_file_exists, ensure_output_dir,
    REPO_ROOT,
    dataset_root, artifact_root, input_subpath, artifact_subpath, parsed_documents_path,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=os.path.join('debug', 'retriever.log'),
                    filemode='a')















METADATA_CONFIG_PATH     = f"{REPO_ROOT}/config/retriever/retriever_metadata.yaml"
RUN_CONFIG_PATH          = f"{REPO_ROOT}/config/retriever/retriever_config.yaml"



class Retriever:
    
    def __init__(self, config_path: str = RUN_CONFIG_PATH, cli_args=None):
        
        # 1) Load default config from YAML
        self._metadata_config = read_yaml(METADATA_CONFIG_PATH)
        default_config        = read_yaml(config_path)
        
        # 2) Parse command line arguments
        args = parse_arguments(cli_args)
        
        print(args)
        
        # 3) Overwrite defaults with user arguments (if provided)
        #    — exactly as user requested:
        if args.run_name is not None:
            default_config["run_name"] = args.run_name
        if args.run_mode is not None:
            default_config["run_mode"] = args.run_mode
        if args.target_dataset is not None:
            default_config["target_dataset"] = args.target_dataset
        if args.embedding_model is not None:
            default_config["embedding_model"] = args.embedding_model
        if args.force_overwrite is not None:
            default_config["force_overwrite"] = args.force_overwrite
            
        # Overwrite the nested "parameters" with relevant parse
        params = default_config.get("parameters", {})
        if args.parameter_beamwidth is not None:
            params["beam_width"] = args.parameter_beamwidth
        if args.parameter_numiterations is not None:
            params["num_iterations"] = args.parameter_numiterations
        if args.parameter_hopnum is not None:
            params["hop_mode"] = args.parameter_hopnum
        if args.parameter_targetlevel is not None:
            params["target_level"] = args.parameter_targetlevel
        if args.parameter_topk is not None:
            params["top_k"] = args.parameter_topk
        if args.parameter_rootk is not None:
            params["ablation_root_k"] = args.parameter_rootk
        if args.parameter_tilek is not None:
            params["ablation_tile_k"] = args.parameter_tilek
        default_config["parameters"] = params
        
        if args.lowlevel_text is not None:
            default_config["low_level_embeddings"]["text"] = args.lowlevel_text
        if args.lowlevel_table is not None:
            default_config["low_level_embeddings"]["table"] = args.lowlevel_table
        if args.lowlevel_image is not None:
            default_config["low_level_embeddings"]["image"] = args.lowlevel_image
        
        # 4) If run_name is still missing, auto-generate it
        #    using the logic specified.
        if not default_config.get("run_name", None):
            run_name_auto = auto_generate_run_name(default_config)
            default_config["run_name"] = run_name_auto
            
        self._run_config = default_config  
        
        # Set metadata
        self._root_path             = self._metadata_config["root_path"]
        
        # QA Pairs
        self._target_dataset        = self._run_config["target_dataset"]
        self._benchmark_dir         = dataset_root(self._metadata_config, self._target_dataset)
        self._artifact_dir          = artifact_root(self._metadata_config, self._target_dataset)
        self._parsed_documents_dir  = parsed_documents_path(self._metadata_config, self._target_dataset, "test")
        self._images_dir            = input_subpath(self._metadata_config,  self._target_dataset, "image_components_dirname", "test")
        self._subimages_dir         = artifact_subpath(self._metadata_config, self._target_dataset, "subimage_components_dirname", "test")
        self._summaries_dir         = artifact_subpath(self._metadata_config, self._target_dataset, "image_summaries_dirname", "test")
        
        self._target_embedder       = self._run_config["embedding_model"]
        self._infovqa_fact_to_parent = None
        self._infovqa_indexed_fact_targets = None
        
        # Current run
        self._run_name              = self._run_config["run_name"]
        self._output_dir            = os.path.join(self._root_path, self._metadata_config["subpath"]["algorithm_results"], self._metadata_config["algorithm_name"], self._target_dataset, "retrieval", self._run_config["run_name"])
        ensure_output_dir(self._output_dir, force_overwrite = self._run_config["force_overwrite"])
        
        self._run_function_mode     = self._run_config["run_mode"]
        self._modality_mode         = self._run_config["parameters"]["modality_mode"]
        
        # Query embedding
        self.initaite_questions_manager()
        
        # Embedding indexer
        self.initiate_indexers()
        
        # Initiate multimodal graph
        self.initiate_graph()
        
        if self._run_function_mode == "iterative_late_interaction":
            gpu_num = self._run_config["low_level_embeddings"].get("gpu_num", 0)
            device_str = f"cuda:{gpu_num}" if torch.cuda.is_available() and gpu_num >= 0 else "cpu"
            self._subindexer_low = Subindexer()
            self._subindexer_low.init_whole_embeddings(self.level_to_indexer["low"], device=device_str)
        else:
            self._subindexer_low = None
                
        return
        
        

    
    
    def initaite_questions_manager(self) -> QuestionsManager:
        """
        Load the questions and subqueries from the JSONL files.
        """
        questions_path                              = os.path.join(self._benchmark_dir, self._metadata_config["dataset_metadata"][self._target_dataset]["filename"])
        subqueries_path                             = artifact_subpath(self._metadata_config, self._target_dataset, "query_decomposition_dirname", "test", self._metadata_config["metadata_files"]["subqueries"])

        emb_dir = artifact_subpath(self._metadata_config, self._target_dataset, "embeddings_dirname", self._target_embedder)

        query_embeddings_path                       = os.path.join(emb_dir, self._metadata_config["metadata_files"]["query_embeddings"])
        query_indices_path                          = os.path.join(emb_dir, self._metadata_config["metadata_files"]["query_embeddings_index"])

        modality_agnostic_subquery_embeddings_path  = os.path.join(emb_dir, self._metadata_config["metadata_files"]["subqueries_modality_agnostic"])
        modality_agnostic_subquery_indices_path     = os.path.join(emb_dir, self._metadata_config["metadata_files"]["subqueries_modality_agnostic_index"])

        modality_aware_subquery_embeddings_path     = os.path.join(emb_dir, self._metadata_config["metadata_files"]["subqueries_modality_aware"])
        modality_aware_subquery_indices_path        = os.path.join(emb_dir, self._metadata_config["metadata_files"]["subqueries_modality_aware_index"])
        
        self._questions_manager = QuestionsManager(
            questions_path                              = questions_path,
            subqueries_path                             = subqueries_path,
            query_embeddings_path                       = query_embeddings_path,
            query_indices_path                          = query_indices_path,
            modality_aware_subquery_embeddings_path     = modality_aware_subquery_embeddings_path,
            modality_aware_subquery_indices_path        = modality_aware_subquery_indices_path,
            modality_agnostic_subquery_embeddings_path  = modality_agnostic_subquery_embeddings_path,
            modality_agnostic_subquery_indices_path     = modality_agnostic_subquery_indices_path
        )
        
        return 

    def initiate_indexers(self):
        """Load individual modality embeddings, concatenate them, and
        build unified index maps for top‑ and low‑level granularity."""

        self.level_to_indexer = {
            "top": Indexer(),
            "low": Indexer(),
            "fact": Indexer(),
        }

        self._target_embedder = self._run_config["embedding_model"]
        emb_dir = artifact_subpath(self._metadata_config, self._target_dataset, "embeddings_dirname", self._target_embedder)

        top_level_pairs = [
            # (os.path.join(emb_dir, self._run_config["top_level_embeddings"]["text"]  + ".pt"),
            #  os.path.join(emb_dir, self._run_config["top_level_embeddings"]["text"]  + ".json")),
            # (os.path.join(emb_dir, self._run_config["top_level_embeddings"]["table"] + ".pt"),
            #  os.path.join(emb_dir, self._run_config["top_level_embeddings"]["table"] + ".json")),
            (os.path.join(emb_dir, self._run_config["top_level_embeddings"]["image"] + ".pt"),
             os.path.join(emb_dir, self._run_config["top_level_embeddings"]["image"] + ".json"))
        ]
        # filter by existence — datasets like MP-DocVQA only have a subset
        # of component types (e.g. only image at the top level).
        existing_top_level_pairs = []
        for pair in top_level_pairs:
            if os.path.exists(pair[0]) and os.path.exists(pair[1]):
                existing_top_level_pairs.append(pair)
            else:
                print(f"[Retriever] Top-level embedding pair not found: {pair[0]}")
        top_level_pairs = existing_top_level_pairs

        self.level_to_indexer["top"].load_embeddings(top_level_pairs, show_progress = True)
        gpu_num = self._run_config["top_level_embeddings"].get("gpu_num", -1)
        self.level_to_indexer["top"].create_index(gpu_id = gpu_num)
        
        low_level_pairs = [
            # (os.path.join(emb_dir, self._run_config["low_level_embeddings"]["text"]  + ".pt"),
            #  os.path.join(emb_dir, self._run_config["low_level_embeddings"]["text"]  + ".json")),
            # (os.path.join(emb_dir, self._run_config["low_level_embeddings"]["table"] + ".pt"),
            #  os.path.join(emb_dir, self._run_config["low_level_embeddings"]["table"] + ".json")),
            (os.path.join(emb_dir, self._run_config["low_level_embeddings"]["image"] + ".pt"),
             os.path.join(emb_dir, self._run_config["low_level_embeddings"]["image"] + ".json")),
        ]
        facts_pt = os.path.join(emb_dir, "tile_fact.pt")
        facts_json = os.path.join(emb_dir, "tile_fact.json")
        if os.path.exists(facts_pt) and os.path.exists(facts_json):
            low_level_pairs.append((facts_pt, facts_json))
        fact_pairs = (
            [(facts_pt, facts_json)]
            if os.path.exists(facts_pt) and os.path.exists(facts_json)
            else []
        )
        # filter by existence
        existing_low_level_pairs = []
        for pair in low_level_pairs:
            if os.path.exists(pair[0]) and os.path.exists(pair[1]):
                existing_low_level_pairs.append(pair)
            else:
                print(f"[Retriever] Low-level embedding pair not found: {pair[0]}, {pair[1]}")
        low_level_pairs = existing_low_level_pairs
        
        self.level_to_indexer["low"].load_embeddings(low_level_pairs, show_progress = True)
        gpu_num = self._run_config["low_level_embeddings"].get("gpu_num", -1)
        self.level_to_indexer["low"].create_index(gpu_id = gpu_num)
        if fact_pairs:
            self.level_to_indexer["fact"].load_embeddings(fact_pairs, show_progress=True)
            self.level_to_indexer["fact"].create_index(gpu_id=gpu_num)

        self._target_level = self._run_config["parameters"]["target_level"]
        if self._target_level == "both":
            self.level_to_indexer["both"] = Indexer()
            self.level_to_indexer["both"].load_embeddings(top_level_pairs + low_level_pairs, show_progress = True)
            gpu_num = self._run_config["top_level_embeddings"].get("gpu_num", -1)
            self.level_to_indexer["both"].create_index(gpu_id = gpu_num)
        else:
            self.level_to_indexer["both"] = None

        low_index = self.level_to_indexer["fact"]
        print("[DEBUG] all low index targets:")
        for index, target in list(low_index._idx2target.items())[:30]:
            print(index, repr(target))
        return
    
    def initiate_graph(self):
        component_dir = artifact_subpath(
            self._metadata_config,
            self._target_dataset,
            "component_dirname",
        )
        self._graph_path = os.path.join(component_dir, "graph.pickle")
        os.makedirs(component_dir, exist_ok=True)

        # The tile manifest is the source of truth for the three-level
        # InfoVQA graph; do not fall back to parse_documents for it.
        tile_manifest_candidates = (
            os.path.join(self._benchmark_dir, "tiles", "manifest.json"),
            os.path.join(REPO_ROOT, "datasets", "InfoVQA", "tiles", "manifest.json"),
        )
        tile_manifest = next(
            (path for path in tile_manifest_candidates if os.path.exists(path)),
            None,
        )

        use_tile_graph = (
            self._target_dataset == "InfoVQA"
            and tile_manifest is not None
        )
        facts_directory = os.path.join(
            REPO_ROOT,
            "artifacts",
            self._target_dataset,
            "facts_each_tile",
        )

        if check_file_exists(self._graph_path) and not use_tile_graph:
            print("[Retriever] Loading existing graph …")
            with open(self._graph_path, "rb") as f:
                self.graph = pickle.load(f)
            print(f"[Retriever] Graph loaded from {self._graph_path}")
        else:
            self.graph = Graph(
                multimodal_documents_directory=self._parsed_documents_dir,
                images_directory=REPO_ROOT if use_tile_graph else self._images_dir,
                subimages_directory=self._subimages_dir,
                summaries_directory=self._summaries_dir,
            )
            if use_tile_graph:
                print("use tile graph")
                tile_manifest = os.path.join(facts_directory, "manifest.json")
                self.graph.load_tile_manifest(tile_manifest, facts_directory)
                documents_with_facts = 0
                for filename, edges in self.graph.intra_document_edges.items():
                    tile_ids = [
                        component_id for component_id in edges
                        if component_id != "i_1"
                        and "_t" in component_id
                        and "_f" not in component_id
                    ]
                    if edges.get("i_1") and any(
                        edges.get(tile_id) for tile_id in tile_ids
                    ):
                        documents_with_facts += 1
                if not documents_with_facts:
                    raise ValueError(
                        "Tile manifest graph did not produce the expected "
                        "i_1 -> tile -> fact hierarchy."
                    )
                print(
                    "[Retriever] Loaded three-level graph: "
                    "original image -> tiles -> facts"
                )
            else:
                self.graph.parse_documents()
            with open(self._graph_path, "wb") as f:
                pickle.dump(self.graph, f)
            print(f"[Retriever] Graph saved to {self._graph_path}")
        return

    def _get_infovqa_ablation_maps(self, fact_index):
        """Build graph/index lookup maps once for all ablation queries."""
        if (
            self._infovqa_fact_to_parent is not None
            and self._infovqa_indexed_fact_targets is not None
        ):
            return (
                self._infovqa_fact_to_parent,
                self._infovqa_indexed_fact_targets,
            )

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

        indexed_fact_targets = {}
        for indexed_target in (fact_index._idx2target or {}).values():
            normalized = self._target_parts(indexed_target)
            if self._is_fact_target(normalized):
                indexed_fact_targets[normalized] = tuple(indexed_target)

        self._infovqa_fact_to_parent = fact_to_parent
        self._infovqa_indexed_fact_targets = indexed_fact_targets
        print(
            "[Retriever] Prepared InfoVQA ablation maps: "
            f"{len(fact_to_parent):,} facts, {len(indexed_fact_targets):,} indexed facts",
            flush=True,
        )
        return fact_to_parent, indexed_fact_targets
        
        
    
    
    
    
    
    
        
    def run(self):
        
        # qids = self._labeled_benchmark.get_qid_list()
        # target_qids = ['41142.jpeg-2', '43600.jpeg-1', '43600.jpeg-3']
        qids = self._questions_manager.get_qid_list()
        # qids = [qid for qid in qids if qid in target_qids]
        
        for qid in tqdm(qids, desc = "Retrieving"):
            
            question_instance: Question = self._questions_manager.get_question_instance_by_qid(qid)
            question_embedding = question_instance.get_embedding()
            subquery_embeddings = question_instance.get_subquery_embedding_list(self._modality_mode)
            if self._run_function_mode == "iterative_late_interaction":
                self._beam_width = self._run_config["parameters"]["beam_width"]
                self._num_iterations = self._run_config["parameters"]["num_iterations"]
                self.retrieve_iterative_late_interaction(qid, question_embedding, subquery_embeddings)
            elif self._run_function_mode == "infovqa_ablation":
                self.retrieve_infovqa_ablation(qid, question_embedding)
            else:
                raise ValueError(
                    f"Unsupported retrieval mode: {self._run_function_mode}. "
                    "Use iterative_late_interaction or infovqa_ablation."
                )
            
        run_config_path = os.path.join(self._output_dir, "run_config.yaml")
        with open(run_config_path, "w") as f:
            yaml.dump(self._run_config, f)

        return

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
            _, component_id = Retriever._target_parts(target)
        except ValueError:
            return False
        return (
            component_id.startswith("i_1_t")
            and "_f" not in component_id
        )


    @staticmethod
    def _is_fact_target(target):
        try:
            _, component_id = Retriever._target_parts(target)
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

    def retrieve_infovqa_ablation(self, qid, query_vec, variant=None,
                                  root_k=None, tile_k=None, k_ret=None,
                                  path_reranking=False, path_weights=(1.0, 0.0, 0.0),
                                  normalization="raw", missing_path_policy="error"):
        """Dispatch one of the four InfoVQA candidate-set experiments."""
        variant = variant or self._run_config["parameters"].get(
            "ablation_variant", "root_tile_facts"
        )
        methods = {
            "flat_facts": self.retrieve_infovqa_flat_facts,
            "root_facts": self.retrieve_infovqa_root_facts,
            "tile_facts": self.retrieve_infovqa_tile_facts,
            "root_tile_facts": self.retrieve_infovqa_root_tile_facts,
        }
        try:
            retrieve_variant = methods[variant]
        except KeyError as exc:
            raise ValueError(f"Unknown InfoVQA ablation variant: {variant}") from exc
        return retrieve_variant(
            qid, query_vec, root_k, tile_k, k_ret, path_reranking,
            path_weights, normalization, missing_path_policy,
        )

    def _retrieve_infovqa_candidate_facts(
        self, qid, query_vec, candidate_facts, selected_roots, selected_tiles,
        variant, k_ret, root_scores=None, tile_scores=None,
        path_reranking=False, path_weights=(1.0, 0.0, 0.0),
        normalization="raw", missing_path_policy="error",
    ):
        """Score, rank, and serialize a common InfoVQA fact candidate set."""
        started = time.perf_counter()
        fact_index = self.level_to_indexer.get("fact")
        if fact_index is None or fact_index.get_embeddings() is None:
            raise RuntimeError("InfoVQA ablation requires tile_fact embeddings.")
        fact_to_parent, indexed_fact_targets = self._get_infovqa_ablation_maps(
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
            "ablation_variant": variant,
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
                for fact in sorted(path_options)
                for option in path_options[fact]
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

    def _infovqa_selection(self, query_vec, root_k, tile_k, k_ret):
        top_index = self.level_to_indexer["top"]
        low_index = self.level_to_indexer["low"]
        all_roots = [
            (self._target_parts(item["target"]), float(item["score"]))
            for item in top_index.knn_search(
                query_vec, top_k=top_index.get_embeddings().shape[0]
            ) if self._is_infographic_target(item["target"])
        ]
        all_tiles = [
            (self._target_parts(item["target"]), float(item["score"]))
            for item in low_index.knn_search(
                query_vec, top_k=low_index.get_embeddings().shape[0]
            ) if self._is_tile_target(item["target"])
        ]
        roots = all_roots[:root_k]
        tiles = all_tiles[:tile_k]
        ranked_nodes = sorted(
            dict(roots + tiles).items(), key=lambda item: item[1], reverse=True
        )[:k_ret]
        return roots, tiles, ranked_nodes, dict(all_roots), dict(all_tiles)

    def _infovqa_k_values(self, root_k, tile_k, k_ret):
        params = self._run_config["parameters"]
        k_ret = k_ret or int(params.get("top_k", 10))
        root_k = root_k or int(params.get("ablation_root_k", 100))
        tile_k = tile_k or int(params.get("ablation_tile_k", root_k))
        return root_k, tile_k, k_ret

    def _retrieve_infovqa_selected(self, qid, query_vec, variant,
                                   root_k, tile_k, k_ret, mode,
                                   path_reranking=False, path_weights=(1.0, 0.0, 0.0),
                                   normalization="raw", missing_path_policy="error"):
        root_k, tile_k, k_ret = self._infovqa_k_values(root_k, tile_k, k_ret)
        roots, tiles, ranked_nodes, all_root_scores, all_tile_scores = self._infovqa_selection(
            query_vec, root_k, tile_k, k_ret
        )
        fact_to_parent, _ = self._get_infovqa_ablation_maps(
            self.level_to_indexer["fact"]
        )
        root_set, tile_set = dict(roots), dict(tiles)
        if mode == "root":
            selected = {target for target, _ in roots}
        elif mode == "tile":
            selected = {target for target, _ in tiles}
        else:
            selected = {target for target, _ in ranked_nodes}
        candidate_facts = {
            fact for fact, paths in fact_to_parent.items()
            if any(
                (mode == "root" and root in selected)
                or (mode == "tile" and tile in selected)
                or (mode == "root_tile" and (root in selected or tile in selected))
                for root, tile in paths
            )
        }
        selected_roots = list(root_set) if mode == "root" else []
        selected_tiles = list(tile_set) if mode == "tile" else []
        if mode == "root_tile":
            selected_roots = [target for target, _ in ranked_nodes if target in root_set]
            selected_tiles = [target for target, _ in ranked_nodes if target in tile_set]
        return self._retrieve_infovqa_candidate_facts(
            qid, query_vec, candidate_facts, selected_roots, selected_tiles,
            variant, k_ret, all_root_scores, all_tile_scores, path_reranking, path_weights,
            normalization, missing_path_policy,
        )

    def retrieve_infovqa_flat_facts(self, qid, query_vec, root_k=None,
                                    tile_k=None, k_ret=None, path_reranking=False,
                                    path_weights=(1.0, 0.0, 0.0), normalization="raw",
                                    missing_path_policy="error"):
        del root_k, tile_k
        k_ret = k_ret or int(self._run_config["parameters"].get("top_k", 10))
        fact_index = self.level_to_indexer.get("fact")
        if fact_index is None:
            raise RuntimeError("InfoVQA ablation requires tile_fact embeddings.")
        facts = {
            self._target_parts(target) for target in (fact_index._idx2target or {}).values()
            if self._is_fact_target(target)
        }
        return self._retrieve_infovqa_candidate_facts(
            qid, query_vec, facts, [], [], "flat_facts", k_ret, {},
            {}, path_reranking, path_weights, normalization, missing_path_policy,
        )

    def retrieve_infovqa_root_facts(self, qid, query_vec, root_k=None,
                                    tile_k=None, k_ret=None, path_reranking=False,
                                    path_weights=(1.0, 0.0, 0.0), normalization="raw",
                                    missing_path_policy="error"):
        del tile_k
        return self._retrieve_infovqa_selected(
            qid, query_vec, "root_facts", root_k, None, k_ret, "root",
            path_reranking, path_weights, normalization, missing_path_policy,
        )

    def retrieve_infovqa_tile_facts(self, qid, query_vec, root_k=None,
                                    tile_k=None, k_ret=None, path_reranking=False,
                                    path_weights=(1.0, 0.0, 0.0), normalization="raw",
                                    missing_path_policy="error"):
        del root_k
        return self._retrieve_infovqa_selected(
            qid, query_vec, "tile_facts", None, tile_k, k_ret, "tile",
            path_reranking, path_weights, normalization, missing_path_policy,
        )

    def retrieve_infovqa_root_tile_facts(self, qid, query_vec, root_k=None,
                                         tile_k=None, k_ret=None,
                                         path_reranking=False,
                                         path_weights=(1.0, 0.0, 0.0),
                                         normalization="raw",
                                         missing_path_policy="error"):
        return self._retrieve_infovqa_selected(
            qid, query_vec, "root_tile_facts", root_k, tile_k, k_ret, "root_tile",
            path_reranking, path_weights, normalization, missing_path_policy,
        )
    
    
    def retrieve_iterative_late_interaction(
        self,
        qid: str,
        query_vec: torch.Tensor,
        query_vec_list: List[torch.Tensor],
        k_ret: int | None = None,
    ):
        """
        Iterative multi-hop retrieval (with a fixed "1_hop" at each iteration),
        using "late interaction" re-ranking on top-level subgraph edges.
        After each iteration, we collect up to 'top_k' distinct nodes to feed
        into the next iteration, thereby expanding the subgraph exploration.
        """
        logging.info(f'-------------qid: {qid}------------')
        if k_ret is None:
            k_ret = int(self._run_config["parameters"]["top_k"])
        beam_width = self._run_config["parameters"]["beam_width"]
        # Number of iterations
        num_iterations = self._run_config["parameters"].get("num_iterations", 1)

        if not query_vec_list:
            query_vec_list = [query_vec]

        # --------------------------------------------
        # 0) We do one knn search on the low-level index
        #    to get an initial set of top-level nodes.
        # --------------------------------------------
        total_start = time.perf_counter()

        # Prepare a dictionary to accumulate time across iterations
        timing_acc = {
            "knn_search(ms)": 0.0,
            "top_level_gcid_organizing(ms)": 0.0,
            "subgraph_extraction(ms)": 0.0,
            "gpu_calculation(ms)": 0.0,
            "cpu_late_interaction(ms)": 0.0,
            "in_edge_reranking(ms)": 0.0
        }

        # 0.1) k-NN in low-level index
        t_knn0 = time.perf_counter()
        low_level_indexer = self.level_to_indexer["low"]
        # Pull many results (e.g. 2048) to avoid losing potential neighbors
        # logging.info(f"query_vec: {query_vec}")
        initial_knn_results = low_level_indexer.knn_search(query_vec, top_k = 2048)
        logging.info(f"initial_knn_results: {initial_knn_results}")
        t_knn1 = time.perf_counter()
        timing_acc["knn_search(ms)"] += (t_knn1 - t_knn0) * 1000

        # 0.2) Convert them to top-level GCIDs & keep distinct
        t_top0 = time.perf_counter()
        low_level_gcid_list = [r["target"] for r in initial_knn_results]
        top_level_gcid_list = [top_level_gcid_by_low_level_gcid(g) for g in low_level_gcid_list]
        logging.info(f"low_level_gcid_list: {low_level_gcid_list}")
        logging.info(f"top_level_gcid_list: {top_level_gcid_list}")

        seen = set()
        distinct_top_gcid_list = []
        for gcid in top_level_gcid_list:
            if gcid not in seen:
                seen.add(gcid)
                distinct_top_gcid_list.append(gcid)

        # Truncate to beam_width for the initial iteration
        if len(distinct_top_gcid_list) > beam_width:
            distinct_top_gcid_list = distinct_top_gcid_list[:beam_width]
        t_top1 = time.perf_counter()
        timing_acc["top_level_gcid_organizing(ms)"] += (t_top1 - t_top0) * 1000

        iteration_top_level_nodes = distinct_top_gcid_list
        logging.info(f"iteration_top_level_nodes: {iteration_top_level_nodes}")
        final_scored_edges = []


        # --------------------------------------------
        # Iterative expansion
        # --------------------------------------------
        for it in range(num_iterations):
            # logging.info(f"-------qid {qid} - iteration [{it}]--------")
            
            # 1) Build subgraph from iteration_top_level_nodes
            t_sg0 = time.perf_counter()
            subgraph = Subgraph(
                graph=self.graph,
                global_top_level_component_id_list=iteration_top_level_nodes,
            )
            # Force a 1-hop expansion so neighbors are included
            subgraph.extract_edges("1_hop")
            subgraph_top_gcid_list = subgraph.get_top_level_gcids_list()
            logging.info(f"subgraph_top_gcid_list: {subgraph_top_gcid_list}") # top-level nodes with 1-hop in the graph 
            t_sg1 = time.perf_counter()
            timing_acc["subgraph_extraction(ms)"] += (t_sg1 - t_sg0) * 1000

            # 2) Build subindex + compute subquery scores on GPU
            t_gpu0 = time.perf_counter()
            self._subindexer_low.build_subindex(subgraph_top_gcid_list, self.graph)
            self._subindexer_low.compute_subquery_scores(query_vec_list)
            low_level_gcids_num = self._subindexer_low.get_low_gcids_num()
            t_gpu1 = time.perf_counter()
            timing_acc["gpu_calculation(ms)"] += (t_gpu1 - t_gpu0) * 1000

            # 3) Score each edge in CPU using late interaction
            t_cpu0 = time.perf_counter()
            edge_list = subgraph.get_retrieval_units_list()
            logging.info(f"edge_list: {edge_list}")
            scored = []
            for e_pair in edge_list:
                if len(e_pair) == 1:
                    e_tup = (tuple(e_pair[0]),)
                    nodes, sc = self._subindexer_low.score_node(e_tup)
                elif len(e_pair) == 2:
                    e_tup = [tuple(e_pair[0]), tuple(e_pair[1])]
                    nodes, sc = self._subindexer_low.score_edge(e_tup)
                else:
                    raise ValueError(f"Invalid edge pair: {e_pair}")
                scored.append({"edge": nodes, "score": sc})
            logging.info(f"scored: {scored}")
            t_cpu1 = time.perf_counter()
            timing_acc["cpu_late_interaction(ms)"] += (t_cpu1 - t_cpu0) * 1000

            # 4) In-edge re-ranking and selecting top edges
            t_re0 = time.perf_counter()
            logging.info("In-edge re-ranking and selecting top edges...")
            # 4.1) Remove duplicates, keep max score
            unique_map = {}
            for item in scored:
                edge_key = frozenset(item["edge"])
                if edge_key not in unique_map:
                    unique_map[edge_key] = item
                else:
                    if item["score"] > unique_map[edge_key]["score"]:
                        unique_map[edge_key] = item
            scored = list(unique_map.values())

            # 4.2) Sort & keep top-k edges
            scored.sort(key=lambda x: x["score"], reverse=True)
            top_k_edges = scored[:k_ret] if len(scored) > k_ret else scored

            # 4.3) Re‑order GCIDs within each edge by relevancy to the query
            relevancy_cache = {}
            top_emb_matrix = self.level_to_indexer["top"].get_embeddings()
            device = top_emb_matrix.device
            query_on_dev = query_vec.to(device, dtype=top_emb_matrix.dtype)

            def get_relevancy(gcid: tuple[str, str]) -> float:
                if gcid in relevancy_cache:
                    return relevancy_cache[gcid]
                idxer = self.level_to_indexer["top"]
                try:
                    row_idx = idxer.get_vector_idx_for_target(gcid)
                except KeyError:
                    logging.error(f"[Retriever] KeyError in in-edge reranking: {gcid}")
                    relevancy_cache[gcid] = 0.0
                    return 0.0
                node_vec = top_emb_matrix[row_idx, :]
                score = float(torch.dot(node_vec, query_on_dev).item())
                relevancy_cache[gcid] = score
                return score

            # logging.info(f"top_k_edges before get_relevancy(): {top_k_edges}")
            for item in top_k_edges:
                if len(item["edge"]) > 1:
                    sorted_edge = sorted(
                        item["edge"], 
                        key=lambda g: get_relevancy(g), 
                        reverse=True
                    )
                    item["edge"] = sorted_edge
            t_re1 = time.perf_counter()
            timing_acc["in_edge_reranking(ms)"] += (t_re1 - t_re0) * 1000

            # Save the final edges of this iteration
            final_scored_edges = top_k_edges

            # --------------------------------------------
            # 5) Gather up to beam_width distinct nodes from top_k_edges
            #    to pass to the next iteration
            # --------------------------------------------
            new_top_level_nodes = []
            seen_nodes = set()
            for item in top_k_edges:
                for node_gcid in item["edge"]:
                    if node_gcid not in seen_nodes:
                        seen_nodes.add(node_gcid)
                        new_top_level_nodes.append(node_gcid)
                        if len(new_top_level_nodes) >= beam_width:
                            break
                if len(new_top_level_nodes) >= beam_width:
                    break

            # If no new nodes appear, we've converged
            if not new_top_level_nodes:
                break

            iteration_top_level_nodes = new_top_level_nodes
            logging.info(f"iteration_top_level_nodes: {iteration_top_level_nodes}")

        # --------------------------------------------
        # After all iterations, produce final retrieval_obj
        # --------------------------------------------
        total_end = time.perf_counter()

        retrieval_obj = {
            "qid": qid,
            "retrieved_units": [],
            "middle_results": {
                "low_level_gcids_num": low_level_gcids_num,
                "subgraph_nodes_num": len(subgraph_top_gcid_list),
                "subgraph_edges_num": len(edge_list),
            },
            "time": {
                "retrieval_time(ms)": (total_end - total_start) * 1000,
                **timing_acc  # Spread out the accumulated times
            },
        }

        # Convert final_scored_edges into the legacy "retrieved_units" structure
        retrieved_units = []
        for it in final_scored_edges:
            edge_nodes = it["edge"]
            if len(edge_nodes) == 2:
                retrieved_units.append({
                    "nodes": [list(edge_nodes[0]), list(edge_nodes[1])],
                    "edges": [(0, 1)],
                    "score": it["score"],
                    "specific_scores": {},
                    "type": "edge",
                })
            else:
                # single node
                retrieved_units.append({
                    "nodes": [list(edge_nodes[0])],
                    "edges": [],
                    "score": it["score"],
                    "specific_scores": {},
                    "type": "node",
                })
        retrieval_obj["retrieved_units"] = retrieved_units

        # Append to JSONL
        outfile = os.path.join(self._output_dir, self._run_name + ".jsonl")
        append_to_jsonl_file(retrieval_obj, outfile)
        return retrieval_obj
    
    
    
    
    
    
    # ───────────────────────────────────────────────────────────────────────
    # Late-interaction re-ranking over top-level edges
    # ───────────────────────────────────────────────────────────────────────
    
def main():
    retriever = Retriever()
    retriever.run()
    return
    
def parse_arguments(argv=None) -> argparse.Namespace:
    """
    Parse command-line arguments, each one overriding the default YAML config.
    """

    parser = argparse.ArgumentParser(
        description="Override default retriever_config.yaml settings."
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of the run (if not given, will be auto-generated)."
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        choices=["iterative_late_interaction", "infovqa_ablation"],
        default=None,
        help="Choose iterative late interaction or InfoVQA ablation."
    )
    parser.add_argument(
        "--target_dataset",
        type=str,
        choices=["InfoVQA", "SlideVQA", "MP-DocVQA", "MultimodalQA", "MMCoQA"],
        default=None,
        help="Which dataset to use."
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        choices=["MM-Embed", "LLaVE-7B", "NV-Embed-v2", "UniME", "QQMM", "VLM2Vec", "MMe5", "Jasper"],
        default=None,
        help="Which embedding model to use."
    )
    
    parser.add_argument(
        "--parameter_topk",
        type=int,
        default=None,
        help="Override for 'top_k' in config (default=200)."
    )
    parser.add_argument(
        "--parameter_rootk",
        type=int,
        default=None,
        help="Number of top-level root candidates for tree traversal.",
    )
    parser.add_argument(
        "--parameter_tilek",
        type=int,
        default=None,
        help="Number of tile candidates for tree traversal.",
    )
    parser.add_argument(
        "--parameter_beamwidth",
        type=int,
        default=None,
        help="Override for 'beam_width' in config (default=50)."
    )
    parser.add_argument(
        "--parameter_numiterations",
        type=int,
        default=None,
        help="Number of retrieval iterations."
    )
    
    parser.add_argument(
        "--lowlevel_text",
        type = str,
        default = None,        
    )
    parser.add_argument(
        "--lowlevel_table",
        type = str,
        default = None,        
    )
    parser.add_argument(
        "--lowlevel_image",
        type = str,
        default = None,        
    )
    
    parser.add_argument(
        "--force_overwrite", 
        type = bool,
        help = "Force overwrite of existing output files."
    )
    
    parser.add_argument(
        "--parameter_hopnum",
        type=str,
        default=None,
        choices=["1_hop"],  # Expand if more hop modes exist
        help="Override for 'hop_mode' in config (default=1_hop)."
    )
    parser.add_argument(
        "--parameter_targetlevel",
        type=str,
        choices=["top", "low", "both"],
        default=None,
        help="Override for 'target_level' in config."
    )
    
    args = parser.parse_args(argv)

    return args


def auto_generate_run_name(config: Dict[str, Any]) -> str:
    """
    If run_name is not given, build it from:
      embedding_model + run_mode + relevant param-value pairs.
    The relevant parameters differ by run_mode:
      single_knn -> (target_level, top_k)
      single_topdown -> (beam_width, top_k)
      decomposed_topdown -> (beam_width, top_k)
      late_interaction -> (beam_width, hop_mode, target_level, top_k)

    The appended string for each parameter is simply paramName + paramValue,
    all in lower-case, removing underscores in the paramName, e.g. 'targetlevellow_topk200'.
    """

    embedding_model = config.get("embedding_model", "UnknownModel")
    run_mode        = config.get("run_mode", "UnknownMode")

    # Define relevant parameters for each run_mode
    relevant_map = {
        "iterative_late_interaction": ["beam_width", "num_iterations", "top_k"],
        "infovqa_ablation":   ["top_k", "ablation_root_k", "ablation_tile_k"],
    }

    # Find relevant params. If run_mode is not recognized,
    # we skip or assume an empty param list.
    relevant_params = relevant_map.get(run_mode, [])

    # Build the suffix part for run_name
    parts = []
    for p in relevant_params:
        # config["parameters"] holds these
        param_val = config["parameters"].get(p, None)
        if param_val is not None:
            # e.g. p = "target_level" -> paramNameNoUnderscore = "targetlevel"
            paramNameNoUnderscore = p.replace("_", "")
            # e.g. paramVal -> "low"
            part = f"{paramNameNoUnderscore}{param_val}"
            parts.append(part.lower())

    # Combine
    suffix = "_".join(parts) if parts else "defaultparams"
    run_name = f"{embedding_model}_{run_mode}_{suffix}"
    return run_name





if __name__ == "__main__":
    
    main()