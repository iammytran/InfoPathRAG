import os
import sys
import yaml
import time
import json
import argparse
import math
import torch
import multiprocessing as mp
import tempfile
import shutil
import re
import tempfile
from pathlib import Path

from typing import Dict, Any, List
from tqdm import tqdm

# --------------- Existing imports from your code base ---------------
from src.lilac.basic_class.graph import Graph
from src.lilac.basic_class.component import Component
from src.lilac.basic_class.image import Image
from src.lilac.basic_class.text import Text
from src.experiment.utils.constants import BenchmarkType, AlgorithmName
from src.experiment.parser.retrieval_result_parser import parse_retrieval_results
from src.utils.utils import (
    read_json_or_jsonl, read_yaml, ensure_output_dir, REPO_ROOT,
    dataset_root, artifact_subpath,
)
from src.models.mllm.qwen2_5_vl_7b import Qwen2_5_VL
from src.lilac.prompts.prompts import (
    INSTRUCTION_PROMPT_NONIMAGE,
    INSTRUCTION_PROMPT_IMAGE,
    INSTRUCTION_PROMPT_PATH,
    DEMONSTRATION_PROMPT,
    DEMONSTRATION_PROMPT_PATH,
    PAGE_PROMPT
)


GENERATOR_DEFAULT_YAML_CONFIG_PATH  = f"{REPO_ROOT}/config/retriever/generator_config.yaml"
METADATA_CONFIG_PATH                = f"{REPO_ROOT}/config/retriever/retriever_metadata.yaml"








# --------------------------------------------------------------------
# 1) Argparse + config
# --------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final answers with Qwen2.5-VL model (parallel).")

    parser.add_argument("--run_name",                 type=str,   help="Name of the run.")
    parser.add_argument("--target_dataset",           type=str,   choices=["MultimodalQA","MMCoQA","WebQA","MP-DocVQA","SlideVQA","InfoVQA"], help="Which dataset to use.")
    parser.add_argument("--generation_model",         type=str,   help="Which generation model to use. (Example: Qwen2.5-VL-7B)")
    parser.add_argument("--retrieval_results_path",   type=str,   help="Path to the retrieval results (.jsonl).")
    parser.add_argument("--num_components",           type=int,   help="Number of components to use from retrieval results.")
    parser.add_argument("--num_paths",                type=int,   help="Number of retrieved paths to use.")
    parser.add_argument("--use_retrieved_paths",      action="store_true", help="Use top-k retrieved paths instead of flattened components.")
    parser.add_argument("--force_overwrite",          type=bool,  help="Force overwrite of existing output files.")
    parser.add_argument("--num_gpus",                 type=int,   help="Number of GPUs to use for parallel generation.")
    parser.add_argument("--path_to_model",            type=str,   help="Path to Qwen2.5-VL-7B or similar.")
    parser.add_argument("--images_subpath",           type=str,   choices = ["image_components", "pdf_pages"], help="Subpath to the images in the dataset directory.")
    parser.add_argument("--text_only",                type=bool)
    args = parser.parse_args()
    return args

def load_config(config_path: str, cli_args: argparse.Namespace) -> Dict[str, Any]:
    config = read_yaml(config_path)

    if cli_args.run_name is not None:
        config["run_name"] = cli_args.run_name
    if cli_args.target_dataset is not None:
        config["target_dataset"] = cli_args.target_dataset
    if cli_args.generation_model is not None:
        config["generation_model"] = cli_args.generation_model
    if cli_args.retrieval_results_path is not None:
        config["retrieval_results_path"] = cli_args.retrieval_results_path
    if cli_args.num_components is not None:
        config["num_components"] = cli_args.num_components
    if cli_args.num_paths is not None:
        config["num_paths"] = cli_args.num_paths
    if cli_args.use_retrieved_paths:
        config["use_retrieved_paths"] = True
    if cli_args.force_overwrite is not None:
        config["force_overwrite"] = cli_args.force_overwrite
    if cli_args.num_gpus is not None:
        config["num_gpus"] = cli_args.num_gpus
    if cli_args.path_to_model is not None:
        config["path_to_model"] = cli_args.path_to_model
    if cli_args.images_subpath is not None:
        config["images_subpath"] = cli_args.images_subpath
    if cli_args.text_only is not None:
        config["text_only"] = cli_args.text_only

    if not config.get("run_name"):
        config["run_name"] = f"{config['generation_model']}_{config['target_dataset']}_run"
    return config









# --------------------------------------------------------------------
# 2) The parallel-enabled Generator class
# --------------------------------------------------------------------
class Generator:

    def __init__(self, config: Dict[str, Any]):
        mp.set_start_method("spawn", force=True)  # needed for CUDA + multiproc

        self.config = config
        self._metadata_config  = read_yaml(METADATA_CONFIG_PATH)

        self.run_name          = config["run_name"]
        self.target_dataset    = config["target_dataset"]
        self.generation_model  = config["generation_model"]
        self.retrieval_results_path = "/workspace/LILaC/algorithm_results/LILaC/InfoVQA/retrieval_backup/info_vqa_tree_traversal_copy/info_vqa_tree_traversal.jsonl"
        self.num_components    = config["num_components"]
        self.num_paths         = 1
        self.use_retrieved_paths = config.get("use_retrieved_paths", False)
        self.force_overwrite   = config["force_overwrite"]
        self.num_gpus          = config["num_gpus"]
        self.path_to_model     = config.get("path_to_model", None)
        self.text_only         = config.get("text_only", False)
        
        self.images_subpath    = config["images_subpath"]
        
        root_path      = self._metadata_config["root_path"]
        algo_results   = self._metadata_config["subpath"]["algorithm_results"]
        algorithm_name = self._metadata_config["algorithm_name"] 

        self.output_dir = os.path.join(
            root_path,
            algo_results,
            algorithm_name,
            self.target_dataset,
            "generation",
            self.run_name
        )
        ensure_output_dir(self.output_dir, force_overwrite=self.force_overwrite)
        self.output_jsonl_file = os.path.join(self.output_dir, f"{self.run_name}.jsonl")

        # Load Graph & QA
        self.graph           = self._initiate_graph()
        self.qid_to_question = self._load_questions()

    # ----------------------------------------------------------------
    # 2.1) Graph + QA loading
    # ----------------------------------------------------------------
    def _initiate_graph(self) -> Graph:
        graph_path = artifact_subpath(
            self._metadata_config, self.target_dataset, "component_dirname", "graph.pickle"
        )

        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"No graph found at {graph_path}")

        print(f"[Generator] Loading graph from {graph_path}...")
        import pickle
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        print(f"[Generator] Graph loaded")
        return graph

    def _load_questions(self) -> Dict[str, str]:
        question_filename = self._metadata_config["dataset_metadata"][self.target_dataset]["filename"]
        questions_path    = os.path.join(dataset_root(self._metadata_config, self.target_dataset), question_filename)

        print(f"[Generator] Loading questions from {questions_path}...")
        qa_list = read_json_or_jsonl(questions_path)

        qid_to_text = {}
        for item in qa_list:
            qid_to_text[item["qid"]] = item["question"]
        print(f"[Generator] Loaded {len(qid_to_text)} questions.")
        return qid_to_text










    # ----------------------------------------------------------------
    # 2.2) Generate pipeline (multiprocess)
    # ----------------------------------------------------------------
    def run(self):
        # 1) Prepare tasks: each is (qid, prompt, image_paths, serialization_time)
        tasks = self._build_tasks()

        print("Now we'll spawn multiple processes to generate answers.")

        # 2) chunk among GPUs
        total     = len(tasks)
        num_gpus  = self.num_gpus
        chunk_sz  = math.ceil(total / num_gpus)

        # 3) spawn processes
        
        tmp_dir = tempfile.mkdtemp(prefix="gen_shards_" + self.run_name + "_")
        procs   = []
        used_gpus = 0
        for g in range(num_gpus):
            start = g * chunk_sz
            end   = min((g + 1) * chunk_sz, total)
            if start >= end:
                break
            tasks_slice = tasks[start:end]

            p = mp.Process(
                target=self._worker_process,
                args=(g, tasks_slice, tmp_dir)
            )
            p.start()
            procs.append(p)
            used_gpus += 1

        # 4) wait & check
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"[Main] Worker {p.name} exit code: {p.exitcode}")

        # 5) merge partial results
        final_path = self.output_jsonl_file
        self._merge_shards(tmp_dir, final_path, used_gpus)

        # 6) cleanup
        shutil.rmtree(tmp_dir)
        print(f"[Generator] Done. Merged final results to {final_path}")





    # ----------------------------------------------------------------
    # 2.3) Build tasks + retrieval, prompts
    # ----------------------------------------------------------------
    def _build_tasks(self) -> List[tuple]:
        
        """ Return a list of (qid, prompt, image_paths, serialization_time). """
        retrieval_map = self._load_retrieval_results()
        print(f"[Generator] Building tasks for {len(retrieval_map)} questions...")
        
        tasks = []
        for qid in retrieval_map:
            start_t = time.time()
            prompt, image_paths = self.build_prompt(qid, retrieval_map[qid])
            serialization_time = (time.time() - start_t) * 1000
            tasks.append((qid, prompt, image_paths, serialization_time))
            
        print(f"[Generator] Built {len(tasks)} tasks.")
        return tasks

    def _load_retrieval_results(self) -> Dict[str, List[Any]]:
        
        print("[Generator] Loading retrieval results...")
        
        # Decide data type from the central dataset registry (no name-based branch).
        ds_meta = self._metadata_config["dataset_metadata"][self.target_dataset]
        data_type = BenchmarkType.from_value(ds_meta["type"])


        algo_name = AlgorithmName.OMG  # placeholder
        if "VisRAG" in self.retrieval_results_path:
            algo_name = AlgorithmName.VISRAG
        
        qa_path_for_parser = os.path.join(
            dataset_root(self._metadata_config, self.target_dataset),
            self._metadata_config["dataset_metadata"][self.target_dataset]["filename"],
        )

        retrieval_manager = parse_retrieval_results(
            algorithm_name        = algo_name,
            data_type             = data_type,
            qa_data_path          = qa_path_for_parser,
            retrieval_result_path = self.retrieval_results_path,
        )

        qid_to_rresult = retrieval_manager.get_qid_to_rresult()
        out_map = {}
        for qid, srres in qid_to_rresult.items():
            if self.use_retrieved_paths and srres.get_retrieved_paths():
                print(f"lấy paths")
                print(f"self.num_paths: {self.num_paths}")
                top = srres.get_retrieved_paths()[:1]
                print(f"top: {top}")
            else:
                top = srres.get_retrieved_components()[: self.num_components]
            out_map[qid] = top
            
        print(f"[Generator] Loaded {len(out_map)} retrieval results.")
        return out_map

    def build_prompt(self, qid: str, top_gcids: List[Any]) -> tuple[str, List[str]]:
        """ Return (text_prompt, image_paths). """
        
        question_text = self.qid_to_question[qid]

        if not top_gcids:
            raise ValueError(f"No retrieved results found for question {qid}.")

        if isinstance(top_gcids[0], tuple):
            if top_gcids[0][0] in top_gcids[0][1]:
                top_gcids = [it[1] for it in top_gcids]
        
        if isinstance(top_gcids[0], str):
            
            # image_components is read from datasets/<DS>/ (input); image_summaries are pipeline artifacts.
            self.images_dir = os.path.join(dataset_root(self._metadata_config, self.target_dataset), self.images_subpath, "dev")
            ds_meta = self._metadata_config["dataset_metadata"][self.target_dataset]
            has_page_summaries = ds_meta.get("has_page_summaries", False)
            if has_page_summaries:
                self.summaries_path = artifact_subpath(self._metadata_config, self.target_dataset, "image_summaries_dirname", "dev")

            # only one component
            text_prompt = ""

            for gcid in top_gcids:
                file_basename = gcid
                for filename in os.listdir(self.images_dir):
                    if filename.startswith(file_basename):
                        image_paths = [os.path.join(self.images_dir, filename)]
                        break

                if has_page_summaries:
                    for filename in os.listdir(self.summaries_path):
                        if filename.startswith(file_basename):
                            with open(os.path.join(self.summaries_path, filename), "r") as f:
                                summary_text = f.read()
                                text_prompt += f"/* \n\n Image summary: {summary_text}\n\n */\n"

            text_prompt += PAGE_PROMPT.format(
                question = question_text
            )
        
        else:
            serialized_parts = []
            image_paths = []
            next_image_idx = 1
            
            if isinstance(top_gcids[0], dict):
                for path_idx, path in enumerate(top_gcids[:1], start=1):
                    serialized_path = [f"/*", f"[Hierarchical Path {path_idx}]"]
                    for node_idx, node in enumerate(path.get("nodes", [])):
                        print(f"node_idx: {node_idx}")
                        print(f"node: {node}")
                        if len(node) == 2:
                            filename, component_id = node
                        elif len(node) == 3:
                            filename, component_id = node[0], node[2]
                        else:
                            raise ValueError(f"Unexpected path node format: {node!r}")

                        document_filename = self._resolve_graph_document_filename(str(filename))
                        component: Component = self.graph.get_component_by_gcid(
                            document_filename, str(component_id)
                        )
                        if isinstance(component, Image):
                            label = "Tile"
                        else:
                            label = "Atomic Fact"

                        if isinstance(component, Image):
                            original_component = self.graph.get_component_by_gcid(
                                document_filename, "i_1"
                            )
                            original_image_ref = ""
                            original_summary = None
                            if isinstance(original_component, Image):
                                original_image_path = self._resolve_prompt_image_path(
                                    original_component, "i_1"
                                )
                                if original_image_path is not None:
                                    original_image_ref = f"<Original image {next_image_idx}>"
                                    image_paths.append(original_image_path)
                                    next_image_idx += 1
                                original_summary = self._get_original_image_summary(
                                    document_filename
                                )

                            image_ref = ""
                            image_path = self._resolve_prompt_image_path(
                                component, str(component_id)
                            )
                            print(f"image_path: {image_path}")
                            if image_path is not None:
                                image_ref = f"<Tile image {next_image_idx}>"

                                image_paths.append(image_path)
                                next_image_idx += 1
                            caption = component.component_obj.get("caption", {}).get(
                                "text", ""
                            )
                            description = " ".join(
                                part
                                for part in (
                                    f"Original image: {original_image_ref}",
                                    image_ref,
                                    f"({caption})" if caption else "",
                                    (
                                        f"Original image summary: {original_summary}"
                                        if original_summary
                                        else ""
                                    ),
                                )
                                if part
                            ).strip()
                        elif isinstance(component, Text):
                            description = f'"{component.text}"'
                        else:
                            serialized_text, cmp_image_paths, updated_idx = (
                                component.serialize_into_prompt(next_image_idx)
                            )
                            description = serialized_text.strip()
                            image_paths.extend(cmp_image_paths)
                            next_image_idx = updated_idx

                        serialized_path.append(f"- {label}: {description}")

                    serialized_path.append("*/")
                    serialized_parts.append("\n".join(serialized_path))
            else:
                for gcid in top_gcids:
                    filename, component_id = gcid
                    document_filename = self._resolve_graph_document_filename(filename)
                    component: Component = self.graph.get_component_by_gcid(document_filename, component_id)

                    serialized_text, cmp_image_paths, updated_idx = component.serialize_into_prompt(next_image_idx)
                    serialized_parts.append(serialized_text)
                    image_paths.extend(cmp_image_paths)
                    next_image_idx = updated_idx

            # choose prompt
            if len(image_paths) > 0:
                instruction_prompt = INSTRUCTION_PROMPT_IMAGE
            else:
                instruction_prompt = INSTRUCTION_PROMPT_NONIMAGE

            print(f"serialized_parts: {serialized_parts}")

            # combine
            text_prompt = (
                INSTRUCTION_PROMPT_PATH
                + DEMONSTRATION_PROMPT_PATH
                + "\n"
                + "\n".join(serialized_parts)
                + "\n"
                + f"Question = {question_text}\nThe answer is: "
            )



        if self.text_only:
            image_paths = []
        
        return text_prompt, image_paths

    def _get_original_image_summary(self, document_filename: str) -> str | None:
        original = self.graph.get_component_by_gcid(document_filename, "i_1")
        if not isinstance(original, Image):
            return None
        filename = original.component_obj.get("filename")
        if not filename:
            return None
        return original._get_image_summary(str(filename))

    def _resolve_prompt_image_path(
        self, component: Image, component_id: str
    ) -> str | None:
        """Resolve original and tile images from their generation-time directories."""
        filename = component.component_obj.get("filename")
        if not filename:
            return None

        if component_id == "i_1":
            image_dir = Path(REPO_ROOT) / "datasets/InfoVQA/image_components/test"
        elif "_t" in component_id:
            image_dir = Path(REPO_ROOT) / "artifacts/InfoVQA/tiles_after_process"
        else:
            return component._get_image_abs_path(filename)

        basename = Path(str(filename)).name
        direct_path = image_dir / basename
        if direct_path.is_file():
            return str(direct_path)

        stem = Path(basename).stem
        matches = sorted(path for path in image_dir.rglob(f"{stem}.*") if path.is_file())
        if matches:
            return str(matches[0])

        raise FileNotFoundError(
            f"Image for component {component_id} was not found in {image_dir}: "
            f"{basename}"
        )

    def _resolve_graph_document_filename(self, filename: str) -> str:
        """Resolve retrieval filenames against the keys stored by the graph.

        Retrieval outputs may identify a document by its source filename
        (for example, ``36966.jpeg``), while parsed-document graphs commonly
        use the generated JSON filename (``36966.json``).
        """
        graph_filenames = self.graph.filename_to_document
        candidates = [filename]
        if not filename.endswith(".json"):
            candidates.append(f"{filename}.json")
            stem, _ = os.path.splitext(filename)
            candidates.append(f"{stem}.json")

        for candidate in candidates:
            if candidate in graph_filenames:
                return candidate

        raise ValueError(
            f"Document with filename {filename} not found in graph. "
            f"Tried: {', '.join(candidates)}."
        )










    # ----------------------------------------------------------------
    # 2.4) Worker process
    # ----------------------------------------------------------------
    def _worker_process(self, local_gpu_id: int, tasks_slice: List[tuple], tmp_dir: str):
        """
        Each worker sets CUDA_VISIBLE_DEVICES = local_gpu_id, 
        loads Qwen2.5, processes tasks, writes partial JSONL results.
        """
        # 1) pin GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_gpu_id)

        # 2) load Qwen on "cuda:0"
        model = Qwen2_5_VL(
            model_name_or_path = self.path_to_model,
            device             = "cuda:0",
            limit_mm_per_prompt = {"image": 8, "video": 0},
        )

        results = []
        start = time.time()
        for (qid, prompt, image_paths, serialization_time) in tqdm(tasks_slice, desc=f"[Worker GPU {local_gpu_id}]"):
            input_objects = [{
                "text": prompt,
                "images": image_paths
            }]
            t0 = time.time()
            outs = model.infer(
                objects=input_objects,
                batch_size = 1,
                temperature = 0.0,
                top_p = 0.9,
                repetition_penalty = 1.05,
                max_tokens = 32,
            )
            t1 = time.time()
            raw_answer = outs[0]
            predicted_answer = self._postprocess_answer(raw_answer)
            generation_time = (t1 - t0) * 1000

            # we do or do not have question text easily. We'll do:
            question_text = self.qid_to_question.get(qid, "???")

            item = {
                "qid": qid,
                "question": question_text,
                "time": {
                    "serialization_time": serialization_time,
                    "generation_time": generation_time
                },
                "predicted_answer": predicted_answer,
                "original_generation_result": raw_answer,
            }
            results.append(item)
        end = time.time()

        partial_path = os.path.join(tmp_dir, f"results_{local_gpu_id}.jsonl")
        with open(partial_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[Worker GPU {local_gpu_id}] done. {len(results)} QAs in {end - start:.1f}s => {partial_path}")


    def _postprocess_answer(self, raw_answer: str) -> str:
        
        if "herefore, the answer is:" in raw_answer:
            raw_answer = raw_answer.split("herefore, the answer is:")[1].strip()
            
        if "answer is:" in raw_answer:
            raw_answer = raw_answer.split("answer is:")[1].strip()
        
        pattern_col = r"f_answers\((.*?)\)"
        try:
            raw_answer = re.findall(pattern_col, raw_answer, re.S)[0].strip()
        except Exception:
            raw_answer = raw_answer
            
        try:
            raw_answer = json.loads(raw_answer)
        except json.JSONDecodeError:
            raw_answer = raw_answer
        
        return raw_answer

    # ----------------------------------------------------------------
    # 2.5) Merge partial JSONL
    # ----------------------------------------------------------------
    def _merge_shards(self, tmp_dir: str, final_jsonl_path: str, parts: int):
        with open(final_jsonl_path, "w", encoding="utf-8") as outf:
            for gpu_id in range(parts):
                partial_path = os.path.join(tmp_dir, f"results_{gpu_id}.jsonl")
                if not os.path.exists(partial_path):
                    continue
                with open(partial_path, "r", encoding="utf-8") as pf:
                    for line in pf:
                        outf.write(line)

        print(f"[Generator] Merged {parts} partial shards -> {final_jsonl_path}")





# --------------------------------------------------------------------
# 3) main
# --------------------------------------------------------------------
def main():
    cli_args = parse_arguments()
    config   = load_config(GENERATOR_DEFAULT_YAML_CONFIG_PATH, cli_args)

    generator = Generator(config)
    generator.run()

if __name__ == "__main__":
    main()