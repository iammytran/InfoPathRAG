from datasets import load_dataset
import pandas as pd
import json
import logging
import os
from pathlib import Path
import shutil
# import logger

val_dir = "infographicsVQA_val_v1.0_withQT copy.json"
visrag_test_dir = "openbmb/VisRAG-Ret-Test-InfoVQA"

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=os.path.join('debug', 'val_dataset.log'),
                    filemode='a'
                    )

logger = logging.getLogger(__name__)

def download_visrag_ret_test():
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="openbmb/VisRAG-Ret-Test-InfoVQA",
        repo_type="dataset",
        local_dir="./VisRAG-Ret-Test-InfoVQA",
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=8
    )
    print("Đã tải toàn bộ repo về thư mục ./VisRAG-Ret-Test-InfoVQA")

def get_queries_from_hf():
    # # Chỉ đọc duy nhất 1 cột 'query'
    # df = pd.read_parquet("VisRAG-Ret-Test-InfoVQA/queries/train-00000-of-00001.parquet", columns=["query"])

    # # Chuyển thành list
    # all_queries = df["query"].tolist()
    # return all_queries
    file_path = "VisRAG-Ret-Test-InfoVQA/queries/train-00000-of-00001.parquet"

    # Đọc chỉ 2 cột cần lấy (tiết kiệm bộ nhớ)
    df = pd.read_parquet(file_path, columns=["query-id", "query"])
    records = df.to_dict(orient="records")
    return records

def get_corpus_from_hf():
    # # Chỉ đọc duy nhất 1 cột 'query'
    # df = pd.read_parquet("VisRAG-Ret-Test-InfoVQA/queries/train-00000-of-00001.parquet", columns=["query"])

    # # Chuyển thành list
    # all_queries = df["query"].tolist()
    # return all_queries
    file_path = "VisRAG-Ret-Test-InfoVQA/corpus/train-00000-of-00001.parquet"

    # Đọc chỉ 2 cột cần lấy (tiết kiệm bộ nhớ)
    df = pd.read_parquet(file_path, columns=["corpus-id"])
    records = df.to_dict(orient="records")
    return records

def map_query_with_qa_val(queries_from_hf):
    qa_val_json = "datasets/InfoVQA/QAs_val.json"
    data = []
    output_data = []
    output_file = "datasets/QAs_val_718.json"

    # get qid and question from qa_val.json
    with open(qa_val_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # # Lọc chỉ lấy qid và question
    # filtered_list_qa_val = [{"qid": item["qid"], "question": item["question"]} for item in data]

    # tracing
    for query_hf in queries_from_hf:
        query_id = query_hf['query-id']
        query = query_hf['query']
        for query_in_qa_val in data:
            query_id_in_qa_val = query_in_qa_val['qid']
            query_str_in_qa_val = query_in_qa_val['question']
            if query.casefold() == query_str_in_qa_val.casefold():
                query_in_qa_val['qid'] = query_id
                output_data.append(query_in_qa_val)
                logging.info(f"For {query_id}: {query}, found in qa_val at {query_id_in_qa_val}")
            # if qu
            #     logging.info(f"For {query_id}: {query}, not found any match")

    with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
    return

def filter_images_from_corpus(corpus):
    output_path = Path("datasets/InfoVQA/image_components/test")
    target_ids = {str(item["corpus-id"]).split(".")[0] for item in corpus}

    image_dir = Path("datasets/InfoVQA/image_components/dev")
    copied_images = []

    # 2. Duyệt qua thư mục và copy nếu khớp ID
    for img_path in image_dir.rglob("*"):
        if img_path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
            if img_path.stem in target_ids:
                dest_path = output_path / img_path.name

                # Copy file sang folder đích
                shutil.copy2(img_path, dest_path)
                copied_images.append(dest_path)

    print(f"Đã copy thành công {len(copied_images)} ảnh sang {output_path}")
    return copied_images

def filter_test_crops_out(target_corpus):
    dev_crops_out_path = Path("artifacts/InfoVQA/crops_out/dev")
    test_crops_out_path = Path("artifacts/InfoVQA/crops_out/test")

    target_ids = {str(item["corpus-id"]).split(".")[0] for item in target_corpus}
    count = 0
    moved = []

    for folder in dev_crops_out_path.iterdir():
        if folder.name in target_ids:
            dst_dir = test_crops_out_path / folder.name
            shutil.copytree(folder, dst_dir, dirs_exist_ok=True)
            count += 1
            moved.append(folder.name)

    diff_a_b = list(set(target_ids) - set(moved))
    return count, diff_a_b

def filter_test_mineru_outputs(target_corpus):
    dev_mineru_output_path = Path("artifacts/InfoVQA/mineru_outputs_pipeline/dev")
    test_mineru_output_path = Path("artifacts/InfoVQA/mineru_outputs_pipeline/test")

    target_ids = {str(item["corpus-id"]).split(".")[0] for item in target_corpus}
    count = 0

    for folder in dev_mineru_output_path.iterdir():
        if folder.name in target_ids:
            dst_dir = test_mineru_output_path / folder.name
            shutil.copytree(folder, dst_dir, dirs_exist_ok=True)
            count += 1

    return count

def filter_test_image_summarization(target_corpus):
    dev_image_summaries_path = Path("artifacts/InfoVQA/image_summaries/dev")
    test_image_summaries_path = Path("artifacts/InfoVQA/image_summaries/test")

    target_ids = {str(item["corpus-id"]).split(".")[0] for item in target_corpus}

    copied_txt = []
    for txt_path in dev_image_summaries_path.iterdir():
        if txt_path.stem in target_ids:
            dest_path = test_image_summaries_path / txt_path.name

            # Copy file sang folder đích
            shutil.copy2(txt_path, dest_path)
            copied_txt.append(dest_path)

    return copied_txt

def main():
    return

if __name__ == "__main__":
    download_visrag_ret_test()
    # query_records = get_queries_from_hf()
    # map_query_with_qa_val(query_records)
    corpus = get_corpus_from_hf()
    test_images = filter_images_from_corpus(corpus)
    print(len(test_images))
    count_move_crops_out,diff = filter_test_crops_out(corpus)
    count_move_mineru_outputs = filter_test_mineru_outputs(corpus)
    print(f"moved {count_move_crops_out} crops_out folder")
    # print(f"moved {count_move_mineru_outputs} mineru_outputs folder")
    # print(diff)
    copied_summaries = filter_test_image_summarization(corpus)
    print(f"copied_summaries: {len(copied_summaries)}")