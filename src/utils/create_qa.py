import json
import re

def convert_infovqa_dataset(input_path: str, output_path: str):
    # 1. Đọc file JSON gốc
    with open(input_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)

    # 2. Lấy danh sách câu hỏi từ trường "data"
    items_list = raw_json.get("data", [])

    # 3. Chuyển đổi từng phần tử sang schema mong muốn
    output_list = []
    for item in items_list:
        formatted_item = {
            "qid": item.get("questionId"),
            "question": item.get("question", ""),
            "answers": [
                {"answer": str(ans)} for ans in item.get("answers", [])
            ],
            "evidences": [
                {"gold_image": item["image_local_name"]}
            ] if "image_local_name" in item else [],
            "data_split": item.get("data_split", raw_json.get("dataset_split", ""))
        }
        output_list.append(formatted_item)

    # 4. Ghi ra file output dưới dạng 1 JSON List
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2)

    print(f"Đã chuyển đổi thành công {len(output_list)} câu hỏi vào '{output_path}'.")

if __name__ == "__main__":
    input_path = "/Users/mytnguyen/Documents/LILaC/infographicsVQA_val_v1.0_withQT copy.json"
    output_path = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/QAs_val.json"
    convert_infovqa_dataset(input_path, output_path)