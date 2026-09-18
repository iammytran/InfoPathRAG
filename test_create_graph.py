import pickle
from src.lilac.basic_class.graph import Graph, Subgraph, top_level_gcid_by_low_level_gcid

def initiate_graph():
    _parsed_documents_dir = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/parsed_documents/dev"
    _images_dir = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/image_components/dev"
    _subimages_dir = "/Users/mytnguyen/Documents/LILaC/artifacts/InfoVQA/image_components_sub/dev"
    _summaries_dir = "/Users/mytnguyen/Documents/LILaC/artifacts/InfoVQA/image_summaries/dev"
    _graph_path = "/Users/mytnguyen/Documents/LILaC/artifacts/InfoVQA/components/graph.pickle"
    graph = Graph(
        multimodal_documents_directory = _parsed_documents_dir,
        images_directory    = _images_dir,
        subimages_directory = _subimages_dir,
        summaries_directory = _summaries_dir
    )
    graph.parse_documents()
    with open(_graph_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"[Retriever] Graph saved to {_graph_path}")

def main():
    return

# print("manifest:", tile_manifest)
# print("facts directory:", facts_directory)
# print("exists:", os.path.exists(facts_directory))

# for filename, edges in self.graph.intra_document_edges.items():
#     print(f"\nDocument: {filename}")
#     print("Edges:", edges)

#     for component_id, children in edges.items():
#         print(f"  {component_id} -> {children}")
file = "/workspace/LILaC/artifacts/InfoVQA/facts_each_tile/raw/10022_merged_i_1_t0002_i_1_t0003.json"
import os
import json

def check_empty_json_files(folder_path):
    print(f"Đang kiểm tra các file JSON trong thư mục: {folder_path}\n")
    
    empty_files_count = 0
    
    # Duyệt qua tất cả các file trong thư mục
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Kiểm tra nếu dữ liệu là một list và độ dài bằng 0
                    if isinstance(data, list) and len(data) == 0:
                        print(f"[RỖNG] {filename}")
                        empty_files_count += 1
                    # Hoặc nếu là một list chứa các dict mà không có fact nào (tùy cấu trúc thực tế)
            except json.JSONDecodeError:
                print(f"[LỖI ĐỌC FILE] {filename} (File JSON không hợp lệ)")
            except Exception as e:
                print(f"[LỖI KHÁC] {filename}: {e}")
                
    print(f"\n--- Hoàn tất ---")
    print(f"Tìm thấy tổng cộng {empty_files_count} file có danh sách rỗng.")

# Đường dẫn thư mục của bạn

import re
import json
from pathlib import Path

import re
import json

def _repair_json_text(raw: str) -> str:
    """Recover common Qwen JSON mistakes, including missing commas between items and truncation."""
    raw = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()

    start = next((index for index, char in enumerate(raw) if char in "[{"), None)
    if start is None:
        return raw
    raw = raw[start:]

    # 1. Nếu chuỗi kết thúc lửng lơ ở đuôi, cắt bỏ object cuối cùng bị lỗi
    if not raw.endswith("]") and not raw.endswith("}"):
        last_brace = raw.rfind("}")
        if last_brace != -1:
            raw = raw[:last_brace + 1]

    # 2. Tự động sửa lỗi thiếu dấu phẩy giữa các object (ví dụ: "} {" thành "} , {")
    # Biểu thức chính quy này tìm dấu đóng ngoặc nhọn, khoảng trắng/xuống dòng và dấu mở ngoặc nhọn tiếp theo mà thiếu dấu phẩy
    raw = re.sub(r'}\s*({)', r'}, \1', raw)

    # 3. Theo dõi stack ngoặc để đóng các ngoặc còn thiếu ở cuối
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in raw:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            stack.append("]" if char == "[" else "}")
        elif char in "]}":
            if stack and char == stack[-1]:
                stack.pop()

    raw = re.sub(r",\s*$", "", raw)
    return raw + "".join(reversed(stack))


def process_and_save_json_files(input_folder: str, output_folder: str) -> None:
    """Đọc file raw, trích xuất facts và lưu theo format {"facts": [...]} ra thư mục output."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"Thư mục nguồn không tồn tại: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    processed_count = 0

    for file_path in sorted(input_path.rglob("*.json")):
        # 1. Đọc nội dung file gốc
        original_text = file_path.read_text(encoding="utf-8")
        
        # 2. Trích xuất ra danh sách các dict chứa ocr và fact
        extracted_items = extract_facts_directly(original_text)
        
        # 3. Chỉ lấy phần "fact" để tạo danh sách (list) các câu fact
        facts_list = [item["fact"] for item in extracted_items]
        
        # 4. Đóng gói lại thành cấu trúc {"facts": [...]}
        final_data = {
            "facts": facts_list
        }
        
        # 5. Format thành JSON chuẩn
        cleaned_text = json.dumps(final_data, ensure_ascii=False, indent=4)
        
        # 6. Xác định đường dẫn và ghi file vào thư mục output
        relative_path = file_path.relative_to(input_path)
        destination_file = output_path / relative_path
        
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        destination_file.write_text(cleaned_text + "\n", encoding="utf-8")
        processed_count += 1
        print(f"[SAVED] {destination_file}")

    print(f"\n--- Hoàn tất ---")
    print(f"Đã xử lý và lưu thành công {processed_count} file vào: {output_path}")

import re

def extract_facts_directly(raw: str) -> list[dict]:
    """
    Bóc tách trực tiếp các cặp ocr và fact bằng regex, 
    bỏ qua hoàn toàn việc phải kiểm tra cú pháp JSON hợp lệ.
    """
    # Xóa markdown nếu có
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()

    # Regex tìm pattern dạng "ocr": "...", "fact": "..."
    # re.DOTALL giúp bắt được các text kéo dài qua nhiều dòng
    pattern = r'"ocr"\s*:\s*"(.*?)",\s*"fact"\s*:\s*"(.*?)"'
    
    matches = re.findall(pattern, raw, flags=re.DOTALL)
    
    results = []
    for ocr_val, fact_val in matches:
        # Xử lý lại các ký tự escape cơ bản nếu cần (ví dụ \" thành ")
        clean_ocr = ocr_val.replace(r'\"', '"')
        clean_fact = fact_val.replace(r'\"', '"')
        
        results.append({
            "ocr": clean_ocr,
            "fact": clean_fact
        })
        
    return results


def update_manifest_facts(
    manifest_path: str,
    facts_folder: str | None = None,
) -> None:
    """Update each manifest tile's ``facts`` from its processed JSON file."""
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_file}")

    facts_path = Path(facts_folder) if facts_folder else manifest_file.parent
    if not facts_path.is_dir():
        raise NotADirectoryError(f"Facts folder does not exist: {facts_path}")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("infographics"), list):
        raise ValueError("Manifest must contain an 'infographics' list.")

    updated_tiles = 0
    missing_files = []

    for infographic in manifest["infographics"]:
        for tile in infographic.get("tiles", []):
            tile_filename = tile.get("filename")
            if not tile_filename:
                raise ValueError("Every tile must contain a 'filename'.")

            fact_file = facts_path / f"{Path(tile_filename).stem}.json"
            if not fact_file.is_file():
                missing_files.append(str(fact_file))
                continue

            fact_data = json.loads(fact_file.read_text(encoding="utf-8"))
            if isinstance(fact_data, dict):
                facts = fact_data.get("facts", [])
            elif isinstance(fact_data, list):
                facts = fact_data
            else:
                raise ValueError(
                    f"Expected a list or object with 'facts' in {fact_file}"
                )

            if not isinstance(facts, list):
                raise ValueError(f"'facts' must be a list in {fact_file}")

            tile["facts"] = facts
            updated_tiles += 1

    if missing_files:
        preview = "\n".join(f"  - {path}" for path in missing_files[:10])
        suffix = (
            f"\n  ... and {len(missing_files) - 10} more"
            if len(missing_files) > 10
            else ""
        )
        raise FileNotFoundError(
            f"Missing fact files for {len(missing_files)} tile(s):\n"
            f"{preview}{suffix}"
        )

    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Updated facts for {updated_tiles} tile(s): {manifest_file}")


# # --- Test thử với chuỗi lỗi của bạn ---
# raw_text = """
# [
#     {"ocr":"Sore Throat","fact":"Sore throat is one of the most common reported symptoms of Coronavirus."},
#     {"ocr":"Headache","fact":"Headache is another frequently reported symptom among those affected by the virus."},
#     ...
# ]
# """

# facts_list = extract_facts_directly(raw_text)
# print(facts_list)


if __name__ == "__main__":
    raw_folder = "/workspace/LILaC/artifacts/InfoVQA/facts_each_tile/raw"
    cleaned_folder = "/workspace/LILaC/artifacts/InfoVQA/facts_each_tile"
    manifest_file = f"{cleaned_folder}/manifest.json"

    # process_and_save_json_files(raw_folder, cleaned_folder)
    update_manifest_facts(manifest_file, cleaned_folder)
    
    # initiate_graph()
    