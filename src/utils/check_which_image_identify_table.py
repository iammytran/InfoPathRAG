from pathlib import Path
import json

mineru_path = Path("artifacts/InfoVQA/mineru_outputs/dev")
has_table_output_file = "debug/has_table.txt"
table_ref_image_output_file = "debug/table_ref_img.txt"
def main():
    identify_table = set()
    table_ref_image = set()
    for file_path in mineru_path.glob("**/vlm/*_content_list.json"):
        # 1. Lấy Document ID trực tiếp từ thư mục cha của thư mục 'vlm'
        # file_path.parts sẽ là ('dev', '1002', 'vlm', '1002_content_list.json')
        # doc_id = file_path.parts[1] 
        doc_id = file_path.name.split("_")[0]
        
        # 2. Đọc nội dung file JSON
        with open(file_path, "r", encoding="utf-8") as f:
            parsed_document = json.load(f)
        
        for block in parsed_document:
            # if block['type'] == "table" and doc_id == "10090":
            #     table_content = block.get('table_body', '')
            #     print(f"{table_content} --> {'<img' in table_content.lower()}")
                # print('<img' in table_content.lower())
            # if doc_id == "10090":
            #     print("10090")
            #     print(block.get('table_body', ''))
                # print(block.get('table_body', '').contains)
            if block['type'] == "table":
                identify_table.add(doc_id)
                # break
            if block['type'] == "table" and 'img' in block.get('table_body', '').lower():
                table_ref_image.add(doc_id) 

    identify_table= sorted(identify_table)
    table_ref_image = sorted(table_ref_image)
            
    print(f"có {len(identify_table)} có dữ liệu bảng")
    print(f"có {len(table_ref_image)} có dữ liệu bảng mà có ref img")
    
    with open(has_table_output_file, 'w') as file:
        for image in identify_table:
            file.write(f"{image}\n")
        print(f"Finished writing has_table_output_file that has which image has table.")

    with open(table_ref_image_output_file, 'w') as file:
        for image in table_ref_image:
            file.write(f"{image}\n")
        print(f"Finished writing table_ref_image!")

    
    # with open(mineru_path, 'r') as file:

    return 

if __name__== "__main__":
    main()
