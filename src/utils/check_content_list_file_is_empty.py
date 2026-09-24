from pathlib import Path
import json

mineru_output_dir = Path("artifacts/InfoVQA/mineru_outputs/dev")

def main():
    mineru_outputs_dir = Path("artifacts/InfoVQA/mineru_outputs/dev")
    content_list_file_paths = sorted(mineru_outputs_dir.rglob("*_content_list.json"))
    content = ""

    empty_json = []

    for path in content_list_file_paths:
        with open(path, 'r') as file:
            content = json.load(file)

        # print(content)
    
        if len(content) == 0:
            empty_json.append(path)

    print(len(empty_json))
    print(empty_json)
    return

if __name__ == "__main__":
    main()