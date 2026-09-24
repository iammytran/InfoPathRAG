import json
from pathlib import Path


def parse_doc_id_from_unit(unit: dict) -> str:
    """Trích xuất ID tài liệu từ một retrieval unit (bỏ phần .json)."""
    # unit["nodes"] có dạng [["36966.json", "i_1"]]
    nodes = unit.get("nodes", [])
    if nodes and len(nodes[0]) > 0:
        raw_doc_name = nodes[0][0]  # "36966.json"
        return raw_doc_name.replace(".json", "")
    return ""


def get_ground_truth_doc_id(qid: str) -> str:
    """Trích xuất ID ảnh gốc từ qid.

    Ví dụ: '36966.jpeg-1' -> '36966'
    """
    # Tách theo dấu chấm hoặc gạch ngang để lấy id đầu tiên
    clean_qid = str(qid).split(".")[0]
    return clean_qid.split("-")[0]


def analyze_retrieval_misses(
    input_jsonl_path: str, output_txt_path: str, top_k: int = 9
):
    input_file = Path(input_jsonl_path)
    output_file = Path(output_txt_path)

    total_queries = 0
    missed_queries = 0

    with open(input_file, "r", encoding="utf-8") as f_in, open(
        output_file, "w", encoding="utf-8"
    ) as f_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            qid = data.get("qid", "")
            target_gt_id = get_ground_truth_doc_id(qid)

            # Lấy top-k retrieved units
            retrieved_units = data.get("retrieved_units", [])[:top_k]

            # Danh sách doc IDs dự đoán trong top-k
            predicted_doc_ids = [
                parse_doc_id_from_unit(unit) for unit in retrieved_units
            ]

            total_queries += 1

            # Kiểm tra xem target có nằm trong top-k không
            if target_gt_id not in predicted_doc_ids:
                missed_queries += 1

                # Ghi thông tin chi tiết vào file output
                f_out.write("=" * 80 + "\n")
                f_out.write(f"QID: {qid}\n")
                f_out.write(f"Ground Truth Document ID: {target_gt_id}\n")
                f_out.write(f"Top-{top_k} Retrieved Units:\n")

                for rank, (unit, doc_id) in enumerate(
                    zip(retrieved_units, predicted_doc_ids), 1
                ):
                    score = unit.get("score", 0.0)
                    nodes = unit.get("nodes", [])
                    f_out.write(
                        f"  Rank {rank:2d} | Doc ID: {doc_id:<10} | Score: {score:.5f} | Node: {nodes}\n"
                    )
                f_out.write("\n")

        # Ghi header tổng kết lên đầu hoặc in ra console
        print(f"Hoàn thành phân tích:")
        print(f"- Tổng số query: {total_queries}")
        print(
            f"- Số query trúng Top-{top_k}: {total_queries - missed_queries} (Recall@{top_k}: {(total_queries - missed_queries) / total_queries * 100:.2f}%)"
        )
        print(f"- Số query KHÔNG có trong Top-{top_k}: {missed_queries}")
        print(f"- Chi tiết các case lỗi đã được ghi vào: {output_file.resolve()}")


if __name__ == "__main__":
    # Thay đổi đường dẫn tương ứng với file của bạn
    INPUT_PATH = "/workspace/LILaC/algorithm_results/LILaC/InfoVQA/retrieval/mmembed_22082315/mmembed_22082315.jsonl"
    OUTPUT_PATH = "debug/missed_top9_queries.txt"

    analyze_retrieval_misses(INPUT_PATH, OUTPUT_PATH, top_k=9)