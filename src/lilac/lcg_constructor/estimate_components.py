import cv2
import numpy as np
from pathlib import Path
import json


def _non_max_suppression_boxes(boxes: list[tuple[int, int, int, int]], iou_threshold: float = 0.4) -> list[tuple[int, int, int, int]]:
    """Loại bỏ các bounding box bị lồng hoặc đè lên nhau quá nhiều."""
    if not boxes:
        return []

    boxes_arr = np.array(boxes, dtype=float)
    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = x1 + boxes_arr[:, 2]
    y2 = y1 + boxes_arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = areas.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(boxes[i])

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        smaller_area = np.minimum(areas[i], areas[order[1:]])
        overlap = inter / np.maximum(smaller_area, 1e-6)

        inds = np.where(overlap <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def estimate_components_opencv(
    image_path: Path | str,
    min_area_ratio: float = 0.0001,
    debug_dir: Path | None = None,
) -> int:
    """Ước tính số lượng components trong infographic."""
    img = cv2.imread(str(image_path))
    if img is None:
        return 4

    h, w = img.shape[:2]
    total_area = h * w

    # 1. Grayscale & làm mờ nhẹ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Phát hiện cạnh bằng Canny
    edges = cv2.Canny(blurred, threshold1=30, threshold2=120)

    # 3. Khử các đường kẻ thẳng của bảng / khung viền lớn
    line_min_w = max(30, int(w * 0.04))
    line_min_h = max(30, int(h * 0.04))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_w, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_min_h))

    lines_h = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
    lines_v = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
    grid_lines = cv2.add(lines_h, lines_v)

    content_edges = cv2.subtract(edges, grid_lines)

    # 4. Gom các dòng chữ liền kề nhau thành nguyên một khối văn bản
    # Tăng chiều cao kernel lên 15 để nối các dòng text theo phương dọc
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 15))
    dilated = cv2.dilate(content_edges, kernel, iterations=1)

    # 5. Tìm contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_area = total_area * min_area_ratio
    raw_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)

        # Lọc bỏ các khối viền bao chiếm gần hết ảnh
        if bh > 0.85 * h or bw > 0.90 * w:
            continue

        # Lọc bỏ các vệt viền mảnh rác dính sát 2 mép lề trái / phải
        is_left_edge_artifact = (bx <= 10 and bw < 35)
        is_right_edge_artifact = (bx + bw >= w - 10 and bw < 35)
        if bw < 12 or is_left_edge_artifact or is_right_edge_artifact:
            continue

        raw_boxes.append((bx, by, bw, bh))

    # 6. Lọc chồng lấn NMS
    final_boxes = _non_max_suppression_boxes(raw_boxes, iou_threshold=0.6)

    # 7. Lưu ảnh debug
    if debug_dir is not None:
        debug_img = img.copy()
        for bx, by, bw, bh in final_boxes:
            cv2.rectangle(debug_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"debug_{Path(image_path).stem}.png"), debug_img)

    return max(2, len(final_boxes))


def main():
    image_folder = Path("datasets/InfoVQA/image_components/test")
    debug_dir = Path("debug/contours")
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    component_count = {}

    for path in sorted(image_folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in valid_extensions:
            continue

        estimated_components = estimate_components_opencv(path, debug_dir=debug_dir)
        component_count[str(path)] = estimated_components

    output_json = Path("debug/estimated_components.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(component_count, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()