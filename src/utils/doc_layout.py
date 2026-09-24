import cv2
from doclayout_yolo import YOLOv10

# Load the pre-trained model
model = YOLOv10("models/DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt")

# Perform prediction
det_res = model.predict(
    "datasets/InfoVQA/image_components/dev/10002.jpeg",   # Image to predict
    imgsz=1024,        # Prediction image size
    conf=0.2,          # Confidence threshold
    device="cuda:0"    # Device to use (e.g., 'cuda:0' or 'cpu')
)

# Annotate and save the result
annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
cv2.imwrite("result.jpg", annotated_frame)