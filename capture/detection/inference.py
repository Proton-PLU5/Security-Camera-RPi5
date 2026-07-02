from ultralytics import YOLO

model = YOLO("./detection/yolo26s.onnx")
results = model("./detection/test.jpg")

for r in results:
    boxes = r.boxes.xyxy  # bounding boxes
    scores = r.boxes.conf  # confidence scores
    classes = r.boxes.cls  # class ids
    r.show()  # visualize