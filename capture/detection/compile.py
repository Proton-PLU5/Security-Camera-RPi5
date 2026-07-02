import torch
import ultralytics

if __name__ == "__main__":
    model = ultralytics.YOLO("./detection/yolo26s.pt")
    model.export(format="onnx")