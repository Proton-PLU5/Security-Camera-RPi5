import ultralytics

if __name__ == "__main__":
    model = ultralytics.YOLO("./capture/detection/yolo26s.pt")
    # Export to NCNN format with optimization options
    model.export(
        format="ncnn", 
        half=True,           # Use FP16 for smaller model size and faster inference
        device="cpu",        # CPU is sufficient for conversion
        imgsz=(960, 540),           # Input image size (default)
        simplify=True,        # Simplify the model architecture
    )
    print("NCNN model exported successfully!")