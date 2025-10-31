from ultralytics import YOLO
import torch
from PIL import Image
import io

model_name = "yolo11n.pt"
MODEL = YOLO(model_name)


def setModelName(name: str):
    global MODEL
    MODEL = YOLO(name)

def getInference(img):
    """
    Run inference on an image.
    
    Args:
        img: Can be a file path (str), PIL Image, numpy array, or image bytes
    """
    # If img is bytes, convert to PIL Image
    if isinstance(img, bytes):
        img = Image.open(io.BytesIO(img))
    
    results = MODEL(img)
    return results[0]

def filterBoundingBoxes(result, whitelist : list):
    filtered_boxes = []
    for i in range(len(result.boxes)):
        class_idx = int(result.boxes.cls[i])
        if result.names[class_idx] in whitelist:
            filtered_boxes.append(result.boxes.xywh[i])
        elif len(whitelist) == 0:
            filtered_boxes.append(result.boxes.xywh[i])
    return filtered_boxes

def filterResult(result, whitelist: list):
    """
    Filter a YOLO result object to only include classes in the whitelist.
    Returns a new result object with filtered detections.
    
    Args:
        result: YOLO result object
        whitelist: List of class names to include (empty list = include all)
    
    Returns:
        Filtered YOLO result object
    """
    if len(whitelist) == 0 or result.boxes is None or len(result.boxes) == 0:
        return result
    
    # Find indices of boxes that match the whitelist
    keep_indices = []
    for i in range(len(result.boxes)):
        class_idx = int(result.boxes.cls[i])
        if result.names[class_idx] in whitelist:
            keep_indices.append(i)
    
    # If no boxes match, return result with empty boxes
    if len(keep_indices) == 0:
        result.boxes = None
        return result
    
    # Filter the boxes
    result.boxes.data = result.boxes.data[keep_indices]
    
    return result

# Function to get the largest bounding box from the results
# Used in future to pick object to track.
def getLargestBoundingBox(result) -> torch.Tensor:
    largest_box : torch.Tensor = torch.tensor(0)
    largest_area = 0

    for i in range(0, result.boxes.xywh.size()[0]):
        width = result.boxes.xywh[i,2]
        height = result.boxes.xywh[i,3]

        if (width*height > largest_area):
            largest_area = width*height
            largest_box = result.boxes.xywh[i,:]
    
    return largest_box