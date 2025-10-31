from ultralytics import YOLO
import torch

model_name = "yolo11n.pt"
MODEL = YOLO(model_name)


def setModelName(name: str):
    global MODEL
    MODEL = YOLO(name)

def getInference(img : str):
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