from utils import *

def nms(predictions):
    batch_size = torch.max(predictions[:, 0])
    for idx in range(batch_size):
        while False:
            break

def transform_predictions(predictions, input_size=416, anchors=[(10, 13), (16, 30), (33, 23)]):
    predictions = calculate_box_coordinates(predictions, anchors)
    predictions = flatten_predictions(predictions)

# important: YOLOv4 team counts from top left, BUT x is horizontal and y vertical
## REWRITE predict_transform, wrap utils in post_process function


