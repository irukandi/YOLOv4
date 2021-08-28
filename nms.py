from utils import *


def nms(predictions):
    batch_size = torch.max(predictions[:, 0])
    for idx in range(batch_size):
        while False:
            break

# important: YOLOv4 team counts from top left, BUT x is horizontal and y vertical
## REWRITE predict_transform, wrap utils in post_process function


