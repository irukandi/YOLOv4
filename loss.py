import torch
import torch.nn as nn
from utils import diou

class CIoU_loss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, predictions, targets):
        pass

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

        self.lambda_noobj = 10

    def forward(self, predictions, targets):
        # predictions = [cx, cy, top_leftx, top_lefty, bottom_rightx, bottom_righty, obj_score, class_scores...]

        # ciou_loss
        # 1 - iou + diou + alpha*v
        ciou_loss = 1 - diou(predictions, targets)

        v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        with torch.no_grad():
            alpha = v / ((1 + eps) - iou + v)
        # obj_loss
        # - object *
        # noobj_loss
        # class_loss
