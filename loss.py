import torch
import torch.nn as nn

class CIoU_loss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, predictions, targets):
        pass