from network_parts import *
from utils import * ## can be removed once post_processing is a single functions
from nms import nms
import time

class DarkNet_53_Mish(nn.Module):
    def __init__(self, classes=30):
        super().__init__()
        self.layer_1 = nn.Sequential(
            Conv(in_c=3, out_c=32, k=3, s=1, p=1),
            Conv(in_c=32, out_c=64, k=3, s=2, p=1),
            CSPBlock(n=1, in_c=32, prime=True),
        )
        self.layer_2 = nn.Sequential(
            Conv(in_c=128, out_c=64, k=1, s=1, p=0),
            Conv(in_c=64, out_c=128, k=3, s=2, p=1),
            CSPBlock(n=2, in_c=64),
        )
        self.layer_3 = nn.Sequential(
            Conv(in_c=128, out_c=128, k=1, s=1, p=0),
            Conv(in_c=128, out_c=256, k=3, s=2, p=1),
            CSPBlock(n=8, in_c=128),
        )
        self.layer_4 = nn.Sequential(
            Conv(in_c=256, out_c=256, k=1, s=1, p=0),
            Conv(in_c=256, out_c=512, k=3, s=2, p=1),
            CSPBlock(n=8, in_c=256),
        )
        self.layer_5 = nn.Sequential(
            Conv(in_c=512, out_c=512, k=1, s=1, p=0),
            Conv(in_c=512, out_c=1024, k=3, s=2, p=1),
            CSPBlock(n=4, in_c=512),
        )
        self.out = nn.Sequential(
            Conv(in_c=1024, out_c=1024, k=1, s=1, p=0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=1024, out_channels=classes, kernel_size=1, stride=1, padding=0),
            nn.Softmax(),
        )
        print(f'Initiated DarkNet_53_Mish with {classes} classes\n')

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.out(x)

        return torch.flatten(x)


class YOLOv4_Mish_416(nn.Module):
    def __init__(self, classes=80, sam_enabled=False, verbose=True):
        super().__init__()
        self.layer_1 = nn.Sequential(
            Conv(in_c=3, out_c=32, k=3, s=1, p=1),
            Conv(in_c=32, out_c=64, k=3, s=2, p=1),
            CSPBlock(n=1, in_c=32, prime=True),
            Conv(in_c=128, out_c=64, k=1, s=1, p=0),
        )
        self.layer_2 = nn.Sequential(
            Conv(in_c=64, out_c=128, k=3, s=2, p=1),
            CSPBlock(n=2, in_c=64),
            Conv(in_c=128, out_c=128, k=1, s=1, p=0),
        )
        self.layer_3 = nn.Sequential(
            Conv(in_c=128, out_c=256, k=3, s=2, p=1),
            CSPBlock(n=8, in_c=128),
            Conv(in_c=256, out_c=256, k=1, s=1, p=0),
        )
        self.layer_4 = nn.Sequential(
            Conv(in_c=256, out_c=512, k=3, s=2, p=1),
            CSPBlock(n=8, in_c=256),
            Conv(in_c=512, out_c=512, k=1, s=1, p=0),
        )
        self.layer_5 = nn.Sequential(
            Conv(in_c=512, out_c=1024, k=3, s=2, p=1),
            CSPBlock(n=4, in_c=512),
            Conv(in_c=1024, out_c=1024, k=1, s=1, p=0),
        )
        self.layer_6 = nn.Sequential(
            Conv(in_c=1024, out_c=512, k=1, s=1, p=0),  # Dense Block ?
            Conv(in_c=512, out_c=1024, k=1, s=1, p=0),  # here too
            Conv(in_c=1024, out_c=512, k=1, s=1, p=0),  # still dense
            SPP_YOLO()
        )
        self.PANet = PANet(classes=classes, sam_enabled=sam_enabled)  # contains YOLO detection
        self.verbose = verbose
        if verbose:
            print(f'Initiated YOLOv4_Mish_416 with {classes} classes')
            if sam_enabled:
                print('\tSpatial-Attention-Module has been enabled.')

    def forward(self, x, train=False):
        x = self.layer_1(x)
        x = self.layer_2(x)
        pan_1 = self.layer_3(x)

        pan_2 = self.layer_4(pan_1)
        x = self.layer_5(pan_2)
        x = self.layer_6(x)

        pred_52, pred_26, pred_13 = self.PANet(x, pan_1, pan_2)

        pred_52 = transform_predictions(pred_52)
        pred_26 = transform_predictions(pred_26)
        pred_13 = transform_predictions(pred_13)

        predictions = torch.cat((pred_52, pred_26, pred_13), 1)
        predictions[:, :, 4] = torch.sigmoid(predictions[:, :, 4])
        predictions = post_processing(predictions, confidence=0.5, verbose=self.verbose)

        if not train:
            predictions = nms(predictions, verbose=self.verbose)

        return predictions


model = YOLOv4_Mish_416(classes=5, sam_enabled=True)
x = torch.rand(3, 3, 416, 416)
predictions = model(x)
