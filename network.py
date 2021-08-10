import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def predict_transform(predictions, input_shape=416, anchors=None):
    if anchors is None:
        anchors = [(10, 13), (16, 30), (33, 23)]
    batch_size = predictions.size(0)
    grid_size = predictions.size(2)
    stride = input_shape // grid_size
    bbox_attrs = predictions.size(1) // 3
    num_anchors = len(anchors)

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    predictions = predictions.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    predictions = predictions.transpose(1, 2).contiguous()
    predictions = predictions.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # Sigmoid the  centre_X, centre_Y. and object confidence
    predictions[:, :, 0] = torch.sigmoid(predictions[:, :, 0])
    predictions[:, :, 1] = torch.sigmoid(predictions[:, :, 1])
    predictions[:, :, 4] = torch.sigmoid(predictions[:, :, 4])

    # Add the center offsets
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    predictions[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    predictions[:, :, 2:4] = torch.exp(predictions[:, :, 2:4]) * anchors

    # Sigmoid the class scores
    predictions[:, :, 5: bbox_attrs] = torch.sigmoid((predictions[:, :, 5: 5 + bbox_attrs]))
    predictions[:, :, :4] *= stride

    return predictions


class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()  # x*tanh(softplus(x)), softplus(x) = 1/β∗log(1+exp(β∗x))


class Conv(nn.Module):
    def __init__(self, in_c, out_c, k, s, p):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)
        self.batch = nn.BatchNorm2d(out_c)
        self.act = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_c, prime=False):
        super().__init__()
        if prime:
            self.convs = nn.Sequential(
                Conv(in_c * 2, in_c, k=1, s=1, p=0),
                Conv(in_c, in_c * 2, k=3, s=1, p=1)
            )
        else:
            self.convs = nn.Sequential(
                Conv(in_c, in_c, k=1, s=1, p=0),
                Conv(in_c, in_c, k=3, s=1, p=1)
            )

    def forward(self, x):
        y = self.convs(x)
        h = torch.add(x, y)
        return h


class CSPBlock(nn.Module):
    def __init__(self, n, in_c, prime=False):
        super().__init__()
        if prime:
            self.conv1 = Conv(in_c, in_c * 2, k=1, s=1, p=0)
            self.conv2 = Conv(in_c * 2, in_c * 2, k=1, s=1, p=0)
            self.conv3 = Conv(in_c * 2, in_c * 2, k=1, s=1, p=0)
        else:
            self.conv1 = Conv(in_c, in_c, k=1, s=1, p=0)
            self.conv2 = Conv(in_c, in_c, k=1, s=1, p=0)
            self.conv3 = Conv(in_c * 2, in_c, k=1, s=1, p=0)
        self.residual = nn.Sequential(
            *[ResidualBlock(in_c, prime) for _ in range(n)]
        )

    def forward(self, x):
        c_2 = x.size(1) // 2
        x_2 = x[:, c_2:, :]
        x_2 = self.conv1(x_2)
        x_2 = self.residual(x_2)
        x_2 = self.conv2(x_2)
        x = self.conv3(x)  # So this changes any channels into half their size, making x the size of x_2. But why? WHY?
        h = torch.cat((x, x_2), 1)  # Made my peace with it don't even care
        return h  # I've been thinking maybe it's not supposed to do that. Like how the PANet downsampling doesn't
        # Maybe it's just a way of communication access to a prior layer
        # But THATS not in line with the post about yolov4_tiny claiming they have the same structure
        # fuck this


class SPP_YOLO(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)  # those might have to be adjusted to allow
        self.maxpool9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)  # multi-sized input
        self.maxpool13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)  # (e.g shape[2]/1 and 2 # and 3)

    def forward(self, x):
        x1 = self.maxpool5(x)
        x2 = self.maxpool9(x)
        x3 = self.maxpool13(x)
        x = torch.cat((x, x1, x2, x3), 1)
        return x


class PANet_blocks(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.convs = nn.Sequential(
            Conv(in_c=in_c, out_c=in_c // 2, k=1, s=1, p=0),
            Conv(in_c=in_c // 2, out_c=in_c, k=3, s=1, p=1),
            Conv(in_c=in_c, out_c=in_c // 2, k=1, s=1, p=0),
            Conv(in_c=in_c // 2, out_c=in_c, k=3, s=1, p=1),
            Conv(in_c=in_c, out_c=in_c // 2, k=1, s=1, p=0),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv = Conv(in_c=in_c, out_c=in_c // 2, k=1, s=1, p=0)

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=(2, 2))  # 100% sure there is a better way to do this
        return x


class Downsample(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv = Conv(in_c=in_c, out_c=in_c * 2, k=3, s=2,
                         p=1)  # stride=2, this is the PANet thing (mentioned in paper)

    def forward(self, x):
        x = self.conv(x)
        return x


class PANet(nn.Module):
    def __init__(self, classes=80):
        super().__init__()
        self.layer_1 = nn.Sequential(
            Conv(in_c=2048, out_c=512, k=1, s=1, p=0),
            Conv(in_c=512, out_c=1024, k=3, s=1, p=1),
            Conv(in_c=1024, out_c=512, k=1, s=1, p=0),
        )
        self.upsample_1 = Upsample(in_c=512)
        self.pan_conv2 = Conv(in_c=512, out_c=256, k=1, s=1, p=0)
        self.layer_2 = PANet_blocks(in_c=512)
        self.upsample_2 = Upsample(in_c=256)
        self.pan_conv1 = Conv(in_c=256, out_c=128, k=1, s=1, p=0)
        self.layer_3 = PANet_blocks(in_c=256)
        self.yolo_52 = nn.Sequential(
            Conv(in_c=128, out_c=256, k=3, s=1, p=1),
            nn.Conv2d(in_channels=256, out_channels=((5 + classes) * 3), kernel_size=1, stride=1, padding=0),  # 52x52x255
        )
        self.downsample_1 = Downsample(in_c=128)
        self.layer_4 = PANet_blocks(in_c=512)
        self.yolo_26 = nn.Sequential(
            Conv(in_c=256, out_c=512, k=3, s=1, p=1),
            nn.Conv2d(in_channels=512, out_channels=((5 + classes) * 3), kernel_size=1, stride=1, padding=0),  # 26x26x255
        )
        self.downsample_2 = Downsample(in_c=256)
        self.layer_5 = PANet_blocks(in_c=1024)
        self.yolo_13 = nn.Sequential(
            Conv(in_c=512, out_c=1024, k=3, s=1, p=1),
            nn.Conv2d(in_channels=1024, out_channels=((5 + classes) * 3), kernel_size=1, stride=1, padding=0),  # 13x13x255
        )

    def forward(self, x, pan_1, pan_2):
        x_13 = self.layer_1(x)

        x = self.upsample_1(x_13)
        pan_2 = self.pan_conv2(pan_2)  # not very verbose, but it's the 2nd connection
        x_26 = self.layer_2(torch.cat((x, pan_2), 1))

        x = self.upsample_2(x_26)
        pan_1 = self.pan_conv1(pan_1)  # it's the 1st connection respectively
        x_52 = self.layer_3(torch.cat((x, pan_1), 1))

        x = self.downsample_1(x_52)
        x_26 = self.layer_4(torch.cat((x, x_26), 1))

        x = self.downsample_2(x_26)
        x_13 = self.layer_5(torch.cat((x, x_13), 1))

        pred_52 = self.yolo_52(x_52)
        pred_26 = self.yolo_26(x_26)
        pred_13 = self.yolo_13(x_13)
        return pred_52, pred_26, pred_13


class DarkNet_53_Mish(nn.Module):
    def __init__(self):
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
            nn.Conv2d(in_channels=1024, out_channels=30, kernel_size=1, stride=1, padding=0),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.out(x)
        return torch.flatten(x)


class YOLOv4_Mish_416(nn.Module):
    def __init__(self, classes=80):
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
        self.PANet = PANet(classes=classes)  # contains YOLO detection

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        pan_1 = self.layer_3(x)
        pan_2 = self.layer_4(pan_1)
        x = self.layer_5(pan_2)
        x = self.layer_6(x)
        pred_52, pred_26, pred_13 = self.PANet(x, pan_1, pan_2)
        pred_52 = predict_transform(pred_52)
        pred_26 = predict_transform(pred_26)
        pred_13 = predict_transform(pred_13)
        predictions = torch.cat((pred_52, pred_26, pred_13), 1)
        return predictions


model = YOLOv4_Mish_416(classes=5)
x = torch.rand(1, 3, 416, 416)
predictions = model(x)
print(predictions.size())
