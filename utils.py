import torch
import numpy as np


def predict_transform(predictions, input_shape=416, anchors=None):
    # have to check if all this is legit
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
    grid_len = torch.arange(grid_size)
    a, b = torch.meshgrid(grid_len, grid_len)

    x_offset = b.reshape(-1, 1)
    y_offset = a.reshape(-1, 1)

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


def batch_indexing(predictions):
    image_index = torch.ones_like(predictions[:, :, 1]) * torch.arange(predictions.size()[0]).unsqueeze(1)
    predictions = torch.cat((image_index.unsqueeze(2), predictions), dim=2)
    predictions = torch.flatten(predictions, end_dim=1)
    return predictions

def background_removal(predictions, confidence):
    predictions = predictions[predictions[:, 5] > confidence]
    print(predictions)
    if predictions.size()[0] > 0:
        class_mask = torch.argmax(predictions[:, 5:], dim=1)
        predictions = torch.cat((predictions[:, :6], class_mask.unsqueeze(1)), dim=1)
    return predictions

def coordinate_transform(predictions):
    if predictions.size()[0] > 0:
        box_corners = torch.ones(predictions.size()[0], 4)
        box_corners[:, 0] = predictions[:, 1] - torch.div(predictions[:, 3], 2, rounding_mode='floor')
        box_corners[:, 1] = predictions[:, 2] - torch.div(predictions[:, 4], 2, rounding_mode='floor')
        box_corners[:, 2] = predictions[:, 1] + torch.div(predictions[:, 3], 2, rounding_mode='floor')
        box_corners[:, 3] = predictions[:, 2] + torch.div(predictions[:, 4], 2, rounding_mode='floor')

        predictions = torch.cat((predictions[:, :5], box_corners, predictions[:, 5:]), dim=1)
    return predictions

def iou(box1, box2):
    pass

def diou(box1, box2):
    pass