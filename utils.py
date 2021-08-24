import torch
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


def get_detections(predictions, confidence):
    ## select objects
    global detections
    object_mask = (predictions[:, :, 4] > confidence).float().unsqueeze(2)
    predictions *= object_mask

    ## Required for multiple classes per bounding box
    # class_mask = (predictions[:, :, 5:] > confidence).float()
    # predictions[:, :, 5:] *= class_mask

    batch_size = predictions.size(0)
    write = False
    for idx in range(batch_size):
        ## remove background predictions
        image_predictions = predictions[idx]
        non_zero_ind = torch.nonzero(image_predictions[:, 4])
        try:
            image_predictions = image_predictions[non_zero_ind[:, 0]]
        except:
            continue

        class_idx = torch.argmax(image_predictions[:, 5:], 1)
        # class_score, class_idx = torch.max(image_predictions[:, 5:], 1)
        class_idx = class_idx.unsqueeze(1)
        image_predictions = torch.cat((idx*torch.ones_like(class_idx), image_predictions[:, :5], class_idx), 1) # class_score.unsqueeze(1)

        if not write:
            detections = image_predictions
            write = True
        else:
            detections = torch.cat((detections, image_predictions), 0)
    return detections
