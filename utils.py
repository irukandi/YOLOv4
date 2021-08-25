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
    object_mask = (predictions[:, :, 4] > confidence).float().unsqueeze(2)
    predictions *= object_mask

    ## Required for multiple classes per bounding box
    # class_mask = (predictions[:, :, 5:] > confidence).float()
    # predictions[:, :, 5:] *= class_mask

    box_corners = torch.ones_like(predictions[:, :, :4])
    box_corners[:, :, 0] = predictions[:, :, 0] - torch.div(predictions[:, :, 2], 2, rounding_mode='floor')
    box_corners[:, :, 1] = predictions[:, :, 0] - torch.div(predictions[:, :, 3], 2, rounding_mode='floor')
    box_corners[:, :, 2] = predictions[:, :, 1] + torch.div(predictions[:, :, 2], 2, rounding_mode='floor')
    box_corners[:, :, 3] = predictions[:, :, 1] + torch.div(predictions[:, :, 3], 2, rounding_mode='floor')
    predictions[:, :, :4] = box_corners

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
        image_predictions = torch.cat((idx * torch.ones_like(class_idx), image_predictions[:, :5], class_idx),
                                      1)  # class_score.unsqueeze(1)
        image_predictions = NMS(image_predictions, nms_confidence=0.5)

        if not write:
            predictions_buffer = image_predictions
            write = True
        else:
            predictions_buffer = torch.cat((predictions_buffer, image_predictions), 0)
    return predictions_buffer


def NMS(image_predictions, nms_confidence):
    classes = torch.unique(image_predictions[:, -1])
    write = False
    for cls in classes:
        try:
            class_predictions = image_predictions[image_predictions[:, -1] == cls]
        except:
            continue

        class_predictions_idx = torch.sort(class_predictions[:, 4], descending=True)[1]
        class_predictions = class_predictions[class_predictions_idx]

        amount = class_predictions.size()[0]
        for idx in range(amount):
            #try:
            dious = DIoU(class_predictions[idx], class_predictions[idx+1:])
            #except ValueError:
            #    break
            #except IndexError:
            #    break

            class_predictions = class_predictions[idx+1:][dious > nms_confidence]

        if not write:
            class_predictions_buffer = class_predictions
            write = True
        else:
            class_predictions_buffer = torch.cat((class_predictions_buffer, class_predictions), 0)
    return class_predictions_buffer


def DIoU(box1, box2):
    # 1 - IoU +
    # can be performed for a list of boxes as box2

    box_1_center = torch.zeros(4)
    box_1_center[0] = box1[1] + torch.div((box1[3] - box1[1]), 2, rounding_mode='floor')
    box_1_center[1] = box1[2] + torch.div((box1[4] - box1[2]), 2, rounding_mode='floor')
    box_1_center[2] = box1[3] - box1[1]
    box_1_center[3] = box1[4] - box1[2]

    box_2_center = torch.ones_like(box2[:,1:5])
    box_2_center[:, 0] = box2[:, 1] + torch.div((box2[:, 3] - box2[:, 1]), 2, rounding_mode='floor')
    box_2_center[:, 1] = box2[:, 2] + torch.div((box2[:, 4] - box2[:, 2]), 2, rounding_mode='floor')
    box_2_center[:, 2] = box2[:, 3] - box2[:, 1]
    box_2_center[:, 3] = box2[:, 4] - box2[:, 2]

    surrounding_box = torch.zeros(4)
    surrounding_box[0] = torch.min(torch.tensor((box1[1], box1[3], box2[1], box2[3])))
    surrounding_box[1] = torch.min(torch.tensor((box1[2], box1[4], box2[2], box2[4])))
    surrounding_box[2] = torch.max(torch.tensor((box1[1], box1[3], box2[1], box2[3])))
    surrounding_box[3] = torch.max(torch.tensor((box1[2], box1[4], box2[2], box2[4])))

    diou = torch.div(torch.cdist(box_1_center[:2], box_2_center[:2]), torch.cdist(surrounding_box[:2], surrounding_box[2:]))
    pass

def iou(box1, box2):
    pass
