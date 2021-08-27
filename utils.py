import torch


def batch_indexing(predictions):
    image_index = torch.ones_like(predictions[:, :, 1]) * torch.arange(predictions.size()[0]).unsqueeze(1)
    predictions = torch.cat((image_index.unsqueeze(2), predictions), dim=2)
    predictions = torch.flatten(predictions, end_dim=1)
    return predictions

def background_removal(predictions, confidence):
    predictions = predictions[predictions[:, 5] > confidence]
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

def diou(box1, box2):
    surrounding_box = torch.ones(box2.size()[0], 4)
    surrounding_box[:, 0] = torch.min(torch.cat((box1[[4, 6]].unsqueeze(0), box2[:, [4, 6]])))
    surrounding_box[:, 1] = torch.min(torch.cat((box1[[5, 7]].unsqueeze(0), box2[:, [5, 7]])))
    surrounding_box[:, 2] = torch.max(torch.cat((box1[[4, 6]].unsqueeze(0), box2[:, [4, 6]])))
    surrounding_box[:, 3] = torch.max(torch.cat((box1[[5, 7]].unsqueeze(0), box2[:, [5, 7]])))

    center_dist = torch.cdist(box1[:2].unsqueeze(0), box2[:, :2]).flatten()
    corner_dist = torch.square(surrounding_box[:, 2] - surrounding_box[:, 0]) +\
                  torch.square(surrounding_box[:, 3] - surrounding_box[:, 1])

    relative_distance = torch.div(center_dist, corner_dist)

    intersection_box = torch.ones(box2.size()[0], 4)
    intersection_box[:, 0] = torch.maximum(box1[4], box2[:, 4])
    intersection_box[:, 1] = torch.maximum(box1[5], box2[:, 5])
    intersection_box[:, 2] = torch.minimum(box1[6], box2[:, 6])
    intersection_box[:, 3] = torch.minimum(box1[7], box2[:, 7])

    intersection = torch.mul(intersection_box[:, 2] - intersection_box[:, 0],
                             intersection_box[:, 3] - intersection_box[:, 1])

    union = torch.mul(box1[6] - box1[4], box1[7] - box1[5]) +\
            torch.mul(box2[:, 6] - box2[:, 4], box2[:, 7] - box2[:, 5]) - intersection

    iou = torch.div(intersection, union)
    diou = iou - relative_distance
    return diou