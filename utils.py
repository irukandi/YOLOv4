import torch
import pandas as pd


def calculate_box_coordinates(predictions, anchors, image_size=(416, 416)):
    anchors_amount = len(anchors)
    data_per_prediction = torch.div(predictions.size(1), anchors_amount)
    prediction_grid_height = predictions.size()[-2]
    prediction_grid_width = predictions.size()[-1]

    x_step_size = image_size[0] / prediction_grid_height
    y_step_size = image_size[1] / prediction_grid_width

    center_indexes = torch.cat([torch.arange(data_per_prediction * i, data_per_prediction * i + 2, dtype=torch.long)
                                for i in range(anchors_amount)])
    h_w_indexes = torch.cat([torch.arange(data_per_prediction * i + 2, data_per_prediction * i + 4, dtype=torch.long)
                             for i in range(anchors_amount)])
    x_coordinates = torch.arange(0, image_size[0], x_step_size)
    y_coordinates = torch.arange(0, image_size[1], y_step_size)
    x_coordinates_grid, y_coordinates_grid = torch.meshgrid(x_coordinates, y_coordinates)

    predictions[:, center_indexes] = torch.sigmoid(predictions[:, center_indexes])
    predictions[:, center_indexes[::2]] *= x_step_size
    predictions[:, center_indexes[1::2]] *= y_step_size
    predictions[:, center_indexes] = torch.round(predictions[:, center_indexes])
    predictions[:, center_indexes[::2]] += x_coordinates_grid.view(1, 1, prediction_grid_height, prediction_grid_width)
    predictions[:, center_indexes[1::2]] += y_coordinates_grid.view(1, 1, prediction_grid_height, prediction_grid_width)

    predictions[:, h_w_indexes] = torch.exp(predictions[:, h_w_indexes])
    predictions[:, h_w_indexes[::2]] *= torch.Tensor(anchors)[:, 0].view(1, 3, 1, 1)
    predictions[:, h_w_indexes[1::2]] *= torch.Tensor(anchors)[:, 1].view(1, 3, 1, 1)

    return predictions


def flatten_predictions(predictions, anchors_amount):
    batch_size = predictions.size()[0]
    attribute_size = torch.div(predictions.size()[1], anchors_amount, rounding_mode='floor')
    predictions = torch.flatten(predictions, start_dim=2)
    predictions = predictions.transpose(1, 2).contiguous()
    predictions = predictions.view(batch_size, -1, attribute_size)

    return predictions


def batch_indexing(predictions):
    # [[cx, cy, w, h, obj_score, classes..], ...] -> [img_idx, cx, cy, w, h, obj_score, classes...]
    image_index = torch.ones_like(predictions[:, :, 0]) * torch.arange(predictions.size()[0]).unsqueeze(1)
    predictions = torch.cat((image_index.unsqueeze(2), predictions), dim=2)
    predictions = torch.flatten(predictions, end_dim=1)

    return predictions


def background_removal(predictions, confidence):
    # [img_idx, cx, cy, w, h, obj_score, classes...] -> [img_idx, cx, cy, w, h, obj_score, class]
    if predictions.size()[0] > 0:
        class_mask = torch.argmax(predictions[:, 6:], dim=1)
        predictions[:, 5] *= predictions[torch.arange(predictions.size()[0]), class_mask + 6]
        predictions = torch.cat((predictions[:, :6], class_mask.unsqueeze(1)), dim=1)
        predictions = predictions[predictions[:, 5] > confidence]

    return predictions


def coordinate_transform(predictions, interference=True, image_size=(416, 416)):
    # [img_idx, cx, cy, w, h, obj_score, class] ->
    # [img_idx, cx, cy, top_leftx, top_lefty, bottom_rightx, bottom_righty, obj_score, class]
    if predictions.size()[0] > 0:
        box_corners = torch.ones(predictions.size()[0], 4)
        box_corners[:, 0] = predictions[:, 1] - torch.div(predictions[:, 4], 2, rounding_mode='floor')
        box_corners[:, 1] = predictions[:, 2] - torch.div(predictions[:, 3], 2, rounding_mode='floor')
        box_corners[:, 2] = predictions[:, 1] + torch.div(predictions[:, 4], 2, rounding_mode='floor')
        box_corners[:, 3] = predictions[:, 2] + torch.div(predictions[:, 3], 2, rounding_mode='floor')

        box_corners[box_corners < 0] = 0
        box_corners[:, 2][box_corners[:, 2] > image_size[0]] = image_size[0]
        box_corners[:, 3][box_corners[:, 3] > image_size[1]] = image_size[1]

        predictions = torch.cat((predictions[:, :3], box_corners, predictions[:, 5:]), dim=1)

    return predictions


def diou(box1, box2):
    # box_shape: [img_idx, cx, cy, top_leftx, top_lefty, bottom_rightx, bottom_righty, obj_score, class]
    surrounding_box = torch.ones(box2.size()[0], 4)
    surrounding_box[:, 0] = torch.minimum(box1[:, 3], box2[:, 3])
    surrounding_box[:, 1] = torch.minimum(box1[:, 4], box2[:, 4])
    surrounding_box[:, 2] = torch.maximum(box1[:, 5], box2[:, 5])
    surrounding_box[:, 3] = torch.maximum(box1[:, 6], box2[:, 6])

    center_dist = torch.cdist(box1[:, :2].unsqueeze(0), box2[:, :2]).flatten()
    corner_dist = torch.square(surrounding_box[:, 2] - surrounding_box[:, 0]) + \
                  torch.square(surrounding_box[:, 3] - surrounding_box[:, 1])

    relative_distance = torch.div(center_dist, corner_dist)

    intersection_box = torch.ones(box2.size()[0], 4)
    intersection_box[:, 0] = torch.maximum(box1[:, 3], box2[:, 3])
    intersection_box[:, 1] = torch.maximum(box1[:, 4], box2[:, 4])
    intersection_box[:, 2] = torch.minimum(box1[:, 5], box2[:, 5])
    intersection_box[:, 3] = torch.minimum(box1[:, 6], box2[:, 6])
    intersection_height = torch.maximum(torch.zeros(1), intersection_box[:, 2] - intersection_box[:, 0])
    intersection_width = torch.maximum(torch.zeros(1), intersection_box[:, 3] - intersection_box[:, 1])

    intersection = torch.mul(intersection_height, intersection_width)

    union = torch.mul(box1[:, 5] - box1[:, 3], box1[:, 6] - box1[:, 4]) + \
            torch.mul(box2[:, 5] - box2[:, 3], box2[:, 6] - box2[:, 4]) - intersection

    iou = torch.div(intersection, union)
    diou_value = iou - relative_distance

    return diou_value


def get_target_data(image_path, data_frame):
    image_name = image_path.rsplit("/", 1)[-1]
    bbox_columns = ['tl_x', 'tl_y', 'br_x', 'br_y']
    rows = data_frame[data_frame.iloc[:, 0] == image_name]

    return rows['class'].values, rows[bbox_columns].values


def read_labels(classes_file_path):
    class_labels = pd.read_csv(classes_file_path, header=None)
    class_labels = dict(class_labels.values)
    return class_labels


def labels_to_id(labels, labels_id):
    labels = [labels_id[label] for label in labels]
    return labels
