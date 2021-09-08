import torch
from utils import diou


def nms(predictions, nms_threshold=0.5, verbose=True):
    # [img_idx, cx, cy, top_leftx, top_lefty, bottom_rightx, bottom_righty, obj_score, class]
    if verbose:
        print('\tRunning Non-Maximum-Suppression on all detections...')
    final_detection_buffer = None
    image_indexes = torch.unique(predictions[:, 0])
    for image_idx in image_indexes:
        image_data = predictions[predictions[:, 0] == image_idx]
        _, sort_idx = torch.sort(image_data, dim=0, descending=True)
        image_data = image_data[sort_idx[:, -2]]
        class_indexes = torch.unique(image_data[:, -1])
        for class_idx in class_indexes:
            class_data = image_data[image_data[:, -1] == class_idx]
            idx = 0
            while idx < class_data.size()[0]:
                dious = diou(class_data[idx], class_data[idx + 1:])
                class_data[idx + 1:, -2][dious > nms_threshold] = 0
                class_data = class_data[torch.nonzero(class_data[:, -2]).flatten()]
                idx += 1
            if final_detection_buffer is None:
                final_detection_buffer = class_data
            else:
                final_detection_buffer = torch.cat((final_detection_buffer, class_data), dim=0)
    if verbose:
        print(f'\tNon-Maximum-Suppression successful, {final_detection_buffer.size()[0]} detections kept.')

    return final_detection_buffer

# important: YOLOv4 team counts from top left, BUT x is horizontal and y vertical
