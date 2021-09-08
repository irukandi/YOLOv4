import cv2
import torch
import torchvision.transforms as transforms
from fastai.vision.all import get_image_files
import pandas as pd
from torch.utils.data import Dataset
import random
from utils import get_target_data, read_labels, labels_to_id


class Data(Dataset):
    def __init__(self, folder_path, images_path, classes_file_path, annotations_file_path=None, header=None, transform=None):
        super().__init__()
        if header is None:
            header = ['img_path', 'tl_x', 'tl_y', 'br_x', 'br_y', 'class']
        if transform is None:
            transform = transforms.ToTensor()
        if annotations_file_path is None:
            self.annotations = None
        else:
            self.annotations = pd.read_csv(folder_path + annotations_file_path, names=header)
        self.images = get_image_files(folder_path + images_path)
        self.transform = transform
        self.classes = read_labels(folder_path + classes_file_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = str(self.images[index])
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # image = torch.as_tensor(image, dtype=torch.int32)#self.transform(image)
        if self.annotations is not None:
            target = {}
            labels, boxes = get_target_data(image_path, self.annotations)
            labels = labels_to_id(labels, self.classes)
            iscrowd = torch.zeros((boxes.shape[0],))
            image_id = torch.tensor([index])
            labels = torch.as_tensor(labels, dtype=torch.float32)
            boxes = torch.as_tensor(boxes, dtype=torch.int32)
            target['boxes'] = boxes
            target['labels'] = labels
            target['iscrowd'] = iscrowd
            target['image_id'] = image_id

            return image, target
        else:
            return image

    def example_image(self, index=None):
        if index is None:
            image_path = str(random.choice(self.images))
        else:
            image_path = str(self.images[index])

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if self.annotations is not None:
            labels, boxes = get_target_data(image_path, self.annotations)

            for idx in range(len(boxes)):
                box = boxes[idx]
                label = labels[idx]
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                cv2.putText(image, label, (box[0], box[1] - 5), 0, 0.5, (255, 0, 0))

        cv2.imshow('Image', image)
        cv2.waitKey(0)


# root = "Example_Dataset/"
# data = Data(root, images_path="train", classes_file_path="class_names.csv",
#             annotations_file_path="train/_annotations.csv")
# data.example_image()
