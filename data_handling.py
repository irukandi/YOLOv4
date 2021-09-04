import os
import cv2
import matplotlib.pyplot as plt
import torch
#from pathlib import Path
#from fastai import *
import torchvision.transforms as transforms
from fastai.vision.all import get_image_files
from fastai.imports import Path
import pandas as pd
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, csv_file, images_path, header=None, transform=None):
        super().__init__()
        if header is None:
            header = ['img_path', 'tl_x', 'tl_y', 'br_x', 'br_y', 'class']
        if transform is None:
            transform = transforms.ToTensor()
        self.images = get_image_files(images_path)
        self.annotations = pd.read_csv(csv_file, names=header)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index].name
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        target = {}
        labels, boxes = get_target_data(img_path, self.annotations)
        iscrowd = torch.zeros((boxes.shape[0],))
        image_id = torch.tensor([index])
        print(labels) # need to convert labels to int
        labels = torch.as_tensor(labels, dtype=torch.float32)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target['boxes'] = boxes
        target['labels'] = labels
        target['iscrowd'] = iscrowd
        target['image_id'] = image_id

        image = self.transform(image)

        return image, target


def get_target_data(img_path, data_frame):
    bbox_columns = ['tl_x', 'tl_y', 'br_x', 'br_y']
    rows = data_frame[data_frame.iloc[:, 0] == img_path]
    return rows['class'].values, rows[bbox_columns].values


root = Path("Example_Dataset")

train_images = get_image_files(root/"test")

img = cv2.imread(str(train_images[-1]), cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

data = Data(csv_file=root/"test/_annotations.csv", images_path=root/"test")

data.__getitem__(3)

