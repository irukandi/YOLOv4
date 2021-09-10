from network import YOLOv4_Mish_416
from data_handling import Data
from torch.utils.data import DataLoader

batch_size = 4

model = YOLOv4_Mish_416(classes=12, sam_enabled=True, verbose=True)
dataset = Data(folder_path="Example_Dataset/", images_path="train", classes_file_path="class_names.csv",
               annotations_file_path="train/_annotations.csv")

dataset.example_image()

loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
