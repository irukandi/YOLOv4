from network import YOLOv4_Mish_416
from data_handling import Data
from loss import Loss
from torch.utils.data import DataLoader
import torch.optim as optim

batch_size = 4
epochs = 100

model = YOLOv4_Mish_416(classes=12, sam_enabled=True, verbose=True)
dataset = Data(folder_path="Example_Dataset/", images_path="train", classes_file_path="class_names.csv",
               annotations_file_path="train/_annotations.csv")

dataset.example_image()

loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=0.001)  # select parameters manually, insert cosine annealing scheduler

for epoch in range(epochs):
  for data in loader:
    X, y = data
    model.zero_grad()
    output = model(X)
    loss = Loss(output, y)
    loss.backward()
    optimizer.step()
    
   print(loss)
  
# perform test with torch.no_grad()
