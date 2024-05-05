import os
import urllib.request
import zipfile
import shutil
import tarfile
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import scipy.io
import torch
from torch.utils.data import random_split
from torch import Tensor
import matplotlib.pyplot as plt
from model import VisionTransformer
from datetime import datetime
# Path setup
root_path = Path('/home/achazhoor/Documents/2024/VIT_pytorch_obj_det')
path_images = root_path / "101_ObjectCategories/airplanes"
path_annot = root_path / "Annotations/Airplanes_Side_2"

# Define the dataset class
class Caltech101Dataset(Dataset):
    def __init__(self, img_dir, annot_dir, transform=None):
        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
        self.annotations = sorted([f for f in os.listdir(annot_dir) if os.path.isfile(os.path.join(annot_dir, f))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        annot_path = os.path.join(self.annot_dir, self.annotations[idx])
        image = Image.open(img_path).convert("RGB")
        annot = scipy.io.loadmat(annot_path)["box_coord"][0]
        top_left_x, top_left_y = annot[2], annot[0]
        bottom_right_x, bottom_right_y = annot[3], annot[1]

        if self.transform:
            image = self.transform(image)
        _, h, w = image.shape

        box = torch.tensor([top_left_x / w, top_left_y / h, bottom_right_x / w, bottom_right_y / h])

        return image, box

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset and data loaders
dataset = Caltech101Dataset(path_images, path_annot, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


print("the length of the dataset is ", len(dataset))

if len(dataset) > 0:
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
else:
    print("Dataset is empty. Check data paths and contents.")

# The outpur shape of the image is 32, 3, 224, 224 and of the tensor is 32, 4 
# 4 is the bounding box coordinates


from torch.optim import AdamW
from torch.nn import MSELoss
import numpy as np


import os

# Define the directory where the model will be saved
model_save_path = '/home/achazhoor/Documents/2024/VIT_pytorch_obj_det/saved_model'
os.makedirs(model_save_path, exist_ok=True)  # Create the directory if it doesn't exist
import os
import torch
import numpy as np
from torch.optim import AdamW
from torch.nn import MSELoss

def bounding_box_intersection_over_union(boxA, boxB):
    # Assuming boxA and boxB are in the format [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def train_model(model, train_loader, val_loader, device, learning_rate, weight_decay, num_epochs):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = MSELoss()

    # Early stopping settings
    patience = 25
    best_val_loss = np.inf
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_pred_boxes = []
        all_true_boxes = []

        for images, targets in train_loader:
            images = images.float().to(device)  # Ensure the data is the correct type and on the right device
            targets = targets.float().to(device) 
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            print("train outputs", outputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            # Collect all predictions and targets for IoU calculation
            all_pred_boxes.extend(outputs.detach().cpu().numpy())
            all_true_boxes.extend(targets.detach().cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_iou = np.mean([bounding_box_intersection_over_union(pred, gt) for pred, gt in zip(all_pred_boxes, all_true_boxes)])
        history['train_loss'].append(epoch_loss)
        history['train_iou'].append(epoch_iou)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_pred_boxes = []
        val_true_boxes = []

        with torch.no_grad():
            for images, targets in test_loader:
                images = images.float().to(device)  # Ensure the data is the correct type and on the right device
                targets = targets.float().to(device) 

                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                print("valid outputs", outputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * images.size(0)

                val_pred_boxes.extend(outputs.detach().cpu().numpy())
                val_true_boxes.extend(targets.detach().cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_iou = np.mean([bounding_box_intersection_over_union(pred, gt) for pred, gt in zip(val_pred_boxes, val_true_boxes)])
        history['val_loss'].append(epoch_val_loss)
        history['val_iou'].append(epoch_val_iou)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train IoU: {epoch_iou:.4f}, Val IoU: {epoch_val_iou:.4f}")

        # Early Stopping Check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            model_file_name = f'vit_object_detector_{timestamp}.pth'
            model_file_path = os.path.join(model_save_path, model_file_name)
            torch.save(model.state_dict(), model_file_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return history


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Sample synthetic image (1 image, 3 color channels, image_size x image_size)
image_size = 224  # Define image size
# Create an instance of the VisionTransformer model
patch_size = 32
num_patches = (image_size // patch_size) ** 2
projection_dim = 128
num_heads = 8
mlp_dim = 256
transformer_layers = 8
mlp_head_units = [512, 4]  # Assuming the last layer outputs 4 coordinates
learning_rate = 0.001
weight_decay = 0.0001
num_epochs = 200

import torch
import torch.optim as optim
from torch.utils.data import DataLoader



# Initialize your Vision Transformer model here
model = VisionTransformer(
    input_shape=(3, image_size, image_size),
    patch_size=patch_size,
    num_patches=num_patches,
    projection_dim=projection_dim,
    num_heads=num_heads,
    mlp_dim=mlp_dim,
    transformer_layers=transformer_layers,
    mlp_head_units=mlp_head_units
)

criterion = torch.nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)





history = train_model(model, train_loader, test_loader, device, learning_rate, weight_decay, num_epochs)

def plot_history(history):
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_history(history)
