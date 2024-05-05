import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image

from model import VisionTransformer

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

# Path to the directory with the test images
image_folder = '/home/achazhoor/Documents/2024/VIT_pytorch_obj_det/test_images'

# Path to the saved weights of the model
model_path = '/home/achazhoor/Documents/2024/VIT_pytorch_obj_det/saved_model/vit_object_detector_2024-05-04_22-56-04.pth'

# Load the model and its weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(
    input_shape=(3, image_size, image_size),
    patch_size=patch_size,
    num_patches=num_patches,
    projection_dim=projection_dim,
    num_heads=num_heads,
    mlp_dim=mlp_dim,
    transformer_layers=transformer_layers,
    mlp_head_units=mlp_head_units
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size expected by the model
    transforms.ToTensor()  # Convert the image to a tensor
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Function to load and process images
def load_and_transform_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return transform(image)

# Loop through the images in the directory and perform inference
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for image files
        image_path = os.path.join(image_folder, filename)
        image = load_and_transform_image(image_path, transform).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = model(image).cpu().numpy()[0]  # Forward pass and move predictions to CPU
        print("predictions for", filename, ":", preds)
        # Display the image and its bounding box
        fig, ax = plt.subplots()
        pil_image = Image.open(image_path)
        ax.imshow(pil_image)
        x, y, x2, y2 = preds  # Assuming preds are [x1, y1, x2, y2]
        rect = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()
