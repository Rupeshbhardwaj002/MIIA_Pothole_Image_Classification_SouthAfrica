# src/train.py
import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from huggingface_hub import hf_hub_download
from dataset import PotholeTrainDataset

# -------------------------
# CONFIG
# -------------------------
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# DOWNLOAD DATASET
# -------------------------
print("Downloading dataset from HuggingFace Hub...")
zip_path = hf_hub_download(
    repo_id="rupesh002/Patholes_Dataset",
    repo_type="dataset",
    filename="pothole_dataset.zip"
)

os.makedirs("pothole_data", exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("pothole_data")

print("Dataset extracted.")

# -------------------------
# LOAD CSV
# -------------------------
train_csv = "pothole_data/pothole_dataset/train_ids_labels.csv"
train_df = pd.read_csv(train_csv)

image_root = "pothole_data/pothole_dataset/all_data"

def make_path(img_id):
    for ext in [".JPG", ".jpg", ".jpeg", ".png", ".PNG"]:
        path = os.path.join(image_root, f"{img_id}{ext}")
        if os.path.exists(path):
            return path
    return None

img_col = "Image_ID" if "Image_ID" in train_df.columns else train_df.columns[0]
train_df[img_col] = train_df[img_col].apply(make_path)
train_df = train_df.dropna().reset_index(drop=True)
print(f"Total training images: {len(train_df)}")

# -------------------------
# TRANSFORMS
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

# -------------------------
# DATASET & DATALOADER
# -------------------------
train_dataset = PotholeTrainDataset(train_df, transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------
# MODEL
# -------------------------
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------
# TRAINING LOOP
# -------------------------
print("Training started...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss:.4f}")

# -------------------------
# SAVE MODEL
# -------------------------
torch.save(model.state_dict(), "outputs/model.pth")
print("Model saved to outputs/model.pth")
