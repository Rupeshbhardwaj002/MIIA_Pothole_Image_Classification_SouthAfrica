# src/predict.py
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import PotholeTestDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# Load test CSV
test_csv = "pothole_data/pothole_dataset/test_ids_only.csv"
test_df = pd.read_csv(test_csv)

# Convert IDs to paths
image_root = "pothole_data/pothole_dataset/all_data"
def find_image_path(img_id):
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        path = os.path.join(image_root, img_id + ext)
        if os.path.exists(path):
            return path
    return None

test_df["Image_ID"] = test_df["Image_ID"].apply(find_image_path)
test_df = test_df.dropna().reset_index(drop=True)

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Dataset & Loader
test_dataset = PotholeTestDataset(test_df, transform)
test_loader = DataLoader(test_dataset, batch_size=32)

# Model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("outputs/model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Predictions
predictions = []
with torch.no_grad():
    for images, ids in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        predictions.extend(zip(ids, probs.cpu().numpy()))

# Submission file
submission_df = pd.DataFrame(predictions, columns=["Image_Path", "Pothole_Probability"])
submission_df["Image_ID"] = submission_df["Image_Path"].apply(lambda x: os.path.basename(x).split(".")[0])
submission_df = submission_df[["Image_ID", "Pothole_Probability"]]
submission_df.to_csv("outputs/final_submission.csv", index=False)
print("Final submission saved: outputs/final_submission.csv")
