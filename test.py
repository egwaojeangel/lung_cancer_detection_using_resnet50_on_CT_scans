# test.py

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# 0) Install gdown if missing
# ===========================
try:
    import gdown
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "gdown"])
    import gdown

# ===========================
# 1) DOWNLOAD MODEL AUTOMATICALLY
# ===========================

model_path = "./resnet50_lung_cancer_binary.pth"
model_url = "https://drive.google.com/uc?export=download&id=1Vc8dM9IsCOhLWreL0JSWNdi4L3TjrHBf"

if not os.path.exists(model_path):
    print("Downloading model...")
    gdown.download(model_url, model_path, quiet=False)
    print("Model download complete!")

# ===========================
# 2) DOWNLOAD TEST DATASET AUTOMATICALLY
# ===========================

test_folder = "./Lung_CT_test_images"
test_folder_drive_link = "https://drive.google.com/drive/folders/1Gy0fecxzm7d3i_0ibT6XGf5Db_5cW7kV?usp=drive_link"

if not os.path.exists(test_folder):
    print("Downloading test dataset folder...")
    gdown.download_folder(test_folder_drive_link, output=test_folder, quiet=False, use_cookies=False)
    print("Test dataset download complete!")

# ===========================
# 3) DATASET CLASS
# ===========================

class_folders = {'non_cancerous': 0, 'cancerous': 1}

class LungCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        for cls, label in class_folders.items():
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.images.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ===========================
# 4) TEST DATA TRANSFORMS
# ===========================

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===========================
# 5) LOAD TEST DATASET
# ===========================

test_dataset = LungCancerDataset(test_folder, transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Found {len(test_dataset)} test images.")

# ===========================
# 6) LOAD MODEL
# ===========================

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_ftrs, 2)
)

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ===========================
# 7) TESTING LOOP
# ===========================

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

# ===========================
# 8) METRICS
# ===========================

conf_matrix = confusion_matrix(all_labels, all_preds)

TN = conf_matrix[0,0]
FP = conf_matrix[0,1]
FN = conf_matrix[1,0]
TP = conf_matrix[1,1]

accuracy = (TP + TN) / (TP + TN + FP + FN)
sensitivity = TP / (TP + FN)       # Recall
specificity = TN / (TN + FP)
precision = TP / (TP + FP)
f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

print("\nTest Metrics:")
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score:    {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# ===========================
# 9) CONFUSION MATRIX PLOT
# ===========================

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-cancerous","Cancerous"],
            yticklabels=["Non-cancerous","Cancerous"])
plt.xlabel("Detected")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

