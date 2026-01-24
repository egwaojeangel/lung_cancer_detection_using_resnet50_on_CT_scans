# test.py

import os
import urllib.request
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# 1) MODEL DOWNLOAD SETUP
# ===========================

model_path = "./resnet50_lung_cancer_binary.pth"
model_url = "https://drive.google.com/uc?export=download&id=1Vc8dM9IsCOhLWreL0JSWNdi4L3TjrHBf"

# Download automatically if model does not exist
if not os.path.exists(model_path):
    print("Model file not found. Downloading now ...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Download complete!")

# ===========================
# 2) DATASET CLASS
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
# 3) TEST DATA TRANSFORMS
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
# 4) LOAD TEST DATASET
# ===========================

test_path = r"./Lung_CT_test_images"  # Your folder with non_cancerous & cancerous
test_dataset = LungCancerDataset(test_path, transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Found {len(test_dataset)} test images.")

# ===========================
# 5) LOAD MODEL
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
# 6) TESTING LOOP
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
# 7) METRICS
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
# 8) CONFUSION MATRIX PLOT
# ===========================

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-cancerous","Cancerous"],
            yticklabels=["Non-cancerous","Cancerous"])
plt.xlabel("Detected")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
