import os
import shutil
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
zip_path = r"C:\Users\egwao\Downloads\archive (3).zip"
extra_malignant_path = r"C:\Users\egwao\OneDrive\Desktop\lung_cancer_detection\dataset_split\val\malignant"
extra_benign_path = r"C:\Users\egwao\OneDrive\Desktop\lung_cancer_detection\dataset_split\test\benign"
original_dataset_path = r"C:\Users\egwao\OneDrive\Desktop\lung_cancer_detection\original_dataset"
balanced_dataset_path = r"C:\Users\egwao\OneDrive\Desktop\lung_cancer_detection\balanced_dataset"
train_path = r"C:\Users\egwao\OneDrive\Desktop\lung_cancer_detection\train"
val_path = r"C:\Users\egwao\OneDrive\Desktop\lung_cancer_detection\val"
test_path = r"C:\Users\egwao\OneDrive\Desktop\lung_cancer_detection\test"

# Step 1: Extract the zip file (force re-extraction)
extracted_path = r"C:\Users\egwao\OneDrive\Desktop\lung_cancer_detection\extracted_archive"
shutil.rmtree(extracted_path, ignore_errors=True)
os.makedirs(extracted_path)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_path)
print("Zip file extracted fresh.")

# Debug zip contents
def print_dir_tree(path, prefix=""):
    contents = os.listdir(path)
    for item in contents:
        item_path = os.path.join(path, item)
        print(f"{prefix}{item}")
        if os.path.isdir(item_path):
            print_dir_tree(item_path, prefix + "  ")
print("Zip contents at extracted_path:")
print_dir_tree(extracted_path)

# Step 2: Combine all original images
class_folders = {'non_cancerous': 0, 'cancerous': 1}
shutil.rmtree(original_dataset_path, ignore_errors=True)
os.makedirs(original_dataset_path)
for cls in class_folders:
    os.makedirs(os.path.join(original_dataset_path, cls), exist_ok=True)

normal_count = 0
benign_count = 0
malignant_count = 0
extra_benign_count = 0
extra_malignant_count = 0

# From archive (3).zip
for root, dirs, files in os.walk(extracted_path):
    for cls in ['normal', 'benign', 'malignant']:
        if cls.lower() in root.lower():
            for img_name in files:
                if img_name.endswith('.jpg'):
                    src = os.path.join(root, img_name)
                    if cls in ['normal', 'benign']:
                        dst = os.path.join(original_dataset_path, 'non_cancerous', f"{cls}_{img_name}")
                        shutil.copy(src, dst)
                        if cls == 'normal':
                            normal_count += 1
                        else:
                            benign_count += 1
                    elif cls == 'malignant':
                        dst = os.path.join(original_dataset_path, 'cancerous', img_name)
                        shutil.copy(src, dst)
                        malignant_count += 1

# Add extra benign and malignant
if os.path.exists(extra_benign_path):
    benign_files = os.listdir(extra_benign_path)
    print(f"Found {len(benign_files)} extra benign images at {extra_benign_path}")
    for img_name in benign_files:
        src = os.path.join(extra_benign_path, img_name)
        dst = os.path.join(original_dataset_path, 'non_cancerous', f"extra_benign_{img_name}")
        shutil.copy(src, dst)
        extra_benign_count += 1
else:
    print(f"Extra benign path {extra_benign_path} does not exist!")

if os.path.exists(extra_malignant_path):
    malignant_files = os.listdir(extra_malignant_path)
    print(f"Found {len(malignant_files)} extra malignant images at {extra_malignant_path}")
    for img_name in malignant_files:
        src = os.path.join(extra_malignant_path, img_name)
        dst = os.path.join(original_dataset_path, 'cancerous', f"extra_malignant_{img_name}")
        shutil.copy(src, dst)
        extra_malignant_count += 1
else:
    print(f"Extra malignant path {extra_malignant_path} does not exist!")

print(f"Original counts from zip - Normal: {normal_count}, Benign: {benign_count}, Malignant: {malignant_count}")
print(f"Extra counts - Benign: {extra_benign_count}, Malignant: {extra_malignant_count}")
non_cancerous_total = normal_count + benign_count + extra_benign_count
cancerous_total = malignant_count + extra_malignant_count
print(f"Non-cancerous total: {non_cancerous_total}, Cancerous total: {cancerous_total}")

# Step 3: Balance dataset by undersampling cancerous to match non_cancerous
shutil.rmtree(balanced_dataset_path, ignore_errors=True)
os.makedirs(balanced_dataset_path)
for cls in class_folders:
    os.makedirs(os.path.join(balanced_dataset_path, cls), exist_ok=True)

non_cancerous_files = os.listdir(os.path.join(original_dataset_path, 'non_cancerous'))
cancerous_files = os.listdir(os.path.join(original_dataset_path, 'cancerous'))

# Balance by taking min of both classes (778)
min_count = min(non_cancerous_total, cancerous_total)  # 778
random.shuffle(cancerous_files)
cancerous_files = cancerous_files[:min_count]  # Reduce cancerous from 922 to 778

# Copy balanced set
for img in non_cancerous_files:
    shutil.copy(os.path.join(original_dataset_path, 'non_cancerous', img),
                os.path.join(balanced_dataset_path, 'non_cancerous', img))
for img in cancerous_files:
    shutil.copy(os.path.join(original_dataset_path, 'cancerous', img),
                os.path.join(balanced_dataset_path, 'cancerous', img))

print(f"Balanced counts - Non-cancerous: {len(os.listdir(os.path.join(balanced_dataset_path, 'non_cancerous')))}, "
      f"Cancerous: {len(os.listdir(os.path.join(balanced_dataset_path, 'cancerous')))}")

# Step 4: Split into train, val, test with clearing directories
for cls in class_folders:
    shutil.rmtree(os.path.join(train_path, cls), ignore_errors=True)
    shutil.rmtree(os.path.join(val_path, cls), ignore_errors=True)
    shutil.rmtree(os.path.join(test_path, cls), ignore_errors=True)
    os.makedirs(os.path.join(train_path, cls))
    os.makedirs(os.path.join(val_path, cls))
    os.makedirs(os.path.join(test_path, cls))

for cls in class_folders:
    cls_path = os.path.join(balanced_dataset_path, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)
    total = len(images)  # 778 for both classes
    train_size = int(0.7 * total)  # 544
    val_size = int(0.15 * total)   # 116
    
    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]
    
    for img in train_images:
        shutil.copy(os.path.join(cls_path, img), os.path.join(train_path, cls, img))
    for img in val_images:
        shutil.copy(os.path.join(cls_path, img), os.path.join(val_path, cls, img))
    for img in test_images:
        shutil.copy(os.path.join(cls_path, img), os.path.join(test_path, cls, img))

# Verify split counts
train_non_cancerous = len(os.listdir(os.path.join(train_path, 'non_cancerous')))
train_cancerous = len(os.listdir(os.path.join(train_path, 'cancerous')))
val_non_cancerous = len(os.listdir(os.path.join(val_path, 'non_cancerous')))
val_cancerous = len(os.listdir(os.path.join(val_path, 'cancerous')))
test_non_cancerous = len(os.listdir(os.path.join(test_path, 'non_cancerous')))
test_cancerous = len(os.listdir(os.path.join(test_path, 'cancerous')))

print(f"Train split - Non-cancerous: {train_non_cancerous}, Cancerous: {train_cancerous}")
print(f"Val split - Non-cancerous: {val_non_cancerous}, Cancerous: {val_cancerous}")
print(f"Test split - Non-cancerous: {test_non_cancerous}, Cancerous: {test_cancerous}")

# Step 5: Define dataset class
class LungCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for cls, label in class_folders.items():
            cls_dir = os.path.join(root_dir, cls)
            cls_images = [os.path.join(cls_dir, img) for img in os.listdir(cls_dir)]
            self.images.extend(cls_images)
            self.labels.extend([label] * len(cls_images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transforms (augmentation during training only)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 6: Load datasets
train_dataset = LungCancerDataset(train_path, transform=train_transform)
val_dataset = LungCancerDataset(val_path, transform=val_test_transform)
test_dataset = LungCancerDataset(test_path, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

# Step 7: Load pre-trained ResNet50 and unfreeze layer4
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_ftrs, 2)
)

model = model.to('cpu')

# Step 8: Define loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001, weight_decay=0.01
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Step 9: Training loop
print("Starting training...")
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cpu'), labels.to('cpu')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_accuracy = 100 * train_correct / train_total
    scheduler.step()

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to('cpu'), labels.to('cpu')
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * val_correct / val_total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, "
          f"Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

# Step 10: Test the model with detailed metrics
model.eval()
test_correct = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to('cpu'), labels.to('cpu')
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Compute additional metrics
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')
conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1])

# Print metrics
print("\nDetailed Test Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print(f"True Negatives (Non-cancerous correct): {conf_matrix[0,0]}")
print(f"False Positives (Non-cancerous as cancerous): {conf_matrix[0,1]}")
print(f"False Negatives (Cancerous as non-cancerous): {conf_matrix[1,0]}")
print(f"True Positives (Cancerous correct): {conf_matrix[1,1]}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-cancerous', 'Cancerous'], 
            yticklabels=['Non-cancerous', 'Cancerous'])
plt.xlabel('Detected')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step 11: Save the model
torch.save(model.state_dict(), r"C:\Users\egwao\OneDrive\Desktop\lung_cancer_detection\resnet50_lung_cancer_binary.pth")
print("Model saved.")