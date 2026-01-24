import os
import shutil
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
original_dataset_path = r"C:\Users\egwao\OneDrive\Desktop\lung_cancer_detection\original_dataset"
imbalanced_test_path = r"C:\Users\egwao\OneDrive\Desktop\lung_cancer_detection\imbalanced_test"
model_path = r"C:\Users\egwao\OneDrive\Desktop\lung_cancer_detection\resnet50_lung_cancer_binary.pth"

# Step 1: Create imbalanced test set
def create_imbalanced_test_set(ratio_non_cancerous, total_images=260):
    test_path = os.path.join(imbalanced_test_path, f"ratio_{int(ratio_non_cancerous*100)}")
    shutil.rmtree(test_path, ignore_errors=True)
    os.makedirs(test_path)
    for cls in ['non_cancerous', 'cancerous']:
        os.makedirs(os.path.join(test_path, cls))

    num_non_cancerous = int(total_images * ratio_non_cancerous)
    num_cancerous = total_images - num_non_cancerous

    non_cancerous_files = os.listdir(os.path.join(original_dataset_path, 'non_cancerous'))
    cancerous_files = os.listdir(os.path.join(original_dataset_path, 'cancerous'))

    if len(non_cancerous_files) < num_non_cancerous or len(cancerous_files) < num_cancerous:
        raise ValueError(f"Not enough images for ratio {ratio_non_cancerous}: "
                         f"Need {num_non_cancerous} non-cancerous, {num_cancerous} cancerous")

    random.shuffle(non_cancerous_files)
    random.shuffle(cancerous_files)
    selected_non_cancerous = non_cancerous_files[:num_non_cancerous]
    selected_cancerous = cancerous_files[:num_cancerous]

    for img in selected_non_cancerous:
        shutil.copy(os.path.join(original_dataset_path, 'non_cancerous', img),
                    os.path.join(test_path, 'non_cancerous', img))
    for img in selected_cancerous:
        shutil.copy(os.path.join(original_dataset_path, 'cancerous', img),
                    os.path.join(test_path, 'cancerous', img))

    return len(selected_non_cancerous), len(selected_cancerous), test_path

# Step 2: Dataset class
class LungCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        class_folders = {'non_cancerous': 0, 'cancerous': 1}
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

# Step 3: Transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    # Step 4: Load model
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to('cpu')
    model.eval()
    print("Model loaded.")

    # Step 5: Test across multiple ratios
    ratios = [0.5, 0.6, 0.7, 0.8, 0.9]  # Non-cancerous ratios: 50%, 60%, 70%, 80%, 90%
    total_images = 260
    results = {'ratio': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'conf_matrix': []}

    plt.ioff()  # Disable interactive plotting
    for ratio in ratios:
        print(f"\nTesting with {ratio*100}% non-cancerous, {(1-ratio)*100}% cancerous")
        try:
            non_cancerous_count, cancerous_count, test_path = create_imbalanced_test_set(ratio, total_images)
            print(f"Test set created - Non-cancerous: {non_cancerous_count}, Cancerous: {cancerous_count}")

            imbalanced_test_dataset = LungCancerDataset(test_path, transform=test_transform)
            imbalanced_test_loader = DataLoader(imbalanced_test_dataset, batch_size=32, shuffle=False, num_workers=0)
            print(f"Imbalanced test samples: {len(imbalanced_test_dataset)}")

            test_correct = 0
            test_total = 0
            all_preds = []
            all_labels = []

            print("Starting inference...")
            with torch.no_grad():
                for inputs, labels in imbalanced_test_loader:
                    inputs, labels = inputs.to('cpu'), labels.to('cpu')
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            test_accuracy = 100 * test_correct / test_total
            precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
            conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1])

            results['ratio'].append(ratio)
            results['accuracy'].append(test_accuracy)
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1'].append(f1)
            results['conf_matrix'].append(conf_matrix)

            print(f"Accuracy: {test_accuracy:.2f}%")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print("Confusion Matrix:")
            print(conf_matrix)
            print(f"True Negatives (Non-cancerous correct): {conf_matrix[0,0]}")
            print(f"False Positives (Non-cancerous as cancerous): {conf_matrix[0,1]}")
            print(f"False Negatives (Cancerous as non-cancerous): {conf_matrix[1,0]}")
            print(f"True Positives (Cancerous correct): {conf_matrix[1,1]}")

            # Save confusion matrix plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Non-cancerous', 'Cancerous'],
                        yticklabels=['Non-cancerous', 'Cancerous'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix (Ratio {ratio*100}:{(1-ratio)*100})')
            plt.savefig(os.path.join(imbalanced_test_path, f"confusion_matrix_ratio_{ratio*100}.png"))
            plt.close()

        except Exception as e:
            print(f"Error processing ratio {ratio}: {str(e)}")

    # Step 6: Plot metrics vs. ratio
    plt.figure(figsize=(10, 6))
    plt.plot(results['ratio'], results['accuracy'], marker='o', label='Accuracy')
    plt.plot(results['ratio'], results['precision'], marker='o', label='Precision')
    plt.plot(results['ratio'], results['recall'], marker='o', label='Recall')
    plt.plot(results['ratio'], results['f1'], marker='o', label='F1 Score')
    plt.xlabel('Non-cancerous Ratio')
    plt.ylabel('Metric Value')
    plt.title('Performance Metrics vs. Imbalance Ratio')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(imbalanced_test_path, "metrics_vs_ratio.png"))
    plt.close()

    print("\nTesting complete. Results and plots saved in imbalanced_test directory.")

if __name__ == '__main__':
    main()