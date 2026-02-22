# 🫁 Lung Cancer Detection Using ResNet50 on CT Scans

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=flat-square&logo=flask)
![Accuracy](https://img.shields.io/badge/Accuracy-98.46%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

**Core Stack:** Python (PyTorch, Flask) · Transfer Learning · ResNet50 · Medical Imaging · CT Scans

A deep learning system for early lung cancer detection from CT scan images, achieving **98.46% accuracy** using a fine-tuned ResNet50 model. Deployed as a full web application — **LUNNY** — designed to assist healthcare professionals with real-time diagnostic support.

---

## 📋 Table of Contents
- [Results](#results)
- [How to Run](#how-to-run)
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Web Application — LUNNY](#web-application-deployment-lunny)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Disclaimer](#disclaimer)

---

## Results

The model achieved strong performance on 260 unseen test images:

| Metric | Score |
|---|---|
| Accuracy | **98.46%** |
| Recall (Sensitivity) | **99.21%** |
| Precision (Specificity) | **97.69%** |
| F1 Score | **98.45%** |

![Test Results](Screenshots%20Of%20Results%20and%20System%20Deployment/test_results.png)

### Confusion Matrix

![Confusion Matrix](Screenshots%20Of%20Results%20and%20System%20Deployment/confusion_matrix.png)

Out of 130 non-cancerous images, 129 were correctly classified (1 misclassified). Out of 130 cancerous images, 127 were correctly identified (3 misclassified). These results confirm strong and reliable classification performance across both classes.

---

## How to Run

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the repository

```bash
git clone https://github.com/egwaojeangel/lung_cancer_detection_using_resnet50_on_CT_scans.git
cd lung_cancer_detection_using_resnet50_on_CT_scans
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the test script

The test script will automatically download the pre-trained model and test dataset if not already present:

```bash
python test.py
```

### 5. Run the web application

```bash
python app.py
```

Then open your browser at: **http://127.0.0.1:5000**

### Test Dataset

The test dataset is available on Google Drive if you want to run it manually:

👉 [Download Test Images](https://drive.google.com/drive/folders/1Gy0fecxzm7d3i_0ibT6XGf5Db_5cW7kV?usp=drive_link)

After downloading, place the folder in the root of the repository:

```
lung_cancer_detection/
├── Lung_CT_test_images/
│   ├── cancerous/
│   └── non_cancerous/
├── app.py
├── test.py
└── ...
```

---

## Overview

Lung cancer remains the leading cause of cancer-related deaths worldwide, largely due to delayed diagnosis. Manual analysis of CT scans is time-consuming and error-prone, especially under high clinical workloads.

This project demonstrates how deep learning can be applied to improve diagnostic accuracy, reduce delays, and support clinical decision-making in lung cancer detection. The system uses a transfer learning approach based on ResNet50, fine-tuned for binary classification of CT scans as cancerous or non-cancerous.

---

## Dataset

CT scan images were obtained from three publicly available lung imaging datasets:

| Dataset | Description | Images |
|---|---|---|
| IQ-OTH/NCCD | Iraq-Oncology Teaching Hospital / National Center for Cancer Diseases | 977 |
| BIR Lung Dataset | Barnard Institute of Radiology, Madras Medical College, Chennai | 476 |
| LIDC-IDRI | Lung Image Database Consortium with expert radiologist annotations | 327 |

**Total: 1,782 CT images** (922 cancerous, 860 non-cancerous). After undersampling to balance classes, **1,720 images** were used — 860 per class.

### Data Split
- Training: 70%
- Validation: 15%
- Testing: 15%

> ⚠️ Due to size and privacy constraints, the full datasets are not included in this repository.

---

## Methodology

### Image Preprocessing
- Resized to **224 × 224 pixels** (ResNet50 input requirement)
- Converted to RGB format
- Normalization applied for training stability
- Independent preprocessing applied per split to prevent data leakage

### Data Augmentation (Training Only)
- Horizontal and vertical flipping
- Rotation (±10°)
- Random cropping
- Color jittering

Augmentations applied to training set only to improve generalization without contaminating validation or test sets.

---

## Model Architecture

- **Base Model:** ResNet50 (ImageNet pre-trained)
- **Approach:** Transfer learning with fine-tuning
- **Early layers:** Frozen to preserve learned low-level features
- **Final layers:** Fine-tuned for binary lung cancer classification
- **Dropout rate:** 0.5 to reduce overfitting

**Output Classes:** Cancerous / Non-cancerous

### System Architecture

![System Architecture](Screenshots%20Of%20Results%20and%20System%20Deployment/System_Archictecture_%20Of_%20The_Lung_Cancer_Detection_System.png)

The pipeline covers data collection, balancing, preprocessing, augmentation, model training, evaluation, and deployment as a full web application.

---

## Training Details

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | Adam |
| Activation | ReLU |
| Learning Rate | Scheduled |
| Early Stopping | Based on validation performance |

---

## Web Application Deployment (LUNNY)

LUNNY is a web-based interface designed for licensed healthcare professionals to upload CT scans and receive real-time diagnostic support.

### Landing Page
![Landing Page](Screenshots%20Of%20Results%20and%20System%20Deployment/landing_page.png)

Access gate for healthcare professionals. Requires hospital name, admin ID, email, and password to register or log in. Includes a password strength analyser during registration.

### Terms and Conditions
![Terms and Conditions](Screenshots%20Of%20Results%20and%20System%20Deployment/terms_and_conditions_page.png)

Legal and ethical disclaimers are presented before system access. Users must accept terms before proceeding.

### Post-Login Dashboard
![Post Login Page](Screenshots%20Of%20Results%20and%20System%20Deployment/post_login_page.png)

Central dashboard with three options: upload a new scan, access patient records, or log out. Displays a personalised welcome message.

### Upload Scan Interface
![Upload Scan](Screenshots%20Of%20Results%20and%20System%20Deployment/upload_scan_page.png)

File input for CT scan upload. Shows a loader animation during model inference. Displays motivational health quotes to keep the interface engaging.

### Analysis Result
![Analysis Result](Screenshots%20Of%20Results%20and%20System%20Deployment/Analysis_result.png)

Displays binary classification result (Positive/Negative), Lung-RADS score, and confidence percentage. Medical disclaimer shown alongside result.

### Add Patient Records
![Add Records](Screenshots%20Of%20Results%20and%20System%20Deployment/Add_Records_page.png)

After viewing results, users can Save, Print, Share, or Delete. Saving redirects to a form for full patient data entry including demographics, scan result, passport photo, and medical history.

### Patient Records Table
![Patient Records](Screenshots%20Of%20Results%20and%20System%20Deployment/patients_record_page.png)

Searchable, interactive table of all saved patient records. Supports add, edit, delete, and detailed view per record.

### View Patient Record
![View Patient Record](Screenshots%20Of%20Results%20and%20System%20Deployment/view_patient_record.png)

Detailed read-only view of selected patient data, scan result, and medical notes in a clean printable layout.

### Printable Patient Result
![Print Patient Result](Screenshots%20Of%20Results%20and%20System%20Deployment/print_patient_result.png)

Print preview of the scan image alongside the result, formatted for hard copy documentation.

---

## Requirements

```
torch
torchvision
Pillow
numpy
scikit-learn
matplotlib
seaborn
gdown
flask
flask-cors
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Limitations
- Binary classification only (cancerous vs non-cancerous) — does not distinguish cancer subtypes
- 2D image-based, not full 3D volumetric CT analysis
- No explainability layer (Grad-CAM not yet integrated)
- No clinical validation with expert radiologists
- Dataset size relatively small compared to clinical-grade systems

---

## Future Work
- Add Grad-CAM explainability for visual interpretation of model decisions
- Extend to multi-class classification (cancer subtypes)
- 3D or multi-view CNN for better spatial understanding
- Expand dataset with larger, more diverse sources
- Clinical validation with expert radiologist review

---

## Disclaimer

This system is intended **strictly for research and educational purposes**. It is not a certified medical device and must not be used as a replacement for professional medical diagnosis, treatment, or clinical decision-making.

All outputs generated by this system should be reviewed and interpreted by qualified healthcare professionals.

---

## Author

**Angel Egwaoje**
Machine Learning Engineer | Computer Vision & Medical Imaging

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/angel-egwaoje-416927280)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/egwaojeangel)































