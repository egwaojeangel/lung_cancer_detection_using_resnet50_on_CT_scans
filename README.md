**Core stack:** Python (PyTorch, Flask) · Deep Learning · Medical Imaging

# Lung Cancer Detection System Using ResNet50 on CT Scans 🫁

## Overview
This project presents a deep learning–based lung cancer detection system
developed using Computed Tomography (CT) scan images. The system utilizes
a transfer learning approach based on the ResNet50 architecture to classify
lung CT scans as either **cancerous** or **non-cancerous**.

The trained model was integrated into a web-based application named **LUNNY**,
designed to support early lung cancer detection and assist healthcare
professionals in medical image analysis.

---

## Motivation
Lung cancer remains the leading cause of cancer-related deaths worldwide,
largely due to delayed diagnosis. Manual analysis of CT scans can be
time-consuming and error-prone, especially under high clinical workloads.

This project demonstrates how deep learning can be used to improve diagnostic
accuracy, reduce delays, and support clinical decision-making in lung cancer
detection.

---

### Dataset
CT scan images were obtained from three publicly available lung imaging datasets to ensure diversity in imaging characteristics and improve the generalizability of the proposed model.

- **IQ-OTH/NCCD Lung Cancer Dataset**  
  (Iraq-Oncology Teaching Hospital / National Center for Cancer Diseases)  
  977 CT images including normal, benign, and malignant lung cases.

- **BIR Lung Dataset**  
  (Biomedical Imaging Research Lung Dataset)  
  476 CT images acquired at the Barnard Institute of Radiology, Madras Medical College, Chennai, India.

- **LIDC-IDRI Dataset**  
  (Lung Image Database Consortium and Image Database Resource Initiative)  
  327 selected CT images with expert radiologist annotations.

A total of **1,782 CT images** were collected (922 cancerous and 860 non-cancerous). After balancing(undersampling cancerous CT scans), **1,720 images** were used, consisting of 860 cancerous and 860 non-cancerous cases.

### Data Split
- Training: 70%
- Validation: 15%
- Testing: 15%

> ⚠️ Due to size and privacy constraints, the datasets are not included in
> this repository.

---

## Methodology

### Image Preprocessing
- Images resized to **224 × 224 pixels**
- Converted to RGB format
- Normalization applied for stable training
- Independent preprocessing for each dataset split

### Data Augmentation (Training Only)
- Horizontal and vertical flipping
- Rotation (±10°)
- Random cropping
- Color jittering

---

## Model Architecture
- Base Model: **ResNet50 (ImageNet pre-trained)**
- Transfer learning approach
- Early layers frozen
- Final layers fine-tuned for binary classification
- Dropout rate: 0.5

**Output Classes**
- Cancerous
- Non-cancerous

---

## Training Details
- Framework: PyTorch
- Optimizer: Adam
- Learning rate scheduling
- ReLU activation
- Early stopping based on validation performance

---

## Results
The model achieved strong performance on unseen test data:

- **Accuracy:** 98.46%
- **Recall:** 99.21%
- **Precision:** 97.69%
- **F1-score:** 98.45%
### Confusion Matrix
![Confusion Matrix](Screenshots%20Of%20Results%20and%20System%20Deployment/confusion_matrix.png)

### Test Results
![Test Results](Screenshots%20Of%20Results%20and%20System%20Deployment/test_results.png)

---

## System Architecture
![System Architecture](Screenshots%20Of%20Results%20and%20System%20Deployment/System_Archictecture_%20Of_%20The_Lung_Cancer_Detection_System.png)

# System Architecture Overview:

The proposed lung cancer detection system follows an end-to-end deep learning pipeline. Lung CT images are collected from multiple datasets, balanced to ensure equal cancerous and non-cancerous samples, and split into training (70%), validation (15%), and testing (15%) sets using stratified sampling.

All images are preprocessed to meet ResNet50 input requirements, while data augmentation is applied only to the training set to improve generalization. A pre-trained ResNet50 model is fine-tuned using transfer learning for binary lung cancer classification and evaluated using standard performance metrics. The trained model is then deployed as a web-based application for real-time lung cancer detection, with the entire workflow fully documented for reproducibility.

---

## Web Application Deployment (LUNNY)

### Landing Page
![Landing Page](Screenshots%20Of%20Results%20and%20System%20Deployment/landing_page.png)

### Terms and Conditions Page
![Terms and Conditions](Screenshots%20Of%20Results%20and%20System%20Deployment/terms_and_conditions_page.png)

### Post-Login Dashboard
![Post Login Page](Screenshots%20Of%20Results%20and%20System%20Deployment/post_login_page.png)

### Upload & Analysis Result
![Analysis Result](Screenshots%20Of%20Results%20and%20System%20Deployment/Analysis_result.png)

### Add Patient Records
![Add Records](Screenshots%20Of%20Results%20and%20System%20Deployment/Add_Records_page.png)

### Patient Records Table
![Patient Records](Screenshots%20Of%20Results%20and%20System%20Deployment/patients_record_page.png)

### View Patient Record
![View Patient Record](Screenshots%20Of%20Results%20and%20System%20Deployment/view_patient_record.png)

### Printable Patient Result
![Print Patient Result](Screenshots%20Of%20Results%20and%20System%20Deployment/print_patient_result.png)

---

## Disclaimer
This system is intended **strictly for research and educational purposes**.
It is not a certified medical device and must not be used as a replacement
for professional medical diagnosis, treatment, or clinical decision-making.

All outputs generated by this system should be reviewed and interpreted by
qualified healthcare professionals.

---

## Author
**Angel Egwaoje**

---

## Future Work
- Expansion of dataset size
- Add explainability methods (e.g., Grad‑CAM)
- Extend to 3D or multi‑view CNNs for better spatial understanding
- Clinical validation with expert radiologists








