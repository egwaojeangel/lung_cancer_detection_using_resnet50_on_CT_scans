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

## Download Test Dataset
The test dataset is hosted on Google Drive.
👉 Download the Test Images here:
https://drive.google.com/drive/folders/1Gy0fecxzm7d3i_0ibT6XGf5Db_5cW7kV?usp=drive_link

After downloading, place the folder in the root of the repository like this:

Lung_CT_test_images/
    ├── non_cancerous/
    └── cancerous/

✅ Note: The test.py script will automatically download this dataset folder if it is not already present.

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

#### Confusion Matrix:
The confusion matrix output further illustrates the model’s accuracy. Out of the 130 non-cancerous images, 129 were correctly classified as non-cancerous, while only 1 was misclassified. Similarly, 127 out of 130 cancerous images were accurately predicted, with 3 instances misclassified as non-cancerous. These results confirm the model’s reliability and its ability to distinguish between cancerous and non-cancerous lung tissues with high precision.


### Test Results
![Test Results](Screenshots%20Of%20Results%20and%20System%20Deployment/test_results.png)

#### Test Results:
After training, the model was tested on 260 images, comprising 130 cancerous and 130 non-cancerous samples. The results demonstrated the high performance of the system. The accuracy of the model was measured at 98.46%, indicating that it made correct detections in nearly all test instances. The model achieved a precision (specificity) of 0.9922, meaning that 99.22% of non-cancerous images were correctly identified, with minimal false positives. The sensitivity (recall) score was 0.9769, signifying that 97.69% of cancerous cases were correctly detected by the system. The F1 score, which balances precision and recall, was calculated to be 0.9845, reflecting strong and reliable classification performance across both classes.


---

## System Architecture
![System Architecture](Screenshots%20Of%20Results%20and%20System%20Deployment/System_Archictecture_%20Of_%20The_Lung_Cancer_Detection_System.png)

####  System Architecture Overview:
The proposed lung cancer detection system follows an end-to-end deep learning pipeline. Lung CT images are collected from multiple datasets, balanced to ensure equal cancerous and non-cancerous samples, and split into training (70%), validation (15%), and testing (15%) sets using stratified sampling.

All images are preprocessed to meet ResNet50 input requirements, while data augmentation is applied only to the training set to improve generalization. A pre-trained ResNet50 model is fine-tuned using transfer learning for binary lung cancer classification and evaluated using standard performance metrics. The trained model is then deployed as a web-based application for real-time lung cancer detection, with the entire workflow fully documented for reproducibility.

---

## Web Application Deployment (LUNNY)

### Landing Page 
![Landing Page](Screenshots%20Of%20Results%20and%20System%20Deployment/landing_page.png)

#### Landing Page (Authentication & Access Control):
The application begins at the Landing Page (Sign In/Register). This serves as the access gate for licensed healthcare professionals. Users are required to enter their hospital name, admin ID, email, and password to register or log in. The interface also features a password strength analyzer, ensuring that only strong and secure credentials are accepted during registration. Successful login or registration transitions the user to the Terms and Conditions page.


### Terms and Conditions Page
![Terms and Conditions](Screenshots%20Of%20Results%20and%20System%20Deployment/terms_and_conditions_page.png)

#### Terms of Use and Liability Disclaimer:
The Terms and Conditions Page presents the legal and ethical disclaimers associated with using the 
system. It reminds users that the tool is intended as a support system and should not replace professional medical judgment. This page also displays the system’s contact information. Users must accept the terms before they can proceed further into the application.


### Post-Login Dashboard
![Post Login Page](Screenshots%20Of%20Results%20and%20System%20Deployment/post_login_page.png)

#### Post-Login Dashboard:
Upon accepting the terms, the user is navigated to the Post-Login Page. This page acts as a central dashboard, offering three main options: upload a new scan for analysis, access patient records, or log out. The interface also displays a personalized welcome message based on the logged-in user’s credentials.


### Upload Scan Interface
![Upload Scan](Screenshots%20Of%20Results%20and%20System%20Deployment/upload_scan_page.png)

#### Upload Scan Interface:
There is a file input system for uploading CT scan images and it displays motivational health quotes to keep the interface engaging. After a user uploads an image and clicks the “Analyze” button, the system begins processing the image using the integrated ResNet50 model. A loader animation appears during this time to indicate progress.


### Analysis Result Interface
![Analysis Result](Screenshots%20Of%20Results%20and%20System%20Deployment/Analysis_result.png)

#### Analysis Result Interface:
Once analysis is complete, the result is displayed in a dedicated Result Section. The result consists of a binary classification: either “Positive” or “Negative” and the Lung-RADS score, along with a confidence score indicating the model’s certainty. The result and a medical disclaimer appear within the same container to ensure users clearly understand that the tool supports, but does not replace, formal diagnosis.


### Add Patient Records
![Add Records](Screenshots%20Of%20Results%20and%20System%20Deployment/Add_Records_page.png)

#### Add Patient Records:
After viewing the result, users have the option to Save, Print, Share, or Delete the scan result. Clicking “Save As” redirects the user to the Records Page, where a form allows them to input full patient data including demographics, scan result, passport photo, and medical history. Saving this form adds the entry to the records database.


### Patient Records Table
![Patient Records](Screenshots%20Of%20Results%20and%20System%20Deployment/patients_record_page.png)

#### Patient Records Table
Saved records are then made accessible via the Patient Records Page. This page displays all saved patient data in a searchable, interactive table. Users can add new records, edit existing ones, delete records, or view them in detail. Each record entry supports selection through checkboxes and expandable rows.


### View Patient Record
![View Patient Record](Screenshots%20Of%20Results%20and%20System%20Deployment/view_patient_record.png)

#### View Patient Record
Selecting the “View” option on any record navigates to the View Record Page. This interface presents a detailed read-only view of the selected patient's data, scan result, medical notes, and other saved information in a clean, printable layout.


### Printable Patient Result
![Print Patient Result](Screenshots%20Of%20Results%20and%20System%20Deployment/print_patient_result.png)

#### Printable Patient Result
If the user opts to print the analysis result directly after a scan, they are navigated to the Print Preview Page. This page shows the scan image alongside the result in a formatted layout prepared for hard copy documentation. A print button is included to trigger the browser’s print functionality.


### Requirements

All required Python packages are listed in requirements.txt:

torch, torchvision, Pillow, numpy, scikit-learn, matplotlib, seaborn, gdown

Install with: pip install -r requirements.txt

### Run Testing

The test.py script will automatically download the pre-trained model and test dataset if not already present:

python test.py


### Installation & Setup

#### Clone the repository:

git clone https://github.com/egwaojeangel/lung_cancer_detection_using_resnet50_on_CT_scans.git

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





















