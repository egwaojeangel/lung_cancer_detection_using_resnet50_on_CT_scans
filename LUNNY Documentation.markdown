# LUNNY - Lung Cancer Detection System Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Pages and Their Interfaces](#pages-and-their-interfaces)
   - [Landing Page (Sign In/Register)](#landing-page-sign-inregister)
   - [Terms and Conditions Page](#terms-and-conditions-page)
   - [Post-Login Page](#post-login-page)
   - [Scan Page](#scan-page)
   - [Print Preview Page](#print-preview-page)
   - [Records Page](#records-page)
   - [Patient Records Page](#patient-records-page)
   - [View Record Page](#view-record-page)
4. [Button Interfaces and States](#button-interfaces-and-states)
5. [JavaScript Functionality](#javascript-functionality)
6. [CSS Styling](#css-styling)
7. [Conclusion](#conclusion)

---

## Overview
**LUNNY** is a web-based application designed to:
- Authenticate users (registration and sign-in).
- Allow users to upload lung CT scans for AI-based analysis.
- Manage patient records, including demographics, medical history, imaging results, and treatment plans.
- Provide functionalities like printing, sharing, and deleting scan results and patient records.

The system uses a client-server architecture, with the frontend built using HTML, CSS, and JavaScript, and a backend (assumed to be at `http://127.0.0.1:5000`) handling authentication, scan analysis, and record management. The frontend communicates with the backend via RESTful API endpoints (`/register`, `/login`, `/detect`, `/add_record`, `/update_record`, `/delete_record`, `/get_records`).

---

## System Architecture
- **Frontend**: A single-page application with multiple views (pages) toggled via JavaScript. Each page is a `<div>` with the class `.page`, and only one page is active at a time.
- **Backend**: A server (not provided in the code) that processes authentication, AI-based scan analysis, and database operations for patient records.
- **Navigation**: Managed through a `navigationHistory` array, allowing forward and backward navigation.
- **State Management**: Global variables like `records`, `userEmail`, `isEditMode`, `currentScanImage`, `selectedPatientId`, and `currentResult` maintain the application's state.

---

## Pages and Their Interfaces

### Landing Page (Sign In/Register)
**Purpose**: The entry point for users to register or sign in.
**HTML Structure**:
- **Container**: `.login-container` with inputs for hospital name, admin ID, email, password, and a password strength bar.
- **Buttons**:
  - `Register` (`#register-btn`): Initiates registration.
  - `Sign In` (`#signin-btn`): Initiates login.
- **Other Elements**:
  - Password strength indicator (`.password-strength` and `#strength-bar`).
  - Authentication message (`#auth-message`).

**Interfaces**:
- **Register Button**:
  - **Enabled State**: When password meets criteria (at least 8 characters, 1 uppercase, 1 lowercase, 1 number, 1 special character).
  - **Disabled State**: Default state or when password criteria are not met.
  - **Loading State**: When clicked, adds `.loading` class with a spinner animation.
- **Sign In Button**:
  - **Enabled State**: Always enabled unless in loading state.
  - **Loading State**: When clicked, adds `.loading` class with a spinner animation.

**Functionality**:
- Validates input fields.
- Checks password strength for registration.
- Sends authentication requests to the backend.

### Terms and Conditions Page
**Purpose**: Displays terms of use for users to accept or decline.
**HTML Structure**:
- **Container**: `#terms-page` with a back button, terms content, and two buttons.
- **Buttons**:
  - `Accept`: Navigates to the post-login page.
  - `Decline`: Navigates back to the landing page.
- **Other Elements**: Back button (`.back-btn`).

**Interfaces**:
- **Accept Button**:
  - **Enabled State**: Always enabled.
  - **Action**: Triggers `navigate('post-login-page')`.
- **Decline Button**:
  - **Enabled State**: Always enabled.
  - **Action**: Triggers `navigateBack()`.
- **Back Button**:
  - **Enabled State**: Always enabled.
  - **Action**: Triggers `navigateBack()`.

**Functionality**:
- Displays static terms and conditions.
- Allows users to proceed or return to the landing page.

### Post-Login Page
**Purpose**: The main dashboard after authentication, providing options for further actions.
**HTML Structure**:
- **Container**: `#post-login-page` with a welcome message, instruction text, and three buttons.
- **Buttons**:
  - `Upload New Scan`: Navigates to the scan page.
  - `Access Patient Records`: Navigates to the patient records page.
  - `Logout`: Initiates logout.
- **Other Elements**: Back button (`.back-btn`).

**Interfaces**:
- **Upload New Scan Button**:
  - **Enabled State**: Always enabled.
  - **Action**: Triggers `navigate('scan-page')`.
- **Access Patient Records Button**:
  - **Enabled State**: Always enabled.
  - **Action**: Triggers `navigate('patient-records')`.
- **Logout Button**:
  - **Enabled State**: Always enabled.
  - **Action**: Triggers `confirmLogout()`, prompting for confirmation.
- **Back Button**:
  - **Enabled State**: Always enabled.
  - **Action**: Triggers `navigateBack()`.

**Functionality**:
- Displays a personalized welcome message based on the user's email.
- Provides navigation to key functionalities.

### Scan Page
**Purpose**: Allows users to upload and analyze lung CT scans.
**HTML Structure**:
- **Container**: `.scan-container` with file input, file label, and analysis result display.
- **Buttons**:
  - `Upload CT Scan` (`.file-label` for `#file-input`): Triggers file selection.
  - `Analyze Scan`: Initiates scan analysis.
  - `Save As` (`#save-as-btn`): Saves the scan result as a patient record.
  - `Print Result` (`#print-result-btn`): Opens print preview.
  - `Share Result` (`#share-result-btn`): Copies result to clipboard.
  - `Delete` (`#delete-btn`): Deletes the current scan.
- **Other Elements**:
  - File name display (`#file-name`).
  - Loader text (`#loader`).
  - Result display (`#result`).
  - Advice quotes (`.advice-quote`, `.advice-citation`).
  - Back button (`.back-btn`).

**Interfaces**:
- **Upload CT Scan Button**:
  - **Enabled State**: Always enabled.
  - **Action**: Opens file explorer for selecting PNG, JPEG, or DICOM files.
- **Analyze Scan Button**:
  - **Enabled State**: Always enabled.
  - **Action**: Triggers `analyzeScan()`, sending the file to the backend for analysis.
- **Save As Button**:
  - **Enabled State**: Visible only after analysis result is displayed.
  - **Action**: Triggers `saveAsRecord()`, navigating to the records page with pre-filled AI output.
- **Print Result Button**:
  - **Enabled State**: Visible only after analysis result is displayed.
  - **Action**: Triggers `printResult()`, navigating to the print preview page.
- **Share Result Button**:
  - **Enabled State**: Visible only after analysis result is displayed.
  - **Action**: Triggers `shareResult()`, copying the result text to the clipboard.
- **Delete Button**:
  - **Enabled State**: Visible only after analysis result is displayed.
  - **Action**: Triggers `confirmDeleteScan()`, prompting to clear the scan data.
- **Back Button**:
  - **Enabled State**: Always enabled.
  - **Action**: Triggers `navigateBack()`.

**Functionality**:
- Uploads and preprocesses images (resizes to 224x224).
- Sends images to the backend for AI analysis.
- Displays results with confidence scores and disclaimers.
- Rotates health-related quotes every 5 seconds.

### Print Preview Page
**Purpose**: Displays a preview of the scan result for printing.
**HTML Structure**:
- **Container**: `#print-preview` with an image and result preview.
- **Buttons**:
  - `Print` (`#print-document-btn`): Initiates printing.
- **Other Elements**:
  - Image preview (`#preview-image`).
  - Result preview (`#preview-result`).
  - Back button (`.back-btn`).

**Interfaces**:
- **Print Button**:
  - **Enabled State**: Always enabled.
  - **Action**: Triggers `printDocument()`, opening the browser's print dialog.
- **Back Button**:
  - **Enabled State**: Always enabled.
  - **Action**: Triggers `navigateBack()`.

**Functionality**:
- Displays the scan image and analysis result.
- Hides buttons during printing via CSS `@media print`.

### Records Page
**Purpose**: Allows users to add or edit patient records.
**HTML Structure**:
- **Container**: `.record-form` with multiple sections for patient data.
- **Buttons**:
  - `Save Record` (`#save-record-btn`): Saves or updates the record.
- **Other Elements**:
  - Passport upload (`#passport-upload`).
  - Form fields for demographics, medical history, imaging results, etc.
  - Record message (`#record-message`).
  - Back button (`.back-btn`).

**Interfaces**:
- **Save Record Button**:
  - **Enabled State**: Always enabled.
  - **Action**: Triggers `saveRecord()`, validating and sending data to the backend.
- **Back Button**:
  - **Enabled State**: Always enabled.
  - **Action**: Triggers `navigateBack()`.

**Functionality**:
- Collects comprehensive patient data.
- Supports passport image/PDF upload with preview.
- Validates required fields (Patient ID, Full Name).
- Sends data to the backend for storage.

### Patient Records Page
**Purpose**: Displays a table of patient records with search and management options.
**HTML Structure**:
- **Container**: `#patient-records` with a search bar and patient table.
- **Buttons**:
  - `Add Record` (`#add-btn`): Navigates to the records page for adding a new record.
  - `Edit Record` (`#edit-btn`): Navigates to the records page for editing a selected record.
  - `View Record` (`#view-btn`): Navigates to the view record page.
  - `Delete Record` (`#delete-record-btn`): Deletes the selected record.
- **Other Elements**:
  - Search input (`#search-patient`).
  - Patient table (`.patient-table`).
  - No record message (`#no-record-message`).
  - Back button (`.back-btn`).

**Interfaces**:
- **Add Record Button**:
  - **Enabled State**: Always enabled.
  - **Action**: Triggers `addRecord()`, navigating to the records page in add mode.
- **Edit Record Button**:
  - **Enabled State**: Enabled when a patient is selected.
  - **Disabled State**: Disabled when no patient is selected.
  - **Action**: Triggers `editRecord()`, navigating to the records page in edit mode.
- **View Record Button**:
  - **Enabled State**: Enabled when a patient is selected.
  - **Disabled State**: Disabled when no patient is selected.
  - **Action**: Triggers `viewRecord()`, navigating to the view record page.
- **Delete Record Button**:
  - **Enabled State**: Enabled when a patient is selected.
  - **Disabled State**: Disabled when no patient is selected.
  - **Action**: Triggers `deleteRecord()`, prompting for confirmation and deleting the record.
- **Back Button**:
  - **Enabled State**: Always enabled.
  - **Action**: Triggers `navigateBack()`.

**Functionality**:
- Loads patient records from the backend.
- Supports searching by Patient ID or name.
- Allows selection via checkboxes for editing, viewing, or deleting.

### View Record Page
**Purpose**: Displays a detailed view of a patient record.
**HTML Structure**:
- **Container**: `.patient-record` with sections for patient data.
- **Buttons**:
  - `View Documents` (`#view-documents-btn`): Navigates back to the patient records page.
- **Other Elements**:
  - Back button (`.back-btn`).

**Interfaces**:
- **View Documents Button**:
  - **Enabled State**: Always enabled.
  - **Action**: Triggers `navigate('patient-records')`.
- **Back Button**:
  - **Enabled State**: Always enabled.
  - **Action**: Triggers `navigateBack()`.

**Functionality**:
- Displays a formatted view of patient data.
- Includes a passport image and calculated age.

---

## Button Interfaces and States
Below is a consolidated list of all buttons, their interfaces, and states:

| **Page**                | **Button**                | **ID/Class**                     | **Enabled State**                              | **Disabled State**                     | **Loading State**                     | **Action**                              |
|-------------------------|---------------------------|----------------------------------|-----------------------------------------------|----------------------------------------|---------------------------------------|-----------------------------------------|
| Landing Page            | Register                  | `#register-btn`                  | Password meets criteria                       | Password criteria not met              | `.loading` with spinner               | `handleAuth('register')`                |
| Landing Page            | Sign In                   | `#signin-btn`                    | Always enabled                                | N/A                                    | `.loading` with spinner               | `handleAuth('signin')`                  |
| Terms Page              | Accept                    | `.btn` (inline onclick)          | Always enabled                                | N/A                                    | N/A                                   | `navigate('post-login-page')`           |
| Terms Page              | Decline                   | `.btn` (inline onclick)          | Always enabled                                | N/A                                    | N/A                                   | `navigateBack()`                        |
| Post-Login Page         | Upload New Scan           | `.btn` (inline onclick)          | Always enabled                                | N/A                                    | N/A                                   | `navigate('scan-page')`                 |
| Post-Login Page         | Access Patient Records    | `.btn` (inline onclick)          | Always enabled                                | N/A                                    | N/A                                   | `navigate('patient-records')`           |
| Post-Login Page         | Logout                    | `.btn` (inline onclick)          | Always enabled                                | N/A                                    | N/A                                   | `confirmLogout()`                       |
| Scan Page               | Upload CT Scan            | `.file-label`                    | Always enabled                                | N/A                                    | N/A                                   | Opens file explorer                     |
| Scan Page               | Analyze Scan              | `.btn` (not `#delete-btn`)       | Always enabled                                | N/A                                    | N/A                                   | `analyzeScan()`                         |
| Scan Page               | Save As                   | `#save-as-btn`                   | Visible after analysis                        | Hidden before analysis                 | N/A                                   | `saveAsRecord()`                        |
| Scan Page               | Print Result              | `#print-result-btn`              | Visible after analysis                        | Hidden before analysis                 | N/A                                   | `printResult()`                         |
| Scan Page               | Share Result              | `#share-result-btn`              | Visible after analysis                        | Hidden before analysis                 | N/A                                   | `shareResult()`                         |
| Scan Page               | Delete                    | `#delete-btn`                    | Visible after analysis                        | Hidden before analysis                 | N/A                                   | `confirmDeleteScan()`                   |
| Print Preview Page      | Print                     | `#print-document-btn`            | Always enabled                                | N/A                                    | N/A                                   | `printDocument()`                       |
| Records Page            | Save Record               | `#save-record-btn`               | Always enabled                                | N/A                                    | N/A                                   | `saveRecord()`                          |
| Patient Records Page    | Add Record                | `#add-btn`                       | Always enabled                                | N/A                                    | N/A                                   | `addRecord()`                           |
| Patient Records Page    | Edit Record               | `#edit-btn`                      | Patient selected                              | No patient selected                    | N/A                                   | `editRecord()`                          |
| Patient Records Page    | View Record               | `#view-btn`                      | Patient selected                              | No patient selected                    | N/A                                   | `viewRecord()`                          |
| Patient Records Page    | Delete Record             | `#delete-record-btn`             | Patient selected                              | No patient selected                    | N/A                                   | `deleteRecord()`                        |
| View Record Page        | View Documents            | `#view-documents-btn`            | Always enabled                                | N/A                                    | N/A                                   | `navigate('patient-records')`           |
| All Pages               | Back                      | `.back-btn`                      | Always enabled                                | N/A                                    | N/A                                   | `navigateBack()`                        |

**Notes**:
- **Loading State**: Only applicable to authentication buttons (`Register`, `Sign In`) with a spinner animation.
- **Visibility State**: Buttons on the scan page (`Save As`, `Print Result`, `Share Result`, `Delete`) are hidden until an analysis result is available.
- **Disabled State**: Primarily used in the patient records page for `Edit`, `View`, and `Delete` buttons when no patient is selected.

---

## JavaScript Functionality
The JavaScript code is organized in a modular `<script type="module">` block, handling all client-side logic. Key functions include:

- **Authentication**:
  - `checkPasswordStrength()`: Validates password strength and updates the UI.
  - `handleAuth(type)`: Manages registration and login requests.
  - `resetLandingPage()`: Clears the landing page form.

- **Navigation**:
  - `navigate(pageId)`: Switches between pages and updates navigation history.
  - `navigateBack()`: Returns to the previous page.
  - `confirmLogout()`: Clears user data and navigates to the landing page.

- **Scan Analysis**:
  - `analyzeScan()`: Preprocesses and sends images to the backend for analysis.
  - `resetScanPage()`: Clears scan page data.
  - `startQuoteRotation()` / `stopQuoteRotation()`: Manages quote rotation.

- **Record Management**:
  - `addRecord()` / `editRecord()`: Navigates to the records page in add/edit mode.
  - `deleteRecord()`: Deletes a selected record.
  - `saveRecord()`: Saves or updates a patient record.
  - `loadPatientFolders()`: Fetches and displays patient records.
  - `viewRecord()`: Displays a detailed patient record.

- **Utilities**:
  - `checkServerStatus()`: Verifies backend connectivity.
  - `promptForBackendURL()`: Allows users to update the backend URL.
  - `fetchWithRetry()`: Implements exponential backoff for API requests.
  - `fileToBase64()`: Converts files to base64 for storage.
  - `calculateAge()`: Computes patient age from DOB.

**Event Listeners**:
- Attached in `window.onload` for all buttons and inputs.
- Inline `onclick` attributes are also used for some buttons (e.g., in terms page).

---

## CSS Styling
The CSS provides a modern, responsive design with the following features:
- **Global Styles**: Resets margins/padding, uses a dark theme (`#1a1a1a` background), and sets a consistent font (`Segoe UI`).
- **Page Transitions**: Smooth transitions for page changes (opacity and transform).
- **Button Styles**:
  - `.btn`: Gradient background, hover effects, and loading spinner.
  - `.delete-btn`: Red gradient for delete actions.
- **Responsive Design**: Media queries for mobile devices (`max-width: 600px`).
- **Print Styles**: Hides buttons and adjusts layout for printing.
- **Components**:
  - `.login-container`, `.scan-container`, `.record-form`, etc., use consistent padding, shadows, and rounded corners.
  - `.patient-table`: Sticky headers and hover effects for rows.
  - `.passport-preview`: Handles image and PDF previews.

---

## Conclusion
**LUNNY** is a robust system for lung cancer detection and patient record management. The button interfaces are designed to be intuitive, with clear enabled, disabled, and loading states. The JavaScript code is well-organized, handling complex tasks like authentication, scan analysis, and record management, while the CSS ensures a visually appealing and responsive UI. The documentation highlights the purpose and functionality of each page and button, making it easy to understand the system's workflow and interfaces.

For further development:
- Implement error handling for edge cases (e.g., invalid file uploads).
- Enhance backend validation and security.
- Add support for multiple languages or accessibility features.
- Optimize performance for large patient record datasets.