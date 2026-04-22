# Robust In-Cabin Distracted Driver Detection

## Project Overview
This project presents a real-time, edge-deployable product prototype (Track 2) designed to detect seven distinct driver distraction behaviors. The system features a custom **3-stage software-based illumination recovery pipeline** that enables standard RGB cameras to function in dark or shadowed vehicle cabins, bypassing the need for expensive infrared (IR) hardware.

## Dependencies
The project requires Python 3.12+ and the following libraries:
*   `ultralytics` (YOLOv8 implementation)
*   `opencv-python` (Image processing)
*   `numpy` (Mathematical transformations)
*   `matplotlib` (Visualization)
*   `torch` (Deep learning backend)

## Code Structure
The project is provided in two formats for reproducibility:
1.  **`ECE570_Final_Project_Code.ipynb`**: A sequential Jupyter Notebook optimized for Google Colab.
2.  **`ECE570_Final_Project_Code.py`**: A standalone Python script for local execution.

## Run Instructions
1.  **Environment Setup**: Install dependencies via `pip install ultralytics opencv-python numpy matplotlib torch`.
2.  **Data Preparation**: Ensure your dataset is structured in the directory defined by `BASE_PATH`.
3.  **Execution**: Run all cells in the Notebook sequentially. The script will automatically:
    *   Purge redundant classes (c6, c7, c8).
    *   Execute the illumination recovery pipeline on all images.
    *   Perform an 80/20 train-validation split.
    *   Fine-tune the YOLOv8-nano model for 15 epochs.
    *   Run a live inference test on `image_test.png`.
    *   ## Run Instructions & Reproducibility

## Authorship and Attribution
**Author**: Mohamed Adel Elsayed Abdelmoeti Ibrahim

### 1. Original Code (Written Entirely by Me)
The following components were engineered and implemented from scratch by the author:
*   **Illumination Recovery Pipeline (Lines 55–85)**: The mathematical implementation of the 3-stage enhancement (`apply_night_enhancement`) including the Median Filter ($k=3$), Gamma Correction ($\gamma=1.5$), and CLAHE (`clipLimit=2.0`).
*   **Data Logic & Purging (Lines 25–50)**: The programmatic logic used to clean the dataset and remove redundant classes.
*   **Automated Data Segregation (Lines 145–175)**: The script used to enforce a strict 80/20 split and move files into appropriate directories to prevent data leakage.
*   **Custom Inference Script (Lines 235–285)**: The logic for mapping raw neural network outputs to human-readable telemetry and testing on unseen real-world images.

### 2. Adapted Code
*   **Model Initialization & Training (Lines 185–215)**: The YOLOv8 model initialization and the `model.train()` loop parameters were adapted from the [Official Ultralytics Documentation](https://docs.ultralytics.com/).

## Dataset Download Policy (Formal Exception)
**Reason for Exception: Proprietary and Restricted Human-Subject Data.**

The dataset used in this project cannot be downloaded automatically via script as it is not publicly accessible. The dataset is a private collection of human-subject data gathered by a research professor at an external academic institution. 

Because the dataset contains sensitive **identifiable human faces**, the author is legally bound by a strict **Non-Disclosure Agreement (NDA)**. Under the terms of this agreement:
1.  The bulk raw dataset remains legally restricted and cannot be distributed, uploaded to public repositories, or shared with evaluators.
2.  A few representative sample images have been included in the final PDF report purely for qualitative demonstration of the enhancement pipeline's efficacy.

This statement serves as a formal declaration of the **Proprietary Data Exception** as outlined in the ECE570 course project deliverables.

### **IMPORTANT: Use the .py file for testing**
The provided **`ECE570_Final_Project_Code.py`** has been optimized for external reproduction. It removes all personal Google Drive dependencies and includes automated error handling for dataset extraction. 

**To run the optimized pipeline:**
1. Upload `ai_distracted2.zip` to your environment.
2. Run: `python ECE570_Final_Project_Code.py`

### **Note on the Jupyter Notebook (.ipynb)**
The `.ipynb` file contains the original development logs and training curves. However, it contains hardcoded paths to the author's Google Drive (`drive.mount`). 
**If you wish to run the Notebook version:** 
You must manually create the directory path `/content/drive/MyDrive/` and place the `ai_distracted2.zip` file there before running, or simply refer to the `.py` script for a seamless execution.
