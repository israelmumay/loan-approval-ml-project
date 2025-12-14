# Loan Approval Prediction Using KNN and SVM

This repository contains the full workflow for a machine learning project predicting loan approval outcomes using K-Nearest Neighbors (KNN) and Support Vector Machines (SVM).  
The project includes preprocessing, model tuning, evaluation, Tableau dashboards, and a professional final report.

---
## 📁 Project Structure
loan-approval-ml-project/
│
├── README.md
├── data/
│ └── loan_data.csv
│
├── code/
│ └── knn_svm_analysis.R
│
├── report/
│ └── Final_Report.pdf


---
## 📂 Code

All clean and documented R scripts are stored in the **/code** folder.

Key processes include:
- Outlier handling (1st and 99th percentile thresholds)
- Log transformations for skewed features
- Dummy encoding for categorical variables
- Scaling (z-score standardization)
- KNN tuning with 5-fold cross-validation
- SVM grid search for cost and gamma
- Model evaluation using Accuracy, Error Rate, AUC, and Confusion Matrix

---

##  Dataset

The dataset used in this project is stored in `/data/loan_data.csv`.

Original source:  
https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data

---

##  Final Report

Full PDF report with all analysis, figures, and interpretation:

 **[Download Final Report](report/Final_Report.pdf)**

---

##  Results Summary

- **Best Model:** SVM (Radial Kernel)  
- **Accuracy:** 92.0% on test set  
- **AUC:** 0.958  
- **Strongest Predictor:** `previous_loan_defaults_on_file`

---

##  Requirements

- R  
- Libraries: `caret`, `e1071`, `class`, `pROC`, `tidyverse`, `ggplot2`

---

##  License

This project is open-source and available for educational use.
