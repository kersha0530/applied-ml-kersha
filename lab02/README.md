# 🚢 Titanic Dataset Analysis - Machine Learning Lab 2

## 📌 Overview
This project analyzes the Titanic dataset to explore survival patterns using **Python, pandas, seaborn, and scikit-learn**. The goal is to **perform data exploration, feature engineering, and prepare the dataset for machine learning models**.

---

## 📊 Dataset Information
The dataset contains **passenger details** from the Titanic disaster, with features such as:
- **Survival (`survived`)**: 1 = Survived, 0 = Did not survive
- **Passenger Class (`pclass`)**: 1st, 2nd, or 3rd class
- **Sex (`sex`)**: Male or Female
- **Age (`age`)**: Passenger's age
- **Siblings/Spouses aboard (`sibsp`)**: Number of siblings/spouses aboard
- **Parents/Children aboard (`parch`)**: Number of parents/children aboard
- **Fare (`fare`)**: Ticket fare
- **Embarkation (`embarked`)**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

---

## 🔍 Analysis Process

### **1️. Load & Inspect the Data**
- Import dataset from **Seaborn** and inspect missing values.
- Display summary statistics and feature correlations.

### **2️. Data Exploration & Preparation**
- Visualize distributions with **scatter plots, histograms, and count plots**.
- Handle missing values and create new features:
  - **Family size** (Total family members aboard)
  - **Categorical encoding** (Convert `sex` and `embarked` to numeric)

### **3️. Feature Selection & Justification**
- Selected input features: `age`, `fare`, `pclass`, `sex`, `family_size`
- Target variable: `survived`

### **4️. Train-Test Splitting**
- **Basic train-test split** (80% training, 20% testing)
- **Stratified split** (Ensures balanced class distribution)

---

## 📌 Required Libraries & Dependencies
To run this project, install the required packages using:

```sh
pip install -r requirements.txt

📂 Repository: GitHub - applied-ml-kersha

