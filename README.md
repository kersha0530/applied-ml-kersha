# 📌 California Housing Price Prediction 🏠


## **Project Overview**

This project focuses on predicting median house values in California using the California Housing dataset from Scikit-Learn. The goal is to analyze relationships between various housing features and price, build a regression model, and evaluate its performance.

## **Dataset Information**
- **Dataset:** California Housing Prices (Scikit-Learn)
- **Features:**
  - `MedInc` - Median income in block group
  - `AveRooms` - Average number of rooms per household
- **Target:** `MedHouseVal` - Median house price ($100,000s)

## **Project Workflow**
1. **Load Dataset** 📂
2. **Exploratory Data Analysis (EDA)** 📊
3. **Feature Selection** 🏗
4. **Train ML Model (Linear Regression)** 🤖
5. **Model Evaluation (R², MAE, RMSE)** 📉
6. **Visualize Predictions** 📌

## **Results**
- **R² Score:** 0.46
- **MAE:** 0.62
- **MSE:** 0.84
- **RMSE:** 0.91

## **How to Run**
1. Clone the repository
   ```bash
   git clone https://github.com/kersha0530/applied-ml-ahsrek
#### Install dependencies:
```bash

pip install -r requirements.txt
```

#### Run the Python script:
```bash

python ml01_housing.py
```



### 📂  Dataset

- The dataset includes housing statistics based on the 1990 California census. It contains the following columns:

- MedInc: Median income of households in a block group (in tens of thousands)

- HouseAge: Median age of houses in the block group

- AveRooms: Average number of rooms per household

- AveBedrms: Average number of bedrooms per household

- Population: Total population of the block group

- AveOccup: Average number of household members

- Latitude: Latitude of the block group’s location

- Longitude: Longitude of the block group’s location

- MedHouseVal: Median house value (Target Variable)

### 🛠 Installation & Setup

1️⃣ Create a Virtual Environment

python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run Jupyter Notebook

jupyter notebook

### 🔍 Exploratory Data Analysis (EDA)

#### Steps Performed:

- 1. Load the dataset and check for missing values

- 2. Generate summary statistics

- 3. Visualize feature distributions (histograms, boxplots, scatterplots)

- 4. Plot correlation heatmaps to identify relationships between variables

### 🤖 Machine Learning Model: Linear Regression

#### Steps:

- 1. Define Features (X) and Target (y)

- 2. Split dataset into training and testing sets

- 3. Train a Linear Regression model using Scikit-Learn

- 4. Evaluate the model using R², MAE, and RMSE

### 📌 Repository

### 🔗 GitHub Repository: https://github.com/kersha0530/applied-ml-kersha

### 💡 Author: Kersha Broussard




