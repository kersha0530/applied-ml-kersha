# 🍄 Mushroom Classification - Midterm Project

**Author**: Kersha Broussard 
**Date**: 3/25/2025
**Course**: Applied Machine Learning  

**Imports**
# Core Libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Evaluation
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


## 📘 Project Overview

This project demonstrates how classification models can be applied to predict whether a mushroom is edible or poisonous based on 22 categorical features. The dataset comes from the UCI Machine Learning Repository.

## 🔍 What’s Included

- Mushroom dataset (pre-cleaned & encoded)
- Two classification models: Decision Tree and Random Forest
- Visualizations (distributions, confusion matrices)
- Performance evaluations (accuracy, precision, recall, F1)
- Reflections for each step of the process

## 🚀 Key Results

- **Best Model**: Random Forest
- **Accuracy**: ~ 1.0 %
- Features like **odor**, **gill-size**, and **spore print color** were strong predictors.

## 📁 Files

- `midterm/classification_kbroussard.ipynb`: Full analysis notebook with reflections
- `midterm/data/mushrooms.csv`: Dataset used
- `midterm/peer_review.md`: Review of a classmate’s work

## ✅ How to Run

1. Clone the repo and navigate to the `midterm/` folder.
2. Set up your virtual environment and activate it.
3. Run Jupyter:
   ```bash
   jupyter notebook

   🔗 Links
📓 View My Notebook https://github.com/kersha0530/applied-ml-kersha/blob/main/midterm/classification_kbroussard.ipynb 

📝 View My Peer Review https://github.com/kersha0530/applied-ml-kersha/blob/main/midterm/peer_review.md 


