# 📌 Iris Dataset Analysis - Machine Learning Lab
# Author: Kersha Broussard
# Date: March 2025
# Repository: GitHub - applied-ml-kersha
# Dataset: Iris (Seaborn Library)


# 1️. Load & Inspect the Data
# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

# Load Iris Dataset
iris = sns.load_dataset('iris')

# Display Basic Dataset Info
iris.info()

# Display First 10 Rows
print(iris.head(10))

# Check for Missing Values
iris.isnull().sum()

# Display Summary Statistics
print(iris.describe())

# Check Numeric Feature Correlations
print(iris.corr(numeric_only=True))

# Reflection 1:
### 1. How many data instances and features exist?

### 2. What are the feature names?

### 3. Are there missing values?

### 4. Which features are numeric vs categorical?

### 5. Are there any correlations?


# 2️. Data Exploration & Preparation
## 2.1 Explore Data Patterns & Distributions

# Scatter Matrix of Numeric Attributes
attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
scatter_matrix(iris[attributes], figsize=(10, 10))
plt.show()

# Scatter Plot: Sepal Length vs Petal Length by Species
sns.scatterplot(x=iris['sepal_length'], y=iris['petal_length'], hue=iris['species'])
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Sepal Length vs Petal Length by Species')
plt.show()

# Histogram of Sepal Length
sns.histplot(iris['sepal_length'], kde=True)
plt.title('Sepal Length Distribution')
plt.show()

# Count Plot: Species Distribution
sns.countplot(x='species', data=iris)
plt.title('Species Distribution')
plt.show()

# Reflection 2.1:
### 1. What patterns or anomalies do you notice?

### 2. Do any features stand out as predictors?

### 3. Are there any class imbalances?



#2.2 Handle Missing Values & Clean Data

# Check for Missing Values
iris.isnull().sum()
# No missing values in this dataset!


# 2.3 Feature Engineering
## Convert Categorical to Numeric (Species Encoding)
encoder = LabelEncoder()
iris['species'] = encoder.fit_transform(iris['species'])

# Reflection 2.3:
### 1. Why might categorical encoding be useful?

### 2. Why do we convert categorical data to numeric?


# 3. Feature Selection & Justification
## Define Features & Target
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris['species']


# Reflection 3:
# 1. Why are these features selected?

# 2. Which features might be highly predictive?

#4. Train-Test Splitting & Comparison


### Basic Train/Test Split
train_set, test_set = train_test_split(X, y, test_size=0.2, random_state=123)

print('Train size:', len(train_set))
print('Test size:', len(test_set))


### Stratified Train/Test Split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)

for train_indices, test_indices in splitter.split(X, y):
    train_set = X.iloc[train_indices]
    test_set = X.iloc[test_indices]

print('Train size:', len(train_set))
print('Test size:', len(test_set))

### Compare Results
print("Original Class Distribution:\n", y.value_counts(normalize=True))
print("Train Set Class Distribution:\n", train_set['sepal_length'].value_counts(normalize=True))
print("Test Set Class Distribution:\n", test_set['sepal_length'].value_counts(normalize=True))


# Reflection 4:
### 1. Why might stratification improve model performance?

### 2. How close are training & test distributions to the original dataset?

### 3. Which split method produced better class balance?


