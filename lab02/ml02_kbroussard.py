### Titanic Dataset Analysis - Machine Learning Lab 2
# **📌 Titanic Dataset Analysis**

**Author:** Kersha Broussard  
**Date:** March 2025  
**Repository:** [GitHub - applied-ml-kersha](https://github.com/kersha0530/applied-ml-kersha)  
**Dataset:** Titanic (Seaborn Library)  

## **1️ Load & Inspect the Data**

```python
# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# Load Titanic Dataset
titanic = sns.load_dataset('titanic')

# Display Basic Dataset Info
titanic.info()

# Display First 10 Rows
print(titanic.head(10))

# Check for Missing Values
titanic.isnull().sum()

# Display Summary Statistics
print(titanic.describe())

# Check Numeric Feature Correlations
print(titanic.corr(numeric_only=True))
```

### **Reflection 1:**
- How many data instances and features exist?
- What are the feature names?
- Are there missing values?
- Which features are numeric vs categorical?
- Are there any correlations?

---

## **2️. Data Exploration & Preparation**

### **2.1 Explore Data Patterns & Distributions**

```python
# Scatter Matrix of Numeric Attributes
attributes = ['age', 'fare', 'pclass']
scatter_matrix(titanic[attributes], figsize=(10, 10))
plt.show()

# Scatter Plot: Age vs Fare by Gender
plt.scatter(titanic['age'], titanic['fare'], c=titanic['sex'].apply(lambda x: 0 if x == 'male' else 1))
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Age vs Fare by Gender')
plt.show()

# Histogram of Age
sns.histplot(titanic['age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Count Plot: Class & Survival
sns.countplot(x='class', hue='survived', data=titanic)
plt.title('Class Distribution by Survival')
plt.show()
```

### **Reflection 2.1:**
- What patterns or anomalies do you notice?
- Do any features stand out as predictors?
- Are there any class imbalances?

---

### **2.2 Handle Missing Values & Clean Data**

```python
# Fixing the warning by using explicit assignment
titanic.loc[:, "age"] = titanic["age"].fillna(titanic["age"].median())
titanic.loc[:, "embark_town"] = titanic["embark_town"].fillna(titanic["embark_town"].mode()[0])

```

### **2.3 Feature Engineering**

```python
# Create Family Size Feature
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1

# Convert Categorical to Numeric
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})
titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Create Binary Feature for Alone
titanic['alone'] = titanic['alone'].astype(int)
```

### **Reflection 2.3:**
- Why might family size be a useful predictor?
- Why do we convert categorical data to numeric?

---

## **3️. Feature Selection & Justification**

```python
# Define Features & Target
X = titanic[['age', 'fare', 'pclass', 'sex', 'family_size']]
y = titanic['survived']
```

### **Reflection 3:**
- Why are these features selected?
- Which features might be highly predictive?

---

## **4️. Train-Test Splitting & Comparison**

### **Basic Train/Test Split**

```python
train_set, test_set = train_test_split(X, y, test_size=0.2, random_state=123)

print('Train size:', len(train_set))
print('Test size:', len(test_set))
```

### **Stratified Train/Test Split**

```python
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)

for train_indices, test_indices in splitter.split(X, y):
    train_set = X.iloc[train_indices]
    test_set = X.iloc[test_indices]

print('Train size:', len(train_set))
print('Test size:', len(test_set))
```

### **Compare Results**

```python
print("Original Class Distribution:\n", y.value_counts(normalize=True))
print("Train Set Class Distribution:\n", train_set['pclass'].value_counts(normalize=True))
print("Test Set Class Distribution:\n", test_set['pclass'].value_counts(normalize=True))
```

### **Reflection 4:**
- Why might stratification improve model performance?
- How close are training & test distributions to the original dataset?
- Which split method produced better class balance?

