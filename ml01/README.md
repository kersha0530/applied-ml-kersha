# 📊 Post-College Salary Prediction: Machine Learning Project

## 🚀 Project Overview
This project explores how **college majors** influence **mid-career salaries** using a dataset from Kaggle. We clean, analyze, and apply machine learning models to predict salaries based on early-career pay and other features.

## 📂 Dataset
- **Source:** [Kaggle - Post-College Salaries](https://www.kaggle.com/datasets/rathoddharmendra/post-college-salaries)
- **File Used:** `final-post-college-salaries.csv`
- **Key Features:**
  - `Rank`: Major ranking based on salaries
  - `Major`: Field of study
  - `Degree Type`: Type of degree (e.g., Bachelor's)
  - `Early Career Pay`: Median salary for early-career graduates
  - `Mid-Career Pay`: Median salary for mid-career professionals
  - `% High Meaning`: Percentage of people who find their job meaningful

## 📌 Process
### 1️⃣ Data Preprocessing
- Removed **missing values**
- Converted **salary columns** from text (`$98,100`) to numeric (`98100`)
- Applied **one-hot encoding** to categorical variables (`Major`, `Degree Type`)

### 2️⃣ Exploratory Data Analysis (EDA)
- **Correlation Heatmap:** Examined relationships between salary and features
- **Scatter Plots:** Analyzed Early Career Pay vs. Mid-Career Pay
- **Boxplots:** Visualized salary distributions across majors

### 3️⃣ Machine Learning Models
- **Linear Regression**
  - `R² Score`: 0.95
  - `Mean Squared Error`: 30227082.54
- **Random Forest Regressor** (Improved Model)
  - `R² Score`: 1.00
  - `Mean Squared Error`: 72307.23

## 🛠 How to Run the Project
1. **Clone the repository:**
   ```sh
   git clone https://github.com/kersha0530/applied-ml-ahsrek.git
   ```
2. **Navigate to the project folder:**
   ```sh
   cd applied-ml-kersha/ml01
   ```
3. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate  # Windows
   ```
4. **Install required dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
5. **Run the Python script:**
   ```sh
   python applied-ml-kersha.py
   ```
6. **Open the Jupyter Notebook (Optional):**
   ```sh
   jupyter notebook ml01/applied-ml-kersha.ipynb
   ```

## 📌 Key Takeaways
✅ **Early Career Pay** strongly correlates with **Mid-Career Pay**
✅ Certain **majors** lead to significantly higher earnings
✅ **Random Forest** outperformed **Linear Regression** for salary predictions

## 📌 Next Steps
🔹 Try **more advanced ML models** (e.g., Gradient Boosting, XGBoost)
🔹 Explore **external datasets** to enrich analysis
🔹 Deploy the model as a **web app** for salary predictions

📧 **Contact:** kbroussard321@gmail.com  
🔗 **GitHub Repository:** [applied-ml-kersha](https://github.com/kersha0530/applied-ml-ahsrek)

