🤖 No-Code ML Model Training System

A fully interactive Streamlit-based No-Code Machine Learning Platform that allows users to upload a dataset, clean it, preprocess it, configure ML models, train them, evaluate them, and optionally download the cleaned dataset — all without writing a single line of code.

This project includes a beautiful custom UI, automatic missing value handling, one-click ML model training, and accuracy evaluation.

📂 Project Structure

No Code ML Model Training System/
│── main.py               # Main Streamlit application
│── ml_utility.py         # Data reading, preprocessing, model training utilities
│── styles.css            # Custom theme & UI styling (dark violet/aqua theme)
│── requirements.txt      # Required Python libraries
│── Images/               # (Optional) App screenshots

✨ Key Features
📂 1. Upload Any Dataset
── Accepts CSV, XLS, XLSX
── Auto-detects file type
── Displays first preview
── Shows dataset structure

📊 2. Dataset Overview & Cleaning
── The app automatically shows:
    ── Total rows
    ── Total columns
    ── Duplicate rows
    ── Total missing values

── Missing Value Handling:
    ── User chooses how to fill missing values:
    ── Mean
    ── Median
    ── Mode

── Summary table shows:
    ── Column name
    ── Method used
    ── Fill value

── Dataset Cleaning:
    ── Removes duplicate rows
    ── User can preview cleaned dataset

── Download:
    ── One-click download of cleaned dataset (dataset_cleaned.csv)

⚙️ 3. ML Model Configuration

After cleaning, users can configure:

Target Column
── Choose any column as the prediction label.
  
Model Selection
Available ML Models:
── Logistic Regression
── Support Vector Classifier (SVC)
── Random Forest
── Decision Tree
── KNN
── Gradient Boosting
── XGBoost

Feature Scaling
── Standard Scaler
── MinMax Scaler

Train/Test Split
── Slider for choosing split ratio (60%–90% training)

Random State
── Ensures reproducibility

🚀 4. Train & Evaluate ML Models
With one click:
── Data is preprocessed:
    ── One-hot encoding for categoricals
    ── Label encoding for target (if needed)
    ── Scaling using Standard/MinMax
── Train-test split
── Model fit & evaluation
── Final accuracy displayed as metric card

Output includes:
── Accuracy score (%)
── Model details
── Summary of configuration used

🎨 UI & Theme
── Custom styles.css creates a modern neon-violet theme:
── Gradient background
── Aqua glowing headings
── Styled buttons
── CSS-polished input fields
── Animated hover effects
── Branded metric cards
── Clean dark-themed tables

🛠 Installation
Install dependencies
  ── pip install -r requirements.txt

▶️ Run the Application
streamlit run main.py

📘 How to Use the App
Step 1: Upload a CSV/XLS/XLSX file
→ Preview loads automatically.

Step 2: Clean the data
→ Fill missing values per-column.
→ Review the cleaned dataset.
→ Download cleaned file if needed.

Step 3: Configure ML model
→ Select target column, ML model, scaler, split ratio.

Step 4: Train Model
→ Click Train Model and Evaluate
→ See accuracy in real-time.

📦 Requirements
As found in requirements.txt:
streamlit==1.32.2
streamlit-option-menu==0.3.12
pandas>=1.3.0
openpyxl==3.1.2
xlrd==2.0.1
scikit-learn==1.4.1.post1
xgboost==2.0.3

