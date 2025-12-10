import os
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------- Streamlit Page Config ----------
st.set_page_config(page_title="Automate ML ⚡️", page_icon="🚀", layout="wide")

# ---------- Load External CSS ----------
css_file = os.path.join(os.path.dirname(__file__), "styles.css")
if os.path.exists(css_file):
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown("<h1>🤖 No-Code ML Model Training Platform ⚡️</h1>", unsafe_allow_html=True)

# ---------- Initialize DataFrame ----------
if 'df' not in st.session_state:
    st.session_state['df'] = None

# ---------- Upload Section ----------
st.markdown('<div class="section-title">📂 Upload Dataset</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a CSV or Excel file to begin the magic!", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if 'uploaded_file_name' not in st.session_state or st.session_state['uploaded_file_name'] != uploaded_file.name:
        st.session_state['uploaded_file_name'] = uploaded_file.name
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
        st.session_state['df'] = df
        st.success(f"Dataset Loaded Successfully: **{uploaded_file.name}**")
else:
    st.session_state['df'] = None
    if 'uploaded_file_name' in st.session_state:
        del st.session_state['uploaded_file_name']

df = st.session_state['df']

# ---------- Function to show Streamlit dataframe with column selection ----------
def show_dataframe(df, key_name):
    if df is None or df.empty:
        st.info("No data to display.")
        return
    all_cols = df.columns.tolist()
    selected_cols = st.multiselect(f"Select Columns to Display ({key_name})", all_cols, default=all_cols, key=f"cols_{key_name}")
    if selected_cols:
        st.dataframe(df[selected_cols], use_container_width=True)
    else:
        st.info("Select at least one column to display.")

# ---------- Dataset Summary & Cleaning ----------
if df is not None:
    st.markdown('<div class="section-title">📊 Dataset Overview & Cleaning</div>', unsafe_allow_html=True)

    st.subheader("Raw Data Preview")
    show_dataframe(df, key_name="raw")

    st.markdown("---")
    col_r, col_c, col_d, col_m = st.columns(4)
    col_r.metric("Total Rows", f"{df.shape[0]}")
    col_c.metric("Total Columns", f"{df.shape[1]}")
    col_d.metric("Duplicate Rows", f"{df.duplicated().sum()}")
    col_m.metric("Missing Values", f"{int(df.isnull().sum().sum())}")

    df_cleaned = df.drop_duplicates()
    missing_cols = [col for col in df_cleaned.columns if df_cleaned[col].isnull().any()]
    fill_info = []

    st.subheader("⚡ Missing Value Fill Summary")
    if missing_cols:
        st.markdown("Select the fill method for each column (Mean / Median / Mode):")
        cols_layout = st.columns(3)
        for idx, col in enumerate(missing_cols):
            col_widget = cols_layout[idx % 3]

            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                default_method = "Mean"
            else:
                default_method = "Mode"

            method = col_widget.selectbox(
                f"{col}",
                options=["Mean", "Median", "Mode"],
                index=["Mean", "Median", "Mode"].index(default_method),
                key=f"fill_{col}"
            )

            if method == "Mean":
                fill_value = df_cleaned[col].mean()
            elif method == "Median":
                fill_value = df_cleaned[col].median()
            else:
                fill_value = df_cleaned[col].mode()[0]

            df_cleaned[col].fillna(fill_value, inplace=True)
            fill_info.append({'Column': col, 'Fill Method': method, 'Fill Value': fill_value})

        fill_df = pd.DataFrame(fill_info)
        show_dataframe(fill_df, key_name="fill_summary")
    else:
        st.info("No missing values found; no filling required.")

    st.markdown("---")
    st.subheader("🧼 Cleaned Data Preview")
    show_dataframe(df_cleaned, key_name="cleaned")

    st.subheader("🧼 Cleaned Data Overview")
    col_r, col_c, col_miss = st.columns(3)
    col_r.metric("Rows After Cleaning", f"{df_cleaned.shape[0]}")
    col_c.metric("Columns After Cleaning", f"{df_cleaned.shape[1]}")
    col_miss.metric("Missing Values After Cleaning", f"{int(df_cleaned.isnull().sum().sum())}")
    st.markdown("<br>", unsafe_allow_html=True)

    if uploaded_file is not None:
        file_name, file_ext = os.path.splitext(uploaded_file.name)
        cleaned_file_name = f"{file_name}_cleaned{file_ext}"
        csv = df_cleaned.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"⬇ Download Cleaned Dataset ({cleaned_file_name})",
            data=csv,
            file_name=cleaned_file_name
        )

    st.session_state['df_cleaned'] = df_cleaned
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Model Functions ----------
def preprocess_data(df, target_col, scaler_type='standard', test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    X = pd.get_dummies(X, drop_first=True)

    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

# ---------- Model Configuration ----------
if df is not None:
    st.markdown('<div class="section-title">⚙ Model Configuration & Training</div>', unsafe_allow_html=True)

    model_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Classifier": SVC(probability=True, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(random_state=42),
        "XGBoost Classifier": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "K-Neighbors Classifier": KNeighborsClassifier(),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42)
    }

    df_config = st.session_state['df_cleaned']
    col1, col2, col3 = st.columns(3)
    target_column = col1.selectbox("🎯 Target Column", df_config.columns)
    selected_model = col2.selectbox("🤖 Select ML Model", list(model_dict.keys()))
    scaler_type = col3.selectbox("⚖ Feature Scaling Method", ["standard", "minmax"])

    colA, colB = st.columns(2)
    split = colA.slider("🧪 Train/Test Split Ratio (%)", 60, 90, 80)
    random_state = colB.number_input("🔁 Random State", value=42, min_value=1)

    st.markdown(
        f"""
        <div style='background-color:#1c0738; padding:10px; border-radius:10px;
        border:1px solid #6A0DAD; text-align:center; width:50%; margin-left:0;'>
            <span style='color: #00FFFF; font-weight:bold;'>📌 Data Split:</span> 
            <span style='color: #FFC300;'>{split}% Train</span> — 
            <span style='color: #FFC300;'>{100-split}% Test</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    if st.button("🚀 Train Model and Evaluate", key="train_btn", use_container_width=True):
        if target_column:
            with st.spinner(f"Training **{selected_model}**..."):
                try:
                    X_train, X_test, y_train, y_test = preprocess_data(
                        df_config,
                        target_column,
                        scaler_type,
                        test_size=(100 - split)/100,
                        random_state=int(random_state)
                    )
                    model = train_model(X_train, y_train, model_dict[selected_model])
                    acc = evaluate_model(model, X_test, y_test) * 100
                    st.success("🎉 Model Training Completed Successfully!")
                    st.markdown(f"**Selected Model:** {selected_model} | **Target:** {target_column} | **Scaler:** {scaler_type.upper()}")
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.metric("Model Test Accuracy", f"{acc:.2f}%")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please select a Target Column.")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("📌 Upload a Dataset to unlock Model Configuration & Training.")
