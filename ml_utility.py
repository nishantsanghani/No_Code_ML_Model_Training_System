import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score


# -------------------- Step 1: Read Data --------------------
def read_data(file_name):
    """
    Read CSV or Excel file from 'data' folder.
    Returns a pandas DataFrame.
    """
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(working_dir)
    file_path = os.path.join(parent_dir, "data", file_name)

    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


# -------------------- Step 2: Preprocess Data --------------------
def preprocess_data(df, target_column, scaler_type='standard', test_size=0.2, random_state=42):
    """
    Preprocess data:
    - Drop missing target rows
    - Split train/test
    - Impute missing values
    - One-hot encode categorical features
    - Scale numeric features
    - Remove low-variance features
    Returns X_train, X_test, y_train, y_test
    """
    try:
        # Drop rows with missing target
        df = df.dropna(subset=[target_column])

        # Split features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Stratify if target has few unique classes (classification)
        stratify_param = y if y.nunique() < 20 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )

        # Identify numerical and categorical columns
        numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        # ---- Numeric: Impute & Scale ----
        if numerical_cols:
            num_imputer = SimpleImputer(strategy='mean')
            X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
            X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

            scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
            X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
            X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

        # ---- Categorical: Impute & One-Hot Encode ----
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
            X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
            X_test_encoded = encoder.transform(X_test[categorical_cols])

            feature_names = encoder.get_feature_names_out(categorical_cols)
            X_train_encoded = pd.DataFrame(X_train_encoded, columns=feature_names, index=X_train.index)
            X_test_encoded = pd.DataFrame(X_test_encoded, columns=feature_names, index=X_test.index)

            X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded], axis=1)
            X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_encoded], axis=1)

        # ---- Remove Low-Variance Features ----
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(X_train)
        selected_features = X_train.columns[selector.get_support()]

        X_train = pd.DataFrame(selector.transform(X_train), columns=selected_features, index=X_train.index)
        X_test = pd.DataFrame(selector.transform(X_test), columns=selected_features, index=X_test.index)

        return X_train, X_test, y_train, y_test

    except Exception as e:
        raise ValueError(f"Error in preprocessing: {e}")


# -------------------- Step 3: Train Model --------------------
def train_model(X_train, y_train, model):
    """
    Train the ML model on training data.
    Returns the trained model.
    """
    try:
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise ValueError(f"Error in training: {e}")


# -------------------- Step 4: Evaluate Model --------------------
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model accuracy on test data.
    Returns float between 0-1.
    """
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return float(accuracy)
    except Exception as e:
        raise ValueError(f"Error in evaluation: {e}")
