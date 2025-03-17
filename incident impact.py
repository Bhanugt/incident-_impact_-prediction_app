import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")  # Ensure this file is uploaded in Streamlit cloud
    return df

df = load_data()

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define features and target
target_col = "impact"
X = df.drop(columns=[target_col], errors="ignore")
y = df[target_col]

# Remove low-variance features
constant_filter = VarianceThreshold(threshold=0.01)
X = constant_filter.fit_transform(X)

# Standardize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Feature selection using SelectKBest
selector = SelectKBest(f_classif, k=10)
X = selector.fit_transform(X, y)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train Decision Tree Model
dt = DecisionTreeClassifier(class_weight='balanced', random_state=42)
dt.fit(X_resampled, y_resampled)

# Save the trained model
joblib.dump(dt, "decision_tree_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selector, "selector.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

# Streamlit UI
st.title("Decision Tree Classifier App")
st.write("Enter feature values to predict the impact class.")

# Collect user inputs
user_input = {}
for col in df.columns[:-1]:  # Excluding target variable
    user_input[col] = st.text_input(f"Enter value for {col}")

if st.button("Predict"):
    try:
        # Convert input into DataFrame
        input_df = pd.DataFrame([user_input])

        # Encode categorical variables
        for col, le in label_encoders.items():
            if col in input_df:
                input_df[col] = le.transform(input_df[col])

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Select top 10 features
        input_selected = selector.transform(input_scaled)

        # Load trained model
        model = joblib.load("decision_tree_model.pkl")

        # Make prediction
        prediction = model.predict(input_selected)
        st.success(f"Predicted Impact: {prediction[0]}")

    except Exception as e:
        st.error(f"Error: {e}")
