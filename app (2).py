import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# 🔹 Load dataset for feature reference
df = pd.read_csv("Incident_Event_Log.csv")

# 🔹 Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 🔹 Define Features (X) and Target (y)
target_col = "impact"  # Change if needed
X = df.drop(columns=[target_col])
y = df[target_col]

# 🔹 Train a new Decision Tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# 🔹 Streamlit UI
st.title("Incident Impact Prediction")

# Generate input fields dynamically based on dataset columns
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"Enter {col}", min_value=0, value=int(df[col].mean()))

# Convert input into DataFrame
input_df = pd.DataFrame([input_data])

# Encode categorical variables
for col in label_encoders:
    if col in input_df:
        input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

# Predict Impact
if st.button("Predict Impact"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Impact: {prediction[0]}")
