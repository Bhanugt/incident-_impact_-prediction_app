import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.tree import DecisionTreeClassifier

# Load saved model and preprocessing tools
model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Streamlit UI
st.title("Incident Impact Prediction App")
st.write("Enter details to predict the incident impact.")

# Load sample dataset to get feature names
df = pd.read_csv("data.csv")
features = df.drop(columns=["impact"]).columns.tolist()

# Create input fields dynamically
user_input = {}
for feature in features:
    user_input[feature] = st.text_input(f"Enter {feature}")

if st.button("Predict"):
    try:
        # Convert input into DataFrame
        input_df = pd.DataFrame([user_input])

        # Encode categorical features
        for col, le in label_encoders.items():
            if col in input_df:
                input_df[col] = le.transform(input_df[col])

        # Scale and select features
        input_scaled = scaler.transform(input_df)
        input_selected = selector.transform(input_scaled)

        # Make prediction
        prediction = model.predict(input_selected)

        st.success(f"Predicted Impact: {prediction[0]}")

    except Exception as e:
        st.error(f"Error: {e}")
