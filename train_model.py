import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("data.csv")

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

    # Add "Unknown" as a valid class
    le.classes_ = np.append(le.classes_, "Unknown")

    label_encoders[col] = le

# Define features and target
target_col = "impact"
X = df.drop(columns=[target_col], errors="ignore")
y = df[target_col]

# Remove constant/low-variance features
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

# Save trained model and preprocessing tools
joblib.dump(dt, "decision_tree_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selector, "selector.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Model and preprocessing tools saved successfully!")
