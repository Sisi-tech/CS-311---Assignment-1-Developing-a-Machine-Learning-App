"""
InsureRate Predictor
Predicts future insurance premiums based on historical policy and claim data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


# 1. Load Dataset
data = pd.read_csv("insurance_data.csv")


# 2. Explore Dataset
print("First 5 rows of dataset:\n", data.head())
print("\nDataset info:\n")
print(data.info())


# 3. Preprocessing
numerical_features = ['Age', 'Claim_History', 'Loss_Ratio']
categorical_features = ['Vehicle_Type', 'Region', 'Policy_Type']

# Separate features and target
X = data[numerical_features + categorical_features]
y = data['Premium']  # Target: future premium

# Preprocessing: scaling numerical, encoding categorical
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Transform features
X_processed = preprocessor.fit_transform(X)

# 4. Split into Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)


# 5. Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# 6. Model Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel RMSE: {rmse:.2f}")
print(f"Model R²: {r2:.2f}")


# 7. Save Model and Preprocessor
joblib.dump(model, 'insure_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
print("\nModel and preprocessor saved successfully!")


# 8. Predict New Policy Premium
def predict_premium(new_data: dict):
    """
    new_data: dictionary with keys as feature names
    Example:
    new_policy = {
        'Age': 35,
        'Claim_History': 1,
        'Loss_Ratio': 0.2,
        'Vehicle_Type': 'Sedan',
        'Region': 'East',
        'Policy_Type': 'Comprehensive'
    }
    """
    df = pd.DataFrame([new_data])
    X_new = preprocessor.transform(df)
    prediction = model.predict(X_new)
    return round(prediction[0], 2)


# Example prediction
new_policy = {
    'Age': 30,
    'Claim_History': 2,
    'Loss_Ratio': 0.25,
    'Vehicle_Type': 'SUV',
    'Region': 'West',
    'Policy_Type': 'Comprehensive'
}

predicted_premium = predict_premium(new_policy)
print(f"\nPredicted Premium for New Policy: ${predicted_premium}")