# model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import joblib

print("🚀 Training started...")

# Load dataset
df = pd.read_csv('real_estate.csv')
print("✅ Data loaded.")

# Drop unused columns
df.drop(columns=["availability", "society"], inplace=True, errors='ignore')

# Drop rows with missing values in essential columns
required_columns = ["location", "BHK", "total_sqft", "bathrooms", "balcony", "Corrected_Percent_Flooded_Area", "price"]
df.dropna(subset=required_columns, inplace=True)

# Convert data types safely
numeric_cols = ["BHK", "total_sqft", "bathrooms", "balcony", "Corrected_Percent_Flooded_Area", "price"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Drop rows with invalid numeric values
df.dropna(inplace=True)

# Encode 'location' using LabelEncoder
le = LabelEncoder()
df['location_encoded'] = le.fit_transform(df['location'])

# Create risk_score from flood area
df['risk_score'] = df['Corrected_Percent_Flooded_Area'] / 100.0

# Select features and target
features = ["location_encoded", "BHK", "bathrooms", "balcony", "total_sqft", "risk_score"]
X = df[features]
y = df["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ R² Score:", round(r2_score(y_test, y_pred), 2))
print("✅ RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))

# Save model and encoder
joblib.dump(model, "final_model.pkl")
joblib.dump(le, "location_label_encoder.pkl")

print("🎯 final_model.pkl and location_label_encoder.pkl saved successfully!")


