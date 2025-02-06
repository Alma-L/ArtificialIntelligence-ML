import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from scipy.stats import randint

# Load dataset (You may want to load only a sample for faster experimentation)
df = pd.read_csv("Data/flightdata.csv", dtype={
    'from_airport_code': 'category',
    'dest_airport_code': 'category',
    'from_country': 'category',
    'dest_country': 'category',
    'airline_name': 'category',
    'aircraft_type': 'category',
    'currency': 'category'
})

# Step 1: Clean 'aircraft_type' (and similar columns)
df["aircraft_type"] = df["aircraft_type"].str.split('|').str[0]

# Step 2: Remove duplicate rows
df.drop_duplicates(inplace=True)

# Step 3: Label Encoding for categorical columns efficiently
categorical_columns = ["from_airport_code", "dest_airport_code", "from_country", "dest_country",
                       "airline_name", "aircraft_type", "currency"]
for col in categorical_columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))  # Label encode all columns

# Step 4: Handle 'co2_percentage' column and ensure numeric types
df["co2_percentage"] = df["co2_percentage"].replace({"None%": None, "nan": None, "": None, "None": None})
df["co2_percentage"] = df["co2_percentage"].str.replace('%', '', regex=True).astype(float)
df["co2_percentage"] = df["co2_percentage"].fillna(df["co2_percentage"].median())  # Avoid inplace

# Step 5: Convert datetime columns to timestamp (seconds since the epoch)
df["departure_time"] = pd.to_datetime(df["departure_time"], errors='coerce')
df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors='coerce')
df["scan_date"] = pd.to_datetime(df["scan_date"], errors='coerce')

df["departure_time"] = df["departure_time"].astype('int64') // 10 ** 9
df["arrival_time"] = df["arrival_time"].astype('int64') // 10 ** 9
df["scan_date"] = df["scan_date"].astype('int64') // 10 ** 9

# Step 6: Handle missing values and outliers
# Fill missing values in numeric columns
df.fillna(df.median(numeric_only=True), inplace=True)

# Handle outliers using IQR for 'price'
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1
df = df[(df["price"] >= Q1 - 1.5 * IQR) & (df["price"] <= Q3 + 1.5 * IQR)]

# Step 7: Prepare features (X) and target variable (y)
X = df.drop(columns=["price"])  # Drop the target column
y = df["price"]

# Convert all columns to numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Step 8: Replace infinite values with NaN and fill them with median
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# Step 9: Handle large values (cap them at 1e10 or smaller than -1e10)
X[X > 1e10] = 1e10
X[X < -1e10] = -1e10

# Step 10: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 11: Hyperparameter tuning using RandomizedSearchCV for Random Forest
param_grid = {
    'n_estimators': randint(100, 200),  # Reduced search space
    'max_depth': [10, 20, None],
    'min_samples_split': randint(2, 6),  # Reduced search space
    'min_samples_leaf': randint(1, 4),
    'bootstrap': [True, False]
}

rf = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=3, cv=2, n_jobs=-1, verbose=2)  # Reduced n_iter and cv

# Handle cases where fits fail
try:
    random_search.fit(X_train, y_train)
except ValueError as e:
    print(f"ValueError: {e}. Check if there are any non-numeric or infinite values in your data.")
    print("Cleaning the dataset and re-running the fitting process.")
    # You can add any additional handling or retry logic here

# Get the best parameters and model
best_rf_model = random_search.best_estimator_

# Step 12: Define models to train (with tuned RandomForest)
models = {
    "Random Forest": best_rf_model,
    "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42),
    "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42),
    "CatBoost": cb.CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, verbose=False)
}

# Step 13: Train and evaluate models
results = {}
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MAE": mae, "RMSE": rmse, "R²": r2}
        print(f"🔹 {name} Performance: MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    except ValueError as e:
        print(f"Error during fitting model {name}: {e}")

# Step 14: Convert results to DataFrame for visualization
results_df = pd.DataFrame(results).T
results_df = results_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
results_df.rename(columns={"index": "Model"}, inplace=True)

# Step 15: Visualization of Model Performance
fig, ax = plt.subplots(figsize=(10, 6))

# Create a bar plot using matplotlib
for metric in results_df['Metric'].unique():
    subset = results_df[results_df['Metric'] == metric]
    ax.bar(subset['Model'], subset['Score'], label=metric)

ax.set_title("Model Performance Comparison")
ax.set_xlabel("Algorithms")
ax.set_ylabel("Score")
ax.legend(title="Metrics")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 16: Save the best model (RandomForest)
joblib.dump(best_rf_model, 'best_rf_model.pkl')
