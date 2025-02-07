import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("Data/flightdata.csv", dtype={
    'from_airport_code': 'category',
    'dest_airport_code': 'category',
    'from_country': 'category',
    'dest_country': 'category',
    'airline_name': 'category',
    'aircraft_type': 'category',
    'currency': 'category',
    'price': np.float64,  # Ensure price is numeric
    'co2_percentage': 'str',  # Keep as string initially for cleaning
    'departure_time': 'str',  # Keep as string to process later
    'arrival_time': 'str',
    'scan_date': 'str'
})

# Drop columns that aren't needed for the model
df.drop(columns=["from_country", "dest_country", "currency", "scan_date"], inplace=True)

# Clean 'aircraft_type' and similar columns, splitting any combined categories
df["aircraft_type"] = df["aircraft_type"].str.split('|').str[0]

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Label Encoding for categorical columns efficiently
categorical_columns = ["from_airport_code", "dest_airport_code", "airline_name", "aircraft_type"]
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Clean 'co2_percentage' and ensure numeric types
df["co2_percentage"] = df["co2_percentage"].replace({"None%": None, "nan": None, "": None, "None": None})
df["co2_percentage"] = df["co2_percentage"].str.replace('%', '', regex=True).astype(float)
df["co2_percentage"].fillna(df["co2_percentage"].median())  # Fixed FutureWarning

# Convert datetime columns to timestamps
df["departure_time"] = pd.to_datetime(df["departure_time"], errors='coerce').astype('int64') // 10 ** 9
df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors='coerce').astype('int64') // 10 ** 9

# Handle missing values for 'price' column directly
df["price"] = df["price"].fillna(df["price"].median())

# Handle outliers using IQR for the price column
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1
df = df[(df["price"] >= Q1 - 1.5 * IQR) & (df["price"] <= Q3 + 1.5 * IQR)]

# Separate features (X) and target variable (y)
X = df.drop(columns=["price"])  # Features
y = df["price"]  # Target variable

# Handle missing values in X using SimpleImputer
numeric_columns = X.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='median')
X[numeric_columns] = imputer.fit_transform(X[numeric_columns])

# Apply imputation for categorical columns (if any)
non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns
categorical_imputer = SimpleImputer(strategy='most_frequent')
X[non_numeric_columns] = categorical_imputer.fit_transform(X[non_numeric_columns])

# Replace any non-numeric values in X (like 'multi') with NaN and then impute them
X = X.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric (non-numeric becomes NaN)
X.fillna(X.median(), inplace=True)  # Impute NaN values with the column median

# **Fixing large values and infinities**
# Clip large values to a reasonable range (e.g., 1e10)
X = X.clip(-1e10, 1e10)

# Handle NaN values in target variable 'y'
y = y.fillna(y.median())  # Impute the target variable if any NaN exists

# **Fix any infinity or large values in y**
y = np.clip(y, -1e10, 1e10)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare a dictionary to store results
results = {}

# --- RandomForest ---
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
results["Random Forest"] = {"MAE": mae_rf, "RMSE": rmse_rf, "R²": r2_rf}


# --- XGBoost ---
xgb_model = XGBRegressor(n_estimators=50, random_state=42, tree_method='hist', n_jobs=-1)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)
results["XGBoost"] = {"MAE": mae_xgb, "RMSE": rmse_xgb, "R²": r2_xgb}

# --- LightGBM ---
lgb_model = lgb.LGBMRegressor(n_estimators=50, random_state=42, device='gpu', n_jobs=-1)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
r2_lgb = r2_score(y_test, y_pred_lgb)
results["LightGBM"] = {"MAE": mae_lgb, "RMSE": rmse_lgb, "R²": r2_lgb}

# --- CatBoost ---
cb_model = cb.CatBoostRegressor(iterations=50, depth=6, learning_rate=0.1, verbose=False, task_type='CPU')
cb_model.fit(X_train, y_train)
y_pred_cb = cb_model.predict(X_test)
mae_cb = mean_absolute_error(y_test, y_pred_cb)
rmse_cb = np.sqrt(mean_squared_error(y_test, y_pred_cb))
r2_cb = r2_score(y_test, y_pred_cb)
results["CatBoost"] = {"MAE": mae_cb, "RMSE": rmse_cb, "R²": r2_cb}

# Save models
joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(xgb_model, 'xgboost_model.pkl')
joblib.dump(lgb_model, 'lightgbm_model.pkl')
joblib.dump(cb_model, 'catboost_model.pkl')

# Convert the results dictionary to a DataFrame
results_df = pd.DataFrame(results).T  # Transpose so models are rows
results_df = results_df.reset_index()  # Reset index to create a column for models
results_df = results_df.melt(id_vars="index", var_name="Metric", value_name="Score")  # Reshape for plotting

# Plotting
plt.figure(figsize=(12, 8))

# Define colors for each metric
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue for MAE, Orange for RMSE, Green for R²

# Bar width
bar_width = 0.25
# Positioning of bars
position = range(len(results_df['index'].unique()))

# Plot bars for each metric
for i, metric in enumerate(results_df['Metric'].unique()):
    data = results_df[results_df['Metric'] == metric]
    plt.bar([p + i * bar_width for p in position], data['Score'], width=bar_width, label=metric, color=colors[i])

# Customize the plot
plt.title("Model Performance Comparison", fontsize=16)
plt.xlabel("Algorithms", fontsize=14)
plt.ylabel("Score", fontsize=14)
plt.xticks([p + bar_width for p in position], results_df['index'].unique(), rotation=45, fontsize=12)
plt.legend(title="Metrics", loc='upper left')

# Add value annotations to bars
for i, metric in enumerate(results_df['Metric'].unique()):
    for j, value in enumerate(results_df[results_df['Metric'] == metric]['Score']):
        plt.text(position[j] + i * bar_width, value + 10, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

# Tight layout to prevent clipping
plt.tight_layout()

# Save the plot to a PNG file
plt.savefig("results_comparison.png")
plt.close()

print("Model comparison plot saved as model_comparison.png")
