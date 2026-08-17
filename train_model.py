import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RANDOM_STATE = 42


# 1. Load data
df = pd.read_csv("Laptop_price.csv")
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

# 2. Quick EDA
print("\nSummary stats:\n", df.describe())
print("\nCorrelation with Price:\n", df.corr(numeric_only=True)["Price"].sort_values(ascending=False))
print("\nBrand counts:\n", df["Brand"].value_counts())

# 3. Split features/target
X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

categorical_cols = ["Brand"]
numeric_cols = ["Processor_Speed", "RAM_Size", "Storage_Capacity", "Screen_Size", "Weight"]

# 4. Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])

# 5. Compare candidate models
candidates = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
}

results = []
fitted_pipelines = {}

for name, model in candidates.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()

    results.append({
        "model": name, "test_r2": r2, "test_mae": mae,
        "test_rmse": rmse, "cv_r2_mean": cv_r2,
    })
    fitted_pipelines[name] = pipe

results_df = pd.DataFrame(results).sort_values("test_r2", ascending=False)
print("\nModel comparison:\n", results_df.to_string(index=False))

# 6. Pick best model and save it
best_name = results_df.iloc[0]["model"]
best_pipeline = fitted_pipelines[best_name]
print(f"\nBest model: {best_name}")

joblib.dump(best_pipeline, "model.pkl")
print("Saved trained pipeline to model.pkl")

# Save reference values for the app's input widgets (min/max/options)
meta = {
    "brands": sorted(df["Brand"].unique().tolist()),
    "ranges": {
        col: (float(df[col].min()), float(df[col].max())) for col in numeric_cols
    },
    "best_model": best_name,
    "test_r2": float(results_df.iloc[0]["test_r2"]),
    "test_mae": float(results_df.iloc[0]["test_mae"]),
}
joblib.dump(meta, "meta.pkl")
print("Saved input metadata to meta.pkl")
