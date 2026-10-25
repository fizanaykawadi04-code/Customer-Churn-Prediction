
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)

# Load dataset
df = pd.read_csv(
    "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

df = df.drop(columns=["customerID"], errors="ignore")
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

valid = y.notna()
X, y = X.loc[valid], y.loc[valid].astype(int)

numeric_cols = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = X.select_dtypes(
    exclude="number"
).columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), numeric_cols),
    ("categorical", Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
}

results = []

for name, classifier in models.items():
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy (%)": round(
            accuracy_score(y_test, predictions) * 100, 2
        ),
        "Precision (%)": round(
            precision_score(y_test, predictions,
                            zero_division=0) * 100, 2
        ),
        "Recall (%)": round(
            recall_score(y_test, predictions,
                         zero_division=0) * 100, 2
        ),
        "F1 Score (%)": round(
            f1_score(y_test, predictions,
                     zero_division=0) * 100, 2
        )
    })

    if name == "Random Forest":
        joblib.dump(model, "random_forest_model.pkl")

results_df = pd.DataFrame(results)
print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)
print("\nComparison saved to model_comparison.csv")
print("Random Forest model saved successfully!")
