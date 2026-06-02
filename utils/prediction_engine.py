import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error


def detect_problem_type(y):
    if y.dtype == "object" or y.nunique() <= 10:
        return "classification"
    return "regression"


def train_prediction_model(df, target_column):
    df = df.copy()
    df = df.dropna(subset=[target_column])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    problem_type = detect_problem_type(y)
    target_encoder = None

    if problem_type == "classification":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    if problem_type == "classification":
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    if problem_type == "classification":
        metrics = {
            "Accuracy": accuracy_score(y_test, predictions),
            "F1 Score": f1_score(y_test, predictions, average="weighted")
        }
    else:
        metrics = {
            "R2 Score": r2_score(y_test, predictions),
            "MAE": mean_absolute_error(y_test, predictions)
        }

    return {
        "model": pipeline,
        "problem_type": problem_type,
        "metrics": metrics,
        "predictions": predictions[:20],
        "actual": y_test[:20],
        "feature_columns": X.columns.tolist(),
        "X_sample": X,
        "target_encoder": target_encoder
    }


def predict_single_input(model, input_data, feature_columns, target_encoder=None):
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    prediction = model.predict(input_df)[0]

    if target_encoder is not None:
        prediction = target_encoder.inverse_transform([int(prediction)])[0]

    return prediction