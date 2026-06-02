import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df, contamination=0.05):
    df = df.copy()
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return df, "No numeric columns available for anomaly detection."

    clean_numeric = numeric_df.fillna(numeric_df.mean())

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    predictions = model.fit_predict(clean_numeric)

    df["anomaly_flag"] = predictions
    df["anomaly_status"] = df["anomaly_flag"].apply(
        lambda x: "Anomaly" if x == -1 else "Normal"
    )

    anomaly_count = int((df["anomaly_status"] == "Anomaly").sum())

    return df, f"Detected {anomaly_count} potential mission or biological anomalies."


def rule_based_alerts(df):
    alerts = []

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()

        if std == 0 or pd.isna(std):
            continue

        upper = mean + 3 * std
        lower = mean - 3 * std

        outliers = df[(df[col] > upper) | (df[col] < lower)]

        if len(outliers) > 0:
            alerts.append(
                f"⚠️ {col}: {len(outliers)} extreme values detected outside normal mission range."
            )

    if not alerts:
        alerts.append("✅ No major rule-based anomalies detected.")

    return alerts