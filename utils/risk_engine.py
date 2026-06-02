import pandas as pd


def calculate_mission_risk(df):
    score = 0
    reasons = []

    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    if missing > 0:
        score += min(20, missing * 0.5)
        reasons.append(f"Missing values detected: {missing}")

    if duplicates > 0:
        score += min(10, duplicates * 0.3)
        reasons.append(f"Duplicate records detected: {duplicates}")

    numeric_df = df.select_dtypes(include="number")

    for col in numeric_df.columns:
        std = numeric_df[col].std()

        if pd.notna(std) and std > numeric_df[col].mean() * 0.8:
            score += 8
            reasons.append(f"High variability detected in {col}")

    score = min(100, score)

    if score < 30:
        level = "LOW"
    elif score < 65:
        level = "MODERATE"
    else:
        level = "CRITICAL"

    return {
        "risk_score": round(score, 2),
        "risk_level": level,
        "reasons": reasons if reasons else ["Mission data appears stable."]
    }