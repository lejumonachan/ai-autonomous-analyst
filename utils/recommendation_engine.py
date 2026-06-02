import pandas as pd


def generate_experiment_recommendations(df):
    recommendations = []

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        return [
            "Upload numeric telemetry data to generate scientific experiment recommendations."
        ]

    for col in numeric_cols:
        mean_value = df[col].mean()
        min_value = df[col].min()
        max_value = df[col].max()

        recommendations.append(
            f"For {col}, observed range is {round(min_value, 2)} to {round(max_value, 2)}. "
            f"Recommended baseline control value: {round(mean_value, 2)}."
        )

    recommendations.append(
        "Maintain stable environmental conditions before initiating high-risk biological growth experiments."
    )

    recommendations.append(
        "Use anomaly detection results before adjusting experiment parameters."
    )

    recommendations.append(
        "Run multiple mission simulations before finalizing biological experiment conditions."
    )

    return recommendations