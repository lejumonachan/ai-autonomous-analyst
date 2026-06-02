import pandas as pd


def profile_experiment_data(df):
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "data_types": df.dtypes.astype(str).to_dict(),
        "numeric_summary": df.describe().to_dict()
        if len(df.select_dtypes(include="number").columns) > 0
        else {}
    }


def create_profile_text(df):
    profile = profile_experiment_data(df)

    return f"""
Experiment Dataset Profile

Shape:
{profile["shape"]}

Columns:
{profile["columns"]}

Missing Values:
{profile["missing_values"]}

Duplicate Rows:
{profile["duplicates"]}

Data Types:
{profile["data_types"]}
"""