import pandas as pd


# -----------------------------
# CLEAN BASIC DATA
# -----------------------------
def clean_data(df):
    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Clean column names
    df.columns = df.columns.str.strip()

    # Strip string values
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype(str).str.strip()

    return df


# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
def handle_missing(df, strategy="mean"):

    df = df.copy()

    for col in df.columns:

        # Try converting to numeric
        numeric_col = pd.to_numeric(df[col], errors='coerce')

        # If numeric column
        if numeric_col.notna().sum() > 0:

            df[col] = numeric_col

            if strategy == "median":
                fill_value = df[col].median()
            else:
                fill_value = df[col].mean()

            df[col] = df[col].fillna(fill_value)

        else:
            # Categorical column
            df[col] = df[col].astype(str)

            if df[col].mode().empty:
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    return df


# -----------------------------
# ENCODE DATA
# -----------------------------
def encode_data(df):
    return pd.get_dummies(df, drop_first=True)


# -----------------------------
# DETECT PROBLEM TYPE
# -----------------------------
def detect_problem_type(y):
    if y.nunique() < 10:
        return "classification"
    return "regression"