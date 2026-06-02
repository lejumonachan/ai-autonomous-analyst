import os


def ensure_directories():

    folders = [
        "uploads",
        "reports",
        "models",
        "vector_store"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def dataframe_to_context(df, max_rows=20):

    try:
        context = df.head(max_rows).to_string()
        return context

    except Exception:
        return str(df.head(max_rows))