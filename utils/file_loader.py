import pandas as pd
import PyPDF2


def load_csv(file):
    encodings = ["utf-8", "utf-8-sig", "windows-1252", "latin1", "ISO-8859-1"]

    for encoding in encodings:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            continue

    file.seek(0)
    return pd.read_csv(file, encoding="latin1", errors="replace")


def load_excel(file):
    file.seek(0)
    return pd.read_excel(file)


def load_pdf(file):
    file.seek(0)
    text = ""

    pdf_reader = PyPDF2.PdfReader(file)

    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text


def load_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df = load_csv(uploaded_file)
        return "dataframe", df

    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        df = load_excel(uploaded_file)
        return "dataframe", df

    elif file_name.endswith(".pdf"):
        text = load_pdf(uploaded_file)
        return "pdf", text

    else:
        return None, None


def get_dataframe_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude=["number"]).columns.tolist()
    }