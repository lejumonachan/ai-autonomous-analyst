import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import clean_data, handle_missing
from utils.model_selector import run_ml_pipeline, predict_user_input
from utils.llm_engine import ask_llm

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Data Platform",
    page_icon="🚀",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #1e3a8a 100%);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* File uploader card */
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, #ffffff 0%, #dbeafe 100%);
    padding: 18px;
    border-radius: 18px;
    border: 2px dashed #60a5fa;
    box-shadow: 0px 8px 25px rgba(37, 99, 235, 0.25);
}

/* Upload button */
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #2563eb, #06b6d4) !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 700 !important;
}

/* Uploaded file text */
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] span {
    color: #0f172a !important;
}

/* Main spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Hero card */
.hero-card {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    padding: 35px;
    border-radius: 24px;
    color: white;
    box-shadow: 0px 20px 40px rgba(37, 99, 235, 0.25);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 18px;
    opacity: 0.95;
}

/* Metric cards */
.metric-card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 8px 24px rgba(15, 23, 42, 0.08);
    border: 1px solid #e5e7eb;
}

.metric-label {
    font-size: 14px;
    color: #64748b;
    font-weight: 600;
}

.metric-value {
    font-size: 32px;
    font-weight: 800;
    color: #0f172a;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.3rem;
    font-weight: 700;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #6d28d9);
    color: white;
}

.stDownloadButton > button {
    background: #0f172a;
    color: white;
    border-radius: 12px;
    border: none;
    font-weight: 700;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("##  AI Data Platform")
st.sidebar.markdown("Enterprise AutoML & Analytics")

menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Cleaning", "Modeling", "AI Insights"]
)

st.sidebar.markdown("### 📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")

st.sidebar.markdown("## 👨‍💻 Developer")

st.sidebar.markdown("### Leju Monachan")

st.sidebar.markdown(
    "[🔗 LinkedIn](https://www.linkedin.com/in/leju-monachan757/)"
)

st.sidebar.markdown(
    "[💻 GitHub](https://github.com/lejumonachan)"
)

st.sidebar.markdown(
    "[🌐 Live Demo](https://ai-autonomous-analyst-rbxj8yagslvp4ojp9mbagn.streamlit.app/)"
)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero-card">
    <div class="hero-title">AI Data Intelligence Platform</div>
    <div class="hero-subtitle">
        Upload data, clean it, visualize patterns, train models, and generate AI-powered insights.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# MAIN APP
# =========================
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # =========================
    # DASHBOARD
    # =========================
    if menu == "Dashboard":

        st.markdown("##  Executive Dashboard")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Rows</div>
                <div class="metric-value">{df.shape[0]}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Columns</div>
                <div class="metric-value">{df.shape[1]}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Missing Values</div>
                <div class="metric-value">{df.isnull().sum().sum()}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📄 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        all_cols = df.columns.tolist()

        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for visualization.")
        else:
            st.markdown("###  Interactive Visual Analytics")

            chart_tabs = st.tabs([
                "Bar Chart",
                "Line Chart",
                "Pie Chart",
                "Histogram",
                "Box Plot",
                "Scatter Plot",
                "Heatmap"
            ])

            with chart_tabs[0]:
                c1, c2 = st.columns(2)
                x_bar = c1.selectbox("X-axis", all_cols, key="bar_x")
                y_bar = c2.selectbox("Y-axis", numeric_cols, key="bar_y")
                color_bar = st.selectbox("Color Group", [None] + all_cols, key="bar_color")
                fig = px.bar(df, x=x_bar, y=y_bar, color=color_bar, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            with chart_tabs[1]:
                c1, c2 = st.columns(2)
                x_line = c1.selectbox("X-axis", all_cols, key="line_x")
                y_line = c2.selectbox("Y-axis", numeric_cols, key="line_y")
                color_line = st.selectbox("Color Group", [None] + all_cols, key="line_color")
                fig = px.line(df, x=x_line, y=y_line, color=color_line, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            with chart_tabs[2]:
                pie_col = st.selectbox("Category Column", all_cols, key="pie_col")
                pie_data = df[pie_col].value_counts().reset_index()
                pie_data.columns = [pie_col, "Count"]
                fig = px.pie(pie_data, names=pie_col, values="Count", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            with chart_tabs[3]:
                hist_col = st.selectbox("Histogram Column", numeric_cols, key="hist_col")
                bins = st.slider("Bins", 10, 100, 30)
                fig = px.histogram(df, x=hist_col, nbins=bins, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            with chart_tabs[4]:
                c1, c2 = st.columns(2)
                x_box = c1.selectbox("X-axis Optional", [None] + all_cols, key="box_x")
                y_box = c2.selectbox("Y-axis", numeric_cols, key="box_y")
                fig = px.box(df, x=x_box, y=y_box, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            with chart_tabs[5]:
                c1, c2 = st.columns(2)
                x_scatter = c1.selectbox("X-axis", numeric_cols, key="scatter_x")
                y_scatter = c2.selectbox("Y-axis", numeric_cols, key="scatter_y")
                color_scatter = st.selectbox("Color Group", [None] + all_cols, key="scatter_color")
                fig = px.scatter(df, x=x_scatter, y=y_scatter, color=color_scatter, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            with chart_tabs[6]:
                if len(numeric_cols) > 1:
                    corr = df[numeric_cols].corr()
                    fig = px.imshow(corr, text_auto=True, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 2 numeric columns for heatmap.")

    # =========================
    # CLEANING
    # =========================
    elif menu == "Cleaning":

        st.markdown("##  Data Cleaning Studio")

        st.markdown("### Missing Values Before Cleaning")
        st.dataframe(df.isnull().sum(), use_container_width=True)

        strategy = st.selectbox("Numeric Missing Value Strategy", ["mean", "median"])

        if st.button("Auto Clean Data"):
            df = clean_data(df)
            df = handle_missing(df, strategy=strategy)

            st.success("✅ Data cleaned successfully")

            st.markdown("### Missing Values After Cleaning")
            st.dataframe(df.isnull().sum(), use_container_width=True)

        st.markdown("### Cleaned Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Cleaned Dataset",
            csv,
            "cleaned_data.csv",
            "text/csv"
        )

    # =========================
    # MODELING
    # =========================
    elif menu == "Modeling":

        st.markdown("##  AutoML Modeling Engine")

        target = st.selectbox("Target Column", df.columns)

        numeric_strategy = st.selectbox(
            "Numeric Missing Value Strategy",
            ["mean", "median"]
        )

        feature_k = st.selectbox(
            "Feature Selection",
            ["all", 5, 10, 15]
        )

        if "ml_output" not in st.session_state:
            st.session_state.ml_output = None

        if st.button("Run Models"):

            try:
                output = run_ml_pipeline(
                    df=df,
                    target_column=target,
                    numeric_strategy=numeric_strategy,
                    feature_k=feature_k
                )

                st.session_state.ml_output = output
                st.success("✅ ML Pipeline completed successfully")

            except Exception as e:
                st.error(f"❌ Modeling Error: {e}")

        if st.session_state.ml_output is not None:

            output = st.session_state.ml_output

            st.markdown("### Detected Problem Type")
            st.info(output["problem_type"])

            st.markdown("### Model Comparison")
            st.dataframe(output["results"], use_container_width=True)

            st.markdown("### 🏆 Best Model")
            st.success(output["best_model_name"])

            st.markdown("### 🔮 Sample Predictions")
            prediction_df = pd.DataFrame({
                "Actual": output["actual"],
                "Predicted": output["predictions"]
            })
            st.dataframe(prediction_df, use_container_width=True)

            st.divider()

            st.markdown("##  Live Prediction Console")

            user_input = {}
            X_sample = output["X_sample"]
            feature_columns = output["feature_columns"]

            for col in feature_columns:

                if pd.api.types.is_numeric_dtype(X_sample[col]):
                    min_val = float(X_sample[col].min())
                    max_val = float(X_sample[col].max())
                    mean_val = float(X_sample[col].mean())

                    user_input[col] = st.number_input(
                        col,
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val
                    )

                else:
                    options = X_sample[col].dropna().astype(str).unique().tolist()
                    if len(options) == 0:
                        options = ["Unknown"]

                    user_input[col] = st.selectbox(col, options)

            if st.button("Predict Target Value"):

                try:
                    prediction = predict_user_input(
                        model=output["best_model"],
                        input_data=user_input,
                        feature_columns=feature_columns,
                        target_encoder=output["target_encoder"]
                    )

                    st.success(f"✅ Predicted {target}: {prediction}")

                except Exception as e:
                    st.error(f"❌ Prediction Error: {e}")

    # =========================
    # AI INSIGHTS
    # =========================
    elif menu == "AI Insights":

        st.markdown("##  AI Analyst")

        st.write("Generate executive-level AI insights from your dataset.")

        if st.button("Generate Insights"):

            summary = f"""
            Dataset shape: {df.shape}
            Columns: {list(df.columns)}
            Missing:
            {df.isnull().sum().to_string()}
            """

            try:
                result = ask_llm(summary)
                st.markdown(result)
            except Exception as e:
                st.error(f"LLM Error: {e}")

else:
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">Welcome to AI Data Platform</div>
        <div class="hero-subtitle">
            Upload a CSV file from the sidebar to begin your analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("Upload a CSV file to start.")
