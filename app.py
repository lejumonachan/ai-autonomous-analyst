import streamlit as st
import pandas as pd
import plotly.express as px

from utils.file_loader import load_file, get_dataframe_summary
from utils.helpers import ensure_directories

from utils.monitoring import (
    get_monitoring_columns,
    create_time_series_chart,
    create_scatter_chart,
    create_distribution_chart,
    create_correlation_heatmap,
    calculate_health_score,
    create_health_gauge
)

from utils.rag_engine import (
    chunk_text,
    build_faiss_index,
    rag_scientific_answer
)

from utils.ai_engine import (
    scientific_answer,
    generate_scientific_summary
)

from utils.anomaly_engine import (
    detect_anomalies,
    rule_based_alerts
)

from utils.risk_engine import calculate_mission_risk

from utils.prediction_engine import (
    train_prediction_model,
    predict_single_input
)

from utils.recommendation_engine import generate_experiment_recommendations

from utils.knowledge_graph import (
    build_knowledge_graph,
    get_graph_edges
)

from utils.report_generator import generate_mission_report


ensure_directories()

st.set_page_config(
    page_title="Helios AI | Space Biolab Intelligence",
    page_icon="🧬",
    layout="wide"
)

# ======================
# PREMIUM CSS
# ======================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(34,211,238,0.18), transparent 28%),
        radial-gradient(circle at 90% 5%, rgba(168,85,247,0.22), transparent 30%),
        radial-gradient(circle at 50% 90%, rgba(20,184,166,0.12), transparent 35%),
        linear-gradient(135deg, #020617 0%, #050816 45%, #0f172a 100%);
    color: #e5e7eb;
}

.block-container {
    padding-top: 1.5rem;
    padding-left: 4rem;
    padding-right: 4rem;
    max-width: 1400px;
}

section[data-testid="stSidebar"] {
    background: rgba(2, 6, 23, 0.96);
    border-right: 1px solid rgba(34,211,238,0.16);
    box-shadow: 10px 0 40px rgba(0,0,0,0.42);
}

section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

.sidebar-brand {
    background: linear-gradient(135deg, rgba(34,211,238,0.14), rgba(124,58,237,0.18));
    border: 1px solid rgba(34,211,238,0.25);
    padding: 22px;
    border-radius: 24px;
    margin-bottom: 22px;
    box-shadow: 0 18px 45px rgba(34,211,238,0.10);
}

.sidebar-brand .logo {
    font-size: 34px;
    margin-bottom: 8px;
}

.sidebar-brand h2 {
    margin: 0;
    font-size: 25px;
    font-weight: 900;
}

.sidebar-brand p {
    margin-top: 8px;
    color: #a5f3fc !important;
    font-size: 13px;
    line-height: 1.45;
}

section[data-testid="stSidebar"] label {
    background: rgba(255,255,255,0.045);
    padding: 13px 15px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 8px;
}

section[data-testid="stSidebar"] label:hover {
    background: rgba(34,211,238,0.16);
    border: 1px solid rgba(34,211,238,0.32);
}

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 22px 26px;
    border-radius: 24px;
    background: rgba(15,23,42,0.58);
    border: 1px solid rgba(148,163,184,0.18);
    backdrop-filter: blur(18px);
    box-shadow: 0 18px 55px rgba(2,6,23,0.35);
    margin-bottom: 26px;
}

.topbar-title {
    font-size: 22px;
    font-weight: 900;
    color: white;
}

.topbar-subtitle {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 3px;
}

.status-pill {
    display: inline-block;
    padding: 9px 14px;
    border-radius: 999px;
    background: rgba(16,185,129,0.12);
    color: #86efac;
    border: 1px solid rgba(16,185,129,0.32);
    font-size: 13px;
    font-weight: 800;
}

.hero {
    position: relative;
    overflow: hidden;
    min-height: 360px;
    background:
        radial-gradient(circle at 85% 22%, rgba(34,211,238,0.35), transparent 24%),
        radial-gradient(circle at 18% 80%, rgba(168,85,247,0.28), transparent 26%),
        linear-gradient(135deg, rgba(2,6,23,0.98) 0%, rgba(15,23,42,0.92) 40%, rgba(30,64,175,0.72) 100%);
    padding: 54px;
    border-radius: 36px;
    color: white;
    box-shadow: 0 35px 90px rgba(34,211,238,0.18);
    border: 1px solid rgba(255,255,255,0.16);
    margin-bottom: 32px;
}

.hero-badge {
    display: inline-block;
    background: rgba(34,211,238,0.12);
    border: 1px solid rgba(34,211,238,0.28);
    padding: 9px 15px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 900;
    margin-bottom: 18px;
    color: #a5f3fc;
    letter-spacing: 0.5px;
}

.hero-title {
    font-size: 58px;
    font-weight: 950;
    letter-spacing: -2px;
    line-height: 1.05;
    max-width: 1000px;
}

.hero-gradient {
    background: linear-gradient(90deg, #67e8f9, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 18px;
    opacity: 0.92;
    margin-top: 20px;
    max-width: 980px;
    line-height: 1.75;
    color: #cbd5e1;
}

.section-title {
    font-size: 30px;
    font-weight: 950;
    color: #f8fafc;
    margin-bottom: 14px;
}

.glass-card {
    background: rgba(15,23,42,0.72);
    backdrop-filter: blur(18px);
    padding: 30px;
    border-radius: 28px;
    border: 1px solid rgba(148,163,184,0.22);
    box-shadow: 0 18px 55px rgba(2,6,23,0.42);
    margin-bottom: 24px;
}

.metric-card {
    position: relative;
    overflow: hidden;
    background: linear-gradient(180deg, rgba(30,41,59,0.95), rgba(15,23,42,0.98));
    padding: 26px;
    border-radius: 26px;
    border: 1px solid rgba(34,211,238,0.18);
    box-shadow: 0 16px 42px rgba(2,6,23,0.42);
    min-height: 150px;
}

.metric-card::before {
    content: "";
    position: absolute;
    height: 3px;
    left: 0;
    top: 0;
    width: 100%;
    background: linear-gradient(90deg, #06b6d4, #7c3aed);
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    color: #f8fafc;
    font-size: 30px;
    font-weight: 950;
    margin-top: 12px;
}

.metric-note {
    color: #67e8f9;
    font-size: 12px;
    margin-top: 10px;
    line-height: 1.4;
}

.feature-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 26px;
    padding: 24px;
    min-height: 190px;
}

.feature-icon {
    font-size: 30px;
    margin-bottom: 14px;
}

.feature-card h4 {
    color: #e0f2fe;
    margin-bottom: 10px;
    font-size: 18px;
}

.feature-card p {
    color: #cbd5e1;
    font-size: 14px;
    line-height: 1.6;
}

.stButton > button {
    background: linear-gradient(135deg, #06b6d4, #7c3aed);
    color: white;
    border-radius: 16px;
    border: none;
    font-weight: 900;
    padding: 0.8rem 1.5rem;
    box-shadow: 0 14px 30px rgba(6,182,212,0.25);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0891b2, #6d28d9);
    color: white;
}

[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ======================
# SIDEBAR
# ======================

st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="logo">🧬</div>
    <h2>Helios AI</h2>
    <p>Autonomous Space Biolab Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Mission Navigation",
    [
        "Mission Control",
        "Experiment Upload",
        "Biolab Monitoring",
        "AI Scientific Assistant",
        "Anomaly Detection",
        "Prediction Engine",
        "Mission Risk",
        "Experiment Recommendations",
        "Knowledge Graph",
        "Mission Report"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Developer")
st.sidebar.markdown("**Leju Monachan**")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/leju-monachan757/)")
st.sidebar.markdown("[GitHub](https://github.com/lejumonachan)")


# ======================
# TOP BAR + HERO
# ======================

st.markdown("""
<div class="topbar">
    <div>
        <div class="topbar-title">Helios AI Mission Interface</div>
        <div class="topbar-subtitle">Scientific AI • Space Biology • Autonomous Experiment Intelligence</div>
    </div>
    <div class="status-pill"> SYSTEM ONLINE</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-badge">SPACE BIOTECH • AUTONOMOUS LABS • SCIENTIFIC AI</div>
    <div class="hero-title">
        Intelligence layer for <span class="hero-gradient">space-based biological experimentation</span>
    </div>
    <div class="hero-subtitle">
        Helios AI is a premium mission-control platform for autonomous biolab monitoring,
        anomaly detection, biological prediction, scientific RAG intelligence, experiment recommendations,
        and executive mission reporting in space-like environments.
    </div>
</div>
""", unsafe_allow_html=True)


# ======================
# MISSION CONTROL
# ======================

if page == "Mission Control":

    st.markdown('<div class="section-title"> Mission Control Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        ("Experiment State", "Active", "Orbital biolab simulation"),
        ("Biolab AI", "Online", "Scientific assistant ready"),
        ("RAG System", "Ready", "Research document intelligence"),
        ("Risk Engine", "Standby", "Mission stability monitoring")
    ]

    for col, metric in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{metric[0]}</div>
                <div class="metric-value">{metric[1]}</div>
                <div class="metric-note">{metric[2]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3> Mission Objective</h3>
        <p>
            Helios AI is designed as an enterprise-style scientific AI prototype for biotechnology,
            autonomous laboratories, drug discovery, tissue engineering, and space-based experimentation.
            The platform simulates how AI can monitor biological experiments, detect anomalies,
            predict outcomes, and generate scientific recommendations remotely.
        </p>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)

    features = [
        ("", "Biolab Telemetry", "Track mission biological and environmental signals such as radiation, pH, humidity, oxygen, temperature, and growth rate."),
        ("", "Anomaly Intelligence", "Detect abnormal experiment behavior, biological deviations, unstable conditions, and mission-level operational risks."),
        ("", "Predictive Biology", "Estimate experiment success probability, growth behavior, and risk levels using machine learning models.")
    ]

    for col, item in zip([f1, f2, f3], features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{item[0]}</div>
                <h4>{item[1]}</h4>
                <p>{item[2]}</p>
            </div>
            """, unsafe_allow_html=True)


# ======================
# EXPERIMENT UPLOAD
# ======================

elif page == "Experiment Upload":

    st.markdown('<div class="section-title"> Experiment Upload</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        Upload experiment telemetry datasets, biological measurements, mission logs,
        or scientific research documents. Supported formats: CSV, Excel, and PDF.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Scientific File",
        type=["csv", "xlsx", "xls", "pdf"]
    )

    if uploaded_file:

        file_type, data = load_file(uploaded_file)

        if file_type == "dataframe":

            st.success(" Experiment dataset uploaded successfully")

            st.session_state["uploaded_data"] = data
            st.session_state["file_type"] = "dataframe"

            summary = get_dataframe_summary(data)

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Rows", summary["rows"])
            c2.metric("Columns", summary["columns"])
            c3.metric("Missing Values", summary["missing_values"])
            c4.metric("Duplicate Rows", summary["duplicate_rows"])

            st.markdown("### 📊 Experiment Dataset Preview")
            st.dataframe(data.head(25), use_container_width=True)

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("###  Numeric Columns")
                st.write(summary["numeric_columns"])

            with c2:
                st.markdown("###  Categorical Columns")
                st.write(summary["categorical_columns"])

            st.markdown("###  Statistical Summary")
            st.dataframe(data.describe(include="all"), use_container_width=True)

        elif file_type == "pdf":

            st.success(" Scientific PDF uploaded successfully")

            st.session_state["uploaded_text"] = data
            st.session_state["file_type"] = "pdf"

            if data.strip():
                try:
                    with st.spinner("Building scientific RAG index..."):
                        chunks = chunk_text(data)
                        index, stored_chunks = build_faiss_index(chunks)

                    st.session_state["rag_index"] = index
                    st.session_state["rag_chunks"] = stored_chunks

                    st.success(" Scientific RAG index created successfully")

                except Exception as e:
                    st.error(f"RAG Index Error: {e}")

            st.markdown("### 📄 Scientific Document Preview")
            st.text_area("Extracted PDF Text", data[:6000], height=350)

            st.metric("Extracted Characters", len(data))

        else:
            st.error("Unsupported file type")


# ======================
# BIOLAB MONITORING
# ======================

elif page == "Biolab Monitoring":

    st.markdown('<div class="section-title"> Biolab Monitoring</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        Autonomous mission telemetry monitoring for biological experiments,
        environmental conditions, and scientific system stability.
    </div>
    """, unsafe_allow_html=True)

    if "uploaded_data" not in st.session_state:

        st.warning("Please upload an experiment dataset first from the Experiment Upload page.")

    else:

        df = st.session_state["uploaded_data"]

        numeric_cols, all_cols = get_monitoring_columns(df)
        health_score = calculate_health_score(df)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Mission Rows", df.shape[0])
        c2.metric("Telemetry Features", df.shape[1])
        c3.metric("Numeric Signals", len(numeric_cols))
        c4.metric("Mission Health", f"{health_score}%")

        st.markdown("###  Mission Stability Gauge")
        st.plotly_chart(create_health_gauge(health_score), use_container_width=True)

        st.markdown("###  Experiment Telemetry Preview")
        st.dataframe(df.head(25), use_container_width=True)

        st.markdown("###  Scientific Telemetry Analytics")

        tabs = st.tabs([
            "Time Series",
            "Scatter Analysis",
            "Distribution",
            "Correlation Heatmap"
        ])

        with tabs[0]:
            if len(numeric_cols) > 0:
                c1, c2 = st.columns(2)
                x_col = c1.selectbox("Timeline Column", all_cols, key="time_x")
                y_col = c2.selectbox("Telemetry Signal", numeric_cols, key="time_y")
                color_col = st.selectbox("Color Group", [None] + all_cols, key="time_color")

                fig = create_time_series_chart(df, x_col, y_col, color_col)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric telemetry columns detected.")

        with tabs[1]:
            if len(numeric_cols) >= 2:
                c1, c2 = st.columns(2)
                x_col = c1.selectbox("X-axis", numeric_cols, key="scatter_x")
                y_col = c2.selectbox("Y-axis", numeric_cols, key="scatter_y")
                color_col = st.selectbox("Color Group", [None] + all_cols, key="scatter_color")

                fig = create_scatter_chart(df, x_col, y_col, color_col)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least two numeric columns.")

        with tabs[2]:
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select Telemetry Signal", numeric_cols, key="dist_col")
                fig = create_distribution_chart(df, selected_col)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric telemetry signals available.")

        with tabs[3]:
            if len(numeric_cols) >= 2:
                st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)
            else:
                st.info("Need at least two numeric columns.")


# ======================
# AI SCIENTIFIC ASSISTANT
# ======================

elif page == "AI Scientific Assistant":

    st.markdown('<div class="section-title"> AI Scientific Assistant</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        Ask scientific questions about uploaded experiment datasets, mission logs,
        or research PDFs. PDF documents use local RAG-based retrieval for contextual answers.
    </div>
    """, unsafe_allow_html=True)

    if "file_type" not in st.session_state:
        st.warning("Please upload a CSV, Excel, or PDF first from the Experiment Upload page.")

    else:

        file_type = st.session_state["file_type"]

        if file_type == "pdf":

            st.success("Scientific PDF is ready for RAG-based AI analysis.")

            question = st.text_input(
                "Ask a scientific question",
                placeholder="Example: What biological risks are mentioned in this document?"
            )

            if st.button("Ask Scientific AI"):

                if question.strip() == "":
                    st.warning("Please enter a question.")

                elif "rag_index" not in st.session_state:
                    st.error("RAG index not found. Please upload the PDF again.")

                else:
                    with st.spinner("Retrieving scientific context and generating answer..."):
                        answer = rag_scientific_answer(
                            question,
                            st.session_state["rag_index"],
                            st.session_state["rag_chunks"]
                        )

                    st.markdown("### Scientific AI Answer")
                    st.markdown(answer)

        elif file_type == "dataframe":

            df = st.session_state["uploaded_data"]

            st.success("Experiment dataset is ready for AI analysis.")

            context = f"""
Dataset Shape: {df.shape}

Columns:
{list(df.columns)}

Missing Values:
{df.isnull().sum().to_string()}

Dataset Preview:
{df.head(25).to_string()}

Statistical Summary:
{df.describe(include='all').to_string()}
"""

            question = st.text_input(
                "Ask a question about the experiment dataset",
                placeholder="Example: What telemetry signals look important?"
            )

            if st.button("Analyze Dataset With Scientific AI"):

                if question.strip() == "":
                    st.warning("Please enter a question.")

                else:
                    with st.spinner("Scientific AI is analyzing the dataset..."):
                        answer = scientific_answer(question, context)

                    st.markdown("###  Scientific AI Answer")
                    st.markdown(answer)


# ======================
# ANOMALY DETECTION
# ======================

elif page == "Anomaly Detection":

    st.markdown('<div class="section-title"> Anomaly Detection</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        Detect abnormal telemetry behavior, biological instability,
        environmental deviations, and potential mission risks using Isolation Forest.
    </div>
    """, unsafe_allow_html=True)

    if "uploaded_data" not in st.session_state:

        st.warning("Please upload an experiment dataset first from the Experiment Upload page.")

    else:

        df = st.session_state["uploaded_data"]

        contamination = st.slider(
            "Anomaly Sensitivity",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01
        )

        if st.button("Run Anomaly Detection"):

            with st.spinner("Analyzing experiment telemetry for anomalies..."):
                anomaly_df, message = detect_anomalies(df, contamination=contamination)
                alerts = rule_based_alerts(df)

            st.session_state["anomaly_df"] = anomaly_df

            st.success(message)

            normal_count = int((anomaly_df["anomaly_status"] == "Normal").sum())
            anomaly_count = int((anomaly_df["anomaly_status"] == "Anomaly").sum())

            c1, c2, c3 = st.columns(3)

            c1.metric("Normal Records", normal_count)
            c2.metric("Anomalies", anomaly_count)
            c3.metric("Sensitivity", contamination)

            st.markdown("###  Mission Alerts")

            for alert in alerts:
                if "" in alert:
                    st.warning(alert)
                else:
                    st.success(alert)

            st.markdown("### Anomaly Detection Results")
            st.dataframe(anomaly_df.head(100), use_container_width=True)

            if anomaly_count > 0:
                st.markdown("###  Detected Anomaly Records")
                st.dataframe(
                    anomaly_df[anomaly_df["anomaly_status"] == "Anomaly"],
                    use_container_width=True
                )

            csv = anomaly_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Anomaly Report CSV",
                csv,
                "helios_anomaly_report.csv",
                "text/csv"
            )


# ======================
# PREDICTION ENGINE
# ======================

elif page == "Prediction Engine":

    st.markdown('<div class="section-title"> Prediction Engine</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        Train a machine learning model to predict experiment outcomes,
        biological response, mission stability, or success conditions.
    </div>
    """, unsafe_allow_html=True)

    if "uploaded_data" not in st.session_state:

        st.warning("Please upload an experiment dataset first.")

    else:

        df = st.session_state["uploaded_data"]

        target = st.selectbox("Select Prediction Target", df.columns)

        if st.button("Train Prediction Model"):

            try:
                with st.spinner("Training biological prediction model..."):
                    output = train_prediction_model(df, target)

                st.session_state["prediction_output"] = output

                st.success(" Prediction model trained successfully")

                st.markdown("### Problem Type")
                st.info(output["problem_type"])

                st.markdown("### Model Metrics")
                st.json(output["metrics"])

                pred_df = pd.DataFrame({
                    "Actual": output["actual"],
                    "Predicted": output["predictions"]
                })

                st.markdown("### Sample Predictions")
                st.dataframe(pred_df, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction Error: {e}")

        if "prediction_output" in st.session_state:

            output = st.session_state["prediction_output"]

            st.markdown("##  Live Experiment Prediction")

            user_input = {}
            X_sample = output["X_sample"]
            feature_columns = output["feature_columns"]

            for col in feature_columns:

                if pd.api.types.is_numeric_dtype(X_sample[col]):
                    user_input[col] = st.number_input(
                        col,
                        value=float(X_sample[col].mean())
                    )

                else:
                    options = X_sample[col].dropna().astype(str).unique().tolist()

                    if len(options) == 0:
                        options = ["Unknown"]

                    user_input[col] = st.selectbox(col, options)

            if st.button("Predict Experiment Outcome"):

                try:
                    prediction = predict_single_input(
                        output["model"],
                        user_input,
                        feature_columns,
                        output["target_encoder"]
                    )

                    st.success(f" Predicted {target}: {prediction}")

                except Exception as e:
                    st.error(f"Live Prediction Error: {e}")


# ======================
# MISSION RISK
# ======================

elif page == "Mission Risk":

    st.markdown('<div class="section-title"> Mission Risk Assessment</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        Evaluate biological mission stability using telemetry quality,
        environmental consistency, anomaly indicators, and experiment signal behavior.
    </div>
    """, unsafe_allow_html=True)

    if "uploaded_data" not in st.session_state:

        st.warning("Please upload an experiment dataset first.")

    else:

        df = st.session_state["uploaded_data"]

        if st.button("Run Mission Risk Assessment"):

            with st.spinner("Analyzing mission telemetry risk..."):
                risk_result = calculate_mission_risk(df)

            risk_score = risk_result["risk_score"]
            risk_level = risk_result["risk_level"]
            reasons = risk_result["reasons"]

            c1, c2, c3 = st.columns(3)

            c1.metric("Mission Risk Score", risk_score)
            c2.metric("Risk Level", risk_level)

            if risk_level == "LOW":
                status = "Stable"
                color = "#10b981"
            elif risk_level == "MODERATE":
                status = "Monitor Closely"
                color = "#f59e0b"
            else:
                status = "Critical"
                color = "#ef4444"

            c3.metric("Mission Status", status)

            st.markdown(f"""
            <div style="
                background: rgba(15,23,42,0.72);
                border: 1px solid {color};
                border-radius: 24px;
                padding: 30px;
                margin-top: 20px;
                margin-bottom: 25px;
                box-shadow: 0 18px 45px rgba(0,0,0,0.35);
            ">
                <h2 style="color:{color}; margin-bottom:10px;">
                    Mission Risk Level: {risk_level}
                </h2>
                <p style="color:#cbd5e1; font-size:16px;">
                    Mission telemetry and biological experiment conditions were analyzed
                    to estimate operational stability and scientific mission risk.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Risk Analysis Factors")

            for reason in reasons:
                if "stable" in reason.lower():
                    st.success(reason)
                else:
                    st.warning(reason)


# ======================
# EXPERIMENT RECOMMENDATIONS
# ======================

elif page == "Experiment Recommendations":

    st.markdown('<div class="section-title"> Experiment Recommendations</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        Generate scientific recommendations for experiment conditions based on uploaded telemetry,
        biological measurements, and mission data behavior.
    </div>
    """, unsafe_allow_html=True)

    if "uploaded_data" not in st.session_state:

        st.warning("Please upload an experiment dataset first.")

    else:

        df = st.session_state["uploaded_data"]

        if st.button("Generate Experiment Recommendations"):

            with st.spinner("Analyzing experiment parameters..."):
                recommendations = generate_experiment_recommendations(df)

            st.markdown("###  AI Experiment Recommendations")

            for rec in recommendations:
                st.info(rec)


# ======================
# KNOWLEDGE GRAPH
# ======================

elif page == "Knowledge Graph":

    st.markdown('<div class="section-title"> Scientific Knowledge Graph</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        Extract scientific entities and relationships from uploaded research PDFs or mission documents.
    </div>
    """, unsafe_allow_html=True)

    if "uploaded_text" not in st.session_state:

        st.warning("Please upload a scientific PDF first.")

    else:

        text = st.session_state["uploaded_text"]

        if st.button("Build Scientific Knowledge Graph"):

            graph, entities = build_knowledge_graph(text)
            edges = get_graph_edges(graph)

            st.success("Knowledge graph generated successfully")

            st.markdown("### Extracted Scientific Entities")
            st.write(entities)

            st.markdown("### Scientific Relationships")

            if len(edges) > 0:
                edge_df = pd.DataFrame(edges, columns=["Entity 1", "Entity 2"])
                st.dataframe(edge_df, use_container_width=True)
            else:
                st.info("No relationships detected.")


# ======================
# MISSION REPORT
# ======================

elif page == "Mission Report":

    st.markdown('<div class="section-title"> Mission Report</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        Generate a downloadable scientific mission report including observations,
        risks, recommendations, and AI-generated summaries.
    </div>
    """, unsafe_allow_html=True)

    if "uploaded_data" not in st.session_state and "uploaded_text" not in st.session_state:

        st.warning("Please upload experiment data or scientific PDF first.")

    else:

        if "uploaded_data" in st.session_state:

            df = st.session_state["uploaded_data"]

            context = f"""
Dataset Shape: {df.shape}
Columns: {list(df.columns)}

Missing Values:
{df.isnull().sum().to_string()}

Preview:
{df.head(25).to_string()}

Statistical Summary:
{df.describe(include='all').to_string()}
"""

        else:

            context = st.session_state["uploaded_text"][:8000]

        if st.button("Generate Mission Report"):

            with st.spinner("Generating scientific mission report..."):
                summary = generate_scientific_summary(context)

            st.markdown("###  Scientific Mission Summary")
            st.markdown(summary)

            report_path = generate_mission_report(
                title="Helios AI Scientific Mission Report",
                content=summary
            )

            with open(report_path, "rb") as file:
                st.download_button(
                    label="📥 Download Mission Report",
                    data=file,
                    file_name="helios_mission_report.pdf",
                    mime="application/pdf"
                )