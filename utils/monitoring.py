import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def get_monitoring_columns(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()
    return numeric_cols, all_cols


def create_time_series_chart(df, x_col, y_col, color_col=None):
    return px.line(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        template="plotly_dark",
        title=f"{y_col} Trend Over Time"
    )


def create_scatter_chart(df, x_col, y_col, color_col=None):
    return px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        template="plotly_dark",
        title=f"{x_col} vs {y_col}"
    )


def create_distribution_chart(df, column):
    return px.histogram(
        df,
        x=column,
        template="plotly_dark",
        title=f"{column} Distribution"
    )


def create_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    return px.imshow(
        corr,
        text_auto=True,
        template="plotly_dark",
        title="Biolab Telemetry Correlation Heatmap"
    )


def calculate_health_score(df):
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return 75

    missing_penalty = df.isnull().sum().sum()
    duplicate_penalty = df.duplicated().sum()

    score = 100 - (missing_penalty * 0.5) - (duplicate_penalty * 0.3)
    score = max(0, min(100, score))

    return round(score, 2)


def create_health_gauge(score):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Mission Health Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#22d3ee"},
                "steps": [
                    {"range": [0, 40], "color": "#7f1d1d"},
                    {"range": [40, 70], "color": "#78350f"},
                    {"range": [70, 100], "color": "#064e3b"}
                ],
            },
        )
    )

    fig.update_layout(template="plotly_dark")
    return fig