import pandas as pd
import numpy as np
import plotly.express as px


def safe_nunique(series: pd.Series) -> int:
    try:
        return series.nunique(dropna=True)
    except TypeError:
        return series.astype(str).nunique(dropna=True)


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        try:
            df[col].nunique()
        except TypeError:
            df[col] = df[col].astype(str)
    return df


def basic_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = sanitize_df(df)
    results = []

    for col in df.columns:
        series = df[col]
        unique_count = safe_nunique(series)
        missing_count = int(series.isnull().sum())
        missing_pct = f"{series.isnull().mean():.1%}"

        if pd.api.types.is_numeric_dtype(series):
            col_min = series.min()
            col_max = series.max()
            col_mean = round(series.mean(), 4)
        else:
            col_min = col_max = col_mean = "—"

        results.append({
            "Column":         col,
            "Dtype":          str(series.dtype),
            "Unique Values":  unique_count,
            "Missing Values": missing_count,
            "Missing %":      missing_pct,
            "Min":            col_min,
            "Max":            col_max,
            "Mean":           col_mean,
        })

    return pd.DataFrame(results)


def plot_correlation_heatmap(df: pd.DataFrame, template: str = "plotly_dark"):
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(title="Correlation Matrix", template=template, height=500)
    return fig


def plot_missing_heatmap(df: pd.DataFrame, template: str = "plotly_dark"):
    if df.isnull().sum().sum() == 0:
        return None

    fig = px.imshow(
        df.isnull().astype(int),
        color_continuous_scale="Reds",
        aspect="auto",
    )
    fig.update_layout(title="Missing Values Heatmap", template=template, height=400)
    return fig


def plot_histogram(df: pd.DataFrame, column: str, template: str = "plotly_dark"):
    if column not in df.columns:
        return None
    if not pd.api.types.is_numeric_dtype(df[column]):
        return None

    return px.histogram(
        df, x=column, nbins=30,
        title=f"Distribution of {column}",
        template=template,
    )


def plot_boxplot(df: pd.DataFrame, column: str, template: str = "plotly_dark"):
    if column not in df.columns:
        return None
    if not pd.api.types.is_numeric_dtype(df[column]):
        return None

    return px.box(df, y=column, title=f"Boxplot of {column}", template=template)


def plot_countplot(df: pd.DataFrame, column: str, template: str = "plotly_dark"):
    if column not in df.columns:
        return None
    if pd.api.types.is_numeric_dtype(df[column]):
        return None

    value_counts = df[column].astype(str).value_counts().reset_index()
    value_counts.columns = [column, "Count"]

    return px.bar(
        value_counts, x="Count", y=column, orientation="h",
        title=f"Value Counts of {column}",
        template=template,
    )


def plot_pairplot(df: pd.DataFrame, template: str = "plotly_dark"):
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return None

    fig = px.scatter_matrix(numeric_df, template=template)
    fig.update_layout(title="Pair Plot", height=600)
    return fig
