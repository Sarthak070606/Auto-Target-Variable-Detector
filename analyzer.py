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


def is_datetime_col(series: pd.Series) -> bool:
    """Check if a column is datetime or looks like one."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype == object:
        sample = series.dropna().head(50)
        if sample.empty:
            return False
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            # Only call it datetime if 80%+ of sample parses successfully
            return parsed.notna().mean() >= 0.8
        except Exception:
            return False
    return False


def try_parse_datetime(series: pd.Series) -> pd.Series:
    """Try to convert a series to datetime, return NaT-filled if it fails."""
    try:
        return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
    except Exception:
        return pd.Series([pd.NaT] * len(series), index=series.index)


def basic_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = sanitize_df(df)
    results = []

    for col in df.columns:
        series = df[col]
        unique_count  = safe_nunique(series)
        missing_count = int(series.isnull().sum())
        missing_pct   = f"{series.isnull().mean():.1%}"

        if pd.api.types.is_numeric_dtype(series):
            col_type = "Numeric"
            col_min  = round(float(series.min()), 4) if not series.dropna().empty else "—"
            col_max  = round(float(series.max()), 4) if not series.dropna().empty else "—"
            col_mean = round(float(series.mean()), 4) if not series.dropna().empty else "—"

        elif is_datetime_col(series):
            col_type = "DateTime"
            parsed   = try_parse_datetime(series)
            col_min  = str(parsed.min().date()) if not parsed.dropna().empty else "—"
            col_max  = str(parsed.max().date()) if not parsed.dropna().empty else "—"
            col_mean = "—"

        else:
            col_type = "Text / Category"
            col_min = col_max = col_mean = "—"

        results.append({
            "Column":         col,
            "Type":           col_type,
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
        zmin=-1, zmax=1,
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

    series = df[column]

    # --- Numeric ---
    if pd.api.types.is_numeric_dtype(series):
        if series.dropna().empty:
            return None
        return px.histogram(
            df, x=column, nbins=30,
            title=f"Distribution of {column}",
            template=template,
            color_discrete_sequence=["#00d4aa"],
        )

    # --- DateTime: bar chart of counts per month ---
    if is_datetime_col(series):
        parsed = try_parse_datetime(series).dropna()
        if parsed.empty:
            return None
        tmp = (
            parsed.dt.to_period("M").astype(str)
            .value_counts().sort_index().reset_index()
        )
        tmp.columns = ["Month", "Count"]
        fig = px.bar(
            tmp, x="Month", y="Count",
            title=f"Events over time — {column}",
            template=template,
            color_discrete_sequence=["#00d4aa"],
        )
        fig.update_xaxes(tickangle=45)
        return fig

    # Not plottable (pure text with too many unique values, etc.)
    return None


def plot_boxplot(df: pd.DataFrame, column: str, template: str = "plotly_dark"):
    if column not in df.columns:
        return None

    series = df[column]

    # --- Numeric ---
    if pd.api.types.is_numeric_dtype(series):
        if series.dropna().empty:
            return None
        return px.box(df, y=column, title=f"Boxplot of {column}", template=template)

    # --- DateTime: boxplot of year ---
    if is_datetime_col(series):
        parsed = try_parse_datetime(series).dropna()
        if parsed.empty:
            return None
        tmp = pd.DataFrame({"Year": parsed.dt.year.astype(int)})
        return px.box(
            tmp, y="Year",
            title=f"Year distribution — {column}",
            template=template,
        )

    return None


def plot_countplot(df: pd.DataFrame, column: str, template: str = "plotly_dark"):
    if column not in df.columns:
        return None

    series = df[column]

    # Skip numeric (histogram covers it)
    if pd.api.types.is_numeric_dtype(series):
        return None

    # --- DateTime: count by year ---
    if is_datetime_col(series):
        parsed = try_parse_datetime(series).dropna()
        if parsed.empty:
            return None
        tmp = (
            parsed.dt.year.astype(int)
            .value_counts().sort_index().reset_index()
        )
        tmp.columns = ["Year", "Count"]
        return px.bar(
            tmp, x="Count", y="Year", orientation="h",
            title=f"Counts by year — {column}",
            template=template,
            color="Count", color_continuous_scale="Teal",
        )

    # --- Text / Category: top 30 to avoid chart overload ---
    value_counts = series.astype(str).value_counts().head(30).reset_index()
    value_counts.columns = [column, "Count"]

    if value_counts.empty:
        return None

    return px.bar(
        value_counts, x="Count", y=column, orientation="h",
        title=f"Value Counts of {column}",
        template=template,
        color="Count", color_continuous_scale="Teal",
    )


def plot_datetime_timeline(df: pd.DataFrame, column: str, template: str = "plotly_dark"):
    """Dedicated line chart timeline for datetime columns."""
    if column not in df.columns:
        return None
    if not is_datetime_col(df[column]):
        return None

    parsed = try_parse_datetime(df[column]).dropna()
    if parsed.empty:
        return None

    tmp = (
        parsed.dt.to_period("M").astype(str)
        .value_counts().sort_index().reset_index()
    )
    tmp.columns = ["Month", "Count"]

    fig = px.line(
        tmp, x="Month", y="Count",
        title=f"Timeline — {column}",
        template=template,
        markers=True,
        color_discrete_sequence=["#00d4aa"],
    )
    fig.update_xaxes(tickangle=45)
    return fig


def plot_pairplot(df: pd.DataFrame, template: str = "plotly_dark"):
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return None

    fig = px.scatter_matrix(numeric_df, template=template)
    fig.update_layout(title="Pair Plot", height=600)
    return fig