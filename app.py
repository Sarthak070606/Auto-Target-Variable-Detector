import streamlit as st
import pandas as pd
import plotly.express as px

from data_loader import load_data
import analyzer as az
import target_detector as td
from target_detector import NoTargetResult, TargetSuggestion

st.set_page_config(page_title="Auto Target Detector", layout="wide")

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

DARK = {
    "bg":           "#0e1117",
    "card_bg":      "#1a1a2e",
    "text":         "#ffffff",
    "subtext":      "#aaaaaa",
    "border":       "#2a2a2a",
    "plot":         "plotly_dark",
    "footer_bg":    "linear-gradient(90deg, #0f0f0f, #1a1a2e, #0f0f0f)",
    "footer_border":"#2a2a2a",
    "footer_sub":   "#888888",
}

LIGHT = {
    "bg":           "#f5f7fa",
    "card_bg":      "#ffffff",
    "text":         "#111111",
    "subtext":      "#111111",
    "border":       "#dddddd",
    "plot":         "plotly_white",
    "footer_bg":    "linear-gradient(90deg, #e8eaf6, #ffffff, #e8eaf6)",
    "footer_border": "#cccccc",
    "footer_sub":   "#666666",
}

T = DARK if st.session_state.theme == "dark" else LIGHT

st.markdown(f"""
    <style>
        .stApp {{
            background-color: {T['bg']};
            color: {T['text']};
        }}
        .block-container {{
            padding-top: 1.5rem;
        }}
        .stDataFrame, .stMetric {{
            background-color: {T['card_bg']};
            border-radius: 8px;
        }}
        div[data-testid="stMetricValue"] {{
            color: {T['text']};
        }}
        label, .stSelectbox label, .stFileUploader label {{
            color: {T['text']} !important;
        }}
    </style>
""", unsafe_allow_html=True)

top_left, top_right = st.columns([6, 1])
with top_left:
    st.title(":rainbow[Auto Target Variable Detector]")
    st.markdown("Upload your dataset :green[(CSV, TSV, XLSX, JSON, TXT)]")
with top_right:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.theme == "dark":
        if st.button("☀️ Light Mode"):
            st.session_state.theme = "light"
            st.rerun()
    else:
        if st.button("🌙 Dark Mode"):
            st.session_state.theme = "dark"
            st.rerun()


def _render_target_profile(df: pd.DataFrame, col: str) -> None:
    st.markdown(f"#### 📌 Target Column Profile: `{col}`")
    series = df[col]

    ts1, ts2, ts3, ts4 = st.columns(4)
    ts1.metric("Unique Values",  int(series.nunique()))
    ts2.metric("Missing Values", int(series.isnull().sum()))
    ts3.metric("Data Type",      str(series.dtype))

    is_numeric = pd.api.types.is_numeric_dtype(series)
    if series.nunique() <= 20 or not is_numeric:
        majority = series.value_counts(normalize=True).max()
        ts4.metric("Class Balance", f"{majority:.0%} majority")
    else:
        ts4.metric("Std Dev", f"{series.std():.2f}")

    if is_numeric and series.nunique() > 20:
        fig = px.histogram(
            df, x=col, nbins=30,
            title=f"Distribution of: {col}",
            template=T["plot"],
        )
    else:
        vc = series.astype(str).value_counts().reset_index()
        vc.columns = [col, "Count"]
        fig = px.bar(
            vc, x=col, y="Count",
            title=f"Class Distribution of: {col}",
            template=T["plot"],
            color="Count", color_continuous_scale="Teal",
        )
    st.plotly_chart(fig, use_container_width=True)


uploaded_file = st.file_uploader(
    ":red[Upload Dataset]",
    type=["csv", "tsv", "xlsx", "xls", "json", "txt"],
)

if uploaded_file is None:
    st.info(":orange[Please upload a dataset file]")
    st.stop()

df, error = load_data(uploaded_file)

if error:
    st.error(f" Error loading file: {error}")
    st.stop()

for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col], errors="raise")
    except (ValueError, TypeError):
        pass

st.success("File loaded successfully!")

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("📈 Dataset Information")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows",                 df.shape[0])
c2.metric("Columns",              df.shape[1])
c3.metric("Numeric Columns",      df.select_dtypes(include="number").shape[1])
c4.metric("Total Missing Values", int(df.isnull().sum().sum()))

st.subheader("Column Analysis")
st.dataframe(az.basic_analysis(df), use_container_width=True)

st.subheader(" Column Names")
st.write(list(df.columns))

st.divider()

st.subheader("🎯 Target Variable Detection")

suggestion = td.detect_target(df)

if suggestion is None:
    st.warning("⚠️ Dataset needs at least 2 columns for target detection.")

elif isinstance(suggestion, NoTargetResult):
    st.error("🧠 No clear target variable detected in this dataset.")

    info_c1, info_c2 = st.columns([1, 2])
    with info_c1:
        st.markdown("#### Dataset Type Identified")
        st.markdown(f"## {suggestion.dataset_type}")
    with info_c2:
        st.markdown("#### Why no target was detected")
        st.info(suggestion.reason)

    st.markdown("#### 💡 What you can do with this dataset")
    for s in suggestion.suggestions:
        st.markdown(f"- {s}")

    st.markdown("#### ✏️ Override — Pick a Target Manually")
    st.caption("If you believe a column can serve as a target, select it below.")
    manual_target = st.selectbox(
        "Select a column to use as target:",
        options=["— skip —"] + list(df.columns),
        index=0,
    )

    if manual_target != "— skip —":
        _render_target_profile(df, manual_target)

    with st.expander("📊 Column Score Ranking (for reference)"):
        st.dataframe(td.get_all_scores(df), use_container_width=True)

elif isinstance(suggestion, TargetSuggestion):
    badge_colour = {"High": "green", "Medium": "orange", "Low": "red"}.get(
        suggestion.confidence, "gray"
    )

    r1, r2, r3 = st.columns([2, 1.2, 1.2])
    with r1:
        st.markdown("#### 🏆 Recommended Target Column")
        st.markdown(
            f"<h2 style='color:#00d4aa; margin:0'>{suggestion.column}</h2>",
            unsafe_allow_html=True,
        )
    with r2:
        task_icon = {
            "Binary Classification":      "🔵",
            "Multi-class Classification": "🟣",
            "Regression":                 "📈",
        }.get(suggestion.ml_task, "❓")
        st.markdown("#### ML Task Type")
        st.markdown(f"### {task_icon} {suggestion.ml_task}")
    with r3:
        st.markdown("#### Confidence")
        st.markdown(
            f"<h3 style='color:{badge_colour}'>"
            f"{suggestion.confidence} &nbsp;({suggestion.score} pts)</h3>",
            unsafe_allow_html=True,
        )

    with st.expander("💡 Why was this column chosen?", expanded=True):
        for reason in suggestion.reasons:
            st.markdown(f"- {reason}")

    with st.expander("📊 Full Column Score Ranking"):
        scores_df = td.get_all_scores(df)
        st.dataframe(scores_df, use_container_width=True)

        fig_scores = px.bar(
            scores_df, x="Score", y="Column", orientation="h",
            color="Score", color_continuous_scale="Teal",
            title="Target Variable Likelihood Score per Column",
            template=T["plot"], text="Score",
        )
        fig_scores.update_layout(yaxis=dict(autorange="reversed"), height=400)
        st.plotly_chart(fig_scores, use_container_width=True)

    if suggestion.alternatives:
        st.markdown("#### 🔁 Alternative Candidate Columns")
        alt_cols = st.columns(len(suggestion.alternatives))
        for i, (alt_col, alt_score, alt_task) in enumerate(suggestion.alternatives):
            with alt_cols[i]:
                st.info(f"**{alt_col}**\n\n{alt_task}\n\nScore: {alt_score}")

    st.markdown("#### ✏️ Override Target (Optional)")
    manual_target = st.selectbox(
        "Select your own target variable if the suggestion doesn't match:",
        options=["— use suggestion —"] + list(df.columns),
        index=0,
    )
    final_target = (
        suggestion.column if manual_target == "— use suggestion —" else manual_target
    )

    _render_target_profile(df, final_target)

    numeric_df = df.select_dtypes(include="number")
    if final_target in numeric_df.columns and numeric_df.shape[1] >= 2:
        corr_df = (
            numeric_df.corr()[final_target]
            .drop(final_target)
            .abs()
            .sort_values(ascending=False)
            .reset_index()
        )
        corr_df.columns = ["Feature", "Abs Correlation"]

        with st.expander("📐 Feature Correlation with Target"):
            fig_corr = px.bar(
                corr_df, x="Abs Correlation", y="Feature", orientation="h",
                title=f"Absolute Correlation of Features with '{final_target}'",
                template=T["plot"],
                color="Abs Correlation", color_continuous_scale="Blues",
            )
            fig_corr.update_layout(yaxis=dict(autorange="reversed"), height=400)
            st.plotly_chart(fig_corr, use_container_width=True)

st.divider()

st.subheader("📊 Data Visualizations")

if "viz_mode" not in st.session_state:
    st.session_state.viz_mode = None

btn1, btn2 = st.columns(2)
with btn1:
    if st.button("📊 Column Visualization"):
        st.session_state.viz_mode = "column"
with btn2:
    if st.button("📈 Row Visualization"):
        st.session_state.viz_mode = "row"

if st.session_state.viz_mode == "column":
    st.subheader("📊 Column Visualization")

    heatmap = az.plot_correlation_heatmap(df, T["plot"])
    if heatmap:
        st.plotly_chart(heatmap, use_container_width=True)

    missing_plot = az.plot_missing_heatmap(df)
    if missing_plot:
        st.plotly_chart(missing_plot, use_container_width=True)

    selected_column = st.selectbox("Select a column", df.columns)

    hist  = az.plot_histogram(df, selected_column, T["plot"])
    box   = az.plot_boxplot(df, selected_column, T["plot"])
    count = az.plot_countplot(df, selected_column, T["plot"])

    colA, colB = st.columns(2)
    if hist:
        with colA:
            st.plotly_chart(hist, use_container_width=True)
    if box:
        with colB:
            st.plotly_chart(box, use_container_width=True)
    if count:
        st.plotly_chart(count, use_container_width=True)

    if not hist and not box and not count:
        st.info("No chart available for the selected column.")

if st.session_state.viz_mode == "row":
    st.subheader("📈 Row Visualization")

    row_index = st.number_input(
        "Select Row Index", min_value=0, max_value=len(df) - 1, step=1
    )
    row_data = df.iloc[int(row_index)]
    st.dataframe(row_data.to_frame().T, use_container_width=True)

    numeric_cols = df.select_dtypes(include="number").columns
    numeric_row  = row_data[numeric_cols]

    if not numeric_row.empty:
        fig_row = px.bar(
            x=numeric_row.index, y=numeric_row.values,
            title=f"Numeric Values of Row {row_index}",
            template=T["plot"],
            labels={"x": "Column", "y": "Value"},
        )
        st.plotly_chart(fig_row, use_container_width=True)
    else:
        st.info("Selected row has no numeric values to visualize.")

st.divider()

st.markdown(
    f"""
    <div style='
        text-align: center;
        padding: 18px 0 10px 0;
        background: {T['footer_bg']};
        border-top: 1px solid {T['footer_border']};
        border-radius: 8px;
        margin-top: 20px;
    '>
        <span style='
            font-size: 16px;
            font-weight: 600;
            letter-spacing: 1px;
            background: linear-gradient(90deg, #00d4aa, #7b61ff, #ff6b6b, #00d4aa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        '>
            ❤️ Developed &amp; Designed by Sarthak Jain ❤️
        </span>
        <br><br>
        <span style='
            font-size: 13px;
            color: {T['footer_sub']};
            letter-spacing: 0.5px;
        '>
            
    </div>
    """,
    unsafe_allow_html=True,
)
