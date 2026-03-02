import streamlit as st
import pandas as pd
import plotly.express as px

from data_loader import load_data
import analyzer as az
import target_detector as td
from target_detector import NoTargetResult, TargetSuggestion

st.set_page_config(
    page_title="Target Variable Finder",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --teal:    #00d4aa;
    --teal2:   #00ffcc;
    --red:     #ff4b4b;
    --purple:  #7c6af7;
    --blue:    #4a90d9;
    --bg:      #0a0e1a;
    --bg2:     #0f1629;
    --card:    rgba(255,255,255,0.04);
    --border:  rgba(0,212,170,0.18);
    --text:    #e8eaf6;
    --muted:   #8892b0;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

.block-container { padding: 1.5rem 2rem 4rem 2rem !important; max-width: 1200px; }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stStatusWidget"] { display: none; }
button[title="View fullscreen"] { display: none; }
button[kind="header"] { display: none !important; }
.styles_viewerBadge__CvC9N { display: none; }
#stDecoration { display: none; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--teal); border-radius: 3px; }

h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text) !important; }
p, span, div, label { font-family: 'Space Grotesk', sans-serif !important; }

hr { border-color: rgba(0,212,170,0.15) !important; margin: 2rem 0 !important; }

div[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 1.2rem 1.4rem !important;
    transition: border-color 0.3s, transform 0.3s;
}
div[data-testid="stMetric"]:hover {
    border-color: var(--teal) !important;
    transform: translateY(-2px);
}
div[data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="stMetricValue"] {
    color: var(--teal) !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

div[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    transition: border-color 0.3s;
}
div[data-testid="stFileUploader"]:hover {
    border-color: var(--teal) !important;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--teal), #00a88a) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.25s !important;
    letter-spacing: 0.02em !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, var(--teal2), var(--teal)) !important;
    box-shadow: 0 6px 20px rgba(0,212,170,0.35) !important;
    transform: translateY(-2px) !important;
}

div[data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
div[data-testid="stSelectbox"] label {
    color: var(--muted) !important;
    font-size: 0.85rem !important;
}

div[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 0.75rem !important;
    overflow: hidden !important;
}
div[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: var(--text) !important;
}
div[data-testid="stExpander"] summary:hover {
    color: var(--teal) !important;
}
div[data-testid="stExpander"] summary svg {
    color: var(--teal) !important;
}
div[data-testid="stExpander"] details[open] {
    border-color: rgba(0,212,170,0.35) !important;
}

div[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
    font-weight: 500 !important;
}

div[data-testid="stNumberInput"] input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
button[data-testid="baseButton-header"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<script>
(function() {
    function fixArrows() {
        document.querySelectorAll('[data-testid="stExpander"] summary').forEach(function(s) {
            s.childNodes.forEach(function(n) {
                if (n.nodeType === 3) { n.textContent = ''; }
            });
        });
    }
    setInterval(fixArrows, 200);
})();
</script>
""", unsafe_allow_html=True)


def section_tag(text: str):
    st.markdown(
        f"<p style='font-family:JetBrains Mono,monospace; font-size:0.72rem;"
        f"color:#00d4aa; letter-spacing:0.18em; text-transform:uppercase;"
        f"margin-bottom:0.2rem'>{text}</p>",
        unsafe_allow_html=True,
    )


def render_table(dataframe: pd.DataFrame) -> None:
    headers = "".join([
        f"<th style='background:#0f1e35;color:#00d4aa;padding:9px 14px;"
        f"font-family:JetBrains Mono,monospace;font-size:0.73rem;"
        f"letter-spacing:0.07em;border-bottom:1px solid rgba(0,212,170,0.25);"
        f"white-space:nowrap;text-align:left'>{col}</th>"
        for col in dataframe.columns
    ])
    rows_html = ""
    for i, (_, row) in enumerate(dataframe.iterrows()):
        bg = "rgba(255,255,255,0.02)" if i % 2 == 0 else "rgba(0,0,0,0)"
        cells = "".join([
            f"<td style='padding:8px 14px;color:#e8eaf6;font-size:0.84rem;"
            f"border-bottom:1px solid rgba(255,255,255,0.04);"
            f"white-space:nowrap'>{val}</td>"
            for val in row
        ])
        rows_html += f"<tr style='background:{bg}'>{cells}</tr>"
    st.markdown(
        f"<div style='overflow-x:auto;border:1px solid rgba(0,212,170,0.18);"
        f"border-radius:12px;margin-bottom:1rem'>"
        f"<table style='width:100%;border-collapse:collapse;background:#0a0e1a'>"
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True,
    )


def card(content_html: str, glow: bool = False):
    border = "rgba(0,212,170,0.5)" if glow else "rgba(0,212,170,0.18)"
    st.markdown(
        f"""<div style='background:rgba(255,255,255,0.04);border:1px solid {border};
        border-radius:14px;padding:1.4rem 1.6rem;margin-bottom:0.75rem;
        box-shadow:{"0 0 24px rgba(0,212,170,0.08)" if glow else "none"}'>
        {content_html}</div>""",
        unsafe_allow_html=True,
    )


def render_target_profile(df: pd.DataFrame, col: str, plot_template: str) -> None:
    section_tag("column info")
    st.markdown(
        f"<h4 style='margin-bottom:1rem'>Checking column "
        f"<code style='color:#00d4aa;background:rgba(0,212,170,0.1);"
        f"padding:2px 8px;border-radius:6px'>{col}</code></h4>",
        unsafe_allow_html=True,
    )
    series = df[col]
    ts1, ts2, ts3, ts4 = st.columns(4)
    ts1.metric("Unique Values",  int(series.nunique()))
    ts2.metric("Missing Values", int(series.isnull().sum()))
    ts3.metric("Data Type",      str(series.dtype))

    is_numeric = pd.api.types.is_numeric_dtype(series)
    if series.nunique() <= 20 or not is_numeric:
        majority = series.value_counts(normalize=True).max()
        ts4.metric("Most Common %", f"{majority:.0%}")
    else:
        ts4.metric("Std Dev", f"{series.std():.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    if is_numeric and series.nunique() > 20:
        fig = px.histogram(df, x=col, nbins=30,
                           title=f"How is '{col}' distributed?",
                           template=plot_template,
                           color_discrete_sequence=["#00d4aa"])
    else:
        vc = series.astype(str).value_counts().reset_index()
        vc.columns = [col, "Count"]
        fig = px.bar(vc, x=col, y="Count",
                     title=f"Value counts for '{col}'",
                     template=plot_template,
                     color="Count", color_continuous_scale="Teal")
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Space Grotesk",
    )
    st.plotly_chart(fig, use_container_width=True)


PLOT = "plotly_dark"

st.markdown("""
<div style='text-align:center; padding: 2.5rem 0 1.5rem 0'>
    <h1 style='font-size:clamp(2.2rem,5vw,3.8rem); font-weight:700;
        line-height:1.1; margin:0 0 0.8rem 0'>
        <span style='color:#ff4b4b'>Auto</span>
        <span style='color:#e8eaf6'> Target </span>
        <span style='background:linear-gradient(135deg,#00d4aa,#4a90d9);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text'>Detector</span>
    </h1>
    <p style='color:#8892b0; font-size:1.05rem; max-width:560px;
        margin:0 auto 1.5rem auto; line-height:1.7'>
        Upload your dataset and this app will try to guess
        which column you should predict
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("### 📁 :green[Upload a Dataset]")
st.markdown(
    "<p style='color:#8892b0;font-size:0.9rem;margin-bottom:1rem'>"
    "Supports CSV, TSV, Excel, JSON, TXT files</p>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    label="Drop your file here or click to browse",
    type=["csv", "tsv", "xlsx", "xls", "json", "txt"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown("""
    <div style='text-align:center;padding:2.5rem;background:rgba(0,212,170,0.03);
        border:2px dashed rgba(0,212,170,0.2);border-radius:14px;margin-top:0.5rem'>
        <div style='font-size:2.5rem;margin-bottom:0.75rem'>📂</div>
        <p style='color:#8892b0;margin:0;font-size:0.95rem'>
            Upload any dataset file to get started</p>
        <p style='color:rgba(0,212,170,0.6);margin:0.4rem 0 0 0;font-size:0.8rem;
            font-family:JetBrains Mono,monospace'>
            CSV · TSV · XLSX · JSON · TXT</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df, error = load_data(uploaded_file)
if error:
    st.markdown(
        f"<div style='background:rgba(255,75,75,0.08);border:1px solid rgba(255,75,75,0.3);"
        f"border-radius:10px;padding:1rem 1.2rem;color:#ff4b4b'>❌ Couldn't load file: {error}</div>",
        unsafe_allow_html=True,
    )
    st.stop()

for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col], errors="raise")
    except (ValueError, TypeError):
        pass

st.markdown(
    "<div style='background:rgba(0,212,170,0.08);border:1px solid rgba(0,212,170,0.25);"
    "border-radius:10px;padding:0.75rem 1.2rem;color:#00d4aa;font-weight:600'>"
    "✅ File Uploaded Successfully</div>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

section_tag("Quick look at the data")
st.markdown("### :red[Basic Info]")
st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Rows",     f"{df.shape[0]:,}")
c2.metric("Total Columns",  df.shape[1])
c3.metric("Number Columns", df.select_dtypes(include="number").shape[1])
c4.metric("Empty Cells",    int(df.isnull().sum().sum()))

st.markdown("<br>", unsafe_allow_html=True)

with st.expander(" First 10 rows"):
    render_table(df.head(10))

with st.expander(" Column info"):
    render_table(az.basic_analysis(df))

with st.expander(" All column names"):
    cols_html = "".join([
        f"<span style='background:rgba(0,212,170,0.1);border:1px solid rgba(0,212,170,0.2);"
        f"border-radius:6px;padding:3px 10px;margin:3px;display:inline-block;"
        f"font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#00d4aa'>{c}</span>"
        for c in df.columns
    ])
    st.markdown(f"<div style='padding:0.5rem 0'>{cols_html}</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

section_tag("finding the target column")
st.markdown("### :rainbow[Which column looks like the target?]")
st.markdown("<br>", unsafe_allow_html=True)

suggestion = td.detect_target(df)

if suggestion is None:
    st.markdown("""
    <div style='background:rgba(255,193,7,0.08);border:1px solid rgba(255,193,7,0.3);
        border-radius:10px;padding:1rem 1.2rem;color:#ffc107'>
         Need at least 2 columns to detect a target. Try a bigger dataset!
    </div>""", unsafe_allow_html=True)

elif isinstance(suggestion, NoTargetResult):
    st.markdown(f"""
    <div style='background:rgba(255,75,75,0.06);border:1px solid rgba(255,75,75,0.25);
        border-radius:14px;padding:1.5rem 1.8rem;margin-bottom:1.5rem'>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;
            color:#ff4b4b;letter-spacing:0.15em;margin-bottom:0.5rem'>
            Couldn't find a clear target column</div>
        <h3 style='margin:0 0 0.4rem 0;color:#e8eaf6'>{suggestion.dataset_type}</h3>
        <p style='margin:0;color:#8892b0;font-size:0.92rem'>{suggestion.reason}</p>
    </div>
    """, unsafe_allow_html=True)

    section_tag("// what can you do with this?")
    st.markdown("#### 💡 Some ideas for this dataset")
    cols = st.columns(min(len(suggestion.suggestions), 3))
    for i, s in enumerate(suggestion.suggestions):
        with cols[i % 3]:
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.04);"
                f"border:1px solid rgba(0,212,170,0.18);border-radius:12px;"
                f"padding:1rem 1.1rem;font-size:0.88rem;color:#e8eaf6;height:100%'>{s}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    section_tag(" pick one yourself")
    st.markdown("####  Try selecting a target manually")
    manual_target = st.selectbox(
        "Pick any column and we'll analyze it:",
        options=["— skip —"] + list(df.columns), index=0,
    )
    if manual_target != "— skip —":
        st.markdown("<br>", unsafe_allow_html=True)
        render_target_profile(df, manual_target, PLOT)

    with st.expander("📊 How each column scored"):
        render_table(td.get_all_scores(df))

elif isinstance(suggestion, TargetSuggestion):
    conf_colors = {"High": "#00d4aa", "Medium": "#f7b731", "Low": "#ff4b4b"}
    conf_bg     = {"High": "rgba(0,212,170,0.08)", "Medium": "rgba(247,183,49,0.08)", "Low": "rgba(255,75,75,0.08)"}
    conf_border = {"High": "rgba(0,212,170,0.3)",  "Medium": "rgba(247,183,49,0.3)",  "Low": "rgba(255,75,75,0.3)"}
    cc  = conf_colors.get(suggestion.confidence, "#8892b0")
    cb  = conf_bg.get(suggestion.confidence, "rgba(255,255,255,0.04)")
    cbr = conf_border.get(suggestion.confidence, "rgba(255,255,255,0.1)")

    task_icon = {
        "Binary Classification":      "🔵",
        "Multi-class Classification": "🟣",
        "Regression":                 "📈",
    }.get(suggestion.ml_task, "❓")

    st.markdown(f"""
    <div style='background:rgba(0,212,170,0.05);border:1px solid rgba(0,212,170,0.3);
        border-radius:16px;padding:1.8rem 2rem;margin-bottom:1.5rem;
        box-shadow:0 0 40px rgba(0,212,170,0.06)'>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;
            color:#8892b0;letter-spacing:0.15em;margin-bottom:0.6rem'>
            Our Guess for the Target Column</div>
        <div style='font-size:clamp(1.8rem,4vw,2.8rem);font-weight:700;
            color:#00d4aa;margin-bottom:1.2rem;letter-spacing:-0.01em'>
            {suggestion.column}</div>
        <div style='display:flex;flex-wrap:wrap;gap:1rem'>
            <div style='background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);
                border-radius:10px;padding:0.7rem 1.2rem'>
                <div style='font-size:0.7rem;color:#8892b0;
                    font-family:JetBrains Mono,monospace;letter-spacing:0.1em'>Type of Problem</div>
                <div style='font-size:1.1rem;font-weight:600;color:#e8eaf6;margin-top:0.2rem'>
                    {task_icon} {suggestion.ml_task}</div>
            </div>
            <div style='background:{cb};border:1px solid {cbr};
                border-radius:10px;padding:0.7rem 1.2rem'>
                <div style='font-size:0.7rem;color:#8892b0;
                    font-family:JetBrains Mono,monospace;letter-spacing:0.1em'>Confidence Level</div>
                <div style='font-size:1.1rem;font-weight:700;color:{cc};margin-top:0.2rem'>
                    {suggestion.confidence} · {suggestion.score} pts</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(" Why did we pick this column?", expanded=True):
        for reason in suggestion.reasons:
            st.markdown(
                f"<div style='padding:0.4rem 0;color:#e8eaf6;font-size:0.9rem'>"
                f"<span style='color:#00d4aa;margin-right:0.5rem'>▸</span>{reason}</div>",
                unsafe_allow_html=True,
            )

    with st.expander(" Score for every column"):
        scores_df = td.get_all_scores(df)
        render_table(scores_df)
        fig_scores = px.bar(
            scores_df, x="Score", y="Column", orientation="h",
            color="Score", color_continuous_scale="Teal",
            title="How likely is each column to be the target?",
            template=PLOT, text="Score",
        )
        fig_scores.update_layout(
            yaxis=dict(autorange="reversed"), height=420,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_family="Space Grotesk",
        )
        st.plotly_chart(fig_scores, use_container_width=True)

    if suggestion.alternatives:
        section_tag(" other possible targets")
        st.markdown("####  Other columns that could work")
        alt_cols = st.columns(len(suggestion.alternatives))
        for i, (alt_col, alt_score, alt_task) in enumerate(suggestion.alternatives):
            with alt_cols[i]:
                st.markdown(
                    f"<div style='background:rgba(255,255,255,0.04);"
                    f"border:1px solid rgba(0,212,170,0.2);border-radius:12px;"
                    f"padding:1rem 1.2rem;text-align:center'>"
                    f"<div style='color:#00d4aa;font-weight:700;font-size:1rem'>{alt_col}</div>"
                    f"<div style='color:#8892b0;font-size:0.82rem;margin:0.3rem 0'>{alt_task}</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;color:#e8eaf6;"
                    f"font-size:0.9rem'>{alt_score} pts</div></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    section_tag("// try your own col pick")
    st.markdown("### Choose your own target variable")
    manual_target = st.selectbox(
        "Choose your own column:",
        options=["— use suggestion —"] + list(df.columns), index=0,
    )
    final_target = suggestion.column if manual_target == "— use suggestion —" else manual_target

    st.markdown("<br>", unsafe_allow_html=True)
    render_target_profile(df, final_target, PLOT)

    numeric_df = df.select_dtypes(include="number")
    if final_target in numeric_df.columns and numeric_df.shape[1] >= 2:
        corr_df = (
            numeric_df.corr()[final_target]
            .drop(final_target).abs()
            .sort_values(ascending=False).reset_index()
        )
        corr_df.columns = ["Feature", "Correlation with Target"]
        with st.expander(" Which columns are related to the target?"):
            fig_corr = px.bar(
                corr_df, x="Correlation with Target", y="Feature", orientation="h",
                title=f"Feature correlation with '{final_target}'",
                template=PLOT, color="Correlation with Target",
                color_continuous_scale="Blues",
            )
            fig_corr.update_layout(
                yaxis=dict(autorange="reversed"), height=420,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_family="Space Grotesk",
            )
            st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

section_tag("Explore the data")
st.markdown("###  Charts & Graphs")
st.markdown("<br>", unsafe_allow_html=True)

if "viz_mode" not in st.session_state:
    st.session_state.viz_mode = None

btn1, btn2 = st.columns(2)
with btn1:
    if st.button(" Explore by Column", use_container_width=True):
        st.session_state.viz_mode = "column"
with btn2:
    if st.button(" Explore by Row", use_container_width=True):
        st.session_state.viz_mode = "row"

st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.viz_mode == "column":
    section_tag("// column explorer")
    st.markdown("#### 📊 Column-level Charts")
    st.markdown("<br>", unsafe_allow_html=True)

    heatmap = az.plot_correlation_heatmap(df, PLOT)
    if heatmap:
        heatmap.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_family="Space Grotesk",
        )
        st.plotly_chart(heatmap, use_container_width=True)

    missing_plot = az.plot_missing_heatmap(df)
    if missing_plot:
        missing_plot.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_family="Space Grotesk",
        )
        st.plotly_chart(missing_plot, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    selected_column = st.selectbox("Pick a column to explore", df.columns)

    hist  = az.plot_histogram(df, selected_column, PLOT)
    box   = az.plot_boxplot(df, selected_column, PLOT)
    count = az.plot_countplot(df, selected_column, PLOT)

    def clean_fig(fig):
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_family="Space Grotesk",
        )
        return fig

    colA, colB = st.columns(2)
    if hist:
        with colA: st.plotly_chart(clean_fig(hist), use_container_width=True)
    if box:
        with colB: st.plotly_chart(clean_fig(box), use_container_width=True)
    if count:
        st.plotly_chart(clean_fig(count), use_container_width=True)
    if not hist and not box and not count:
        st.info("Nothing to chart for this column — might be all text.")

if st.session_state.viz_mode == "row":
    section_tag("// row explorer")
    st.markdown("#### 📈 Row-level View")
    st.markdown("<br>", unsafe_allow_html=True)

    row_index = st.number_input(
        "Which row do you want to look at?", min_value=0, max_value=len(df) - 1, step=1
    )
    row_data = df.iloc[int(row_index)]
    render_table(row_data.to_frame().T)

    numeric_cols = df.select_dtypes(include="number").columns
    numeric_row  = row_data[numeric_cols]

    if not numeric_row.empty:
        fig_row = px.bar(
            x=numeric_row.index, y=numeric_row.values,
            title=f"All numeric values in row {row_index}",
            template=PLOT,
            labels={"x": "Column", "y": "Value"},
            color=numeric_row.values,
            color_continuous_scale="Teal",
        )
        fig_row.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_family="Space Grotesk",
        )
        st.plotly_chart(fig_row, use_container_width=True)
    else:
        st.info("No numeric values in this row to chart.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;padding:1.5rem 0 2rem 0'>
    <div style='font-size:1.8rem;margin-bottom:0.6rem'></div>
    <div style='
        font-size:1rem; font-weight:600; letter-spacing:0.05em;
        background:linear-gradient(90deg,#00d4aa,#7c6af7,#ff6b6b,#00d4aa);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        background-clip:text; background-size:200%;
    '>Made with ❤️ by Sarthak Jain</div>
</div>
""", unsafe_allow_html=True)