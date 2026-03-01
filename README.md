<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Auto%20Target%20Detector&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=AI-Powered%20ML%20Target%20Variable%20Detection%20Engine&descAlignY=55&descSize=16"/>

<p>
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PRs-Welcome-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Made%20with-❤️-red?style=for-the-badge"/>
</p>

<br/>

> ### 🎯 *Upload any dataset. Get instant ML target variable detection, task classification, confidence scoring, and rich visualizations — all in one click.*

<br/>

</div>


## 📌 Table of Contents

| # | Section |
|---|---------|
| 1 | [✨ What is Auto Target Detector?](#-what-is-auto-target-detector) |
| 2 | [🚀 Live Demo & Screenshots](#-live-demo--screenshots) |
| 3 | [⚙️ How It Works](#️-how-it-works) |
| 4 | [🗂️ Supported File Formats](#️-supported-file-formats) |
| 5 | [📦 Installation](#-installation) |
| 6 | [▶️ Running the App](#️-running-the-app) |
| 7 | [🧠 Detection Logic](#-detection-logic) |
| 8 | [📊 Features Breakdown](#-features-breakdown) |
| 9 | [🗃️ Project Structure](#️-project-structure) |
| 10 | [🤝 Contributing](#-contributing) |

---

## ✨ What is Auto Target Detector?

**Auto Target Detector** is a smart Streamlit web app that analyzes any dataset and automatically identifies the most likely **target variable** for machine learning — so you spend less time guessing and more time building models.

It doesn't just guess randomly. It uses a **multi-signal scoring engine** that examines column names, data types, cardinality, missing values, rating patterns, and dataset-level signals to produce a confident, explainable recommendation.

You upload a CSV  →  App analyzes every column  →  Recommends target + ML task type  →  Shows why

### 🌟 Why use this?

- ⏱️ **Save hours** of manual EDA before starting a project
- 🤖 **Beginner-friendly** — no ML knowledge needed to get started
- 🔍 **Explainable** — every recommendation comes with reasons
- 🧹 **Handles messy data** — strings, nulls, timestamps, mixed types
- 🐦 **Smart enough to say "no target"** — detects tweet/log/raw datasets

---

## 🚀 Live Demo & Screenshots

<div align="center">

### 🎯 Target Detection in Action

┌─────────────────────────────────────────────────────────────────┐
│  🏆 Recommended Target    │  ML Task Type   │  Confidence       │
│                           │                 │                   │
│   rate                    │  📈 Regression  │  🟢 High (78 pts) │
└─────────────────────────────────────────────────────────────────┘


### 💡 Why was this column chosen?

  ✅  Column name matches common target variable keywords
  ✅  Column contains rating-like values (e.g. '4.1/5')
  ✅  No missing values — typical for a well-formed target column
  ✅  Last column in dataset — conventionally placed target position
```

### 🐦 No-Target Detection (Tweet / Log datasets)

┌──────────────────────────────────────────────────────────────────┐
│  🧠 No clear target variable detected in this dataset.           │
│                                                                  │
│  Dataset Type: 🐦 Social Media / Tweet Data                      │
│                                                                  │
│  💡 What you can do:                                             │
│   • 🔤 Sentiment Analysis — Positive / Negative / Neutral        │
│   • 📌 Topic Modeling — LDA or BERTopic                          │
│   • 📈 Engagement Prediction — predict likes/retweets            │
└──────────────────────────────────────────────────────────────────┘


</div>



## ⚙️ How It Works

```mermaid
flowchart TD
    A[📁 Upload Dataset] --> B[🔄 Load & Parse File]
    B --> C[🧹 Type Coercion & Cleaning]
    C --> D{Dataset Type Check}
    D -->|Social / Log / Text / Metadata| E[🚫 NoTargetResult]
    D -->|Structured ML Dataset| F[📊 Profile All Columns]
    F --> G[🧮 Score Each Column]
    G --> H{Best Score > 0?}
    H -->|No| E
    H -->|Yes| I[🎯 TargetSuggestion]
    I --> J[📈 Visualize Distribution]
    I --> K[📐 Correlation Analysis]
    I --> L[🔁 Show Alternatives]
    E --> M[💡 Show Suggestions]

    style A fill:#1e3a5f,color:#fff
    style I fill:#00d4aa,color:#000
    style E fill:#ff6b6b,color:#fff
    style G fill:#4a90d9,color:#fff



## 🗂️ Supported File Formats

<div align="center">

| Format | Extension | Notes |
|--------|-----------|-------|
| 📄 CSV | `.csv` | Most common, auto-detected delimiter |
| 📑 TSV | `.tsv` | Tab-separated values |
| 📊 Excel | `.xlsx`, `.xls` | Multi-sheet support |
| 🗃️ JSON | `.json` | Flat and nested structures |
| 📝 Text | `.txt` | Delimiter auto-detected |
| 🛢️ SQL | `.sql` | SQLite database files |

</div>




## 📦 Installation

### Prerequisites
- Python **3.8+**
- pip

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/auto-target-detector.git
cd auto-target-detector

### Step 2 — Create a virtual environment *(recommended)*
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate


### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

<details>
<summary>📋 <b>requirements.txt contents</b></summary>

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
openpyxl>=3.1.0


</details>

---

## ▶️ Running the App

```bash
streamlit run app.py


Then open your browser at:

http://localhost:8501
```


## 🧠 Detection Logic

The heart of the app is a **multi-signal scoring engine** in `target_detector.py`.

### 🔍 Column Scoring System

Every column is profiled and scored on the following signals:

<div align="center">

| Signal | Points | Condition |
|--------|--------|-----------|
| 🔑 Keyword Match | `+40` | Column name matches 60+ target keywords (`rate`, `price`, `label`, `churn`...) |
| ⭐ Rating Pattern | `+30` | Values look like `"4.1/5"` or parseable floats |
| 🔵 Binary Column | `+25` | Exactly 2 unique values — perfect for classification |
| 🟣 Low Cardinality Categorical | `+18` | ≤20 unique string values |
| 🔢 Low Cardinality Numeric | `+15` | ≤15 unique numeric values |
| 📈 Continuous Numeric | `+10` | High cardinality numeric — regression candidate |
| ✅ No Missing Values | `+5` | Clean target columns have no nulls |
| 📐 Good Skewness | `+3` | Skewness < 3 — reasonable distribution |
| 📌 Last Column | `+8` | Convention bonus |
| 🪪 ID / Free Text | `-40` | Column looks like an ID, URL, name, or tweet |
| ❗ Very High Cardinality | `-30` | >95% unique values |
| ⚠️ High Missing Rate | `-20` | >30% null values |

</div>

### 🗂️ Dataset-Type Detection

Before scoring columns, the app checks if the **entire dataset** is a raw/unlabeled type:

| Dataset Pattern | Detection Signals | Response |
|----------------|-------------------|----------|
| 🐦 Social Media | `tweet`, `hashtag`, `screen_name`, `retweet` | `NoTargetResult` + NLP suggestions |
| 📋 Log / Event | 2+ of: `log`, `event`, `timestamp`, `session` | `NoTargetResult` + analytics suggestions |
| 📝 Free Text | >50% columns are long-text, no target keywords | `NoTargetResult` + NLP suggestions |
| 🗂️ Metadata | >75% columns are IDs/URLs/names | `NoTargetResult` + engineering suggestions |

### 🎯 Confidence Levels

Score ≥ 50  AND  gap from runner-up ≥ 15  →  🟢 HIGH
Score ≥ 25  AND  gap from runner-up ≥ 8   →  🟡 MEDIUM
Everything else                            →  🔴 LOW



## 📊 Features Breakdown

<details>
<summary>📁 <b>Dataset Preview & Info</b></summary>

- First 5 rows preview
- Row count, column count, numeric column count
- Total missing values counter
- Full column analysis table (dtype, unique count, missing %, etc.)

</details>

<details>
<summary>🎯 <b>Target Variable Detection</b></summary>

- Recommended target column with color-coded name
- ML task type: Binary Classification / Multi-class Classification / Regression
- Confidence level: High / Medium / Low with score
- Expandable "Why was this chosen?" reasoning list
- Full column score ranking bar chart
- Up to 3 alternative candidate columns
- Manual override dropdown — pick your own target

</details>

<details>
<summary>📌 <b>Target Column Profile</b></summary>

- Unique value count
- Missing value count
- Data type
- Class balance (majority %) or Std Dev for continuous
- Distribution histogram (numeric) or class bar chart (categorical)
- Feature correlation heatmap with target

</details>

<details>
<summary>🐦 <b>No-Target Dataset Handling</b></summary>

- Identifies dataset type: Social Media, Log, Text, or Metadata
- Explains why no target was detected
- Gives 4–5 specific actionable suggestions
- Still allows manual target override
- Shows column score table for reference

</details>

<details>
<summary>📊 <b>Data Visualizations</b></summary>

**Column Mode:**
- Correlation heatmap (numeric columns)
- Missing values heatmap
- Per-column: Histogram + Box plot + Count plot

**Row Mode:**
- Select any row by index
- View full row as a table
- Bar chart of all numeric values in that row

</details>


## 🗃️ Project Structure


auto-target-detector/
│
├── 📄 app.py                  # Main Streamlit app & UI logic
├── 🧠 target_detector.py      # Scoring engine, dataset type detection
├── 📊 analyzer.py             # Column analysis & visualization helpers
├── 📁 data_loader.py          # File loading for CSV/TSV/XLSX/JSON/SQL
├── 📋 requirements.txt        # Python dependencies
└── 📖 README.md               # You are here



## 🤝 Contributing

Contributions are welcome! Here's how to get started:

```bash
# 1. Fork the repo on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/auto-target-detector.git

# 3. Create a feature branch
git checkout -b feature/my-awesome-feature

# 4. Make your changes, then commit
git add .
git commit -m "✨ Add: my awesome feature"

# 5. Push and open a Pull Request
git push origin feature/my-awesome-feature
```

### 💡 Ideas for contributions
- [ ] Add support for more file formats (Parquet, Feather)
- [ ] Export detected target + ML task as a JSON config
- [ ] Add a "suggested model" recommendation after target detection
- [ ] Dark/light theme toggle
- [ ] Batch file upload support



<div align="center">

### 🌟 If this project helped you, please give it a star!

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer"/>

**Built with ❤️ by [Sarthak Jain](https://github.com/YOUR_USERNAME)**

</div>