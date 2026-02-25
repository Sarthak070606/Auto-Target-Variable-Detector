import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List



@dataclass
class ColumnProfile:
    name: str
    dtype: str
    unique_count: int
    unique_ratio: float
    missing_ratio: float
    is_numeric: bool
    is_binary: bool
    cardinality: str
    skewness: Optional[float]
    has_target_keyword: bool
    looks_like_rating: bool
    looks_like_id: bool
    looks_like_text: bool
    score: float = 0.0
    reasons: list = field(default_factory=list)


@dataclass
class TargetSuggestion:
    column: str
    score: float
    ml_task: str
    confidence: str
    reasons: list
    alternatives: list


@dataclass
class NoTargetResult:
    reason: str
    dataset_type: str
    suggestions: list


TARGET_KEYWORDS = [
    "target", "label", "output", "class", "result", "outcome",
    "predict", "prediction", "response", "y",
    "churn", "fraud", "survived", "survival", "default", "diagnosis",
    "status", "flag", "approved", "passed", "failed", "success",
    "failure", "grade", "decision", "voted", "winner", "loser",
    "purchased", "clicked", "converted", "retained", "attrition",
    "loan", "risk", "category", "type", "group", "cluster",
    "price", "salary", "revenue", "sales", "demand", "income",
    "spend", "cost", "amount", "value", "total", "profit", "loss",
    "rating", "rate", "score", "rank", "stars", "review_score",
    "aggregate", "votes", "vote", "likes", "review",
    "price_range", "price_for_two", "approx_cost", "annual_income",
    "quality", "performance", "level", "priority", "sentiment",
]

ID_KEYWORDS = [
    "id", "uuid", "guid", "index", "key", "url", "link",
    "image", "photo", "img", "phone", "address", "email",
    "name", "title", "description", "comment", "text", "tweet",
    "post", "message", "content", "body", "review_list",
    "reviews_list", "menu", "menu_item", "dish", "caption",
    "hashtag", "username", "user", "handle", "screen_name",
    "timestamp", "created_at", "date", "time", "datetime",
    "location", "place", "city", "country", "coordinates",
    "source", "device", "lang", "language",
]

RAW_DATA_KEYWORDS = [
    "tweet", "post", "message", "text", "body", "content",
    "caption", "hashtag", "mention", "retweet", "reply",
    "log", "event", "action", "activity", "session",
    "raw", "unstructured", "feed", "stream",
]


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


def get_cardinality(unique_ratio: float, unique_count: int) -> str:
    if unique_count <= 2:
        return "binary"
    if unique_count <= 20 or unique_ratio < 0.05:
        return "low"
    if unique_ratio < 0.50:
        return "medium"
    return "high"


def has_target_keyword(col_name: str) -> bool:
    col_lower = str(col_name).lower().replace("-", "_").replace(" ", "_")
    return any(keyword in col_lower for keyword in TARGET_KEYWORDS)


def looks_like_id_col(col_name: str) -> bool:
    col_lower = str(col_name).lower().replace("-", "_").replace(" ", "_")
    return any(kw in col_lower for kw in ID_KEYWORDS)


def looks_like_raw_data_col(col_name: str) -> bool:
    col_lower = str(col_name).lower().replace("-", "_").replace(" ", "_")
    return any(kw in col_lower for kw in RAW_DATA_KEYWORDS)


def looks_like_rating_col(series: pd.Series, col_name: str) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return False
    try:
        sample = series.dropna().astype(str).head(50)
        slash_matches = sample.str.match(r"^\s*[\d\.]+\s*/\s*[\d\.]+\s*$").sum()
        if slash_matches / max(len(sample), 1) > 0.5:
            return True
        numeric_matches = pd.to_numeric(sample.str.strip(), errors="coerce").notna().sum()
        if numeric_matches / max(len(sample), 1) > 0.7:
            return True
    except Exception:
        pass
    return False


def looks_like_text_col(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return False
    try:
        sample = series.dropna().astype(str).head(100)
        avg_words = sample.str.split().str.len().mean()
        avg_chars = sample.str.len().mean()
        if avg_words > 5 and avg_chars > 30:
            return True
    except Exception:
        pass
    return False


def detect_dataset_type(df: pd.DataFrame):
    col_names = [str(c).lower() for c in df.columns]
    total_cols = len(df.columns)

    raw_col_count   = sum(1 for c in col_names if any(kw in c for kw in RAW_DATA_KEYWORDS))
    id_col_count    = sum(1 for c in col_names if any(kw in c for kw in ID_KEYWORDS))
    text_col_count  = sum(1 for col in df.columns if looks_like_text_col(df[col]))
    target_kw_count = sum(1 for c in col_names if any(kw in c for kw in TARGET_KEYWORDS))
    noise_ratio     = (raw_col_count + id_col_count + text_col_count) / max(total_cols, 1)

    social_signals = [
        "tweet", "retweet", "hashtag", "mention", "handle",
        "screen_name", "username", "follower", "following",
    ]
    if any(any(kw in c for c in col_names) for kw in social_signals):
        return (
            True,
            "🐦 Social Media / Tweet Data",
            "Dataset contains social media signals (tweets, hashtags, usernames) with no clear prediction target.",
            [
                "🔤 **Sentiment Analysis** — classify tweets as Positive / Negative / Neutral",
                "📌 **Topic Modeling** — use LDA or BERTopic to discover themes",
                "📈 **Engagement Prediction** — predict likes/retweets if those columns exist",
                "🔍 **Named Entity Recognition (NER)** — extract people, places, brands",
                "📊 **Exploratory Analysis** — hashtag frequency, word clouds, posting trends",
            ],
        )

    log_signals = ["log", "event", "action", "activity", "session",
                   "timestamp", "created_at", "datetime"]
    if sum(1 for kw in log_signals if any(kw in c for c in col_names)) >= 2:
        return (
            True,
            "📋 Log / Event Data",
            "Dataset appears to be event or activity log data with timestamps and actions but no target label.",
            [
                "⏱️ **Anomaly Detection** — find unusual events or sessions",
                "🔮 **Next Event Prediction** — predict the next user action",
                "📊 **Usage Pattern Analysis** — cohort analysis, funnel analysis",
                "⚠️ **Fraud / Bot Detection** — if user behavior data is present",
            ],
        )

    if text_col_count / max(total_cols, 1) > 0.5 and target_kw_count == 0:
        return (
            True,
            "📝 Unstructured Text Data",
            "Most columns contain free-form text with no structured target variable.",
            [
                "🔤 **Text Classification** — manually label a sample, then train a classifier",
                "📌 **Clustering** — group similar documents using K-Means or DBSCAN",
                "📝 **Summarization** — use NLP to summarize documents",
                "🔍 **Keyword Extraction** — TF-IDF or YAKE for important terms",
            ],
        )

    if noise_ratio > 0.75 and target_kw_count == 0:
        return (
            True,
            "🗂️ Metadata / ID-heavy Dataset",
            "Most columns appear to be identifiers, URLs, or metadata rather than ML features.",
            [
                "🔗 **Join with another dataset** — this may be a lookup/reference table",
                "📊 **Descriptive Statistics only** — explore distributions and patterns",
                "🧹 **Feature Engineering** — extract useful features from text/URL columns first",
            ],
        )

    return (False, None, None, None)


def profile_columns(df: pd.DataFrame) -> List[ColumnProfile]:
    total_rows = len(df)
    profiles = []

    for col in df.columns:
        series = df[col]
        values = series.dropna()
        is_numeric = pd.api.types.is_numeric_dtype(series)

        unique_count  = safe_nunique(series)
        unique_ratio  = unique_count / total_rows if total_rows > 0 else 0
        missing_ratio = series.isnull().mean()

        skewness = None
        if is_numeric and len(values) > 2:
            try:
                skewness = float(values.skew())
            except Exception:
                pass

        profile = ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            missing_ratio=missing_ratio,
            is_numeric=is_numeric,
            is_binary=(unique_count == 2),
            cardinality=get_cardinality(unique_ratio, unique_count),
            skewness=skewness,
            has_target_keyword=has_target_keyword(col),
            looks_like_rating=looks_like_rating_col(series, col),
            looks_like_id=looks_like_id_col(col),
            looks_like_text=looks_like_text_col(series),
        )
        profiles.append(profile)

    return profiles


def score_column(profile: ColumnProfile) -> ColumnProfile:
    score = 0.0
    reasons = []

    if profile.looks_like_id or profile.looks_like_text:
        score -= 40
        reasons.append("Column appears to be an ID, free-text, or metadata field — unlikely target")

    if profile.has_target_keyword:
        score += 40
        reasons.append("Column name matches common target variable keywords")

    if profile.looks_like_rating:
        score += 30
        reasons.append("Column contains rating-like values (e.g. '4.1/5') — strong regression target")

    if profile.is_binary:
        score += 25
        reasons.append("Binary column (2 unique values) — ideal for binary classification")

    elif profile.cardinality == "low" and not profile.is_numeric:
        score += 18
        reasons.append(f"Low cardinality categorical ({profile.unique_count} classes) — good for classification")

    elif profile.cardinality == "low" and profile.is_numeric and profile.unique_count <= 15:
        score += 15
        reasons.append(f"Low cardinality numeric ({profile.unique_count} values) — likely encoded class labels")

    elif profile.cardinality in ("medium", "high") and profile.is_numeric:
        score += 10
        reasons.append("Continuous numeric column — possible regression target")

    elif profile.cardinality == "medium" and not profile.is_numeric:
        score += 8
        reasons.append("Medium cardinality categorical — possible multi-class target")

    if profile.unique_ratio > 0.95 and profile.unique_count > 50:
        score -= 30
        reasons.append("Very high cardinality — likely an ID or free-text column")

    if profile.missing_ratio > 0.3:
        score -= 20
        reasons.append(f"High missing rate ({profile.missing_ratio:.0%}) — unusual for a target variable")
    elif profile.missing_ratio > 0.05:
        score -= 8
        reasons.append(f"Moderate missing values ({profile.missing_ratio:.0%})")
    elif profile.missing_ratio == 0:
        score += 5
        reasons.append("No missing values — typical for a well-formed target column")

    if profile.is_numeric and profile.skewness is not None:
        if abs(profile.skewness) < 3:
            score += 3
            reasons.append("Reasonable skewness — plausible continuous target")

    profile.score = max(score, 0.0)
    profile.reasons = reasons
    return profile


def get_ml_task(profile: ColumnProfile) -> str:
    if profile.looks_like_rating:
        return "Regression"
    if profile.is_binary:
        return "Binary Classification"
    if profile.cardinality == "low":
        return "Multi-class Classification"
    if profile.is_numeric and profile.cardinality in ("medium", "high"):
        return "Regression"
    if not profile.is_numeric and profile.cardinality == "medium":
        return "Multi-class Classification"
    return "Unknown"


def get_confidence(score: float, runner_up_score: float) -> str:
    gap = score - runner_up_score
    if score >= 50 and gap >= 15:
        return "High"
    if score >= 25 and gap >= 8:
        return "Medium"
    return "Low"


def detect_target(df: pd.DataFrame):
    if df.shape[1] < 2:
        return None

    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    df = sanitize_df(df)

    is_raw, dataset_type, reason, suggestions = detect_dataset_type(df)
    if is_raw:
        return NoTargetResult(
            reason=reason,
            dataset_type=dataset_type,
            suggestions=suggestions,
        )

    profiles = profile_columns(df)
    last_col = df.columns[-1]

    scored = []
    for p in profiles:
        p = score_column(p)
        if p.name == last_col:
            p.score += 8
            p.reasons.append("Last column in dataset — conventionally placed target position")
        scored.append(p)

    scored.sort(key=lambda p: p.score, reverse=True)
    best = scored[0]
    runner_up_score = scored[1].score if len(scored) > 1 else 0

    if best.score <= 0:
        return NoTargetResult(
            reason="No column in this dataset has characteristics of a typical ML target variable.",
            dataset_type="❓ Unknown / Raw Dataset",
            suggestions=[
                "🏷️ **Manually label your data** — add a target column based on your goal",
                "📊 **Use for EDA only** — explore distributions, correlations, and patterns",
                "🔍 **Unsupervised Learning** — try clustering (K-Means) or anomaly detection",
            ],
        )

    alternatives = [
        (p.name, round(p.score, 1), get_ml_task(p))
        for p in scored[1:4]
        if p.score > 5
    ]

    return TargetSuggestion(
        column=best.name,
        score=round(best.score, 1),
        ml_task=get_ml_task(best),
        confidence=get_confidence(best.score, runner_up_score),
        reasons=best.reasons,
        alternatives=alternatives,
    )


def get_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    df = sanitize_df(df)

    profiles = profile_columns(df)
    last_col = df.columns[-1]
    rows = []

    for p in profiles:
        p = score_column(p)
        if p.name == last_col:
            p.score += 8
        rows.append({
            "Column":        p.name,
            "Score":         round(p.score, 1),
            "ML Task":       get_ml_task(p),
            "Cardinality":   p.cardinality,
            "Missing %":     f"{p.missing_ratio:.1%}",
            "Unique Values": p.unique_count,
            "Keyword Match": "✅" if p.has_target_keyword else "❌",
            "Rating Column": "✅" if p.looks_like_rating else "❌",
            "Text Column":   "✅" if p.looks_like_text else "❌",
        })

    rows.sort(key=lambda r: r["Score"], reverse=True)
    return pd.DataFrame(rows)