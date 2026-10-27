from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from scipy import stats
from scipy.stats import chi2_contingency, f_oneway

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(
    page_title="Social Media Engagement Analysis",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------------------------
# DESIGN SYSTEM (front end only: colors, fonts, chart theme, helpers)
# ---------------------------------------------------------------------------
INK = "#1E1B3A"
MUTED = "#6B6785"
LINE = "#E6E2F2"
VIOLET = "#5B3FD9"
MAGENTA = "#D6246E"
ORANGE = "#FF8A3D"
PALETTE = [VIOLET, MAGENTA, ORANGE, "#2BB3A3", "#F2B134", "#8B7CF0"]

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=DM+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #1E1B3A;
  --muted: #6B6785;
  --line: #E6E2F2;
  --violet: #5B3FD9;
  --magenta: #D6246E;
  --orange: #FF8A3D;
}

.stApp {
  background: #F6F4FB;
  color: var(--ink);
}
.stApp, .stMarkdown, p, label, input, button, textarea, li, td, th {
  font-family: 'DM Sans', system-ui, sans-serif;
}
.block-container {
  padding-top: 1.6rem;
  padding-bottom: 3rem;
  max-width: 1280px;
}
h1, h2, h3 {
  font-family: 'Bricolage Grotesque', 'DM Sans', sans-serif !important;
  color: var(--ink);
  letter-spacing: -0.02em;
}
h3 { font-weight: 800; }

/* Hero */
.hero {
  background: linear-gradient(120deg, #3B2BA8 0%, #7A2FC0 38%, #D6246E 72%, #FF8A3D 100%);
  border-radius: 28px;
  padding: 2.8rem 3rem 2.4rem 3rem;
  color: #fff;
  position: relative;
  overflow: hidden;
  margin-bottom: 0.6rem;
}
.hero::after {
  content: "";
  position: absolute;
  right: -90px; top: -90px;
  width: 340px; height: 340px;
  border-radius: 50%;
  border: 38px solid rgba(255,255,255,0.10);
}
.hero h1 {
  font-family: 'Bricolage Grotesque', sans-serif !important;
  font-weight: 800;
  font-size: 3rem;
  line-height: 1.05;
  color: #fff !important;
  margin: 0 0 0.8rem 0;
  padding: 0;
  max-width: 760px;
}
.hero p {
  font-size: 1.12rem;
  line-height: 1.55;
  max-width: 640px;
  color: rgba(255,255,255,0.92);
  margin: 0 0 1.4rem 0;
}
.hero .chips { display: flex; gap: 0.6rem; flex-wrap: wrap; }
.hero .chip {
  background: rgba(255,255,255,0.16);
  border: 1px solid rgba(255,255,255,0.35);
  border-radius: 999px;
  padding: 0.35rem 0.95rem;
  font-size: 0.92rem;
  font-weight: 500;
}

/* Metric cards */
[data-testid="stMetric"] {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1.1rem 1.3rem 1rem 1.3rem;
  box-shadow: 0 6px 20px rgba(60, 40, 140, 0.06);
  border-top: 4px solid var(--violet);
}
[data-testid="stColumn"]:nth-of-type(2) [data-testid="stMetric"] { border-top-color: var(--magenta); }
[data-testid="stColumn"]:nth-of-type(3) [data-testid="stMetric"] { border-top-color: var(--orange); }
[data-testid="stColumn"]:nth-of-type(4) [data-testid="stMetric"] { border-top-color: #2BB3A3; }
[data-testid="stMetricLabel"] p {
  color: var(--muted);
  font-size: 0.95rem;
  font-weight: 500;
}
[data-testid="stMetricValue"] {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 800;
  color: var(--ink);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.4rem;
  background: #fff;
  padding: 0.4rem;
  border-radius: 16px;
  border: 1px solid var(--line);
}
.stTabs [data-baseweb="tab"] {
  border-radius: 12px;
  padding: 0.6rem 1.1rem;
  font-weight: 600;
  color: var(--muted);
  height: auto;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(120deg, var(--violet), var(--magenta));
  color: #fff !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: #fff;
  border-right: 1px solid var(--line);
}
.side-title {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 800;
  font-size: 1.35rem;
  color: var(--ink);
  margin-bottom: 0.2rem;
}
.side-sub {
  color: var(--muted);
  font-size: 0.92rem;
  margin-bottom: 1rem;
  line-height: 1.45;
}

/* Callouts and chart titles */
.callout {
  background: #fff;
  border: 1px solid var(--line);
  border-left: 5px solid var(--violet);
  border-radius: 14px;
  padding: 0.9rem 1.2rem;
  margin: 0.4rem 0 1.2rem 0;
  color: var(--ink);
  line-height: 1.55;
}
.callout strong { display: block; margin-bottom: 0.15rem; }
.callout span { color: var(--muted); }
.chart-title {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 800;
  font-size: 1.15rem;
  color: var(--ink);
  margin: 0.6rem 0 0.4rem 0;
}
.test-title {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 800;
  font-size: 1.15rem;
  margin: 0.8rem 0 0.2rem 0;
}

/* Buttons */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  background: linear-gradient(120deg, var(--violet), var(--magenta));
  color: #fff;
  border: none;
  border-radius: 12px;
  padding: 0.6rem 1.4rem;
  font-weight: 600;
  box-shadow: 0 6px 16px rgba(214, 36, 110, 0.25);
}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
  color: #fff;
  filter: brightness(1.08);
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
  outline: 3px solid var(--orange);
  outline-offset: 2px;
}

/* Forms, tables, alerts */
[data-testid="stForm"] {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 1.4rem;
  box-shadow: 0 6px 20px rgba(60, 40, 140, 0.06);
}
[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 14px;
  overflow: hidden;
}
[data-testid="stAlert"] { border-radius: 14px; }
hr { border-color: var(--line) !important; }

@media (max-width: 760px) {
  .hero { padding: 1.8rem 1.4rem; border-radius: 20px; }
  .hero h1 { font-size: 2rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

# Chart theme so every plot matches the page
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": LINE,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#EEEAF7",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 10,
})
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "brand", [VIOLET, "#FFFFFF", MAGENTA]
)


def callout(title, body):
    st.markdown(
        f'<div class="callout"><strong>{title}</strong>'
        f'<span>{body}</span></div>',
        unsafe_allow_html=True,
    )


def chart_title(text):
    st.markdown(f'<div class="chart-title">{text}</div>', unsafe_allow_html=True)


def test_title(text):
    st.markdown(f'<div class="test-title">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# APP LOGIC (unchanged)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Instagram_Analytics.csv"

CATEGORICAL_FEATURES = [
    "account_type", "media_type", "content_category",
    "traffic_source", "day_of_week"
]

NUMERIC_FEATURES = [
    "follower_count", "has_call_to_action", "post_hour",
    "caption_length", "hashtags_count"
]

TARGET = "performance_bucket_label"


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop_duplicates().copy()
    data["post_datetime"] = pd.to_datetime(
        data["post_datetime"], errors="coerce"
    )
    return data


@st.cache_resource
def train_model(data):
    features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    model_data = data[features + [TARGET]].dropna()

    X = model_data[features]
    y = model_data[TARGET].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    preprocessing = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES
        ),
        ("numeric", StandardScaler(), NUMERIC_FEATURES)
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )

    return pipeline, accuracy, report, len(X_train), len(X_test)


if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Place Instagram_Analytics.csv "
        "inside the data folder."
    )
    st.stop()

df = load_data()

required_columns = (
    CATEGORICAL_FEATURES + NUMERIC_FEATURES + [
        TARGET, "likes", "comments", "shares", "saves",
        "reach", "impressions", "engagement_rate",
        "followers_gained", "account_id", "post_id",
        "post_date"
    ]
)

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

# Hero header
st.markdown(
    """
<div class="hero">
<h1>📊 Social Media Engagement Analysis</h1>
<p>Explore post engagement, perform statistical tests, and predict performance categories using machine learning.</p>
<div class="chips">
<span class="chip">📈 See what is working</span>
<span class="chip">🧪 Test your assumptions</span>
<span class="chip">🤖 Predict a new post</span>
</div>
</div>
""",
    unsafe_allow_html=True,
)

st.caption(
    "Dataset: Instagram analytics CSV. Findings describe this dataset "
    "and should not automatically be generalized to all Instagram users."
)

# Sidebar filters
st.sidebar.markdown(
    '<div class="side-title">🎛️ Filter the dataset</div>'
    '<div class="side-sub">Every number, chart and test on the '
    'first four tabs updates to match your selection.</div>',
    unsafe_allow_html=True,
)

media_options = sorted(df["media_type"].dropna().unique().tolist())
selected_media = st.sidebar.multiselect(
    "Media type",
    media_options,
    default=media_options
)

account_options = sorted(df["account_type"].dropna().unique().tolist())
selected_accounts = st.sidebar.multiselect(
    "Account type",
    account_options,
    default=account_options
)

performance_options = sorted(
    df[TARGET].dropna().unique().tolist()
)
selected_performance = st.sidebar.multiselect(
    "Performance category",
    performance_options,
    default=performance_options
)

filtered_df = df[
    df["media_type"].isin(selected_media)
    & df["account_type"].isin(selected_accounts)
    & df[TARGET].isin(selected_performance)
].copy()

if filtered_df.empty:
    st.warning("No records match these filters. Change your selections.")
    st.stop()

# Main metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Posts", f"{len(filtered_df):,}")
col2.metric(
    "Average engagement rate",
    f"{filtered_df['engagement_rate'].mean():.2%}"
)
col3.metric(
    "Average likes",
    f"{filtered_df['likes'].mean():.1f}"
)
col4.metric(
    "Accounts",
    f"{filtered_df['account_id'].nunique():,}"
)

st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview",
    "🔎 Explore Data",
    "🧪 Statistical Tests",
    "🔗 Relationships",
    "🤖 ML Prediction"
])

# TAB 1: Overview
with tab1:
    st.subheader("Engagement Overview")
    callout(
        "What you are looking at",
        "Engagement rate shows how strongly people react to a post. "
        "The left chart shows how posts are spread across engagement "
        "levels, and the right chart shows how many posts fall into "
        "each performance category."
    )

    left, right = st.columns(2)

    with left:
        chart_title("Engagement rate distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(
            filtered_df["engagement_rate"].dropna(),
            bins=30,
            kde=True,
            color=VIOLET,
            edgecolor="white",
            ax=ax
        )
        if ax.lines:
            ax.lines[0].set_color(MAGENTA)
            ax.lines[0].set_linewidth(2.2)
        ax.set_xlabel("Engagement rate")
        ax.set_ylabel("Number of posts")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        chart_title("Posts by performance category")
        counts = filtered_df[TARGET].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        counts.plot(
            kind="bar",
            ax=ax,
            color=[PALETTE[i % len(PALETTE)] for i in range(len(counts))],
            width=0.7
        )
        ax.set_xlabel("Performance category")
        ax.set_ylabel("Number of posts")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    chart_title("Average engagement by media type")
    media_summary = (
        filtered_df.groupby("media_type")["engagement_rate"]
        .mean()
        .sort_values(ascending=False)
    )
    st.dataframe(
        (media_summary * 100).round(2).rename(
            "Average engagement (%)"
        ).to_frame(),
        use_container_width=True
    )

# TAB 2: Explore data
with tab2:
    st.subheader("Dataset Explorer")
    callout(
        "Browse the raw posts",
        "Type any word to search every column. The table follows the "
        "filters in the sidebar, and you can download what you see."
    )

    search = st.text_input("Search the table", "", placeholder="e.g. Reel, Friday, Fitness")

    display_df = filtered_df.copy()

    if search:
        matching_rows = display_df.astype(str).apply(
            lambda column: column.str.contains(
                search, case=False, na=False
            )
        ).any(axis=1)
        display_df = display_df[matching_rows]

    st.write(f"Showing **{len(display_df):,}** rows")
    st.dataframe(display_df, use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.caption("Count, average, spread and range for every numeric column.")
    numeric_summary = filtered_df.select_dtypes(
        include="number"
    ).describe().T
    st.dataframe(numeric_summary.round(3), use_container_width=True)

    st.download_button(
        "⬇️ Download filtered CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_instagram_data.csv",
        mime="text/csv"
    )

# TAB 3: Statistical tests
with tab3:
    st.subheader("Confidence Interval and Hypothesis Testing")
    callout(
        "How to read the results",
        "A p-value below 0.05 means the pattern is unlikely to be "
        "random chance, so the result is called statistically "
        "significant. Above 0.05, the data does not give enough "
        "evidence either way."
    )

    test_data = filtered_df["engagement_rate"].dropna()

    if len(test_data) < 2:
        st.warning("At least two valid observations are required.")
    else:
        reference_mean = st.number_input(
            "Reference mean engagement rate",
            min_value=0.0,
            max_value=1.0,
            value=0.04,
            step=0.01,
            format="%.3f"
        )

        confidence = st.selectbox(
            "Confidence level",
            [0.90, 0.95, 0.99],
            index=1
        )

        mean_value = test_data.mean()
        standard_error = stats.sem(test_data)

        ci = stats.t.interval(
            confidence=confidence,
            df=len(test_data) - 1,
            loc=mean_value,
            scale=standard_error
        )

        a, b = st.columns(2)
        a.metric("Sample mean", f"{mean_value:.4%}")
        b.metric(
            f"{confidence:.0%} confidence interval",
            f"{ci[0]:.4%} to {ci[1]:.4%}"
        )

        t_stat, t_p = stats.ttest_1samp(
            test_data, popmean=reference_mean
        )

        test_title("One-sample t-test")
        st.write("H₀: The population mean equals the reference mean.")
        st.write("H₁: The population mean differs from the reference mean.")
        st.write(f"t-statistic: {t_stat:.4f}")
        st.write(f"p-value: {t_p:.6g}")

        if t_p < 0.05:
            st.success(
                "Reject H₀ at the 5% significance level."
            )
        else:
            st.info(
                "Fail to reject H₀ at the 5% significance level."
            )

        st.caption(
            "The reference mean is an illustrative value, not an "
            "official Instagram benchmark. The test assumes independent "
            "observations; posts from the same account may not be independent."
        )

    st.divider()
    st.subheader("Chi-square Test: Account Type vs Performance")
    st.caption("Asks whether some account types land in certain performance categories more often than others.")

    table = pd.crosstab(
        filtered_df["account_type"],
        filtered_df[TARGET]
    )

    if table.shape[0] >= 2 and table.shape[1] >= 2:
        chi2, p_value, dof, expected = chi2_contingency(table)

        st.write("Observed contingency table")
        st.dataframe(table, use_container_width=True)
        st.write(f"Chi-square statistic: {chi2:.4f}")
        st.write(f"Degrees of freedom: {dof}")
        st.write(f"p-value: {p_value:.6g}")

        if p_value < 0.05:
            st.write(
                "Result: evidence of an association between account type "
                "and performance category."
            )
        else:
            st.write(
                "Result: insufficient evidence of an association."
            )
    else:
        st.info("Select at least two account types and two performance categories.")

    st.divider()
    st.subheader("One-way ANOVA: Engagement by Media Type")
    st.caption("Asks whether average engagement differs between media types such as Reels, Photos and Carousels.")

    anova_groups = [
        group["engagement_rate"].dropna().values
        for _, group in filtered_df.groupby("media_type")
        if group["engagement_rate"].notna().sum() > 0
    ]

    if len(anova_groups) >= 2:
        f_stat, p_value = f_oneway(*anova_groups)
        st.write(f"F-statistic: {f_stat:.4f}")
        st.write(f"p-value: {p_value:.6g}")

        if p_value < 0.05:
            st.write(
                "At least one media-type mean differs statistically."
            )
        else:
            st.write(
                "No statistically significant difference was detected."
            )
    else:
        st.info("At least two media types are needed for ANOVA.")

# TAB 4: Relationships
with tab4:
    st.subheader("Correlation Analysis")
    callout(
        "Pick any two numbers",
        "Correlation runs from -1 to +1. Close to +1 means they rise "
        "together, close to -1 means one rises as the other falls, and "
        "near 0 means no clear link."
    )

    pick1, pick2 = st.columns(2)

    with pick1:
        x_column = st.selectbox(
            "First numerical variable",
            [
                "reach", "likes", "comments", "shares", "saves",
                "follower_count", "engagement_rate", "impressions",
                "followers_gained", "caption_length", "hashtags_count"
            ],
            index=0
        )

    with pick2:
        y_column = st.selectbox(
            "Second numerical variable",
            [
                "reach", "likes", "comments", "shares", "saves",
                "follower_count", "engagement_rate", "impressions",
                "followers_gained", "caption_length", "hashtags_count"
            ],
            index=1
        )

    pair = filtered_df[[x_column, y_column]].dropna()

    if len(pair) >= 2 and x_column != y_column:
        pearson_r, pearson_p = stats.pearsonr(
            pair[x_column], pair[y_column]
        )
        spearman_r, spearman_p = stats.spearmanr(
            pair[x_column], pair[y_column]
        )

        c1, c2 = st.columns(2)
        c1.metric("Pearson correlation", f"{pearson_r:.4f}")
        c2.metric("Spearman correlation", f"{spearman_r:.4f}")

        st.write(f"Pearson p-value: {pearson_p:.6g}")
        st.write(f"Spearman p-value: {spearman_p:.6g}")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=pair,
            x=x_column,
            y=y_column,
            alpha=0.5,
            color=MAGENTA,
            edgecolor="white",
            ax=ax
        )
        ax.set_title(f"{x_column} vs {y_column}")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Select two different variables with valid observations.")

    st.subheader("Numerical Correlation Heatmap")
    st.caption("Violet cells move in opposite directions, pink cells move together. The stronger the color, the stronger the link.")
    correlation = filtered_df.select_dtypes(
        include="number"
    ).corr()

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(
        correlation,
        cmap=HEATMAP_CMAP,
        center=0,
        linewidths=0.5,
        linecolor="white",
        ax=ax
    )
    st.pyplot(fig)
    plt.close(fig)

# TAB 5: Machine learning
with tab5:
    st.subheader("Predict Post Performance")
    callout(
        "How this works",
        "Train a Random Forest classifier using post and account metadata. "
        "The model does not use likes, comments, reach, impressions, or "
        "engagement rate as input features."
    )

    if st.button("Train / Load Classifier"):
        with st.spinner("Training the model..."):
            model, accuracy, report, train_rows, test_rows = train_model(df)

        st.session_state["model"] = model
        st.session_state["accuracy"] = accuracy
        st.session_state["report"] = report
        st.session_state["train_rows"] = train_rows
        st.session_state["test_rows"] = test_rows

    if "model" not in st.session_state:
        st.info("Click the button above to train the classifier.")
    else:
        st.metric(
            "Test accuracy",
            f"{st.session_state['accuracy']:.2%}"
        )

        st.write(
            f"Training rows: {st.session_state['train_rows']:,} "
            f"| Testing rows: {st.session_state['test_rows']:,}"
        )

        chart_title("Classification report")
        st.dataframe(
            pd.DataFrame(st.session_state["report"]).T.round(3),
            use_container_width=True
        )

        st.divider()
        chart_title("Enter details for a new post")

        with st.form("prediction_form"):
            f1, f2 = st.columns(2)

            with f1:
                account_type = st.selectbox(
                    "Account type", sorted(df["account_type"].dropna().unique())
                )
                media_type = st.selectbox(
                    "Media type", sorted(df["media_type"].dropna().unique())
                )
                content_category = st.selectbox(
                    "Content category",
                    sorted(df["content_category"].dropna().unique())
                )
                traffic_source = st.selectbox(
                    "Traffic source",
                    sorted(df["traffic_source"].dropna().unique())
                )
                day_of_week = st.selectbox(
                    "Day of week",
                    sorted(df["day_of_week"].dropna().unique())
                )

            with f2:
                follower_count = st.number_input(
                    "Follower count", min_value=0, value=5000
                )
                has_cta = st.selectbox(
                    "Has call to action?", [0, 1],
                    format_func=lambda value: "Yes" if value == 1 else "No"
                )
                post_hour = st.slider("Post hour", 0, 23, 12)
                caption_length = st.number_input(
                    "Caption length", min_value=0, value=120
                )
                hashtags_count = st.number_input(
                    "Number of hashtags", min_value=0, value=8
                )

            submitted = st.form_submit_button("Predict Performance")

        if submitted:
            input_df = pd.DataFrame([{
                "account_type": account_type,
                "media_type": media_type,
                "content_category": content_category,
                "traffic_source": traffic_source,
                "day_of_week": day_of_week,
                "follower_count": follower_count,
                "has_call_to_action": has_cta,
                "post_hour": post_hour,
                "caption_length": caption_length,
                "hashtags_count": hashtags_count
            }])

            prediction = st.session_state["model"].predict(input_df)[0]
            st.success(f"Predicted performance category: {prediction}")

            st.caption(
                "This is a model prediction based on patterns in the "
                "provided dataset, not a guarantee of future performance."
            )

st.divider()
st.caption("Social Media Engagement Analysis | Statistical Analysis + Machine Learning")