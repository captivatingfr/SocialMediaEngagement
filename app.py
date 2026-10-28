import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import chi2_contingency, f_oneway

from utils.config import (
    CATEGORICAL_FEATURES, DATA_PATH, NUMERIC_FEATURES, TARGET,
    hour_label, order_days,
)
from utils.theme import (
    HEATMAP_CMAP, MAGENTA, VIOLET, PALETTE,
    apply_theme, callout, chart_title, show_df, test_title,
)
from utils.modeling import train_model, compare_models, feature_importance
from utils.explain import explain_prediction
from utils.recommend import recommend_setups
from utils.charts import (
    comparison_chart, confusion_chart, contribution_chart,
    day_hour_heatmap, day_hour_tables, importance_chart,
    media_type_chart, roc_chart,
)

st.set_page_config(
    page_title="Social Media Engagement Analysis",
    page_icon="📊",
    layout="wide"
)
apply_theme()


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop_duplicates().copy()
    data["post_datetime"] = pd.to_datetime(
        data["post_datetime"], errors="coerce"
    )
    return data


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
<span class="chip">🤖 Predict and explain</span>
<span class="chip">🎯 Get a posting plan</span>
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
    'analysis tabs updates to match your selection. The model tabs '
    'always use the full dataset.</div>',
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

tab1, tab_times, tab2, tab3, tab4, tab5, tab_compare, tab_reco = st.tabs([
    "📈 Overview",
    "🔥 Best Times",
    "🔎 Explore Data",
    "🧪 Statistical Tests",
    "🔗 Relationships",
    "🤖 ML Prediction",
    "🏆 Model Comparison",
    "🎯 Post Recommender"
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
    show_df(
        (media_summary * 100).round(2).rename(
            "Average engagement (%)"
        ).to_frame()
    )

# TAB: Best times (day x hour heatmap)
with tab_times:
    st.subheader("When Should You Post?")
    callout(
        "How to read this",
        "Each square is one day and hour. Darker, warmer squares mean "
        "posts made at that time earned more on average. Grey squares "
        "have too few posts to judge."
    )

    h1, h2 = st.columns(2)
    with h1:
        metric_choice = st.selectbox(
            "Measure",
            ["engagement_rate", "likes", "comments", "shares",
             "saves", "reach", "impressions", "followers_gained"],
            index=0
        )
    with h2:
        min_posts = st.slider(
            "Hide time slots with fewer than this many posts",
            1, 30, 5
        )

    average_table, count_table = day_hour_tables(
        filtered_df, metric_choice, min_posts
    )

    if average_table.notna().sum().sum() == 0:
        st.info(
            "No time slot has enough posts. Lower the minimum or "
            "widen the sidebar filters."
        )
    else:
        is_rate = metric_choice == "engagement_rate"

        stacked = average_table.stack()
        best_day, best_hour = stacked.idxmax()
        best_value = stacked.max()
        worst_day, worst_hour = stacked.idxmin()
        overall = filtered_df[metric_choice].mean()

        def fmt(value):
            return f"{value:.2%}" if is_rate else f"{value:,.1f}"

        m1, m2, m3 = st.columns(3)
        m1.metric("Best time slot", f"{best_day}, {hour_label(best_hour)}")
        m2.metric(
            "Average in that slot", fmt(best_value),
            delta=f"{best_value / overall - 1:+.1%} vs overall"
        )
        m3.metric(
            "Weakest time slot", f"{worst_day}, {hour_label(worst_hour)}"
        )

        fig = day_hour_heatmap(average_table, as_percent=is_rate)
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Small slots can look strong by luck. Raise the minimum post "
            "count to see which patterns hold up."
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
    show_df(display_df)

    st.subheader("Descriptive Statistics")
    st.caption("Count, average, spread and range for every numeric column.")
    numeric_summary = filtered_df.select_dtypes(
        include="number"
    ).describe().T
    show_df(numeric_summary.round(3))

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
            confidence,
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
        show_df(table)
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

    numeric_choices = [
        "reach", "likes", "comments", "shares", "saves",
        "follower_count", "engagement_rate", "impressions",
        "followers_gained", "caption_length", "hashtags_count"
    ]

    pick1, pick2 = st.columns(2)

    with pick1:
        x_column = st.selectbox(
            "First numerical variable", numeric_choices, index=0
        )

    with pick2:
        y_column = st.selectbox(
            "Second numerical variable", numeric_choices, index=1
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
        st.cache_resource.clear()
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
        show_df(
            pd.DataFrame(st.session_state["report"]).T.round(3)
        )

        st.divider()
        chart_title("What the model pays attention to overall")
        st.caption(
            "Each bar is the share of the model's decisions that rely on "
            "that column. Larger means more influential."
        )
        fig = importance_chart(feature_importance(st.session_state["model"]))
        st.pyplot(fig)
        plt.close(fig)

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
                    order_days(df["day_of_week"].dropna().unique())
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

            st.markdown("**Engagement metrics**")
            f3, f4 = st.columns(2)
            with f3:
                likes = st.number_input("Likes", min_value=0, value=100)
                comments = st.number_input("Comments", min_value=0, value=10)
                shares = st.number_input("Shares", min_value=0, value=5)
                saves = st.number_input("Saves", min_value=0, value=5)
            with f4:
                reach = st.number_input("Reach", min_value=0, value=1000)
                impressions = st.number_input("Impressions", min_value=0, value=1500)
                engagement_rate = st.number_input(
                    "Engagement rate", min_value=0.0, max_value=1.0,
                    value=0.04, step=0.01, format="%.4f"
                )
                followers_gained = st.number_input(
                    "Followers gained", min_value=0, value=5
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
                "hashtags_count": hashtags_count,
                "likes": likes,
                "comments": comments,
                "shares": shares,
                "saves": saves,
                "reach": reach,
                "impressions": impressions,
                "engagement_rate": engagement_rate,
                "followers_gained": followers_gained,
            }])

            prediction = st.session_state["model"].predict(input_df)[0]
            st.success(f"Predicted performance category: {prediction}")

            st.caption(
                "This is a model prediction based on patterns in the "
                "provided dataset, not a guarantee of future performance."
            )

            # Explain WHY the model predicted this category
            chart_title(f"Why the model predicted '{prediction}'")
            st.caption(
                "Pink bars pushed the prediction toward this category. "
                "Violet bars pushed it away. Longer bars mean a "
                "stronger effect."
            )
            try:
                with st.spinner("Working out what drove this prediction..."):
                    contributions = explain_prediction(
                        st.session_state["model"], input_df, prediction
                    )
                fig = contribution_chart(contributions, prediction)
                st.pyplot(fig)
                plt.close(fig)
            except ImportError:
                st.info(
                    "Install the explanation library with: pip install shap"
                )
            except Exception as error:
                st.warning(f"Could not explain this prediction: {error}")

# TAB: Model comparison
with tab_compare:
    st.subheader("Which Model Predicts Best?")
    callout(
        "A fair test",
        "Several models are scored with 5-fold cross-validation, which "
        "means each one is tested on data it has never seen, five "
        "different ways. A baseline that always guesses the most common "
        "category is included, because a model is only useful if it "
        "beats that."
    )

    if st.button("Run model comparison"):
        st.cache_resource.clear()
        with st.spinner("Cross-validating models. This can take a minute..."):
            st.session_state["comparison"] = compare_models(df)

    if "comparison" not in st.session_state:
        st.info("Click the button above to compare the models.")
    else:
        comparison = st.session_state["comparison"]
        results = comparison["results"]
        best_name = comparison["best_name"]
        best_row = results[results["Model"] == best_name].iloc[0]
        edge = best_row["Accuracy"] - comparison["baseline_acc"]

        st.markdown(
            f'<span class="badge">🏆 Best model: {best_name}</span>',
            unsafe_allow_html=True
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("Best accuracy", f"{best_row['Accuracy']:.2%}")
        k2.metric("Best macro F1", f"{best_row['Macro F1']:.3f}")
        k3.metric(
            "Accuracy vs baseline guessing",
            f"{edge * 100:+.1f} pts",
        )

        if edge < 0.02:
            st.warning(
                "The best model is barely better than always guessing the "
                "most common category. In this dataset, the available "
                "post details do not predict performance well."
            )
        else:
            st.success(
                f"The best model beats the baseline by {edge * 100:.1f} "
                "percentage points, so the post details carry real signal."
            )

        left, right = st.columns([3, 2])
        with left:
            chart_title("Model scores")
            fig = comparison_chart(results, best_name)
            st.pyplot(fig)
            plt.close(fig)
        with right:
            chart_title("Full results")
            show_df(
                results.set_index("Model").round(3)
            )

        st.caption(
            f"Scored on {comparison['n_rows']:,} posts using "
            f"{comparison['folds']}-fold cross-validation. Macro F1 gives "
            "every performance category equal weight, so it is fairer "
            "than accuracy when categories are uneven. The thin lines on "
            "the bars show variation between folds."
        )

        st.divider()
        c_left, c_right = st.columns(2)
        with c_left:
            chart_title(f"Confusion matrix: {best_name}")
            st.caption(
                "Rows are the real category, columns are the prediction. "
                "A bright diagonal means correct predictions."
            )
            fig = confusion_chart(
                comparison["confusion"], comparison["class_names"]
            )
            st.pyplot(fig)
            plt.close(fig)
        with c_right:
            chart_title(f"ROC curves: {best_name}")
            st.caption(
                "Curves bending toward the top-left are better. "
                "The dashed line is random guessing."
            )
            fig = roc_chart(comparison["roc"])
            st.pyplot(fig)
            plt.close(fig)

# TAB: Post recommender
with tab_reco:
    st.subheader("Plan Your Next Post")
    callout(
        "How the recommendation works",
        "Tell us about your account and post. The app tries every "
        "combination of media type, day, hour and call-to-action, "
        "scores each with the trained model, and shows the setups with "
        "the highest expected engagement."
    )

    with st.form("recommend_form"):
        r1, r2 = st.columns(2)
        with r1:
            r_account = st.selectbox(
                "Account type",
                sorted(df["account_type"].dropna().unique()),
                key="r_account"
            )
            r_category = st.selectbox(
                "Content category",
                sorted(df["content_category"].dropna().unique()),
                key="r_category"
            )
            r_traffic = st.selectbox(
                "Traffic source",
                sorted(df["traffic_source"].dropna().unique()),
                key="r_traffic"
            )
        with r2:
            r_followers = st.number_input(
                "Follower count", min_value=0, value=5000, key="r_followers"
            )
            r_caption = st.number_input(
                "Planned caption length", min_value=0, value=120,
                key="r_caption"
            )
            r_hashtags = st.number_input(
                "Planned number of hashtags", min_value=0, value=8,
                key="r_hashtags"
            )

        find_plan = st.form_submit_button("Find my best posting setups")

    if find_plan:
        with st.spinner("Scoring every posting combination..."):
            reco_model = train_model(df)[0]
            ranked, top_class, average = recommend_setups(
                reco_model, df, r_account, r_category, r_traffic,
                r_followers, r_caption, r_hashtags
            )
        st.session_state["reco"] = (ranked, top_class, average)

    if "reco" not in st.session_state:
        st.info("Fill in the form and click the button to see your plan.")
    else:
        ranked, top_class, average = st.session_state["reco"]

        chart_title("Your top 3 setups")
        cards = st.columns(3)
        for position, (_, row) in enumerate(ranked.head(3).iterrows()):
            cta_text = "with a call-to-action" if row["has_call_to_action"] == 1 else "without a call-to-action"
            first = " first" if position == 0 else ""
            cards[position].markdown(
                f'<div class="rec-card{first}">'
                f'<div class="rec-rank">Recommendation {position + 1}</div>'
                f'<div class="rec-title">{row["media_type"]} on {row["day_of_week"]} at {hour_label(row["post_hour"])}</div>'
                f'<div class="rec-meta">Post {cta_text}.<br>'
                f'Chance of landing in "{top_class}": {row["top_tier_probability"]:.0%}</div>'
                f'<div class="rec-score">{row["expected_engagement"]:.2%}</div>'
                f'<div class="rec-uplift">{row["uplift"]:+.1%} vs an average setup</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.write("")
        left, right = st.columns([3, 2])
        with left:
            chart_title("Top 10 setups")
            table = ranked.head(10).copy()
            table["post_hour"] = table["post_hour"].apply(hour_label)
            table["has_call_to_action"] = table["has_call_to_action"].map(
                {1: "Yes", 0: "No"}
            )
            table["expected_engagement"] = (table["expected_engagement"] * 100).round(2)
            table["top_tier_probability"] = (table["top_tier_probability"] * 100).round(1)
            table["uplift"] = (table["uplift"] * 100).round(1)
            table = table[[
                "media_type", "day_of_week", "post_hour",
                "has_call_to_action", "expected_engagement",
                "top_tier_probability", "uplift"
            ]].rename(columns={
                "media_type": "Media",
                "day_of_week": "Day",
                "post_hour": "Hour",
                "has_call_to_action": "Call-to-action",
                "expected_engagement": "Expected engagement (%)",
                "top_tier_probability": f"Chance of '{top_class}' (%)",
                "uplift": "vs average (%)",
            })
            show_df(table, hide_index=True)
        with right:
            chart_title("Best result for each media type")
            fig = media_type_chart(ranked)
            st.pyplot(fig)
            plt.close(fig)

        st.caption(
            "Expected engagement blends the model's category probabilities "
            "with each category's real average engagement. It ranks setups "
            "against each other. It is not a promise of exact results."
        )

st.divider()
st.caption("Social Media Engagement Analysis | Statistical Analysis + Machine Learning")