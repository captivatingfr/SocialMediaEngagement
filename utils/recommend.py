import itertools

import pandas as pd

from utils.config import FEATURES, TARGET, order_days


def class_engagement_means(data):
    """Average real engagement rate of each performance category."""
    return data.groupby(data[TARGET].astype(str))["engagement_rate"].mean()


def build_grid(data, account_type, content_category, traffic_source,
               follower_count, caption_length, hashtags_count):
    """Every media type x day x hour x call-to-action combination."""
    media_types = sorted(data["media_type"].dropna().unique().tolist())
    days = order_days(data["day_of_week"].dropna().unique().tolist())

    rows = [
        {
            "account_type": account_type,
            "media_type": media,
            "content_category": content_category,
            "traffic_source": traffic_source,
            "day_of_week": day,
            "follower_count": follower_count,
            "has_call_to_action": cta,
            "post_hour": hour,
            "caption_length": caption_length,
            "hashtags_count": hashtags_count,
        }
        for media, day, hour, cta in itertools.product(
            media_types, days, range(24), (0, 1)
        )
    ]
    return pd.DataFrame(rows)[FEATURES]


def recommend_setups(pipeline, data, account_type, content_category,
                     traffic_source, follower_count, caption_length,
                     hashtags_count):
    """
    Score every setup with the trained classifier.

    Score = expected engagement rate: each category's probability is weighted
    by that category's real average engagement rate in the dataset. This works
    with any category names, no ordering has to be assumed.
    """
    grid = build_grid(
        data, account_type, content_category, traffic_source,
        follower_count, caption_length, hashtags_count,
    )

    classes = [str(c) for c in pipeline.named_steps["classifier"].classes_]
    means = class_engagement_means(data)
    weights = [float(means.get(c, means.mean())) for c in classes]

    proba = pipeline.predict_proba(grid)
    grid = grid.copy()
    grid["expected_engagement"] = proba @ pd.Series(weights).values

    top_class = classes[int(pd.Series(weights).idxmax())]
    grid["top_tier_probability"] = proba[:, classes.index(top_class)]

    average = grid["expected_engagement"].mean()
    grid["uplift"] = grid["expected_engagement"] / average - 1

    ranked = grid.sort_values("expected_engagement", ascending=False)
    return ranked.reset_index(drop=True), top_class, average