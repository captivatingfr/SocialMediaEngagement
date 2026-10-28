from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Instagram_Analytics.csv"

CATEGORICAL_FEATURES = [
    "account_type", "media_type", "content_category",
    "traffic_source", "day_of_week"
]

NUMERIC_FEATURES = [
    "follower_count", "has_call_to_action", "post_hour",
    "caption_length", "hashtags_count",
    "likes", "comments", "shares", "saves",
    "reach", "impressions", "engagement_rate", "followers_gained"
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TARGET = "performance_bucket_label"

DAY_ORDERS = [
    ["Monday", "Tuesday", "Wednesday", "Thursday",
     "Friday", "Saturday", "Sunday"],
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
]


def order_days(values):
    """Return day names in calendar order when they match a known format."""
    values = list(values)
    for order in DAY_ORDERS:
        if set(values).issubset(set(order)):
            return [d for d in order if d in values]
    return sorted(values)


def hour_label(hour):
    """18 -> '6 PM', 0 -> '12 AM'."""
    hour = int(hour)
    suffix = "AM" if hour < 12 else "PM"
    return f"{(hour % 12) or 12} {suffix}"