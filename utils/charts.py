import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.config import order_days, hour_label
from utils.theme import (
    INK, LINE, MAGENTA, MUTED, ORANGE, PALETTE, SEQ_CMAP, TEAL, VIOLET,
)


# ---------------------------------------------------------------------------
# Day x hour heatmap
# ---------------------------------------------------------------------------
def day_hour_tables(data, metric, min_posts=1):
    """Return (average table, post-count table) with days as rows, hours as columns."""
    days = order_days(data["day_of_week"].dropna().unique().tolist())
    hours = list(range(24))

    frame = data[["day_of_week", "post_hour", metric]].dropna().copy()
    frame["post_hour"] = frame["post_hour"].astype(int)

    average = frame.pivot_table(
        index="day_of_week", columns="post_hour", values=metric, aggfunc="mean"
    ).reindex(index=days, columns=hours)
    counts = frame.pivot_table(
        index="day_of_week", columns="post_hour", values=metric, aggfunc="count"
    ).reindex(index=days, columns=hours).fillna(0)

    average = average.where(counts >= min_posts)
    return average, counts


def day_hour_heatmap(average, as_percent=False):
    fig, ax = plt.subplots(figsize=(13, 4.6))
    display = average * 100 if as_percent else average
    sns.heatmap(
        display,
        cmap=SEQ_CMAP,
        linewidths=1.5,
        linecolor="white",
        cbar_kws={"label": "Average engagement (%)" if as_percent else "Average value", "shrink": 0.85},
        ax=ax,
    )
    ax.set_xticklabels([hour_label(h) for h in average.columns], rotation=0, fontsize=8)
    ax.set_yticklabels(average.index, rotation=0)
    ax.set_xlabel("Hour posted")
    ax.set_ylabel("")
    return fig


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
def comparison_chart(results, best_name):
    ordered = results.sort_values("Macro F1")
    colors = [
        MAGENTA if m == best_name else (MUTED if m.startswith("Baseline") else VIOLET)
        for m in ordered["Model"]
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        ordered["Model"], ordered["Macro F1"],
        xerr=ordered["Macro F1 SD"], color=colors, height=0.6,
        error_kw={"ecolor": INK, "capsize": 3, "lw": 1},
    )
    ax.set_xlabel("Macro F1 (higher is better)")
    ax.set_xlim(0, max(0.05, float(ordered["Macro F1"].max()) * 1.25))
    return fig


def confusion_chart(cm, class_names):
    row_totals = cm.sum(axis=1, keepdims=True)
    normalised = np.divide(cm, row_totals, out=np.zeros(cm.shape, dtype=float), where=row_totals != 0)
    fig, ax = plt.subplots(figsize=(6, 4.6))
    sns.heatmap(
        normalised, annot=cm, fmt="d", cmap=SEQ_CMAP, vmin=0, vmax=1,
        linewidths=2, linecolor="white",
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "Share of true class", "shrink": 0.85}, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig


def roc_chart(roc):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for i, (cls, (fpr, tpr, auc)) in enumerate(roc.items()):
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=2.2, label=f"{cls} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], color=LINE, lw=1.5, linestyle="--", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# Explainability and importance
# ---------------------------------------------------------------------------
def importance_chart(importance):
    ordered = importance.sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(ordered["feature"], ordered["importance"] * 100, color=VIOLET, height=0.6)
    ax.set_xlabel("Share of the model's decisions (%)")
    return fig


def contribution_chart(contributions, predicted_class):
    top = contributions.head(10).iloc[::-1]
    labels = [f"{f} = {v}" for f, v in zip(top["feature"], top["value"])]
    colors = [MAGENTA if c > 0 else VIOLET for c in top["contribution"]]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(labels, top["contribution"], color=colors, height=0.6)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel(f"Pushes toward '{predicted_class}' (right)  |  away from it (left)")
    return fig


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------
def media_type_chart(ranked):
    best = (
        ranked.groupby("media_type")["expected_engagement"].max().sort_values()
    )
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(best.index, best.values * 100, color=[ORANGE if i == len(best) - 1 else TEAL for i in range(len(best))], height=0.6)
    ax.set_xlabel("Best expected engagement (%)")
    return fig