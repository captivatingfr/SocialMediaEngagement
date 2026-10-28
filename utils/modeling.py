import time

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from utils.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)


# ---------------------------------------------------------------------------
# Original classifier (unchanged behaviour)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Feature importance (grouped back to the original columns)
# ---------------------------------------------------------------------------
def original_feature_name(encoded_name):
    """'categorical__media_type_Reel' -> ('media_type', 'Reel')."""
    rest = encoded_name.split("__", 1)[-1]
    for col in CATEGORICAL_FEATURES:
        if rest.startswith(col + "_"):
            return col, rest[len(col) + 1:]
    return rest, None


def feature_importance(pipeline):
    """Random Forest importance summed over each column's one-hot pieces."""
    pre = pipeline.named_steps["preprocessing"]
    clf = pipeline.named_steps["classifier"]
    names = pre.get_feature_names_out()

    totals = {}
    for name, value in zip(names, clf.feature_importances_):
        col, _ = original_feature_name(name)
        totals[col] = totals.get(col, 0.0) + float(value)

    result = pd.DataFrame(
        {"feature": list(totals.keys()), "importance": list(totals.values())}
    )
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
def _dense_preprocessor():
    return ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            CATEGORICAL_FEATURES,
        ),
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
    ])


def _candidate_models():
    models = {
        "Baseline (most common class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150, random_state=42
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=1,
        )
    except Exception:
        pass  # XGBoost is optional; the comparison still works without it
    return models


@st.cache_resource(show_spinner=False)
def compare_models(data, max_rows=20000, folds=5):
    """Cross-validate several models and evaluate the best one on a hold-out set."""
    model_data = data[FEATURES + [TARGET]].dropna()
    if len(model_data) > max_rows:
        model_data = model_data.sample(max_rows, random_state=42)

    X = model_data[FEATURES]
    encoder = LabelEncoder()
    y = encoder.fit_transform(model_data[TARGET].astype(str))
    class_names = list(encoder.classes_)
    n_classes = len(class_names)

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "f1": "f1_macro"}

    rows = []
    candidates = _candidate_models()
    for name, estimator in candidates.items():
        pipe = Pipeline([("pre", _dense_preprocessor()), ("clf", estimator)])
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Accuracy SD": scores["test_accuracy"].std(),
            "Macro F1": scores["test_f1"].mean(),
            "Macro F1 SD": scores["test_f1"].std(),
            "Fit time (s)": scores["fit_time"].mean(),
        })

    results = pd.DataFrame(rows)
    baseline_name = "Baseline (most common class)"
    baseline_acc = float(
        results.loc[results["Model"] == baseline_name, "Accuracy"].iloc[0]
    )

    contenders = results[results["Model"] != baseline_name]
    best_name = contenders.sort_values("Macro F1", ascending=False).iloc[0]["Model"]

    # Hold-out evaluation of the best model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    best_pipe = Pipeline([
        ("pre", _dense_preprocessor()),
        ("clf", candidates[best_name]),
    ])
    best_pipe.fit(X_train, y_train)
    predictions = best_pipe.predict(X_test)
    proba = best_pipe.predict_proba(X_test)

    cm = confusion_matrix(y_test, predictions, labels=list(range(n_classes)))

    y_bin = (y_test[:, None] == np.arange(n_classes)[None, :]).astype(int)
    roc = {}
    for i, cls in enumerate(class_names):
        if y_bin[:, i].sum() == 0 or y_bin[:, i].sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        roc[cls] = (fpr, tpr, roc_auc_score(y_bin[:, i], proba[:, i]))

    return {
        "results": results,
        "best_name": best_name,
        "baseline_acc": baseline_acc,
        "class_names": class_names,
        "confusion": cm,
        "roc": roc,
        "n_rows": len(model_data),
        "folds": folds,
    }