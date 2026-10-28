import numpy as np
import pandas as pd
import streamlit as st

from utils.modeling import original_feature_name


@st.cache_resource(show_spinner=False)
def _get_explainer(_classifier, cache_key):
    """Build the SHAP explainer once. `cache_key` identifies the model."""
    import shap

    return shap.TreeExplainer(_classifier)


def explain_prediction(pipeline, input_df, predicted_class):
    """
    Return a DataFrame of how much each input pushed the model toward
    `predicted_class`. Positive = pushed toward it, negative = pushed away.
    Columns: feature, value, contribution.
    """
    pre = pipeline.named_steps["preprocessing"]
    clf = pipeline.named_steps["classifier"]

    X = pre.transform(input_df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    names = pre.get_feature_names_out()

    explainer = _get_explainer(clf, id(clf))
    shap_values = explainer.shap_values(X)

    class_index = list(clf.classes_).index(str(predicted_class))

    # SHAP returns a list (older versions) or a 3D array (newer versions)
    if isinstance(shap_values, list):
        row = np.asarray(shap_values[class_index])[0]
    else:
        arr = np.asarray(shap_values)
        row = arr[0, :, class_index] if arr.ndim == 3 else arr[0]

    totals = {}
    for name, value in zip(names, row):
        col, _ = original_feature_name(name)
        totals[col] = totals.get(col, 0.0) + float(value)

    result = pd.DataFrame({
        "feature": list(totals.keys()),
        "value": [input_df.iloc[0][col] for col in totals.keys()],
        "contribution": list(totals.values()),
    })
    order = result["contribution"].abs().sort_values(ascending=False).index
    return result.loc[order].reset_index(drop=True)