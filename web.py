"""Streamlit deployment: python -m streamlit run web.py"""
import hashlib
import pickle
from pathlib import Path

import numpy as np

import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components

BASE_DIR = Path(__file__).resolve().parent
FEATURES = [
    "Age>50", "Gallbladder size(0=Normal, 1=Enlarged)",
    "Gallbladder wall thickening(0=<5mm, 1=≥5mm)", "Blood Type A", "ALP",
    "Hyperlipidemia", "Hypocalcemia", "Operative time",
    "Fasting time(0=<12h, 1=≥12h)",
]
LABELS = ["Age > 50 years", "Enlarged gallbladder", "Gallbladder wall ≥ 5 mm",
          "Blood type A", "ALP", "Hyperlipidemia", "Hypocalcemia",
          "Operative time", "Fasting time ≥ 12 hours"]
CONTINUOUS = ["ALP", "Operative time"]
BINARY = [name for name in FEATURES if name not in CONTINUOUS]
# Units were not specified in the supplied data dictionary.
UNITS = {"ALP": "same unit as training data", "Operative time": "same unit as training data"}


DEFAULT_INPUTS = {'Age>50': 1.0, 'Gallbladder size(0=Normal, 1=Enlarged)': 0.0, 'Gallbladder wall thickening(0=<5mm, 1=≥5mm)': 0.0, 'Blood Type A': 0.0, 'ALP': 76.0, 'Hyperlipidemia': 0.0, 'Hypocalcemia': 0.0, 'Operative time': 50.0, 'Fasting time(0=<12h, 1=≥12h)': 0.0}
MODEL_SHA256 = '484b044873d1a4c48b2863e4c5a546a246d5035b00b4656bd9049013b6b08bf0'

def validate_features(df):
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    x = df.loc[:, FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    if len(x) == 0 or not np.isfinite(x.to_numpy()).all():
        raise ValueError("All nine features must be present, numeric and finite.")
    for name in BINARY:
        if not x[name].isin([0, 1]).all():
            raise ValueError(f"{name} must be 0 (No) or 1 (Yes).")
    for name in CONTINUOUS:
        if (x[name] < 0).any():
            raise ValueError(f"{name} must not be negative.")
    return x


st.set_page_config(page_title="Prediction model for PONV", page_icon="🩺", layout="centered")

@st.cache_resource
def load_model(model_mtime):
    with (BASE_DIR / "ponv_logistic.pkl").open("rb") as stream:
        model = pickle.load(stream)
    if model.named_steps["classifier"].penalty is not None:
        raise ValueError("The deployed model must be unpenalized logistic regression.")
    if hashlib.sha256((BASE_DIR / "ponv_logistic.pkl").read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("The model file does not match this deployment.")
    if list(model.feature_names_in_) != FEATURES:
        raise ValueError("Model feature order does not match this application.")
    # The source CSV uses status=0 for PONV, per the corrected event definition.
    # Training recodes the outcome, so model class 1 is PONV; predictors are not recoded.
    if getattr(model, "outcome_definition_", {}).get("source_positive_value") != 0:
        raise ValueError("This model does not use the corrected PONV outcome definition.")
    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected model classes.")
    return model

def show_force_plot(model, row):
    # Exact interventional SHAP in log-odds for a linear model:
    # phi_i = beta_i * (standardized_x_i - E[standardized_X_i]).
    # StandardScaler centers the training data, so its standardized means are zero.
    classifier = model.named_steps["classifier"]
    contributions = model.named_steps["scaler"].transform(row)[0] * classifier.coef_[0]
    force = shap.force_plot(
        float(classifier.intercept_[0]),
        contributions,
        features=row.iloc[0].to_numpy(),
        feature_names=LABELS,
        out_names="PONV probability",
        link="logit",
        plot_cmap=["#FF0051", "#008BFB"],
    )
    components.html(
        "<html><head>" + shap.getjs() +
        "</head><body style='margin:0;background:white;'>" +
        force.html() + "</body></html>",
        height=200,
        scrolling=True,
    )


def main():
    # Keep the light appearance without requiring .streamlit/config.toml.
    st.markdown("""
    <style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #ffffff; color: #31333f; color-scheme: light;
        --primary-color: #ff4b4b; --background-color: #ffffff;
        --secondary-background-color: #f0f2f6; --text-color: #31333f;
    }
    [data-testid="stSidebar"] {background-color: #f0f2f6; color: #31333f;}
    .stApp h1, .stApp h2, .stApp h3, .stApp p,
    .stApp label, .stApp [data-testid="stMarkdownContainer"] {color: #31333f;}
    .stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"],
    .stApp input, .stApp [data-testid="stNumberInput"] button {
        background-color: #f0f2f6; color: #31333f;
    }
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="popover"] [role="option"] {background-color: #ffffff; color: #31333f;}
    .stApp [data-testid="stButton"] button {
        background-color: #ffffff; color: #31333f; border: 1px solid #d6d6d8;
    }
    .stApp [data-testid="stButton"] button:hover {color: #ff4b4b; border-color: #ff4b4b;}
    </style>
    """, unsafe_allow_html=True)
    st.title("Prediction model for PONV")

    with st.sidebar:
        st.header("Variable Descriptions")
        descriptions = [
            "No: age ≤ 50 years. Yes: age > 50 years.",
            "No: normal gallbladder size. Yes: enlarged gallbladder.",
            "No: wall thickness < 5 mm. Yes: wall thickness ≥ 5 mm.",
            "No: non-A blood type. Yes: blood type A.",
            "Alkaline phosphatase level. Use the same unit as the training data.",
            "No: absence of hyperlipidemia. Yes: presence of hyperlipidemia.",
            "No: absence of hypocalcemia. Yes: presence of hypocalcemia.",
            "Duration of the operation. Use the same unit as the training data.",
            "No: fasting time < 12 hours. Yes: fasting time ≥ 12 hours.",
        ]
        for label, description in zip(LABELS, descriptions):
            st.markdown(f"**{label}**")
            st.write(description)


    try:
        model = load_model((BASE_DIR / "ponv_logistic.pkl").stat().st_mtime_ns)
    except (OSError, ValueError, pickle.UnpicklingError) as exc:
        st.error(f"Model loading failed: {exc}")
        return

    values = {}
    for name, label in zip(FEATURES, LABELS):
        if name in BINARY:
            values[name] = st.selectbox(
                label, options=[0, 1], index=int(DEFAULT_INPUTS[name]),
                format_func=lambda value: "No" if value == 0 else "Yes", key=name)
        else:
            values[name] = st.number_input(
                label, min_value=0.0, value=float(DEFAULT_INPUTS[name]),
                step=1.0, format="%.2f", key=name, help=UNITS[name])

    if st.button("Predict"):
        try:
            row = validate_features(pd.DataFrame([values], columns=FEATURES))
            probability = float(model.predict_proba(row)[0, 1])
        except ValueError as exc:
            st.error(str(exc))
            return
        st.markdown(
            f"### *Based on feature values, predicted probability of PONV is {probability:.2%}.*"
        )
        show_force_plot(model, row)


if __name__ == "__main__":
    main()
