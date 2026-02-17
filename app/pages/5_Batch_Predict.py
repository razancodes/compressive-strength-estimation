"""
Page 5: Batch Predict — CSV upload, batch inference, CSV download.

Security: validates file size, row count, column names, and CSV injection.
Missing values are auto-filled with engineering best-practice defaults.
"""

import sys
import os
import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from styles import apply_styles
from model_loader import (
    engineer_features_batch, predict_batch, load_model, autofill_missing,
    MODEL_TYPES, SUBSET_NAMES, INPUT_FIELDS, DEFAULT_FILL_VALUES,
)

apply_styles()

st.header("Batch Predict")
st.markdown("Upload a CSV of concrete mix designs to generate predictions in bulk. "
            "Missing values are automatically filled with best-practice defaults.")

# ── Constants ───────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 5
MAX_ROWS = 10_000
REQUIRED_COLS = [f["name"] for f in INPUT_FIELDS]

# ── Template Download ───────────────────────────────────────────────────
st.subheader("CSV Template")
st.caption("Download this template, fill in your data, and upload it below. "
           "Leave cells empty if values are unavailable -- they will be auto-filled.")

template = pd.DataFrame({
    "Cement": [280.0, 350.0],
    "Blast_Furnace_Slag": [50.0, ""],
    "Fly_Ash": [50.0, ""],
    "Water": [180.0, 160.0],
    "Superplasticizer": [6.0, ""],
    "Coarse_Aggregate": [970.0, 1000.0],
    "Fine_Aggregate": [770.0, 800.0],
    "Age": [28.0, 7.0],
})

csv_template = template.to_csv(index=False)
st.download_button(
    "Download Template CSV",
    data=csv_template,
    file_name="concrete_mix_template.csv",
    mime="text/csv",
)

# Show defaults reference
with st.expander("Auto-fill defaults reference"):
    defaults_df = pd.DataFrame([
        {"Variable": k, "Default": v,
         "Rationale": "0 = not used in mix" if v == 0 else
                      "28-day standard (ASTM C39)" if k == "Age" else
                      "Training data median"}
        for k, v in DEFAULT_FILL_VALUES.items()
    ])
    st.dataframe(defaults_df, use_container_width=True, hide_index=True)

# ── Upload and Predict ──────────────────────────────────────────────────
st.divider()
st.subheader("Upload and Predict")

c1, c2 = st.columns(2)
with c1:
    model_type = st.selectbox("Model", MODEL_TYPES, index=1, key="batch_model")
with c2:
    subset = st.selectbox("Subset", SUBSET_NAMES, index=3, key="batch_subset")

uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")

if uploaded is not None:

    # ── 1. File size check ──────────────────────────────────────────
    file_size_mb = uploaded.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large ({file_size_mb:.1f} MB). "
                 f"Maximum allowed: {MAX_FILE_SIZE_MB} MB.")
        st.stop()

    # ── 2. Parse CSV safely ─────────────────────────────────────────
    try:
        raw_text = uploaded.getvalue().decode("utf-8", errors="replace")
        if re.search(r'(?:^|,)\s*[=+@]', raw_text):
            st.error("CSV contains cells starting with =, +, or @ which "
                     "are not allowed for security reasons.")
            st.stop()
        df_input = pd.read_csv(io.StringIO(raw_text))
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # ── 3. Row count check ──────────────────────────────────────────
    if len(df_input) == 0:
        st.error("CSV is empty.")
        st.stop()
    if len(df_input) > MAX_ROWS:
        st.error(f"Too many rows ({len(df_input):,}). "
                 f"Maximum allowed: {MAX_ROWS:,}.")
        st.stop()

    # ── 4. Column validation ────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLS if c not in df_input.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    # Keep only required columns
    df_input = df_input[REQUIRED_COLS].copy()

    # ── 5. Coerce to numeric ────────────────────────────────────────
    for col in REQUIRED_COLS:
        df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

    # ── 6. Auto-fill missing values ─────────────────────────────────
    df_input, fill_log = autofill_missing(df_input)

    if fill_log:
        st.info("**Auto-filled missing values:**\n- " + "\n- ".join(fill_log))

    st.markdown(f"Ready: **{len(df_input)}** rows after validation and auto-fill.")

    if st.button("Run Batch Prediction", type="primary", use_container_width=True):
        # Engineer features
        df_feat = engineer_features_batch(df_input)

        # Load model and predict
        model = load_model(model_type, subset)
        predictions = predict_batch(model, df_feat)

        # Combine results
        df_result = df_input.copy()
        df_result["Predicted_Strength_MPa"] = np.round(predictions, 2)

        # Display
        st.subheader("Results")
        st.dataframe(df_result, use_container_width=True, hide_index=True)

        # Summary
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Mean Prediction", f"{predictions.mean():.2f} MPa")
        with c2:
            st.metric("Min", f"{predictions.min():.2f} MPa")
        with c3:
            st.metric("Max", f"{predictions.max():.2f} MPa")

        # Chart
        st.subheader("Batch Predictions Overview")
        fig = px.scatter(
            df_result,
            x="Age",
            y="Predicted_Strength_MPa",
            color="Cement",
            color_continuous_scale="viridis",
            labels={
                "Age": "Age (days)",
                "Predicted_Strength_MPa": "Predicted Strength (MPa)",
                "Cement": "Cement (kg/m3)",
            },
            template="plotly_white",
        )
        fig.update_layout(height=400, margin=dict(l=40, r=40, t=30, b=40))
        st.plotly_chart(fig, use_container_width=True)

        # Download
        csv_result = df_result.to_csv(index=False)
        st.download_button(
            "Download Results CSV",
            data=csv_result,
            file_name="concrete_predictions.csv",
            mime="text/csv",
        )
