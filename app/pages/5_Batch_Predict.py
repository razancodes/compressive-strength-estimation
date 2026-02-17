"""
Page 5: Batch Predict — CSV upload, batch inference, CSV download.
"""

import sys
import os
import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from styles import apply_styles
from model_loader import (
    engineer_features_batch, predict_batch, load_model,
    MODEL_TYPES, SUBSET_NAMES,
)

apply_styles()

st.header("Batch Predict")
st.markdown("Upload a CSV of concrete mix designs to generate predictions in bulk.")

# ── Template Download ───────────────────────────────────────────────────
st.subheader("CSV Template")
st.caption("Download this template, fill in your data, and upload it below.")

template = pd.DataFrame({
    "Cement": [280.0, 350.0],
    "Blast_Furnace_Slag": [50.0, 100.0],
    "Fly_Ash": [50.0, 0.0],
    "Water": [180.0, 160.0],
    "Superplasticizer": [6.0, 10.0],
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

st.dataframe(template, use_container_width=True, hide_index=True)

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
    try:
        df_input = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Validate columns
    required = ["Cement", "Blast_Furnace_Slag", "Fly_Ash", "Water",
                "Superplasticizer", "Coarse_Aggregate", "Fine_Aggregate", "Age"]
    missing = [c for c in required if c not in df_input.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    st.markdown(f"Uploaded **{len(df_input)}** rows.")

    if st.button("Run Batch Prediction", type="primary", use_container_width=True):
        # Engineer features
        df_feat = engineer_features_batch(df_input[required])

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
