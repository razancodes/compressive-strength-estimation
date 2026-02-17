"""
Page 2: Strength Curve — Multi-age strength development projection.
"""

import sys
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from styles import apply_styles
from model_loader import (
    render_input_form, engineer_features, predict, load_model,
    select_subset_for_age, autofill_single_input,
    MODEL_TYPES,
)

apply_styles()

st.header("Strength Development Curve")
st.markdown("Enter a mix design (without age) to project strength development "
            "from 1 to 365 days. Check N/A for any unavailable value.")

# ── Input ───────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.6])

with left:
    st.subheader("Mix Design")
    raw_input = render_input_form(include_age=False, key_prefix="curve_")

    st.divider()

    model_type = st.selectbox("Model", MODEL_TYPES, index=1, key="curve_model")

    show_all = st.checkbox("Overlay all three models", value=False)

    # Age points to predict
    ages = [1, 3, 7, 14, 28, 56, 90, 180, 365]

    run = st.button("Generate Curve", type="primary", use_container_width=True)

with right:
    if run:
        # Auto-fill any N/A values
        filled_input, fill_log = autofill_single_input(raw_input)
        if fill_log:
            st.info("**Auto-filled values:**\n- " + "\n- ".join(fill_log))

        models_to_plot = MODEL_TYPES if show_all else [model_type]
        colors = {"XGBoost": "#2196F3", "CatBoost": "#E64A19", "LightGBM": "#388E3C"}

        fig = go.Figure()
        results_table = []

        for mt in models_to_plot:
            predictions = []
            subsets_used = []

            for age in ages:
                inp = dict(filled_input)
                inp["Age"] = float(age)
                subset = select_subset_for_age(age)
                subsets_used.append(subset)

                model = load_model(mt, subset)
                features_df = engineer_features(inp)
                strength = predict(model, features_df)
                predictions.append(strength)

                if mt == (models_to_plot[0]):
                    results_table.append({
                        "Age (days)": age,
                        "Subset": subset,
                        f"Strength (MPa)": round(strength, 2),
                    })

            fig.add_trace(go.Scatter(
                x=ages,
                y=predictions,
                mode="lines+markers",
                name=mt,
                line=dict(color=colors.get(mt, "#666"), width=2),
                marker=dict(size=8),
            ))

        # Reference lines
        for threshold in [20, 30, 40]:
            fig.add_hline(
                y=threshold, line_dash="dot", line_color="#ccc",
                annotation_text=f"{threshold} MPa",
                annotation_position="right",
            )

        fig.update_layout(
            xaxis_title="Age (days)",
            yaxis_title="Compressive Strength (MPa)",
            xaxis_type="log",
            xaxis=dict(
                tickmode="array",
                tickvals=ages,
                ticktext=[str(a) for a in ages],
            ),
            height=500,
            margin=dict(l=40, r=40, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Results table
        st.subheader("Predicted Values")
        if not show_all:
            df_results = pd.DataFrame(results_table)
            st.dataframe(df_results, use_container_width=True, hide_index=True)
        else:
            # Build multi-model table
            table_data = []
            for age in ages:
                row = {"Age (days)": age, "Subset": select_subset_for_age(age)}
                for mt in MODEL_TYPES:
                    inp = dict(filled_input)
                    inp["Age"] = float(age)
                    subset = select_subset_for_age(age)
                    model = load_model(mt, subset)
                    features_df = engineer_features(inp)
                    row[mt] = round(predict(model, features_df), 2)
                table_data.append(row)
            st.dataframe(pd.DataFrame(table_data), use_container_width=True,
                         hide_index=True)

    else:
        st.info("Enter a mix design and click Generate Curve.")
