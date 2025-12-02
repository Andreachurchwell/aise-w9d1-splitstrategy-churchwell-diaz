import os
import pandas as pd
import plotly.express as px

def generate_partner_b_visuals(y_test, y_pred_test, cv_scores, output_dir="partner_b_visuals"):
    """
    Create Plotly visuals for Partner B results.
    """

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1. 5-Fold CV Bar Chart
    # ------------------------------------------------------------
    cv_df = pd.DataFrame({
        "Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
        "R² Score": cv_scores
    })

    fig_cv = px.bar(
        cv_df,
        x="Fold",
        y="R² Score",
        title="Partner B – 5-Fold CV R² Scores",
        text="R² Score",
        color="R² Score",
        color_continuous_scale="Bluered"
    )
    fig_cv.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_cv.update_layout(template="plotly_white", height=500)

    cv_path = os.path.join(output_dir, "cv_r2_bar_chart.html")
    fig_cv.write_html(cv_path)
    print(f"✔ Saved CV bar chart → {cv_path}")

    # ------------------------------------------------------------
    # 2. Actual vs Predicted Scatter
    # ------------------------------------------------------------
    test_df = pd.DataFrame({
        "Actual": y_test.reset_index(drop=True),
        "Predicted": pd.Series(y_pred_test).reset_index(drop=True)
    })

    fig_scatter = px.scatter(
        test_df,
        x="Actual",
        y="Predicted",
        title="Partner B – Actual vs Predicted (Test Set)",
        opacity=0.7
    )

    # add perfect line
    min_val = min(test_df["Actual"].min(), test_df["Predicted"].min())
    max_val = max(test_df["Actual"].max(), test_df["Predicted"].max())

    fig_scatter.add_shape(
        type="line",
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(color="red", dash="dash")
    )
    fig_scatter.update_layout(template="plotly_white", height=600)

    scatter_path = os.path.join(output_dir, "actual_vs_pred_scatter.html")
    fig_scatter.write_html(scatter_path)
    print(f"✔ Saved scatter plot → {scatter_path}")

    # ------------------------------------------------------------
    # 3. Residual Plot
    # ------------------------------------------------------------
    residuals = y_pred_test - y_test.reset_index(drop=True)

    fig_resid = px.scatter(
        x=y_pred_test,
        y=residuals,
        title="Partner B – Residual Plot",
        labels={"x": "Predicted", "y": "Residuals"},
        opacity=0.7
    )
    fig_resid.add_shape(
        type="line",
        x0=min(y_pred_test), y0=0,
        x1=max(y_pred_test), y1=0,
        line=dict(color="red", dash="dash")
    )
    fig_resid.update_layout(template="plotly_white", height=500)

    resid_path = os.path.join(output_dir, "residuals_scatter.html")
    fig_resid.write_html(resid_path)
    print(f"✔ Saved residual plot → {resid_path}")

    # ------------------------------------------------------------
    # 4. Residual Histogram
    # ------------------------------------------------------------
    fig_hist = px.histogram(
        residuals,
        nbins=30,
        title="Partner B – Distribution of Residuals",
        opacity=0.75
    )
    fig_hist.update_layout(template="plotly_white", height=500)

    hist_path = os.path.join(output_dir, "residuals_histogram.html")
    fig_hist.write_html(hist_path)
    print(f"✔ Saved residual histogram → {hist_path}")

    # ------------------------------------------------------------
    # 5. Sorted Actual vs Predicted Line Chart
    # ------------------------------------------------------------
    y_test_sorted = y_test.reset_index(drop=True).sort_values()
    y_pred_sorted = pd.Series(y_pred_test).reset_index(drop=True).iloc[y_test_sorted.index]

    x_vals = list(range(len(y_test_sorted)))

    fig_sorted = px.line(
        x=x_vals,
        y=y_test_sorted,
        title="Partner B – Sorted Actual vs Predicted",
        labels={"x": "Sorted Samples", "y": "Progression Score"},
    )
    fig_sorted.add_scatter(
        x=x_vals,
        y=y_pred_sorted,
        name="Predicted",
        mode="lines"
    )
    fig_sorted.update_layout(template="plotly_white", height=600)

    sorted_path = os.path.join(output_dir, "sorted_actual_vs_pred.html")
    fig_sorted.write_html(sorted_path)
    print(f"✔ Saved sorted comparison → {sorted_path}")

    print("\n🎉 All Partner B visuals generated successfully!")


# ----------------------------------------------------------------
# Example usage:
# ----------------------------------------------------------------
if __name__ == "__main__":
    print("⚠ This file is designed to be imported, not run directly.")
    print("Import generate_partner_b_visuals() inside eval_partner_b.py")
