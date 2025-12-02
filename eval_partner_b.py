"""
AISE 26 - W9D1 Split Strategy Showdown
Partner B: Ordered Holdout + 5-Fold Time-Aware CV
Author: Jose Diaz
Dataset: Diabetes Progression (#7)
Metric: R²
"""

import os
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# Optional visualizations (Partner B extra credit)
import plotly.express as px

from rich.console import Console

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def generate_partner_b_visuals(y_test, y_pred_test, cv_scores):
    """
    Partner B Plotly visualizations saved under partner_b_visuals/
    """
    os.makedirs("partner_b_visuals", exist_ok=True)

    # --- 1. CV Bar Chart ---
    cv_df = pd.DataFrame({
        "Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
        "R² Score": cv_scores
    })

    fig_cv = px.bar(
        cv_df,
        x="Fold",
        y="R² Score",
        title="Partner B – 5-Fold Time-Aware CV R²",
        text="R² Score",
        color="R² Score",
        color_continuous_scale="Bluered"
    )
    fig_cv.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_cv.write_html("partner_b_visuals/cv_r2_bar_chart.html")

    # --- 2. Actual vs Predicted Scatter ---
    test_df = pd.DataFrame({
        "Actual": y_test.reset_index(drop=True),
        "Predicted": pd.Series(y_pred_test).reset_index(drop=True)
    })

    fig_scatter = px.scatter(
        test_df,
        x="Actual",
        y="Predicted",
        title="Partner B – Actual vs Predicted (Test Set)",
        opacity=0.7,
    )
    fig_scatter.write_html("partner_b_visuals/actual_vs_pred_scatter.html")

    # --- 3. Residual Histogram ---
    residuals = y_pred_test - y_test.reset_index(drop=True)

    fig_resid = px.histogram(
        residuals,
        nbins=30,
        title="Partner B – Residual Distribution (Test Set)"
    )
    fig_resid.write_html("partner_b_visuals/residual_histogram.html")


def main():
    console = Console()

    console.print("\n[bold magenta]=== Partner B – Evaluation Pipeline ===[/bold magenta]")
    console.print("[bold cyan]AISE 26 - W9D1 Split Strategy Showdown[/bold cyan]")
    console.print("[white]Partner B: Ordered Holdout + Time-Aware CV (KFold, shuffle=False)[/white]\n")

    # --------------------------------------------
    # Step 1 – Load Dataset
    # --------------------------------------------
    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target

    # --------------------------------------------
    # Step 2 – Ordered Holdout Split (NOT random)
    # --------------------------------------------
    # Equivalent to time-aware split: first 80% train, last 20% test
    split_idx = int(len(X) * 0.8)

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]

    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    console.print(f"[yellow]Holdout strategy: First 80% → Train, Last 20% → Test[/yellow]")
    console.print(f"[yellow]X_train: {X_train.shape}, X_test: {X_test.shape}[/yellow]\n")

    # --------------------------------------------
    # Step 3 – Build Pipeline
    # --------------------------------------------
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])

    model.fit(X_train, y_train)

    # --------------------------------------------
    # Step 4 – Train/Test Performance
    # --------------------------------------------
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    console.print(f"[green]Train R²: {train_r2:.4f}[/green]")
    console.print(f"[green]Test  R²: {test_r2:.4f}[/green]\n")

    # --------------------------------------------
    # Step 5 – TimeSeriesSplit 
    # --------------------------------------------
    
    tscv = TimeSeriesSplit(n_splits=5)

    cv_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        cv_scores.append(r2_score(y_val, preds))
    
    cv_scores = np.array(cv_scores)

    console.print("[cyan]--- 5-Fold TimeSeriesSplit Results (Partner B) ---[/cyan]")
    console.print(f"Fold scores: {cv_scores}")
    console.print(f"CV Mean R²: {cv_scores.mean():.4f}")
    console.print(f"CV Std  R²: {cv_scores.std():.4f}\n")

    # --------------------------------------------
    # Step 6 – Save to comparison.csv
    # --------------------------------------------
    df_out = pd.DataFrame({
        "strategy": ["partner_b"] * 5,
        "fold": [1, 2, 3, 4, 5],
        "score": cv_scores
    })

    if not os.path.exists("comparison.csv"):
        df_out.to_csv("comparison.csv", index=False)
    else:
        df_out.to_csv("comparison.csv", mode="a", header=False, index=False)

    console.print("[green]✔ Appended Partner B scores to comparison.csv[/green]\n")

    # --------------------------------------------
    # Step 7 – Visualizations
    # --------------------------------------------
    generate_partner_b_visuals(y_test, y_pred_test, cv_scores)

    console.print("[bold green]🎉 Partner B evaluation complete![/bold green]\n")


if __name__ == "__main__":
    main()
