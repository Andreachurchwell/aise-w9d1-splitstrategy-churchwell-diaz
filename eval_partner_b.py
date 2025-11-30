"""
AISE 26 - W9D1 Split Strategy Showdown
Partner B: Stratified Holdout + Stratified 5-Fold CV
Author: Jose Diaz
Dataset: Diabetes Progression (#7)
Metric: R²

NOTE:
The diabetes dataset is regression, so true stratification is impossible.
Partner B requirement: stratified OR time-aware CV.
Solution:
    • Bin continuous target y into 5 quantile bins
    • Use stratified split + StratifiedKFold on binned labels
"""

import os
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from rich.console import Console

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def bin_target(y, bins=5):
    """
    Convert continuous regression target into discrete bins
    so we can perform stratification.
    """
    return pd.qcut(y, q=bins, labels=False, duplicates="drop")


def main():
    console = Console()
    console.print("\n[bold magenta]=== Partner B – Stratified Evaluation Pipeline ===[/bold magenta]")

    # ----------------------------------------
    # Step 1: Load Dataset
    # ----------------------------------------
    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target

    console.print("\n[bold cyan]📥 Loading Diabetes dataset (#7)[/bold cyan]")
    console.print(f"[yellow]X shape: {X.shape}[/yellow]")
    console.print(f"[yellow]y shape: {y.shape}[/yellow]")

    # ----------------------------------------
    # Step 2: Create Binned y for Stratification
    # ----------------------------------------
    console.print("\n[bold cyan]📊 Creating 5-bin stratified target labels...[/bold cyan]")
    y_binned = bin_target(y, bins=5)

    # ----------------------------------------
    # Step 3: Stratified 80/20 Split
    # ----------------------------------------
    console.print("\n[bold cyan]🧪 Stratified 80/20 Train/Test Split...[/bold cyan]")

    X_train, X_test, y_train, y_test, train_bins, test_bins = train_test_split(
        X,
        y,
        y_binned,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_binned
    )

    console.print("[green]✔ Stratified split successful.[/green]")
    console.print(f"X_train shape: {X_train.shape}")
    console.print(f"X_test  shape: {X_test.shape}")

    # ----------------------------------------
    # Step 4: Build Pipeline (Scaler + Ridge)
    # ----------------------------------------
    console.print("\n[bold cyan]🧠 Training Ridge Regression model...[/bold cyan]")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])

    model.fit(X_train, y_train)
    console.print("[green]✔ Model trained.[/green]")

    # ----------------------------------------
    # Step 5: Training R²
    # ----------------------------------------
    y_pred_train = model.predict(X_train)
    train_r2 = r2_score(y_train, y_pred_train)

    # ----------------------------------------
    # Step 6: Test R² on the 20% holdout set
    # ----------------------------------------
    y_pred_test = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)

    console.print("\n[bold cyan]📘 Performance Results[/bold cyan]")
    console.print(f"Train R²: {train_r2:.4f}")
    console.print(f"Test R² : {test_r2:.4f}")

    # ----------------------------------------
    # Step 7: Stratified 5-Fold CV (on training set only)
    # ----------------------------------------
    console.print("\n[bold cyan]📚 Running Stratified 5-Fold CV...[/bold cyan]")

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=skf,
        scoring="r2"
    )

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    console.print("\n--- 5-Fold CV Results (Partner B) ---")
    console.print(f"Fold scores: {cv_scores}")
    console.print(f"CV Mean R²: {cv_mean:.4f}")
    console.print(f"CV Std  R²: {cv_std:.4f}")

    # ----------------------------------------
    # Step 8: Summary Output
    # ----------------------------------------
    console.print("\n✨ Partner B Summary")
    console.print(f"📘 Train R²   : {train_r2:.4f}")
    console.print(f"📗 Test R²    : {test_r2:.4f}")
    console.print(f"📙 CV Mean R² : {cv_mean:.4f}")
    console.print(f"📒 CV Std R²  : {cv_std:.4f}")

    # ----------------------------------------
    # Step 9: Append results to comparison.csv
    # ----------------------------------------
    comp_path = "comparison.csv"

    comp_df = pd.DataFrame({
        "strategy": ["partner_b"] * len(cv_scores),
        "fold": [1, 2, 3, 4, 5],
        "score": cv_scores
    })

    if not os.path.exists(comp_path):
        comp_df.to_csv(comp_path, index=False)
    else:
        comp_df.to_csv(comp_path, mode="a", index=False, header=False)

    console.print("\n✔ Appended Partner B scores to comparison.csv")
    console.print("🎉 Partner B evaluation complete!\n")


if __name__ == "__main__":
    main()
