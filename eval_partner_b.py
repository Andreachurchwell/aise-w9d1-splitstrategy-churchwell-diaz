"""
AISE 26 - W9D1 Split Strategy Showdown
Partner B: Holdout + SAME 5-Fold CV as Partner A
Author: Jose Diaz
Dataset: Diabetes Progression (#7)
Metric: R²

This script mirrors Partner A’s evaluation pipeline but represents
Partner B’s version of the split strategy:

✔ Uses the SAME 80/20 holdout split (random_state=42)
✔ Uses the SAME 5-Fold KFold cross-validation setup (random_state=42)
✔ Uses the SAME model pipeline: StandardScaler → Ridge
✔ Computes R² metrics (Train/Test/CV)
✔ Saves Partner B scores into comparison.csv
"""

import os

import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from rich.console import Console

# Shared global seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def main():
    console = Console()

    # Header
    console.print("\n[bold magenta]=== Partner B – Evaluation Pipeline ===[/bold magenta]")
    console.print("[bold cyan]AISE 26 - W9D1 Split Strategy Showdown[/bold cyan]")
    console.print("Partner B: Holdout + SAME 5-Fold CV as Partner A\n", style="bold")

    # -------------------------------------------------
    # Step 1 — Load Diabetes dataset
    # -------------------------------------------------
    console.print("[bold cyan]📥 Step 1: Loading Diabetes dataset (#7)...[/bold cyan]")

    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target

    console.print(f"[yellow]• Features shape: {X.shape}[/yellow]")
    console.print(f"[yellow]• Target shape:   {y.shape}[/yellow]")

    # -------------------------------------------------
    # Step 2 — Exact same 80/20 holdout split
    # -------------------------------------------------
    console.print("\n[bold cyan]📊 Step 2: Performing 80/20 Holdout Split...[/bold cyan]")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    console.print("[green]✔ Holdout split complete.[/green]")
    console.print(f"X_train: {X_train.shape}")
    console.print(f"X_test : {X_test.shape}")

    # -------------------------------------------------
    # Step 3 — Model Pipeline
    # -------------------------------------------------
    console.print("\n[bold cyan]🧠 Step 3: Building Ridge Regression Pipeline...[/bold cyan]")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])

    model.fit(X_train, y_train)
    console.print("[green]✔ Model fitted on training data.[/green]")

    # -------------------------------------------------
    # Step 4 — Train/Test Performance
    # -------------------------------------------------
    console.print("\n[bold cyan]📘 Step 4: Train/Test R² Performance...[/bold cyan]")

    # Training R²
    y_pred_train = model.predict(X_train)
    train_r2 = r2_score(y_train, y_pred_train)

    # Test R²
    y_pred_test = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)

    console.print(f"[green]• Train R²: {train_r2:.4f}")
    console.print(f"[green]• Test  R²: {test_r2:.4f}")

    # -------------------------------------------------
    # Step 5 — SAME KFold CV as Partner A
    # -------------------------------------------------
    console.print("\n[bold cyan]📚 Step 5: SAME 5-Fold Cross-Validation...[/bold cyan]")

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE  # ensures same 5 folds
    )

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kfold,
        scoring="r2"
    )

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    console.print("\n--- 5-Fold CV Results (Partner B) ---")
    console.print(f"Fold scores: {cv_scores}")
    console.print(f"CV Mean R²: {cv_mean:.4f}")
    console.print(f"CV Std  R²: {cv_std:.4f}")

    # -------------------------------------------------
    # Step 6 — Summary
    # -------------------------------------------------
    console.print("\n✨ [bold]Partner B Summary[/bold]")
    console.print(f"📘 Train R²   : {train_r2:.4f}")
    console.print(f"📗 Test R²    : {test_r2:.4f}")
    console.print(f"📙 CV Mean R² : {cv_mean:.4f}")
    console.print(f"📒 CV Std R²  : {cv_std:.4f}")

    # -------------------------------------------------
    # Step 7 — Append to comparison.csv
    # -------------------------------------------------
    comparison_path = "comparison.csv"

    comp_df = pd.DataFrame({
        "strategy": ["partner_b"] * len(cv_scores),
        "fold": [1, 2, 3, 4, 5],
        "score": cv_scores
    })

    if not os.path.exists(comparison_path):
        comp_df.to_csv(comparison_path, index=False)
        console.print(f"\n[green]✔ Created comparison.csv with Partner B scores[/green]")
    else:
        comp_df.to_csv(comparison_path, mode="a", header=False, index=False)
        console.print(f"\n[green]✔ Appended Partner B scores to comparison.csv[/green]")

    console.print("\n🎉 Partner B evaluation complete!\n")



if __name__ == "__main__":
    main()
