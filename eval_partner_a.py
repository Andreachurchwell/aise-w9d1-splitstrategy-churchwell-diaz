# """
# AISE 26 - W9D1 Split Strategy Showdown
# Partner A: Random Holdout + 5-Fold Standard CV
# Author: Andrea Churchwell
# Dataset: Diabetes Progression (#7)
# Metric: R²

# This file is the required Partner A script for the assignment.
# I'm writing it with very detailed comments so I understand every step.

# What this script does:
#   1. Load the diabetes dataset from sklearn
#   2. Perform an 80/20 random train/test split (Partner A requirement)
#   3. Build a Pipeline: StandardScaler -> Ridge (no hyperparameter tuning)
#   4. Compute:
#        - Train R²
#        - Test R² (20% holdout)
#        - 5-fold cross-validation R² on the training set only
#   5. Generate and SAVE Plotly visualizations into a folder called
#      `partner_a_visuals` so they can be reused in the report / README.
# """

# # ---------------------------------------------------------
# # Step 0: Imports and global random seed
# # ---------------------------------------------------------

# import os  # for creating the visuals folder

# import numpy as np
# import pandas as pd

# from sklearn.datasets import load_diabetes
# from sklearn.model_selection import train_test_split, KFold, cross_val_score
# from sklearn.linear_model import Ridge
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score

# import plotly.express as px  # for all visualizations

# # Set a global random seed so that our results are reproducible.
# # We'll use this same RANDOM_STATE everywhere that supports it.
# RANDOM_STATE = 42
# np.random.seed(RANDOM_STATE)


# def main():
#     """
#     Main function: runs the whole Partner A pipeline.

#     Steps:
#       1. Load the diabetes dataset
#       2. Inspect its shape and a few rows
#       3. Create an 80/20 random train/test split (Partner A requirement)
#       4. Build and train a Ridge regression model (with scaling)
#       5. Compute Train R², Test R², and 5-fold CV R²
#       6. Generate and save multiple visuals under partner_a_visuals/
#     """

#     # -------------------------------------------------
#     # Make sure the visuals folder exists
#     # -------------------------------------------------
#     visuals_dir = "partner_a_visuals"
#     os.makedirs(visuals_dir, exist_ok=True)

#     # ---------------------------------------------
#     # Step 1: Load the diabetes dataset (#7)
#     # ---------------------------------------------
#     data = load_diabetes(as_frame=True)

#     # X = features (10 columns)
#     # y = target (disease progression score)
#     X = data.data
#     y = data.target

#     # Print basic info so we SEE what we're working with
#     print("Shape of X (features):", X.shape)  # (rows, columns)
#     print("Shape of y (target):", y.shape)    # (rows,)

#     print("\nFirst 5 rows of X (features):")
#     print(X.head())

#     print("\nFirst 5 values of y (target):")
#     print(y.head())

#     # ---------------------------------------------
#     # Step 2: 80/20 random train/test split
#     # ---------------------------------------------
#     # Partner A is required to use a simple random 80/20 split.
#     # test_size=0.2  -> 20% of the data will be put into the test set
#     # random_state   -> makes the split reproducible
#     #
#     # After this:
#     #   - X_train, y_train will be used to TRAIN the model
#     #   - X_test,  y_test  will be used ONCE at the end to EVALUATE the model
#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=RANDOM_STATE
#     )

#     # Print shapes so we can see that 80/20 actually happened.
#     print("\n--- After 80/20 Train/Test Split (Random Holdout) ---")
#     print("X_train shape:", X_train.shape)   # should be about (353, 10)
#     print("X_test shape: ", X_test.shape)    # should be about (89, 10)
#     print("y_train shape:", y_train.shape)
#     print("y_test shape: ", y_test.shape)

#     # ---------------------------------------------
#     # Step 3: Build the model pipeline and train it
#     # ---------------------------------------------
#     # For this assignment, we must use Ridge (regression) with no tuning.
#     # We also scale the features first using StandardScaler.
#     #
#     # Pipeline steps:
#     #   1. "scaler" -> StandardScaler()
#     #   2. "ridge"  -> Ridge()
#     model = Pipeline([
#         ("scaler", StandardScaler()),
#         ("ridge", Ridge())
#     ])

#     # Fit the model on the TRAINING data only.
#     # The model will learn a relationship between X_train and y_train.
#     model.fit(X_train, y_train)

#     # ---------------------------------------------
#     # Step 3.5 — Evaluate on the TRAINING set
#     # ---------------------------------------------
#     # This helps us compare:
#     #   - Train R²
#     #   - Test R²
#     #   - Mean CV R²
#     # to understand overfitting or underfitting.
#     y_pred_train = model.predict(X_train)
#     train_r2 = r2_score(y_train, y_pred_train)

#     print("\n--- Training Set Performance ---")
#     print(f"Train R² score: {train_r2:.4f}")

#     # ---------------------------------------------
#     # Step 4: Evaluate on the 20% test set using R²
#     # ---------------------------------------------
#     # We use the trained model to make predictions for X_test
#     y_pred_test = model.predict(X_test)

#     # Now we compute R² between the true y_test and our predictions.
#     test_r2 = r2_score(y_test, y_pred_test)

#     print("\n--- Test Set Performance (20% Holdout) ---")
#     print(f"Test R² score: {test_r2:.4f}")

#     # ---------------------------------------------
#     # Step 5: 5-Fold Standard Cross-Validation (on training data only)
#     # ---------------------------------------------
#     # Partner A must use KFold (not StratifiedKFold, not TimeSeriesSplit).
#     #
#     # VERY IMPORTANT:
#     #   Cross-validation MUST be run on the TRAINING set ONLY.
#     #   We NEVER touch the test set until the very end.
#     #
#     # shuffle=True means the data is mixed before splitting into folds.
#     # random_state makes sure the shuffling is identical every run.
#     kfold = KFold(
#         n_splits=5,
#         shuffle=True,
#         random_state=RANDOM_STATE
#     )

#     # cross_val_score automatically:
#     #   - splits X_train/y_train into 5 folds
#     #   - trains the model 5 separate times
#     #   - evaluates each time using R² (our metric)
#     #   - returns an array of 5 scores
#     cv_scores = cross_val_score(
#         model,
#         X_train,
#         y_train,
#         cv=kfold,
#         scoring="r2"
#     )

#     # Print CV results
#     print("\n--- 5-Fold Cross-Validation Results (Training Set Only) ---")
#     print("Individual fold scores:", cv_scores)
#     print(f"CV Mean R²: {cv_scores.mean():.4f}")
#     print(f"CV Std Dev: {cv_scores.std():.4f}")

#     # ---------------------------------------------------------
#     # Step 6: Summary table (for logging / debugging)
#     # ---------------------------------------------------------
#     summary_df = pd.DataFrame({
#         "metric": ["Train R²", "Test R²", "CV Mean R²", "CV Std R²"],
#         "value": [
#             train_r2,
#             test_r2,
#             cv_scores.mean(),
#             cv_scores.std()
#         ]
#     })
#     print("\n--- Metric Summary ---")
#     print(summary_df)

#     # ---------------------------------------------------------
#     # Step 7: Visualizations (saved as HTML files)
#     # ---------------------------------------------------------

#     # 7.1: 5-Fold CV R² bar chart
#     cv_df = pd.DataFrame({
#         "Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
#         "R² Score": cv_scores
#     })

#     fig_cv_bar = px.bar(
#         cv_df,
#         x="Fold",
#         y="R² Score",
#         title="5-Fold CV R² Scores (Partner A – Random Holdout)",
#         text="R² Score",
#         color="R² Score",
#         color_continuous_scale="Bluered"
#     )
#     fig_cv_bar.update_traces(texttemplate='%{text:.3f}', textposition='outside')
#     fig_cv_bar.update_layout(
#         title_font_size=22,
#         xaxis_title="Fold Number",
#         yaxis_title="R² Score",
#         template="plotly_white",
#         height=500,
#         width=900
#     )
#     cv_bar_path = os.path.join(visuals_dir, "cv_r2_bar_chart.html")
#     fig_cv_bar.write_html(cv_bar_path)
#     print(f"\nSaved CV bar chart to: {cv_bar_path}")

#     # 7.2: Train vs Test vs CV Mean comparison bar chart
#     compare_df = pd.DataFrame({
#         "Metric": ["Train R²", "Test R²", "CV Mean R²"],
#         "Score": [train_r2, test_r2, cv_scores.mean()]
#     })

#     fig_compare = px.bar(
#         compare_df,
#         x="Metric",
#         y="Score",
#         title="Train vs Test vs CV Mean R²",
#         text="Score",
#         color="Metric",
#     )
#     fig_compare.update_traces(texttemplate='%{text:.3f}', textposition='outside')
#     fig_compare.update_layout(
#         template="plotly_white",
#         height=500,
#         width=800
#     )
#     compare_path = os.path.join(visuals_dir, "train_test_cv_comparison.html")
#     fig_compare.write_html(compare_path)
#     print(f"Saved comparison chart to: {compare_path}")

#     # 7.3: Actual vs Predicted scatter (test set)
#     test_results = pd.DataFrame({
#         "Actual": y_test.reset_index(drop=True),
#         "Predicted": pd.Series(y_pred_test).reset_index(drop=True)
#     })

#     fig_scatter = px.scatter(
#         test_results,
#         x="Actual",
#         y="Predicted",
#         title="Actual vs Predicted – Test Set (Ridge Regression, R² Metric)",
#         labels={"Actual": "Actual Progression", "Predicted": "Predicted Progression"},
#         opacity=0.7,
#     )
#     # Add a dashed line for perfect predictions
#     min_val = min(test_results["Actual"].min(), test_results["Predicted"].min())
#     max_val = max(test_results["Actual"].max(), test_results["Predicted"].max())
#     fig_scatter.add_shape(
#         type="line",
#         x0=min_val,
#         y0=min_val,
#         x1=max_val,
#         y1=max_val,
#         line=dict(color="red", dash="dash")
#     )
#     fig_scatter.update_layout(
#         template="plotly_white",
#         height=600,
#         width=800
#     )
#     scatter_path = os.path.join(visuals_dir, "actual_vs_pred_scatter.html")
#     fig_scatter.write_html(scatter_path)
#     print(f"Saved Actual vs Predicted scatter to: {scatter_path}")

#     # 7.4: Residual scatter plot (Predicted vs Residuals)
#     residuals = y_pred_test - y_test.reset_index(drop=True)

#     fig_resid_scatter = px.scatter(
#         x=y_pred_test,
#         y=residuals,
#         title="Residual Plot – Test Set (Predicted vs Residuals)",
#         labels={"x": "Predicted Values", "y": "Residuals (Pred - Actual)"},
#         opacity=0.7,
#     )
#     fig_resid_scatter.add_shape(
#         type="line",
#         x0=min(y_pred_test),
#         y0=0,
#         x1=max(y_pred_test),
#         y1=0,
#         line=dict(color="red", dash="dash")
#     )
#     fig_resid_scatter.update_layout(
#         template="plotly_white",
#         height=500,
#         width=800
#     )
#     resid_scatter_path = os.path.join(visuals_dir, "residuals_scatter.html")
#     fig_resid_scatter.write_html(resid_scatter_path)
#     print(f"Saved residual scatter plot to: {resid_scatter_path}")

#     # 7.5: Residual histogram
#     fig_resid_hist = px.histogram(
#         residuals,
#         nbins=30,
#         title="Distribution of Residuals – Test Set",
#         labels={"value": "Residual"},
#         opacity=0.75,
#     )
#     fig_resid_hist.update_layout(
#         template="plotly_white",
#         height=500,
#         width=800
#     )
#     resid_hist_path = os.path.join(visuals_dir, "residuals_histogram.html")
#     fig_resid_hist.write_html(resid_hist_path)
#     print(f"Saved residual histogram to: {resid_hist_path}")

#     # 7.6: Sorted Actual vs Predicted line chart
#     y_test_series = y_test.reset_index(drop=True)
#     y_pred_series = pd.Series(y_pred_test).reset_index(drop=True)

#     sorted_idx = y_test_series.argsort()
#     actual_sorted = y_test_series.iloc[sorted_idx].reset_index(drop=True)
#     pred_sorted = y_pred_series.iloc[sorted_idx].reset_index(drop=True)

#     x_vals = list(range(len(actual_sorted)))

#     fig_sorted = px.line(
#         x=x_vals,
#         y=actual_sorted,
#         labels={"x": "Sorted Samples", "y": "Progression Score"},
#         title="Actual vs Predicted (Sorted Test Set)"
#     )
#     fig_sorted.add_scatter(
#         x=x_vals,
#         y=pred_sorted,
#         mode="lines",
#         name="Predicted Values"
#     )
#     fig_sorted.update_layout(
#         template="plotly_white",
#         height=600,
#         width=900
#     )
#     sorted_path = os.path.join(visuals_dir, "sorted_actual_vs_pred.html")
#     fig_sorted.write_html(sorted_path)
#     print(f"Saved sorted Actual vs Predicted line chart to: {sorted_path}")

#     print("\nAll Partner A visuals have been saved to the folder:", visuals_dir)


# if __name__ == "__main__":
#     # Command to run in terminal (inside venv):
#     #   python eval_partner_a.py
#     main()


"""
AISE 26 - W9D1 Split Strategy Showdown
Partner A: Random Holdout + 5-Fold Standard CV
Author: Andrea Churchwell
Dataset: Diabetes Progression (#7)
Metric: R²
"""

# ---------------------------------------------------------
# Imports and global random seed
# ---------------------------------------------------------

import os

import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import plotly.express as px

# Fancy terminal output using Rich (no colorama now)
from rich.console import Console
from rich.table import Table

# Global random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def main():
    console = Console()

    # Simple, clean header (no giant ASCII)
    console.print("\n[bold magenta]=== Partner A – Evaluation Pipeline ===[/bold magenta]")
    console.print("[bold cyan]AISE 26 - W9D1 Split Strategy Showdown[/bold cyan]")
    console.print("Partner A: Random Holdout + 5-Fold Standard CV\n", style="bold")

    # -------------------------------------------------
    # Make sure the visuals folder exists
    # -------------------------------------------------
    visuals_dir = "partner_a_visuals"
    os.makedirs(visuals_dir, exist_ok=True)

    # ---------------------------------------------
    # Step 1: Load the diabetes dataset (#7)
    # ---------------------------------------------
    console.print("\n[bold cyan]📥 Step 1: Loading Diabetes dataset (#7)...[/bold cyan]")

    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target

    console.print(f"[yellow]• Shape of X (features): {X.shape}[/yellow]")
    console.print(f"[yellow]• Shape of y (target):   {y.shape}[/yellow]\n")

    console.print("[white]First 5 rows of X (features):[/white]")
    console.print(X.head())

    console.print("\n[white]First 5 values of y (target):[/white]")
    console.print(y.head())

    # ---------------------------------------------
    # Step 2: 80/20 random train/test split
    # ---------------------------------------------
    console.print("\n[bold cyan]📊 Step 2: 80/20 Random Train/Test Split...[/bold cyan]")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    console.print("\n[green]--- After 80/20 Train/Test Split (Random Holdout) ---[/green]")
    console.print(f"X_train shape: {X_train.shape}")
    console.print(f"X_test  shape: {X_test.shape}")
    console.print(f"y_train shape: {y_train.shape}")
    console.print(f"y_test  shape: {y_test.shape}")

    # ---------------------------------------------
    # Step 3: Build the model pipeline and train it
    # ---------------------------------------------
    console.print("\n[bold cyan]🧠 Step 3: Building Ridge Regression Pipeline...[/bold cyan]")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])

    model.fit(X_train, y_train)
    console.print("[green]✔ Model fitted on training data.[/green]")

    # ---------------------------------------------
    # Step 3.5 — Evaluate on the TRAINING set
    # ---------------------------------------------
    y_pred_train = model.predict(X_train)
    train_r2 = r2_score(y_train, y_pred_train)

    console.print("\n[bold cyan]📘 Training Set Performance:[/bold cyan]")
    console.print(f"[green]• Train R² score: {train_r2:.4f}[/green]")

    # ---------------------------------------------
    # Step 4: Evaluate on the 20% test set using R²
    # ---------------------------------------------
    y_pred_test = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)

    console.print("\n📗 Step 4: Test Set Performance (20% Holdout)")

    console.print(f"[green]• Test R² score: {test_r2:.4f}[/green]")

    # ---------------------------------------------
    # Step 5: 5-Fold Standard Cross-Validation
    # ---------------------------------------------
    console.print("\n[bold cyan]📚 Step 5: 5-Fold Cross-Validation on Training Data...[/bold cyan]")

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
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

     # ---------------------------------------------
    # Step 5: Show CV results
    # ---------------------------------------------
    console.print("\n--- 5-Fold Cross-Validation Results (Training Only) ---")
    console.print(f"Fold scores: {cv_scores}")
    console.print(f"CV Mean R²: {cv_mean:.4f}")
    console.print(f"CV Std  R²: {cv_std:.4f}")

    # ---------------------------------------------------------
    # Step 6: Summary of all R² metrics
    # ---------------------------------------------------------
    console.print("\n✨ Step 6: Summary (Partner A - R² Scores)\n")
    console.print(f"📘 Train R²   : {train_r2:.4f}")
    console.print(f"📗 Test R²    : {test_r2:.4f}")
    console.print(f"📙 CV Mean R² : {cv_mean:.4f}")
    console.print(f"📒 CV Std R²  : {cv_std:.4f}")

    # ---------------------------------------------------------
    # Step 7: Visualizations (saved as HTML files)
    # ---------------------------------------------------------
    console.print("\n📈 Step 7: Generating and saving Plotly visuals...")

    # 7.1: 5-Fold CV R² bar chart
    cv_df = pd.DataFrame({
        "Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
        "R² Score": cv_scores
    })

    fig_cv_bar = px.bar(
        cv_df,
        x="Fold",
        y="R² Score",
        title="5-Fold CV R² Scores (Partner A – Random Holdout)",
        text="R² Score",
        color="R² Score",
        color_continuous_scale="Bluered"
    )
    fig_cv_bar.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_cv_bar.update_layout(
        title_font_size=22,
        xaxis_title="Fold Number",
        yaxis_title="R² Score",
        template="plotly_white",
        height=500,
        width=900
    )
    cv_bar_path = os.path.join(visuals_dir, "cv_r2_bar_chart.html")
    fig_cv_bar.write_html(cv_bar_path)
    console.print(f"✔ Saved CV bar chart to: {cv_bar_path}")

    # 7.2: Train vs Test vs CV Mean comparison bar chart
    compare_df = pd.DataFrame({
        "Metric": ["Train R²", "Test R²", "CV Mean R²"],
        "Score": [train_r2, test_r2, cv_mean]
    })

    fig_compare = px.bar(
        compare_df,
        x="Metric",
        y="Score",
        title="Train vs Test vs CV Mean R²",
        text="Score",
        color="Metric",
    )
    fig_compare.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_compare.update_layout(
        template="plotly_white",
        height=500,
        width=800
    )
    compare_path = os.path.join(visuals_dir, "train_test_cv_comparison.html")
    fig_compare.write_html(compare_path)
    console.print(f"✔ Saved comparison chart to: {compare_path}")

    # 7.3: Actual vs Predicted scatter (test set)
    test_results = pd.DataFrame({
        "Actual": y_test.reset_index(drop=True),
        "Predicted": pd.Series(y_pred_test).reset_index(drop=True)
    })

    fig_scatter = px.scatter(
        test_results,
        x="Actual",
        y="Predicted",
        title="Actual vs Predicted – Test Set (Ridge Regression, R² Metric)",
        labels={"Actual": "Actual Progression", "Predicted": "Predicted Progression"},
        opacity=0.7,
    )
    min_val = min(test_results["Actual"].min(), test_results["Predicted"].min())
    max_val = max(test_results["Actual"].max(), test_results["Predicted"].max())
    fig_scatter.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="red", dash="dash")
    )
    fig_scatter.update_layout(
        template="plotly_white",
        height=600,
        width=800
    )
    scatter_path = os.path.join(visuals_dir, "actual_vs_pred_scatter.html")
    fig_scatter.write_html(scatter_path)
    console.print(f"✔ Saved Actual vs Predicted scatter to: {scatter_path}")

    # 7.4: Residual scatter plot (Predicted vs Residuals)
    residuals = y_pred_test - y_test.reset_index(drop=True)

    fig_resid_scatter = px.scatter(
        x=y_pred_test,
        y=residuals,
        title="Residual Plot – Test Set (Predicted vs Residuals)",
        labels={"x": "Predicted Values", "y": "Residuals (Pred - Actual)"},
        opacity=0.7,
    )
    fig_resid_scatter.add_shape(
        type="line",
        x0=min(y_pred_test),
        y0=0,
        x1=max(y_pred_test),
        y1=0,
        line=dict(color="red", dash="dash")
    )
    fig_resid_scatter.update_layout(
        template="plotly_white",
        height=500,
        width=800
    )
    resid_scatter_path = os.path.join(visuals_dir, "residuals_scatter.html")
    fig_resid_scatter.write_html(resid_scatter_path)
    console.print(f"✔ Saved residual scatter plot to: {resid_scatter_path}")

    # 7.5: Residual histogram
    fig_resid_hist = px.histogram(
        residuals,
        nbins=30,
        title="Distribution of Residuals – Test Set",
        labels={"value": "Residual"},
        opacity=0.75,
    )
    fig_resid_hist.update_layout(
        template="plotly_white",
        height=500,
        width=800
    )
    resid_hist_path = os.path.join(visuals_dir, "residuals_histogram.html")
    fig_resid_hist.write_html(resid_hist_path)
    console.print(f"✔ Saved residual histogram to: {resid_hist_path}")

    # 7.6: Sorted Actual vs Predicted line chart
    y_test_series = y_test.reset_index(drop=True)
    y_pred_series = pd.Series(y_pred_test).reset_index(drop=True)

    sorted_idx = y_test_series.argsort()
    actual_sorted = y_test_series.iloc[sorted_idx].reset_index(drop=True)
    pred_sorted = y_pred_series.iloc[sorted_idx].reset_index(drop=True)

    x_vals = list(range(len(actual_sorted)))

    fig_sorted = px.line(
        x=x_vals,
        y=actual_sorted,
        labels={"x": "Sorted Samples", "y": "Progression Score"},
        title="Actual vs Predicted (Sorted Test Set)"
    )
    fig_sorted.add_scatter(
        x=x_vals,
        y=pred_sorted,
        mode="lines",
        name="Predicted Values"
    )
    fig_sorted.update_layout(
        template="plotly_white",
        height=600,
        width=900
    )
    sorted_path = os.path.join(visuals_dir, "sorted_actual_vs_pred.html")
    fig_sorted.write_html(sorted_path)
    console.print(f"✔ Saved sorted Actual vs Predicted chart to: {sorted_path}")

    # ---------------------------------------------------------
    # Save Partner A CV scores to comparison.csv
    # ---------------------------------------------------------
    comparison_path = "comparison.csv"

    comp_df = pd.DataFrame({
        "strategy": ["partner_a"] * len(cv_scores),
        "fold": [1, 2, 3, 4, 5],
        "score": cv_scores
    })

    if not os.path.exists(comparison_path):
        comp_df.to_csv(comparison_path, index=False)
    else:
        comp_df.to_csv(comparison_path, mode="a", header=False, index=False)

    console.print("\n🎉 All Partner A visuals have been saved to the folder: partner_a_visuals")
    console.print("✔ Saved Partner A CV scores to: comparison.csv")
    console.print("Done.\n")


if __name__ == "__main__":
    main()
