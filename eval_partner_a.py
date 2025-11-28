"""
AISE 26 - W9D1 Split Strategy Showdown
Partner A: Random Holdout + 5-Fold Standard CV
Author: Andrea Churchwell
Dataset: Diabetes Progression (#7)
Metric: R²

This file is the required Partner A script for the assignment.
I'm writing it with very detailed comments so I understand every step.
"""

# ---------------------------------------------------------
# Step 0: Imports and global random seed
# ---------------------------------------------------------

# numpy: numerical operations and arrays
import numpy as np

# pandas: nice table-style data structures (DataFrames)
import pandas as pd

# load_diabetes: gives us the diabetes regression dataset (#7)
from sklearn.datasets import load_diabetes

# train_test_split: will do our 80/20 random train/test split
# KFold: will do 5-fold cross-validation on the training set
# cross_val_score: will run the CV and give scores for each fold
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# Ridge: the required regression model for this assignment (no tuning)
from sklearn.linear_model import Ridge

# Pipeline: lets us chain preprocessing + model together
from sklearn.pipeline import Pipeline

# StandardScaler: scales features so Ridge behaves better
from sklearn.preprocessing import StandardScaler

# r2_score: our evaluation metric (R²)
from sklearn.metrics import r2_score

# Set a global random seed so that our results are reproducible.
# We'll use this same RANDOM_STATE everywhere that supports it.
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def main():
    """
    Main function: runs the whole Partner A pipeline.

    Right now it will:
      1. Load the diabetes dataset
      2. Inspect its shape and a few rows
      3. Create an 80/20 random train/test split (Partner A requirement)
    """

    # ---------------------------------------------
    # Step 1: Load the diabetes dataset (#7)
    # ---------------------------------------------
    data = load_diabetes(as_frame=True)

    # X = features (10 columns)
    # y = target (disease progression score)
    X = data.data
    y = data.target

    # Print basic info so we SEE what we're working with
    print("Shape of X (features):", X.shape)  # (rows, columns)
    print("Shape of y (target):", y.shape)    # (rows,)

    print("\nFirst 5 rows of X (features):")
    print(X.head())

    print("\nFirst 5 values of y (target):")
    print(y.head())

    # ---------------------------------------------
    # Step 2: 80/20 random train/test split
    # ---------------------------------------------
    # Partner A is required to use a simple random 80/20 split.
    # test_size=0.2  -> 20% of the data will be put into the test set
    # random_state   -> makes the split reproducible
    #
    # After this:
    #   - X_train, y_train will be used to TRAIN the model
    #   - X_test,  y_test  will be used ONCE at the end to EVALUATE the model
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    # Let's print shapes so we can see that 80/20 actually happened.
    print("\n--- After 80/20 Train/Test Split (Random Holdout) ---")
    print("X_train shape:", X_train.shape)   # should be about (353, 10)
    print("X_test shape: ", X_test.shape)    # should be about (89, 10)
    print("y_train shape:", y_train.shape)
    print("y_test shape: ", y_test.shape)

    # ---------------------------------------------
    # Step 3: Build the model pipeline and train it
    # ---------------------------------------------
    # For this assignment, we must use Ridge (regression) with no tuning.
    # We also scale the features first using StandardScaler.
    #
    # Pipeline steps:
    #   1. "scaler" -> StandardScaler()
    #   2. "ridge"  -> Ridge()
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])

    # Fit the model on the TRAINING data only.
    # The model will learn a relationship between X_train and y_train.
    model.fit(X_train, y_train)

    # ---------------------------------------------
    # Step 4: Evaluate on the 20% test set using R²
    # ---------------------------------------------
    # We use the trained model to make predictions for X_test
    y_pred_test = model.predict(X_test)

    # Now we compute R² between the true y_test and our predictions.
    test_r2 = r2_score(y_test, y_pred_test)

    print("\n--- Test Set Performance (20% Holdout) ---")
    print(f"Test R² score: {test_r2:.4f}")

    # ---------------------------------------------
    # Step 5: 5-Fold Standard Cross-Validation (on training data only)
    # ---------------------------------------------
    # Partner A must use KFold (not StratifiedKFold, not TimeSeriesSplit).
    #
    # VERY IMPORTANT:
    #   Cross-validation MUST be run on the TRAINING set ONLY.
    #   We NEVER touch the test set until the very end.
    #
    # shuffle=True means the data is mixed before splitting into folds.
    # random_state makes sure the shuffling is identical every run.
    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # cross_val_score automatically:
    #   - splits X_train/y_train into 5 folds
    #   - trains the model 5 separate times
    #   - evaluates each time using R² (our metric)
    #   - returns an array of 5 scores
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kfold,
        scoring="r2"
    )
    # Print CV results
    print("\n--- 5-Fold Cross-Validation Results (Training Set Only) ---")
    print("Individual fold scores:", cv_scores)
    print(f"CV Mean R²: {cv_scores.mean():.4f}")
    print(f"CV Std Dev: {cv_scores.std():.4f}")


if __name__ == "__main__":
    # This makes sure main() only runs when we execute this file directly.
    # Command to run in terminal (inside venv):
    #   python eval_partner_a.py
    main()
