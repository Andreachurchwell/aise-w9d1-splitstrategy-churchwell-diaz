from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import numpy as np


def main():
    # 1) Load dataset
    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target

    # 2) Holdout split (Partner B: same split style as Partner A, same seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # 3) Pipeline: StandardScaler + Ridge regression
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(random_state=42)),
        ]
    )

    # 4) Fit on training data
    model.fit(X_train, y_train)

    # 5) Evaluate on test set using R^2
    test_r2 = model.score(X_test, y_test)  # Ridge.score = R^2 for regression

    # 6) KFold CV on full data (same KFold config as Andrea)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="r2"
    )

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    # 7) Print results in a simple, clear way
    print("Diabetes Regression – Partner B (Jose)")
    print("Model: StandardScaler + Ridge")
    print("Metric: R^2\n")

    print(f"Test R^2 (holdout): {test_r2:.4f}")
    print(f"CV R^2 mean (5-fold): {cv_mean:.4f}")
    print(f"CV R^2 std:           {cv_std:.4f}")
    print("All CV fold scores:")
    for i, score in enumerate(cv_scores, start=1):
        print(f"  Fold {i}: {score:.4f}")

    # 8) Save the model

    import joblib
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "diabetes_ridge.joblib"

    joblib.dump(model, model_path)

    print(f"Saved model → {model_path}")



if __name__ == "__main__":
    main()
