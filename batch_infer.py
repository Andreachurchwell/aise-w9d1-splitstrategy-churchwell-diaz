# batch_infer.py
import sys
import time
import json
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path("models/diabetes_ridge.joblib")

def main(input_csv, output_csv):
    start = time.time()

    # -----------------------------
    # 1. Load the trained model
    # -----------------------------
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    # The trained model knows the correct feature names
    try:
        expected_features = model.feature_names_in_
    except:
        raise ValueError("The loaded model does not have feature_names_in_. "
                         "Ensure you trained it using scikit-learn >= 1.0.")

    print(f"Expected features:\n{list(expected_features)}\n")

    # -----------------------------
    # 2. Load input CSV
    # -----------------------------
    df = pd.read_csv(input_csv)
    print(f"Loaded input CSV: {df.shape[0]} rows")

    # -----------------------------
    # 3. Validate schema
    # -----------------------------
    missing = [f for f in expected_features if f not in df.columns]
    extra   = [c for c in df.columns if c not in expected_features]

    if missing:
        raise ValueError(f"❌ Missing columns in input CSV: {missing}")

    if extra:
        print(f"⚠️ Warning: extra columns ignored: {extra}")

    # restrict to correct columns in the correct order
    df_features = df[expected_features]

    # -----------------------------
    # 4. Predict
    # -----------------------------
    preds = model.predict(df_features)

    # -----------------------------
    # 5. Write output
    # -----------------------------
    df_out = df.copy()
    df_out["prediction"] = preds

    Path(output_csv).parent.mkdir(exist_ok=True, parents=True)
    df_out.to_csv(output_csv, index=False)

    duration = time.time() - start
    print(f"\n✅ Batch inference complete.")
    print(f"Rows processed: {df.shape[0]}")
    print(f"Saved predictions → {output_csv}")
    print(f"Time taken: {duration:.3f} seconds")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("\nUsage:")
        print("  python batch_infer.py data/input.csv data/predictions.csv\n")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    main(input_csv, output_csv)
