# Week 9 – Split Strategy Showdown  
## Partner Comparison: Random Holdout vs Time-Aware Split  

This document summarizes and compares the evaluation strategies for:

- **Partner A – Andrea Churchwell**  
  - Random 80/20 holdout  
  - 5-fold KFold cross-validation  

- **Partner B – Jose Diaz**  
  - Ordered 80/20 holdout (first 80% → train, last 20% → test)  
  - 5-fold TimeSeriesSplit (time-aware CV)  

The goal is to:

1. Describe the dataset and model setup.  
2. Summarize Partner A and Partner B results.  
3. Compare cross-validation stability and residual errors.  
4. Explain which strategy is more appropriate for the diabetes dataset and why.  

---

## 1. Dataset Overview  

- **Name:** Diabetes Regression (`sklearn.datasets.load_diabetes`)  
- **Samples:** 442 patients  
- **Features:** 10 numeric health measurements (age, BMI, blood pressure, etc.)  
- **Task Type:** Regression – predict a continuous disease progression score  
- **Metric:** R²  

This is a **small, noisy medical dataset**, which means:

- Different train/test splits can give noticeably different scores.  
- R² values around **0.45–0.55** are normal.  
- Stability (variance across splits) is just as important as the raw R² number.  

---

## 2. Partner A – Random Holdout + KFold (Andrea)  

### Strategy  

- **80/20 random train/test split** using `train_test_split(..., shuffle=True, random_state=42)`.  
- **5-fold KFold** cross-validation on the training data.  
- Model: `StandardScaler` + `Ridge` regression combined in a `Pipeline`.  
- Metric: R².  

Using a `Pipeline` prevents data leakage because the scaler is fit **only on the training data** inside each split or fold.  

### Train/Test Performance  

- **Train R²:** 0.5276  
- **Test R²:** 0.4541  

Interpretation:  

- The model explains about **52–53%** of the variance on the training set.  
- It explains about **45%** of the variance on unseen test data.  
- The drop from train → test is **small and normal** for a small, noisy dataset.  
- This suggests the model is learning real patterns and generalizes reasonably well.  

### 5-Fold CV (KFold) Results  

Fold R² scores:

- Fold 1: 0.4699  
- Fold 2: 0.5381  
- Fold 3: 0.4133  
- Fold 4: 0.4903  
- Fold 5: 0.4922  

Summary:

- **Mean R²:** ~0.4808  
- **Std (variance):** ~0.04  

This shows **low-to-moderate variance**: the model behaves similarly across folds.  
For this dataset, Partner A’s strategy gives **stable** and **reliable** estimates.  

There is some wobble between folds (e.g., Fold 2 higher, Fold 3 lower), which is expected with only 442 samples.  

---

## 3. Partner B – Ordered Holdout + TimeSeriesSplit (Jose)  

### Strategy  

- **Ordered 80/20 split**:  
  - First 80% of rows → training set  
  - Last 20% of rows → test set  
- **5-fold TimeSeriesSplit** on the training data (no shuffling; later folds always validate on “later” rows).  
- Same model as Partner A: `StandardScaler` + `Ridge` in a `Pipeline`.  
- Metric: R².  

This simulates a **time-aware** evaluation: train on “earlier” data, validate on “later” data, even though the diabetes dataset is not truly time series.  

### Train/Test Performance  

- **Train R²:** 0.5088  
- **Test R²:** 0.5413  

Interpretation:  

- On this particular ordered split, the last 20% of the data (test set) is slightly **easier** than the earlier rows.  
- That’s why test R² ends up a bit higher than train R².  
- This is not an error; it can happen on small datasets with non-random splits.  

### 5-Fold TimeSeriesSplit Results  

Fold R² scores:

- Fold 1: 0.2984  
- Fold 2: 0.5760  
- Fold 3: 0.2785  
- Fold 4: 0.5831  
- Fold 5: 0.4492  

Summary:

- **Mean R²:** ~0.4373  
- **Std (variance):** ~0.13  

Compared to Partner A, Partner B’s strategy has:

- A **lower mean R²**, and  
- A **much higher variance** across folds.  

TimeSeriesSplit forces each fold to train on earlier rows and validate on later rows.  
For a dataset that is **not actually temporal**, this mostly increases instability instead of adding realism.  

---

## 4. Cross-Validation Summary  

From `comparison.csv` (combined Partner A + Partner B CV scores):  

| Strategy    | Mean R² | Std (variance) |
|------------|--------:|---------------:|
| Partner A  | ~0.48   | ~0.04          |
| Partner B  | ~0.44   | ~0.13          |

Key points:

- Partner A has a **higher average R²** and **much lower variance**.  
- Partner B has a **lower average R²** and **very bouncy scores** (from ~0.28 to ~0.58).  

This matches what you see in the CV bar charts:  

- `assets/partner_a_cvr2.png`  
- `assets/partner_b_cvr2.png`  

Open these two images side by side in VS Code to visually confirm the difference.  

---

## 5. Visual Notes  

You created PNG screenshots for both partners:  

- CV bar charts  
  - `assets/partner_a_cvr2.png`  
  - `assets/partner_b_cvr2.png`  

- Actual vs Predicted scatterplots  
  - `assets/partner_a_actual_vs_pred.png`  
  - `assets/partner_b_actual_vs_pred.png`  

- Residual histograms  
  - `assets/partner_a_residuals.png`  
  - `assets/partner_b_residuals.png`  

### 5.1 CV Bar Charts  

- **Partner A:** bars are tightly grouped between ~0.41 and ~0.54.  
- **Partner B:** bars swing from ~0.28 to ~0.58.  

This visually reinforces the numeric variances (0.04 vs 0.13).  

### 5.2 Actual vs Predicted  

- **Partner A:**  
  - Scatter points roughly follow an upward trend.  
  - Includes a dashed red trendline that shows the linear relationship.  
  - Points are somewhat spread but follow the line reasonably well.  

- **Partner B:**  
  - Scatter also trends upward, but points appear more scattered.  
  - No trendline in this plot, but the general pattern is noisier.  

### 5.3 Residuals  

- **Partner A residuals:** more tightly clustered around 0, indicating more consistent prediction errors.  
- **Partner B residuals:** wider spread, matching the higher fold-to-fold variance from TimeSeriesSplit.  

---

## 6. Recommendation  

Both partners correctly implemented their assigned strategies:

- Partner A: random 80/20 split + 5-fold KFold CV.  
- Partner B: ordered 80/20 split + 5-fold TimeSeriesSplit (time-aware CV).  

However, the **diabetes dataset is not temporal** — the row order does not represent time.

Because of that:

- Randomized KFold (Partner A) is **better suited** to this dataset.  
- It produces **higher mean R²** and **much more stable** results.  
- TimeSeriesSplit (Partner B) mainly increases variance without adding realism here.  

**Final conclusion:**  

> For the Diabetes Regression dataset, we recommend Partner A’s random holdout + KFold strategy as the primary evaluation method. Partner B’s ordered + time-aware approach is still valuable to understand and would be more appropriate for a true time series problem (e.g., forecasting over months or years), but it is less well-matched to this specific dataset.
