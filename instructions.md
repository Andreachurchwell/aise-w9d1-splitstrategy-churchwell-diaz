# **AISE 26 W9D1 - After-Class Assignment: Split Strategy Showdown**

**Difficulty: 4/10 | Time: 40-50 min | Format: Pairs or Trios**

---

## **Goal**
Work with your partner(s) to compare different evaluation strategies on the same dataset, then synthesize your findings into a joint recommendation report.

---

## **Task Division**

### **Partner A: Random Holdout + 5-Fold Standard CV**
- 80/20 train/test split (`random_state=42`)
- 5-fold CV on training set using `KFold`
- Calculate your chosen metric (see Model & Metric Requirements below)

### **Partner B: Stratified/Time-Aware + 5-Fold Specialized CV**
- 80/20 train/test split (stratified OR time-aware based on data)
- 5-fold CV using `StratifiedKFold` (classification) OR `TimeSeriesSplit` (temporal data)
- Calculate the SAME metric as Partner A

### **Partner C: Nested Holdout Validation** (if trio)
- Split data into 60% train / 20% validation / 20% test
- Train on 60%, validate on 20%, report final score on 20% test
- Run 5-fold CV on the 60% training portion only (not the full dataset)
- This simulates a more realistic pipeline with a held-out validation set
- Calculate the SAME metric as Partners A & B

---

## **Model & Metric Requirements**

⚠️ **IMPORTANT: All partners must use the same model and metric for fair comparison.**

### **Model Selection (No Tuning Allowed)**
- **Classification tasks**: Use `LogisticRegression(random_state=42, max_iter=1000)`
- **Regression tasks**: Use `Ridge(random_state=42)`
- **No hyperparameter tuning** - use these exact parameters
- This ensures differences come from evaluation strategy, not model choice

### **Metric Selection Process**
1. Review your chosen dataset's "Metric recommendation" below
2. **As a team, discuss and agree on ONE metric before coding**
3. Document your choice in `TEAM_INFO.md` (see deliverables)
4. All partners must report this same metric in their code

---

## **Dataset Options**

Choose **ONE** dataset for your pair/trio assignment. Each has unique characteristics that make split/metric choices matter.

### **Classification Datasets**

#### **1. Breast Cancer Wisconsin (Balanced Binary)**
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
```
- **Size**: 569 samples, 30 features
- **Classes**: Malignant (37%) vs Benign (63%) - slightly imbalanced
- **Good for**: Comparing standard vs stratified splits; practicing precision/recall/F1
- **Metric recommendation**: ROC-AUC, PR-AUC, or F1

---

#### **2. Wine Quality (Multiclass)**
```python
from sklearn.datasets import load_wine
data = load_wine(as_frame=True)
X, y = data.data, data.target
```
- **Size**: 178 samples, 13 features
- **Classes**: 3 wine types (balanced)
- **Good for**: Small dataset where CV variance matters; multiclass metrics
- **Metric recommendation**: Accuracy, weighted F1, or ROC-AUC (ovr)

---

#### **3. Credit Card Default (Imbalanced)**
```python
# UCI ML Repository - Credit Card Clients
import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
df = pd.read_excel(url, header=1)
X = df.iloc[:, 1:-1]  # Skip ID, take features
y = df.iloc[:, -1]     # Default payment
```
- **Size**: 30,000 samples, 23 features
- **Classes**: Default (22%) vs No Default (78%) - imbalanced
- **Good for**: Demonstrating why PR-AUC > accuracy for imbalanced data
- **Metric recommendation**: PR-AUC, F1, or precision/recall

---

#### **4. Adult Income (Binary, Large)**
```python
from sklearn.datasets import fetch_openml
data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
X, y = data.data, data.target
# Note: You'll need to handle categorical features (use OneHotEncoder or LabelEncoder)
# and missing values before running your evaluation
```
- **Size**: 48,842 samples, 14 features (mix of numerical/categorical)
- **Classes**: Income >50K (24%) vs ≤50K (76%)
- **Good for**: Larger dataset, preprocessing practice, imbalance
- **Metric recommendation**: ROC-AUC or F1

---

#### **5. Iris (Multiclass, Classic)**
```python
from sklearn.datasets import load_iris
data = load_iris(as_frame=True)
X, y = data.data, data.target
```
- **Size**: 150 samples, 4 features
- **Classes**: 3 flower species (perfectly balanced)
- **Good for**: Simple baseline; showing CV variance on tiny data
- **Metric recommendation**: Accuracy or weighted F1

---

### **Regression Datasets**

#### **6. California Housing (Regression)**
```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
```
- **Size**: 20,640 samples, 8 features
- **Target**: Median house value (continuous)
- **Good for**: Standard regression metrics; comparing MAE vs RMSE
- **Metric recommendation**: MAE, RMSE, or R²

---

#### **7. Diabetes Progression (Regression, Small)**
```python
from sklearn.datasets import load_diabetes
data = load_diabetes(as_frame=True)
X, y = data.data, data.target
```
- **Size**: 442 samples, 10 features
- **Target**: Disease progression (continuous)
- **Good for**: Small dataset where CV stabilizes estimates; low R² challenge
- **Metric recommendation**: MAE, RMSE, or R²

---

### **Time-Series Datasets**

#### **8. Bitcoin Historical Prices (Temporal)**
```python
import pandas as pd
import yfinance as yf
df = yf.download('BTC-USD', start='2020-01-01', end='2024-01-01')
# Create features: lagged prices, rolling means, etc.
# Target: next-day price or price movement
```
- **Size**: ~1,460 daily records
- **Task**: Predict next-day price or price direction (regression or classification)
- **Good for**: **Requires TimeSeriesSplit**; demonstrates look-ahead leakage
- **Metric recommendation**: MAE/RMSE (regression) or accuracy/F1 (classification)

---

#### **9. Airline Passengers (Temporal, Classic)**
```python
import pandas as pd
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url)
# Create lag features, rolling stats for supervised learning
# Target: Passengers (next month)
```
- **Size**: 144 monthly records
- **Task**: Predict passenger count (regression)
- **Good for**: **Must use TimeSeriesSplit**; small temporal dataset
- **Metric recommendation**: MAE or RMSE

---

#### **10. Store Sales (Temporal)**
```python
# Use seaborn flights dataset
import pandas as pd
import seaborn as sns
df = sns.load_dataset('flights')
# Transform to supervised: predict passengers using year, month features
```
- **Size**: 144 records
- **Task**: Predict sales/demand (regression)
- **Good for**: **Requires TimeSeriesSplit**; feature engineering from dates
- **Metric recommendation**: MAE or RMSE

---

### **Dataset Selection Guidelines**

| Choose... | If you want to practice... |
|-----------|---------------------------|
| **#3 or #4** | Imbalanced classification + PR-AUC vs ROC-AUC comparison |
| **#8, #9, or #10** | TimeSeriesSplit + avoiding look-ahead leakage |
| **#2 or #7** | Small datasets where CV variance is visible |
| **#6** | Standard regression with large, clean data |
| **#1 or #5** | Quick setup, focus on evaluation mechanics |

---

## **Required Deliverables**

Submit **one** GitHub repository per pair/trio containing exactly these files:

### **1. Team Info & Dataset Declaration** (`TEAM_INFO.md`)

Create a file with this structure:

```
# Team Information

## Team Members
- **Partner A**: [Full Name] - Random Holdout Strategy
- **Partner B**: [Full Name] - Stratified/Time-Aware Strategy
- **Partner C**: [Full Name] - Nested Holdout Strategy (if trio)

## Dataset Choice
- **Dataset Name**: [e.g., Breast Cancer Wisconsin]
- **Dataset Number**: [e.g., #1]
- **Task Type**: [Classification or Regression]
- **Key Characteristic**: [e.g., Slightly imbalanced, Temporal, Small sample size]

## Why We Chose This Dataset
[1-2 sentences about what makes this dataset interesting for evaluation practice]

## Agreed-Upon Metric
- **Metric**: [e.g., ROC-AUC]
- **Why this metric**: [1 sentence explaining why it's appropriate for this dataset]

## Loading Code
[Exact code to load your chosen dataset - paste the Python code from the dataset options]

## Package Versions
- Python: [version]
- scikit-learn: [version]
- pandas: [version]
- numpy: [version]
[any other relevant packages]
```

### **2. Individual Evaluation Scripts**

Each partner creates their own Python file:
- `eval_partner_a.py` - Partner A's complete evaluation pipeline
- `eval_partner_b.py` - Partner B's complete evaluation pipeline
- `eval_partner_c.py` - Partner C's complete evaluation pipeline *(if trio)*

**Each file must include:**
- Comments with your name and partner role at the top
- Random seed set (`random_state=42` everywhere)
- Complete pipeline from data loading to final results
- Print statements showing all scores (test score, CV mean, CV std, individual fold scores)
- Comments explaining each major step

**Example structure:**
```python
"""
Partner A: Random Holdout + 5-Fold Standard CV
Author: [Your Name]
Dataset: [Dataset Name]
Metric: [Your Metric]
"""

import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression  # or Ridge for regression
# ... other imports

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load data
# [Your code here]

# Split data: 80/20 train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Train model
model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate on test set
test_score = [your metric calculation]

# 5-Fold CV on training set
kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='[your metric]')

# Print results
print(f"Test Score: {test_score:.4f}")
print(f"CV Mean: {cv_scores.mean():.4f}")
print(f"CV Std: {cv_scores.std():.4f}")
print(f"Individual Folds: {cv_scores}")
```

### **3. Comparison Table** (`comparison.csv`)

**For Pairs:**
```csv
strategy,test_score,cv_mean,cv_std,fold1,fold2,fold3,fold4,fold5
random_holdout,0.9234,0.9156,0.0123,0.9245,0.9034,0.9289,0.9101,0.9112
stratified_or_timeseries,0.9267,0.9189,0.0098,0.9278,0.9156,0.9245,0.9134,0.9132
```

**For Trios:**
```csv
strategy,test_score,val_score,cv_mean,cv_std,fold1,fold2,fold3,fold4,fold5
random_holdout,0.9234,N/A,0.9156,0.0123,0.9245,0.9034,0.9289,0.9101,0.9112
stratified_or_timeseries,0.9267,N/A,0.9189,0.0098,0.9278,0.9156,0.9245,0.9134,0.9132
nested_holdout,0.9201,0.9178,0.9145,0.0115,0.9234,0.9023,0.9256,0.9089,0.9123
```

**Important Notes:**
- Replace example numbers with your actual results
- Use 4 decimal places for all scores
- For trios: Partners A & B write `N/A` in the `val_score` column
- Partner C's CV scores come from 5-fold CV on the 60% training portion only

### **4. Joint Recommendation Report** (`RECOMMENDATION.md`)

**Required Length:**
- **Pairs**: 200-250 words
- **Trios**: 300-350 words

**Required Sections:**

Create a file with this structure:

```
# Evaluation Strategy Comparison Report

## Dataset Summary
- **Dataset**: [Name]
- **Size**: [samples x features]
- **Task**: [Classification/Regression]
- **Challenge**: [Imbalanced/Temporal/Small/etc.]
- **Metric Used**: [Your metric]

## Comparison Summary
[2-3 sentences for pairs, 3-4 for trios]
- Which strategy showed lower variance?
- Which had higher mean score?
- [Trios only] How does the nested approach compare? Does having a separate validation set provide additional confidence?

## Split Strategy Rationale
[2-3 sentences]
- Why is Partner B's approach (stratified or time-aware) more appropriate for this specific dataset?
- What makes it safer against data leakage compared to random splitting?
- For time-series data: Why is temporal ordering critical?

## Data Leakage Analysis
[2-3 sentences]
- Identify ONE specific potential leakage risk in your dataset
  - For temporal data: Look-ahead bias (using future to predict past)
  - For imbalanced data: Information leakage through improper splitting
  - For feature-rich data: Target leakage through correlated features
- Explain exactly how your chosen splits avoid or mitigate this risk

## Final Recommendation
[3-4 sentences]
- Which evaluation strategy would you trust most for production deployment? Why?
- What are the practical trade-offs (computational cost, complexity, reliability)?
- [Trios only] When would you use a nested holdout approach vs. simpler strategies?
- Under what circumstances might you choose a different strategy?

## Reproducibility Instructions

### Environment Setup
pip install scikit-learn==X.X.X pandas==X.X.X numpy==X.X.X
[list all required packages with versions]

### Running Evaluations
# Partner A
python eval_partner_a.py

# Partner B
python eval_partner_b.py

# Partner C (if trio)
python eval_partner_c.py

### Expected Runtime
- Partner A: ~[X] seconds
- Partner B: ~[X] seconds
- Partner C: ~[X] seconds (if trio)

### Random Seeds
All scripts use random_state=42 for reproducibility.
```

---

## **Submission Instructions**

1. Create a **single GitHub repository** for your pair/trio
2. Name it: `aise-w9d1-splitstrategy-[lastname1]-[lastname2]-[lastname3]`
3. Ensure your repository contains:
   - `TEAM_INFO.md`
   - `eval_partner_a.py`
   - `eval_partner_b.py`
   - `eval_partner_c.py` (if trio)
   - `comparison.csv`
   - `RECOMMENDATION.md`
4. Make the repository public or give access to instructors
5. **Submit the GitHub repository URL in Canvas**


---

## **Grading Rubric**

### **For Pairs (Total: 100 points)**

| Criteria | Points | Description |
|----------|--------|-------------|
| **Evaluation Pipelines** | 30 | Both partners' scripts run correctly with proper CV implementation, correct model/metric usage, and proper random seeds |
| **Comparison Table** | 20 | Complete CSV with all fold-level results, properly formatted, consistent across partners |
| **Analysis Quality** | 30 | Joint recommendation shows clear understanding of variance/stability differences, appropriate for the dataset chosen |
| **Leakage Check** | 15 | Identifies a real, specific leakage risk for the chosen dataset and explains how splits mitigate it |
| **Reproducibility** | 5 | Seeds documented, package versions listed, clear run instructions, code runs without errors |

### **For Trios (Total: 100 points)**

| Criteria | Points | Description |
|----------|--------|-------------|
| **Evaluation Pipelines** | 30 | All three partners' scripts run correctly with proper CV implementation, Partner C's nested approach correctly implemented on 60% train only |
| **Comparison Table** | 20 | Complete CSV with all fold-level results for all 3 strategies, properly formatted with val_score handled correctly |
| **Analysis Quality** | 30 | Joint recommendation compares all three approaches with clear analysis of when nested holdout adds value |
| **Leakage Check** | 15 | Identifies a real, specific leakage risk for the chosen dataset and explains how splits mitigate it |
| **Reproducibility** | 5 | Seeds documented, package versions listed, clear run instructions, all code runs without errors |

---

## **Important Reminders**

✅ **For time-series datasets (#8-10)**: Partner B **must** use `TimeSeriesSplit`

✅ **For imbalanced datasets (#3-4)**: Discuss why your chosen metric (especially PR-AUC) matters in your recommendation

✅ **For Adult dataset (#4)**: Handle categorical encoding (use `OneHotEncoder` or `OrdinalEncoder`) and document your approach

✅ **All partners must use the SAME metric** - agree on this before you start coding

✅ **All partners must use the SAME model with SAME parameters** - no tuning allowed

✅ **Document any preprocessing steps** in your code comments (scaling, encoding, feature engineering)

✅ **Trios**: Partner C's CV happens on the 60% training set only, not the full 80% train split

✅ **Set `random_state=42`** everywhere for reproducibility

---

## **Tips for Success**

1. **Start by meeting as a team** to choose your dataset and agree on your metric
2. **Write `TEAM_INFO.md` first** to document your decisions
3. **Test your individual scripts** before creating the comparison table
4. **Compare results as you go** - if scores are wildly different, debug before moving forward
5. **For small datasets (#2, #5, #7, #9, #10)**: Expect higher variance in CV scores
6. **For time-series data**: Make sure Partner B sees clear temporal patterns in their analysis
7. **Save your environment** with `pip freeze > requirements.txt` for easy reproducibility

---

## **Common Pitfalls to Avoid**

❌ Using different models or metrics across partners
❌ Forgetting to set random seeds consistently
❌ Running CV on test data (CV should only use training data)
❌ For Partner C: Running CV on the full 80% instead of just the 60% training portion
❌ Using `random_state` inconsistently across splits
❌ Not handling categorical features in Adult dataset (#4)
❌ Using regular cross-validation on time-series data (must use TimeSeriesSplit)
❌ Writing generic leakage checks that don't relate to your specific dataset

---

Good luck! This assignment will give you hands-on experience with the evaluation strategies you'll use in production ML systems. 🚀