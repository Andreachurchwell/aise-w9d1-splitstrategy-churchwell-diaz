<p align="center">
  <img src="assets/cJtc.png" alt="Justice Through Code Banner" width="500"/>
</p>

<h1 align="center">🧪 AISE Week 9 — Split Strategy Showdown</h1>
<h3 align="center">Team: Andrea Churchwell & Jose Diaz</h3>

<p align="center">
  <img src="https://img.shields.io/badge/AISE-2026-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Week-9-informational?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dataset-Diabetes-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Metric-R²-success?style=for-the-badge"/>
</p>

---

## ❤️ Why We Chose the Diabetes Dataset

We selected the Diabetes Regression dataset (#7) because it fits the assignment requirements **and** it carries personal meaning for both of us. José's mother has diabetes, and Andrea’s cocker spaniel, Ace, developed diabetes later in life. Even though this is a technical evaluation project, using a dataset connected to real life makes the work feel more grounded and motivating.

---

## 📌 Project Overview

This repository contains our team implementation for the **AISE 26 W9D1 Split Strategy Showdown**.

Our goal is to compare two evaluation strategies using:

- the **same dataset** (Diabetes Regression #7)
- the **same model** (`Ridge` Regression inside a `Pipeline` with `StandardScaler`)
- the **same metric** (**R²**)
- and produce a clear **comparison + recommendation report** based on both numeric scores and visual diagnostics.

We implemented:

- **Partner A (Andrea)** – Random 80/20 holdout + 5-fold **KFold**  
- **Partner B (Jose)** – Ordered 80/20 holdout + 5-fold **TimeSeriesSplit** (time-aware style)

Both partners use the **same model & metric**, as required.

---

## 🔧 Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-2.3.3-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-2.3.5-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-6.5.0-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/Rich-14.2.0-0D0D0D?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white"/>
</p>

---

## ✅ Current Progress

### ✔ Dataset & Metric

- **Dataset:** Diabetes Regression (`sklearn.datasets.load_diabetes`) – Assignment Dataset #7  
- **Task Type:** Regression  
- **Metric:** **R²** (coefficient of determination) – agreed upon for both partners

### ✔ Partner A (Andrea) – Random Holdout + KFold

- Loaded dataset  
- Built `Pipeline(StandardScaler + Ridge)`  
- 80/20 **random** train/test split (`train_test_split`, `shuffle=True`, `random_state=42`)  
- 5-Fold **KFold** cross-validation on training set  
- All scores printed (train/test + per-fold CV, mean, std)  
- Plotly visuals generated and saved to HTML + PNG:
  - CV bar chart
  - Actual vs predicted scatter
  - Residuals histogram
- `comparison.csv` updated with Partner A metrics  
- Additional notes in `partnerA-notes.txt` and `partner_comparison.md`

### ✔ Partner B (Jose) – Ordered Holdout + TimeSeriesSplit

- Loaded the same dataset & uses the same model/metric  
- 80/20 **ordered** train/test split:
  - First 80% rows → train  
  - Last 20% rows → test  
- 5-Fold **TimeSeriesSplit** on the 80% training portion  
- All scores printed (train/test + per-fold CV, mean, std)  
- Plotly visuals generated and saved to HTML + PNG:
  - CV bar chart
  - Actual vs predicted scatter
  - Residuals histogram
- `comparison.csv` updated with Partner B metrics

### 🧪 Partner Comparison – In Progress (Almost There)

- `partner_comparison.md` created to:
  - Summarize both strategies
  - Compare R² scores and CV variance
  - Analyze residuals and prediction patterns
  - Embed side-by-side chart screenshots from `assets/`
- Final polish and word count checks will happen after RECOMMENDATION is finished.

### 📄 TEAM_INFO & RECOMMENDATION

- `TEAM_INFO.md` – structure ready, content being finalized  
- `RECOMMENDATION.md` – will be written after reviewing the final metrics + visuals together

---

## 📂 Project Structure

```text
├── assets/
│   ├── cJtc.png                   # JTC banner
│   ├── jtc.png                    # JTC icon
│   ├── partner_a_cvr2.png         # Andrea – CV bar chart
│   ├── partner_a_actual_vs_pred.png
│   ├── partner_a_residuals.png
│   ├── partner_b_cvr2.png         # Jose – CV bar chart
│   ├── partner_b_actual_vs_pred.png
│   ├── partner_b_residuals.png
│   └── (other screenshots as needed)
│
├── partner_a_visuals/             # HTML Plotly charts for Partner A
├── partner_b_visuals/             # HTML Plotly charts for Partner B
│
├── eval_partner_a.py              # Andrea – Random Holdout + KFold
├── eval_partner_b.py              # Jose – Ordered Holdout + TimeSeriesSplit
│
├── comparison.csv                 # Combined scores for both strategies
├── TEAM_INFO.md                   # Team + dataset + metric info (per assignment)
├── RECOMMENDATION.md              # Final written recommendation report
│
├── partner_comparison.md          # Visual + narrative comparison (side-by-side charts)
├── partnerA-notes.txt             # Andrea’s working notes
│
├── partner_a_notebook.ipynb       # (Optional) Jupyter notebook for exploration
├── requirements.txt               # Minimal project dependencies
└── README.md      

```

## ⚙️ Setup & How to Run
### From the project root:

### 1. Create and activate virtual environment (if not already)
```
python -m venv venv
source venv/bin/activate    # macOS/Linux
# OR
venv\Scripts\activate       # Windows
```
### 2. Install dependencies
```
pip install -r requirements.txt
```
### 3. Run Partner A pipeline
```
python eval_partner_a.py
```
### 4. Run Partner B pipeline
```
python eval_partner_b.py
```
### Both scripts will:

- Load the Diabetes dataset

- Run their respective split + CV strategies

- Print R² scores (train/test, per-fold, mean, std)

- Save Plotly visuals into their partner_*_visuals/ folders

- comparison.csv aggregates the final metrics for both strategies.

## 🚀 Project Status
| Step                      | Status                        |
| ------------------------- | ----------------------------- |
| Repo created              | ✅ Done                        |
| venv + `requirements.txt` | ✅ Done                        |
| Jupyter kernel configured | ✅ Done                        |
| Dataset selected (#7)     | ✅ Done                        |
| Metric selected (R²)      | ✅ Done                        |
| Partner A code + visuals  | ✅ Done                        |
| Partner B code + visuals  | ✅ Done                        |
| `comparison.csv` updated  | ✅ Done                        |
| `partner_comparison.md`   | ✅ Drafted                     |
| `TEAM_INFO.md`            | ⏳ Finalizing                  |
| `RECOMMENDATION.md`       | ⏳ Pending (after full review) |

### 📝 Notes
This repository is intentionally small and focused on evaluation strategy, not model tuning.
We keep the model and metric fixed and only change how we split and validate, then use:

- cross-validation scores

- variance across folds

- residual analysis

- and side-by-side visuals

to decide which strategy we would trust most for this dataset.

Once RECOMMENDATION.md is complete, this project will be fully ready for submission.


---

<p align="center"><i>Built with ❤️ by Andrea & Jose • JTC AISE 2026</i></p>
<p align="center">
  <img src="assets/jtc.png" alt="JTC Icon" width="90"/>
</p>