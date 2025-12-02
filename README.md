<p align="center">
  <img src="assets/cJtc.png" alt="Justice Through Code Banner" width="500"/>
</p>

<h1 align="center">🧪 AISE Week 9 — Split Strategy Showdown</h1>
<h3 align="center">Team: Andrea Churchwell & Jose Diaz</h3>

<p align="center">
  <img src="https://img.shields.io/badge/AISE-2025-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Week-9-informational?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dataset-Diabetes-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Metric-R²-success?style=for-the-badge"/>
</p>

---

## ❤️ Why We Chose the Diabetes Dataset

We selected the Diabetes Regression dataset (#7) because it fits the assignment requirements and it carries personal meaning for both of us. José's mother has diabetes, and Andrea’s cocker spaniel, Ace, developed diabetes later in life. Even though this is a technical evaluation project, using a dataset connected to real life makes the work feel more grounded and motivating.

---

## 📌 Project Overview

This repository contains our team implementation for the **AISE 26 W9D1 Split Strategy Showdown**.  
Our goal is to compare two evaluation strategies using:

- the **same dataset**
- the **same model** (Ridge Regression)
- the **same metric** (R²)
- and produce a clear comparison + recommendation report.

Currently, **Partner A (Andrea)** has completed all required evaluation steps, including visuals and CSV results.

---

## 🔧 Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-2.3.3-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-2.3.5-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white"/>
</p>

---

## ✅ Current Progress

### ✔ Repository Setup
- Repo initialized  
- `.gitignore` created  
- Required W9D1 assignment files added  
- Project structure cleaned and simplified

### ✔ Virtual Environment Working
- venv created  
- Dependencies installed  
- Jupyter kernel connected (`aise_w9d1_venv`)

### ✔ Dataset & Metric
- **Dataset:** Diabetes Regression (#7)  
- **Metric:** **R²** (coefficient of determination)

### ✔ Partner A (Andrea) — Completed
- Loaded dataset  
- Built Ridge Regression + StandardScaler pipeline  
- 80/20 Random Holdout complete  
- 5-Fold CV complete  
- All scores printed  
- Plotly visuals generated  
- `comparison.csv` updated with Partner A results  
- Internal notes completed (dataset, model, scaling, R² analysis)

### ⏳ Partner B (Jose) — Pending
- Implement stratified/time-aware split (if applicable)  
- Run model + metric  
- Add results to `comparison.csv`

### ⏳ TEAM_INFO.md — Pending Finalization

### ⏳ RECOMMENDATION.md — Pending (after Partner B completes part)
---

## 📂 Project Structure
```
├── assets/
│ ├── cJtc.png
│ └── jtc.png
├── partner_a_visuals/
├── eval_partner_a.py
├── eval_partner_b.py
├── comparison.csv
├── TEAM_INFO.md
├── RECOMMENDATION.md
├── partner_a_notebook.ipynb
└── README.md
```

---

## 🚀 Next Steps

### 🔸 Partner B
- Implement stratified or specialized CV  
- Use **same model + metric**  
- Add results to `comparison.csv`

### 🔸 TEAM_INFO.md
- Add team names  
- Add roles (A = random holdout, B = stratified/time-aware)  
- Add dataset info & loading code  
- Add agreed-upon metric  
- Add package versions

### 🔸 RECOMMENDATION.md
- Analyze variance  
- Compare stability  
- Discuss leakage risks  
- Give final recommendation (200–250 words)

---

## 📝 Notes

This README will continue to evolve as the assignment progresses.  
Once Partner B completes his portion, we will finalize the comparison and recommendation.

---

<p align="center"><i>Built with ❤️ by Andrea & Jose • JTC AISE 2025</i></p>
<p align="center">
  <img src="assets/jtc.png" alt="JTC Icon" width="90"/>
</p>