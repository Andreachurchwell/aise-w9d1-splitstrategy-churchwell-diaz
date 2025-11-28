# aise-w9d1-splitstrategy-churchwell-diaz
Week 9 Split Strategy Showdown
# 🧪 AISE Week 9 — Split Strategy Showdown  
### **Team: Andrea Churchwell & Jose Diaz**

---

## 📌 Project Overview
This repository contains our team implementation for the AISE 26 W9D1 Split Strategy Showdown.
Our goal is to compare different evaluation strategies on the same dataset, using the same model and the same metric, and analyze how each strategy impacts stability, variance, and trustworthiness.

We are now mid-way through Part A development, with the environment fully working and Partner A’s Jupyter workflow set up correctly.

---

## ✅ Current Progress

### ✔ Repository + Structure Ready
- GitHub repo initialized  
- .gitignore created  
- Required scaffold files added:
  - TEAM_INFO.md
  - eval_partner_a.py
  - eval_partner_b.py
  - comparison.csv
  - RECOMMENDATION.md

### ✔ Virtual Environment Working
- venv/ successfully created  
- All dependencies installed  
- Jupyter kernel connected inside VS Code (aise_w9d1_venv)

### ✔ Jupyter Notebook Working
- partner_a_notebook.ipynb created  
- Imports, dataset loading, splitting all verified working

### ✔ Dataset Selected
**Dataset #7 — Diabetes Regression Dataset**

### ✔ Metric Selected
**R² (Coefficient of Determination)**

### ✔ Partner A Code Completed (Functionally)
- Data loaded and explored  
- 80/20 Random Holdout implemented  
- Ridge Regression + StandardScaler pipeline  
- Test R² score printed  
- 5-fold KFold CV implemented  
- CV mean + std printed  

---

## ⏳ Next Steps
### 🔸 1. Partner B (José)
Implement evaluation using:
- same dataset  
- same metric (R²)  
- stratified or specialized CV (based on assignment instructions)

### 🔸 2. Fill in TEAM_INFO.md
Add:
- names & roles  
- dataset (#7)  
- metric (R²)  
- why we chose this dataset  
- code block for loading diabetes dataset  
- package versions

### 🔸 3. Populate comparison.csv
After both scripts run, record:
- test score  
- CV mean  
- CV std  
- fold-by-fold results  
for Partner A and Partner B.

### 🔸 4. Write RECOMMENDATION.md
Final 200–250 word analysis comparing:
- variance  
- stability  
- leakage risk  
- which strategy we'd trust  

---

## 🚀 Project Status

| Step                     | Status      |
|--------------------------|------------ |
| Repo created             | ✅ Done    |
| venv + requirements.txt  | ✅ Done    |
| Jupyter kernel fixed     | ✔️ Done    |
| Dataset selected (#7)    | ✔️ Done    |
| Metric selected (R²)     | ✔️ Done    |
| Partner A code           | ✔️ Done    |
| Partner B code           | ⏳ Pending |
| comparison.csv           | ⏳ Pending |
| TEAM_INFO.md             | ⏳ Pending |
| RECOMMENDATION.md        | ⏳ Pending |


---

## 📝 Notes

This README will continue evolving as we finalize the dataset and metric and begin implementing the required evaluation strategies.


