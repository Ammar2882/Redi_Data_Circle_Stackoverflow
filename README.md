# 2025 Stack Overflow Developer Survey — Compensation Analysis

`#ReDI` `#26s_dcp_data_circle` `#data_project`

## Objective

This project uses a structured, end-to-end analytical workflow to investigate global developer trends using the **Stack Overflow 2025 Developer Survey** dataset. It aims to ensure statistical rigour, reproducibility, and alignment with industry benchmarks, in order to:

- Provide actionable insights into the evolving professional landscape
- Identify high-value skill sets
- Support data-driven career positioning

## Case Study: Predictive Modelling of Developer Compensation

This case study examines the complex factors that influence global developer salaries using advanced regression techniques on the 2025 Stack Overflow dataset. The primary objective is to quantify the predictive power of various indicators — age, formal education level, professional experience, developer role, employment type, work environment, geographic location, etc. — on annual compensation. The study serves as a data-driven benchmark for students to assess their career prospects and negotiate compensation based on objective global market trends.

## Data Source and Scope

The 2025 Stack Overflow Developer Survey — a comprehensive primary source representing the global software development ecosystem, collected via a structured online survey targeting developers of all experience levels and specialisations.

## Project Schedule (9 Weeks)

| Week | Focus | Status |
|---|---|---|
| 1 | Project kickoff, team roles, dataset acquisition, environment & repo setup | ✅ Done |
| 2 | Data cleaning & preparation (missing values, duplicates, outliers, feature selection) | 🔄 In progress |
| 3 | Exploratory Data Analysis (distributions, outliers, relationships, geographic visualisations) | ⬜ Not started |
| 4 | Feature engineering & train/test split (75–80% / 25–30%) | ⬜ Not started |
| 5 | Baseline modelling (e.g. linear regression) | ⬜ Not started |
| 6 | Advanced modelling & hyperparameter tuning (Random Forest, Gradient Boosting, CatBoost) | ⬜ Not started |
| 7 | Model evaluation & diagnostics (SHAP, feature importance) | ⬜ Not started |
| 8 | Dashboard development (Streamlit) | ⬜ Not started |
| 9 | Final report, documentation & delivery | ⬜ Not started |

## Repository Structure

```
.
├── data_cleaning.py   # Week 2: cleaning & preparation of the raw survey data
└── README.md
```

As the project progresses through later weeks, scripts/notebooks for EDA, feature engineering, modelling, and the dashboard will be added here.

## Current Progress — Data Cleaning (`data_cleaning.py`)

Starting from the raw survey responses, the cleaning script:

- Selects the core columns relevant to the compensation case study (`ConvertedCompYearly`, `Country`, `WorkExp`, `YearsCode`, `EdLevel`, `Age`, `DevType`, `Employment`, `OrgSize`, `RemoteWork`, `Industry`, `ICorPM`, `LanguageHaveWorkedWith`)
- Removes duplicate responses (excluding `ResponseId`)
- Drops rows with no reported salary (`ConvertedCompYearly`), since it is the prediction target
- Keeps only respondents who are professional developers (`MainBranch`)
- Detects and removes salary outliers, both globally (99th percentile) and relative to each country's median (below 10% of country median), plus a $1,000 floor to remove obvious data-entry errors
- Groups countries with fewer than 50 respondents into an `Other` category
- Drops rows missing core fields (`WorkExp`, `YearsCode`, `EdLevel`) and fills remaining optional fields with `"Unknown"`
- Outputs the cleaned dataset to `clean.csv`

## Getting Started

```bash
pip install pandas
python data_cleaning.py
```

> Note: file paths in the script currently point to a Google Drive / Colab path and will need to be updated to a local or repo-relative path when running outside Colab.

## Tech Stack

- **Data processing:** Python, pandas
- **Modelling (upcoming):** scikit-learn, CatBoost
- **Interpretability (upcoming):** SHAP
- **Dashboard (upcoming):** Streamlit
