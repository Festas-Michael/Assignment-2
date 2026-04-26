# 🚢 Unsinkable — Titanic Survival Analysis

> *A narrative-driven ML pipeline for the Titanic survival prediction problem.*

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.787-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()

---

## Overview

**Unsinkable** is a structured machine learning project that builds a survival prediction model for the RMS Titanic disaster. Rather than treating this as a mechanical notebook exercise, the project frames every data decision as a question: *what does this column actually tell us about who lived and who died?*

The full pipeline covers:

| Part | Task | Output |
|------|------|--------|
| 1 | Data Cleaning | `train_cleaned.csv` |
| 2 | Feature Engineering | `train_engineered.csv` |
| 3 | Feature Selection | `selected_features.txt`, `feature_importances.csv` |

**Baseline result:** 5-fold CV ROC-AUC = **0.787** with 12 selected features.

---

## Project Structure

```
unsinkable/
│
├── data/
│   ├── train.csv                  ← Raw Titanic dataset
│   ├── train_cleaned.csv          ← After Part 1
│   ├── train_engineered.csv       ← After Part 2
│   ├── feature_importances.csv    ← Random Forest ranking
│   └── selected_features.txt      ← Final 12 features
│
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb   ← Full walkthrough
│
├── scripts/
│   ├── data_cleaning.py           ← Part 1 pipeline
│   ├── feature_engineering.py     ← Part 2 pipeline
│   └── feature_selection.py       ← Part 3 pipeline
│
├── report/
│   └── index.html                 ← Interactive visual report
│
├── README.md
└── requirements.txt
```

---

## Quickstart

### 1. Clone & install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/unsinkable.git
cd unsinkable
pip install -r requirements.txt
```

### 2. Get the dataset

Download from [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data) and place `train.csv` and `test.csv` in the `data/` folder.

### 3. Run the full pipeline

```bash
# Part 1: Data Cleaning
python scripts/data_cleaning.py

# Part 2: Feature Engineering
python scripts/feature_engineering.py

# Part 3: Feature Selection
python scripts/feature_selection.py
```

### 4. Explore the notebook

```bash
jupyter notebook notebooks/Titanic_Feature_Engineering.ipynb
```

### 5. View the visual report

Open `report/index.html` in any browser for the full interactive analysis.

---

## Approach

### Part 1 — Data Cleaning

**Missing Values:**

| Column | Missing % | Strategy | Rationale |
|--------|-----------|----------|-----------|
| `Cabin` | 79.1% | Drop (extract deck first) | Too sparse; deck letter is recoverable |
| `Age` | 19.4% | Grouped median (Pclass × Sex) | Sub-group medians reduce imputation bias |
| `Embarked` | 0.7% | Mode ('S') | Only 6 rows; 'S' is most common port |
| `Fare` | <0.1% | Class-level median | Single missing value |

**Outliers:**
- `Fare`: capped at 99th percentile (£213.15) — 9 passengers affected
- `Age`: clipped to [0.5, 80] — physiologically impossible values only

**Consistency:** Sex values standardised to lowercase; duplicate PassengerIds removed.

---

### Part 2 — Feature Engineering

**11 original features → 37 engineered columns**

| Feature | Formula | Why It Matters |
|---------|---------|---------------|
| `FamilySize` | SibSp + Parch + 1 | Solo & large-group travellers had lower survival |
| `IsAlone` | FamilySize == 1 | Binary flag for solo travellers |
| `Title` | Extracted from Name | Captures gender + social rank simultaneously |
| `Deck` | First letter of Cabin | Physical proximity to lifeboats |
| `AgeGroup` | Child/Teen/Adult/Senior | Non-linear age effect on survival |
| `FarePerPerson` | Fare / FamilySize | Normalised wealth proxy |
| `IsWoman_or_Child` | (Sex==female) OR (Age<13) | Encodes "women and children first" rule |
| `LogFare` | log1p(Fare) | Reduces right-skew for linear models |
| `Age_x_Pclass` | Age × Pclass | Age penalised more in lower classes |
| `Pclass_x_Fare` | Pclass × Fare | Compound wealth × status signal |

**Encoding:** One-hot for nominal features (Sex, Embarked, Title, Deck, AgeGroup); Pclass kept as ordinal integer.

---

### Part 3 — Feature Selection

**37 → 12 features in two steps:**

1. **Correlation filter** (|r| > 0.90): removed 2 redundant features — `Sex_female` (mirror of `Sex_male`) and `Deck_U` (correlated with `Cabin_known`).

2. **Random Forest importance** (300 trees, max_depth=8, 5-fold CV): features ranked by Mean Decrease in Impurity; those above the mean threshold selected.

**Final 12 selected features:**

```
Sex_male, IsWoman_or_Child, Title_Mr, Age_x_Pclass,
LogFare, Fare, LogAge, Age, FarePerPerson,
Pclass_x_Fare, Title_Miss, Title_Mrs
```

---

## Key Findings

- **Sex** is the dominant predictor: females survived at 74.8% vs 22.3% for males (3.4× difference)
- **Class compounded inequality**: 1st-class females had ~95% survival; 3rd-class males ~12%
- **Title** captures gender *and* social rank in one variable — `Master` (young boys) vs `Mr` (adult men)
- **FamilySize** shows a non-linear survival pattern — small families (2–4) fared best
- **Deck** proxies for class: upper decks (A–C) showed higher survival; 'Unknown' deck skews 3rd-class
- **Interaction features** (`Age_x_Pclass`) ranked 4th in importance, outperforming raw features alone

---

## Requirements

See `requirements.txt`. Core dependencies:

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
```

---

## Dataset

- **Source:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- **Training set:** 891 passengers with survival labels
- **Overall survival rate:** 41.5%

---

## License

MIT — free to use for academic and personal projects.

---

*Project: Unsinkable · AI Assignment 2 · Titanic Dataset Analysis*
