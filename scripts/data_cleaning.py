"""
data_cleaning.py
================
Part 1 – Data Cleaning Pipeline for the Titanic Dataset.

Decisions made:
  • Age  : impute with median grouped by Pclass + Sex (more precise than global median)
  • Fare : impute single missing value with median of same Pclass
  • Embarked : impute 2 missing with mode ('S')
  • Cabin : too sparse (>77 % missing) → drop raw column; deck letter extracted later
  • Duplicates : check and remove any exact duplicates on PassengerId
  • Fare outliers : cap at 99th percentile to reduce skew
  • Age outliers  : cap at [1, 80] – plausible human range
"""

import pandas as pd
import numpy as np
import os

RAW_PATH     = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
CLEANED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_cleaned.csv')


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]  shape={df.shape}  survived_rate={df['Survived'].mean():.2%}")
    return df


def report_missing(df: pd.DataFrame) -> None:
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    pct  = (miss / len(df) * 100).round(1)
    print("\n[missing values]")
    print(pd.DataFrame({'count': miss, 'pct': pct}).to_string())


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Age – grouped median imputation
    age_medians = df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
    df['Age_was_missing'] = df['Age'].isnull().astype(int)
    df['Age'] = df['Age'].fillna(age_medians)

    # Fare – class-level median
    fare_median = df.groupby('Pclass')['Fare'].transform('median')
    df['Fare'] = df['Fare'].fillna(fare_median)

    # Embarked – mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Cabin – preserve for deck extraction, then drop
    df['Cabin_known'] = df['Cabin'].notnull().astype(int)
    # raw Cabin column kept until feature_engineering extracts the deck

    print("[missing] After imputation:")
    remaining = df.isnull().sum()
    print(remaining[remaining > 0] if remaining.any() else "  None remaining ✓")
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fare – cap at 99th percentile
    fare_cap = df['Fare'].quantile(0.99)
    outlier_count = (df['Fare'] > fare_cap).sum()
    df['Fare'] = df['Fare'].clip(upper=fare_cap)
    print(f"[outliers] Fare: capped {outlier_count} values at {fare_cap:.2f}")

    # Age – clip to plausible range
    df['Age'] = df['Age'].clip(0.5, 80)
    print("[outliers] Age: clipped to [0.5, 80]")

    return df


def consistency_checks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardise Sex values to lowercase
    df['Sex'] = df['Sex'].str.lower().str.strip()
    invalid_sex = df[~df['Sex'].isin(['male', 'female'])]
    if not invalid_sex.empty:
        print(f"[consistency] Dropped {len(invalid_sex)} rows with invalid Sex value")
        df = df[df['Sex'].isin(['male', 'female'])]

    # Remove duplicate PassengerIds
    before = len(df)
    df = df.drop_duplicates(subset='PassengerId')
    dropped = before - len(df)
    print(f"[consistency] Duplicates removed: {dropped}")

    return df


def clean(path: str = RAW_PATH, save: bool = True) -> pd.DataFrame:
    df = load_data(path)
    report_missing(df)
    df = handle_missing(df)
    df = handle_outliers(df)
    df = consistency_checks(df)

    if save:
        df.to_csv(CLEANED_PATH, index=False)
        print(f"\n[saved] {CLEANED_PATH}  shape={df.shape}")

    return df


if __name__ == '__main__':
    clean()
