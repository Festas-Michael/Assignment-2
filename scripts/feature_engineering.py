"""
feature_engineering.py
=======================
Part 2 – Feature Engineering Pipeline for the Titanic Dataset.

Features created:
  FamilySize     – SibSp + Parch + 1
  IsAlone        – 1 if FamilySize == 1
  Title          – extracted from Name (Mr, Mrs, Miss, Master, Rare)
  Deck           – first letter of Cabin; 'U' if unknown
  AgeGroup       – Child / Teen / Adult / Senior
  FarePerPerson  – Fare / FamilySize
  LogFare        – log1p transform of Fare (reduces right skew)
  LogAge         – log1p transform of Age
  Pclass_x_Fare  – interaction feature
  IsWoman_or_Child – classic domain rule

Encoding:
  One-hot: Sex, Embarked, Title, Deck, AgeGroup
  Ordinal: Pclass kept as integer (1 < 2 < 3 already ordinal)
"""

import pandas as pd
import numpy as np
import os

CLEANED_PATH     = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_cleaned.csv')
ENGINEERED_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_engineered.csv')


def extract_title(name: str) -> str:
    """Pull the title from a Name string like 'Smith, Mr. John'."""
    title = name.split(',')[1].split('.')[0].strip()
    rare_male   = {'Dr', 'Rev', 'Col', 'Major', 'Capt', 'Jonkheer', 'Don', 'Sir'}
    rare_female = {'Lady', 'Countess', 'Dona', 'Mme', 'Mlle', 'Ms'}
    if title in rare_male:
        return 'Rare_Male'
    if title in rare_female:
        return 'Rare_Female'
    if title == 'Miss':
        return 'Miss'
    if title == 'Mrs':
        return 'Mrs'
    if title == 'Mr':
        return 'Mr'
    if title == 'Master':
        return 'Master'
    return 'Other'


def extract_deck(cabin) -> str:
    """Extract deck letter from cabin string; return 'U' if unknown."""
    if pd.isna(cabin) or str(cabin).strip() == '':
        return 'U'
    return str(cabin)[0].upper()


def age_group(age: float) -> str:
    if age < 13:
        return 'Child'
    elif age < 20:
        return 'Teen'
    elif age < 60:
        return 'Adult'
    else:
        return 'Senior'


def engineer(path: str = CLEANED_PATH, save: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[engineer] input shape: {df.shape}")

    # ── Derived features ───────────────────────────────────────────────────────
    df['FamilySize']    = df['SibSp'] + df['Parch'] + 1
    df['IsAlone']       = (df['FamilySize'] == 1).astype(int)
    df['Title']         = df['Name'].apply(extract_title)
    df['Deck']          = df['Cabin'].apply(extract_deck)
    df['AgeGroup']      = df['Age'].apply(age_group)
    df['FarePerPerson'] = (df['Fare'] / df['FamilySize']).round(4)
    df['IsWoman_or_Child'] = (
        (df['Sex'] == 'female') | (df['AgeGroup'] == 'Child')
    ).astype(int)

    # ── Transformations ────────────────────────────────────────────────────────
    df['LogFare'] = np.log1p(df['Fare'])
    df['LogAge']  = np.log1p(df['Age'])

    # ── Interaction features ───────────────────────────────────────────────────
    df['Pclass_x_Fare'] = df['Pclass'] * df['Fare']
    df['Age_x_Pclass']  = df['Age']  * df['Pclass']

    # ── One-hot encoding ───────────────────────────────────────────────────────
    nominal_cols = ['Sex', 'Embarked', 'Title', 'Deck', 'AgeGroup']
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=False, dtype=int)

    # Drop columns not useful for modelling
    drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    print(f"[engineer] output shape: {df.shape}")
    print(f"[engineer] features created: {df.shape[1]} columns")

    if save:
        df.to_csv(ENGINEERED_PATH, index=False)
        print(f"[saved] {ENGINEERED_PATH}")

    return df


if __name__ == '__main__':
    engineer()
