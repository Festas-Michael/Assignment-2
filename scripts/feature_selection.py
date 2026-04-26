"""
feature_selection.py
====================
Part 3 – Feature Selection for the Titanic Dataset.

Methods used:
  1. Correlation analysis  – remove features with |r| > 0.90 with another feature
  2. Random Forest importance – rank by mean decrease in impurity
  3. Threshold selection  – keep features with importance > mean importance

Outputs:
  • selected_features.txt  – list of chosen feature names
  • feature_importances.csv – full ranking table
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

ENGINEERED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_engineered.csv')
OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), '..', 'data')


def correlation_filter(df: pd.DataFrame, target: str = 'Survived',
                        threshold: float = 0.90) -> list:
    """Remove one from each highly-correlated pair (keep the one with higher target corr)."""
    feat_df   = df.drop(columns=[target])
    corr_mat  = feat_df.corr().abs()
    upper     = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    to_drop   = set()

    for col in upper.columns:
        high = upper[col][upper[col] > threshold].index.tolist()
        for h in high:
            # Keep the feature with higher correlation to target
            if abs(df[col].corr(df[target])) >= abs(df[h].corr(df[target])):
                to_drop.add(h)
            else:
                to_drop.add(col)

    print(f"[corr_filter] Dropping {len(to_drop)} highly-correlated features: {to_drop}")
    remaining = [c for c in feat_df.columns if c not in to_drop]
    return remaining


def random_forest_importance(df: pd.DataFrame, features: list,
                              target: str = 'Survived') -> pd.DataFrame:
    """Fit a Random Forest and return a ranked importance DataFrame."""
    X = df[features].fillna(0)
    y = df[target]

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)

    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
    print(f"[RF] 5-fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    importance_df = pd.DataFrame({
        'feature':    features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    return importance_df, rf


def select_features(path: str = ENGINEERED_PATH) -> list:
    df = pd.read_csv(path)
    print(f"[select] Starting with {df.shape[1]-1} features")

    # Step 1: correlation filter
    after_corr = correlation_filter(df)
    print(f"[select] After correlation filter: {len(after_corr)} features")

    # Step 2: Random Forest importance
    imp_df, rf = random_forest_importance(df, after_corr)

    # Step 3: threshold – keep above-mean importance
    mean_imp = imp_df['importance'].mean()
    selected = imp_df[imp_df['importance'] >= mean_imp]['feature'].tolist()
    print(f"\n[select] Mean importance threshold: {mean_imp:.4f}")
    print(f"[select] Selected {len(selected)} features:\n")
    for i, feat in enumerate(selected, 1):
        score = imp_df[imp_df['feature']==feat]['importance'].values[0]
        print(f"  {i:2d}. {feat:<35s} {score:.4f}")

    # Save outputs
    imp_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importances.csv'), index=False)
    with open(os.path.join(OUTPUT_DIR, 'selected_features.txt'), 'w') as f:
        f.write('\n'.join(selected))

    print(f"\n[saved] feature_importances.csv  |  selected_features.txt")
    return selected


if __name__ == '__main__':
    select_features()
