import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.constants import NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL, COLUMN_NAMES
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve
)
# Resampling
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

import matplotlib.pyplot as plt

# Setup structured logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ==========================================
# FUNCTIONAL LOGIC CORES
# ==========================================

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts granular time insights from transaction timestamps."""
    df = df.copy()
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    # Signup profile features
    df['signup_hour'] = df['signup_time'].dt.hour
    df['signup_dayofweek'] = df['signup_time'].dt.day_of_week
    df['signup_month'] = df['signup_time'].dt.month
    df['signup_year'] = df['signup_time'].dt.year

    # Purchase profile features (FIXED: purchasep_year typo resolved)
    df['purchase_hour'] = df['purchase_time'].dt.hour
    df['purchase_dayofweek'] = df['purchase_time'].dt.day_of_week
    df['purchase_month'] = df['purchase_time'].dt.month
    df['purchase_year'] = df['purchase_time'].dt.year
    
    return df


# ==========================================
# SKLEARN PIPELINE COMPLIANT COMPONENT CLASSES
# ==========================================

class FeatureEngineerTransform(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        # FIXED: Added explicit return statement to pass frame down pipeline
        return feature_engineer(X)
    

class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols: Optional[List[str]] = None):
        self.numerical_cols = numerical_cols

    def fit(self, X: pd.DataFrame, y=None):
        if self.numerical_cols is None:
            self.numerical_cols_ = [
                col for col in X.columns
                if pd.api.types.is_numeric_dtype(X[col])
            ]
        else:
            self.numerical_cols_ = [col for col in self.numerical_cols if col in X.columns]

        self.scaler_ = StandardScaler()
        if self.numerical_cols_:
            self.scaler_.fit(X[self.numerical_cols_])
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        if hasattr(self, 'numerical_cols_') and self.numerical_cols_:
            X[self.numerical_cols_] = self.scaler_.transform(X[self.numerical_cols_])
        return X


class DataFrameLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols: Optional[List[str]] = None):
        self.categorical_cols = categorical_cols

    def fit(self, X: pd.DataFrame, y=None):
        present_cols = [
            col for col in X.columns if col in (self.categorical_cols or X.columns)
        ]
        
        # Isolate true strings for encoding, ignoring raw baseline timestamps
        self.cols_to_encode_ = [
            col for col in present_cols
            if (not pd.api.types.is_numeric_dtype(X[col]) or col in (self.categorical_cols or []))
            and col not in ["signup_time", "purchase_time"]
        ]

        self.encoders_ = {}
        self.fallback_maps_ = {}
        
        for col in self.cols_to_encode_:
            
            #Fill missing values and force everything to a clean string type
            filled_series = X[col].fillna('unknown').astype(str)
            # Cast to string and include a universal unseen fallback label
            unique_vals =filled_series.unique().tolist()
            if 'unknown' not in unique_vals:
                unique_vals.append('unknown')
                
            le = LabelEncoder()
            le.fit(unique_vals)
            self.encoders_[col] = le
            
            # Save a fast lookup map containing known classes for testing mutations
            self.fallback_maps_[col] = set(le.classes_)
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, le in self.encoders_.items():
            if col in X.columns:
                # FIXED: Safely intercept values not encountered during fit() 
                # mapping them to our pre-fit 'unknown' array element to avoid ValueError crashes
                valid_set = self.fallback_maps_[col]
                X[col] = X[col].astype(str).apply(lambda val: val if val in valid_set else 'unknown')
                X[col] = le.transform(X[col])
        return X


# ==========================================
# EXECUTION WRAPPER HOOK
# ==========================================

def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fits execution parameters strictly across training folds, morphing evaluation data splits safely."""
    scaler_transformer = FeatureScaler(numerical_cols=num_cols)
    scaler_transformer.fit(X_train)
    
    X_train_scaled = scaler_transformer.transform(X_train)
    X_val_scaled = scaler_transformer.transform(X_val)
    X_test_scaled = scaler_transformer.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler_transformer.scaler_


def evaluate(name, model, X_test_sc, y_test, color='#3498db'):
    """
    Evaluate a trained model.
    Automatically finds the threshold that gives the best F1 score,
    then reports all metrics at that threshold.
    """
    # Get probability scores (not hard labels)
    proba = model.predict_proba(X_test_sc)[:, 1]  # probability of being Default

    # Find the best threshold: sweep all possible cut-offs
    # and pick the one that maximises F1
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_test, proba)
    f1_arr    = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
    best_idx  = np.argmax(f1_arr)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    # Apply the best threshold
    y_pred = (proba >= best_thresh).astype(int)

    # Compute all metrics
    rec   = recall_score(y_test, y_pred, zero_division=0)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    aucpr = average_precision_score(y_test, proba)  # threshold-independent
    aucroc= roc_auc_score(y_test, proba)

    print(f'\n=== {name} ===  (best threshold: {best_thresh:.2f})')
    print(f'  Recall    : {rec:.2%}  ← % of real defaults caught')
    print(f'  Precision : {prec:.2%}  ← % of alerts that are real')
    print(f'  F1-Score  : {f1:.4f}')
    print(f'  AUC-PR    : {aucpr:.4f}  ← primary metric (threshold-independent)')
    print(f'  AUC-ROC   : {aucroc:.4f}')

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred),
        display_labels=['Repaid', 'Default']
    ).plot(ax=ax, cmap='Blues', colorbar=False)
    caught = confusion_matrix(y_test, y_pred)[1, 1]
    total  = y_test.sum()
    ax.set_title(f'{name}\nCaught {caught}/{total} defaults  |  Recall={rec:.0%}',
                 fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.show()

    # Return metrics for later comparison
    return dict(name=name, recall=rec, precision=prec, f1=f1,
                auc_pr=aucpr, auc_roc=aucroc,
                threshold=best_thresh, proba=proba, y_pred=y_pred)

