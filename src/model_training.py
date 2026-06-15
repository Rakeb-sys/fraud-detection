import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split as sk_train_test_split
import matplotlib.pyplot as plt
import shap

from pathlib import Path


def prepare_and_split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Separates features from target and performs a stratified train-test split.
    """
    # 1. Separate features (X) from target variable (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 2. Perform Stratified Train-Test Split (aliased to avoid recursion)
    X_train, X_test, y_train, y_test = sk_train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y  # Preserves class distribution
    )
    
    # Optional diagnostic logging
    print(f"\n--- Distribution Summary for Target: '{target_column}' ---")
    print(f"Train Shape: {X_train.shape} | Test Shape: {X_test.shape}")
    print("Train class ratios:\n", y_train.value_counts(normalize=True).to_dict())
    print("Test class ratios:\n", y_test.value_counts(normalize=True).to_dict())
    
    return X_train, X_test, y_train, y_test

# ---------------------------------------------------------
# Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    # Process Credit Card Data
    try:
        print("Loading creditcard.csv...")
        credit_df = pd.read_csv('creditcard.csv')
        
        X_credit_train, X_credit_test, y_credit_train, y_credit_test = prepare_and_split_data(
            df=credit_df, 
            target_column='Class'
        )
    except FileNotFoundError:
        print("Error: 'creditcard.csv' not found.")

    # Process Fraud Data
    try:
        print("\nLoading Fraud_Data.csv...")
        fraud_df = pd.read_csv('Fraud_Data.csv')
        
        X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = prepare_and_split_data(
            df=fraud_df, 
            target_column='class'
        )
    except FileNotFoundError:
        print("Error: 'Fraud_Data.csv' not found.")

def optimize_threshold(y_true, probas):
    """Finds the decision threshold that maximizes the F1-score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probas)
    # Avoid division by zero
    f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, 
                          out=np.zeros_like(precisions), where=(precisions + recalls) > 0)
    best_idx = np.argmax(f1_scores[:-1]) # last element has recall=0, precision=1
    return thresholds[best_idx], f1_scores[best_idx]

def evaluate_with_cv(model_name, model_obj, X, y, cv_folds=5):
    """
    Performs 5-Fold Stratified Cross-Validation.
    Tracks and averages F1, Precision, Recall, and ROC-AUC dynamically.
    """
    SEED = 42
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    
    metrics = {'F1': [], 'Precision': [], 'Recall': [], 'ROC-AUC': [], 'PR-AUC':[], 'Threshold': []}
    
    # Convert to numpy arrays for reliable indexing across K-Fold splits
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]
        
        # Clone and train model
        model = model_obj
        model.fit(X_train, y_train)
        
        # Predict probabilities
        probas = model.predict_proba(X_val)[:, 1]
        
        # Find best operational threshold on validation fold
        best_thresh, _ = optimize_threshold(y_val, probas)
        preds = (probas >= best_thresh).astype(int)
        
        # Calculate snapshot metrics
        precisions, recalls, thresholds = precision_recall_curve(y_val, probas)
        f1 = f1_score(y_val, preds, zero_division=0)
        report = classification_report(y_val, preds, output_dict=True, zero_division=0)
        auc_roc = roc_auc_score(y_val, probas)
        auc_pr = average_precision_score(y_val, probas) 
        
        metrics['F1'].append(f1)
        metrics['Precision'].append(report['1']['precision'])
        metrics['Recall'].append(report['1']['recall'])
        metrics['ROC-AUC'].append(auc_roc)
        metrics['PR-AUC'].append(auc_pr)
        metrics['Threshold'].append(best_thresh)

        # Printout for engineering visibility
        print(f"   -> Fold {fold+1} | F1: {f1:.4f} | Precision: {report['1']['precision']:.4f} | Recall: {report['1']['recall']:.4f}")

    # Aggregate performance
    summary = {}
    for metric, values in metrics.items():
        summary[f'{metric}_mean'] = np.mean(values)
        summary[f'{metric}_std'] = np.std(values)
        summary[f'{metric}_raw_folds'] = values
    print(f"✓ {model_name} 5-Fold CV Completed. Mean F1: {summary['F1_mean']:.4f}")
    return summary


def plot_feature_importance(model, feature_names, top_n=10, save_path=None):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
    else:
        raise ValueError('Model does not expose feature importances.')

    importance = pd.Series(importance, index=feature_names).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot.barh(ax=ax, color='#3182bd')
    ax.invert_yaxis()
    ax.set_title('Top Feature Importances')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved feature importance plot to {save_path}')
    plt.close(fig)
    return importance


def shap_explainability(model, X_test, y_test, feature_names, save_dir=None, sample_size=1000):
    """
    Generate SHAP plots and display them directly in the notebook.
    
    Parameters:
        model: Trained model (e.g., XGBoost, RandomForest, etc.).
        X_test: Test dataset (features).
        y_test: True labels for the test dataset.
        feature_names: List of feature names.
        save_dir: Directory to save generated SHAP plots (default: None).
        sample_size: Number of samples to use for SHAP analysis (default: 1000).
    """
    # Sample a subset of the test set for SHAP analysis
    sample_idx = np.random.RandomState(42).choice(X_test.shape[0], min(sample_size, X_test.shape[0]), replace=False)
    X_sample = X_test.iloc[sample_idx]

    # Create SHAP explainer and compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Generate and display the SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    if save_dir:
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir_path / 'shap_summary_plot.png', dpi=150, bbox_inches='tight')
    plt.show()  # Ensure plot is displayed in the notebook

    # Identify one true positive, false positive, and false negative using the full test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    # Find indices for true positive, false positive, and false negative
    indices = {
        'true_positive': np.where((y_test == 1) & (y_pred == 1))[0],
        'false_positive': np.where((y_test == 0) & (y_pred == 1))[0],
        'false_negative': np.where((y_test == 1) & (y_pred == 0))[0],
    }

    selected = {}
    for label, idx_array in indices.items():
        if len(idx_array) > 0:
            selected[label] = idx_array[0]

    # Generate and display SHAP force plots for selected instances
    for label, idx in selected.items():
        instance = X_test.iloc[[idx]]
        if idx in sample_idx:
            instance_shap_values = shap_values[np.where(sample_idx == idx)[0][0]]
        else:
            instance_shap_values = explainer.shap_values(instance)[0]

        # Display the force plot directly in the notebook
        shap.force_plot(
            explainer.expected_value,
            instance_shap_values,
            instance,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        if save_dir:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir_path / f'shap_force_plot_{label}.png', dpi=150, bbox_inches='tight')
        plt.show()

    return {
        'feature_names': feature_names,
        'shap_values': shap_values,
        'selected_instances': selected,
        'confusion_matrix': cm,
    }