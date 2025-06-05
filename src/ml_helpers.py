"""
*** LLM Generated ***

Machine Learning helper functions.
Common utilities for model evaluation, visualization, and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import learning_curve, validation_curve
import warnings
warnings.filterwarnings('ignore')


def evaluate_classification_model(y_true, y_pred, y_pred_proba=None, class_names=None):
    """
    Comprehensive evaluation of a classification model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        class_names: Names of classes (optional)
    
    Returns:
        dict: Dictionary with evaluation metrics
    """
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return results


def evaluate_regression_model(y_true, y_pred):
    """
    Comprehensive evaluation of a regression model.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        dict: Dictionary with evaluation metrics
    """
    results = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    print(f"Mean Squared Error: {results['mse']:.4f}")
    print(f"Root Mean Squared Error: {results['rmse']:.4f}")
    print(f"Mean Absolute Error: {results['mae']:.4f}")
    print(f"R² Score: {results['r2']:.4f}")
    
    return results


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(8, 6)):
    """
    Plot confusion matrix with nice formatting.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, figsize=(8, 6)):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        figsize: Figure size
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, figsize=(8, 6)):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        figsize: Figure size
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_learning_curve(model, X, y, cv=5, figsize=(10, 6)):
    """
    Plot learning curve to analyze bias-variance tradeoff.
    
    Args:
        model: Scikit-learn model
        X: Features
        y: Target
        cv: Cross-validation folds
        figsize: Figure size
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_validation_curve(model, X, y, param_name, param_range, cv=5, figsize=(10, 6)):
    """
    Plot validation curve for hyperparameter tuning.
    
    Args:
        model: Scikit-learn model
        X: Features
        y: Target
        param_name: Name of parameter to vary
        param_range: Range of parameter values
        cv: Cross-validation folds
        figsize: Figure size
    """
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=cv, n_jobs=-1, scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=figsize)
    plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title(f'Validation Curve for {param_name}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20, figsize=(10, 8)):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: Names of features
        top_n: Number of top features to show
        figsize: Figure size
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model doesn't have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top_n features
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(top_features)), top_importances[::-1])
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.show()


def plot_regression_results(y_true, y_pred, figsize=(12, 4)):
    """
    Plot regression model results.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Actual vs Predicted
    ax1.scatter(y_true, y_pred, alpha=0.6)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Actual vs Predicted Values')
    ax1.grid(True)
    
    # Residuals
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def compare_models(models_dict, X_train, X_test, y_train, y_test, metric='accuracy'):
    """
    Compare multiple models and return results.
    
    Args:
        models_dict: Dictionary with model names as keys and model objects as values
        X_train, X_test, y_train, y_test: Train/test splits
        metric: Evaluation metric ('accuracy' for classification, 'r2' for regression)
    
    Returns:
        pd.DataFrame: Comparison results
    """
    results = []
    
    for name, model in models_dict.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        if metric == 'accuracy':
            train_score = accuracy_score(y_train, train_pred)
            test_score = accuracy_score(y_test, test_pred)
        elif metric == 'r2':
            train_score = r2_score(y_train, train_pred)
            test_score = r2_score(y_test, test_pred)
        else:
            raise ValueError("Metric must be 'accuracy' or 'r2'")
        
        results.append({
            'Model': name,
            f'Train {metric.capitalize()}': train_score,
            f'Test {metric.capitalize()}': test_score,
            'Overfitting': train_score - test_score
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(f'Test {metric.capitalize()}', ascending=False)
    
    print("Model Comparison Results:")
    print("=" * 50)
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    return results_df


def create_sample_dataset(dataset_type='classification', n_samples=1000, n_features=10, random_state=42):
    """
    Create sample datasets for testing.
    
    Args:
        dataset_type: 'classification' or 'regression'
        n_samples: Number of samples
        n_features: Number of features
        random_state: Random seed
    
    Returns:
        tuple: (X, y) features and target
    """
    from sklearn.datasets import make_classification, make_regression
    
    if dataset_type == 'classification':
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=n_features//2,
            n_redundant=0, n_clusters_per_class=1, random_state=random_state
        )
    elif dataset_type == 'regression':
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features, noise=0.1,
            random_state=random_state
        )
    else:
        raise ValueError("dataset_type must be 'classification' or 'regression'")
    
    return X, y 