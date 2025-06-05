"""
*** LLM Generated ***

Data utilities for machine learning projects.
Common functions for loading, cleaning, and preprocessing data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(filepath, **kwargs):
    """
    Load dataset from various file formats.
    
    Args:
        filepath (str): Path to the dataset file
        **kwargs: Additional arguments passed to pandas read functions
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath, **kwargs)
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        return pd.read_excel(filepath, **kwargs)
    elif filepath.endswith('.json'):
        return pd.read_json(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def explore_dataset(df):
    """
    Quick exploration of a dataset.
    
    Args:
        df (pd.DataFrame): Dataset to explore
    
    Returns:
        dict: Dictionary with exploration results
    """
    exploration = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe() if len(df.select_dtypes(include=[np.number]).columns) > 0 else None,
        'categorical_summary': df.select_dtypes(include=['object']).describe() if len(df.select_dtypes(include=['object']).columns) > 0 else None
    }
    
    print(f"Dataset Shape: {exploration['shape']}")
    print(f"Columns: {exploration['columns']}")
    print(f"\nMissing Values:")
    for col, missing in exploration['missing_values'].items():
        if missing > 0:
            print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
    
    return exploration


def clean_dataset(df, drop_duplicates=True, handle_missing='drop', missing_threshold=0.5):
    """
    Basic dataset cleaning.
    
    Args:
        df (pd.DataFrame): Dataset to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        handle_missing (str): How to handle missing values ('drop', 'fill_mean', 'fill_median', 'fill_mode')
        missing_threshold (float): Drop columns with missing % above this threshold
    
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    df_clean = df.copy()
    
    # Drop duplicates
    if drop_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        print(f"Dropped {initial_rows - len(df_clean)} duplicate rows")
    
    # Handle columns with too many missing values
    missing_pct = df_clean.isnull().sum() / len(df_clean)
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index
    if len(cols_to_drop) > 0:
        df_clean = df_clean.drop(columns=cols_to_drop)
        print(f"Dropped columns with >{missing_threshold*100}% missing: {list(cols_to_drop)}")
    
    # Handle remaining missing values
    if handle_missing == 'drop':
        df_clean = df_clean.dropna()
    elif handle_missing == 'fill_mean':
        df_clean = df_clean.fillna(df_clean.select_dtypes(include=[np.number]).mean())
    elif handle_missing == 'fill_median':
        df_clean = df_clean.fillna(df_clean.select_dtypes(include=[np.number]).median())
    elif handle_missing == 'fill_mode':
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0)
    
    print(f"Final dataset shape: {df_clean.shape}")
    return df_clean


def split_features_target(df, target_column):
    """
    Split dataset into features and target.
    
    Args:
        df (pd.DataFrame): Dataset
        target_column (str): Name of target column
    
    Returns:
        tuple: (X, y) features and target
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y


def prepare_data_for_ml(X, y, test_size=0.2, random_state=42, scale_features=True):
    """
    Prepare data for machine learning: split and optionally scale.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of test set
        random_state (int): Random seed
        scale_features (bool): Whether to scale features
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.dtype == 'object' else None
    )
    
    scaler = None
    if scale_features:
        # Only scale numeric features
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            scaler = StandardScaler()
            X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
            X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, scaler


def encode_categorical_features(X_train, X_test, categorical_columns=None):
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        categorical_columns (list): List of categorical columns to encode
    
    Returns:
        tuple: (X_train_encoded, X_test_encoded, encoders)
    """
    if categorical_columns is None:
        categorical_columns = X_train.select_dtypes(include=['object']).columns
    
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    encoders = {}
    
    for col in categorical_columns:
        if col in X_train.columns:
            encoder = LabelEncoder()
            X_train_encoded[col] = encoder.fit_transform(X_train[col].astype(str))
            X_test_encoded[col] = encoder.transform(X_test[col].astype(str))
            encoders[col] = encoder
    
    return X_train_encoded, X_test_encoded, encoders


def plot_data_distribution(df, columns=None, figsize=(15, 10)):
    """
    Plot distribution of numeric columns.
    
    Args:
        df (pd.DataFrame): Dataset
        columns (list): Columns to plot (if None, plots all numeric columns)
        figsize (tuple): Figure size
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if i < len(axes):
            sns.histplot(data=df, x=col, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
    
    # Hide empty subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, figsize=(12, 8)):
    """
    Plot correlation matrix for numeric columns.
    
    Args:
        df (pd.DataFrame): Dataset
        figsize (tuple): Figure size
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        plt.figure(figsize=figsize)
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough numeric columns for correlation matrix") 