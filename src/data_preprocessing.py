"""
data_preprocessing.py

Functions to:
- load raw LendingClub CSV
- select features
- clean, impute, encode, scale
- produce train/test CSVs under data/processed/

Usage (example):
    from data_preprocessing import preprocess_and_save
    preprocess_and_save('../archive/accepted_2007_to_2018.csv', '../data/processed/')
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# Columns we will use (start with these; you can expand)
SELECTED_COLUMNS = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
    'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
    'purpose', 'addr_state', 'dti', 'open_acc', 'pub_rec', 'total_acc',
    'revol_util', 'loan_status'
]

def load_raw(path, nrows=None):
    print(f'Loading raw CSV: {path}')
    df = pd.read_csv(path, low_memory=False, nrows=nrows)
    return df

def map_target(df):
    """Map loan_status to binary target: 0 -> non-default, 1 -> defaulted"""
    # Define default-like statuses (common in LendingClub)
    default_vals = set([
        'Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off',
        'Late (31-120 days)', 'Late (16-30 days)'  # optionally include late payments
    ])
    # Treat 'Fully Paid' as 0. For safety, treat 'Current' as 0 as well.
    df['target'] = df['loan_status'].apply(lambda x: 1 if x in default_vals else 0)
    return df

def clean_basic(df):
    """Apply basic cleaning and narrow to SELECTED_COLUMNS"""
    # Ensure selected columns exist
    existing = [c for c in SELECTED_COLUMNS if c in df.columns]
    df = df[existing].copy()
    # Convert percentages to numeric
    if 'int_rate' in df.columns:
        if df['int_rate'].dtype == object:
            df['int_rate'] = df['int_rate'].str.rstrip('%').astype(float) / 100.0
    if 'revol_util' in df.columns:
        if df['revol_util'].dtype == object:
            df['revol_util'] = df['revol_util'].str.rstrip('%').astype(float)
    # emp_length map
    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].fillna('0').astype(str)
        def emp_to_years(s):
            s = s.strip()
            if s == '< 1 year': return 0
            if s == '10+ years': return 10
            try:
                return int(s.split()[0])
            except:
                return 0
        df['emp_length_yrs'] = df['emp_length'].apply(emp_to_years)
        df.drop(columns=['emp_length'], inplace=True)
    return df

def impute_and_encode(df, scaler_path=None, fit_scaler=True):
    """Impute missing values, encode categoricals, scale numerics.

    Returns X, y, and saves scaler if scaler_path provided.
    """
    df = df.copy()
    # target
    if 'loan_status' in df.columns and 'target' not in df.columns:
        df = map_target(df)

    if 'target' not in df.columns:
        raise ValueError('target column not found')

    y = df['target'].astype(int).copy()
    df = df.drop(columns=['loan_status', 'target'], errors=True)

    # Fill numeric missing with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        med = df[c].median()
        df[c] = df[c].fillna(med)

    # Fill categorical missing with 'Unknown'
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].fillna('Unknown').astype(str)

    # Reduce cardinality for addr_state (example)
    if 'addr_state' in df.columns:
        top_states = df['addr_state'].value_counts().nlargest(10).index
        df['addr_state'] = df['addr_state'].apply(lambda x: x if x in top_states else 'Other')

    # One-hot encode categorical (limited set)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Add emp_length_yrs if present (already numeric)
    # Scale numeric features
    scaler = StandardScaler()
    if fit_scaler:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        if scaler_path:
            joblib.dump(scaler, scaler_path)
    else:
        if scaler_path and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            df[numeric_cols] = scaler.transform(df[numeric_cols])
        else:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            if scaler_path:
                joblib.dump(scaler, scaler_path)

    X = df
    return X, y

def preprocess_and_save(raw_csv_path, out_dir, nrows=None, test_size=0.2):
    os.makedirs(out_dir, exist_ok=True)
    print('Loading...')
    df = load_raw(raw_csv_path, nrows=nrows)
    print('Mapping target...')
    df = map_target(df)
    print('Cleaning basic columns...')
    df = clean_basic(df)
    print('Imputing & encoding...')
    scaler_path = os.path.join(out_dir, 'scaler.joblib')
    X, y = impute_and_encode(df, scaler_path=scaler_path, fit_scaler=True)

    # Train/test split (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )

    print(f'Saving processed files to {out_dir} ...')
    X_train['target'] = y_train
    X_test['target'] = y_test

    train_path = os.path.join(out_dir, 'train_processed.csv')
    test_path = os.path.join(out_dir, 'test_processed.csv')
    X_train.to_csv(train_path, index=False)
    X_test.to_csv(test_path, index=False)
    print('Saved:', train_path, test_path)
    return train_path, test_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=str, default='../archive/accepted_2007_to_2018.csv')
    parser.add_argument('--out', type=str, default='../data/processed/')
    parser.add_argument('--nrows', type=int, default=None)
    args = parser.parse_args()
    preprocess_and_save(args.raw, args.out, nrows=args.nrows)