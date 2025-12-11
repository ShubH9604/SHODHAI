"""
rl_from_supervised.py

Builds a decision policy from the supervised MLP by finding the decision threshold
that maximizes realized profit on the validation/test data.

Usage:
    # (assuming venv active and preprocessing + supervised model done)
    python src/rl_from_supervised.py \
      --processed data/processed/test_processed.csv \
      --supervised_model models/supervised_mlp.pth \
      --scaler data/processed/scaler.joblib \
      --out reports/rl_supervised_results.csv

Outputs:
- prints best threshold and realized profit
- writes CSV 'reports/rl_supervised_disagreements.csv' with examples of disagreements
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# re-use MLP definition from train_supervised
class MLP(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def load_processed(csv_path):
    df = pd.read_csv(csv_path)
    if 'target' not in df.columns:
        raise ValueError('processed CSV must contain target column')
    y = df['target'].values.astype(int)
    X = df.drop(columns=['target']).copy()
    return df, X, y

def load_model(model_path, input_dim, device='cpu'):
    model = MLP(n_input=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_proba(model, X, device='cpu', batch_size=1024):
    import torch
    model.eval()
    xs = torch.from_numpy(X.values.astype(float)).float().to(device)
    with torch.no_grad():
        probs = model(xs).cpu().numpy().flatten()
    return probs

def compute_realized_profit(df, approve_mask, loan_amnt_col='loan_amnt', int_rate_col='int_rate', actual_default_col='target'):
    """
    Compute realized profit for a binary approve_mask on df.
    Assumes df has loan_amnt and int_rate in *dollar* and decimal (e.g., 0.12) units.
    If those columns are missing or scaled, this function will try to use values present in df.
    """
    if loan_amnt_col not in df.columns or int_rate_col not in df.columns:
        # fallback: if these exact columns are not present, look for something similar
        # Try to find columns containing 'loan_amnt' or 'int_rate' substrings
        candidates = df.columns.tolist()
        loan_cols = [c for c in candidates if 'loan_amnt' in c]
        int_cols = [c for c in candidates if 'int_rate' in c]
        if loan_cols:
            loan_amnt_col = loan_cols[0]
        if int_cols:
            int_rate_col = int_cols[0]
        else:
            raise ValueError('loan_amnt or int_rate column not found in processed CSV. Please ensure these columns exist (unscaled).')

    loan_amnt = df[loan_amnt_col].astype(float).values
    int_rate = df[int_rate_col].astype(float).values
    actual_default = df[actual_default_col].astype(int).values

    profit = np.zeros(len(df), dtype=float)
    # If approve:
    #   if actual_default==0: profit = loan_amnt * int_rate  (interest revenue)
    #   else: profit = - loan_amnt  (loss of principal)
    idx_approve = np.where(approve_mask)[0]
    for i in idx_approve:
        if actual_default[i] == 0:
            profit[i] = loan_amnt[i] * int_rate[i]
        else:
            profit[i] = - loan_amnt[i]
    # if deny -> profit 0
    total_profit = float(profit.sum())
    mean_profit = float(profit.mean())
    return total_profit, mean_profit, profit

def main(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df, X, y = load_processed(args.processed)
    # Load scaler if available and transform X the same way model was trained if needed
    if args.scaler and os.path.exists(args.scaler):
        scaler = joblib.load(args.scaler)
        # scaler expects numeric columns; but since we saved scaled csv, skip applying scaler.
        # We assume the processed CSV is the same format used for training.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.supervised_model, input_dim=X.shape[1], device=device)
    probs = predict_proba(model, X, device=device)
    auc = roc_auc_score(y, probs)
    print(f'Supervised model AUC on provided processed file: {auc:.4f}')

    # Search thresholds 0..1 to maximize realized total profit on this dataset
    thresholds = np.linspace(0.0, 0.9, 91)  # up to 0.9 because very low thresholds approve too many
    best = None
    best_profit = -1e12
    best_details = None
    for t in thresholds:
        approve_mask = (probs < t)  # approve when predicted default prob < t
        total_profit, mean_profit, profit_arr = compute_realized_profit(df, approve_mask,
                                                                       loan_amnt_col=args.loan_amnt_col,
                                                                       int_rate_col=args.int_rate_col,
                                                                       actual_default_col='target')
        if total_profit > best_profit:
            best_profit = total_profit
            best = t
            best_details = (total_profit, mean_profit, profit_arr, approve_mask)
    print(f'Best threshold (max realized total profit) = {best:.4f}, total_profit = {best_details[0]:.2f}, mean_profit_per_loan = {best_details[1]:.6f}')

    # Save disagreements: where supervised proba <0.5 (approve under naive) but threshold policy disagrees, etc.
    probs_series = pd.Series(probs, name='p_default')
    df_out = df.copy()
    df_out['p_default'] = probs_series.values
    df_out['approve_by_threshold'] = best_details[3].astype(int)
    df_out['realized_profit_if_approved'] = best_details[2]
    # A naive supervised policy using 0.5 threshold:
    df_out['approve_by_0.5'] = (df_out['p_default'] < 0.5).astype(int)

    disagree = df_out[df_out['approve_by_threshold'] != df_out['approve_by_0.5']]
    out_csv = os.path.join(os.path.dirname(args.out), 'rl_supervised_disagreements.csv')
    disagree.to_csv(out_csv, index=False)
    print(f'Saved disagreements to {out_csv}')

    # Save summary results
    summary = {
        'best_threshold': float(best),
        'total_profit': float(best_details[0]),
        'mean_profit_per_loan': float(best_details[1]),
        'dataset_rows': int(len(df))
    }
    import json
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print('Saved summary to', args.out)
    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed', type=str, default='data/processed/test_processed.csv')
    parser.add_argument('--supervised_model', type=str, default='models/supervised_mlp.pth')
    parser.add_argument('--scaler', type=str, default='data/processed/scaler.joblib')
    parser.add_argument('--out', type=str, default='reports/rl_supervised_summary.json')
    parser.add_argument('--loan_amnt_col', type=str, default='loan_amnt')
    parser.add_argument('--int_rate_col', type=str, default='int_rate')
    args = parser.parse_args()
    main(args)