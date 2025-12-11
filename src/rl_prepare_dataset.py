"""
rl_prepare_dataset.py

Converts processed CSVs into a d3rlpy-friendly MDPDataset numpy archive.

For our one-step-per-sample problem:
- state = feature vector
- action = historic action: we don't have historic "approved/denied" labels (dataset only contains accepted loans),
  so we treat action = 1 (approved) for all samples unless you augment with rejected samples.
  However for training an offline RL agent, we will simulate actions:
    - If you want to use the supervised policy as behavior policy, estimate it separately.

This script:
- Loads processed train CSV (with 'target' column)
- Builds states, actions (we'll set action=1 for all), rewards as per reward function described.

Saves to: data/processed/rl_dataset.npz
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def build_rl_dataset(processed_csv_path, out_path, behavior_action_policy=None):
    """
    processed_csv_path: CSV with processed features + 'target'
    out_path: where to save npz with arrays: obs, acts, rewards, next_obs, terminals
    behavior_action_policy: if provided, a callable f(states_df) -> action_probs (for estimating behavior policy)
    """

    df = pd.read_csv(processed_csv_path)
    if 'target' not in df.columns:
        raise ValueError('processed CSV must contain target column')

    # Construct state (all columns except target)
    obs = df.drop(columns=['target']).values.astype(np.float32)
    targets = df['target'].values.astype(int)

    # ACTION: historically these are accepted loans so we set action=1 (approve)
    # If you have a behavior policy or offline info, replace accordingly.
    acts = np.ones(len(obs), dtype=np.int32)

    # Reward function:
    # If action == 0: reward = 0
    # If action == 1:
    #   if target==0 (fully paid): reward = loan_amnt * int_rate
    #   if target==1 (default): reward = - loan_amnt
    # Since loan_amnt and int_rate are scaled in processed CSV, we need an approximation.
    # We'll attempt to reconstruct approximate original reward using columns if present; otherwise use proxy.
    # Here we attempt to use 'loan_amnt' and 'int_rate' columns if they exist unscaled in processed CSV.
    col_names = pd.read_csv(processed_csv_path, nrows=0).columns.tolist()
    # Try to find indices for loan_amnt and int_rate
    reward_values = np.zeros(len(obs), dtype=np.float32)
    if 'loan_amnt' in col_names and 'int_rate' in col_names:
        loan_idx = col_names.index('loan_amnt')
        int_idx = col_names.index('int_rate')
        loan_amnts = df['loan_amnt'].values.astype(float)
        int_rates = df['int_rate'].values.astype(float)
        for i in range(len(obs)):
            if acts[i] == 0:
                reward_values[i] = 0.0
            else:
                if targets[i] == 0:
                    reward_values[i] = loan_amnts[i] * int_rates[i]
                else:
                    reward_values[i] = -loan_amnts[i]
    else:
        # fallback: use proxy reward based on target only (1 for paid, -1 for default)
        print('loan_amnt or int_rate not found in processed CSV; using proxy rewards.')
        reward_values = np.where(targets == 0, 1.0, -1.0)

    # next_obs and terminals: since each sample is a 1-step decision, set next_obs = zeros and terminals=1
    next_obs = np.zeros_like(obs)
    terminals = np.ones(len(obs), dtype=bool)

    np.savez_compressed(out_path, obs=obs, acts=acts, rewards=reward_values, next_obs=next_obs, terminals=terminals)
    print(f'Saved RL dataset to {out_path}')
    return out_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed', type=str, default='../data/processed/train_processed.csv')
    parser.add_argument('--out', type=str, default='../data/processed/rl_dataset.npz')
    args = parser.parse_args()
    build_rl_dataset(args.processed, args.out)