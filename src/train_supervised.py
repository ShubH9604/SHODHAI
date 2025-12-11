"""
Train a supervised MLP classifier (PyTorch) on the processed CSV.

Saves the best model and a small metrics JSON.

Usage:
    python src/train_supervised.py --train ../data/processed/train_processed.csv --test ../data/processed/test_processed.csv --out models/
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, precision_recall_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'target' not in df.columns:
        raise ValueError('target column missing in processed CSV')
    y = df['target'].values.astype(float)
    X = df.drop(columns=['target']).values.astype(float)
    return X, y

def train_loop(model, opt, criterion, loader, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).float().unsqueeze(1)
        opt.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def eval_model(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float()
            out = model(xb).cpu().numpy().flatten()
            preds.append(out)
            ys.append(yb.numpy().flatten())
    y_true = np.concatenate(ys)
    y_proba = np.concatenate(preds)
    auc = roc_auc_score(y_true, y_proba)
    return auc, y_true, y_proba

def main(args):
    os.makedirs(args.out, exist_ok=True)
    X_train, y_train = load_data(args.train)
    X_test, y_test = load_data(args.test)

    # optional small val split from train
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train
    )

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)
    test_loader = DataLoader(test_ds, batch_size=512)

    model = MLP(n_input=X_train.shape[1]).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    best_auc = -1
    best_path = os.path.join(args.out, 'supervised_mlp.pth')
    for epoch in range(1, args.epochs + 1):
        loss = train_loop(model, opt, criterion, train_loader, DEVICE)
        val_auc, _, _ = eval_model(model, val_loader, DEVICE)
        print(f'Epoch {epoch}/{args.epochs} loss={loss:.4f} val_auc={val_auc:.4f}')
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_path)
    print('Best val AUC:', best_auc)

    # Load best and evaluate on test
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_auc, y_true, y_proba = eval_model(model, test_loader, DEVICE)
    # choose threshold to maximize F1 on (val set) OR use 0.5
    thresh = 0.5
    y_pred = (y_proba > thresh).astype(int)
    f1 = f1_score(y_true, y_pred)
    print(f'Test AUC: {test_auc:.4f}, F1@{thresh}: {f1:.4f}')

    # Save metrics and artifacts
    metrics = {'test_auc': float(test_auc), 'test_f1': float(f1), 'threshold': float(thresh)}
    with open(os.path.join(args.out, 'supervised_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Saved metrics to', os.path.join(args.out, 'supervised_metrics.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='../data/processed/train_processed.csv')
    parser.add_argument('--test', type=str, default='../data/processed/test_processed.csv')
    parser.add_argument('--out', type=str, default='models/')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    main(args)