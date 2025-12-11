"""
evaluate_supervised.py

Loads saved model and evaluates on test set, producing ROC and PR curves and metrics.

Usage:
    python src/evaluate_supervised.py --test ../data/processed/test_processed.csv --model models/supervised_mlp.pth --out reports/figures/
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, roc_auc_score
import torch
from train_supervised import MLP  # re-use model definition
import json

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    y = df['target'].values.astype(float)
    X = df.drop(columns=['target']).values.astype(float)
    return X, y

def main(args):
    os.makedirs(args.out, exist_ok=True)
    X_test, y_test = load_data(args.test)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLP(n_input=X_test.shape[1]).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X_test).float().to(device)
        y_proba = model(x).cpu().numpy().flatten()
    roc_auc = roc_auc_score(y_test, y_proba)
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={roc_auc:.4f}')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(args.out, 'roc_curve.png'))
    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall')
    plt.savefig(os.path.join(args.out, 'pr_curve.png'))

    # F1 at 0.5
    y_pred = (y_proba > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)
    metrics = {'roc_auc': float(roc_auc), 'f1_0.5': float(f1)}
    with open(os.path.join(args.out, 'supervised_eval.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Saved figures and metrics to', args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='../data/processed/test_processed.csv')
    parser.add_argument('--model', type=str, default='models/supervised_mlp.pth')
    parser.add_argument('--out', type=str, default='reports/figures/')
    args = parser.parse_args()
    main(args)