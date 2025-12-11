"""
Robust RL training + simple OPE script for SHODHAI.

This script is version-tolerant across several d3rlpy releases:
- tries multiple algorithm class names (DiscreteCQL, CQL, DiscreteIQL, IQL)
- constructs algorithm instances by inspecting constructor signatures
- calls fit(dataset, n_steps=...) with minimal args
- saves policy and behavior classifier
- performs a simple one-step Importance Sampling OPE

Usage:
    python src/rl_train_and_eval.py --dataset data/processed/rl_dataset.npz --out models/rl/ --algo cql --steps 20000
"""
import os
import argparse
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# d3rlpy imports (robust)
try:
    from d3rlpy.dataset import MDPDataset
except Exception:
    # older/newer versions might place this differently
    from d3rlpy.datasets import MDPDataset

# Attempt to import algorithm classes; tolerate missing names
try:
    from d3rlpy.algos import DiscreteCQL
except Exception:
    DiscreteCQL = None
try:
    from d3rlpy.algos import CQL
except Exception:
    CQL = None
try:
    from d3rlpy.algos import DiscreteIQL
except Exception:
    DiscreteIQL = None
try:
    from d3rlpy.algos import IQL
except Exception:
    IQL = None

import inspect
import math

def load_npz_dataset(path):
    d = np.load(path)
    obs = d['obs']
    acts = d['acts']
    rewards = d['rewards']
    # next_obs may not be present; fallback to zeros
    next_obs = d['next_obs'] if 'next_obs' in d.files else np.zeros_like(obs)
    terminals = d['terminals'] if 'terminals' in d.files else np.ones(len(obs), dtype=bool)
    return obs, acts, rewards, next_obs, terminals

def pick_algo_class(algo_name):
    name = algo_name.lower()
    if name == 'cql':
        if DiscreteCQL is not None:
            return DiscreteCQL
        if CQL is not None:
            return CQL
        raise RuntimeError("No CQL variant found in d3rlpy.")
    elif name == 'iql':
        if DiscreteIQL is not None:
            return DiscreteIQL
        if IQL is not None:
            return IQL
        # fall back
        if DiscreteCQL is not None:
            return DiscreteCQL
        if CQL is not None:
            return CQL
        raise RuntimeError("No suitable IQL/CQL variant found in d3rlpy.")
    else:
        raise ValueError(f'Unsupported algo name: {algo_name}')

def make_algo(cls):
    """
    Try multiple constructor styles for cls. Return constructed instance.
    """
    # try no-arg
    try:
        return cls()
    except Exception as e_no:
        last_no = e_no

    # inspect __init__ parameters and build sensible args/kwargs
    sig = inspect.signature(cls.__init__)
    params = list(sig.parameters.keys())[1:]  # drop self
    # defaults
    default_config = None
    default_device = 'cpu'
    default_enable_ddp = False
    # build positional args list
    pos_args = []
    for name in params:
        lname = name.lower()
        if lname in ('config', 'cfg', 'configuration'):
            pos_args.append(default_config)
        elif lname in ('device',):
            pos_args.append(default_device)
        elif lname in ('enable_ddp', 'enableddp', 'ddp'):
            pos_args.append(default_enable_ddp)
        else:
            # generic placeholder
            pos_args.append(None)
    try:
        return cls(*pos_args)
    except Exception as e_pos:
        last_pos = e_pos

    # try keyword args for recognized keys
    kwargs = {}
    for name in params:
        lname = name.lower()
        if lname in ('config', 'cfg', 'configuration'):
            kwargs[name] = default_config
        elif lname in ('device',):
            kwargs[name] = default_device
        elif lname in ('enable_ddp', 'enableddp', 'ddp'):
            kwargs[name] = default_enable_ddp
    try:
        return cls(**kwargs)
    except Exception as e_kw:
        raise RuntimeError(f'Could not instantiate {cls}. Errors: no-arg: {last_no}; positional: {last_pos}; kw: {e_kw}')

def prepare_d3rlpy_dataset(obs, acts, rewards, next_obs, terminals):
    # d3rlpy MDPDataset accepts observations, actions, rewards, terminals for our case.
    try:
        dataset = MDPDataset(observations=obs, actions=acts, rewards=rewards, terminals=terminals)
    except TypeError:
        # older names
        dataset = MDPDataset(obs, acts, rewards, terminals)
    return dataset

def estimate_behavior_policy(obs, acts):
    # Fit a simple behavior classifier p(a|s) using logistic regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(obs, acts)
    return clf

def eval_policy_probs_from_d3rlpy_policy(policy, obs, acts):
    """
    Try to obtain pi_e(a|s) for each sample.
    If policy supports predict_value (Q-values), we softmax them.
    Else if policy supports predict (deterministic), we create deterministic probs.
    Returns: pe_a (n,) = probability of the *actually taken* action under eval policy.
    """
    n = obs.shape[0]
    # try predict_value
    try:
        qs = policy.predict_value(obs)  # shape (n, n_actions)
        # stable softmax
        max_q = qs.max(axis=1, keepdims=True)
        exp_q = np.exp(qs - max_q)
        probs = exp_q / (exp_q.sum(axis=1, keepdims=True) + 1e-12)
        taken = acts.astype(int)
        pe_a = np.array([probs[i, taken[i]] for i in range(n)])
        return pe_a
    except Exception:
        pass
    # try deterministic predict
    try:
        # policy.predict may accept a single observation or batch
        preds = []
        for s in obs:
            a = policy.predict([s])
            if isinstance(a, (list, tuple, np.ndarray)):
                a0 = int(np.array(a).flatten()[0])
            else:
                a0 = int(a)
            preds.append(a0)
        preds = np.array(preds, dtype=int)
        taken = acts.astype(int)
        pe_a = (preds == taken).astype(float)  # 1 if matches, 0 otherwise
        # Avoid zero division later by adding small epsilon
        pe_a = pe_a * 0.999 + 1e-6
        return pe_a
    except Exception:
        pass
    # fallback: uniform small prob
    n_actions = int(np.max(acts) + 1)
    taken = acts.astype(int)
    pe_a = np.ones(len(acts)) * (1.0 / max(n_actions, 1))
    return pe_a

def importance_sampling_estimate(eval_policy, behavior_clf, obs, acts, rewards):
    # Get behavior probabilities for taken actions
    pb = behavior_clf.predict_proba(obs)  # shape (n, k)
    # prob of actual action under behavior
    taken = acts.astype(int)
    pb_a = np.array([pb[i, taken[i]] for i in range(len(acts))])
    # Get eval policy probabilities
    pe_a = eval_policy_probs_from_d3rlpy_policy(eval_policy, obs, acts)
    # compute IS weights and estimate
    eps = 1e-9
    weights = pe_a / (pb_a + eps)
    estimate = float(np.mean(weights * rewards))
    stderr = float(np.std(weights * rewards) / math.sqrt(len(rewards)))
    return estimate, stderr

def main(args):
    os.makedirs(args.out, exist_ok=True)
    obs, acts, rewards, next_obs, terminals = load_npz_dataset(args.dataset)
    print('Loaded dataset shapes:', obs.shape, acts.shape, rewards.shape)

    dataset = prepare_d3rlpy_dataset(obs, acts, rewards, next_obs, terminals)

    # pick algorithm class
    algo_cls = pick_algo_class(args.algo)
    print('Using algorithm class:', algo_cls)

    # construct algorithm instance robustly
    algo = make_algo(algo_cls)
    print('Constructed algorithm:', algo)

    # Fit with minimal signature
    print('Training offline RL algorithm...')
    try:
        algo.fit(dataset, n_steps=args.steps)
    except TypeError:
        # Try older signature: algo.fit(dataset, n_steps)
        algo.fit(dataset, args.steps)

    # Save policy (try multiple save methods)
    policy_path = os.path.join(args.out, f'{args.algo}_policy')
    try:
        algo.save_policy(policy_path)
        print('Saved policy to', policy_path)
    except Exception:
        try:
            # some versions use save() on policy object
            pol = algo.create_policy()
            try:
                pol.save(policy_path)
                print('Saved policy via pol.save to', policy_path)
            except Exception:
                # fallback: joblib dump
                joblib.dump(algo, policy_path + '.joblib')
                print('Saved full algo joblib to', policy_path + '.joblib')
        except Exception as e:
            joblib.dump(algo, policy_path + '.joblib')
            print('Saved algo fallback to', policy_path + '.joblib', 'err:', e)

    # Estimate behavior policy and perform simple OPE (IS)
    print('Estimating behavior policy for OPE...')
    behavior_clf = estimate_behavior_policy(obs, acts)
    joblib.dump(behavior_clf, os.path.join(args.out, 'behavior_clf.joblib'))

    # Wrap learned policy
    try:
        learned_policy = algo.create_policy()
    except Exception:
        # maybe algo itself is a policy wrapper
        learned_policy = algo

    est, stderr = importance_sampling_estimate(learned_policy, behavior_clf, obs, acts, rewards)
    print(f'IS estimated policy value: {est:.4f} +/- {1.96*stderr:.4f} (approx 95% CI)')

    # save OPE results
    with open(os.path.join(args.out, 'rl_metrics.txt'), 'w') as f:
        f.write(f'IS_estimate: {est}\nstderr:{stderr}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../data/processed/rl_dataset.npz')
    parser.add_argument('--out', type=str, default='models/rl/')
    parser.add_argument('--algo', type=str, default='cql')
    parser.add_argument('--steps', type=int, default=20000)
    args = parser.parse_args()
    main(args)
