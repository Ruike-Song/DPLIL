"""
PCA + Probe: test whether language information resides in a low-dimensional
subspace (Prediction 3: low-rank separability).

Tests:
  1. PCA to various dimensions, then probe accuracy
  2. Random-label baseline to validate probe meaningfulness
"""
import json
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


def load_layer_states(states_dir, layer_idx, langs):
    states = {}
    for lang in langs:
        fpath = Path(states_dir) / f"layer{layer_idx}_{lang}.pt"
        states[lang] = torch.load(fpath, map_location="cpu", weights_only=True).float()
    return states


def group_center(states, langs):
    stack = torch.stack([states[l] for l in langs], dim=0)
    mean = stack.mean(dim=0, keepdim=True)
    return {lang: stack[i] - mean.squeeze(0) for i, lang in enumerate(langs)}


def build_dataset(states_centered, langs):
    X_parts, y_parts = [], []
    for i, lang in enumerate(langs):
        X_parts.append(states_centered[lang].numpy())
        y_parts.append(np.full(states_centered[lang].shape[0], i))
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def run_probe_cv(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        accs.append(accuracy_score(y[test_idx], clf.predict(X[test_idx])))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, required=True)
    parser.add_argument("--langs", type=str, default="en,zh,ko,de")
    parser.add_argument("--pca_dims", type=str, default="1,2,5,10,50,100,500")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    langs = args.langs.split(",")
    pca_dims = [int(x) for x in args.pca_dims.split(",")]

    all_results = {}
    for layer in layers:
        print(f"\nLayer {layer}")
        states = load_layer_states(args.states_dir, layer, langs)
        centered = group_center(states, langs)
        X, y = build_dataset(centered, langs)

        pca_results = {}
        for n_dim in pca_dims:
            if n_dim > X.shape[1]:
                continue
            pca = PCA(n_components=n_dim, random_state=42)
            X_pca = pca.fit_transform(X)
            var_explained = pca.explained_variance_ratio_.sum()
            acc, std = run_probe_cv(X_pca, y)
            pca_results[n_dim] = {"acc": acc, "std": std, "var_explained": float(var_explained)}
            print(f"  dim={n_dim:>4}: acc={acc:.4f}±{std:.4f}, var={var_explained:.4f}")

        acc_full, std_full = run_probe_cv(X, y)
        pca_results["full"] = {"acc": acc_full, "std": std_full, "var_explained": 1.0}
        print(f"  dim=full: acc={acc_full:.4f}±{std_full:.4f}")

        rand_results = {}
        for n_dim in [5, 10, 50]:
            if n_dim > X.shape[1]:
                continue
            pca_r = PCA(n_components=n_dim, random_state=42)
            X_pca_r = pca_r.fit_transform(X)
            random_accs = []
            for seed in range(10):
                y_rand = np.random.RandomState(seed).permutation(y)
                acc_rand, _ = run_probe_cv(X_pca_r, y_rand)
                random_accs.append(acc_rand)
            rand_results[n_dim] = {
                "mean": float(np.mean(random_accs)),
                "std": float(np.std(random_accs)),
            }
            print(f"  Random (PCA={n_dim}): {rand_results[n_dim]['mean']:.4f}")

        all_results[f"layer{layer}"] = {
            "pca_probe": pca_results,
            "random_label_baselines": rand_results,
        }

    with open(out_dir / "pca_probe_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
