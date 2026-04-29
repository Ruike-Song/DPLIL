"""
Diagnostics on pre-collected hidden states:
  1. Instance-conditioned language probe — validates Prediction 1 (decodability)
  2. Group-centered divergence D_p^(k) — tracks routing zone
  3. Feature-level language modulation score phi_i — classifies LMF vs LIVF
"""
import json
import itertools
import argparse

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from sae_model import SparseAutoencoder


def load_layer_states(states_dir, layer_idx, langs):
    states = {}
    for lang in langs:
        fpath = Path(states_dir) / f"layer{layer_idx}_{lang}.pt"
        states[lang] = torch.load(fpath, map_location="cpu", weights_only=True).float()
    return states


def group_center(states, langs):
    stack = torch.stack([states[l] for l in langs], dim=0)
    mean = stack.mean(dim=0, keepdim=True)
    centered = {}
    for i, lang in enumerate(langs):
        centered[lang] = stack[i] - mean.squeeze(0)
    return centered


def run_probe(states_centered, langs, n_splits=5):
    X_parts, y_parts = [], []
    for i, lang in enumerate(langs):
        X_parts.append(states_centered[lang].numpy())
        y_parts.append(np.full(states_centered[lang].shape[0], i))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        accs.append(accuracy_score(y[test_idx], clf.predict(X[test_idx])))
    return float(np.mean(accs)), float(np.std(accs))


def compute_centered_divergence(states_centered, langs):
    pairs = list(itertools.combinations(range(len(langs)), 2))
    cos = nn.CosineSimilarity(dim=1)
    divs = []
    for i, j in pairs:
        sim = cos(states_centered[langs[i]], states_centered[langs[j]])
        divs.append((1.0 - sim).mean().item())
    return float(np.mean(divs))


def compute_phi_scores(states_dir, layer_idx, sae_path, langs, device="cpu"):
    """Per-feature phi_i = E_g[ Var_l( z_hat_{i,g,l} ) ]."""
    ckpt = torch.load(sae_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    sae = SparseAutoencoder(cfg["input_dim"], cfg["hidden_dim"]).to(device)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()

    states = load_layer_states(states_dir, layer_idx, langs)

    z_all = {}
    for lang in langs:
        with torch.no_grad():
            z_all[lang] = sae.encode(states[lang].to(device)).cpu()

    z_stack = torch.stack([z_all[l] for l in langs], dim=0)
    z_mean = z_stack.mean(dim=0, keepdim=True)
    z_std = z_stack.std(dim=0, keepdim=True) + 1e-8
    z_hat = (z_stack - z_mean) / z_std

    var_per_sample = z_hat.var(dim=0)
    phi = var_per_sample.mean(dim=0)
    return phi.numpy()


def rank_features_by_phi(phi, q_values):
    M = len(phi)
    sorted_idx = np.argsort(phi)[::-1]
    result = {}
    for q in q_values:
        n = max(1, int(M * q / 100))
        result[f"LMVF_top{q}"] = sorted_idx[:n].tolist()
        result[f"LIVF_bot{q}"] = sorted_idx[-n:].tolist()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states_dir", type=str, required=True,
                        help="Directory containing layer{L}_{lang}.pt files")
    parser.add_argument("--sae_path", type=str, required=True,
                        help="Path to trained SAE checkpoint (best.pt)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, required=True,
                        help="Comma-separated layer indices")
    parser.add_argument("--langs", type=str, default="en,zh,ko,de")
    parser.add_argument("--phi_q_values", type=str, default="5,10,20,40",
                        help="Percentile values for LMF/LIVF ranking")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    langs = args.langs.split(",")
    q_values = [int(x) for x in args.phi_q_values.split(",")]

    all_results = {}
    for layer in layers:
        print(f"\nLayer {layer}")
        states = load_layer_states(args.states_dir, layer, langs)
        centered = group_center(states, langs)

        probe_acc, probe_std = run_probe(centered, langs)
        div = compute_centered_divergence(centered, langs)
        print(f"  Probe: {probe_acc:.4f} ± {probe_std:.4f}")
        print(f"  Divergence: {div:.6f}")

        phi = compute_phi_scores(args.states_dir, layer, args.sae_path, langs)
        phi_mean = float(np.mean(phi))
        n_active = int((phi > 0.01).sum())
        print(f"  Phi mean={phi_mean:.4f}, active (phi>0.01)={n_active}/{len(phi)}")

        feature_ranks = rank_features_by_phi(phi, q_values)
        np.save(out_dir / f"phi_layer{layer}.npy", phi)
        with open(out_dir / f"feature_ranks_layer{layer}.json", "w") as f:
            json.dump(feature_ranks, f, indent=2)

        all_results[f"layer{layer}"] = {
            "probe_acc": probe_acc, "probe_std": probe_std,
            "divergence": div, "phi_mean": phi_mean, "n_active_phi": n_active,
        }

    with open(out_dir / "probe_divergence_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
