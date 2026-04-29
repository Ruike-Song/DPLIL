"""
Cross-lingual feature analysis on last-token SAE activations.

For each SAE feature, compute CrossLingualConsistency across languages,
then classify as LIVF (language-invariant) or LMF (language-modulated).
"""
import json
import argparse
import itertools

import torch
import numpy as np
from pathlib import Path

from sae_model import SparseAutoencoder


def load_sae(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    sae = SparseAutoencoder(cfg["input_dim"], cfg["hidden_dim"]).to(device)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


def encode_states(sae, states, device, batch_size=4096):
    all_z = []
    for i in range(0, states.shape[0], batch_size):
        batch = states[i:i+batch_size].to(device)
        with torch.no_grad():
            all_z.append(sae.encode(batch).cpu())
    return torch.cat(all_z, dim=0)


def compute_consistency(act_by_lang, langs, eps=1e-8):
    per_lang_mean = torch.stack([act_by_lang[l].mean(dim=0) for l in langs])
    mu = per_lang_mean.mean(dim=0)
    sigma = per_lang_mean.std(dim=0)
    cv = sigma / (mu + eps)
    consistency = 1.0 - cv.clamp(max=1.0)
    return consistency, mu, per_lang_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states_dir", type=str, required=True)
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, required=True)
    parser.add_argument("--langs", type=str, default="en,zh,ko,de")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Consistency threshold for LIVF/LMF classification")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    langs = args.langs.split(",")
    states_dir = Path(args.states_dir)

    sae = load_sae(args.sae_path, device)
    hidden_dim = sae.hidden_dim

    all_consistency = []
    all_active_mask = None

    for layer_idx in layers:
        print(f"\n--- Layer {layer_idx} ---")
        act_by_lang = {}
        for lang in langs:
            fpath = states_dir / f"layer{layer_idx}_{lang}.pt"
            if not fpath.exists():
                print(f"  WARNING: {fpath} not found")
                continue
            states = torch.load(fpath, weights_only=True)
            act_by_lang[lang] = encode_states(sae, states, device)

        consistency, mu, per_lang_mean = compute_consistency(act_by_lang, langs)
        active_mask = mu > 1e-6

        all_consistency.append(consistency)
        if all_active_mask is None:
            all_active_mask = active_mask
        else:
            all_active_mask = all_active_mask | active_mask

    combined_consistency = torch.stack(all_consistency).mean(dim=0)
    active_mask = all_active_mask
    n_active = active_mask.sum().item()

    tau = args.threshold
    is_livf = (combined_consistency > tau) & active_mask
    is_lmvf = (combined_consistency <= tau) & active_mask
    n_livf = is_livf.sum().item()
    n_lmvf = is_lmvf.sum().item()

    print(f"\nClassification (tau={tau}):")
    print(f"  Active: {n_active} / {hidden_dim}")
    print(f"  LIVF:   {n_livf} ({n_livf/max(1,n_active)*100:.1f}%)")
    print(f"  LMF:    {n_lmvf} ({n_lmvf/max(1,n_active)*100:.1f}%)")

    classification = {
        "threshold": tau,
        "n_features": hidden_dim,
        "n_active": n_active,
        "n_livf": n_livf,
        "n_lmvf": n_lmvf,
    }
    with open(save_dir / "feature_classification.json", "w") as f:
        json.dump(classification, f, indent=2)

    torch.save({
        "combined_consistency": combined_consistency,
        "active_mask": active_mask,
        "is_livf": is_livf,
        "is_lmvf": is_lmvf,
        "lmvf_indices": is_lmvf.nonzero(as_tuple=True)[0],
        "livf_indices": is_livf.nonzero(as_tuple=True)[0],
        "lang_order": langs,
    }, save_dir / "crosslingual_analysis.pt")

    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()
