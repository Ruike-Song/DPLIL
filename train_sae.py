"""
Train Sparse Autoencoder on collected decoder hidden states.
Trains on combined hidden states from all languages so the SAE dictionary
captures features that appear across different linguistic contexts.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
import argparse

from sae_model import SparseAutoencoder


def train(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    print("Loading training data...")
    tokens = torch.load(args.data_path, weights_only=True)
    print(f"  Shape: {tokens.shape}, dtype: {tokens.dtype}")

    dataset = TensorDataset(tokens)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )

    model = SparseAutoencoder(
        input_dim=args.input_dim, hidden_dim=args.hidden_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"SAE: {total_params:,} params ({total_params/1e6:.1f}M)")

    global_step = 0
    best_cosine = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_mse, epoch_cos, epoch_steps = 0.0, 0.0, 0

        for (batch,) in dataloader:
            batch = batch.to(device)
            x_hat, z = model(batch)

            mse_loss = F.mse_loss(x_hat, batch)
            l1_loss = z.abs().sum(dim=-1).mean()
            loss = mse_loss + args.l1_coeff * l1_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if global_step % args.norm_every == 0:
                model.normalize_decoder()

            with torch.no_grad():
                cos = F.cosine_similarity(x_hat, batch, dim=-1).mean().item()
                epoch_mse += mse_loss.item()
                epoch_cos += cos
                epoch_steps += 1
            global_step += 1

        scheduler.step()
        avg_cos = epoch_cos / epoch_steps
        avg_mse = epoch_mse / epoch_steps
        print(f"Epoch {epoch+1}/{args.num_epochs} | mse={avg_mse:.4f} | cos={avg_cos:.4f}")

        if avg_cos > best_cosine:
            best_cosine = avg_cos
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "cosine_sim": avg_cos,
                "config": {
                    "input_dim": args.input_dim,
                    "hidden_dim": args.hidden_dim,
                    "l1_coeff": args.l1_coeff,
                },
            }, save_dir / "best.pt")
            print(f"  -> New best cosine: {best_cosine:.4f}")

    model.eval()
    print("\nFinal evaluation...")
    all_metrics = {"mse": 0, "cosine_sim": 0, "l0_sparsity": 0, "frac_dead_features": 0}
    n_batches = 0
    with torch.no_grad():
        for (batch,) in dataloader:
            batch = batch.to(device)
            metrics = model.compute_metrics(batch)
            for k in all_metrics:
                all_metrics[k] += metrics[k]
            n_batches += 1
    for k in all_metrics:
        all_metrics[k] /= n_batches

    print(f"  MSE:              {all_metrics['mse']:.4f}")
    print(f"  Cosine Similarity:{all_metrics['cosine_sim']:.4f}")
    print(f"  L0 Sparsity:      {all_metrics['l0_sparsity']:.1f} / {args.hidden_dim}")

    with open(save_dir / "train_log.json", "w") as f:
        json.dump({"final_metrics": all_metrics, "best_cosine": best_cosine}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to combined hidden states .pt file")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--input_dim", type=int, required=True,
                        help="Hidden size of the VLM (e.g. 2048 or 4096)")
    parser.add_argument("--hidden_dim", type=int, required=True,
                        help="SAE dictionary size (e.g. 4x overcomplete)")
    parser.add_argument("--l1_coeff", type=float, required=True,
                        help="L1 sparsity penalty coefficient")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--norm_every", type=int, default=100,
                        help="Normalize decoder weights every N steps")
    args = parser.parse_args()
    train(args)
