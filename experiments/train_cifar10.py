"""
Training script for VQVAE on CIFAR-10.
Reproduces Table 3 from "Straightening Out the Straight-Through Estimator" (Huh et al., ICML 2023).

Usage:
    python experiments/train_cifar10.py --config baseline --output-dir experiments/results/baseline
    python experiments/train_cifar10.py --config best --batch-size 128 --output-dir experiments/results/best_bs128
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from configs import TRAIN_DEFAULTS, make_vq_layer
from models import VQVAE


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_perplexity(q, num_codes):
    """Compute perplexity from codebook indices."""
    counts = torch.bincount(q.reshape(-1), minlength=num_codes).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -(probs * probs.log()).sum()
    return entropy.exp()


def effective_dim(X, threshold=0.99):
    """Number of PCA dimensions needed to explain `threshold` fraction of variance.

    Args:
        X: (N, D) tensor of data points.
        threshold: cumulative explained-variance ratio target.

    Returns:
        int: smallest k such that the top-k singular values capture >= threshold of total variance.
    """
    X = X - X.mean(dim=0, keepdim=True)
    S = torch.linalg.svdvals(X)
    var = S ** 2
    cumvar = var.cumsum(0) / var.sum()
    return int((cumvar < threshold).sum().item()) + 1


@torch.no_grad()
def validate(model, val_loader):
    model.eval()
    total_mse = 0.0
    num_batches = 0
    for x, _ in val_loader:
        x = x.cuda()
        x_recon, _ = model(x)
        total_mse += F.mse_loss(x_recon, x).item()
        num_batches += 1
    model.train()
    return total_mse / num_batches


def train(args):
    cfg = TRAIN_DEFAULTS.copy()
    batch_size = args.batch_size if args.batch_size is not None else cfg['batch_size']
    torch.manual_seed(cfg['seed'])

    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    transform = transforms.ToTensor()
    full_train = datasets.CIFAR10(
        root='/orcd/compute/ppliang/001/lxz/vqtorch-data/cifar10', train=True, download=True, transform=transform
    )
    train_dataset = Subset(full_train, range(40000))  # match SQ-VAE 40k/10k split
    val_dataset = Subset(full_train, range(40000, 50000))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model
    use_inplace = (args.config == 'best')
    vq_layer = make_vq_layer(args.config, beta_override=args.beta)
    model = VQVAE(vq_layer, dim_z=cfg['dim_z'], num_rb=cfg['num_rb']).cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
        betas=cfg['betas'],
    )

    total_steps = cfg['epochs'] * len(train_loader)
    warmup_steps = cfg['warmup_epochs'] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Kmeans warmup: run a few batches through VQ layer without gradients
    print(f'Config: {args.config}, batch_size={batch_size}, beta={args.beta}')
    print('Warming up codebook with kmeans...')
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            if i >= 5:
                break
            model(x.cuda())
    model.train()

    # Training loop
    log = []
    for epoch in range(cfg['epochs']):
        epoch_start = time.time()
        epoch_mse = 0.0
        epoch_cmt = 0.0
        epoch_codes = []
        epoch_latents = []
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg["epochs"]}')
        for x, _ in pbar:
            x = x.cuda()
            optimizer.zero_grad()

            x_recon, vq_out = model(x)
            mse_loss = F.mse_loss(x_recon, x)
            cmt_loss = vq_out['loss']

            if use_inplace:
                # With inplace optimizer, only backprop reconstruction loss
                mse_loss.backward()
            else:
                (mse_loss + cfg['alpha'] * cmt_loss).backward()

            optimizer.step()
            scheduler.step()

            epoch_mse += mse_loss.item()
            epoch_cmt += cmt_loss.item()
            epoch_codes.append(vq_out['q'].detach().cpu())
            # Collect encoder latents: (B, C, H, W) -> (B*H*W, C)
            with torch.no_grad():
                z_e = model.encoder(x)
                epoch_latents.append(z_e.permute(0, 2, 3, 1).reshape(-1, z_e.shape[1]).cpu())
            num_batches += 1

            pbar.set_postfix({
                'mse': f'{mse_loss.item():.4f}',
                'cmt': f'{cmt_loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}',
            })

        # Validation
        val_mse = validate(model, val_loader)

        # Epoch stats
        epoch_time = time.time() - epoch_start
        all_codes = torch.cat(epoch_codes, dim=0)
        perplexity = compute_perplexity(all_codes, 1024).item()
        active_pct = all_codes.reshape(-1).unique().numel() / 1024 * 100

        # Effective dimension (99% of variance)
        all_latents = torch.cat(epoch_latents, dim=0).float()
        latent_edim = effective_dim(all_latents)
        cb_weight = model.vq_layer.codebook.weight.detach().cpu().float()
        codebook_edim = effective_dim(cb_weight)

        epoch_log = {
            'epoch': epoch + 1,
            'mse': epoch_mse / num_batches,
            'val_mse': val_mse,
            'cmt_loss': epoch_cmt / num_batches,
            'perplexity': perplexity,
            'active_pct': active_pct,
            'latent_edim': latent_edim,
            'codebook_edim': codebook_edim,
            'epoch_time_sec': epoch_time,
        }
        log.append(epoch_log)
        print(f'  MSE: {epoch_log["mse"]:.5f} | Val MSE: {val_mse:.5f} | '
              f'Perplexity: {perplexity:.1f} | Active: {active_pct:.1f}% | '
              f'Latent eDim: {latent_edim} | CB eDim: {codebook_edim} | '
              f'Time: {epoch_time:.1f}s')

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': args.config,
        'train_defaults': cfg,
        'batch_size': batch_size,
    }, os.path.join(args.output_dir, 'model.pt'))

    with open(os.path.join(args.output_dir, 'train_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    print(f'Saved checkpoint and log to {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, choices=['baseline', 'best'])
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--beta', type=float, default=None, help='Override beta for VQ layer')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    args = parser.parse_args()
    train(args)
