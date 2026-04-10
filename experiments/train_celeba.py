"""
Training script for VQVAE on CelebA (128x128).
Reproduces Table 3 from "Straightening Out the Straight-Through Estimator" (Huh et al., ICML 2023).

Usage:
    python experiments/train_celeba.py --config baseline --output-dir experiments/results/celeba_baseline
    python experiments/train_celeba.py --config best --batch-size 128 --output-dir experiments/results/celeba_best_bs128
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Skip MD5 check for CelebA (files converted from Kaggle CSV format)
datasets.CelebA._check_integrity = lambda self: True
from tqdm import tqdm

from configs import TRAIN_DEFAULTS_CELEBA, make_vq_layer
from models import VQVAE


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_perplexity(q, num_codes):
    counts = torch.bincount(q.reshape(-1), minlength=num_codes).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -(probs * probs.log()).sum()
    return entropy.exp()


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
    cfg = TRAIN_DEFAULTS_CELEBA.copy()
    batch_size = args.batch_size if args.batch_size is not None else cfg['batch_size']
    torch.manual_seed(cfg['seed'])

    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset: CelebA, center crop 140 -> resize 128x128
    transform = transforms.Compose([
        transforms.CenterCrop(140),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    data_root = '/orcd/compute/ppliang/001/lxz/vqtorch-data'
    train_dataset = datasets.CelebA(
        root=data_root, split='train', download=False, transform=transform
    )
    val_dataset = datasets.CelebA(
        root=data_root, split='valid', download=False, transform=transform
    )
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
    model = VQVAE(vq_layer, dim_z=cfg['dim_z'], num_rb=cfg['num_rb'], arch=cfg['arch']).cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
        betas=cfg['betas'],
    )

    total_steps = cfg['epochs'] * len(train_loader)
    warmup_steps = cfg['warmup_epochs'] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Kmeans warmup
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
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg["epochs"]}')
        for x, _ in pbar:
            x = x.cuda()
            optimizer.zero_grad()

            x_recon, vq_out = model(x)
            mse_loss = F.mse_loss(x_recon, x)
            cmt_loss = vq_out['loss']

            if use_inplace:
                mse_loss.backward()
            else:
                (mse_loss + cfg['alpha'] * cmt_loss).backward()

            optimizer.step()
            scheduler.step()

            epoch_mse += mse_loss.item()
            epoch_cmt += cmt_loss.item()
            epoch_codes.append(vq_out['q'].detach().cpu())
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

        epoch_log = {
            'epoch': epoch + 1,
            'mse': epoch_mse / num_batches,
            'val_mse': val_mse,
            'cmt_loss': epoch_cmt / num_batches,
            'perplexity': perplexity,
            'active_pct': active_pct,
            'epoch_time_sec': epoch_time,
        }
        log.append(epoch_log)
        print(f'  MSE: {epoch_log["mse"]:.5f} | Val MSE: {val_mse:.5f} | '
              f'Perplexity: {perplexity:.1f} | Active: {active_pct:.1f}% | '
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
