"""
Evaluation script for VQVAE on CIFAR-10 test set.
Computes MSE, Perplexity, and LPIPS (Table 3 metrics).

Usage:
    python experiments/evaluate.py --checkpoint experiments/results/baseline/model.pt --config baseline
    python experiments/evaluate.py --checkpoint experiments/results/best/model.pt --config best
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from configs import TRAIN_DEFAULTS, make_vq_layer
from models import VQVAE


def compute_perplexity(q, num_codes):
    counts = torch.bincount(q.reshape(-1), minlength=num_codes).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -(probs * probs.log()).sum()
    return entropy.exp()


@torch.no_grad()
def evaluate(args):
    cfg = TRAIN_DEFAULTS.copy()

    # Dataset
    transform = transforms.ToTensor()
    test_dataset = datasets.CIFAR10(
        root='/orcd/compute/ppliang/001/lxz/vqtorch-data/cifar10', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model
    vq_layer = make_vq_layer(args.config)
    model = VQVAE(vq_layer, dim_z=cfg['dim_z'], num_rb=cfg['num_rb']).cuda()

    checkpoint = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # LPIPS (both alex and vgg backbones)
    try:
        import lpips
        lpips_alex = lpips.LPIPS(net='alex').cuda()
        lpips_vgg = lpips.LPIPS(net='vgg').cuda()
    except ImportError:
        print('WARNING: lpips not installed. Skipping LPIPS metric. Install with: pip install lpips')
        lpips_alex = None
        lpips_vgg = None

    total_mse = 0.0
    total_lpips_alex = 0.0
    total_lpips_vgg = 0.0
    all_codes = []
    num_samples = 0

    for x, _ in test_loader:
        x = x.cuda()
        x_recon, vq_out = model(x)

        # MSE
        total_mse += F.mse_loss(x_recon, x, reduction='sum').item()

        # LPIPS (expects input in [-1, 1])
        if lpips_alex is not None:
            x_scaled = x * 2 - 1
            xr_scaled = x_recon * 2 - 1
            total_lpips_alex += lpips_alex(x_scaled, xr_scaled).sum().item()
            total_lpips_vgg += lpips_vgg(x_scaled, xr_scaled).sum().item()

        all_codes.append(vq_out['q'].cpu())
        num_samples += x.size(0)

    # Aggregate
    mse = total_mse / (num_samples * 3 * 32 * 32)  # per-pixel MSE
    all_codes = torch.cat(all_codes, dim=0)
    perplexity = compute_perplexity(all_codes, 1024).item()
    active_pct = all_codes.reshape(-1).unique().numel() / 1024 * 100

    results = {
        'config': args.config,
        'mse': mse,
        'mse_x1e3': mse * 1000,
        'perplexity': perplexity,
        'active_pct': active_pct,
    }

    if lpips_alex is not None:
        results['lpips_alex'] = total_lpips_alex / num_samples
        results['lpips_alex_x1e1'] = results['lpips_alex'] * 10
        results['lpips_vgg'] = total_lpips_vgg / num_samples
        results['lpips_vgg_x1e1'] = results['lpips_vgg'] * 10

    # Print results
    print(f'\n=== Evaluation Results ({args.config}) ===')
    print(f'MSE (x10^-3):    {results["mse_x1e3"]:.2f}')
    print(f'Perplexity:      {perplexity:.1f}')
    print(f'Active codes:    {active_pct:.1f}%')
    if 'lpips_alex_x1e1' in results:
        print(f'LPIPS-alex (x10^-1):  {results["lpips_alex_x1e1"]:.2f}')
        print(f'LPIPS-vgg  (x10^-1):  {results["lpips_vgg_x1e1"]:.2f}')

    print(f'\n--- Table 3 targets (CIFAR-10) ---')
    if args.config == 'baseline':
        print(f'MSE (x10^-3):    5.65')
        print(f'Perplexity:      14.0')
        print(f'LPIPS (x10^-1):  5.43')
    elif args.config == 'best':
        print(f'MSE (x10^-3):    1.74')
        print(f'Perplexity:      608.6')
        print(f'LPIPS (x10^-1):  2.27')

    # Save results to JSON next to checkpoint
    output_path = os.path.join(os.path.dirname(args.checkpoint), 'eval_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True, choices=['baseline', 'best'])
    args = parser.parse_args()
    evaluate(args)
