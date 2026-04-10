"""
Collect all eval results and produce a summary markdown table.

Usage:
    python experiments/collect_results.py
"""

import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'results_summary.md')

# Paper Table 3 reference values
PAPER = {
    'cifar10': {
        'VQVAE baseline': {'mse_x1e3': 5.65, 'perplexity': 14.0, 'lpips_x1e1': 5.43},
        'VQVAE+Affine+OPT+replace+l2': {'mse_x1e3': 1.74, 'perplexity': 608.6, 'lpips_x1e1': 2.27},
    },
    'celeba': {
        'VQVAE baseline': {'mse_x1e3': 10.02, 'perplexity': 16.2, 'lpips_x1e1': 2.71},
        'VQVAE+Affine+OPT+replace+l2': {'mse_x1e3': 4.42, 'perplexity': 872.6, 'lpips_x1e1': 1.36},
    },
}


def load_results():
    cifar10_runs = []
    celeba_runs = []

    for dirname in sorted(os.listdir(RESULTS_DIR)):
        eval_path = os.path.join(RESULTS_DIR, dirname, 'eval_results.json')
        if not os.path.isfile(eval_path):
            continue

        with open(eval_path) as f:
            result = json.load(f)

        # Try to get beta from the saved checkpoint
        ckpt_path = os.path.join(RESULTS_DIR, dirname, 'model.pt')
        beta = None
        if os.path.isfile(ckpt_path):
            import torch
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            cfg = ckpt.get('train_defaults', {})
            # beta is not stored in train_defaults; infer from dirname
            beta = None

        # Infer beta from dirname
        if '_b0995' in dirname:
            beta = 0.995
        elif '_b099' in dirname:
            beta = 0.99
        elif '_b095' in dirname:
            beta = 0.95
        elif result.get('config') == 'best':
            beta = 1.0
        else:
            beta = 0.9  # default

        # Infer batch size: check checkpoint first, then dirname
        batch_size = ckpt.get('batch_size') if os.path.isfile(ckpt_path) else None
        if batch_size is None:
            if '_bs128' in dirname:
                batch_size = 128
            elif '_bs32' in dirname or 'celeba' in dirname:
                batch_size = 32
            else:
                batch_size = 256

        entry = {
            'name': dirname,
            'config': result.get('config', 'unknown'),
            'beta': beta,
            'batch_size': batch_size,
            'mse_x1e3': result.get('mse_x1e3'),
            'perplexity': result.get('perplexity'),
            'active_pct': result.get('active_pct'),
            'lpips_alex_x1e1': result.get('lpips_alex_x1e1', result.get('lpips_x1e1')),
            'lpips_vgg_x1e1': result.get('lpips_vgg_x1e1'),
        }

        if dirname.startswith('celeba'):
            celeba_runs.append(entry)
        else:
            cifar10_runs.append(entry)

    return cifar10_runs, celeba_runs


def format_table(runs, paper_ref):
    header = '| Run | Config | Beta | BS | MSE (x10⁻³) | Perplexity | Active % | LPIPS-vgg (x10⁻¹) | LPIPS-alex (x10⁻¹) |'
    sep = '|---|---|---|---|---|---|---|---|---|'
    rows = [header, sep]

    for r in runs:
        lpips_vgg = f'{r["lpips_vgg_x1e1"]:.2f}' if r.get('lpips_vgg_x1e1') is not None else '—'
        lpips_alex = f'{r["lpips_alex_x1e1"]:.2f}' if r.get('lpips_alex_x1e1') is not None else '—'
        rows.append(
            f'| {r["name"]} | {r["config"]} | {r["beta"]} | {r["batch_size"]} '
            f'| {r["mse_x1e3"]:.2f} | {r["perplexity"]:.1f} | {r["active_pct"]:.1f} '
            f'| {lpips_vgg} | {lpips_alex} |'
        )

    # Add paper reference rows
    for label, vals in paper_ref.items():
        rows.append(
            f'| **Paper: {label}** | — | — | — '
            f'| **{vals["mse_x1e3"]:.2f}** | **{vals["perplexity"]:.1f}** | — '
            f'| **{vals["lpips_x1e1"]:.2f}** | — |'
        )

    return '\n'.join(rows)


def main():
    cifar10_runs, celeba_runs = load_results()

    lines = ['# Results Summary', '']
    lines.append('Comparison against Table 3 of "Straightening Out the Straight-Through Estimator" (Huh et al., ICML 2023).')
    lines.append('')

    lines.append('## CIFAR-10 (32x32)')
    lines.append('')
    if cifar10_runs:
        lines.append(format_table(cifar10_runs, PAPER['cifar10']))
    else:
        lines.append('No results yet.')
    lines.append('')

    lines.append('## CelebA (128x128)')
    lines.append('')
    if celeba_runs:
        lines.append(format_table(celeba_runs, PAPER['celeba']))
    else:
        lines.append('No results yet.')
    lines.append('')

    content = '\n'.join(lines)
    with open(OUTPUT_PATH, 'w') as f:
        f.write(content)
    print(content)
    print(f'\nWritten to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
