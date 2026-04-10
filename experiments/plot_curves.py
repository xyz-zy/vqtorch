"""
Plot training and validation loss curves for all runs.

Usage:
    python experiments/plot_curves.py
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
OUTPUT_DIR = os.path.dirname(__file__)


def load_logs():
    cifar10_logs = {}
    celeba_logs = {}

    for dirname in sorted(os.listdir(RESULTS_DIR)):
        log_path = os.path.join(RESULTS_DIR, dirname, 'train_log.json')
        if not os.path.isfile(log_path):
            continue
        with open(log_path) as f:
            log = json.load(f)
        if dirname.startswith('celeba'):
            celeba_logs[dirname] = log
        else:
            cifar10_logs[dirname] = log

    return cifar10_logs, celeba_logs


def plot_dataset(logs, title, output_path):
    has_val = any('val_mse' in entry for log in logs.values() for entry in log)

    fig, axes = plt.subplots(1, 2 if has_val else 1, figsize=(14 if has_val else 7, 5))
    if not has_val:
        axes = [axes]

    # Train MSE
    ax = axes[0]
    for name, log in logs.items():
        epochs = [e['epoch'] for e in log]
        mse = [e['mse'] for e in log]
        ax.plot(epochs, mse, label=name, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train MSE')
    ax.set_yscale('log')
    ax.set_title(f'{title} — Train MSE')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Val MSE (if any run has it)
    if has_val:
        ax = axes[1]
        for name, log in logs.items():
            if 'val_mse' not in log[0]:
                continue
            epochs = [e['epoch'] for e in log]
            val_mse = [e['val_mse'] for e in log]
            ax.plot(epochs, val_mse, label=name, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val MSE')
        ax.set_yscale('log')
        ax.set_title(f'{title} — Val MSE')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved {output_path}')


def print_timing(logs, title):
    print(f'\n=== {title} — Avg Epoch Time ===')
    for name, log in logs.items():
        times = [e.get('epoch_time_sec') for e in log if e.get('epoch_time_sec') is not None]
        if times:
            avg = sum(times) / len(times)
            total = sum(times)
            print(f'  {name}: {avg:.1f}s/epoch, {total/3600:.2f}h total')
        else:
            print(f'  {name}: no timing data')


def main():
    cifar10_logs, celeba_logs = load_logs()

    if cifar10_logs:
        plot_dataset(cifar10_logs, 'CIFAR-10', os.path.join(OUTPUT_DIR, 'cifar10_curves.png'))
        print_timing(cifar10_logs, 'CIFAR-10')

    if celeba_logs:
        plot_dataset(celeba_logs, 'CelebA', os.path.join(OUTPUT_DIR, 'celeba_curves.png'))
        print_timing(celeba_logs, 'CelebA')


if __name__ == '__main__':
    main()
