"""
Hyperparameter configurations for Table 3 reproduction.
"""

import torch
from vqtorch.nn import VectorQuant


def make_vq_layer(config_name, beta_override=None):
    if config_name == 'baseline':
        return VectorQuant(
            feature_size=64,
            num_codes=1024,
            beta=beta_override if beta_override is not None else 0.9,
            kmeans_init=True,
        )
    elif config_name == 'best':
        inplace_optimizer = lambda *args, **kwargs: torch.optim.SGD(
            *args, **kwargs, lr=50.0, momentum=0.9
        )
        return VectorQuant(
            feature_size=64,
            num_codes=1024,
            beta=1.0,
            kmeans_init=True,
            norm='l2',
            cb_norm='l2',
            affine_lr=10.0,
            replace_freq=20,
            inplace_optimizer=inplace_optimizer,
        )
    else:
        raise ValueError(f'Unknown config: {config_name}')


TRAIN_DEFAULTS_CIFAR10 = dict(
    epochs=90,
    warmup_epochs=10,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.95),
    alpha=5.0,
    dim_z=64,
    num_rb=2,
    arch='32',
    img_size=32,
    seed=42,
)

TRAIN_DEFAULTS_CELEBA = dict(
    epochs=90,
    warmup_epochs=10,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.95),
    alpha=5.0,
    dim_z=64,
    num_rb=6,
    arch='128',
    img_size=128,
    seed=42,
)

# Backward compat alias
TRAIN_DEFAULTS = TRAIN_DEFAULTS_CIFAR10
