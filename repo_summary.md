# vqtorch Repo Summary

**vqtorch** is a PyTorch library for vector quantization, developed for the ICML 2023 paper *"Straightening Out the Straight-Through Estimator"* by Huh et al.

## Repo Structure

```
vqtorch/
├── setup.py                  # Package installation
├── README.md
├── assets/                   # Logo files
├── examples/
│   ├── classification.py     # MNIST VQ classifier (1000 iters). Trains a simple CNN with a VectorQuant layer for 10-class digit classification. Runs four progressive configs: baseline, +kmeans init, +sync update (sync_nu=1.0), +affine parameterization (affine_lr=10).
│   ├── autoencoder.py        # MNIST VQ autoencoder (1000 iters). Encoder-decoder CNN with a VectorQuant bottleneck, trained on L1 reconstruction loss. Same four progressive configs as classification.py but with sync_nu=2.0 and affine_lr=2.0.
│   ├── experimental_group_affine.py  # Same classifier architecture as classification.py but demonstrates the `affine_groups` parameter (set to 8). Runs only VectorQuant with kmeans init and group-affine reparameterization (300 iters).
│   ├── experimental_inplace_update.py  # Same classifier architecture as classification.py but demonstrates `inplace_optimizer` for alternating optimization. Uses an inner SGD optimizer (lr=50, momentum=0.9) to update codebook vectors in-place, bypassing the commitment loss (300 iters). Compares baseline+kmeans vs inplace alt update.
│   └── test.py               # Standalone smoke test for all three VQ layer types: VectorQuant, GroupVectorQuant (4 groups, unshared codebooks), and ResidualVectorQuant (4 groups, shared codebook). Creates each layer, runs a kmeans warmup pass, then a forward pass, and prints quantization error.
└── vqtorch/                  # Library source
    ├── __init__.py
    ├── dists.py              # Distance functions
    ├── math_fns.py           # Math utilities
    ├── norms.py              # Normalization layers
    ├── utils.py
    └── nn/
        ├── vq.py             # VectorQuant (core VQ layer)
        ├── vq_base.py        # Base class for VQ layers
        ├── gvq.py            # GroupVectorQuant (subvector quantization)
        ├── rvq.py            # ResidualVectorQuant
        ├── affine.py         # Affine reparameterization
        ├── pool.py           # MaxVecPool2d, SoftMaxVecPool2d
        └── utils/            # Codebook init (kmeans) & dead code replacement
```

The examples are toy demos on MNIST (1000 iterations each) showing how to use the library with progressively more features (baseline, kmeans init, sync update, affine parameterization). They are not the full training scripts used for the paper's Table 3 results.

## Table 3 — Paper Quotes and Setup

Table 3 ("Generative modeling reconstruction") compares VQVAE methods on CIFAR-10 and CelebA.

### Direct quotes from the paper (Huh et al., ICML 2023)

**Architecture (Appendix A.1):**
> "For generative modeling, we use backbone architecture from (Takida et al., 2022) with 64 channels."

This refers to the SQ-VAE ResNet encoder/decoder from `github.com/sony/sqvae`, specifically `vision/networks/net_32.py` for CIFAR-10. The architecture uses `num_rb=2` residual blocks, `dim_z=64` channels, and produces 8x8 spatial latent maps from 32x32 inputs.

**Dataset (Appendix A.1):**
> "For CIFAR10, we use an image size of 32x32, and for CelebA, we use an image size of 128x128."

**Loss (Appendix A.1):**
> "We use MSE for reconstruction loss, and we do not use any perceptual or discriminative loss."

**Codebook (Appendix A.1):**
> "We use 1024 codes for all our experiments."

**VQ hyperparameters (Appendix A.1):**
> "For VQ hyper-parameters, we use alpha=5 and beta=0.9~0.995 with mean-squared error for the commitment loss."

**Affine alpha (Appendix A.9):**
> "We reduce the weighting of the commitment loss to alpha=2."

**Tuning (Table 3 caption):**
> "The metrics are computed on the test set and hyper-parameters are tuned for each model."

**Metrics (Section 5.2):**
> "We compare the performance of our method against existing techniques [...] using MSE as well as LPIPS perceptual loss (Zhang et al., 2018)."

Table 3 columns: MSE (x10^-3), Perplexity, LPIPS (x10^-1). The paper does not specify which LPIPS backbone (alex/vgg).

**Training config (Table 8):**
| Config | Value |
|---|---|
| optimizer | AdamW |
| base learning rate | 1e-4 |
| weight decay | 1e-4 |
| optimizer momentum | (0.9, 0.95) |
| training epochs | 90 |
| learning rate scheduler | Cosine |
| warmup epochs | 10 |
| augmentations | None |

**Batch size:** Not specified anywhere in the paper for any experiment. The SQ-VAE reference code defaults to batch_size=32.

### What is NOT specified
- Batch size for generative experiments
- LPIPS backbone network (alex vs vgg)
- Exact beta value per method (only range 0.9~0.995)
- Train/val split (SQ-VAE code uses 40k train / 10k val from the 50k CIFAR-10 training set)
