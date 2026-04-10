# Results Summary

Comparison against Table 3 of "Straightening Out the Straight-Through Estimator" (Huh et al., ICML 2023).

## CIFAR-10 (32x32)

| Run | Config | Beta | BS | MSE (x10⁻³) | Perplexity | Active % | LPIPS-vgg (x10⁻¹) | LPIPS-alex (x10⁻¹) |
|---|---|---|---|---|---|---|---|---|
| baseline | baseline | 0.9 | 256 | 10.45 | 33.1 | 5.5 | 4.55 | 2.11 |
| baseline_bs32 | baseline | 0.9 | 32 | 9.12 | 6.5 | 0.8 | 4.11 | 1.80 |
| baseline_bs32_b095 | baseline | 0.95 | 32 | 7.41 | 11.9 | 1.4 | 3.71 | 1.45 |
| baseline_bs32_b099 | baseline | 0.99 | 32 | 6.23 | 22.1 | 2.9 | 3.30 | 1.13 |
| baseline_bs32_b0995 | baseline | 0.995 | 32 | 5.56 | 36.7 | 5.0 | 3.05 | 0.93 |
| best | best | 1.0 | 256 | 3.67 | 998.6 | 100.0 | 2.08 | 0.35 |
| best_bs32 | best | 1.0 | 32 | 3.56 | 991.6 | 100.0 | 1.97 | 0.29 |
| **Paper: VQVAE baseline** | — | — | — | **5.65** | **14.0** | — | **5.43** | — |
| **Paper: VQVAE+Affine+OPT+replace+l2** | — | — | — | **1.74** | **608.6** | — | **2.27** | — |

## CelebA (128x128)

| Run | Config | Beta | BS | MSE (x10⁻³) | Perplexity | Active % | LPIPS-vgg (x10⁻¹) | LPIPS-alex (x10⁻¹) |
|---|---|---|---|---|---|---|---|---|
| celeba_baseline | baseline | 0.9 | 32 | 1.51 | 14.5 | 2.3 | 1.91 | 1.26 |
| celeba_baseline_b095 | baseline | 0.95 | 32 | 1.16 | 17.8 | 2.9 | 1.67 | 1.09 |
| celeba_baseline_b099 | baseline | 0.99 | 32 | 0.89 | 34.8 | 4.9 | 1.41 | 0.89 |
| celeba_baseline_b0995 | baseline | 0.995 | 32 | 0.85 | 44.4 | 6.3 | 1.38 | 0.85 |
| celeba_baseline_bs128 | baseline | 0.9 | 128 | 2.06 | 12.3 | 2.2 | 2.27 | 1.56 |
| celeba_baseline_bs128_b095 | baseline | 0.95 | 128 | 1.40 | 15.2 | 3.9 | 1.91 | 1.28 |
| celeba_baseline_bs128_b099 | baseline | 0.99 | 128 | 1.06 | 32.4 | 5.4 | 1.57 | 1.01 |
| celeba_baseline_bs128_b0995 | baseline | 0.995 | 128 | 0.95 | 37.5 | 6.2 | 1.47 | 0.93 |
| celeba_best | best | 1.0 | 32 | 0.56 | 975.4 | 100.0 | 0.90 | 0.42 |
| celeba_best_bs128 | best | 1.0 | 128 | 0.61 | 977.2 | 100.0 | 0.95 | 0.48 |
| **Paper: VQVAE baseline** | — | — | — | **10.02** | **16.2** | — | **2.71** | — |
| **Paper: VQVAE+Affine+OPT+replace+l2** | — | — | — | **4.42** | **872.6** | — | **1.36** | — |

## Findings

### Beta is the most impactful hyperparameter for the baseline

Across both datasets, increasing beta from 0.9 to 0.995 consistently and substantially improves all metrics. Higher beta shifts the commitment loss to weight the codebook update more heavily, which helps the baseline converge to lower reconstruction error despite using fewer active codes overall.

**CIFAR-10 baseline (BS=32):** Sweeping beta from 0.9 to 0.995 reduced MSE from 9.12 to 5.56 (x10^-3) and LPIPS-vgg from 4.11 to 3.05 (x10^-1). At beta=0.995, MSE (5.56) closely matches the paper's reported 5.65, though perplexity (36.7 vs 14.0) and LPIPS-vgg (3.05 vs 5.43) diverge.

**CelebA baseline (BS=32):** The same trend holds — beta=0.995 gives the best MSE (0.85) and LPIPS-vgg (1.38).

### Batch size affects performance but not dramatically

Larger batch sizes (fewer optimization steps per epoch) consistently produce slightly worse results:

- **CIFAR-10:** BS=256 (MSE 10.45) vs BS=32 (MSE 9.12) at the same beta — a ~15% gap.
- **CelebA baseline:** BS=128 is ~20-35% worse in MSE than BS=32 at matched beta values (e.g., beta=0.9: MSE 2.06 vs 1.51).
- **CelebA best:** BS=128 (MSE 0.61) vs BS=32 (MSE 0.56) — only a ~9% gap. The best config with affine reparameterization and inplace codebook optimization is more robust to batch size.

### The "best" config (Affine+OPT+replace+l2) dramatically outperforms the baseline

On both datasets, the best config achieves 100% codebook utilization (vs 2-6% for the baseline) and far lower reconstruction error:

- **CIFAR-10:** Best MSE 3.56 vs baseline-best MSE 5.56 (1.6x improvement).
- **CelebA:** Best MSE 0.56 vs baseline-best MSE 0.85 (1.5x improvement), with perplexity near the maximum (975 out of 1024).

### LPIPS backbone matters: alex vs vgg

The paper reports LPIPS without specifying the backbone. Our results show LPIPS-alex values are consistently 1.5-3x lower than LPIPS-vgg. The paper's reported LPIPS values (e.g., 5.43 for CIFAR-10 baseline) are much more consistent with LPIPS-vgg than LPIPS-alex, suggesting the paper likely used the VGG backbone.

### CelebA MSE discrepancy with the paper

Our CelebA MSE values (0.56-2.06 x10^-3) are 5-18x lower than the paper's (4.42-10.02 x10^-3). This gap is too large to be explained by hyperparameter tuning and may indicate a difference in how MSE is computed or reported. The LPIPS-vgg values, by contrast, are in a comparable range (0.90-2.27 ours vs 1.36-2.71 paper). The architecture, training procedure, and codebook configuration all match the paper's description, so the MSE discrepancy remains unexplained. One possibility is that the paper computes MSE with a different reduction (e.g., sum over channels rather than mean, or evaluated at a different resolution).

### Epoch timing

All CelebA runs took approximately 352s/epoch (~5.9 min) when running 3 jobs per GPU on H200s. BS=128 and BS=32 showed no meaningful timing difference in this shared-GPU setting; the GPU compute was saturated by running multiple jobs concurrently. Running a single job per GPU would be needed to measure the true batch size speedup.
