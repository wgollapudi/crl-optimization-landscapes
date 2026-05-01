# Project Overview

This project studies whether theorem-backed identifiability assumptions change
the empirical loss landscape of causal representation learning models.

The experimental unit is a forest of independent training runs. Within each
regime, runs use the same dataset and hyperparameters but different random
seeds. This isolates optimizer and initialization variability under a fixed
empirical objective.

The pipeline is intentionally staged:

```text
datasets -> trained checkpoints -> landscape probe artifacts -> analysis tables
```

Each stage writes artifacts to disk so later stages can be rerun without
retraining.

## Regimes

Regime A is a plain VAE trained on observational data. It is the control.

Regime B is a Sparse VAE trained with engineered anchor features. It is the
statistically identifiable regime.

Regime C is a CausalDiscrepancy VAE trained on observational plus single-node
intervention environments.

Regime D uses the same CD-VAE estimator with richer intervention coverage. It is
the stronger causal-identifiability setting, and can also be run with anchor
features when we want statistical and causal constraints together.

## Main Measurement Levels

Local geometry is measured around one trained checkpoint using endpoint losses,
gradient norms, Hessian summaries, random slices, 2D slices, and perturbation
sharpness.

Pairwise geometry is measured within a regime using linear interpolation curves,
barriers, area excess, and parameter distance.

Distributional summaries are computed across the forest. The important object is
not one plot; it is the distribution of geometry descriptors induced by training
under each regime.

