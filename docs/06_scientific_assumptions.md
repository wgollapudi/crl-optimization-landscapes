# Scientific Assumptions and Caveats

This project is an empirical study of theorem-facing regimes, not a proof that
the implemented estimators satisfy every theorem assumption exactly.

## Identifiability Regimes

Regime A is the non-identifiable baseline.

Regime B is designed to face the sparse-decoding statistical identifiability
assumptions by appending engineered anchor features.

Regime C and Regime D use intervention environments for causal identifiability.
The current CD-VAE implementation uses `env_id` and does not receive the true
`intervention_target` as training input.

## dSprites Caveat

The dSprites renderer is quantized and table-based. This is not a smooth
injective observation map in the strict mathematical sense. We use it as a close
experimental approximation because it gives controlled visual factors and a
simple renderer for repeated experiments.

## Graph Identifiability

Causal graph recovery should be interpreted up to the appropriate equivalence
class. Node ordering in the implementation is a convenient topological
parameterization, not a semantic commitment that labels are intrinsically
ordered.

## Comparing Regimes

Identifiable models add constraints and auxiliary losses. Therefore:

- total loss is not directly comparable across regimes
- component losses are more meaningful for cross-regime summaries
- interpolation is only done intra-regime
- Regime A is the control for cross-regime comparisons

The paper should phrase results as associations under this controlled pipeline,
for example:

```text
Under fixed datasets, optimizer settings, and deterministic landscape probes,
Regime D shows lower recon_img interpolation barriers than Regime A.
```

It should not claim that identifiability universally improves optimization.

