Technically the `overlap_support` dataset is not generated from the same master SCM. Hence, it does not follow the same latent distribution. We should fix this eventually, how, I have no idea.

New experimental decomposition:
- Model A: baseline non-identifiable VAE
plain VAE, observational data only.
- Model B: statistically identifiable VAE
sparse/anchor-mask VAE or another identifiable representation-learning variant.
- Model C: interventional causal model without full statistical guarantees
e.g. CausalDiscrepancy VAE under softer or weaker assumptions. This is probably your “causal-ish” regime, but you should be explicit that it may only identify ancestors/transitive closure or depend on sparsity assumptions.
- Model D: strongest identifiability regime you can actually instantiate
likely an intervention-rich model with enough structure to support both latent identification and graph identification. But this may require a synthetic intervention design stronger than ordinary dSprites gives you for free.

Estimators (still a work in progress), work off of Moran et al.
Regime A: plain VAE
Regime B: masked / sparse identifiable VAE
Regime C: CausalDiscrepancy VAE
Regime D: same CausalDiscrepancy-VAE, just trained on the stronger regime

First we need to write
- one shared encoder / decodeder
- shared training harness
- shared latent parameterization
etc.

Start with observational baseline.


