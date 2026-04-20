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
Regime B: masked / sparse identifiable VAE (see Moran et al. 2022), with engineered anchor featrues added to the model. Append 8 scalar anchor channels, a_(1, 1), a(1, 2) depend on scale, a_(2,1),... depend on orientation and so on.
Regime C: CausalDiscrepancy VAE (see Zheng et al. 2023)
Regime D: same CausalDiscrepancy-VAE, just trained on the stronger regime

First we need to write
- one shared encoder / decodeder
- shared training harness
- shared latent parameterization
etc.

Start with observational baseline.

Share input pipeline / encoder trunk / image decoder / training harness
Vary decoder parameterization and loss terms

decoder parameterization
a: direct decoder
b: sparse featrue-selection layer on top of shared decoder outputs or derfore per-featrue decoding
c/d: SCM layer + mixing decoder

Loss
a: ELBO
b: ELBO
c/d: ELBO + MMD + graph sparisty + intervention-target machninery
