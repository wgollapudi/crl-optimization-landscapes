How many of each type of model do we train?
How do we split the data to provide some randomness, or do we only need the random initalizations?

New experimental decomposition:

- Model A: baseline non-identifiable VAE
plain VAE, observational data only.
- Model B: statistically identifiable VAE
sparse/anchor-mask VAE or another identifiable representation-learning variant.
- Model C: interventional causal model without full statistical guarantees
e.g. CausalDiscrepancy VAE under softer or weaker assumptions. This is probably your “causal-ish” regime, but you should be explicit that it may only identify ancestors/transitive closure or depend on sparsity assumptions.
- Model D: strongest identifiability regime you can actually instantiate
likely an intervention-rich model with enough structure to support both latent identification and graph identification. But this may require a synthetic intervention design stronger than ordinary dSprites gives you for free.
