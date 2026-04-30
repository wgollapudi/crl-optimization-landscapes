Concens rn:
When we checkpoint SparseVAE are we saving both decoders?
Saving the right checkpoints
Finish evaluation infra (oh lawd)
Make sure CD-VAE is good
Testing


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

We implement a sparse-decoding VAE with learned continuous feature masks and known hard anchor features, following the sparse DGM parameterization of Moran et al. The implementation uses a continuous relaxation of the sparse mask rather than the full spike-and-slab variational EM estimator.

# Mesurment plan
To make this nonsense scientifically useful, we are going to try and turn "loss landscape analysis" into a fixed battery of measurements at three levels.
- local geometry around one trained solution
- pairwise geometry between two trained solutions
- distributional summaries across the whole forest
mirrors the literatrue - Li et al. focus on normalized slices and curvature comparisons, Frankle et al. on interpolation barriers / connectivity, Brea et al. on symmetry-induced flat structure, and Baldassi et al. on neighborhood volume / wide minima rather than pointwise loss alone.

For each seed in each regime, store one landscape record with
- final train / val loss
- parameter norm
- gradient norm at convergence
- top Hessian eigenvalue
- Hessian trace estimate
- 1D normalized slice through the solution
- 2D normalized slice through the solution
- local sharpness under small pertubations
This is the “single minimum” view. It tells you whether a regime tends to produce sharper, flatter, or more anisotropic minima. Li et al. use normalized slices to make cross-model comparisons meaningful, and a large part of the flat-minima literature treats curvature and perturbation sensitivity as basic local descriptors.

Local curvature

For a trained model theta, estimate:
- $lambda_("max") (H_theta)$ top Hessian eigenvalue
- $tr(H_theta)$: Hessian trace
- maybe $norm(H_theta)_F$ approximetely, if feasible

Interpretation
- larget $lambda_("max")$ means sharper dominant curvatrue
- larger trace means more total curvatrue
- large trace with moderate top eigenvalue can indicate many modestly curved directions rather than one knife-edge direction.

1D and 2D normalized slices
For each solution, choose direction





To make this scientifically useful is to turn “loss landscape analysis” into a **fixed battery of measurements** at three levels:

* **local geometry around one trained solution**
* **pairwise geometry between two trained solutions**
* **distributional summaries across the whole forest**

That mirrors the literature well: Li et al. focus on normalized slices and curvature comparisons, Frankle et al. on interpolation barriers / connectivity, Brea et al. on symmetry-induced flat structure, and Baldassi et al. on neighborhood volume / wide minima rather than pointwise loss alone. ([arXiv][1])

## 1. What you should measure for each trained model

For each seed in each regime, store one **landscape record** with:

* final train / val loss
* parameter norm
* gradient norm at convergence
* top Hessian eigenvalue
* Hessian trace estimate
* 1D normalized slice through the solution
* 2D normalized slice through the solution
* local sharpness under small perturbations

This is the “single minimum” view. It tells you whether a regime tends to produce sharper, flatter, or more anisotropic minima. Li et al. use normalized slices to make cross-model comparisons meaningful, and a large part of the flat-minima literature treats curvature and perturbation sensitivity as basic local descriptors. ([arXiv][1])

A good operationalization is:

### Local curvature

For a trained model ( \theta ), estimate:

* ( \lambda_{\max}(H_\theta) ): top Hessian eigenvalue
* ( \mathrm{tr}(H_\theta) ): Hessian trace estimate
* maybe ( |H_\theta|_F ) approximately, if feasible

Interpretation:

* larger ( \lambda_{\max} ) means sharper dominant curvature
* larger trace means more total curvature
* large trace with moderate top eigenvalue can indicate many modestly curved directions rather than one knife-edge direction

Brea et al. explicitly connect flat directions to symmetry-generated valleys, and many later landscape papers use Hessian summaries as the standard local geometry descriptors. ([arXiv][2])

### 1D and 2D normalized slices

For each solution, choose directions:

* one random direction
* one data-driven direction such as the vector to another seed’s solution

Then evaluate loss on:
[
\theta + \alpha d
]
and in 2D:
[
\theta + \alpha d_1 + \beta d_2
]
using **filter normalization** or an analogous block-wise normalization so comparisons across regimes remain meaningful. This is straight from Li et al. and should be one of your core tools. ([arXiv][1])

### Local perturbation sharpness

Perturb parameters in a small Euclidean ball and record:

* worst-case loss increase over sampled perturbations
* average loss increase over sampled perturbations

This is a crude practical proxy for the “wide basin / local entropy” idea in Baldassi-style work. It is not the full robust-ensemble formalism, but it gets at the same object: how much low-loss volume surrounds the solution. ([PNAS][3])

## 2. What you should measure for each pair of seeds

This is the second major level. For each pair of independently trained models in the same regime, measure:

* endpoint distance ( |\theta_1-\theta_2| )
* linear interpolation barrier
* optionally mode-connectivity path barrier
* loss along the segment between them
* representation similarity if you later want that as a secondary analysis

The key quantity is the **linear interpolation barrier**:
[
\max_{\alpha \in [0,1]} L((1-\alpha)\theta_1+\alpha\theta_2)

* \frac{L(\theta_1)+L(\theta_2)}{2}.
  ]

Frankle et al. use essentially this idea to study linear mode connectivity and stability to SGD noise. Singh et al. treat barrier height as a central topographic descriptor and use it to characterize “mountainsides” and “ridges.” ([arXiv][4])

This gives you regime-level questions like:

* does Regime A produce many disconnected-looking minima?
* does Regime D produce lower barriers between seeds?
* do identifiable regimes reduce or increase basin multiplicity?

You do **not** need to answer these by inspection of one plot; you answer them by the **distribution of pairwise barriers** across the forest.

## 3. What you should aggregate across the forest

This is where the science happens.

For each regime, after training (n) seeds, compute distributions of:

* final losses
* top Hessian eigenvalues
* Hessian traces
* local perturbation sharpness
* pairwise interpolation barriers
* pairwise parameter distances

Then compare those distributions across regimes.

This is important because one pretty slice can be misleading. The real object is:

> the empirical distribution of local and pairwise geometric statistics induced by SGD under each theorem-backed regime.

That gives you a clean inferential target.

I would report, for each metric:

* mean
* median
* standard deviation
* 25/75 percentiles
* maybe bootstrap confidence intervals for regime differences

Not because you need fancy statistics, but because you want to distinguish:

* “the regime really shifts the geometry”
  from
* “one seed happened to look weird.”

## 4. A concrete measurement plan

I would break the study into three phases.

### Phase I: local geometry

Train a forest for each regime, maybe 20 seeds to start. For each trained model:

* final loss
* gradient norm
* top Hessian eigenvalue
* Hessian trace estimate
* 1D slice along a random normalized direction
* 1D slice toward one reference model
* local perturbation sharpness

Goal:

* understand whether regimes systematically differ in curvature and local basin width

This is the fastest phase and probably should come first.

### Phase II: pairwise topology

Within each regime, sample a set of seed pairs and compute:

* linear interpolation barrier
* loss profile along the straight line
* maybe a piecewise-linear or curved low-loss connection later

Goal:

* understand whether solutions lie in the same connected low-loss region or in separated basins

This is the Frankle/Singh-style phase. ([arXiv][4])

### Phase III: representative visualization

Only after you have summary statistics, produce the paper-style figures:

* one or two representative 2D slices per regime
* one interpolation plot per regime
* one boxplot / violin plot per metric across regimes

The representative visualizations should be chosen to illustrate the summaries, not replace them.

## 5. How to process the raw landscape data

The main mistake to avoid is collecting giant arrays of slice values and then not having a statistical object.

You want each computation to reduce to a **small number of comparable scalar descriptors**.

For example:

### For a 1D slice

Do not just save the curve. Also compute:

* minimum value on slice
* maximum value on slice
* width of region within (\varepsilon) of the minimum
* symmetry/asymmetry around the center
* average second finite difference near the center

These turn a curve into analyzable numbers.

### For a 2D slice

Do not just save the heatmap. Also compute:

* area of sublevel set ( {(\alpha,\beta): L \le L(\theta)+\varepsilon} )
* eccentricity of the low-loss region
* whether there are multiple visible low-loss pockets in the window

This gives you “valley width” and anisotropy in a comparable way. Li et al.’s visualization framework is useful here because it makes the slices themselves meaningful enough to summarize. ([arXiv][1])

### For interpolation

Compute:

* barrier height
* integral of excess loss along the path
* number of local peaks above threshold

Barrier height is the main number; the integral is a useful backup because two paths can have the same max barrier but very different overall roughness. Frankle et al. center their analysis on barrier-style quantities. ([arXiv][4])

### For perturbation sharpness

Compute:

* mean loss increase
* 90th percentile loss increase
* max loss increase over sampled perturbations

That gives you a width/robustness profile rather than one number.

## 6. What conclusions are actually justified

This is the most important part.

You should not conclude “identifiability causes X” just because one regime has prettier plots. You can conclude one of the following much more precisely.

### If identifiable regimes show lower Hessian top eigenvalues, larger sublevel-set area, and smaller perturbation sensitivity

Then you can say:

> stronger identifiability assumptions are associated with locally broader / less sharp minima under this optimization setup.

That is a geometry claim, not yet a generalization or optimization-theory claim.

### If identifiable regimes show smaller pairwise interpolation barriers

Then you can say:

> independently trained solutions under stronger identifiability appear to lie in more connected low-loss regions.

That is a connectivity claim, in the Frankle/Singh sense. ([arXiv][4])

### If identifiable regimes show larger barriers but also less variance across seeds

Then you can say:

> identifiability may isolate solutions into fewer, more stable basins rather than creating a giant connected valley.

That would actually be very interesting. You should not assume “identifiable” means “more connected.” It might instead mean “fewer equivalent broad plateaus.”

### If only the symmetry-heavy non-identifiable regime shows many flat directions

Then you can say:

> part of the observed flatness is consistent with degeneracy/symmetry rather than generic optimization ease.

That is where the Brea-style interpretation helps. ([arXiv][2])

## 7. What not to overinterpret

A few guardrails:

* **Low training loss alone tells you almost nothing** about geometry. Zhang et al.-style generalization skepticism still applies. A regime can fit perfectly and still have very different landscape structure. ([arXiv][1])
* **One 2D slice is not the landscape.** It is only a probe of it.
* **Parameter-space flatness is scale-sensitive.** That is why normalized directions matter. Li et al.’s filter normalization exists precisely to make these comparisons less meaningless. ([arXiv][1])
* **Mode connectivity does not prove convexity.** It only shows low-loss paths exist between some solutions.

## 8. The exact objects I would save to disk

For each trained seed:

* checkpoint
* final metrics
* Hessian summary
* perturbation-sharpness summary
* selected 1D slice arrays
* selected 2D slice grid
* metadata: regime, seed, model class, training loss, val loss

For each seed pair:

* endpoints
* endpoint losses
* pairwise distance
* interpolation curve
* barrier summary

Then create one analysis table per regime with one row per seed and one row per pair.

That gives you a very natural downstream analysis structure:

* **seed table** for local geometry
* **pair table** for connectivity/topology

## 9. My recommended first-pass battery

If you want the shortest path to real results, do this first:

For each regime:

* train 20 seeds
* for each seed: top Hessian eigenvalue, trace estimate, local perturbation sharpness
* for 20 random seed pairs: linear interpolation barrier
* for 3 representative seeds: 2D normalized slice

Then compare regime distributions with boxplots and bootstrap confidence intervals.

That is enough to answer:

* are minima sharper or flatter?
* are basins more or less connected?
* is seed-to-seed variability higher or lower?

And it is very well aligned with the literature you pulled. ([arXiv][1])

The next step is to decide the exact scalar definitions and numerical estimators for:

* top Hessian eigenvalue,
* trace,
* perturbation sharpness,
* 1D/2D slice normalization,
* interpolation barrier.

[1]: https://arxiv.org/abs/1712.09913?utm_source=chatgpt.com "Visualizing the Loss Landscape of Neural Nets"
[2]: https://arxiv.org/abs/1907.02911?utm_source=chatgpt.com "Weight-space symmetry in deep networks gives rise to permutation saddles, connected by equal-loss valleys across the loss landscape"
[3]: https://www.pnas.org/doi/10.1073/pnas.1608103113?utm_source=chatgpt.com "Unreasonable effectiveness of learning neural networks"
[4]: https://arxiv.org/pdf/1912.05671?utm_source=chatgpt.com "arXiv:1912.05671v4 [cs.LG] 18 Jul 2020"


Yes.

## 2. Why VAE loss evaluation is noisy

A VAE does not usually compute reconstruction from exactly one deterministic latent. It samples:

[
z = \mu_\phi(x) + \sigma_\phi(x)\epsilon,\qquad \epsilon \sim \mathcal N(0,I).
]

So even with the **same model weights** (\theta), same input batch, and same loss function, the loss can change slightly every time because a new (\epsilon) is sampled.

That is bad for landscape analysis. If you evaluate:

[
L(\theta + \alpha d)
]

along a slice, some wiggles in the curve may be real landscape geometry, but some may just be sampling noise.

You have three options:

**Best default: deterministic posterior mean.**
Use (z=\mu_\phi(x)) during landscape evaluation. This removes sampling noise and makes “loss at (\theta)” a real deterministic quantity.

**Alternative: fixed noise.**
Pre-sample (\epsilon) once per batch and reuse it for every point in the slice/interpolation. This preserves the stochastic VAE objective more faithfully, but is more annoying to implement.

**Expensive alternative: Monte Carlo average.**
Evaluate each point multiple times with different (\epsilon) and average. More faithful, slower.

I’d use **posterior mean** for the primary landscape probes, and maybe fixed-noise/MC average as a robustness check later.

## 4. Why direction normalization matters

Suppose you plot loss along:

[
\theta + \alpha d
]

where (d) is a random direction.

If one layer has large weights and another has tiny weights, a naive random direction may perturb the tiny layer much more aggressively relative to its scale. Then your plot reflects arbitrary parameter scaling, not meaningful geometry.

This is especially dangerous because neural networks have scale symmetries. For example, in a ReLU network, you can often multiply one layer by (c) and the next by (1/c) without changing the function much. The raw parameter coordinates changed, but the function did not.

So direction normalization tries to make perturbations comparable to the scale of each layer.

The simplest useful version is **layer-wise normalization**:

For each parameter tensor (W_\ell), sample random noise (D_\ell), then rescale it so:

[
|D_\ell| = |W_\ell|.
]

Then the perturbation direction changes each layer by a comparable relative amount.

So for a slice:

[
\theta(\alpha)=\theta+\alpha d,
]

(\alpha = 0.01) roughly means “move each layer by about 1% of its norm,” not “move in some arbitrary global direction.”

For our first implementation, I would use:

* layer-wise normalization for weights,
* either include biases with their own norm or set bias directions to zero,
* same rule across all regimes.

That gives cleaner comparisons than raw random directions.

