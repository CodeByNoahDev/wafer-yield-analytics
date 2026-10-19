# Walkthrough: how this project works and why it is built this way

This document explains the project from the ground up, so that anyone (including future me) can follow every decision.

---

## 0. The fab context in one paragraph

A 300 mm silicon wafer goes through hundreds of process steps (deposition, lithography, etch, implant, CMP, metallisation) over several weeks. At the end, a probe card touches every die and runs an electrical test. The result is a **wafer map**: a grid where each die is pass or fail. **Yield** is the fraction that pass. Every point of yield is worth a lot of money, so fabs employ yield engineers whose job is: (1) notice *systematic* failure patterns, (2) trace them to a tool or process step, and (3) keep tools stable with SPC so it doesn't happen again. This project is a small version of that toolkit.

---

## 1. Synthetic wafer maps (`generator.py`)

**Why simulate?** Real fab data is confidential. The public WM-811K dataset exists (and the code can load it), but a simulator lets me control the ground truth. I know exactly which pattern is on each wafer, which makes it possible to *test* the pipeline.

**How a wafer is made:**
1. Make a square grid and mark the dies inside a circle (`wafer_coords`). Each die gets a normalised radius `r` (0 at centre, 1 at edge) and an angle `theta`.
2. For the chosen pattern, compute a **probability of failure for every die**. For example, `edge_ring` uses a sigmoid that rises from 0 to 1 as `r` passes ~0.85, so dies near the edge are likely to fail.
3. Add **background random failures** (0–8 %) because real wafers always have some random particle defects.
4. Combine independent mechanisms correctly: `P(fail) = 1 - (1 - p_pattern) * (1 - p_background)`. A die survives only if it survives both.
5. Draw random numbers: each die fails if `rand < P(fail)`.

**Realism choices (these make the classifier's job harder on purpose):**
- Pattern strength varies (35–95 %), so some patterns are faint.
- Edge rings can be incomplete, which makes them look like a wide edge-loc.
- `loc` clusters can sit near the edge, which makes them look like edge-loc.
- Scratches are curved (quadratic Bézier curves) and can be broken into pieces.
- Wafer size varies (30–56 dies across), like different products.
- Classes are imbalanced (lots of `none`, few `near_full`).

My first version had none of these and the classifier scored 99.9 %, which is a sign the data was too easy, not that the model was great.

---

## 2. Features (`features.py`)

A machine-learning model needs a fixed-length vector of numbers. Wafer maps come in different sizes, so I turn each map into 66 numbers that describe *where* and *how* the fails are arranged.

**Design principle: rotation invariance.** A scratch is a scratch whether it runs north-south or east-west, so the features shouldn't change when the wafer is rotated. There is a unit test that rotates a wafer by 90° and checks the features are identical.

| Group | What | Why |
|---|---|---|
| `fail_rate` | fraction of failing dies | separates `none` / `random` / `near_full` |
| `radial_*` (6) | fail rate in rings from centre to edge | centre vs donut vs edge ring |
| `zone_*` (13) | fail rate in a centre zone, 4 middle sectors and 8 edge sectors, **sorted high-to-low** | sorting throws away *which* sector is bad (rotation) but keeps *that one sector is bad* (edge-loc vs edge-ring) |
| `radon_mean_*`, `radon_std_*` (40) | Radon transform statistics | see below |
| `cluster_*` (6) | shape of the biggest connected group of fails | scratches are long and thin, loc clusters are round |

**The Radon transform.** Imagine shining light through the wafer map from one direction and measuring how much "fail" each beam passes through. That gives one 1-D profile. Do it from 36 angles (every 10°) and you get a *sinogram*, the same maths as a CT scanner. Then:
- the **mean** profile across angles says where the fails are radially, and
- the **std** across angles says how *directional* they are. A ring looks the same from every angle (std ≈ 0); a scratch looks very different along its length than across it (std large).

The profiles are resampled to 20 points so every wafer size gives the same number of features.

**Cluster geometry.** `scipy.ndimage.label` finds connected groups of failing dies (8-connectivity, so diagonal neighbours count). For the biggest one I compute area, mean radius, how many angular sectors it spans, and elongation. Elongation comes from the eigenvalues of the covariance of the cluster's coordinates: one big and one small eigenvalue means a line.

---

## 3. Classification (`classify.py`)

- **Three models**: logistic regression (a linear baseline you always need, to show the fancy model is worth it), random forest and gradient boosting.
- **Stratified k-fold CV**: every fold keeps the class proportions, which matters with rare classes.
- **Macro-F1** as the metric: the average F1 of each class with equal weight. Accuracy would be misleading because predicting `none` for everything already gives 39 %.
- **`class_weight="balanced"`**: rare classes count more in training, so the model doesn't ignore them.
- **Hold-out test set** never used for model choice. The final number is reported on it.
- **Permutation importance**: shuffle one feature at a time on the test set and measure the drop in macro-F1. It is measured on unseen data and is less biased than the tree's built-in "impurity" importance.

**Interpreting the result:** the linear baseline (0.934) is only slightly behind the tree models (0.944–0.948), which tells me most of the work is done by good features rather than model complexity. The hardest class is `loc` (F1 0.73). The confusion matrix shows faint small clusters being called `none`, which is also hard for a human looking at the map.

---

## 4. Yield modelling (`yield_model.py`)

**Die yield models.** If killer defects land at density D0 (per cm²) and a die has area A, it catches on average A·D0 defects.
- If defects are perfectly random (Poisson), P(zero defects) = exp(−A·D0).
- In reality defect density varies between wafers and regions. If D itself follows a Gamma distribution with shape α, averaging exp(−A·D) over it gives the **negative binomial** model (1 + A·D0/α)^(−α). Small α means lots of variation; α → ∞ gives back Poisson.
- Murphy and Seeds are older models that assume particular density distributions (triangular and exponential).

**Monte Carlo check.** For each of 200 virtual wafers I draw a density from Gamma(α=2, mean 0.15), scatter that many defects uniformly on a 300 mm disk, lay a die grid over it (dies must be fully inside the 3 mm edge exclusion), and count dies with zero defects. Repeating for 14 die sizes gives yield vs area, and I fit each model with `scipy.optimize.curve_fit`.

**Result:** the negative binomial fit recovers D0 = 0.150 and α ≈ 2.2, matching the true values. Poisson underestimates D0 (0.130) because it is the wrong model for variable density. The good-dies-per-wafer plot shows why large dies are expensive, and why the industry moved to chiplets.

**Gross dies per wafer:** DPW ≈ πr²/A − πd/√(2A). The first term is wafer area / die area; the second subtracts the partial dies lost around the circular edge. A unit test checks it against a hand calculation.

---

## 5. Statistical Process Control (`spc.py`)

**Control chart basics.** Collect in-control data (Phase I), compute the centre line and σ, then flag new points (Phase II) that look unusual.

- **σ from the moving range**, not the standard deviation. The average of |xᵢ − xᵢ₋₁| divided by d₂ = 1.128 estimates the *short-term* noise. If the Phase I data contains a slow drift, the plain standard deviation would be inflated and the limits too wide.
- **Western Electric rules** catch patterns that a single 3σ limit misses: 2 of 3 points beyond 2σ, 4 of 5 beyond 1σ, 8 in a row on one side, and the Nelson trend rule (6 rising or falling). They are vectorised with cumulative sums so thousands of simulated series can be checked at once.
- **EWMA** (exponentially weighted moving average) smooths the data so a small, *sustained* shift builds up and crosses the limit. λ = 0.2 means each point gets 20 % weight.
- **Capability**: Cp = (USL − LSL)/6σ asks whether the process *could* fit in spec; Cpk = min(USL − μ, μ − LSL)/3σ asks whether it *does*, given where it is centred. Cp/Cpk use short-term σ and Pp/Ppk use overall σ.

**Average Run Length.** The fair way to compare charts is not one example plot but the average number of points before an alarm:
- **ARL₀** (no shift): how often the chart cries wolf. Higher is better. Rule 1 alone gives about 370, which is 1/0.0027 because 0.27 % of a normal distribution is beyond ±3σ.
- **ARL₁** (with a shift): how fast a real problem is caught. Lower is better.
My simulated values match the textbook (370 and ~44 for rule 1 at a 1σ shift), which validates the implementation. The key trade-off: adding run rules makes detection faster but *much* noisier (ARL₀ drops to 78), while EWMA gets the same speed with fewer false alarms.

---

## 6. Testing strategy (`tests/`)

The tests check *science*, not just that code runs:
- encoding and reproducibility of the generator;
- features are rotation-invariant; the Radon transform conserves mass for real-sized shapes and separates a line from a ring;
- yield models: Poisson ≤ Murphy ≤ Seeds, negative binomial → Poisson as α → ∞, DPW matches a hand calculation, fitted parameters recover the truth;
- SPC: rule-1 false-alarm rate ≈ 0.27 %, the moving-range σ estimate is unbiased, ARL matches theory, and the vectorised rules match on 1-D and 2-D input;
- an end-to-end smoke test of the whole pipeline.

Two of my first tests were themselves wrong (one compared two different random wafers, another expected bilinear interpolation to conserve the mass of a single pixel). Fixing them taught me the difference between a bug in the code and a bug in the expectation.

---

## 7. What I would do next

1. Run the classifier on real WM-811K maps and compare with published results.
2. Add a small CNN on the raw maps and compare it to the hand-crafted features.
3. Spatial clustering in the yield simulator (the `cluster_fraction` option) to study when even the negative binomial model breaks down.
4. Link the pieces: flag a tool with SPC, then show the wafer patterns from lots that went through it ("commonality analysis").
