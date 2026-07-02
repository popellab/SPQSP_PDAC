---
marp: true
theme: default
paginate: true
math: mathjax
size: 16:9
style: |
  section {
    font-size: 24px;
  }
  h1 {
    font-size: 38px;
    color: #1a5276;
  }
  h2 {
    font-size: 32px;
    color: #2c3e50;
  }
  section.title {
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.title h1 {
    font-size: 48px;
  }
  section.title h2 {
    font-size: 28px;
    color: #666;
    font-weight: normal;
  }
  section.two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto 1fr;
    gap: 0 30px;
  }
  section.two-col h1 {
    grid-column: 1 / -1;
  }
  table {
    font-size: 18px;
  }
  img {
    display: block;
    margin: 0 auto;
  }
  .small {
    font-size: 18px;
  }
  .footnote {
    font-size: 14px;
    color: #888;
    position: absolute;
    bottom: 30px;
  }
  blockquote {
    border-left: 4px solid #2980b9;
    padding-left: 16px;
    font-style: italic;
    color: #555;
  }
  .highlight {
    color: #c0392b;
    font-weight: bold;
  }
  .good {
    color: #27ae60;
    font-weight: bold;
  }
  .warn {
    color: #e67e22;
    font-weight: bold;
  }
---

<!-- _class: title -->

# Sensitivity Analysis & Restriction Estimation
## Dimensionality Reduction for a 177-Parameter QSP (Quantitative Systems Pharmacology) Model

Joel Eliason
Lab Meeting -- February 25, 2026

---

# The Problem

**177 uncertain parameters**, 31 calibration targets at diagnosis.

- Most parameters probably don't affect the observables we care about
- Downstream calibration via NPE (neural posterior estimation -- a method that learns the mapping from simulated observables to parameter distributions) degrades with unnecessary dimensions
- ~49% of simulations fail (tumor doesn't reach diagnosis size)

**Goal:** Identify which parameters matter, fix the rest, reduce dimensionality.

**Data:** 5,000 parameter sets sampled iid from priors → 2,852 valid simulations (43% NaN rate).

---

# Method 1: PRCC

**Partial Rank Correlation Coefficient** -- how much does $\theta_i$ correlate with observable $y_j$, *after removing the effect of all other parameters*?

1. **Rank-transform** $\theta_i$ and $y_j$ (robust to any monotonic relationship)
2. Compute the rank correlation matrix $\mathbf{C}$ of $(\theta_1, \ldots, \theta_p, y_j)$
3. Invert it: $\mathbf{P} = \mathbf{C}^{-1}$. The partial correlation is:

$$\text{PRCC}(\theta_i, y_j) = -\frac{P_{i,y}}{\sqrt{P_{ii} \cdot P_{yy}}}$$

Equivalent to the residual-regression approach, but a single matrix inversion gives all pairwise PRCCs simultaneously. Result $\in [-1, +1]$ with sign.

**Strength:** Captures linear, saturating, exponential -- anything monotonic.
**Limitation:** Misses threshold/switch effects and interactions ($\text{PRCC} \approx 0$ for non-monotonic relationships).

---

# Method 2: Random Forest Permutation Importance

**Intuition:** Train a random forest (RF) to predict $y_j$ from all 177 parameters. Scramble $\theta_i$. How much does prediction drop?

$$\text{Importance}_i = R^2_{\text{baseline}} - R^2_{\text{after scrambling}}$$

**Strength:** Captures *any* functional relationship -- thresholds, interactions, U-shapes. RF fits the targets well (median $R^2 \sim 0.91$).

**Limitation:** No sign information; importance can leak between correlated parameters.

**Why both?** PRCC catches monotonic effects with directionality. RF catches everything else. Where they disagree $\rightarrow$ nonlinear mechanisms (e.g., k_M2_pol: RF rank 3, PRCC rank 26).

---

# Significance & Combining Methods

**Permutation null test (1,000 perms):** Shuffle the $\theta$-$x$ correspondence, recompute mean $|\text{PRCC}|$ across targets $\rightarrow$ null distribution. Compare observed mean $|\text{PRCC}|$ to null $\rightarrow$ p-value per parameter. Benjamini-Hochberg false discovery rate (FDR) correction at 5%.

**Significance criterion:** A parameter is kept if its PRCC is significant (FDR < 0.05). RF importance is used for ranking only, not for the significance cutoff.

**Aggregate ranking:** For each parameter, take $\max|\text{PRCC}|$ and max RF importance across 31 observables. Rank by each, final rank = average.

| Metric | Value |
|---|---|
| Spearman rank correlation (PRCC vs RF) | **0.865** |
| Top-20 overlap | 16/20 (80%) |
| Top-30 overlap | 24/30 (80%) |

---

# SA Result: Aggregate Ranking

![h:480](../figures/surrogate_sensitivity/aggregate_ranking.png)

---

# SA Result: Cross-Method Agreement

![h:480](../figures/surrogate_sensitivity/cross_method_agreement.png)

---

# SA Result: 97/177 Parameters Significant

**2,852 valid simulations** (of 5,000). PRCC permutation null test (1,000 perms) + BH FDR correction:

**97 parameters significant** (FDR < 0.05) → fixed the other 80 at prior medians.

| Rank (PRCC) | Rank (RF) | Parameter | Module |
|:-:|:-:|---|---|
| 1 | 2 | k_PDGF_sec | Fibroblast (PDGF secretion) |
| 2 | 1 | k_myCAF_prolif | Fibroblast (myCAF proliferation) |
| 3 | 10 | k_C1_growth | Cancer (tumor growth rate) |
| 4 | 17 | k_ECM_myCAF_sec | Fibroblast (ECM secretion) |
| 5 | 4 | k_K_g | Cancer (vascular growth) |

Top 5 are all stroma + tumor intrinsic growth -- consistent with PDAC biology.

---

# SA Result: Parameters 6--13

| Rank (PRCC) | Rank (RF) | Parameter | Module |
|:-:|:-:|---|---|
| 6 | 8 | k_TGFb_sec_myCAF | Fibroblast (TGFb secretion) |
| 7 | 5 | q_nCD8_P_out | T cell (naive CD8 peripheral exit) |
| 8 | 12 | q_CD8_T_in | T cell (CD8 tumor infiltration) |
| 9 | 14 | q_nCD8_LN_in | T cell (naive CD8 LN entry) |
| 10 | 7 | N_IL2_CD4 | T cell (IL-2 CD4 expansion) |

Also in union of top-13: k_myCAF_to_iCAF (14/9), k_cell_clear (22/6), k_M2_pol (26/3).

**k_M2_pol** and **k_cell_clear** are the two parameters with biggest RF/PRCC discrepancy -- both are switch-like nonlinear mechanisms.

---

# PRCC Heatmap (Parameters x Observables)

![h:480](../figures/surrogate_sensitivity/prcc_heatmap.png)

---

# RF Importance Heatmap (Parameters x Observables)

![h:480](../figures/surrogate_sensitivity/rf_importance_heatmap.png)

---

# Biological Interpretation: Dominant Clusters

**The model is dominated by stroma and T cell trafficking:**

| Cluster | # in Top 13 | Key Parameters |
|---|:-:|---|
| **Fibroblast/stroma** | 5 | k_PDGF_sec, k_myCAF_prolif, k_ECM_myCAF_sec, k_TGFb_sec_myCAF, k_myCAF_to_iCAF |
| **CD8 trafficking** | 3 | q_nCD8_P_out, q_CD8_T_in, q_nCD8_LN_in |
| **Tumor intrinsic** | 2 | k_C1_growth, k_K_g |
| **Immune regulation** | 3 | N_IL2_CD4, k_cell_clear, k_M2_pol |

Dense desmoplastic stroma is the defining feature of PDAC. The full thymus → LN → tumor trafficking chain matters -- the bottleneck is *getting T cells there*, not killing efficiency.

---

# Biological Interpretation: CAF Plasticity Circuit

The complete **CAF interconversion circuit** is significant (~24 parameters total):

- All three interconversion pathways (iCAF $\leftrightarrow$ myCAF, PSC $\rightarrow$ iCAF/myCAF/apCAF)
- Proliferation and death for each subtype
- Key secreted factors (ECM, TGFb, CXCL12, IL-6, CCL5)

The model is sensitive to *which* CAF subtype dominates, not just total CAF numbers. This aligns with Biffi 2019 -- the iCAF/myCAF balance is TGFb-driven and determines whether the stroma is inflammatory (iCAF, IL-6) vs desmoplastic (myCAF, ECM/CXCL12).

**apCAF → Treg induction axis** also made the cut: k_apCAF_Treg (rank 42) and k_psc_activation_apCAF (rank 46). Elyada 2019's mechanism is detectable at diagnosis -- apCAF-driven Treg induction measurably contributes to the immunosuppressive TME even without treatment.

---

# Biological Interpretation: CD8 Trafficking Chain

Every step of the CD8 supply chain is individually significant:

```
  Thymus          Periphery         Lymph Node         Tumor
  ──────          ──────────         ──────────         ─────
                  q_nCD8_P_out  →   q_nCD8_LN_in  →   q_CD8_T_in
                  (rank 7/5)        (rank 9/14)        (rank 8/12)
                                         │
                                    k_T_activation     n_T_expansion
                                    _per_APC           _burst
                                    (rank 30/15)       (rank 16/33)
```

Meanwhile, **p_T_kill_per_contact** (killing efficiency per contact) is rank 171 -- *not significant*.

---

# Biological Interpretation: CXCL12 Exclusion

**CXCL12_50_Texcl** (rank 23) and **k_CXCL12_sec_myCAF** (rank 40) are both significant.

This is the Feig 2013 mechanism: myCAF-derived CXCL12 creates a chemokine barrier that excludes T cells from the tumor. The model's representation of the "immune desert" phenotype is validated.

**Therapeutic implication:** CXCR4 inhibitors (e.g., AMD3100/plerixafor) should shift model behavior by breaking the exclusion barrier.

**Contrast:** The CXCR3/chemotaxis module (IFNg → CXCL9 → CD8/NK recruitment) is entirely non-significant at baseline. Constitutive trafficking dominates IFNg-inducible recruitment at diagnosis. This may change under treatment when IFNg spikes.

---

# Biological Interpretation: Metabolism

9 metabolic parameters are significant, including:

| Parameter | What it controls |
|---|---|
| glc_50_C, k_glc_tumor, k_glc_immune | Glucose competition between tumor and immune cells |
| k_lac_Csec, k_lac_clear | Lactate accumulation and clearance |
| pO2_50, k_O2_demand | Hypoxia threshold and oxygen consumption |
| f_HIF_glycolysis | HIF-driven glycolytic switch |

Metabolic competition shapes the TME at diagnosis.

---

# Biological Interpretation: Other Notable Findings

**n_CD8_clones** (rank 25) is a proxy for TMB -- the immune response scales with neoantigen diversity, consistent with clinical TMB-response correlations.

**PD1_50** (rank 83) -- significant but low-ranked. Even without treatment, constitutive PD-L1 expression has a detectable effect on CD8 exhaustion. The checkpoint axis is wired correctly and will matter more under anti-PD-1.

**Notable absences:**
- p_T_kill_per_contact (rank 171) -- killing efficiency is irrelevant
- CXCR3/chemotaxis module -- all non-significant
- k_C1_death (rank 45/99) -- once tumor is viable, death rate barely matters

---

# Restriction Estimator: Motivation

**Problem:** ~49% of prior samples produce invalid simulations (tumor fails to grow to 3.2 cm within evolution window → NaN outputs). This is wasted compute.

**Idea:** What if we could learn which regions of the 177D prior space are "valid" and sample only from those regions?

This is the **RestrictionEstimator** approach (Deistler et al. 2022):
1. Train a binary classifier: $\theta \rightarrow \{\text{valid}, \text{invalid}\}$
2. Use the classifier as a rejection filter on the prior
3. Only propose parameter sets likely to produce valid simulations

If this works, we could cut simulation waste from ~49% to near zero.

---

# Restriction Estimator: Method

**Classifier architecture:**
- ResNet with 512 hidden features, 4 blocks (larger than default to handle 177D input)
- Binary output with softmax: $P(\text{valid} \mid \theta)$
- Trained with class-balanced loss weighting (valid/invalid classes are roughly equal)

**Training procedure:**
1. **Initial round:** 5,000 simulations from the prior, label each as valid/invalid
2. **Refinement (3 rounds):** Sample 1,000 new points from the current RestrictedPrior, simulate, append to training set, retrain
3. Refinement concentrates training data near the decision boundary

**RestrictedPrior:** Rejection sampling with a tunable "allowed false negative" rate -- controls how aggressively invalid samples are filtered.

---

# Restriction Estimator: How Rejection Sampling Works

The classifier produces $P(\text{valid} \mid \theta)$ for any proposed parameter set:

```
  Sample θ from prior
         │
         ▼
  Evaluate P(valid | θ)
         │
    ┌────┴────┐
    │ > thr?  │
    ├─────────┤
    │  Yes    │──── Accept θ, run simulation
    │  No     │──── Reject θ, sample again
    └─────────┘
```

**The threshold** is set by the `allowed_false_negatives` parameter:
- Lower afn $\rightarrow$ stricter filtering $\rightarrow$ fewer wasted sims, but risk rejecting valid ones
- Higher afn $\rightarrow$ more permissive $\rightarrow$ less filtering but safer

**Sweeping afn** shows whether meaningful filtering is achievable at any threshold.

---

# Restriction Estimator: Result -- Null Finding

**Classifier accuracy:** ~65% (barely above the 57% majority-class baseline).

| allowed_false_negatives | acceptance rate | effective rejection |
|---|---|---|
| 0.05 | ~0.03 | 97% |
| 0.10 | ~0.08 | 92% |
| 0.20 | ~0.18 | 82% |

**The dilemma:** At strict thresholds (afn=0.05), the classifier rejects 97% of samples -- including most valid ones. At permissive thresholds (afn=0.20), it barely filters anything useful.

---

# Restriction Estimator: Why It Failed

**Best single predictor:** k_C1_growth / k_C1_death ratio (classification AUC ~0.65). No parameter or low-dimensional combination cleanly separates valid from invalid.

**Root cause:** The valid/invalid boundary is **diffuse in 177D**. There is no sharp decision surface -- validity depends on subtle interactions among many parameters simultaneously.

```
  Sharp boundary (easy)          Diffuse boundary (our case)
  ┌────────────────────┐         ┌────────────────────┐
  │ ○○○○○○ │ ●●●●●●●  │         │ ○●○●○○●●○○●●●○●○○  │
  │ ○○○○○○ │ ●●●●●●●  │         │ ●○●○●●○○●●○●○●○●●  │
  │ ○○○○○○ │ ●●●●●●●  │         │ ○○●●○●●○●○○●○●●○○  │
  │ ○○○○○○ │ ●●●●●●●  │         │ ●●○○●○○●○●●○●○○●●  │
  └────────────────────┘         └────────────────────┘
  Classifier: easy               Classifier: barely above chance
```

---

# Restriction Estimator: What We Learned

**The null result is informative:**

1. **Tumor viability is not a simple criterion** -- it emerges from complex parameter interactions, not a clean low-dimensional manifold
2. **No shortcut to valid priors** -- we can't pre-filter; we have to simulate and discard
3. **49% NaN rate is inherent** to a mechanistic model with broad priors

**What we did instead:**
- Ran 5,000 simulations, dropped NaN rows, worked with 2,567 valid simulations
- The SA itself further reduced dimensionality (177 → 97), which helps indirectly


---

# Restriction Estimator: RF Feature Analysis

As a byproduct, the Random Forest trained for the restriction estimator provided a complementary view of **which parameters most predict simulation validity**:

| Parameter | Univariate AUC | Biological role |
|---|---|---|
| k_C1_growth / k_C1_death | 0.65 | Net tumor growth rate |
| k_K_g | 0.58 | Vascular carrying capacity |
| k_myCAF_prolif | 0.56 | Stromal barrier buildup |

All three relate to whether the tumor can physically reach 3.2 cm -- intrinsic growth and vascular support. But even the best predictor is weak ($\text{AUC} \approx 0.65$).

---

# Summary: SA + Restriction Estimator

| Stage | Method | Result |
|---|---|---|
| **Sensitivity analysis** | PRCC + RF importance (2,852 sims), PRCC permutation null test (1,000 perms), BH FDR correction | **97/177 significant** (FDR < 0.05). Fix 80 parameters at prior medians. |
| **Restriction estimator** | ResNet classifier (512 hidden, 4 blocks), 3 iterative rounds, rejection sampling | **Null result.** Boundary too diffuse in 177D. ~65% accuracy vs 57% baseline. |

**Key biological findings:**
- Stroma (CAF subtypes, ECM) and CD8 trafficking dominate
- Killing efficiency is irrelevant -- the bottleneck is T cell delivery
- Metabolism module has real impact (9 significant params)
- apCAF→Treg and CXCL12 exclusion mechanisms are detectable at diagnosis

---

# Implications for Calibration

**Dimensionality reduction: 177 → 97** makes NPE calibration feasible.

But 97 is still a lot. The SA results suggest further strategies:

1. **Reparameterize correlated pairs** (production/degradation → steady-state/turnover) to reduce effective dimensionality
2. **Staged calibration:** calibrate the 8 most identifiable parameters first, then expand
3. **Target-parameter matching:** PRCC heatmap tells us which observables constrain which parameters -- we can design targeted experiments
4. **Module-based decomposition:** The biological clusters (stroma, trafficking, metabolism) are relatively independent -- calibrate each block separately

---

<!-- _class: title -->

# Questions?

SA code: `workflows/sbi_workflow.py`
Restriction estimator: `workflows/restriction_estimator.py`
Biological interpretation: `notes/sensitivity_analysis_biological_interpretation.md`

---

# Backup: PRCC -- Formal Definition

For parameters $\theta = (\theta_1, \ldots, \theta_p)$ and observable $y$:

1. Rank-transform: $R(\theta_i)$ and $R(y)$
2. Fit linear models on ranks of all other parameters:

$$R(\theta_i) = \alpha_0 + \sum_{k \neq i} \alpha_k R(\theta_k) + \varepsilon_i$$
$$R(y) = \beta_0 + \sum_{k \neq i} \beta_k R(\theta_k) + \varepsilon_y$$

3. PRCC = Pearson correlation of residuals:

$$\text{PRCC}_i = \frac{\sum_j \varepsilon_{i,j} \cdot \varepsilon_{y,j}}{\sqrt{\sum_j \varepsilon_{i,j}^2 \cdot \sum_j \varepsilon_{y,j}^2}}$$

**Why residuals?** Removes confounding from correlated parameters. Without this step, $\theta_i$ might appear important only because it correlates with $\theta_k$, which actually drives $y$.

---

# Backup: RF Permutation Importance -- Formal Definition

For a trained Random Forest $f$ and test set $\{(\theta^{(j)}, y^{(j)})\}$:

1. Baseline score: $S_0 = R^2(y, f(\theta))$

2. For parameter $i$, create $\theta^{\pi_i}$ by randomly permuting column $i$ across samples

3. Permuted score: $S_i = R^2(y, f(\theta^{\pi_i}))$

4. Importance: $I_i = S_0 - S_i$

**Using OOB (out-of-bag) samples** avoids overfitting: each tree in the forest evaluates on the ~37% of training samples it didn't see during bootstrap.

**Advantage over impurity-based importance:** Impurity importance is biased toward high-cardinality features. Permutation importance is unbiased and works on held-out data.

---

# Backup: Benjamini-Hochberg FDR Correction

With 5,487 hypothesis tests (177 params × 31 observables), we need multiple testing correction.

**Bonferroni** (FWER control): adjusted p = raw p × 5,487. Far too conservative -- would miss real effects.

**Benjamini-Hochberg** (FDR control): controls the *expected fraction of false discoveries* among rejected hypotheses.

**Procedure:**
1. Sort all 5,487 p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. Find largest $k$ such that $p_{(k)} \leq \frac{k}{m} \alpha$
3. Reject all hypotheses with $p \leq p_{(k)}$

At FDR $= 0.05$: among the parameters we call "significant," we expect $\leq 5\%$ to be false positives. Much more powerful than Bonferroni while still controlling error.

---

# Backup: Restriction Estimator -- Deistler et al. 2022

**Paper:** "Truncated proposals for scalable and hassle-free simulation-based inference" (PLoS Computational Biology)

**Key idea:** In simulation-based inference (SBI), invalid simulations (NaN, crash, unphysical) waste compute. Rather than discarding them, use them as training signal.

**Algorithm:**
1. Sample $\theta \sim p(\theta)$, simulate $x = f(\theta)$
2. Label: valid if $x$ is finite, invalid otherwise
3. Train classifier $c(\theta) \rightarrow P(\text{valid} \mid \theta)$
4. Define RestrictedPrior: sample from prior, accept with probability proportional to $c(\theta)$
5. Iterate: sample from RestrictedPrior, simulate, retrain

**When it works:** Clear manifold of valid parameters (e.g., conservation laws, stability constraints). Works well when validity depends on a few key parameters.

**When it fails:** Diffuse boundary in high-D space with no clean separating surface -- our case.
