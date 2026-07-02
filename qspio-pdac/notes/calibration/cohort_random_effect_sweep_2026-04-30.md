# Cohort random-effect τ sweep (2026-04-30)

**Branch:** `feature/cohort-random-effect`  •  **Pool:** `multi_scenario_v5_qual_migration_28d85751` (5k baseline-only, with-submodel-sync)
**Logs:** `/tmp/sbi_run_cohort_re_tau{00,03,05,07,10}_ppc_*.log`

## Setup

5-point sweep of `--tau-cohort ∈ {0.0, 0.3, 0.5, 0.7, 1.0}` on the baseline_no_treatment scenario, all with `--run-posterior-predictive` so we get posterior pairwise tension (the table that responds to cohort-RE — prior-PC pairwise φ does NOT, since it reads raw simulator outputs).

Each run reuses the same 5k-sim pool (cohort shift is post-load). Total cost was the τ=0.0 pool generation (~30 min on Rockfish) plus 5× NPE training + 500 PPC sims (~2 min each).

## Headline metrics

| τ | D²_prior | k_eff (of 22) | D²_post / D²_prior | median joint reach |
|---|---|---|---|---|
| 0.0 | 83.9 | 10.1 | 1.650 | 15/22 |
| 0.3 | 65.3 | 10.8 | 1.715 | 15/22 |
| 0.5 | 54.9 | 11.4 | 2.093 | 14/22 |
| 0.7 | 46.8 | **11.6** | 2.367 | 14/22 |
| 1.0 | 38.6 | 11.1 | 2.744 | 13/22 |

- **D²_prior drops monotonically** (-54% from τ=0 to τ=1.0). Cohort-RE absorbs apparent misspec into nuisance noise, so the Mahalanobis distance from x_obs to the noise-augmented prior predictive shrinks.
- **D²_post / D²_prior rises monotonically.** The residual after the posterior pulls is increasingly *cleanly* structural — the cohort component has been peeled off.
- **k_eff peaks at τ=0.7** then drops at τ=1.0 — beyond ~0.7 the posterior is too diffuse and effective dimensionality declines.
- **Joint reach declines slightly** (15 → 13 of 22). Expected: at τ=0 the posterior was over-tight, hitting CIs by overconfidence; cohort-RE widens the posterior honestly.

## Top posterior pairwise tension (PPC)

| τ | top φ | top pair | second φ | second pair |
|---|---|---|---|---|
| 0.0 | **−0.432** | `cd8_treg × total_til` (Cluster A) | −0.271 | `cdc1 × th_exh` (Cluster B) |
| 0.3 | −0.453 | `cd8_treg × total_til` (A) | −0.279 | `cdc1 × th_exh` (B) |
| 0.5 | **−0.361** | `cd8_treg × total_til` (A) — minimum | −0.331 | `cdc1 × th_exh` (B) |
| 0.7 | −0.417 | `cd8_treg × total_til` (A) | −0.387 | `cdc1 × th_exh` (B) |
| 1.0 | −0.483 | **`cdc1 × th_exh` (B)** — swap | −0.407 | `cd8_treg × total_til` (A) |

### Two key reads

1. **Cluster A's top-pair φ is non-monotonic** with a minimum at τ=0.5. Mechanism: φ depends on *both* marginal P(reach target_i) AND joint P(reach both). As τ rises, marginals shrink (posterior wider) faster than joint at high τ — so |φ| algebraically rebounds even though absolute joint reach is still dropping (10.9% → 3.5% from τ=0 to τ=1.0). The *honest* signal at high τ is the joint reach; the φ statistic is most discriminating around τ=0.5.

2. **Diagnostic rotation at τ=1.0.** Cluster B (`cdc1_density × th_exhausted_fraction`) overtakes Cluster A as the top tension — at τ=0.0 their ratio was 0.432/0.271 ≈ 1.6× in Cluster A's favor; at τ=1.0 it's 0.483/0.407 ≈ 1.2× in **Cluster B's** favor. As cohort heterogeneity is absorbed, the structural signal rises in relative importance. This is the cleanest diagnostic separation we have for cohort-driven vs structural misspecification.

## Posterior misfit-driver influence (LOO decomposition)

Influence of each target on D²_post; selected drivers across the sweep:

| Target | τ=0 | τ=0.5 | τ=1.0 | reading |
|---|---|---|---|---|
| `treg_fraction_cd4_hiraoka2006` | 51.5 | 38.4 | (lower) | Cluster A — drops with τ |
| `treg_fraction_cd4` (Liudahl) | 50.4 | 38.0 | (lower) | Cluster A — drops with τ |
| `cdc1_density_baseline` | 38.1 | 24.0 | rising in φ | Cluster B — drops in raw influence but rises in *relative* tension |
| `cd8_treg_ratio_golesworthy2022` | 29.3 | 15.5 | (lower) | Cluster A — drops 47% |
| `cd8_density_baseline` | 18.7 | (out of top-5) | — | Jansen δ_c absorbed it |
| `th_exhausted_fraction` | 15.7 | 16.5 | (rising) | Structural (Cluster B/C) |

The Treg/TIL/CD8 cohort-driven misfits all drop substantially with τ; Th-exhaustion (Cluster B/C) holds steady or rises — same story as the φ table.

## τ recommendation

| τ | Use case |
|---|---|
| **0.5** | **Default for inference.** Minimum top-pair |φ|, k_eff at 11.4, posterior usefully wide, diagnostic preserved. The natural log-space gap between Golesworthy CD8/Treg=3.69 and Liudahl-implied 10.0 is ~1.0; τ=0.5 in copula-Gaussian space puts that at the ~95th percentile. |
| **0.7–1.0** | **Diagnostic mode** — best for *separating* structural from cohort, but posterior too diffuse for trustworthy θ inference. Use to rank structural-fix priorities. At τ=1.0 the rank order of pairwise tensions reflects model structure with cohort heterogeneity peeled off. |
| 0.3 | Too tight — barely moves Cluster A's φ. |

The CLI default in `RunConfig` is `tau_cohort=0.0` (off, no behavior change vs historical). Pass `--tau-cohort 0.5` for inference runs that should account for cohort heterogeneity in the calibration target set.

## Pool reuse property (worth knowing)

Cohort shift is applied to `x_train_t` post-copula transform, downstream of any simulator interaction. Consequently:

- **Same pool serves any τ.** No re-simulation needed across the sweep.
- **PPC pools differ across τ.** Posterior samples are different (wider posterior at higher τ), so the theta-hash-keyed posterior_predictive suffix pool is fresh per τ. Each PPC run takes ~2 min on Rockfish.
- **Prior-PC pairwise φ table is invariant in τ.** That table reads raw simulator outputs (joint reachability of empirical CI95s), which doesn't see the cohort shift. The runbook's pitfalls section now flags this so future debugging doesn't expect prior-PC tension movement.

## Followup work surfaced

- Cluster B (`cdc1_density × th_exhausted_fraction`) is now the **top structural priority** when cohort heterogeneity is properly accounted for. Mechanism: model couples cDC1 abundance directly to CD4 priming via `H_mAPC` (mcDC1 → naive CD4 expansion), so high cDC1 forces low Th-exhausted by construction. Obs has both (cDC1=7.82 cells/mm², Th-exhausted=78%). Decoupling priming from chronic-antigen exhaustion (TGFβ/stiffness-mediated, or PD-1/PD-L1 directly) is the candidate fix.

- Cluster C (apCAF/stroma/M1:M2/Th-exh tetrahedron, Jacobian-cosines structural) remains visible in misfit drivers; the `k_PSC_to_apCAF` overload is the dominant elasticity for cd4_helper, treg_fraction, total_til, collagen1. Adding a second apCAF source (tissue-resident apCAF differentiation independent of PSC pool) would decouple.

- Cluster A's residual after τ=1.0 (φ=−0.407 between Goles and Liudahl-TIL) is roughly the floor — the cohort heterogeneity is large enough that even τ=1.0 doesn't fully equalize. If we ever want true cross-cohort identification, the Path B (full hierarchical SBI with τ as inferred latent) is the principled answer; this prototype is Path A (τ as a fixed scale).
