# IL-1 / M2 cluster trace — 2026-05-12

Single-θ qsp_sim traces motivated by the misspec audit on
`figures/multi_postfix_priors_v1_no_nivo/` (the 3-scenario rerun after
dropping the day-7-truncated gvax_nivo arm). The audit flags 5
MECHANISM_MISSING + 3 DEGENERACY targets, with `active_fraction_IL1`
rail-pinned at the high prior edge and `k_M2_pol` top-driving two
M2-related targets with opposite-sign z's. Goal: identify what pulls
`k_IL1_sec` 15× above prior and whether the z=+4.39 outlier on
`m2tam_pct_nonLA_gvax_d21` is real or an artifact.

## 1. IL-1 axis sweep (baseline_no_treatment, evolve_to_diagnosis @ 1875 d)

Single-parameter sweep over `k_IL1_sec` at constant priors elsewhere. Posterior
median is 1.53e-11 (15× prior median 1e-12).

| k_IL1_sec | V_T.IL1 (nM) | iCAF | myCAF | apCAF | IL10 (nM) | TGFb | M1/M2 | C1 |
|--|--|--|--|--|--|--|--|--|
| 1× (prior) | 3.04e-3 | 1.09e8 | 5.65e8 | 6.75e7 | 2.16e-3 | 5.22e-2 | 1.87 | 3.61e8 |
| 3× | 9.83e-3 | 1.96e8 | 5.44e8 | 6.77e7 | 2.34e-3 | 5.05e-2 | 1.63 | 3.90e8 |
| 6× | 2.18e-2 | 2.74e8 | 5.02e8 | 7.00e7 | 2.46e-3 | 4.71e-2 | 1.53 | 4.33e8 |
| 10× | 4.08e-2 | 3.35e8 | 4.54e8 | 7.45e7 | 2.56e-3 | 4.32e-2 | 1.47 | 4.85e8 |
| 12× | 5.12e-2 | 3.55e8 | 4.34e8 | 7.66e7 | 2.60e-3 | 4.17e-2 | 1.45 | 5.07e8 |
| 15× (post med) | — | — | — | — | — | — | — | **evolve_to_diagnosis FAILED, max 2.97 cm at 7300 d** |

### What the sweep says

- **iCAF responds strongly to k_IL1_sec** (1.1e8 → 3.6e8 over 1–12×, ~3.3×).
  This is the intended `H_IL1_eff → iCAF` mechanism. Mass-balance: myCAF
  drops 1.3× as IL-1 climbs (iCAF↔myCAF interconversion).
- **IL-10 barely moves (1.0 → 1.2×).** iCAFs don't secrete enough IL-10 in
  this model to make k_IL1_sec a viable lever on the IL-10 axis. So the
  posterior is *not* pulling k_IL1_sec up to fit the IL-10 target.
- **M1/M2 drops modestly** (1.87 → 1.45, 1.29×). Same — k_IL1_sec is not
  meaningfully responsible for the M2-polarization fit.
- **C1 (tumor) grows** as k_IL1_sec climbs (3.6e8 → 5.1e8). IL-1 has a
  net pro-tumor effect through the stromal axis at finite values.
- **Above ~12×, evolve_to_diagnosis stalls.** At posterior median (15×)
  the model can't reach diagnosis diameter — there's a bifurcation
  between 12× and 15× where the iCAF-driven feedback prevents tumor
  growth from reaching 3.2 cm. The posterior must be combining high
  k_IL1_sec with compensating params (faster growth, weaker killing,
  etc.) — a degenerate-joint signature.

### So why does the posterior push k_IL1_sec up 15×?

Not for IL-10. Not for M1/M2. Look at the conflict pair from the audit:

> `PDAC_VT_IL1_tumor_tissue_diagnosis_GhassemZadeh2017` ↔ `stromal_fraction` —
> cos −0.69, IL1-alone prior reach 0.01%, stromal-alone 89.12%, **joint 0.00%**.

`stromal_fraction` wants high CAF mass (89% reach), and IL-1 expands iCAF
without losing total CAF (iCAF↔myCAF interconversion conserves total).
So **`k_IL1_sec` is dragged up by `stromal_fraction`, taking the IL-1
target observable with it.** The IL-1 target wants `V_T.IL1 ≈ 8.4e-4 nM`
(= empirical 0.008362 × prior-median `active_fraction=0.1`), but at
prior medians `V_T.IL1` is already 3.04e-3 nM (3.6× too high).
`active_fraction_IL1` rails at the high prior edge trying to absorb the
mismatch, and the audit still finds the target at z=+1.44.

This is a real **target conflict**, not a parameter bug. The two
literature sources (`Ghassem-Zadeh 2017` IL-1 ELISA vs whatever drives
`stromal_fraction`) are mutually inconsistent under the current
IL-1→iCAF model. Options:

1. **Question the IL-1 target.** Whole-tissue ELISA of "active" IL-1
   in tumor lysates is a notoriously noisy measurement; the
   `active_fraction_IL1` bridge prior may already be too narrow.
2. **Question the iCAF-via-IL-1 mechanism.** If iCAF formation is
   substantially TGFβ- or IL-6-driven and only weakly IL-1-driven,
   the stromal expansion shouldn't need high `k_IL1_sec`.
3. **Add IL-1 production from sources other than cancer cells**
   (e.g., damaged stroma) — would let `V_T.IL1` rise without forcing
   `k_IL1_sec` (cancer-cell secretion) up.

## 2. gvax_neoadjuvant day-21 m2tam_pct — the z=+4.39 outlier

Traced the full 21-day gvax_neoadjuvant scenario at prior medians and at
posterior medians (k_IL1_sec, k_M2_pol, k_M1_pol, k_IL10_sec, k_Mac_death,
k_IL1_deg, IL10_50 all set to posterior medians from
`figures/multi_postfix_priors_v1_no_nivo/posterior_summary.csv`).

### Decomposition: M2 absolute vs immune-total

| theta | t (d) | M2 | immune_total | M2/immune | M1/M2 | Treg | MDSC |
|--|--|--|--|--|--|--|--|
| prior | 0 | 4.95e7 | 2.41e8 | 0.206 | 1.87 | 8.55e7 | 1.07e7 |
| prior | 7.0 (post-Cy+GVAX) | 4.96e7 | 2.33e8 | 0.213 | 1.87 | 7.74e7 | 1.07e7 |
| prior | 21 | 4.99e7 | 2.39e8 | 0.208 | 1.87 | 8.34e7 | 1.07e7 |
| posterior | 21 | 1.30e8 | 2.50e8 | 0.520 | 0.15 | 8.40e7 | 1.41e7 |

**Cy is not depleting the denominator enough to spike the M2 fraction.**
Treg dips ~10% post-Cy (day 6→7) and recovers by day 14. MDSC barely
moves. The model's denominator at day 21 is essentially equal to its
denominator at day 0. So the z=+4.39 outlier is **not** a "Cy depletes
the denominator and inflates the M2 fraction" artifact.

### The actual cause — observable code / schema mismatch

The YAML target says:

- `study_interpretation`: "M2 TAM as percentage of **all nucleated cells**
  in non-LA tumor ROIs" (Li et al. Cancer Cell 2022, Fig S5S, mIHC)
- `experimental_denominator`: "All nucleated cells in non-LA tumor ROI (mIHC)"
- `model_denominator_species`: lists `V_T.C1`, `V_T.C_x` (cancer cells)
  and all CAF subtypes alongside immune species

But the `observable.code` function:

```python
total = (V_T.CD8 + V_T.CD8_exh + V_T.Th + V_T.Th_exh + V_T.Treg
       + V_T.Mac_M1 + V_T.Mac_M2 + V_T.MDSC
       + V_T.cDC1 + V_T.cDC2 + V_T.mcDC1 + V_T.mcDC2)
```

**only sums immune species — no `V_T.C1`, no CAFs, no `V_T.C_x`.** The
formula is `M2 / immune_only`, not `M2 / all_nucleated`.

### Quantitative impact

At prior medians, day 21:

- M2 = 4.99e7
- Immune-only total (current code): 2.39e8 → fraction = 0.208
- All-nucleated total (correct denominator: + V_T.C1 ~3.6e8 + iCAF+myCAF+apCAF ~7e8):
  ~1.30e9 → fraction = **0.038**
- Empirical target: 0.0074 (CI95 [0.0017, 0.0259])

The corrected fraction is still ~5× over the empirical median, but
within an order of magnitude — vs the buggy formula's 28× over. The
audit z would drop from +4.39 to roughly +1.5–2.0, plausibly back into
"MARGINAL" territory.

### Tension with `m1_m2_ratio`

`m1_m2_ratio` (baseline) empirical = 0.518, prior-median model = 1.87 →
posterior pushes M2 up to drop M1/M2 to ~0.15. But `m2tam_d21` wants M2
*fraction of all cells* lower (target 0.0074). If we trust both:

- Target → M2 = 0.518 × M1 (from m1_m2)
- Target → M2 / total_nucleated = 0.0074

So total_nucleated ≈ M2 / 0.0074 = 70 × M2 ≈ 36 × M1. Currently
total_nucleated / M1 ≈ 14 (1.3e9 / 9.3e7). The model needs the
non-macrophage compartments to be ~2.5× larger relative to macrophages.
This is plausible — PDAC TME is macrophage-rich but the literature
still shows lymphocytes + cancer cells dominating numerically. The
fix-the-formula correction will *reveal* this tension rather than mask
it under the artificial 5× inflation.

## 3. Headline recommendations

1. **Fix the `m2tam_pct_nonLA_gvax_d21` observable code** to match
   `model_denominator_species`. Same fix applies to the
   `m2tam_pct_nonLA_gvax_nivo_d21` sibling target. This should drop
   the z=+4.39 outlier substantially; rerun the audit to confirm.
2. **Audit other density-vs-fraction targets** for the same
   `code` vs `model_denominator_species` mismatch class
   (see `project_density_obs_formula_bug.md` for prior precedent).
3. **Investigate the `stromal_fraction` ↔ IL-1 conflict** before
   widening the `active_fraction_IL1` prior or k_IL1_sec prior:
   either question the IL-1 target (active-fraction prior may be
   too tight) or the iCAF-via-IL-1 mechanism strength (consider
   whether TGFβ/IL-6 should dominate iCAF formation more than IL-1
   does in the current `H_IL1_eff` form).
4. **`k_M2_pol` is the M1/M2 rate-limiter, not IL-10.** Pushing
   `k_IL10_sec` 22× barely moves M1/M2 (1.87 → 1.09); pushing
   `k_M2_pol` 5.6× flips it (1.87 → 0.258). The `m1_m2_ratio`
   MECHANISM_MISSING / prior_reach=✗ verdict is really "prior on
   `k_M2_pol` is centered an order of magnitude too low". The
   target *is* reachable in the posterior tail — consistent with
   the runbook's reachability-not-prior-coverage principle.
5. **The IL-1 cluster is structurally degenerate at posterior θ.**
   At posterior median `k_IL1_sec` alone, evolve_to_diagnosis stalls
   at 2.97 cm — the model can't reach diagnosis without compensating
   params. The joint posterior is finding a corner of parameter
   space that combines high IL-1 with compensating tumor-growth
   params; this is fragile and likely contributes to the
   DEGENERACY verdicts on `stromal_fraction`,
   `cd8_fc_nonLA_gvax_d21`, and `ccl2_mcp1`.

## 4. Provenance

- Run: `figures/multi_postfix_priors_v1_no_nivo` (3-scenario SBI, 20k
  sims, seed 2025, completed 2026-05-12).
- Posterior medians from `posterior_summary.csv` of that run.
- Traces: `/tmp/il1_sweep_{1,3,6,10,12}x.csv` and
  `/tmp/gvax_{base,post}.csv`.
- Binary: `cpp/sim/build/qsp_sim` (refreshed via `make refresh-cpp`
  with `CVODEBase.cpp` patches from sibling qsp-codegen).
