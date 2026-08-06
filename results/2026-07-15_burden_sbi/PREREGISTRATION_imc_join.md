# Pre-registration — IMC spatial phenotype × burden posterior join

**Written 2026-07-16, BEFORE any correlation was computed.** Analysis set, hypotheses,
features, and decision rules are fixed here. Anything not listed below is EXPLORATORY.

## Rationale

Burden params were inferred from longitudinal SLD + CA19-9 alone (amortized NPE, PFS-clamped
1st-line window). IMC phenotypes are an INDEPENDENT measurement on pre-treatment tissue.
A correlation between them is therefore an orthogonal test of what the burden params mean —
not a circular re-description of the same data.

Direction of inference: baseline tissue state → subsequent burden dynamics.

## Analysis set

- 48 patients: the IMC-49 set minus `100-PM-047` (no `_cells.csv` on disk).
- Arms: ~24 GnP / ~25 mFFX (exact split recorded at runtime).
- Params = posterior MEDIANS from `imc49_pfs_posteriors.csv`.
- **Primary analysis EXCLUDES misspecified patients** (`misspec==True`; ~5 patients).
  A patient whose model does not reproduce their data has meaningless params.
  Sensitivity analysis re-runs WITH them included; both reported.

## Primary hypotheses (confirmatory)

Shared params (H1-H3) pool both arms (n≈43 after misspec exclusion), justified by the
Jul-15 cross-regimen result: shared params showed no significant arm difference (MW p>0.15).

| ID | Burden param | IMC feature(s) | Direction | n | Motivation |
|----|--------------|----------------|-----------|---|------------|
| H1 | `k_C1_growth` | Ki67+ cancer fraction | **positive** | ~43 | ORTHOGONAL VALIDATION: growth inferred from imaging/blood dynamics vs. proliferation measured in tissue. The single strongest falsifiable test available. |
| H2 | `f_ca199_C2` | classical-tumor frac; GATA6+ frac; CLDN18.2+ frac | two-sided | ~43 | CA19-9 is a secretory glycan; `f` encodes CA19-9-above-volume discordance. Tests whether baseline secretory phenotype predicts that discordance. |
| H3 | `r_resist`, `k_resist` | basal+intermediate frac; SMAD4+ frac | two-sided | ~43 | Basal-like PDAC is the chemoresistant subtype; SMAD4 status is a canonical progression driver. |
| H4 | GnP: `k_C_nabp`, `IC50_nabp`<br>mFFX: `k_kill_fu5/oxa/iri`, `IC50_*` | fibroblast frac; myCAF frac | **negative** (kill), **positive** (IC50) | ~21/~22 | Desmoplastic stroma limits drug penetration — the defining PDAC delivery hypothesis. **ARM-STRATIFIED → UNDERPOWERED (~70% power for rho=0.5). Reported as primary-but-weak.** |

## Statistics

- **Spearman rank correlation** (params are lognormal, features are bounded fractions;
  ranks are what the Jul-15 population trend already used).
- Primary tests reported **unadjusted**, with Benjamini-Hochberg FDR **within each
  hypothesis family** (H2 has 3 features, H3 has 2 params × 2 features, H4 arm-stratified).
- Two-sided p unless the Direction column says otherwise; directional hypotheses (H1, H4)
  additionally report whether the observed sign matches prediction — a correct-magnitude
  wrong-sign result counts as a FAILED prediction, not a hit.
- **n=48 is small.** No claim is made on any correlation with |rho| < 0.3 regardless of p.

## Exploratory (clearly labeled, NOT confirmatory)

- Full per-patient composition: lineage / cell_type / subtype fractions, all 11 markers.
- Niche archetypes: KMeans over per-cell `nbhd_*` vectors → per-patient archetype abundance.
  Archetype count K is a researcher degree of freedom; K chosen by silhouette and REPORTED,
  and results shown across a K sweep rather than at one flattering K.
- Full sweep: all 12 params × all features, BH-FDR across the entire sweep.
- Anything here that survives is a HYPOTHESIS FOR A FUTURE COHORT, not a result.

## Known confounders (cannot be fully resolved with data on hand)

- **Biopsy site unknown.** Metadata `Site` = enrolling institution (JHU/MSK/PM/NH/DFC), NOT
  tissue site. Liver-met vs pancreatic-primary cores are not distinguishable → composition
  features (esp. stromal/immune fractions) are confounded by site. Flagged, not fixed.
- **Institution = plausible staining/batch confounder.** Encoded in the subject prefix, so
  checkable: report per-institution feature distributions; if a primary hit tracks
  institution, it is reported as confounded.
- **Single core per patient** — spatial heterogeneity within a tumor is not captured
  (the Jul-1 steer: two biopsies from one patient genuinely differ).
- **`subtype=='proliferating'` is identical to `Ki67`** in every patient checked. ONE
  feature, not two. Must not be double-counted.

## Pre-committed interpretation

- H1 confirmed (rho>0.3, p<0.05, positive) → burden-model growth param is externally valid.
- H1 null → the growth param is a curve-fitting quantity, not a biological growth rate.
  This is a real and publishable negative, and it constrains how the QSP can be used
  predictively. **We report it either way.**
