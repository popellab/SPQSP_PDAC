# Li et al. 2022 — Digitization Results

**Paper:** Li et al., Cancer Cell 2022, DOI: 10.1016/j.ccell.2022.10.001
**Study:** Neoadjuvant GVAX ± nivolumab in resectable PDAC (NCT02451982)
**Tissue:** Surgical resection specimens, ~2 weeks post-treatment
**Region:** Non-LA (bulk tumor), mIHC, % of all nucleated cells in ROI
**Groups:** 1=OS<2y GVAX, 2=OS>2y GVAX, 3=OS<2y GVAX+Nivo, 4=OS>2y GVAX+Nivo

## CD8 T cells (non-LA tumor) — Fig 2E

Source: `cd8t_nonLA_tumor.csv`

| Group | n | Values (%) | Median (%) |
|-------|---|------------|------------|
| 1 (OS<2y, GVAX) | 4 | 1.07, 2.07, 8.04, 11.23 | 5.05 |
| 2 (OS>2y, GVAX) | 5 | 2.32, 7.95, 8.38, 10.22, 14.93 | 8.38 |
| 3 (OS<2y, GVAX+Nivo) | 3 | 5.39, 10.01, 13.08 | 10.01 |
| 4 (OS>2y, GVAX+Nivo) | 7 | 4.09, 5.39, 5.59, 6.36, 12.99, 13.72, 19.09 | 6.36 |

**Pooled by arm:**

| Arm | n | Median (%) | IQR (%) | Bootstrap 95% CI (%) |
|-----|---|------------|---------|----------------------|
| GVAX | 9 | 8.04 | [2.32, 10.22] | [2.07, 11.23] |
| GVAX+Nivo | 10 | 8.19 | [5.44, 13.06] | [5.39, 13.36] |

## Treg (non-LA tumor) — Fig S3L

Source: pasted digitization

| Group | n | Values (%) | Median (%) |
|-------|---|------------|------------|
| 1 (OS<2y, GVAX) | 4 | 0.10, 0.59, 0.65, 1.77 | 0.62 |
| 2 (OS>2y, GVAX) | 5 | 0.32, 0.34, 0.42, 1.14, 1.50 | 0.42 |
| 3 (OS<2y, GVAX+Nivo) | 3 | 0.44, 1.43, 3.55 | 1.43 |
| 4 (OS>2y, GVAX+Nivo) | 7 | 0.18, 0.49, 0.89, 1.68, 2.06, 2.13, 2.42 | 1.68 |

**Pooled by arm:**

| Arm | n | Median (%) | IQR (%) | Bootstrap 95% CI (%) |
|-----|---|------------|---------|----------------------|
| GVAX | 9 | 0.59 | [0.34, 1.14] | [0.32, 1.50] |
| GVAX+Nivo | 10 | 1.56 | [0.59, 2.11] | [0.49, 2.24] |

## M1 TAM (non-LA tumor) — Fig S5S

Source: pasted digitization (negative values floored to 0)

| Group | n | Values (%) | Median (%) |
|-------|---|------------|------------|
| 1 (OS<2y, GVAX) | 4 | 0.00, 0.53, 0.90, 2.25 | 0.71 |
| 2 (OS>2y, GVAX) | 5 | 0.03, 0.14, 0.46, 0.97, 1.18 | 0.46 |
| 3 (OS<2y, GVAX+Nivo) | 3 | 0.00, 0.04, 1.22 | 0.04 |
| 4 (OS>2y, GVAX+Nivo) | 7 | 0.25, 0.26, 0.32, 0.74, 0.87, 1.09, 1.47 | 0.74 |

**Pooled by arm:**

| Arm | n | Median (%) | IQR (%) | Bootstrap 95% CI (%) |
|-----|---|------------|---------|----------------------|
| GVAX | 9 | 0.53 | [0.14, 0.97] | [0.03, 1.18] |
| GVAX+Nivo | 10 | 0.53 | [0.25, 1.04] | [0.15, 1.09] |

## M2 TAM (non-LA tumor) — Fig S5S

Source: pasted digitization (negative values floored to 0)

| Group | n | Values (%) | Median (%) |
|-------|---|------------|------------|
| 1 (OS<2y, GVAX) | 4 | 0.00, 0.74, 2.59, 4.54 | 1.66 |
| 2 (OS>2y, GVAX) | 5 | 0.17, 0.48, 0.67, 0.78, 1.11 | 0.67 |
| 3 (OS<2y, GVAX+Nivo) | 3 | 0.01, 0.23, 2.55 | 0.23 |
| 4 (OS>2y, GVAX+Nivo) | 7 | 0.04, 0.36, 0.37, 0.74, 1.34, 1.71, 2.31 | 0.74 |

**Pooled by arm:**

| Arm | n | Median (%) | IQR (%) | Bootstrap 95% CI (%) |
|-----|---|------------|---------|----------------------|
| GVAX | 9 | 0.74 | [0.48, 1.11] | [0.17, 2.59] |
| GVAX+Nivo | 10 | 0.55 | [0.26, 1.62] | [0.20, 1.82] |

**Cross-check — M1/M2 ratio from pooled medians:**
- GVAX: 0.53/0.74 = 0.72
- GVAX+Nivo: 0.53/0.55 = 0.96 (cf. existing m1_m2_ratio_gvax_nivo_d21 target median=0.30 from non-LA Figure 3B — different because that's per-patient ratios, not ratio of medians)

## CD8+PD-1+ T cells (non-LA tumor) — Fig S4D

Source: pasted digitization

| Group | n | Values (%) | Median (%) |
|-------|---|------------|------------|
| 1 (OS<2y, GVAX) | 4 | 0.05, 0.33, 0.41, 3.26 | 0.37 |
| 2 (OS>2y, GVAX) | 5 | 0.56, 0.57, 0.65, 0.91, 1.05 | 0.65 |
| 3 (OS<2y, GVAX+Nivo) | 3 | 0.14, 0.94, 3.02 | 0.94 |
| 4 (OS>2y, GVAX+Nivo) | 7 | 0.27, 0.42, 1.03, 1.16, 1.38, 1.64, 2.09 | 1.16 |

**Pooled by arm:**

| Arm | n | Median (%) | IQR (%) | Bootstrap 95% CI (%) |
|-----|---|------------|---------|----------------------|
| GVAX | 9 | 0.57 | [0.41, 0.91] | [0.33, 1.05] |
| GVAX+Nivo | 10 | 1.09 | [0.55, 1.58] | [0.42, 1.74] |

**Cross-check — CD8+PD1+ as fraction of total CD8 (ratio of pooled medians):**
- GVAX: 0.57/8.04 = 7.0%
- GVAX+Nivo: 1.09/8.19 = 13.3% (nivo expands PD-1+ CD8 subset)

## CD4 Th (non-LA tumor) — Fig S3H

Source: pasted digitization

| Group | n | Values (%) | Median (%) |
|-------|---|------------|------------|
| 1 (OS<2y, GVAX) | 4 | 2.67, 5.45, 6.58, 9.71 | 6.01 |
| 2 (OS>2y, GVAX) | 5 | 2.19, 4.32, 5.49, 5.92, 6.09 | 5.49 |
| 3 (OS<2y, GVAX+Nivo) | 3 | 4.35, 7.51, 14.55 | 7.51 |
| 4 (OS>2y, GVAX+Nivo) | 7 | 0.41, 2.80, 6.22, 8.18, 8.36, 9.19, 11.41 | 8.18 |

**Pooled by arm:**

| Arm | n | Median (%) | IQR (%) | Bootstrap 95% CI (%) |
|-----|---|------------|---------|----------------------|
| GVAX | 9 | 5.49 | [4.32, 6.09] | [2.67, 6.58] |
| GVAX+Nivo | 10 | 7.84 | [4.82, 8.98] | [4.35, 9.89] |

## NK cells (non-LA tumor) — Fig S3F

Source: pasted digitization

| Group | n | Values (%) | Median (%) |
|-------|---|------------|------------|
| 1 (OS<2y, GVAX) | 4 | 2.01, 3.46, 5.94, 6.97 | 4.70 |
| 2 (OS>2y, GVAX) | 5 | 0.12, 0.97, 1.46, 1.90, 1.93 | 1.46 |
| 3 (OS<2y, GVAX+Nivo) | 3 | 1.41, 1.73, 3.45 | 1.73 |
| 4 (OS>2y, GVAX+Nivo) | 7 | 0.56, 0.68, 1.37, 1.88, 2.07, 3.10, 3.53 | 1.88 |

**Pooled by arm:**

| Arm | n | Median (%) | IQR (%) | Bootstrap 95% CI (%) |
|-----|---|------------|---------|----------------------|
| GVAX | 9 | 1.93 | [1.46, 3.46] | [0.97, 5.94] |
| GVAX+Nivo | 10 | 1.81 | [1.38, 2.84] | [1.05, 3.10] |

## CD4+PD-1+ T cells (non-LA tumor) — Fig S4C

Source: pasted digitization (negative values floored to 0)

| Group | n | Values (%) | Median (%) |
|-------|---|------------|------------|
| 1 (OS<2y, GVAX) | 4 | 0.28, 0.32, 1.33, 2.42 | 0.82 |
| 2 (OS>2y, GVAX) | 5 | 0.28, 0.33, 0.50, 1.18, 1.88 | 0.50 |
| 3 (OS<2y, GVAX+Nivo) | 3 | 0.05, 0.93, 3.21 | 0.93 |
| 4 (OS>2y, GVAX+Nivo) | 7 | 0.00, 0.11, 0.91, 1.31, 1.54, 1.83, 2.27 | 1.31 |

**Pooled by arm:**

| Arm | n | Median (%) | IQR (%) | Bootstrap 95% CI (%) |
|-----|---|------------|---------|----------------------|
| GVAX | 9 | 0.50 | [0.32, 1.33] | [0.28, 1.88] |
| GVAX+Nivo | 10 | 1.12 | [0.31, 1.76] | [0.11, 1.90] |

## Th1 (non-LA tumor) - fig S3I

Source: pasted digitization (negative values floored to 0)

| Group | n | Values (%) | Median (%) |
|-------|---|------------|------------|
| 1 (OS<2y, GVAX) | 4 | 0.06, 0.25, 0.88, 2.65 | 0.57 |
| 2 (OS>2y, GVAX) | 5 | 0.00, 0.01, 0.43, 0.55, 2.24 | 0.43 |
| 3 (OS<2y, GVAX+Nivo) | 3 | 0.24, 0.56, 2.94 | 0.56 |
| 4 (OS>2y, GVAX+Nivo) | 7 | 0.00, 0.03, 0.12, 0.28, 0.46, 1.03, 1.30 | 0.28 |

**Pooled by arm:**

| Arm | n | Median (%) | IQR (%) | Bootstrap 95% CI (%) |
|-----|---|------------|---------|----------------------|
| GVAX | 9 | 0.43 | [0.06, 0.88] | [0.01, 2.24] |
| GVAX+Nivo | 10 | 0.37 | [0.15, 0.91] | [0.12, 1.03] |

## Th2 (non-LA tumor) - figure S3J

Source: pasted digitization (negative values floored to 0)

| Group | n | Values (%) | Median (%) |
|-------|---|------------|------------|
| 1 (OS<2y, GVAX) | 4 | 0.00, 0.35, 0.41, 1.55 | 0.38 |
| 2 (OS>2y, GVAX) | 5 | 0.30, 0.39, 0.42, 0.71, 3.16 | 0.42 |
| 3 (OS<2y, GVAX+Nivo) | 3 | 0.04, 1.62, 1.99 | 1.62 |
| 4 (OS>2y, GVAX+Nivo) | 7 | 0.61, 0.63, 0.63, 0.68, 1.56, 1.94, 2.56 | 0.68 |

**Pooled by arm:**

| Arm | n | Median (%) | IQR (%) | Bootstrap 95% CI (%) |
|-----|---|------------|---------|----------------------|
| GVAX | 9 | 0.41 | [0.35, 0.71] | [0.30, 1.55] |
| GVAX+Nivo | 10 | 1.12 | [0.63, 1.86] | [0.62, 1.94] |

## Th17 (non-LA tumor) — figure 2O

Source: pasted digitization (negative values floored to 0)

| Group | n | Values (%) | Median (%) |
|-------|---|------------|------------|
| 1 (OS<2y, GVAX) | 4 | 0.000, 0.054, 0.090, 0.398 | 0.072 |
| 2 (OS>2y, GVAX) | 5 | 0.023, 0.072, 0.111, 0.200, 0.210 | 0.111 |
| 3 (OS<2y, GVAX+Nivo) | 3 | 0.075, 0.171, 0.274 | 0.171 |
| 4 (OS>2y, GVAX+Nivo) | 7 | 0.275, 0.327, 0.349, 0.568, 0.704, 0.721, 1.029 | 0.568 |

**Pooled by arm:**

| Arm | n | Median (%) | IQR (%) | Bootstrap 95% CI (%) |
|-----|---|------------|---------|----------------------|
| GVAX | 9 | 0.090 | [0.054, 0.200] | [0.023, 0.210] |
| GVAX+Nivo | 10 | 0.338 | [0.274, 0.670] | [0.223, 0.704] |

**Cross-check — CD4 subtype accounting (pooled medians):**

| Subtype | GVAX (%) | GVAX+Nivo (%) |
|---------|----------|---------------|
| CD4 total | 5.49 | 7.84 |
| Treg | 0.59 | 1.55 |
| Th1 | 0.43 | 0.37 |
| Th2 | 0.41 | 1.12 |
| Th17 | 0.09 | 0.34 |
| Sum subtypes | 1.52 | 3.38 |
| Residual (Th0/undiff) | 3.97 | 4.46 |

Note: large residual (~4-4.5%) likely represents undifferentiated Th0 CD4+ T cells not captured by lineage-specific markers.

## DC-LAMP+ (mature DC) — partial digitization

Only boundary points visible; rest clustered near 0.

| Arm | n | Visible range (%) | Notes |
|-----|---|-------------------|-------|
| GVAX | 9 | [0, 0.72] | Most points clustered near 0 |
| GVAX+Nivo | 10 | [0, 0.96] + outlier at 2.84 | Main cluster [0, 0.96], one outlier |

For calibration: approximate as uniform [0, 0.72] for GVAX, [0, 0.96] for GVAX+Nivo (excluding outlier).
Median ~0.36% (GVAX), ~0.48% (GVAX+Nivo). Mature DCs very rare in both arms.

## DC-SIGN+ (immature/tolerogenic DC) — not digitized

Similar to DC-LAMP but slightly higher in both arms. Immature DCs marginally more abundant than mature DCs, consistent with immunosuppressive TME limiting DC maturation.
