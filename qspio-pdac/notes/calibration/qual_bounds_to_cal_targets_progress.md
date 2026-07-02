# Qualitative Bounds → Calibration Targets: Progress Report

## Context

Converting qualitative bounds from `tests/qualitative_bounds.yaml` into proper calibration target YAMLs in `calibration_targets/baseline_no_treatment/`. Each YAML is validated against the `CalibrationTarget` pydantic model from `qsp-llm-workflows`.

## Completed (8 of 12)

| # | Bound | File Created | Source | Status |
|---|-------|-------------|--------|--------|
| 15 | CD8 exhaustion fraction | `cd8_exhausted_fraction_PDAC_deriv002.yaml` | Shen 2017 (n=94, mean 30.62±16.80%) | VALID |
| 16 | CD8:Treg ratio | `cd8_treg_ratio_PDAC_deriv002.yaml` | Golesworthy 2022 (n=166, **directly reported per-patient ratio** 9.8±23.8) | VALID |
| 17 | iCAF fraction | `icaf_fraction_of_caf_PDAC_deriv002.yaml` | Elyada 2019 KPC mouse (n=20, 44.4±3.9%) | VALID |
| 18 | apCAF fraction | `apcaf_fraction_of_caf_PDAC_deriv002.yaml` | Elyada 2019 KPC mouse (n=20, 10.3±1.25%) | VALID |
| 19 | M2 predominance rate | `m2_predominance_rate_PDAC_deriv001.yaml` | Ino 2013 (n=212, 78% M2-predominant) — response_rate aggregation | VALID |
| 20 | Treg fraction of CD4+ | `treg_fraction_cd4_PDAC_deriv002.yaml` | Hiraoka 2006 (n=198, mean 34.6±10.9% SD) | VALID |
| 21 | Absolute lactate | `tumor_lactate_absolute_PDAC_deriv002.yaml` | Battini 2017 (n=44, PA 16.137±4.727 nmol/mg → 16,137±4,727 µM) | VALID |
| 26 | Tumor diameter | `tumor_diameter_PDAC_deriv001.yaml` | Tran 2022 (n=315, median 30mm, IQR 23-35mm, 2D-macro) | VALID |

### Key lesson from #16
The ratio of means ≠ mean of ratios. Golesworthy 2022 reports per-patient CD8:FOXP3 ratio directly (mean 9.8±23.8, n=166 HR/MMR-intact), which differs substantially from the ratio of mean densities (40.5/13.6 = 3.0). Always check if papers report the ratio directly before computing it from marginal distributions.

### Key decision for #19
Ino 2013 only reports "78% M2-predominant" (binary), not continuous M2:M1 ratios. Created as a `response_rate` aggregation target (like the existing HIF positivity rate target), complementing the continuous M1/M2 ratio from Dias Costa 2022 (deriv001).

### Key finding for #20
Hiraoka 2006 **does** report SD — the Methods section states "Values were expressed as mean ± SD." The full text (accessed via JHU institutional proxy) gives 34.6 ± 10.9% (n=198 PDAC). The abstract only says "average (34.6%)" without the SD, which is why it appeared missing from the abstract alone. Fig. 2E shows bar chart with error bars broken down by tumor grade: W/D (n=24, ~27%), M/D (n=140, ~34%), P/D (n=24, ~41%). Beta distribution fit: alpha=6.24, beta=11.80, median=0.340, CI95=[0.151, 0.573].

### Key decisions for #21 and #26

**#21 (lactate absolute)**: Used same Battini 2017 source as deriv001 (fold-change) but extracted the absolute PA concentration (16.137 nmol/mg → 16,137 µM via tissue density assumption). Together, deriv001 pins the fold-change and deriv002 pins the absolute scale. Lognormal fit: median 15,446 µM, CI95 [8,805, 27,251] µM.

**#26 (tumor diameter)**: Takahashi 2018 (qual bound source) only reports binary ≤2cm vs >2cm split, not continuous data. Used Tran et al. 2022 (Cancers, PMC9139767, n=315 resected PDAC from Oslo) which reports 2D-macro: mean 30mm (SD 10), median 30mm (IQR 23-35), range 9-110mm. Lognormal from median+IQR: median 3.0 cm, CI95 [1.63, 5.54]. Takahashi 2018 cited as secondary source.

### Validator lesson: keep inputs in paper units
The `CalibrationTarget` validator checks that input `value` appears in `value_snippet`. When unit conversion is needed (e.g., nmol/mg → µM, mm → cm), keep inputs in original paper units and convert in `distribution_code`. Otherwise validation fails because the converted value doesn't match the snippet.

## Skipped (4 targets)

| # | Bound | Reason for skipping |
|---|-------|-------------------|
| 22 | Absolute glucose | Kamphorst 2015 only has qualitative data. Existing deriv001 uses Battini 2017 quantitative data. Low marginal value. |
| 23 | VEGF tissue conc | Qual bound sources are cross-cancer (Haslene-Hox 2014 = ovarian) or portal blood (Hogendorf 2014). Existing deriv001 uses Delitto 2015 PDAC tissue, which is better. |
| 24 | HIF activity level | Qual bound [0.5, 1.0] is derived from pO2→HIF Hill function (model-derived, not empirical). Existing deriv001 uses Sun 2007 binary IHC positivity rate. No true independent continuous data. |
| 25 | CAF:cancer ratio | Erkan 2012 is a review article with "< 20% cancer cells" — too qualitative for a proper distribution. No primary quantitative source found. |

## Validation command

```bash
.venv/bin/python -c "
import yaml, json
from qsp_llm_workflows.core.calibration import CalibrationTarget
data = yaml.safe_load(open('calibration_targets/baseline_no_treatment/FILENAME.yaml'))
units = json.loads(open('batch_jobs/input_data/species_units.json').read())
CalibrationTarget.model_validate(data, context={'species_units': units})
print('VALID')
"
```

## Summary

8 of 12 qualitative bounds successfully converted to calibration targets. 4 skipped due to insufficient quantitative data or redundancy with existing targets. All 8 created targets pass `CalibrationTarget` validation.