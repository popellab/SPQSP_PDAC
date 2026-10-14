"""Build vpop corpus + cohorts YAMLs from calibration target YAMLs.

Reads every non-excluded target in calibration_targets/{scenario}/*.yaml,
maps it to the vpop readout name from vpop_readouts.py, and writes:
  - results/vpop_pool/corpus.yaml
  - results/vpop_pool/cohorts.yaml

Usage
-----
    .venv/bin/python -I scripts/build_vpop_corpus.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vpop_readouts import READOUTS

REPO = Path(__file__).resolve().parent.parent

# ── calibration_target_id → vpop readout name ────────────────────────
TARGET_TO_READOUT = {
    # Baseline fractions (PASS-01)
    "cancer_fraction_PASS01": "cancer_fraction",
    "cd8_fraction_PASS01": "cd8_fraction",
    "cd4_fraction_PASS01": "cd4_fraction",
    "macrophage_fraction_PASS01": "macrophage_fraction",
    "fibroblast_fraction_PASS01": "fibroblast_fraction",
    "dendritic_fraction_PASS01": "dendritic_fraction",
    "mdsc_fraction_PASS01": "mdsc_fraction",
    "bcell_fraction_PASS01": "bcell_fraction",
    # Baseline sub-fractions
    "cd8_exhausted_fraction_shen2017": "cd8_exhausted_fraction",
    "th_exhausted_fraction": "th_exhausted_fraction",
    "treg_fraction_baseline": "treg_fraction_of_tcells",
    "treg_fraction_cd4": "treg_fraction_cd4",
    "treg_fraction_cd4_hiraoka2006": "treg_fraction_cd4",
    "stromal_fraction": "stromal_fraction",
    "apc_maturation_ratio": "apc_maturation_ratio",
    "icaf_fraction_of_caf": "icaf_fraction_caf",
    "mycaf_fraction_of_caf": "mycaf_fraction_caf",
    "apcaf_fraction_of_caf": "apcaf_fraction_caf",
    # Baseline densities
    "cd8_density_baseline": "cd8_density",
    "cd8_density_baseline_golesworthy2022": "cd8_density",
    "cd8_density_baseline_liu2015": "cd8_density",
    "cd4_helper_density": "cd4_helper_density",
    "treg_density_baseline": "treg_density",
    "tam_density": "tam_density",
    "mdsc_density": "mdsc_density",
    "cdc1_density_baseline": "cdc1_density",
    "cdc2_density_baseline": "cdc2_density",
    "total_til_density": "total_til_density",
    # Baseline ratios
    "cd8_treg_ratio_golesworthy2022": "cd8_treg_ratio",
    "m1_m2_ratio": "m1_m2_ratio",
    # Baseline fold
    "collagen1_protein_level": "collagen_fold",
    # Baseline concentrations
    "pdac_plasma_tgfb1_vizio2012": "tgfb_concentration",
    "pdac_serum_il12p70_dorman2022_censored": "il12_concentration",
    "pdac_serum_arg1_cane2023": "arginase_concentration",
    "PDAC_IL10_tumor_tissue_Herremans2023": "il10_concentration",
    "PDAC_VT_IL1_tumor_tissue_diagnosis_GhassemZadeh2017": "il1_concentration",
    "PDAC_VT_IL6_tumor_tissue_diagnosis_GhassemZadeh2017": "il6_concentration",
    "pdac_tumor_ccl2_mcp1_concentration_ghassemzadeh2017": "ccl2_concentration",
    "vegf_tissue_concentration": "vegf_concentration",
    # GVAX d21
    "cd4_pct_nonLA_gvax_d21": "cd4_pct_gvax_d21",
    "cd8_pct_nonLA_gvax_d21": "cd8_pct_gvax_d21",
    "cd8exh_pct_nonLA_gvax_d21": "cd8exh_pct_gvax_d21",
    "m1tam_pct_nonLA_gvax_d21": "m1tam_pct_gvax_d21",
    "m2tam_pct_nonLA_gvax_d21": "m2tam_pct_gvax_d21",
    "treg_pct_nonLA_gvax_d21": "treg_pct_gvax_d21",
    "cd8_fc_nonLA_gvax_d21": "cd8_fc_gvax_d21",
    "cd8gzmb_fc_nonLA_gvax_d21": "cd8gzmb_fc_gvax_d21",
    # GVAX+nivo d21
    "cd4_pct_nonLA_gvax_nivo_d21": "cd4_pct_gvax_nivo_d21",
    "cd8_pct_nonLA_gvax_nivo_d21": "cd8_pct_gvax_nivo_d21",
    "cd8exh_pct_nonLA_gvax_nivo_d21": "cd8exh_pct_gvax_nivo_d21",
    "m1tam_pct_nonLA_gvax_nivo_d21": "m1tam_pct_gvax_nivo_d21",
    "m2tam_pct_nonLA_gvax_nivo_d21": "m2tam_pct_gvax_nivo_d21",
    "treg_pct_nonLA_gvax_nivo_d21": "treg_pct_gvax_nivo_d21",
    "cd8_fc_nonLA_gvax_nivo_d21": "cd8_fc_gvax_nivo_d21",
    "cd8gzmb_fc_nonLA_gvax_nivo_d21": "cd8gzmb_fc_gvax_nivo_d21",
    "treg_fraction_gvax_nivo_d21": "treg_fraction_immune_gvax_nivo",
    "m1_m2_ratio_gvax_nivo_d21": "m1_m2_ratio_gvax_nivo_d21",
}

# ── readout → quantity_kind (SCALE_BY_KIND key) ─────────────────────
READOUT_KIND = {}
for r in READOUTS:
    if "fraction" in r or "pct" in r or "maturation" in r:
        READOUT_KIND[r] = "fraction"
    elif "density" in r or "til_density" in r:
        READOUT_KIND[r] = "density"
    elif "ratio" in r:
        READOUT_KIND[r] = "ratio"
    elif "fold" in r or "_fc_" in r:
        READOUT_KIND[r] = "foldchange"
    elif "concentration" in r:
        READOUT_KIND[r] = "concentration"
    else:
        READOUT_KIND[r] = "count"

# ── scenario directories ────────────────────────────────────────────
SCENARIO_DIRS = {
    "baseline_no_treatment": "baseline",
    "gvax_neoadjuvant": "gvax_d21",
    "gvax_nivo_neoadjuvant": "gvax_nivo_d21",
}


def main():
    cal_dir = REPO / "calibration_targets"
    readout_set = set(READOUTS)

    targets = []
    cohort_map = {}  # (source_tag, scenario, n) → cohort_id

    for scenario_dir, scenario_name in SCENARIO_DIRS.items():
        d = cal_dir / scenario_dir
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.yaml")):
            if f.name == "reference_values.yaml":
                continue
            with open(f) as fh:
                data = yaml.safe_load(fh)

            tid = data.get("calibration_target_id", "")
            if tid not in TARGET_TO_READOUT:
                print(f"  SKIP (no readout mapping): {tid}")
                continue

            readout = TARGET_TO_READOUT[tid]
            if readout not in readout_set:
                print(f"  SKIP (readout not in READOUTS): {readout}")
                continue

            ed = data.get("empirical_data", {})
            ps = data.get("primary_data_source", {})
            median_list = ed.get("median", [])
            ci95_list = ed.get("ci95", [])
            n = ed.get("sample_size")
            source = ps.get("source_tag", "unknown")

            if not median_list or not ci95_list or n is None:
                print(f"  SKIP (missing data): {tid}")
                continue

            median = median_list[0]
            ci_lo, ci_hi = ci95_list[0]

            # Build cohort key
            cohort_key = (source, scenario_name, n)
            if cohort_key not in cohort_map:
                slug = source.lower().replace(" ", "_")
                suffix = {"baseline": "bl", "gvax_d21": "gv",
                           "gvax_nivo_d21": "gn"}[scenario_name]
                cohort_id = f"{slug}_{suffix}"
                # Deduplicate
                if cohort_id in {v for v in cohort_map.values()}:
                    cohort_id = f"{cohort_id}_{n}"
                cohort_map[cohort_key] = cohort_id
            cohort_id = cohort_map[cohort_key]

            kind = READOUT_KIND.get(readout, "count")
            units = "dimensionless" if kind == "fraction" else "cells/mm2" if kind == "density" else "dimensionless"

            # Build statistics from median + CI95
            statistics = [
                {"stat": "quantile", "p": 0.5, "value": float(median)},
            ]
            if ci_lo is not None and ci_lo > 0:
                statistics.insert(0,
                    {"stat": "quantile", "p": 0.025, "value": float(ci_lo)})
            if ci_hi is not None:
                statistics.append(
                    {"stat": "quantile", "p": 0.975, "value": float(ci_hi)})

            targets.append({
                "target_id": tid,
                "cohort_id": cohort_id,
                "readout": readout,
                "quantity_kind": kind,
                "units": units,
                "assay_modality": "IHC" if kind in ("fraction", "density") else "serum",
                "measurement_context": {},
                "statistics": statistics,
            })

    # ── Build cohorts ────────────────────────────────────────────────
    cohorts_list = []
    for (source, scenario_name, n), cohort_id in sorted(
            cohort_map.items(), key=lambda x: x[1]):
        cohorts_list.append({
            "cohort_id": cohort_id,
            "description": f"{source}, {scenario_name}, n={n}",
            "scenarios": [scenario_name],
            "n_c": n,
            "source_tag": source,
        })

    # ── Write corpus YAML ────────────────────────────────────────────
    out_dir = REPO / "results" / "vpop_pool"
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = out_dir / "corpus.yaml"
    corpus_doc = {"targets": targets}
    with open(corpus_path, "w") as fh:
        yaml.dump(corpus_doc, fh, default_flow_style=False, sort_keys=False,
                  allow_unicode=True, width=120)
    print(f"\nWrote {len(targets)} targets to {corpus_path}")

    cohorts_path = out_dir / "cohorts.yaml"
    cohorts_doc = {"cohorts": cohorts_list, "blocks": []}
    with open(cohorts_path, "w") as fh:
        yaml.dump(cohorts_doc, fh, default_flow_style=False, sort_keys=False,
                  allow_unicode=True, width=120)
    print(f"Wrote {len(cohorts_list)} cohorts to {cohorts_path}")

    # ── Summary ──────────────────────────────────────────────────────
    readouts_covered = sorted({t["readout"] for t in targets})
    readouts_missing = sorted(readout_set - set(readouts_covered))
    print(f"\nReadouts covered: {len(readouts_covered)}/{len(READOUTS)}")
    if readouts_missing:
        print(f"Readouts with no data: {readouts_missing}")


if __name__ == "__main__":
    main()
