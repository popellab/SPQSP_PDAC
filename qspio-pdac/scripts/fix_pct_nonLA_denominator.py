"""Patch _pct_nonLA_ calibration targets to use CD45+ denominator (immune-only).

Li 2022 (Cancer Cell) computes these percentages as 'fraction of CD45+ cells'
(see Fig 2E and text at page ~7: "calculated as the percentage of total CD45+
cells"). The extracted YAMLs incorrectly sum all cells in V_T — cancer + CAFs +
immune — for the denominator, yielding a value 10-100x smaller than the paper's.
This drives ~134/160 of the multi-scenario Mahalanobis D² observed in SBI.
"""

from pathlib import Path

OLD_DENOM = (
    "total = species_dict['V_T.C1'] + species_dict['V_T.C_x'] "
    "+ species_dict['V_T.CD8'] + species_dict['V_T.CD8_exh'] "
    "+ species_dict['V_T.Th'] + species_dict['V_T.Th_exh'] + species_dict['V_T.Treg'] "
    "+ species_dict['V_T.Mac_M1'] + species_dict['V_T.Mac_M2'] + species_dict['V_T.MDSC'] "
    "+ species_dict['V_T.cDC1'] + species_dict['V_T.cDC2'] "
    "+ species_dict['V_T.mcDC1'] + species_dict['V_T.mcDC2'] "
    "+ species_dict['V_T.iCAF'] + species_dict['V_T.myCAF'] + species_dict['V_T.apCAF'] "
    "+ species_dict['V_T.qPSC']"
)

# CD45+ = immune cells only (T cells + TAMs + MDSCs + DCs).
# Excludes: cancer (C1, C_x), CAFs (iCAF, myCAF, apCAF), quiescent PSC (qPSC).
NEW_DENOM = (
    "total = species_dict['V_T.CD8'] + species_dict['V_T.CD8_exh'] "
    "+ species_dict['V_T.Th'] + species_dict['V_T.Th_exh'] + species_dict['V_T.Treg'] "
    "+ species_dict['V_T.Mac_M1'] + species_dict['V_T.Mac_M2'] + species_dict['V_T.MDSC'] "
    "+ species_dict['V_T.cDC1'] + species_dict['V_T.cDC2'] "
    "+ species_dict['V_T.mcDC1'] + species_dict['V_T.mcDC2']"
)


def patch(path: Path) -> bool:
    text = path.read_text()
    if OLD_DENOM not in text:
        return False
    new = text.replace(OLD_DENOM, NEW_DENOM)
    path.write_text(new)
    return True


if __name__ == "__main__":
    targets = sorted(Path("calibration_targets").rglob("*pct_nonLA*.yaml"))
    n_patched = 0
    for t in targets:
        if patch(t):
            print(f"  patched: {t}")
            n_patched += 1
        else:
            print(f"  skipped (pattern not found): {t}")
    print(f"\nPatched {n_patched}/{len(targets)} files")
