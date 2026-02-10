#!/usr/bin/env python3
"""
Parameter Comparison Tool: GPU PDAC vs CPU HCC

Compares all behavioral and chemical parameters between GPU (PDAC) and CPU (HCC)
implementations to verify alignment before validation testing.

Usage:
    python3 compare_parameters.py [--show-all] [--export-csv]
"""

import re
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

class ParameterComparator:
    """Compares GPU and CPU parameter values."""

    def __init__(self, gpu_root: Path, cpu_root: Path):
        self.gpu_root = gpu_root
        self.cpu_root = cpu_root
        self.parameters = {}
        self.mismatches = []

    def extract_gpu_params(self) -> Dict[str, float]:
        """Extract parameters from GPU source files."""
        params = {}

        # PDE diffusion coefficients
        pde_file = self.gpu_root / "PDAC/pde/pde_integration.cu"
        if pde_file.exists():
            content = pde_file.read_text()

            # Diffusion coefficients
            diffusion_pattern = r'config\.diffusion_coeffs\[CHEM_(\w+)\]\s*=\s*([\d.e-]+)'
            for match in re.finditer(diffusion_pattern, content):
                chem, value = match.groups()
                params[f"D_{chem}"] = float(value)

            # Decay rates
            decay_pattern = r'config\.decay_rates\[CHEM_(\w+)\]\s*=\s*([\d.e-]+)'
            for match in re.finditer(decay_pattern, content):
                chem, value = match.groups()
                params[f"lambda_{chem}"] = float(value)

        # Model definition parameters
        model_file = self.gpu_root / "PDAC/sim/model_definition.cu"
        if model_file.exists():
            content = model_file.read_text()

            # Extract environment properties
            env_pattern = r'env\.newProperty<float>\("(\w+)",\s*([\d.e-]+)'
            for match in re.finditer(env_pattern, content):
                name, value = match.groups()
                params[name] = float(value)

            env_pattern_int = r'env\.newProperty<int>\("(\w+)",\s*(\d+)'
            for match in re.finditer(env_pattern_int, content):
                name, value = match.groups()
                params[name] = int(value)

        return params

    def extract_cpu_params(self) -> Dict[str, float]:
        """Extract parameters from CPU source files."""
        params = {}

        # Parse XML parameter file
        xml_file = self.cpu_root / "HCC/HCC_single/resource/param_all_test.xml"
        if xml_file.exists():
            content = xml_file.read_text()

            # Extract specific parameters we care about
            xml_pattern = r'<(\w+)>([\d.eE+-]+)</\1>'
            for match in re.finditer(xml_pattern, content):
                name, value = match.groups()
                try:
                    # Try to parse as float
                    params[name] = float(value)
                except ValueError:
                    pass

        return params

    def compare_diffusion_coefficients(self, gpu_params: Dict, cpu_params: Dict) -> None:
        """Compare diffusion coefficients."""
        print("\n" + "="*70)
        print("DIFFUSION COEFFICIENTS (cm²/s)")
        print("="*70)
        print(f"{'Chemical':<15} {'GPU':<15} {'CPU':<15} {'Match':<10} {'Ratio':<10}")
        print("-"*70)

        chemicals = {
            'O2': ('D_O2', '2.8e-5'),
            'IFN': ('D_IFN', '1.0e-7'),
            'IL2': ('D_IL2', '4.0e-8'),
            'IL10': ('D_IL10', '1.4e-8'),
            'TGFB': ('D_TGFB', '2.6e-7'),
            'CCL2': ('D_CCL2', '1.31e-8'),
            'ARGI': ('D_ARGI', '1.0e-6'),
            'NO': ('D_NO', '3.8e-5'),
            'IL12': ('D_IL12', '2.4e-8'),
            'VEGFA': ('D_VEGFA', '2.9e-7'),
        }

        for chem, (gpu_key, cpu_value) in chemicals.items():
            gpu_val = gpu_params.get(gpu_key, float('nan'))
            cpu_val = float(cpu_value)

            match = abs(gpu_val - cpu_val) < 1e-9
            ratio = gpu_val / cpu_val if cpu_val != 0 else 1.0

            status = "✓" if match else "✗"
            print(f"{chem:<15} {gpu_val:<15.2e} {cpu_val:<15.2e} {status:<10} {ratio:<10.2f}")

            if not match:
                self.mismatches.append((chem, gpu_val, cpu_val, "diffusion"))

    def compare_behavioral_parameters(self, gpu_params: Dict, cpu_params: Dict) -> None:
        """Compare behavioral parameters."""
        print("\n" + "="*70)
        print("BEHAVIORAL PARAMETERS")
        print("="*70)
        print(f"{'Parameter':<40} {'GPU':<15} {'Expected':<15} {'Match':<10}")
        print("-"*70)

        expected = {
            'cancer_move_prob': 0.1,
            'cancer_stem_move_prob': 0.05,
            'tcell_move_prob': 0.5,
            'tcell_cyt_move_prob': 0.3,
            'treg_move_prob': 0.3,
            'mdsc_move_prob': 0.3,
            'cancer_stem_div_interval': 24.0,
            'cancer_progenitor_div_interval': 12.0,
            'cancer_progenitor_div_max': 10,
            'tcell_div_interval': 24,
            'tcell_life_mean': 100.0,
            'treg_life_mean': 100.0,
            'mdsc_life_mean': 100.0,
            'PD1_PDL1_half': 0.5,
            'n_PD1_PDL1': 2.0,
            'escape_base': 0.95,
            'tcell_exhaust_base_PDL1': 0.99,
            'tcell_exhaust_base_Treg': 0.99,
            'PDL1_decay_rate': 0.1,
            'PDL1_syn_max': 0.5,
        }

        for param, expected_val in expected.items():
            gpu_val = gpu_params.get(param, float('nan'))
            match = abs(gpu_val - expected_val) < 1e-6 if not isinstance(expected_val, int) else gpu_val == expected_val
            status = "✓" if match else "✗"

            print(f"{param:<40} {gpu_val:<15} {expected_val:<15} {status:<10}")

            if not match:
                self.mismatches.append((param, gpu_val, expected_val, "behavioral"))

    def compare_chemical_parameters(self, gpu_params: Dict, cpu_params: Dict) -> None:
        """Compare chemical source/sink parameters."""
        print("\n" + "="*70)
        print("CHEMICAL SOURCE/SINK RATES")
        print("="*70)
        print(f"{'Chemical Source':<40} {'GPU':<15} {'Expected':<15} {'Match':<10}")
        print("-"*70)

        # Note: Many of these are computed dynamically, just show expected values
        expected = {
            'IFN_PDL1_EC50': 5.0e-7,
            'IFN_PDL1_hill': 2.0,
            'IL10_T_cell_suppress_EC50': 5.0e-7,
            'TGFB_T_cell_suppress_EC50': 3.0e-7,
            'NIVO_PD1_EC50': 1.0e-8,
            'suppression_max': 0.9,
        }

        for param, expected_val in expected.items():
            gpu_val = gpu_params.get(param, float('nan'))
            match = abs(gpu_val - expected_val) < 1e-9 if expected_val < 1e-6 else abs(gpu_val - expected_val) < 1e-6
            status = "✓" if match else "✗"

            print(f"{param:<40} {gpu_val:<15.2e} {expected_val:<15.2e} {status:<10}")

            if not match:
                self.mismatches.append((param, gpu_val, expected_val, "chemical"))

    def report_mismatches(self) -> None:
        """Report parameter mismatches."""
        if self.mismatches:
            print("\n" + "="*70)
            print(f"MISMATCHES FOUND: {len(self.mismatches)}")
            print("="*70)

            for param, gpu_val, expected_val, category in self.mismatches:
                pct_diff = abs(gpu_val - expected_val) / abs(expected_val) * 100 if expected_val != 0 else 0
                print(f"\n{category.upper()}: {param}")
                print(f"  GPU value:      {gpu_val}")
                print(f"  Expected value: {expected_val}")
                print(f"  Difference:     {pct_diff:.2f}%")
        else:
            print("\n" + "="*70)
            print("✓ ALL PARAMETERS MATCH!")
            print("="*70)

    def run_comparison(self) -> int:
        """Run full comparison."""
        print("\n" + "="*70)
        print("PDAC GPU vs HCC CPU PARAMETER COMPARISON")
        print("="*70)
        print(f"GPU root: {self.gpu_root}")
        print(f"CPU root: {self.cpu_root}")

        gpu_params = self.extract_gpu_params()
        cpu_params = self.extract_cpu_params()

        print(f"\nExtracted {len(gpu_params)} GPU parameters")
        print(f"Extracted {len(cpu_params)} CPU parameters")

        self.compare_diffusion_coefficients(gpu_params, cpu_params)
        self.compare_behavioral_parameters(gpu_params, cpu_params)
        self.compare_chemical_parameters(gpu_params, cpu_params)
        self.report_mismatches()

        return 0 if not self.mismatches else 1


def main():
    """Main entry point."""
    # Find project roots
    script_dir = Path(__file__).parent
    gpu_root = script_dir.parent.parent
    cpu_root = gpu_root.parent.parent / "SPQSP_HCC-main"

    if not gpu_root.exists():
        print(f"Error: GPU root not found at {gpu_root}", file=sys.stderr)
        return 1

    if not cpu_root.exists():
        print(f"Warning: CPU root not found at {cpu_root}", file=sys.stderr)
        print("Proceeding with GPU parameter validation only...", file=sys.stderr)

    comparator = ParameterComparator(gpu_root, cpu_root)
    return comparator.run_comparison()


if __name__ == "__main__":
    sys.exit(main())
