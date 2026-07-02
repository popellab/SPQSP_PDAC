"""Run prior predictive coverage tests for baseline and GVAX multiomics scenarios."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

subprocess.run([
    sys.executable, "-m", "pytest",
    "tests/scenarios/test_prior_predictive_coverage.py",
    "-m", "slow", "-v"
], cwd=REPO_ROOT)
