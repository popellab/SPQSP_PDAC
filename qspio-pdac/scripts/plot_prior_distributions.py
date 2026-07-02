"""Generate prior distributions figure for presentation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qsp_inference.priors import load_prior

PRIORS_CSV = 'parameters/pdac_priors.csv'
OUTPUT_FILE = 'figures/prior_distributions.png'

# Load prior
prior = load_prior(PRIORS_CSV)
prior_df = pd.read_csv(PRIORS_CSV)

# Select subset of key parameters for presentation (12 max)
KEY_PARAMS = [
    'k_C1_growth', 'k_CD8_pro', 'k_CD8_death', 'k_Treg_pro',
    'k_psc_activation', 'k_ECM_apsc_sec', 'k_APC_mat', 'k_M1_pol',
    'initial_tumour_diameter', 'K_T_Treg', 'n_CD8_clones', 'ECM_50_T_mot'
]

# Get indices of key parameters
param_indices = [i for i, name in enumerate(prior_df['name']) if name in KEY_PARAMS]
param_names_subset = [prior_df['name'].iloc[i] for i in param_indices]

# Sample from prior
n_samples = 10000
prior_samples = prior.sample((n_samples,))

# Create figure
n_params = len(param_indices)
ncols = 4
nrows = (n_params + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
axes = axes.flatten()

for idx, (param_idx, param_name) in enumerate(zip(param_indices, param_names_subset)):
    ax = axes[idx]
    samples = prior_samples[:, param_idx].numpy()

    # Log-spaced bins
    bins = np.logspace(np.log10(samples.min()), np.log10(samples.max()), 40)
    ax.hist(samples, bins=bins, density=True, alpha=0.7, color='steelblue', edgecolor='none')

    # Stats
    median = np.median(samples)
    ax.axvline(median, color='red', linestyle='--', linewidth=2, label=f'Median: {median:.2e}')

    # Bounds
    row = prior_df.iloc[param_idx]
    if pd.notna(row['lower_bound']):
        ax.axvline(row['lower_bound'], color='darkred', linestyle=':', linewidth=1.5, alpha=0.7)
    if pd.notna(row['upper_bound']):
        ax.axvline(row['upper_bound'], color='darkgreen', linestyle=':', linewidth=1.5, alpha=0.7)

    ax.set_xscale('log')
    ax.set_title(param_name.replace('_', ' '), fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, which='both')

# Hide empty subplots
for idx in range(len(param_indices), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Prior Distributions (Log-Normal)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_FILE}")
