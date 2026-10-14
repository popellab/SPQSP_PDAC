#!/usr/bin/env zsh
# Generate diverse LLM expert elicitations using persona-based prompting.
# Each persona gets the same prompt template but with a different expertise framing.
# Starts at v3 since v1 and v2 were generated with the generic prompt.

set -euo pipefail

PROMPT_TEMPLATE="notes/inference/llm_elicitation/llm_elicitation_prompt_personas.md"
OUTDIR="notes/inference/llm_elicitation"

PERSONAS=(
  "a senior GI pathologist with 20 years of experience reading PDAC resections. You have personally reviewed thousands of H&E and IHC-stained slides from pancreatic tumor specimens. Your estimates are grounded in what you actually see under the microscope — cell morphology, staining patterns, and spatial distributions in tissue sections."
  "a tumor immunologist who runs a wet lab studying T cell biology in pancreatic cancer. You focus on flow cytometry and single-cell RNA-seq of tumor-infiltrating lymphocytes. You think in terms of immune cell phenotypes, activation states, and exhaustion markers. You tend to see slightly higher immune infiltration than pathologists because your dissociation protocols capture cells that are hard to see on tissue sections."
  "a stromal biologist specializing in cancer-associated fibroblasts and the tumor extracellular matrix. You have deep expertise in CAF subtypes (iCAF, myCAF, apCAF), stellate cell activation, and collagen remodeling. You tend to emphasize the dominance of the stromal compartment and may estimate higher stromal fractions and ECM levels than others."
  "a myeloid cell biologist who studies tumor-associated macrophages and MDSCs. You are an expert on macrophage polarization, myeloid-derived suppressor cell biology, and innate immune evasion in PDAC. You tend to estimate higher myeloid cell densities than T cell-focused researchers because you know these cells are often underappreciated."
  "a clinical oncologist who treats PDAC patients and runs translational clinical trials. Your perspective is shaped by clinical outcomes and correlative tissue studies from trial biopsies. You think about what predicts response vs resistance and tend to be more conservative in your estimates, reflecting the heterogeneity you see across patients."
  "a computational systems biologist who builds mathematical models of the tumor microenvironment. You think quantitatively about cell densities, turnover rates, and steady-state balances. You tend to anchor on mass-balance constraints and known physiological reference values, then derive other quantities from those anchors."
  "a comparative oncologist who studies PDAC across species (mouse models, organoids, human tissue). You are keenly aware of differences between mouse and human PDAC and tend to adjust your human estimates based on known cross-species scaling factors. You may give slightly different estimates than pure human-tissue researchers."
  "a single-cell genomics researcher who has profiled the PDAC tumor microenvironment using scRNA-seq and spatial transcriptomics (MERFISH, Visium). You think in terms of cell-type proportions from clustering analyses and gene expression signatures. You tend to detect more rare cell populations (like cDC1s and apCAFs) than IHC-based studies."
)

for i in {1..${#PERSONAS[@]}}; do
  VERSION=$((i + 2))  # v3 through v10
  PERSONA="${PERSONAS[$i]}"
  OUTFILE="${OUTDIR}/llm_elicited_targets_v${VERSION}.csv"

  echo "=== Round $i / ${#PERSONAS[@]} (v${VERSION}) ==="
  PROMPT=$(sed "s|{PERSONA}|${PERSONA}|g" "$PROMPT_TEMPLATE")

  echo "Generating $OUTFILE ..."
  FULL_OUTPUT=$(claude --allowedTools "" --print "$PROMPT")

  # Extract CSV block (from header line to end)
  echo "$FULL_OUTPUT" | sed -n '/^observable,estimate/,$p' | sed '/^$/d' > "$OUTFILE"

  # Validate we got data
  NLINES=$(wc -l < "$OUTFILE")
  echo "  -> $NLINES lines (expected ~28)"
  if [ "$NLINES" -lt 5 ]; then
    echo "  WARNING: Too few lines, output may have failed"
  fi
  echo ""
done

echo "=== Aggregating all elicitations ==="
.venv/bin/python scripts/aggregate_llm_elicitations.py

echo "Done!"