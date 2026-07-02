#!/bin/bash
# Generate 10 LLM expert elicitations for treatment scenario observables.
# Uses the personas prompt for step-by-step reasoning, then extracts the CSV block.
#
# Usage: bash scripts/generate_treatment_elicitations.sh

set -e

PROMPT_FILE="notes/inference/llm_elicitation/llm_elicitation_prompt_treatment_personas.md"
OUTPUT_DIR="notes/inference/llm_elicitation"

PERSONAS=(
  "a senior pancreatic surgical oncologist who has operated on over 500 PDAC patients and routinely reviews their pathology slides and mIHC panels"
  "an immunologist specializing in tumor immunology and checkpoint immunotherapy clinical trials, with deep expertise in PD-1 blockade mechanisms"
  "a computational systems biologist who builds mechanistic models of the tumor microenvironment and thinks in terms of cell counts, rates, and mass balance"
  "a pathologist specializing in gastrointestinal cancers who reads hundreds of PDAC resection specimens per year and scores IHC panels"
  "a translational researcher focused on cancer vaccines (including GVAX), with expertise in antigen presentation, T cell priming, and tertiary lymphoid structures"
  "a clinical trialist who has designed and run multiple neoadjuvant immunotherapy trials in PDAC, with deep familiarity with treatment response patterns"
  "a tumor immunologist focused on myeloid cell biology in PDAC — macrophage polarization, MDSCs, and how immunotherapy shifts the myeloid compartment"
  "a quantitative pharmacologist who models drug PK/PD and immune checkpoint receptor occupancy, thinking about dose-response and timing of immune effects"
  "a cancer biologist specializing in the PDAC tumor microenvironment, particularly cancer-associated fibroblasts, desmoplasia, and how stroma modulates immune infiltration"
  "a biostatistician who designs and analyzes correlative studies in immunotherapy trials, with strong intuitions about effect sizes and inter-patient variability in mIHC data"
)

PROMPT_TEMPLATE=$(cat "$PROMPT_FILE")

N_EXPERTS=${1:-10}  # Pass count as first arg, default 10

for i in $(seq 1 $N_EXPERTS); do
  PERSONA="${PERSONAS[$((i-1))]}"
  PROMPT="${PROMPT_TEMPLATE/\{PERSONA\}/$PERSONA}"
  OUTPUT="$OUTPUT_DIR/llm_elicited_treatment_v${i}.csv"

  echo "Expert $i: ${PERSONA:0:80}..."

  # Use claude CLI to generate, extract CSV block, strip code fences
  claude --allowedTools "" --print "$PROMPT" 2>/dev/null | \
    awk '/^observable,/{found=1} found{print}' | \
    grep -v '^```' | \
    head -20 > "$OUTPUT"

  n_lines=$(wc -l < "$OUTPUT" | tr -d ' ')
  echo "  → $OUTPUT ($n_lines lines)"

  if [ "$n_lines" -lt 2 ]; then
    echo "  WARNING: Fewer than 2 lines — CSV extraction may have failed"
  fi
done

echo ""
echo "Done. Run: python scripts/aggregate_llm_elicitations.py"
echo "to combine baseline + treatment into consensus."
