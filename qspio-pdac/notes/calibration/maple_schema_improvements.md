# Maple Schema Improvement Ideas

Collected during submodel target extraction sessions. These are not blocking but would improve data integrity.

---

## `derived_arithmetic` input type

**Problem:** `unit_conversion` inputs skip snippet validation, so hallucinated values can hide behind plausible rationales. Common cases:
- Physics conversions: G' (storage modulus) → E (Young's modulus) via `E = 3*G'`
- Arithmetic: "2 ROIs × 2 gels = 4 observations"
- Unit scaling: nmol/mg → µM via tissue density

These are all deterministic and traceable, but the validator can't verify them.

**Proposal:** A `derived_arithmetic` input type that requires:
```yaml
- name: E_stiff_kPa
  value: 51.0
  units: kilopascal
  input_type: derived_arithmetic
  source_inputs: [Gprime_stiff_kPa]  # must reference existing inputs
  formula: "3 * Gprime_stiff_kPa"     # validator can eval and check
  rationale: "E = 2*(1+nu)*G' with nu=0.5 for incompressible PEG hydrogels"
```

The validator could:
1. Check that all `source_inputs` exist as named inputs
2. Evaluate the formula and verify it matches `value`
3. Still require a `rationale` for the derivation logic

This keeps raw paper values as `direct_measurement` (snippet-validated) while making derived values explicitly traceable.

**Workaround (current):** Use `direct_fit` with `unit_conversion` inputs. The rationale documents the formula and the snippet contains the raw value. Acceptable but not machine-verifiable.

**Alternative workaround:** Use `algebraic` forward model type so conversions happen in code. But `algebraic` can't handle per-error-model x-values (each error model entry gets the same `compute` call), making it incompatible with multi-condition dose-response designs.

---

## `direct_fit` with x-value transformation

**Problem:** `direct_fit` evaluates the Hill function directly at the `x_input` value. If the paper reports stiffness in G' but the model uses Young's modulus E, there's no way to do the conversion inside `direct_fit` — you must pre-convert the input.

**Proposal:** Add an optional `x_transform` field to `direct_fit`:
```yaml
forward_model:
  type: direct_fit
  curve: hill
  ec50: "E_50_fib"
  n_hill: "n_fib"
  baseline: "0.0"
  maximum: "YAP_max"
  x_transform: "x * 3.0"  # applied to x_input before Hill evaluation
```

This would let raw G' values stay as `direct_measurement` inputs while the forward model handles the conversion.