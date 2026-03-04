# QSP Parameter Enum Usage Guide

## Overview

The new `QSPParam` class provides type-safe, named access to QSP parameters via enums, replacing the need to memorize vector indices.

## Step 1: Finding Parameter Indices from Param.cpp

### How Parameters are Stored

Parameters are added to `_paramDesc` in order in `Param.cpp::setupParam()`. The line number in that sequence maps to the vector index.

**Example from Param.cpp:**
```cpp
void Param::setupParam() {
    // ... initialization code ...

    // Line 1 of _paramDesc entries:
    _paramDesc.push_back({"Param.QSP.init_value.Compartment.V_C", "", "pos"});    // INDEX: 0

    // Line 2:
    _paramDesc.push_back({"Param.QSP.init_value.Compartment.V_P", "", "pos"});    // INDEX: 1

    // Line 3:
    _paramDesc.push_back({"Param.QSP.init_value.Compartment.V_T", "", "pos"});    // INDEX: 2

    // ... more parameters ...

    // Line 150 (example):
    _paramDesc.push_back({"Param.QSP.init_value.Species.V_T_T0", "", "pos"});     // INDEX: 149

    // Line 151:
    _paramDesc.push_back({"Param.QSP.init_value.Species.V_T_T1", "", "pos"});     // INDEX: 150
}
```

### Finding Indices Systematically

1. **Open `/PDAC/qsp/ode/Param.cpp`**
2. **Find the `setupParam()` method**
3. **Count the `_paramDesc.push_back()` calls** for each parameter
4. **The count becomes the index:**

```cpp
// Param.cpp example: Line numbers show order
Line 100: _paramDesc.push_back({"Param.QSP.init_value.Species.V_T_T0", "", ""});
Line 101: _paramDesc.push_back({"Param.QSP.init_value.Species.V_T_T1", "", ""});
Line 102: _paramDesc.push_back({"Param.QSP.init_value.Species.V_T_C1", "", ""});

// So:
// V_T_T0 is at INDEX 99 (100 lines down, but 0-indexed)
// V_T_T1 is at INDEX 100
// V_T_C1 is at INDEX 101
```

### Quick Method: Search and Count

```bash
# Count how many _paramDesc entries exist before your target
grep -n "_paramDesc.push_back" /PDAC/qsp/ode/Param.cpp | grep "V_T_T0"

# This shows line number, then search above it to count entries
# Or use awk to get the count directly
awk '/_paramDesc.push_back.*V_T_T0/ {print NR}' /PDAC/qsp/ode/Param.cpp
```

## Step 2: Building Your Enum

### Template for Adding Parameters

For each parameter you need, add an entry:

```cpp
enum QSPParamFloat {
    // STEP 1: Find XML path in param_all_test.xml
    // STEP 2: Find corresponding _paramDesc entry in Param.cpp
    // STEP 3: Count position (0-indexed) = INDEX
    // STEP 4: Add to enum

    // Format: QSP_<COMPARTMENT>_<SPECIES> = <INDEX>,

    QSP_V_C = 0,           // Central volume - line 100 of Param.cpp
    QSP_V_P = 1,           // Peripheral volume - line 101 of Param.cpp
    QSP_V_T = 2,           // Tumor volume - line 102 of Param.cpp
    QSP_V_LN = 3,          // Lymph node volume - line 103 of Param.cpp

    QSP_V_T_T0 = 99,       // Tumor naive T cells - line 199 of Param.cpp
    QSP_V_T_T1 = 100,      // Tumor activated T cells - line 200 of Param.cpp
    QSP_V_T_C1 = 101,      // Tumor cytotoxic T cells - line 201 of Param.cpp
    QSP_V_T_IFNg = 102,    // Tumor IFN-gamma - line 202 of Param.cpp

    // Continue for all parameters you need...

    QSP_PARAM_FLOAT_COUNT
};
```

### Naming Convention

- `QSP_<COMPARTMENT>_<SPECIES>`
- Compartments: `V_C` (central), `V_P` (peripheral), `V_T` (tumor), `V_LN` (lymph node)
- Species: `T0` (naive T), `T1` (activated T), `C1` (cytotoxic T), `IFNg`, `IL2`, etc.

**Examples:**
```cpp
QSP_V_C_nT0           // Central compartment: naive CD8 T cells
QSP_V_T_T0            // Tumor compartment: naive CD8 T cells
QSP_V_T_IFNg          // Tumor compartment: IFN-gamma concentration
QSP_V_LN_IL2          // Lymph node compartment: IL-2 concentration
QSP_V_P_T0            // Peripheral blood: naive CD8 T cells
```

## Step 3: Using the Enum in Code

### Method 1: Direct Enum Access (Type-Safe)

```cpp
// In LymphCentral_wrapper.cpp or any QSP integration code

// Single parameter access
double tumor_t0 = qsp_params.getFloat(QSP_V_T_T0);
double tumor_ifng = qsp_params.getFloat(QSP_V_T_IFNg);

// Multiple parameters
double naive_t = qsp_params.getFloat(QSP_V_T_T0);
double activated_t = qsp_params.getFloat(QSP_V_T_T1);
double cytotoxic_t = qsp_params.getFloat(QSP_V_T_C1);

// Calculate total tumor T cells
double total_t_cells = naive_t + activated_t + cytotoxic_t;
```

**Benefits:**
- ✅ Compiler checks enum exists
- ✅ IDE autocomplete suggests available parameters
- ✅ No magic numbers
- ✅ Easy to find usages (search for `QSP_V_T_IFNg`)

### Method 2: Named Convenience Methods

```cpp
// For frequently-used parameters, use convenience methods:

double tumor_ifng = qsp_params.getTumorCytokine_IFNg();
double tumor_t_naive = qsp_params.getTumorTCell_Naive();
double tumor_mdsc = qsp_params.getTumorImmune_MDSC();

// Convenience methods are in QSPParam class, e.g.:
// inline double getTumorCytokine_IFNg() const {
//     return getFloat(QSP_V_T_IFNg);
// }
```

**When to use:**
- Use enums for rare/one-time access
- Use convenience methods for frequently-accessed parameters

### Method 3: Backward Compatible (Old Style)

```cpp
// Old code still works via inherited Param class
double val = qsp_params.getVal(99);  // Direct index access (still works)

// But prefer the new enum style:
double val = qsp_params.getFloat(QSP_V_T_T0);  // Much clearer!
```

## Step 4: Integration Examples

### In LymphCentral_wrapper - Getting State for ABM

```cpp
// File: LymphCentral_wrapper.cpp

QSPState LymphCentral_wrapper::get_state_for_abm() const {
    QSPState state;

    // Get current concentrations from QSP model (type-safe enums!)
    state.tumor_ifng_conc = _qsp_params.getFloat(QSP_V_T_IFNg);
    state.tumor_il2_conc = _qsp_params.getFloat(QSP_V_T_IL2);
    state.tumor_il10_conc = _qsp_params.getFloat(QSP_V_T_IL10);
    state.tumor_tgfb_conc = _qsp_params.getFloat(QSP_V_T_TGFb);
    state.tumor_il12_conc = _qsp_params.getFloat(QSP_V_T_IL12);

    // Get T cell populations
    state.tumor_naive_t = _qsp_params.getFloat(QSP_V_T_T0);
    state.tumor_activated_t = _qsp_params.getFloat(QSP_V_T_T1);
    state.tumor_cytotoxic_t = _qsp_params.getFloat(QSP_V_T_C1);

    // Get immunosuppression signals
    state.tumor_mdsc_count = _qsp_params.getFloat(QSP_V_T_MDSC);
    state.tumor_argi_conc = _qsp_params.getFloat(QSP_V_T_ArgI);
    state.tumor_no_conc = _qsp_params.getFloat(QSP_V_T_NO);

    return state;
}
```

### In ABM Feedback - Updating QSP from ABM

```cpp
// File: LymphCentral_wrapper.cpp

void LymphCentral_wrapper::_apply_abm_feedback(const ABMFeedback& feedback) {
    // Example: ABM killed X cancer cells, recruit Y T cells
    // Update QSP state accordingly

    // Current approach: _qsp_model->setSpeciesValue(index, value)
    // New approach with enums:

    // Recruit T cells from lymph node to tumor
    double ln_t0_current = _qsp_params.getFloat(QSP_V_LN_T0);
    double t_recruitment = feedback.recruited_t_cells;

    // This would call your ODE system's species update
    // _qsp_model->setSpecies(QSP_V_LN_T0, ln_t0_current - t_recruitment);
    // _qsp_model->setSpecies(QSP_V_T_T0, tumor_t0_current + t_recruitment);
}
```

## Complete Workflow Example

### Before (Index-Based - Error-Prone)

```cpp
// You have to remember indices!
double ifng = _qsp_model._paramFloat[47];      // Which parameter is 47?
double il2 = _qsp_model._paramFloat[48];       // Are these adjacent?
double tumor_cells = _qsp_model._paramFloat[52]; // What's at 52 again?

// Easy to make mistakes:
double wrong_param = _qsp_model._paramFloat[47];  // Is this IFNg or something else?
// No IDE support, hard to debug
```

### After (Enum-Based - Clear & Safe)

```cpp
// Clear, self-documenting names
double ifng = _qsp_params.getFloat(QSP_V_T_IFNg);
double il2 = _qsp_params.getFloat(QSP_V_T_IL2);
double tumor_cells = _qsp_params.getFloat(QSP_V_T_C1);

// Or even clearer:
double ifng = _qsp_params.getTumorCytokine_IFNg();

// IDE autocomplete helps you find the right parameter
// Easy to search codebase for all uses of this parameter
// Compile-time type checking
```

## Adding New Parameters

### When You Need a New Parameter

1. **Find it in Param.cpp setupParam()** and get its index
2. **Add to QSPParam enum:**
   ```cpp
   enum QSPParamFloat {
       // ... existing parameters ...
       QSP_V_NEW_PARAM = 150,  // NEW: Add index here
   };
   ```
3. **Add convenience method (optional):**
   ```cpp
   inline double getNewParam() const {
       return getFloat(QSP_V_NEW_PARAM);
   }
   ```
4. **Use in code:**
   ```cpp
   double val = qsp_params.getFloat(QSP_V_NEW_PARAM);
   // or
   double val = qsp_params.getNewParam();
   ```

## Summary

| Approach | Pro | Con | Use When |
|----------|-----|-----|----------|
| **Enum (getFloat)** | Type-safe, named, searchable | Need to find indices | Primary method |
| **Convenience method** | Most readable | More code | Frequently-used params |
| **Direct index (getVal)** | Compatible with old code | Error-prone, unclear | Legacy code only |

The enum approach gives you the best of both worlds: the clarity of named parameters with the efficiency of vector indexing!
