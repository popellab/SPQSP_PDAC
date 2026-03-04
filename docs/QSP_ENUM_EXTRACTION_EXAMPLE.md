# QSP Parameter Enum - Extraction Walkthrough

## Step-by-Step Example: Finding Indices for Tumor Cytokines

### Scenario
You want to add these parameters to your enum:
- Tumor IFN-gamma concentration (V_T_IFNg)
- Tumor IL-2 concentration (V_T_IL2)
- Tumor IL-10 concentration (V_T_IL10)

### Step 1: Open Param.cpp and Find setupParam()

File: `/PDAC/qsp/ode/Param.cpp`

Search for: `void Param::setupParam()`

### Step 2: Locate the Parameter Entries

Look for entries matching your parameter names. They'll look like:

```cpp
void Param::setupParam() {
    // ... other setup code ...

    // Search for "V_T_IFNg", "V_T_IL2", etc.
    _paramDesc.push_back({"Param.QSP.init_value.Species.V_T_IFNg", "molecules", "pos"});
    _paramDesc.push_back({"Param.QSP.init_value.Species.V_T_IL2", "molecules", "pos"});
    _paramDesc.push_back({"Param.QSP.init_value.Species.V_T_IL10", "molecules", "pos"});
}
```

### Step 3: Count the Line Position

The order of `_paramDesc.push_back()` calls determines the index.

**Method A: Manual Counting**
```cpp
// Assuming setupParam() starts at line 50, with these entries:

Line 50: setupParam() {
Line 51:     _paramDesc.push_back(...); // INDEX 0
Line 52:     _paramDesc.push_back(...); // INDEX 1
...
Line 150: _paramDesc.push_back({"V_T_IFNg", ...});  // INDEX 99
Line 151: _paramDesc.push_back({"V_T_IL2", ...});   // INDEX 100
Line 152: _paramDesc.push_back({"V_T_IL10", ...});  // INDEX 101
```

**Method B: Using grep + awk**
```bash
# Find which line Param.cpp has the V_T_IFNg entry
grep -n "V_T_IFNg" /PDAC/qsp/ode/Param.cpp

# Output might be: 150:    _paramDesc.push_back({"Param.QSP.init_value.Species.V_T_IFNg", ...

# Now count how many _paramDesc entries exist up to line 150
awk 'NR <= 150 && /_paramDesc.push_back/ {count++} END {print count-1}' /PDAC/qsp/ode/Param.cpp
# The count-1 gives you the 0-indexed position
```

### Step 4: Extract All Relevant Parameters

Create a script or manually extract:

```bash
# Find all parameters we need for ABM coupling
grep -n "_paramDesc.push_back" /PDAC/qsp/ode/Param.cpp | \
  grep -E "V_T_IFNg|V_T_IL2|V_T_IL10|V_T_TGFB|V_T_MDSC|V_T_C1"

# Output (example):
# 150: {"Param.QSP.init_value.Species.V_T_C1", ...}
# 151: {"Param.QSP.init_value.Species.V_T_IFNg", ...}
# 152: {"Param.QSP.init_value.Species.V_T_IL2", ...}
# 153: {"Param.QSP.init_value.Species.V_T_IL10", ...}
# 154: {"Param.QSP.init_value.Species.V_T_TGFB", ...}
# 155: {"Param.QSP.init_value.Species.V_T_MDSC", ...}
```

### Step 5: Fill in Your Enum

Based on the grep output, you know that:
- V_T_C1 is at line 150 → INDEX = 149 (0-indexed)
- V_T_IFNg is at line 151 → INDEX = 150
- V_T_IL2 is at line 152 → INDEX = 151
- V_T_IL10 is at line 153 → INDEX = 152
- V_T_TGFB is at line 154 → INDEX = 153
- V_T_MDSC is at line 155 → INDEX = 154

Now update `QSPParam.h`:

```cpp
enum QSPParamFloat {
    // Tumor T Cell Species
    QSP_V_T_C1 = 149,      // Tumor cytotoxic CD8 T cells
    QSP_V_T_IFNg = 150,    // Tumor IFN-gamma concentration
    QSP_V_T_IL2 = 151,     // Tumor IL-2 concentration
    QSP_V_T_IL10 = 152,    // Tumor IL-10 concentration
    QSP_V_T_TGFB = 153,    // Tumor TGF-beta concentration
    QSP_V_T_MDSC = 154,    // Tumor MDSC count

    QSP_PARAM_FLOAT_COUNT
};
```

## Practical: Complete Parameter Discovery Script

### Bash Script to Extract All Tumor Parameters

```bash
#!/bin/bash
# File: extract_qsp_params.sh

PARAM_FILE="/PDAC/qsp/ode/Param.cpp"

echo "=== QSP Parameter Extraction ==="
echo ""
echo "Searching for tumor compartment parameters (V_T_*):"
echo ""

# Find all V_T_* parameters with their line numbers
grep -n "_paramDesc.push_back.*V_T_" "$PARAM_FILE" | while read line; do
    # Extract line number and parameter name
    line_num=$(echo "$line" | cut -d: -f1)
    param_name=$(echo "$line" | grep -oE "V_T_[A-Za-z0-9]*" | head -1)

    # Calculate 0-indexed position (rough estimate)
    # Need to count all _paramDesc entries before this line
    entries_before=$(awk -v target="$line_num" 'NR < target && /_paramDesc.push_back/ {count++} END {print count}' "$PARAM_FILE")

    # Index is entries_before - 1 (because we're 0-indexed)
    index=$((entries_before - 1))

    echo "QSP_V_T_$param_name = $index,    // Line $line_num"
done
```

**Run it:**
```bash
chmod +x extract_qsp_params.sh
./extract_qsp_params.sh > /tmp/qsp_params.txt

# Review output
cat /tmp/qsp_params.txt
```

## Real-World Example: Complete Enum for ABM Coupling

Based on typical QSP model structure, here's what you might extract:

```cpp
enum QSPParamFloat {
    // ===== COMPARTMENT VOLUMES =====
    QSP_V_C = 0,                // Central compartment
    QSP_V_P = 1,                // Peripheral compartment
    QSP_V_T = 2,                // Tumor compartment
    QSP_V_LN = 3,               // Lymph node compartment

    // ===== TUMOR COMPARTMENT - T CELLS =====
    QSP_V_T_T0 = 50,            // Naive CD8 T cells in tumor
    QSP_V_T_T1 = 51,            // Activated CD8 T cells in tumor
    QSP_V_T_C1 = 52,            // Cytotoxic CD8 T cells in tumor

    // ===== TUMOR COMPARTMENT - CYTOKINES =====
    QSP_V_T_IFNg = 100,         // IFN-gamma concentration
    QSP_V_T_IL2 = 101,          // IL-2 concentration
    QSP_V_T_IL10 = 102,         // IL-10 concentration
    QSP_V_T_TGFb = 103,         // TGF-beta concentration
    QSP_V_T_IL12 = 104,         // IL-12 concentration

    // ===== TUMOR COMPARTMENT - IMMUNOSUPPRESSION =====
    QSP_V_T_MDSC = 110,         // Myeloid-derived suppressor cells
    QSP_V_T_ArgI = 111,         // Arginase-I concentration
    QSP_V_T_NO = 112,           // Nitric oxide concentration
    QSP_V_T_CCL2 = 113,         // CCL2 chemokine

    // ===== LYMPH NODE COMPARTMENT =====
    QSP_V_LN_T0 = 200,          // Naive CD8 T cells
    QSP_V_LN_T1 = 201,          // Activated CD8 T cells
    QSP_V_LN_IL2 = 202,         // IL-2 concentration

    // ===== PERIPHERAL COMPARTMENT =====
    QSP_V_P_T0 = 300,           // Naive CD8 T cells
    QSP_V_P_T1 = 301,           // Activated CD8 T cells

    QSP_PARAM_FLOAT_COUNT
};
```

## Using Your Extracted Enum

Once you've filled in the indices, you can use them like:

```cpp
// In LymphCentral_wrapper.cpp

void LymphCentral_wrapper::initialize(const std::string& param_file) {
    // Load QSP parameters from XML
    _qsp_params.initializeParams(param_file);

    // Now you can access any parameter by name!
    double tumor_vol = _qsp_params.getFloat(QSP_V_T);
    double tumor_ifng = _qsp_params.getFloat(QSP_V_T_IFNg);
    double tumor_il2 = _qsp_params.getFloat(QSP_V_T_IL2);
}

QSPState LymphCentral_wrapper::get_state_for_abm() {
    QSPState state;

    // Highly readable, type-safe access
    state.tumor_ifng = _qsp_params.getFloat(QSP_V_T_IFNg);
    state.tumor_il2 = _qsp_params.getFloat(QSP_V_T_IL2);
    state.tumor_il10 = _qsp_params.getFloat(QSP_V_T_IL10);
    state.tumor_t_cytotoxic = _qsp_params.getFloat(QSP_V_T_C1);
    state.tumor_mdsc = _qsp_params.getFloat(QSP_V_T_MDSC);

    return state;
}
```

## Validation: Verify Your Indices

After you've extracted the indices, verify them:

```bash
# For each parameter, run:
grep -A1 -B1 "V_T_IFNg" /PDAC/qsp/ode/Param.cpp

# Count preceding entries manually to confirm your index is correct
```

Or add a debug print in code:

```cpp
// In LymphCentral_wrapper.cpp initialization
void LymphCentral_wrapper::initialize(const std::string& param_file) {
    _qsp_params.initializeParams(param_file);

    // Debug: Verify indices
    std::cout << "QSP V_T_IFNg = " << _qsp_params.getFloat(QSP_V_T_IFNg) << std::endl;
    std::cout << "QSP V_T_IL2 = " << _qsp_params.getFloat(QSP_V_T_IL2) << std::endl;
    // Should print reasonable values, not 0 or garbage
}
```

## Summary

1. **Find parameters** in Param.cpp setupParam() using grep
2. **Count positions** (0-indexed) from line numbers
3. **Fill enum** with name = index pairs
4. **Use throughout code** for type-safe access
5. **Verify** that values are reasonable when accessed

This is a one-time setup cost with huge long-term benefits!
