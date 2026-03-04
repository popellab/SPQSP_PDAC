# QSP Enum-Driven Parameter System

## The Problem (And Solution)

### ❌ Previous Approach (Index-Based)
```cpp
enum QSPParamFloat {
    QSP_V_C = 0,           // Hardcoded index
    QSP_V_P = 1,           // Must match Param.cpp order
    QSP_V_T = 2,           // If you add a param anywhere, ALL indices after shift!
    QSP_V_T_IFNg = 99,     // Must recalculate
    QSP_V_T_IL2 = 100,     // And update all these
};

// Problem: Adding QSP_V_NEWPARAM at position 50 means:
// - Shift QSP_V_T_IFNg from 99 to 100
// - Shift QSP_V_T_IL2 from 100 to 101
// - Update all other indices...
// - ERROR-PRONE! 😱
```

### ✅ New Approach (Enum-Driven)
```cpp
enum QSPParamFloat {
    QSP_V_C,               // Auto-index: 0 (from enum position)
    QSP_V_P,               // Auto-index: 1 (from enum position)
    QSP_V_T,               // Auto-index: 2 (from enum position)
    QSP_V_T_IFNg,          // Auto-index: 9 (from enum position)
    QSP_V_T_IL2,           // Auto-index: 10 (from enum position)
    QSP_NEW_PARAM_ANYWHERE,// Auto-index: 11 (from its enum position, wherever you put it)
    QSP_PARAM_FLOAT_COUNT  // Sentinel: knows total count
};

// The XML path mapping array mirrors enum order:
const char* QSP_PARAM_FLOAT_XML_PATHS[] = {
    "Param.QSP.init_value.Compartment.V_C",      // Index 0 (enum position)
    "Param.QSP.init_value.Compartment.V_P",      // Index 1 (enum position)
    "Param.QSP.init_value.Compartment.V_T",      // Index 2 (enum position)
    "Param.QSP.init_value.Species.V_T_IFNg",     // Index 9 (enum position)
    "Param.QSP.init_value.Species.V_T_IL2",      // Index 10 (enum position)
    "Param.QSP.init_value.Species.V_NEW",        // Index 11 (enum position)
};

// Benefit: Add new parameter ANYWHERE in enum and XML_PATHS array
// Everything else stays the same! ✅
```

## Key Insight: Enum Position = Vector Index

The breakthrough is that **the enum definition order IS the contract for indices**.

```cpp
enum QSPParamFloat {
    QSP_V_C,        // Position 0 in enum → Vector index 0
    QSP_V_P,        // Position 1 in enum → Vector index 1
    QSP_V_T,        // Position 2 in enum → Vector index 2
    //                ...
    QSP_V_T_IFNg,   // Position 9 in enum → Vector index 9
    //                ...
    QSP_PARAM_FLOAT_COUNT  // Tells you total count
};

// When you call:
double ifng = qsp_params.getFloat(QSP_V_T_IFNg);
// → Automatically uses index 9 (the enum position)
```

## How It Works

### 1. Define Parameters Logically

Group parameters by meaning, NOT by Param.cpp order:

```cpp
enum QSPParamFloat {
    // ===== COMPARTMENT VOLUMES =====
    QSP_V_C,        // 0
    QSP_V_P,        // 1
    QSP_V_T,        // 2
    QSP_V_LN,       // 3

    // ===== TUMOR T CELLS =====
    QSP_V_T_T0,     // 4
    QSP_V_T_T1,     // 5
    QSP_V_T_C1,     // 6

    // ===== TUMOR CYTOKINES =====
    QSP_V_T_IFNg,   // 7
    QSP_V_T_IL2,    // 8
    QSP_V_T_IL10,   // 9

    QSP_PARAM_FLOAT_COUNT  // = 10
};
```

### 2. Add Corresponding XML Paths

In **the exact same order**:

```cpp
const char* QSP_PARAM_FLOAT_XML_PATHS[] = {
    // ===== COMPARTMENT VOLUMES =====
    "Param.QSP.init_value.Compartment.V_C",      // Index 0
    "Param.QSP.init_value.Compartment.V_P",      // Index 1
    "Param.QSP.init_value.Compartment.V_T",      // Index 2
    "Param.QSP.init_value.Compartment.V_LN",     // Index 3

    // ===== TUMOR T CELLS =====
    "Param.QSP.init_value.Species.V_T_T0",       // Index 4
    "Param.QSP.init_value.Species.V_T_T1",       // Index 5
    "Param.QSP.init_value.Species.V_T_C1",       // Index 6

    // ===== TUMOR CYTOKINES =====
    "Param.QSP.init_value.Species.V_T_IFNg",     // Index 7
    "Param.QSP.init_value.Species.V_T_IL2",      // Index 8
    "Param.QSP.init_value.Species.V_T_IL10",     // Index 9
};

// IMPORTANT: Array size must equal QSP_PARAM_FLOAT_COUNT
// Static assert checks this at compile-time!
```

### 3. XML Reader Loads by Enum Position

```cpp
void QSPParam::readParamsFromXml(std::string file) {
    for (int enum_idx = 0; enum_idx < QSP_PARAM_FLOAT_COUNT; ++enum_idx) {
        std::string xml_path = QSP_PARAM_FLOAT_XML_PATHS[enum_idx];

        // Read XML value using the path
        double value = ptree.get<double>(xml_path);

        // Store at vector index = enum position (automatic!)
        _paramFloat[enum_idx] = value;
    }
}
```

The XML can be in **any order** - the reader knows where to put each value based on enum position!

## Adding a New Parameter (Simple!)

### Scenario
You want to add a new immunosuppression marker `QSP_V_T_PD1L1`.

### Before (Index-Based - HARD)
1. Find where it fits in Param.cpp
2. Count the index (e.g., 150 + some offset = 247)
3. Add to enum: `QSP_V_T_PD1L1 = 247,`
4. Update ALL subsequent indices:
   - `QSP_V_T_MDSC = 248,` (was 247)
   - `QSP_V_T_ArgI = 249,` (was 248)
   - ... (20+ more updates)
5. Fix any code that assumes hardcoded indices

### After (Enum-Driven - EASY)
1. Add to enum (anywhere that makes logical sense):
   ```cpp
   enum QSPParamFloat {
       // ... existing ...
       QSP_V_T_MDSC,
       QSP_V_T_ArgI,
       QSP_V_T_NO,
       QSP_V_T_PD1L1,     // NEW: Just add it!
       QSP_V_T_CCL2,
       // ... rest stays the same ...
   };
   ```

2. Add corresponding XML path (same position in array):
   ```cpp
   const char* QSP_PARAM_FLOAT_XML_PATHS[] = {
       // ... existing paths ...
       "Param.QSP.init_value.Species.V_T_MDSC",
       "Param.QSP.init_value.Species.V_T_ArgI",
       "Param.QSP.init_value.Species.V_T_NO",
       "Param.QSP.init_value.Species.V_T_PD1L1",  // NEW: Just add XML path!
       "Param.QSP.init_value.Species.V_T_CCL2",
       // ... rest stays the same ...
   };
   ```

3. Use in code:
   ```cpp
   double pd1l1 = qsp_params.getFloat(QSP_V_T_PD1L1);  // Done!
   ```

**That's it!** No index recalculation. No cascading updates. ✨

## Real-World Example: Adding Drug Concentrations

### You want to track NIVO and CABO concentrations in tumor and central compartments

```cpp
// STEP 1: Add to enum
enum QSPParamFloat {
    // ... existing parameters ...

    // ===== TUMOR COMPARTMENT - CYTOKINES =====
    QSP_V_T_IFNg,
    QSP_V_T_IL2,
    QSP_V_T_IL10,

    // ===== TUMOR COMPARTMENT - DRUGS =====  (NEW SECTION)
    QSP_V_T_NIVO,              // Nivolumab (tumor)
    QSP_V_T_CABO,              // Cabozantinib (tumor)
    QSP_V_C_NIVO,              // Nivolumab (central)
    QSP_V_C_CABO,              // Cabozantinib (central)

    QSP_PARAM_FLOAT_COUNT
};
```

```cpp
// STEP 2: Add to XML paths (SAME ORDER as enum)
const char* QSP_PARAM_FLOAT_XML_PATHS[] = {
    // ... existing paths ...

    // ===== TUMOR COMPARTMENT - CYTOKINES =====
    "Param.QSP.init_value.Species.V_T_IFNg",
    "Param.QSP.init_value.Species.V_T_IL2",
    "Param.QSP.init_value.Species.V_T_IL10",

    // ===== TUMOR COMPARTMENT - DRUGS =====  (NEW SECTION)
    "Param.QSP.init_value.Species.V_T_NIVO",
    "Param.QSP.init_value.Species.V_T_CABO",
    "Param.QSP.init_value.Species.V_C_NIVO",
    "Param.QSP.init_value.Species.V_C_CABO",
};
```

```cpp
// STEP 3: Add XML values to param_all_test.xml
<Species>
    <V_T_IFNg>0.00192558</V_T_IFNg>
    <V_T_IL2>100.0</V_T_IL2>
    <V_T_IL10>50.0</V_T_IL10>
    <V_T_NIVO>0.0</V_T_NIVO>        <!-- NEW -->
    <V_T_CABO>0.0</V_T_CABO>        <!-- NEW -->
    <V_C_NIVO>0.0</V_C_NIVO>        <!-- NEW -->
    <V_C_CABO>0.0</V_C_CABO>        <!-- NEW -->
</Species>
```

```cpp
// STEP 4: Use in code
void LymphCentral_wrapper::get_state_for_abm(QSPState& state) {
    state.nivo_tumor = qsp_params.getFloat(QSP_V_T_NIVO);
    state.cabo_tumor = qsp_params.getFloat(QSP_V_T_CABO);
    state.nivo_central = qsp_params.getFloat(QSP_V_C_NIVO);
    state.cabo_central = qsp_params.getFloat(QSP_V_C_CABO);
}
```

**Result**: Everything works automatically. No index conflicts, no recalculations. ✅

## Compile-Time Safety

The `static_assert` ensures your arrays stay synchronized:

```cpp
static_assert(sizeof(QSP_PARAM_FLOAT_XML_PATHS) / sizeof(QSP_PARAM_FLOAT_XML_PATHS[0])
              == QSP_PARAM_FLOAT_COUNT,
              "XML_PATHS array size must match QSP_PARAM_FLOAT_COUNT");
```

If you add an enum value but forget to add an XML path → **Compilation error!** ✅

## Comparison: Old vs New

| Aspect | Index-Based (❌) | Enum-Driven (✅) |
|--------|---|---|
| **Adding parameter** | Recalculate all later indices | Just add to enum + path array |
| **Index visibility** | Hardcoded magic numbers | Transparent (enum position) |
| **Error potential** | High (easy to get indices wrong) | Low (compiler catches mismatches) |
| **Code clarity** | `getVal(247)` - What is 247? | `getFloat(QSP_V_T_PD1L1)` - Clear! |
| **Maintenance** | Fragile - cascading updates | Robust - isolated changes |
| **XML dependency** | Tight coupling to Param.cpp order | Flexible - any XML structure |

## Implementation Checklist

- [ ] Define enum `QSPParamFloat` with no hardcoded indices
- [ ] Create `QSP_PARAM_FLOAT_XML_PATHS` array in **same order**
- [ ] Add `static_assert` to verify array sizes match
- [ ] Implement `readParamsFromXml()` to iterate by enum position
- [ ] Add convenience methods: `getTumorCytokine_IFNg()`, etc.
- [ ] Update XML file with values
- [ ] Use in code: `qsp_params.getFloat(QSP_V_T_IFNg)`
- [ ] Test by adding a new parameter (should be trivial!)

## When You Need to Add a Parameter

1. **Add to enum** (logical section, any position)
2. **Add to XML_PATHS array** (same position as enum)
3. **Add to param_all_test.xml** (if providing initial values)
4. **Compile** - static_assert catches size mismatches
5. **Use** - `getFloat(QSP_NEW_PARAM)` automatically works

That's it! No manual index tracking. The enum order IS the API.

