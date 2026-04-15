#!/usr/bin/env python3
"""
abm_param_codegen.py — Generate gpu_param.h, gpu_param.cu, and derived_params.inc
from annotated param_all_test.xml.

The XML is the single source of truth. Parameters have these attributes:
  name="PARAM_XXX"      — FLAMEGPU environment property name (also C++ enum entry)
  type="float|int|bool"  — C++ type
  val="pr|pos|"          — validation rule (pr=positive real, pos=positive int/real)
  units="..."            — optional documentation

Derived parameters (QSP→ABM) are in a <DerivedParams> section:
  <D name="..." type="float" comment="..."><![CDATA[C++ expression]]></D>
  <Local var="..." type="double"><![CDATA[C++ expression]]></Local>
  <Call comment="..."><![CDATA[C++ statement]]></Call>
  <Computed name="..." type="float"><![CDATA[expression using getFloat/getInt]]></Computed>

Usage:
    python abm_param_codegen.py <xml_file> <output_dir>

Generates:
    <output_dir>/gpu_param.h
    <output_dir>/gpu_param.cu
    <output_dir>/derived_params.inc
"""

import xml.etree.ElementTree as ET
import sys
import os
import textwrap
from datetime import datetime


# ─── XML parsing ──────────────────────────────────────────────────────────────

def build_xml_path(elem, parent_map):
    """Build dotted XML path for an element (e.g., 'Param.ABM.Environment.SecPerSlice')."""
    path = []
    while elem is not None:
        path.append(elem.tag)
        elem = parent_map.get(elem)
    path.reverse()
    return '.'.join(path)


def collect_params(root):
    """Walk XML tree and collect all elements with 'name' attribute.
    Returns dict with float/int/bool lists, each entry being a dict with
    keys: name, xml_path, type, val, units.
    """
    params = {'float': [], 'int': [], 'bool': []}
    parent_map = {c: p for p in root.iter() for c in p}
    parent_map[root] = None

    # Skip elements inside DerivedParams
    derived_section = root.find('.//DerivedParams')
    derived_elems = set()
    if derived_section is not None:
        for e in derived_section.iter():
            derived_elems.add(e)

    for elem in root.iter():
        if elem in derived_elems:
            continue
        name = elem.get('name')
        if name is None:
            continue
        ptype = elem.get('type', 'float')
        val = elem.get('val', '')
        units = elem.get('units', '')
        xml_path = build_xml_path(elem, parent_map)

        if ptype not in params:
            print(f"WARNING: unknown type '{ptype}' for {name}", file=sys.stderr)
            continue

        params[ptype].append({
            'name': name,
            'xml_path': xml_path,
            'type': ptype,
            'val': val,
            'units': units,
        })

    return params


def collect_derived(root):
    """Collect <DerivedParams> section entries.
    Returns list of dicts with keys: kind, name/var, type, expr, comment.
    """
    entries = []
    dp = root.find('.//DerivedParams')
    if dp is None:
        return entries

    for elem in dp:
        tag = elem.tag
        text = (elem.text or '').strip()
        comment = elem.get('comment', '')

        if tag == 'D':
            entries.append({
                'kind': 'property',
                'name': elem.get('name', ''),
                'type': elem.get('type', 'float'),
                'expr': text,
                'comment': comment,
            })
        elif tag == 'Local':
            entries.append({
                'kind': 'local',
                'var': elem.get('var', ''),
                'type': elem.get('type', 'double'),
                'expr': text,
                'comment': comment,
            })
        elif tag == 'Call':
            entries.append({
                'kind': 'call',
                'expr': text,
                'comment': comment,
            })
        elif tag == 'Computed':
            entries.append({
                'kind': 'computed',
                'name': elem.get('name', ''),
                'type': elem.get('type', 'float'),
                'expr': text,
                'comment': comment,
            })

    return entries


# ─── Adhesion matrix parsing ─────────────────────────────────────────────────

# Maps XML tag name (lowercase) to C++ StateCounterIdx enum name.
SC_TAG_MAP = {
    'cancer_stem':       'SC_CANCER_STEM',
    'cancer_prog':       'SC_CANCER_PROG',
    'cancer_sen':        'SC_CANCER_SEN',
    'cd8_eff':           'SC_CD8_EFF',
    'cd8_cyt':           'SC_CD8_CYT',
    'cd8_sup':           'SC_CD8_SUP',
    'cd8_naive':         'SC_CD8_NAIVE',
    'th':                'SC_TH',
    'treg':              'SC_TREG',
    'tfh':               'SC_TFH',
    'tcd4_naive':        'SC_TCD4_NAIVE',
    'mdsc':              'SC_MDSC',
    'mac_m1':            'SC_MAC_M1',
    'mac_m2':            'SC_MAC_M2',
    'fib_quiescent':     'SC_FIB_QUIESCENT',
    'fib_mycaf':         'SC_FIB_MYCAF',
    'fib_icaf':          'SC_FIB_ICAF',
    'fib_frc':           'SC_FIB_FRC',
    'vas_tip':           'SC_VAS_TIP',
    'vas_phalanx':       'SC_VAS_PHALANX',
    'vas_collapsed':     'SC_VAS_COLLAPSED',
    'vas_hev':           'SC_VAS_HEV',
    'bcell_naive':       'SC_BCELL_NAIVE',
    'bcell_act':         'SC_BCELL_ACT',
    'bcell_plasma':      'SC_BCELL_PLASMA',
    'dc_cdc1_immature':  'SC_DC_CDC1_IMMATURE',
    'dc_cdc1_mature':    'SC_DC_CDC1_MATURE',
    'dc_cdc2_immature':  'SC_DC_CDC2_IMMATURE',
    'dc_cdc2_mature':    'SC_DC_CDC2_MATURE',
}


def collect_adhesion_matrix(root):
    """Parse <AdhesionMatrix> sparse block.
    Returns list of (row_sc_name, col_sc_name, value_str) tuples.
    """
    entries = []
    am = root.find('.//AdhesionMatrix')
    if am is None:
        return entries

    for row_elem in am:
        row_tag = row_elem.tag
        if row_tag not in SC_TAG_MAP:
            print(f"WARNING: AdhesionMatrix row tag '{row_tag}' not in SC_TAG_MAP", file=sys.stderr)
            continue
        row_sc = SC_TAG_MAP[row_tag]

        for col_elem in row_elem:
            col_tag = col_elem.tag
            if col_tag not in SC_TAG_MAP:
                print(f"WARNING: AdhesionMatrix col tag '{col_tag}' not in SC_TAG_MAP", file=sys.stderr)
                continue
            col_sc = SC_TAG_MAP[col_tag]
            val = (col_elem.text or '0').strip()
            entries.append((row_sc, col_sc, val))

    return entries


# ─── Code generation ──────────────────────────────────────────────────────────

HEADER = "// Auto-generated by abm_param_codegen.py from param_all_test.xml — do not edit manually"


def gen_header(params):
    """Generate gpu_param.h."""
    lines = [
        '#ifndef GPU_PARAM_H',
        '#define GPU_PARAM_H',
        '',
        HEADER,
        '',
        '#include "../core/ParamBase.h"',
        '#include <string>',
        '',
        '// Forward declaration to avoid including CUDA headers in .cpp compilation',
        'namespace flamegpu {',
        '    class EnvironmentDescription;',
        '}',
        '',
        'namespace PDAC {',
        '',
        '// Type-safe enums for parameter access',
    ]

    for ptype, enum_name, count_name in [
        ('float', 'GPUParamFloat', 'GPU_PARAM_FLOAT_COUNT'),
        ('int',   'GPUParamInt',   'GPU_PARAM_INT_COUNT'),
        ('bool',  'GPUParamBool',  'GPU_PARAM_BOOL_COUNT'),
    ]:
        lines.append(f'enum {enum_name} {{')
        for p in params[ptype]:
            lines.append(f'    {p["name"]},')
        lines.append(f'    {count_name}')
        lines.append('};')
        lines.append('')

    lines.extend([
        'class GPUParam : public SP_QSP_IO::ParamBase {',
        'public:',
        '    GPUParam();',
        '    ~GPUParam() {}',
        '',
        '    // Type-safe accessors',
        '    inline float getFloat(GPUParamFloat idx) const {',
        '        if (idx >= GPU_PARAM_FLOAT_COUNT) return 0.0f;',
        '        return static_cast<float>(_paramFloat[idx]);',
        '    }',
        '    inline int getInt(GPUParamInt idx) const {',
        '        if (idx >= GPU_PARAM_INT_COUNT) return 0;',
        '        return _paramInt[idx];',
        '    }',
        '    inline bool getBool(GPUParamBool idx) const {',
        '        if (idx >= GPU_PARAM_BOOL_COUNT) return false;',
        '        return _paramBool[idx];',
        '    }',
        '',
        '    // Populate FLAMEGPU environment from loaded parameters',
        '    void populateFlameGPUEnvironment(flamegpu::EnvironmentDescription& env) const;',
        '',
        'private:',
        '    virtual void setupParam() override;',
        '    virtual void processInternalParams() override;',
        '};',
        '',
        '} // namespace PDAC',
        '#endif',
        '',
    ])

    return '\n'.join(lines)


def gen_source(params, derived):
    """Generate gpu_param.cu."""
    lines = [
        '#include "gpu_param.h"',
        '#include "flamegpu/flamegpu.h"',
        '#include <iostream>',
        '',
        HEADER,
        '',
        'namespace PDAC {',
        '',
        '// XML path mappings - maps index to {XML_path, units, validation_type}',
        '// validation_type: "pr" = positive real, "pos" = positive integer, "" = unconstrained',
        'const char* _gpu_param_description[][3] = {',
    ]

    # Float, then int, then bool — must match enum order
    for ptype in ['float', 'int', 'bool']:
        if params[ptype]:
            lines.append(f'    // {ptype.capitalize()} parameters')
        for p in params[ptype]:
            lines.append(f'    {{"{p["xml_path"]}", "{p["units"]}", "{p["val"]}"}},  // {p["name"]}')

    lines.extend([
        '};',
        '',
        '// Verify that _gpu_param_description has the correct number of entries',
        'static_assert(sizeof(_gpu_param_description) / sizeof(_gpu_param_description[0]) == ',
        '        static_cast<int>(GPU_PARAM_FLOAT_COUNT)+static_cast<int>(GPU_PARAM_INT_COUNT)+static_cast<int>(GPU_PARAM_BOOL_COUNT),',
        '        "GPU parameter description array size mismatch");',
        '',
        'GPUParam::GPUParam() : ParamBase() {',
        '    setupParam();',
        '}',
        '',
        'void GPUParam::setupParam() {',
        '    _paramFloat.resize(GPU_PARAM_FLOAT_COUNT, 0.0);',
        '    _paramInt.resize(GPU_PARAM_INT_COUNT, 0);',
        '    _paramBool.resize(GPU_PARAM_BOOL_COUNT, false);',
        '',
        '    for (size_t i = 0; i < static_cast<int>(GPU_PARAM_FLOAT_COUNT)+',
        '                                static_cast<int>(GPU_PARAM_INT_COUNT)+',
        '                                static_cast<int>(GPU_PARAM_BOOL_COUNT); i++) {',
        '        std::vector<std::string> desc(',
        '            _gpu_param_description[i],',
        '            _gpu_param_description[i] + 3);',
        '        _paramDesc.push_back(desc);',
        '    }',
        '}',
        '',
        'void GPUParam::processInternalParams() {',
        '    // ParamBase handles validation based on the "pr" and "pos" tags',
        '}',
        '',
    ])

    # populateFlameGPUEnvironment
    lines.append('void GPUParam::populateFlameGPUEnvironment(flamegpu::EnvironmentDescription& env) const {')

    for ptype, getter in [('float', 'getFloat'), ('int', 'getInt'), ('bool', 'getBool')]:
        if params[ptype]:
            cpp_type = ptype
            for p in params[ptype]:
                lines.append(f'    env.newProperty<{cpp_type}>("{p["name"]}", {getter}({p["name"]}));')
            lines.append('')

    # Computed params from DerivedParams section
    computed = [d for d in derived if d['kind'] == 'computed']
    if computed:
        lines.append('    // Computed properties')
        for c in computed:
            if c['comment']:
                lines.append(f'    // {c["comment"]}')
            lines.append(f'    env.newProperty<{c["type"]}>("{c["name"]}", {c["expr"]});')
        lines.append('')

    lines.extend([
        '}',
        '',
        '} // namespace PDAC',
        '',
    ])

    return '\n'.join(lines)


def gen_derived_inc(derived, adh_matrix_entries=None):
    """Generate derived_params.inc for #include in model_functions.cu."""
    lines = [
        HEADER,
        '// This file is #included in model_functions.cu inside set_internal_params()',
        '// It requires: QP() macro, AVOGADROS, PI, SEC_PER_DAY constants,',
        '//              flamegpu::EnvironmentDescription env in scope.',
        '',
    ]

    for d in derived:
        kind = d['kind']

        if kind == 'computed':
            # Computed params are emitted in populateFlameGPUEnvironment, skip here
            continue

        if kind == 'local':
            if d.get('comment'):
                lines.append(f'    // {d["comment"]}')
            lines.append(f'    {d["type"]} {d["var"]} = {d["expr"]};')
            lines.append('')

        elif kind == 'property':
            if d.get('comment'):
                lines.append(f'    // {d["comment"]}')
            lines.append(f'    env.newProperty<{d["type"]}>("{d["name"]}", ')
            lines.append(f'                    {d["expr"]});')
            lines.append('')

        elif kind == 'call':
            if d.get('comment'):
                lines.append(f'    // {d["comment"]}')
            lines.append(f'    {d["expr"]};')
            lines.append('')

    # ── Adhesion matrix device array initialization ──
    if adh_matrix_entries:
        lines.append('    // ── Adhesion matrix (cell-cell, sparse from XML <AdhesionMatrix>) ──')
        lines.append('    {')
        lines.append('        constexpr int N = ABM_STATE_COUNTER_SIZE;')
        lines.append('        float h_adh_matrix[N * N] = {0};')
        for row_sc, col_sc, val in adh_matrix_entries:
            lines.append(f'        h_adh_matrix[{row_sc} * N + {col_sc}] = {val}f;')
        lines.append('        float* d_adh_matrix = nullptr;')
        lines.append('        cudaMalloc(&d_adh_matrix, N * N * sizeof(float));')
        lines.append('        cudaMemcpy(d_adh_matrix, h_adh_matrix, N * N * sizeof(float), cudaMemcpyHostToDevice);')
        lines.append('        env.newProperty<unsigned long long>("adh_matrix_ptr",')
        lines.append('            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_adh_matrix)));')
        lines.append('    }')
        lines.append('')
    else:
        # No adhesion matrix in XML — allocate zeroed array (no adhesion)
        lines.append('    // ── Adhesion matrix (no <AdhesionMatrix> found, all zeros) ──')
        lines.append('    {')
        lines.append('        constexpr int N = ABM_STATE_COUNTER_SIZE;')
        lines.append('        float* d_adh_matrix = nullptr;')
        lines.append('        cudaMalloc(&d_adh_matrix, N * N * sizeof(float));')
        lines.append('        cudaMemset(d_adh_matrix, 0, N * N * sizeof(float));')
        lines.append('        env.newProperty<unsigned long long>("adh_matrix_ptr",')
        lines.append('            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_adh_matrix)));')
        lines.append('    }')
        lines.append('')

    return '\n'.join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <xml_file> <output_dir>")
        sys.exit(1)

    xml_file = sys.argv[1]
    output_dir = sys.argv[2]

    tree = ET.parse(xml_file)
    root = tree.getroot()

    params = collect_params(root)
    derived = collect_derived(root)
    adh_matrix = collect_adhesion_matrix(root)

    nf = len(params['float'])
    ni = len(params['int'])
    nb = len(params['bool'])
    nd = len([d for d in derived if d['kind'] == 'property'])
    nl = len([d for d in derived if d['kind'] == 'local'])
    nc = len([d for d in derived if d['kind'] == 'call'])
    ncomp = len([d for d in derived if d['kind'] == 'computed'])

    print(f"Parsed: {nf} float, {ni} int, {nb} bool params")
    print(f"Derived: {nd} properties, {nl} locals, {nc} calls, {ncomp} computed")
    print(f"Adhesion matrix: {len(adh_matrix)} non-zero entries")

    os.makedirs(output_dir, exist_ok=True)

    # Generate gpu_param.h
    h_path = os.path.join(output_dir, 'gpu_param.h')
    with open(h_path, 'w') as f:
        f.write(gen_header(params))
    print(f"Wrote {h_path}")

    # Generate gpu_param.cu
    cu_path = os.path.join(output_dir, 'gpu_param.cu')
    with open(cu_path, 'w') as f:
        f.write(gen_source(params, derived))
    print(f"Wrote {cu_path}")

    # Generate derived_params.inc
    inc_path = os.path.join(output_dir, 'derived_params.inc')
    with open(inc_path, 'w') as f:
        f.write(gen_derived_inc(derived, adh_matrix))
    print(f"Wrote {inc_path}")


if __name__ == '__main__':
    main()
