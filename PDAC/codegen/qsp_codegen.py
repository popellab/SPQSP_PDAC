#!/usr/bin/env python3
"""
qsp_codegen.py — Generate C++ CVODE ODE code from SBML.

Reads:
  PDAC_QSP/PDAC_model.sbml — SimBiology SBML Level 2 v4 export

Generates (into PDAC/qsp/ode/):
  QSP_enum.h     — species enum, parameter enum, QSP file param enum
  ODE_system.h   — class declaration (CVODEBase interface)
  ODE_system.cpp — RHS function, setup_class_parameters, init assignments
  QSPParam.h     — parameter reader declaration
  QSPParam.cpp   — parameter reader implementation

Usage:
  python3 PDAC/codegen/qsp_codegen.py [path/to/model.sbml]

Default SBML path: PDAC_QSP/PDAC_model.sbml
Run only when QSP model structure changes, not for parameter value tweaks.
"""

import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_SBML = os.path.join(REPO_ROOT, "PDAC_QSP", "PDAC_model.sbml")
OUTPUT_DIR = os.path.join(REPO_ROOT, "PDAC", "qsp", "ode")

SBML_NS = "{http://www.sbml.org/sbml/level2/version4}"
MATH_NS = "{http://www.w3.org/1998/Math/MathML}"


# =========================================================================
# SBML Parser
# =========================================================================

class SBMLModel:
    """Parse an SBML Level 2 v4 file into code-generation-ready structures."""

    def __init__(self, sbml_path: str):
        tree = ET.parse(sbml_path)
        root = tree.getroot()
        self.model = root.find(f"{SBML_NS}model")
        if self.model is None:
            raise ValueError("No <model> element found in SBML")

        # ID → name lookup (covers compartments, species, parameters, rules)
        self.id_to_name: Dict[str, str] = {}
        self.id_to_comp: Dict[str, str] = {}   # species id → compartment name
        self.name_to_id: Dict[str, str] = {}

        self.compartments: List[dict] = []
        self.species: List[dict] = []
        self.parameters: List[dict] = []
        self.reactions: List[dict] = []
        self.assignment_rules: List[dict] = []    # repeatedAssignment
        self.initial_assignments: List[dict] = []
        self.unit_defs: Dict[str, float] = {}     # unit_id → SI factor

        self._parse_units()
        self._parse_compartments()
        self._parse_parameters()  # before species (species may ref param units)
        self._parse_species()
        self._parse_reactions()
        self._parse_rules()
        self._parse_initial_assignments()

    # --- Units -----------------------------------------------------------

    def _parse_units(self):
        """Parse <listOfUnitDefinitions> → SI conversion factor per unit ID."""
        for ud in self.model.findall(f".//{SBML_NS}unitDefinition"):
            uid = ud.get("id")
            factor = 1.0
            for u in ud.findall(f".//{SBML_NS}unit"):
                kind = u.get("kind", "dimensionless")
                exp = float(u.get("exponent", "1"))
                scale = int(u.get("scale", "0"))
                mult = float(u.get("multiplier", "1"))
                # SI base factor for this unit component:
                #   value_SI = value_model * (mult * 10^scale * base_SI)^exp
                # For SBML L2: base_SI for 'metre'=1, 'second'=1, 'mole'=1,
                #   'gram'=1e-3 (kg), 'dimensionless'=1
                base_si = {"metre": 1.0, "second": 1.0, "mole": 1.0,
                           "kilogram": 1.0, "gram": 1e-3,
                           "dimensionless": 1.0, "item": 1.0 / 6.02214076e23,
                           "litre": 1e-3, "liter": 1e-3}
                base = base_si.get(kind, 1.0)
                component = (mult * (10.0 ** scale) * base) ** exp
                factor *= component
            self.unit_defs[uid] = factor

        # Built-in units always available
        self.unit_defs.setdefault("dimensionless", 1.0)
        self.unit_defs.setdefault("substance", 1.0)
        self.unit_defs.setdefault("volume", 1.0)
        self.unit_defs.setdefault("time", 1.0)

    # Fallback SI factors for bare unit names not in listOfUnitDefinitions
    _BARE_SI = {"litre": 1e-3, "liter": 1e-3, "metre": 1.0, "meter": 1.0,
                "second": 1.0, "kilogram": 1.0, "gram": 1e-3,
                "mole": 1.0, "item": 1.0 / 6.02214076e23,
                "dimensionless": 1.0}

    def get_si_factor(self, unit_id: str) -> float:
        """Get SI conversion factor for a unit ID. Returns 1.0 if unknown."""
        if not unit_id:
            return 1.0
        factor = self.unit_defs.get(unit_id)
        if factor is not None:
            return factor
        factor = self._BARE_SI.get(unit_id)
        if factor is not None:
            return factor
        print(f"  WARNING: unknown unit '{unit_id}', assuming factor=1",
              file=sys.stderr)
        return 1.0

    # --- Compartments ----------------------------------------------------

    def _parse_compartments(self):
        for c in self.model.findall(f".//{SBML_NS}compartment"):
            cid = c.get("id")
            name = c.get("name")
            self.id_to_name[cid] = name
            self.name_to_id[name] = cid
            self.compartments.append({
                "id": cid,
                "name": name,
                "size": float(c.get("size", "1")),
                "spatial_dim": int(c.get("spatialDimensions", "3")),
                "units": c.get("units", ""),
            })

    # --- Parameters ------------------------------------------------------

    def _parse_parameters(self):
        for p in self.model.findall(
                f"{SBML_NS}listOfParameters/{SBML_NS}parameter"):
            pid = p.get("id")
            name = p.get("name")
            self.id_to_name[pid] = name
            self.name_to_id[name] = pid
            self.parameters.append({
                "id": pid,
                "name": name,
                "value": float(p.get("value", "0")),
                "units": p.get("units", ""),
                "constant": p.get("constant", "true") == "true",
            })

    # --- Species ---------------------------------------------------------

    def _parse_species(self):
        comp_id_to_name = {c["id"]: c["name"] for c in self.compartments}
        for sp in self.model.findall(f".//{SBML_NS}species"):
            sid = sp.get("id")
            name = sp.get("name")
            comp_id = sp.get("compartment")
            comp_name = comp_id_to_name.get(comp_id, comp_id)
            full_name = f"{comp_name}.{name}"

            self.id_to_name[sid] = full_name
            self.id_to_comp[sid] = comp_name
            self.name_to_id[full_name] = sid

            init = sp.get("initialAmount", sp.get("initialConcentration", "0"))
            self.species.append({
                "id": sid,
                "name": full_name,
                "base_name": name,
                "compartment": comp_name,
                "initial_value": float(init),
                "units": sp.get("substanceUnits", ""),
                "has_only_substance_units":
                    sp.get("hasOnlySubstanceUnits", "false") == "true",
            })

    # --- Reactions -------------------------------------------------------

    def _parse_reactions(self):
        for rxn in self.model.findall(f".//{SBML_NS}reaction"):
            rl = rxn.find(f"{SBML_NS}listOfReactants")
            pl = rxn.find(f"{SBML_NS}listOfProducts")

            reactant_ids = [sr.get("species")
                            for sr in (rl if rl is not None else [])]
            product_ids = [sr.get("species")
                           for sr in (pl if pl is not None else [])]

            # Parse MathML rate law
            kl = rxn.find(f"{SBML_NS}kineticLaw")
            rate_expr = ""
            if kl is not None:
                math = kl.find(f"{MATH_NS}math")
                if math is not None:
                    rate_expr = self._mathml_to_infix(math)

            self.reactions.append({
                "name": rxn.get("name", f"rxn_{len(self.reactions)}"),
                "reactant_ids": reactant_ids,
                "product_ids": product_ids,
                "reactant_names": [self.id_to_name.get(r, r) for r in reactant_ids],
                "product_names": [self.id_to_name.get(p, p) for p in product_ids],
                "rate_law": rate_expr,
            })

    # --- Assignment Rules -----------------------------------------------

    def _parse_rules(self):
        for ar in self.model.findall(f".//{SBML_NS}assignmentRule"):
            var_id = ar.get("variable")
            var_name = self.id_to_name.get(var_id, var_id)
            math = ar.find(f"{MATH_NS}math")
            expr = self._mathml_to_infix(math) if math is not None else "0"
            self.assignment_rules.append({
                "variable_id": var_id,
                "variable_name": var_name,
                "expression": expr,
            })

    def _parse_initial_assignments(self):
        for ia in self.model.findall(f".//{SBML_NS}initialAssignment"):
            sym_id = ia.get("symbol")
            sym_name = self.id_to_name.get(sym_id, sym_id)
            math = ia.find(f"{MATH_NS}math")
            expr = self._mathml_to_infix(math) if math is not None else "0"
            self.initial_assignments.append({
                "variable_id": sym_id,
                "variable_name": sym_name,
                "expression": expr,
            })

    # --- MathML → Infix Converter ----------------------------------------

    def _mathml_to_infix(self, node) -> str:
        """Convert a MathML <math> or <apply> tree to infix C++ string.

        Resolves SBML IDs to human-readable names.
        """
        tag = node.tag.replace(MATH_NS, "")

        if tag == "math":
            children = list(node)
            if len(children) == 1:
                return self._mathml_to_infix(children[0])
            # Multiple children at top level — shouldn't happen, join with ;
            return "; ".join(self._mathml_to_infix(c) for c in children)

        if tag == "cn":
            # Numeric constant
            typ = node.get("type", "real")
            text = (node.text or "").strip()
            if typ == "e-notation":
                # <cn type="e-notation"> mantissa <sep/> exponent </cn>
                sep = node.find(f"{MATH_NS}sep")
                if sep is not None:
                    mantissa = (node.text or "").strip()
                    exponent = (sep.tail or "").strip()
                    return f"{mantissa}e{exponent}"
            if typ == "integer":
                return f"{text}.0"  # Avoid C++ integer division
            return text

        if tag == "ci":
            # Identifier — resolve SBML ID to name
            sbml_id = (node.text or "").strip()
            return self.id_to_name.get(sbml_id, sbml_id)

        if tag == "apply":
            children = list(node)
            if not children:
                return "0"
            op_node = children[0]
            op_tag = op_node.tag.replace(MATH_NS, "")

            # SimBiology exports max/min as <ci>max</ci> instead of <max/>
            if op_tag == "ci":
                func_name = (op_node.text or "").strip()
                _CI_FUNCTIONS = {"max", "min", "abs", "floor", "ceil",
                                 "exp", "log", "ln", "sqrt", "power"}
                if func_name in _CI_FUNCTIONS:
                    op_tag = func_name

            # Handle <root> specially: extract <degree> child
            if op_tag == "root":
                degree_val = "2"  # default: square root
                value_nodes = []
                for c in children[1:]:
                    ct = c.tag.replace(MATH_NS, "")
                    if ct == "degree":
                        # <degree><cn>N</cn></degree>
                        inner = list(c)
                        if inner:
                            degree_val = self._mathml_to_infix(inner[0])
                    else:
                        value_nodes.append(c)
                if value_nodes:
                    val = self._mathml_to_infix(value_nodes[0])
                else:
                    val = "0"
                if degree_val == "2":
                    return f"std::sqrt({val})"
                return f"std::pow({val}, 1.0 / {degree_val})"

            args = [self._mathml_to_infix(c) for c in children[1:]]
            return self._apply_op(op_tag, args)

        if tag == "piecewise":
            return self._convert_piecewise(node)

        if tag in ("sep", "degree"):
            return ""

        # Fallback
        return f"/* unknown MathML tag: {tag} */"

    def _apply_op(self, op: str, args: List[str]) -> str:
        """Convert a MathML <apply> operation to infix C++."""
        # Binary arithmetic
        if op == "plus":
            return " + ".join(f"({a})" if " - " in a or " + " in a else a
                              for a in args) if len(args) > 1 else args[0]
        if op == "minus":
            if len(args) == 1:
                return f"(-{args[0]})"
            return f"({args[0]} - {args[1]})"
        if op == "times":
            return " * ".join(f"({a})" if (" + " in a or " - " in a) else a
                              for a in args)
        if op == "divide":
            num = f"({args[0]})" if (" + " in args[0] or " - " in args[0]) else args[0]
            den = f"({args[1]})" if (" + " in args[1] or " - " in args[1]
                                     or " * " in args[1] or " / " in args[1]) else args[1]
            return f"{num} / {den}"
        if op == "power":
            return f"std::pow({args[0]}, {args[1]})"

        # Comparison
        if op == "lt":
            return f"({args[0]} < {args[1]})"
        if op == "leq":
            return f"({args[0]} <= {args[1]})"
        if op == "gt":
            return f"({args[0]} > {args[1]})"
        if op == "geq":
            return f"({args[0]} >= {args[1]})"
        if op == "eq":
            return f"({args[0]} == {args[1]})"
        if op == "neq":
            return f"({args[0]} != {args[1]})"

        # Logic
        if op == "and":
            return " && ".join(f"({a})" for a in args)
        if op == "or":
            return " || ".join(f"({a})" for a in args)
        if op == "not":
            return f"(!({args[0]}))"

        # Functions
        func_map = {
            "ln": "std::log", "log": "std::log", "log2": "std::log2",
            "log10": "std::log10", "exp": "std::exp", "sqrt": "std::sqrt",
            "abs": "std::abs", "floor": "std::floor", "ceiling": "std::ceil",
            "sin": "std::sin", "cos": "std::cos", "tan": "std::tan",
        }
        if op in func_map:
            return f"{func_map[op]}({', '.join(args)})"

        if op == "max":
            if len(args) == 2:
                return f"std::max({args[0]}, {args[1]})"
            # Nested max for >2 args
            result = args[0]
            for a in args[1:]:
                result = f"std::max({result}, {a})"
            return result
        if op == "min":
            if len(args) == 2:
                return f"std::min({args[0]}, {args[1]})"
            result = args[0]
            for a in args[1:]:
                result = f"std::min({result}, {a})"
            return result

        if op == "root":
            # <apply><root/><degree><cn>2</cn></degree><ci>x</ci></apply>
            # args[0] is degree, args[1] is the value (if degree present)
            if len(args) == 2:
                return f"std::pow({args[1]}, 1.0 / {args[0]})"
            return f"std::sqrt({args[0]})"

        return f"/* unknown op: {op} */({', '.join(args)})"

    def _convert_piecewise(self, node) -> str:
        """Convert MathML <piecewise> to C++ ternary."""
        pieces = node.findall(f"{MATH_NS}piece")
        otherwise = node.find(f"{MATH_NS}otherwise")

        parts = []
        for piece in pieces:
            children = list(piece)
            if len(children) >= 2:
                val = self._mathml_to_infix(children[0])
                cond = self._mathml_to_infix(children[1])
                parts.append((cond, val))

        default = "0.0"
        if otherwise is not None:
            children = list(otherwise)
            if children:
                default = self._mathml_to_infix(children[0])

        # Build nested ternary
        result = default
        for cond, val in reversed(parts):
            result = f"({cond} ? {val} : {result})"
        return result


# =========================================================================
# Identifier classification for C++ macro wrapping
# =========================================================================

def _sanitize(name: str) -> str:
    """V_T.C1 → V_T_C1"""
    return name.replace(".", "_")


def classify_identifiers(sbml: SBMLModel) -> dict:
    """Build identifier → C++ macro mapping.

    Returns dict: human_name → "SPVAR(SP_xxx)" | "PARAM(P_xxx)" | "AUX_VAR_xxx"
    """
    mapping = {}

    species_names = {sp["name"] for sp in sbml.species}
    param_names = {p["name"] for p in sbml.parameters}
    rule_vars = {r["variable_name"] for r in sbml.assignment_rules}
    comp_names = {c["name"] for c in sbml.compartments}

    for sp in sbml.species:
        mapping[sp["name"]] = f'SPVAR(SP_{_sanitize(sp["name"])})'
        # Also map base_name if unambiguous
        # (skip for now — base names can be ambiguous across compartments)

    for r in sbml.assignment_rules:
        vn = r["variable_name"]
        mapping[vn] = f'AUX_VAR_{_sanitize(vn)}'

    # Parameters that are NOT overridden by rules
    for p in sbml.parameters:
        if p["name"] not in rule_vars:
            mapping[p["name"]] = f'PARAM(P_{_sanitize(p["name"])})'

    # Compartment names that appear as rule variables → AUX_VAR
    # Compartment names that are just fixed params → PARAM
    for c in sbml.compartments:
        if c["name"] in rule_vars:
            pass  # already handled above
        elif c["name"] not in mapping:
            mapping[c["name"]] = f'PARAM(P_{_sanitize(c["name"])})'

    return mapping


def wrap_expression(expr: str, mapping: dict) -> str:
    """Replace human-readable names in an expression with C++ macros."""
    # Sort by length descending for longest-match-first
    sorted_names = sorted(mapping.keys(), key=len, reverse=True)

    result = expr
    replacements = []
    for name in sorted_names:
        pattern = re.escape(name)
        full_pattern = r'(?<![a-zA-Z0-9_\.])' + pattern + r'(?![a-zA-Z0-9_\.])'
        placeholder = f"__PH{len(replacements)}__"
        if re.search(full_pattern, result):
            result = re.sub(full_pattern, placeholder, result)
            replacements.append((placeholder, mapping[name]))

    for ph, repl in replacements:
        result = result.replace(ph, repl)

    return result


# =========================================================================
# Rule dependency ordering
# =========================================================================

def order_rules(rules: List[dict], all_rule_names: Set[str]) -> List[dict]:
    """Topologically sort assignment rules so dependencies come first."""
    name_to_rule = {r["variable_name"]: r for r in rules}

    deps = {}
    for r in rules:
        vn = r["variable_name"]
        deps[vn] = set()
        for other_vn in all_rule_names:
            if other_vn == vn:
                continue
            pattern = re.escape(other_vn)
            if re.search(r'(?<![a-zA-Z0-9_\.])' + pattern + r'(?![a-zA-Z0-9_\.])',
                         r["expression"]):
                deps[vn].add(other_vn)

    # Kahn's algorithm
    in_degree = {vn: len(deps[vn]) for vn in name_to_rule}
    queue = sorted([vn for vn in name_to_rule if in_degree[vn] == 0])
    ordered = []
    while queue:
        node = queue.pop(0)
        ordered.append(name_to_rule[node])
        for other in name_to_rule:
            if node in deps.get(other, set()):
                deps[other].discard(node)
                in_degree[other] -= 1
                if in_degree[other] == 0:
                    queue.append(other)
        queue.sort()

    if len(ordered) != len(rules):
        remaining = set(name_to_rule.keys()) - {r["variable_name"] for r in ordered}
        print(f"WARNING: circular rule dependencies: {remaining}", file=sys.stderr)
        for vn in sorted(remaining):
            ordered.append(name_to_rule[vn])

    return ordered


# =========================================================================
# Volume scaling for ydot
# =========================================================================

def needs_volume_scaling(species: dict, sbml: SBMLModel) -> Optional[str]:
    """Check if a species' ydot needs 1/V_compartment scaling.

    Amount-based species: no scaling (ydot in moles/s).
    Concentration-based: ydot = flux_sum / V_compartment.

    Returns compartment name if scaling needed, None otherwise.
    """
    # hasOnlySubstanceUnits=true → amount, no scaling
    if species.get("has_only_substance_units"):
        return None

    # Check unit: cell, nanomole, milligram → amount
    unit_id = species.get("units", "")
    si_factor = sbml.get_si_factor(unit_id)

    # Heuristic: if the unit involves only moles (or moles*other without m^-3),
    # it's amount-based. If it involves m^-3 (concentration) or m^-2 (surface
    # density), it needs scaling.
    # Use the unit definition structure directly.
    ud = None
    for u in sbml.model.findall(f".//{SBML_NS}unitDefinition"):
        if u.get("id") == unit_id:
            ud = u
            break

    if ud is None:
        return None

    has_inverse_volume = False
    for u in ud.findall(f".//{SBML_NS}unit"):
        kind = u.get("kind", "dimensionless")
        exp = float(u.get("exponent", "1"))
        if kind == "metre" and exp < 0:
            has_inverse_volume = True
        if kind in ("litre", "liter") and exp < 0:
            has_inverse_volume = True

    if has_inverse_volume:
        return species["compartment"]
    return None


# =========================================================================
# Stoichiometry
# =========================================================================

def build_stoichiometry(reactions: list, species_names: set) -> Dict[str, List[Tuple[int, int]]]:
    """species_name → [(flux_1based, +1/-1), ...]"""
    stoich: Dict[str, List[Tuple[int, int]]] = {sp: [] for sp in species_names}
    for i, rxn in enumerate(reactions):
        flux_idx = i + 1
        for sp in rxn["reactant_names"]:
            if sp in stoich:
                stoich[sp].append((flux_idx, -1))
        for sp in rxn["product_names"]:
            if sp in stoich:
                stoich[sp].append((flux_idx, +1))
    return stoich


# =========================================================================
# Code generators
# =========================================================================

def _get_constant_compartments(sbml: SBMLModel) -> list:
    """Compartments whose volumes are constant (not set by assignment rules)."""
    rule_vars = {r["variable_name"] for r in sbml.assignment_rules}
    param_names = {p["name"] for p in sbml.parameters}
    return [c for c in sbml.compartments
            if c["name"] not in rule_vars and c["name"] not in param_names]


def gen_enum_h(sbml: SBMLModel) -> str:
    const_comps = _get_constant_compartments(sbml)

    lines = [
        "#pragma once",
        "",
        "// Auto-generated by qsp_codegen.py from SBML — do not edit manually",
        "",
        "#include <utility>",
        "",
        "namespace CancerVCT{",
        "",
        "// QSP Species Enum (ODE state vector indices)",
        "enum QSPSpeciesEnum",
        "{",
    ]
    for sp in sbml.species:
        lines.append(f'SP_{_sanitize(sp["name"])},')
    lines.append(f"QSP_SPECIES_COUNT  // = {len(sbml.species)}")
    lines.append("};")
    lines.append("")
    lines.append("enum QSPNonSpeciesEnum { QSP_NON_SPECIES_COUNT = 0 };")
    lines.append("")
    lines.append("// QSP Parameter Enum (class parameters with SI conversion)")
    lines.append("enum QSPParamEnum")
    lines.append("{")
    for p in sbml.parameters:
        lines.append(f'P_{_sanitize(p["name"])},')
    for c in const_comps:
        lines.append(f'P_{_sanitize(c["name"])},')
    lines.append(f"QSP_PARAM_COUNT")
    lines.append("};")
    lines.append("")

    # Unified file param enum: compartment ICs + species ICs + model parameters
    # This matches XML hierarchy: init_value.Compartment, init_value.Species,
    # init_value.Parameter
    lines.append("// QSP File Param Enum (XML file indices)")
    lines.append("// Order: Compartment ICs, Species ICs, Model Parameters")
    lines.append("enum QSPFileParamEnum")
    lines.append("{")
    lines.append("// --- Compartment initial values ---")
    for c in sbml.compartments:
        lines.append(f'QSP_{_sanitize(c["name"])},')
    lines.append("// --- Species initial values ---")
    for sp in sbml.species:
        lines.append(f'QSP_{_sanitize(sp["name"])},')
    lines.append("// --- Model parameters ---")
    for p in sbml.parameters:
        lines.append(f'QSP_{_sanitize(p["name"])},')
    lines.append(f"QSP_FILE_PARAM_COUNT")
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace CancerVCT")
    return "\n".join(lines) + "\n"


def gen_ode_h() -> str:
    return '''#ifndef __CancerVCT_ODE__
#define __CancerVCT_ODE__

// Auto-generated by qsp_codegen.py from SBML — do not edit manually

#include "../cvode/CVODEBase.h"
#include "QSPParam.h"

namespace CancerVCT{

class ODE_system :
    public CVODEBase
{
public:
    static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
    static int g(realtype t, N_Vector y, realtype *gout, void *user_data);
    static std::string getHeader();
    static void setup_class_parameters(QSPParam& param);
    static double get_class_param(unsigned int i);
    static void set_class_param(unsigned int i, double v);

    template<class Archive>
    static void classSerialize(Archive & ar, const unsigned int version);

    static bool use_steady_state;
    static bool use_resection;
    static double _QSP_weight;
private:
    static state_type _class_parameter;
public:
    ODE_system();
    ODE_system(const ODE_system& c);
    ~ODE_system();
    void setup_instance_tolerance(QSPParam& param);
    void setup_instance_variables(QSPParam& param);
    void eval_init_assignment(void);
    unsigned int get_num_variables(void)const { return _species_var.size(); };
    unsigned int get_num_params(void)const { return _class_parameter.size(); };

protected:
    void setupVariables(void);
    void setupEvents(void);
    void initSolver(realtype t0);
    void update_y_other(void);
    void adjust_hybrid_variables(void);
    bool triggerComponentEvaluate(int i, realtype t, bool curr);
    bool eventEvaluate(int i);
    bool eventExecution(int i, bool delay, realtype& dt);
    realtype get_unit_conversion_species(int i) const;
    realtype get_unit_conversion_nspvar(int i) const;
    bool allow_negative(int i)const{ return false; };
private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int /*version*/);
};

template<class Archive>
inline void ODE_system::serialize(Archive & ar, const unsigned int){
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(CVODEBase);
}
template<class Archive>
void ODE_system::classSerialize(Archive & ar, const unsigned int){
    ar & BOOST_SERIALIZATION_NVP(_class_parameter);
}
inline double ODE_system::get_class_param(unsigned int i){
    return _class_parameter[i];
}
inline void ODE_system::set_class_param(unsigned int i, double v){
    _class_parameter[i] = v;
}

}  // namespace CancerVCT
#endif
'''


def gen_ode_cpp(sbml: SBMLModel, mapping: dict) -> str:
    lines = []
    lines.append('// Auto-generated by qsp_codegen.py from SBML — do not edit manually')
    lines.append('#include "ODE_system.h"')
    lines.append('#include "QSP_enum.h"')
    lines.append('')
    lines.append('#define SPVAR(x) NV_DATA_S(y)[x]')
    lines.append('#define NSPVAR(x) ptrOde->_nonspecies_var[x]')
    lines.append('#define PARAM(x) _class_parameter[x]')
    lines.append('#define PFILE(x) param.getVal(x)')
    lines.append('')
    lines.append('namespace CancerVCT{')
    lines.append('#define QSP_W ODE_system::_QSP_weight')
    lines.append('')
    lines.append('bool ODE_system::use_steady_state = false;')
    lines.append('bool ODE_system::use_resection = false;')
    lines.append('double ODE_system::_QSP_weight = 1.0;')
    lines.append('')

    # Constructor/destructor
    lines.append('ODE_system::ODE_system() :CVODEBase() {')
    lines.append('    setupVariables(); setupEvents(); setupCVODE(); update_y_other();')
    lines.append('}')
    lines.append('ODE_system::ODE_system(const ODE_system& c) { setupCVODE(); }')
    lines.append('ODE_system::~ODE_system() {}')
    lines.append('')
    lines.append('void ODE_system::initSolver(realtype t){')
    lines.append('    restore_y(); int flag;')
    lines.append('    flag = CVodeInit(_cvode_mem, f, t, _y);')
    lines.append('    check_flag(&flag, "CVodeInit", 1);')
    lines.append('    flag = CVodeRootInit(_cvode_mem, _nroot, g);')
    lines.append('    check_flag(&flag, "CVodeRootInit", 1);')
    lines.append('}')
    lines.append('')

    # Count total params (parameters + compartment-as-params)
    rule_vars = {r["variable_name"] for r in sbml.assignment_rules}
    param_names_set = {p["name"] for p in sbml.parameters}
    extra_comp_params = [c for c in sbml.compartments
                         if c["name"] not in rule_vars and c["name"] not in param_names_set]
    total_params = len(sbml.parameters) + len(extra_comp_params)

    lines.append(f'state_type ODE_system::_class_parameter = state_type({total_params}, 0);')
    lines.append('')

    # setup_class_parameters
    lines.append('void ODE_system::setup_class_parameters(QSPParam& param){')
    for i, p in enumerate(sbml.parameters):
        factor = sbml.get_si_factor(p["units"])
        san = _sanitize(p["name"])
        lines.append(f'    //{p["name"]}, index: {i}, units: {p["units"]}')
        lines.append(f'    _class_parameter[P_{san}] = PFILE(QSP_{san}) * {factor:.15g};')
    for j, c in enumerate(extra_comp_params):
        factor = sbml.get_si_factor(c["units"])
        san = _sanitize(c["name"])
        idx = len(sbml.parameters) + j
        lines.append(f'    //{c["name"]} (compartment), index: {idx}, units: {c["units"]}')
        lines.append(f'    _class_parameter[P_{san}] = PFILE(QSP_{san}) * {factor:.15g};')
    lines.append('}')
    lines.append('')

    # setupVariables
    lines.append('void ODE_system::setupVariables(void){')
    lines.append(f'    _species_var = std::vector<realtype>({len(sbml.species)}, 0);')
    lines.append(f'    _nonspecies_var = std::vector<realtype>(0, 0);')
    lines.append(f'    _species_other = std::vector<realtype>(0, 0);')
    lines.append('}')
    lines.append('')

    # Stubs
    lines.append('void ODE_system::setupEvents(void){ _nevent = 0; _nroot = 0; }')
    lines.append('void ODE_system::update_y_other(void){ }')
    lines.append('void ODE_system::adjust_hybrid_variables(void){ }')
    lines.append('bool ODE_system::triggerComponentEvaluate(int, realtype, bool){ return false; }')
    lines.append('bool ODE_system::eventEvaluate(int){ return false; }')
    lines.append('bool ODE_system::eventExecution(int, bool, realtype&){ return false; }')
    lines.append('realtype ODE_system::get_unit_conversion_species(int) const { return 1.0; }')
    lines.append('realtype ODE_system::get_unit_conversion_nspvar(int) const { return 1.0; }')
    lines.append('')

    lines.append('void ODE_system::setup_instance_tolerance(QSPParam&){')
    lines.append('    _abstol = 1e-12; _reltol = 1e-6;')
    lines.append('}')
    lines.append('')
    lines.append('void ODE_system::setup_instance_variables(QSPParam& param){')
    for i, sp in enumerate(sbml.species):
        factor = sbml.get_si_factor(sp["units"])
        san = _sanitize(sp["name"])
        lines.append(f'    //{sp["name"]}, index: {i}, units: {sp["units"]}')
        lines.append(f'    _species_var[SP_{san}] = PFILE(QSP_{san}) * {factor:.15g};')
    lines.append('}')
    lines.append('')

    # getHeader
    header = ",".join(sp["name"] for sp in sbml.species)
    lines.append('std::string ODE_system::getHeader(){')
    lines.append(f'    return "{header}";')
    lines.append('}')
    lines.append('')

    lines.append('int ODE_system::g(realtype, N_Vector, realtype*, void*){ return 0; }')
    lines.append('')

    # eval_init_assignment
    lines.append('void ODE_system::eval_init_assignment(void){')
    for ia in sbml.initial_assignments:
        vn = ia["variable_name"]
        san = _sanitize(vn)
        cpp_expr = wrap_expression(ia["expression"], mapping)
        # For init, SPVAR reads from _y directly
        cpp_init = cpp_expr.replace("SPVAR(", "NV_DATA_S(_y)[")
        lines.append(f'    // {vn} = {ia["expression"][:80]}')
        lines.append(f'    NV_DATA_S(_y)[SP_{san}] = {cpp_init};')
    lines.append('}')
    lines.append('')

    # ---------------------------------------------------------------
    # RHS function f()
    # ---------------------------------------------------------------
    lines.append('int ODE_system::f(realtype t, N_Vector y, N_Vector ydot, void *user_data){')
    lines.append('')
    lines.append('    ODE_system* ptrOde = static_cast<ODE_system*>(user_data);')
    lines.append('')

    # 1) Assignment rules (topologically ordered)
    lines.append('    //Assignment rules:')
    lines.append('')
    rule_names = {r["variable_name"] for r in sbml.assignment_rules}
    ordered = order_rules(sbml.assignment_rules, rule_names)
    for r in ordered:
        vn = r["variable_name"]
        san = _sanitize(vn)
        cpp_expr = wrap_expression(r["expression"], mapping)
        lines.append(f'    realtype AUX_VAR_{san} = {cpp_expr};')
        lines.append('')

    # 2) Reaction fluxes
    lines.append('    //Reaction fluxes:')
    lines.append('')
    for i, rxn in enumerate(sbml.reactions):
        flux_idx = i + 1
        cpp_rate = wrap_expression(rxn["rate_law"], mapping)
        lines.append(f'    realtype ReactionFlux{flux_idx} = {cpp_rate};')
        lines.append('')

    # 3) ydot assembly
    lines.append('    //ODE right-hand side:')
    lines.append('')
    sp_names = {sp["name"] for sp in sbml.species}
    stoich = build_stoichiometry(sbml.reactions, sp_names)

    for sp in sbml.species:
        spn = sp["name"]
        san = _sanitize(spn)
        entries = stoich.get(spn, [])

        if not entries:
            lines.append(f'    NV_DATA_S(ydot)[SP_{san}] = 0.0;')
            continue

        flux_terms = []
        for fidx, sign in entries:
            flux_terms.append(f"{'+ ' if sign > 0 else '- '}ReactionFlux{fidx}")
        flux_expr = " ".join(flux_terms)
        if flux_expr.startswith("+ "):
            flux_expr = flux_expr[2:]

        comp_name = needs_volume_scaling(sp, sbml)
        if comp_name is not None:
            comp_san = _sanitize(comp_name)
            if comp_name in rule_names:
                vol = f"AUX_VAR_{comp_san}"
            else:
                vol = f"PARAM(P_{comp_san})"
            lines.append(f'    NV_DATA_S(ydot)[SP_{san}] = 1/{vol}*({flux_expr});')
        else:
            lines.append(f'    NV_DATA_S(ydot)[SP_{san}] = {flux_expr};')

    lines.append('')
    lines.append('    return 0;')
    lines.append('}')
    lines.append('')
    lines.append('}  // namespace CancerVCT')
    return "\n".join(lines) + "\n"


def gen_qsp_param_h() -> str:
    return '''#ifndef __CancerVCT_QSPParam__
#define __CancerVCT_QSPParam__

// Auto-generated by qsp_codegen.py from SBML — do not edit manually

#include "../ParamBase.h"

namespace CancerVCT{

class QSPParam : public ParamBase
{
public:
    QSPParam();
    ~QSPParam(){};
    double getVal(int n) const;
    void printParam(void) const;
private:
    std::vector<double> _param;
    void _readParameters(const std::string& filename);
    static const char* _xml_paths[];
};

}  // namespace CancerVCT
#endif
'''


def gen_qsp_param_cpp(sbml: SBMLModel) -> str:
    total = len(sbml.compartments) + len(sbml.species) + len(sbml.parameters)

    lines = []
    lines.append('// Auto-generated by qsp_codegen.py from SBML — do not edit manually')
    lines.append('#include "QSPParam.h"')
    lines.append('#include "QSP_enum.h"')
    lines.append('#include <boost/property_tree/xml_parser.hpp>')
    lines.append('#include <iostream>')
    lines.append('')
    lines.append('namespace CancerVCT{')
    lines.append('')

    # XML paths: Compartment ICs, Species ICs, Model Parameters
    # Must match QSPFileParamEnum order exactly
    lines.append('const char* QSPParam::_xml_paths[] = {')
    lines.append('    // Compartment initial values')
    for c in sbml.compartments:
        san = _sanitize(c["name"])
        lines.append(f'    "Param.QSP.init_value.Compartment.{san}",')
    lines.append('    // Species initial values')
    for sp in sbml.species:
        san = _sanitize(sp["name"])
        lines.append(f'    "Param.QSP.init_value.Species.{san}",')
    lines.append('    // Model parameters')
    for p in sbml.parameters:
        san = _sanitize(p["name"])
        lines.append(f'    "Param.QSP.init_value.Parameter.{san}",')
    lines.append('};')
    lines.append('')
    lines.append(f'QSPParam::QSPParam() :_param({total}, 0) {{}}')
    lines.append('')
    lines.append('double QSPParam::getVal(int n) const { return _param[n]; }')
    lines.append('')
    lines.append('void QSPParam::_readParameters(const std::string& filename){')
    lines.append('    namespace pt = boost::property_tree;')
    lines.append('    pt::ptree tree;')
    lines.append('    pt::read_xml(filename, tree, pt::xml_parser::trim_whitespace);')
    lines.append(f'    for (int i = 0; i < {total}; i++){{')
    lines.append('        try { _param[i] = tree.get<double>(_xml_paths[i]); }')
    lines.append('        catch (const pt::ptree_bad_path&) {')
    lines.append('            std::cerr << "WARNING: QSP param not found: "')
    lines.append('                      << _xml_paths[i] << std::endl;')
    lines.append('        }')
    lines.append('    }')
    lines.append('}')
    lines.append('')
    lines.append('void QSPParam::printParam(void) const {')
    lines.append(f'    for (int i = 0; i < {total}; i++)')
    lines.append('        std::cout << _xml_paths[i] << " = " << _param[i] << std::endl;')
    lines.append('}')
    lines.append('')
    lines.append('}  // namespace CancerVCT')
    return "\n".join(lines) + "\n"


def gen_xml_snippet(sbml: SBMLModel) -> str:
    """Generate XML <QSP> section for param_all_test.xml."""
    lines = ['<QSP>']
    lines.append('  <simulation>')
    lines.append('    <weight_qsp>0.8</weight_qsp>')
    lines.append('    <t_steadystate>5000</t_steadystate>')
    lines.append('    <use_resection>0</use_resection>')
    lines.append('    <t_resection>1000</t_resection>')
    lines.append('    <presimulation_diam_frac>1.00</presimulation_diam_frac>')
    lines.append('    <start>0</start>')
    lines.append('    <step>1</step>')
    lines.append('    <n_step>360</n_step>')
    lines.append('    <tol_rel>1e-09</tol_rel>')
    lines.append('    <tol_abs>1e-12</tol_abs>')
    lines.append('  </simulation>')
    lines.append('  <init_value>')

    # Compartments
    lines.append('    <Compartment>')
    for c in sbml.compartments:
        san = _sanitize(c["name"])
        lines.append(f'      <{san}>{c["size"]}</{san}>')
    lines.append('    </Compartment>')

    # Species
    lines.append('    <Species>')
    for sp in sbml.species:
        san = _sanitize(sp["name"])
        lines.append(f'      <{san}>{sp["initial_value"]}</{san}>')
    lines.append('    </Species>')

    # Parameters
    lines.append('    <Parameter>')
    for p in sbml.parameters:
        san = _sanitize(p["name"])
        lines.append(f'      <{san}>{p["value"]}</{san}>')
    lines.append('    </Parameter>')

    lines.append('  </init_value>')
    lines.append('</QSP>')
    return "\n".join(lines)


# =========================================================================
# Main
# =========================================================================

def main():
    sbml_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SBML
    print(f"Parsing SBML: {sbml_path}")

    sbml = SBMLModel(sbml_path)
    print(f"  Species:          {len(sbml.species)}")
    print(f"  Compartments:     {len(sbml.compartments)}")
    print(f"  Reactions:        {len(sbml.reactions)}")
    print(f"  Parameters:       {len(sbml.parameters)}")
    print(f"  Assignment rules: {len(sbml.assignment_rules)}")
    print(f"  Initial assigns:  {len(sbml.initial_assignments)}")
    print(f"  Unit definitions: {len(sbml.unit_defs)}")

    mapping = classify_identifiers(sbml)
    print(f"  Identifier mappings: {len(mapping)}")

    print("\nGenerating C++ files...")
    files = {
        "QSP_enum.h": gen_enum_h(sbml),
        "ODE_system.h": gen_ode_h(),
        "ODE_system.cpp": gen_ode_cpp(sbml, mapping),
        "QSPParam.h": gen_qsp_param_h(),
        "QSPParam.cpp": gen_qsp_param_cpp(sbml),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fname, content in files.items():
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, "w") as f:
            f.write(content)
        print(f"  {fname}: {content.count(chr(10))} lines")

    # XML snippet
    xml = gen_xml_snippet(sbml)
    xml_path = os.path.join(OUTPUT_DIR, "qsp_params_xml_snippet.xml")
    with open(xml_path, "w") as f:
        f.write(xml)
    print(f"  qsp_params_xml_snippet.xml (paste into param_all_test.xml)")

    print(f"\nDone! Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
