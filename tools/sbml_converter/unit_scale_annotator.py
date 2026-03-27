"""Compute per-parameter unit conversion factors for model-unit ODE generation.

Walks SBML ASTs computing the SI scale of each node. At additive nodes
where children have mismatched scales, traces the correction back to a
parameter and records a load-time conversion factor. This makes expressions
evaluate with numerically identical intermediate values to SimBiology.
"""

import libsbml
from sbml_parser import SBMLModel
from name_mapper import NameMapper


def _scales_match(a: float, b: float, tol: float = 1e-6) -> bool:
    if a == 0.0 or b == 0.0:
        return a == b
    return abs(a / b - 1.0) < tol


def _try_get_constant(node: libsbml.ASTNode):
    if node is None:
        return None
    ntype = node.getType()
    if ntype == libsbml.AST_INTEGER:
        return float(node.getInteger())
    if ntype in (libsbml.AST_REAL, libsbml.AST_REAL_E, libsbml.AST_RATIONAL):
        return node.getReal()
    if ntype == libsbml.AST_MINUS and node.getNumChildren() == 1:
        v = _try_get_constant(node.getChild(0))
        return -v if v is not None else None
    return None


def _find_parameters_in(node: libsbml.ASTNode, param_ids: set,
                        exponent: float = 1.0) -> list[tuple[str, float]]:
    """Find all parameter IDs in an AST subtree with their effective exponent.

    Returns list of (param_id, exponent) where exponent is +1 for numerator,
    -1 for denominator, etc.
    """
    if node is None:
        return []
    ntype = node.getType()

    if ntype == libsbml.AST_NAME:
        name = node.getName()
        if name in param_ids:
            return [(name, exponent)]
        return []

    result = []

    if ntype == libsbml.AST_DIVIDE:
        # Numerator keeps current exponent, denominator inverts it
        result.extend(_find_parameters_in(node.getChild(0), param_ids, exponent))
        result.extend(_find_parameters_in(node.getChild(1), param_ids, -exponent))
        return result

    if ntype in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER):
        exp_val = _try_get_constant(node.getChild(1))
        if exp_val is not None:
            result.extend(_find_parameters_in(node.getChild(0), param_ids,
                                              exponent * exp_val))
        return result

    for i in range(node.getNumChildren()):
        result.extend(_find_parameters_in(node.getChild(i), param_ids, exponent))
    return result


class UnitScaleAnnotator:
    """Computes per-parameter conversion factors for model-unit mode."""

    def __init__(self, model: SBMLModel, mapper: NameMapper, time_scale: float):
        self.model = model
        self.mapper = mapper
        self.time_scale = time_scale

        # SI scale for each element (modified as conversions are applied)
        self.element_scales: dict[str, float] = {}
        # Implicit scales for assignment rule variables
        self.implicit_scales: dict[str, float] = {}
        # Per-parameter conversion factors (applied at load time)
        self.param_conversions: dict[str, float] = {}
        # Set of all parameter IDs (for fast lookup)
        self._param_ids: set[str] = set()
        # Overall SI scale per kinetic law (computed with final scales)
        self.kl_scales: dict[int, float] = {}
        # Per-species substance scale
        self.substance_scales: dict[str, float] = {}
        # Per-stoichiometry-term factor
        self.stoich_factors: dict[tuple[int, str], float] = {}

        # Set of assignment rule targets (these can't be converted at load time)
        self._rule_targets: set[str] = {ar.variable_id for ar in model.assignment_rules}

        self._build_element_scales()

    def _build_element_scales(self):
        for comp in self.model.compartments:
            self.element_scales[comp.id] = comp.unit_si_scaling
        for sp in self.model.species:
            self.element_scales[sp.id] = sp.unit_si_scaling
        for param in self.model.parameters:
            self.element_scales[param.id] = param.unit_si_scaling
            # Only constant parameters (not rule targets) can be converted at load time
            if param.id not in self._rule_targets:
                self._param_ids.add(param.id)
        # Also include compartments (they're loaded as parameters)
        for comp in self.model.compartments:
            if comp.id not in self._rule_targets:
                self._param_ids.add(comp.id)

    def process_all(self):
        """Find parameter conversions, then compute stoich factors.

        Processes expressions in dependency order: fix mismatches in each
        assignment rule before computing its implicit scale, so downstream
        rules see corrected scales.
        """
        # Single pass through assignment rules in dependency order,
        # with re-checking after each round of corrections within a rule
        for ar in self.model.assignment_rule_order:
            for _retry in range(5):
                # Recompute all implicit scales up to this point
                # (corrections may have changed upstream rule scales)
                for prev_ar in self.model.assignment_rule_order:
                    self.implicit_scales[prev_ar.variable_id] = (
                        self._compute_scale(prev_ar.math_ast))
                    if prev_ar.variable_id == ar.variable_id:
                        break
                corrections = {}
                self._find_mismatches(ar.math_ast, corrections)
                if not corrections:
                    break
                self._apply_corrections(corrections)
            self.implicit_scales[ar.variable_id] = self._compute_scale(ar.math_ast)

        # Single pass through kinetic laws (with retry)
        for i, rxn in enumerate(self.model.reactions):
            if rxn.kinetic_law_ast:
                for _retry in range(5):
                    corrections = {}
                    self._find_mismatches(rxn.kinetic_law_ast, corrections)
                    if not corrections:
                        break
                    self._apply_corrections(corrections)
                self.kl_scales[i] = self._compute_scale(rxn.kinetic_law_ast)
            else:
                self.kl_scales[i] = 1.0

        # Process event expressions (with retry)
        for event in self.model.events:
            for ast in [event.trigger_ast] + [ea.math_ast for ea in event.event_assignments]:
                if ast:
                    for _retry in range(5):
                        corrections = {}
                        self._find_mismatches(ast, corrections)
                        if not corrections:
                            break
                        self._apply_corrections(corrections)

        self._compute_substance_scales()
        self._compute_stoich_factors()

    def _apply_corrections(self, corrections: dict):
        for param_id, correction in corrections.items():
            if correction == 0.0 or not (1e-30 < abs(correction) < 1e30):
                continue
            self.param_conversions[param_id] = (
                self.param_conversions.get(param_id, 1.0) * correction
            )
            self.element_scales[param_id] /= correction

    def _find_all_mismatches(self) -> dict[str, float]:
        """Walk all expressions, find additive mismatches, return param corrections."""
        corrections: dict[str, float] = {}

        for ar in self.model.assignment_rule_order:
            self._find_mismatches(ar.math_ast, corrections)

        for rxn in self.model.reactions:
            if rxn.kinetic_law_ast:
                self._find_mismatches(rxn.kinetic_law_ast, corrections)

        for event in self.model.events:
            if event.trigger_ast:
                self._find_mismatches(event.trigger_ast, corrections)
            for ea in event.event_assignments:
                if ea.math_ast:
                    self._find_mismatches(ea.math_ast, corrections)

        return corrections

    # ---- Scale computation (no C++ generation) ----

    def _compute_scale(self, node: libsbml.ASTNode) -> float:
        """Compute the SI scale of an AST node."""
        if node is None:
            return 1.0
        ntype = node.getType()

        if ntype in (libsbml.AST_INTEGER, libsbml.AST_REAL, libsbml.AST_REAL_E,
                     libsbml.AST_RATIONAL):
            return 1.0

        if ntype == libsbml.AST_NAME:
            name = node.getName()
            return self.implicit_scales.get(name, self.element_scales.get(name, 1.0))

        if ntype == libsbml.AST_NAME_TIME:
            return self.time_scale

        if ntype in (libsbml.AST_CONSTANT_PI, libsbml.AST_CONSTANT_E,
                     libsbml.AST_CONSTANT_TRUE, libsbml.AST_CONSTANT_FALSE):
            return 1.0

        if ntype == libsbml.AST_TIMES:
            scale = 1.0
            for i in range(node.getNumChildren()):
                scale *= self._compute_scale(node.getChild(i))
            return scale

        if ntype == libsbml.AST_DIVIDE:
            left = self._compute_scale(node.getChild(0))
            right = self._compute_scale(node.getChild(1))
            return left / right if right != 0 else left

        if ntype in (libsbml.AST_PLUS, libsbml.AST_MINUS):
            if ntype == libsbml.AST_MINUS and node.getNumChildren() == 1:
                return self._compute_scale(node.getChild(0))
            # Return first child's scale (assumes corrections have been applied)
            return self._compute_scale(node.getChild(0))

        if ntype in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER):
            base_s = self._compute_scale(node.getChild(0))
            exp_val = _try_get_constant(node.getChild(1))
            if exp_val is not None and base_s > 0:
                return base_s ** exp_val
            return 1.0

        if ntype == libsbml.AST_FUNCTION_ROOT:
            if node.getNumChildren() == 2:
                degree_val = _try_get_constant(node.getChild(0))
                arg_s = self._compute_scale(node.getChild(1))
                if degree_val and degree_val != 0 and arg_s > 0:
                    return arg_s ** (1.0 / degree_val)
            else:
                arg_s = self._compute_scale(node.getChild(0))
                return arg_s ** 0.5 if arg_s > 0 else 1.0
            return 1.0

        # Dimensionless results
        if ntype in (libsbml.AST_FUNCTION_EXP, libsbml.AST_FUNCTION_LN,
                     libsbml.AST_FUNCTION_LOG, libsbml.AST_FUNCTION_SIN,
                     libsbml.AST_FUNCTION_COS, libsbml.AST_FUNCTION_TAN,
                     libsbml.AST_RELATIONAL_EQ, libsbml.AST_RELATIONAL_NEQ,
                     libsbml.AST_RELATIONAL_LT, libsbml.AST_RELATIONAL_GT,
                     libsbml.AST_RELATIONAL_LEQ, libsbml.AST_RELATIONAL_GEQ,
                     libsbml.AST_LOGICAL_AND, libsbml.AST_LOGICAL_OR,
                     libsbml.AST_LOGICAL_NOT, libsbml.AST_LOGICAL_XOR):
            return 1.0

        # Scale-preserving functions
        if ntype in (libsbml.AST_FUNCTION_ABS, libsbml.AST_FUNCTION_FLOOR,
                     libsbml.AST_FUNCTION_CEILING):
            return self._compute_scale(node.getChild(0))

        # Min/Max/Piecewise — return first value child's scale
        if ntype in (libsbml.AST_FUNCTION_MIN, libsbml.AST_FUNCTION_MAX):
            return self._compute_scale(node.getChild(0))

        if ntype == libsbml.AST_FUNCTION_PIECEWISE:
            if node.getNumChildren() > 0:
                return self._compute_scale(node.getChild(0))
            return 1.0

        # Generic functions
        if ntype == libsbml.AST_FUNCTION:
            fname = node.getName()
            if fname in ("exp", "log", "log10"):
                return 1.0
            if fname == "abs":
                return self._compute_scale(node.getChild(0))
            if fname == "sqrt":
                s = self._compute_scale(node.getChild(0))
                return s ** 0.5 if s > 0 else 1.0
            if fname == "pow":
                base_s = self._compute_scale(node.getChild(0))
                exp_val = _try_get_constant(node.getChild(1))
                if exp_val is not None and base_s > 0:
                    return base_s ** exp_val
                return 1.0
            if fname == "nthroot":
                x_s = self._compute_scale(node.getChild(0))
                n_val = _try_get_constant(node.getChild(1))
                if n_val and n_val != 0 and x_s > 0:
                    return x_s ** (1.0 / n_val)
                return 1.0
            if fname in ("min", "max"):
                return self._compute_scale(node.getChild(0))
            return 1.0

        if ntype == libsbml.AST_LAMBDA:
            return self._compute_scale(node.getChild(node.getNumChildren() - 1))

        return 1.0

    # ---- Mismatch finding ----

    def _find_mismatches(self, node: libsbml.ASTNode, corrections: dict):
        """Recursively find additive/relational mismatches and record corrections."""
        if node is None:
            return
        ntype = node.getType()

        # Recurse into children first
        for i in range(node.getNumChildren()):
            self._find_mismatches(node.getChild(i), corrections)

        # Check additive nodes
        if ntype in (libsbml.AST_PLUS, libsbml.AST_MINUS) and node.getNumChildren() >= 2:
            self._check_additive_mismatch(node, corrections)

        # Check min/max
        if ntype in (libsbml.AST_FUNCTION_MIN, libsbml.AST_FUNCTION_MAX):
            self._check_additive_mismatch(node, corrections)

        # Check generic min/max
        if ntype == libsbml.AST_FUNCTION and node.getName() in ("min", "max"):
            self._check_additive_mismatch(node, corrections)

        # Check relational operators
        if ntype in (libsbml.AST_RELATIONAL_EQ, libsbml.AST_RELATIONAL_NEQ,
                     libsbml.AST_RELATIONAL_LT, libsbml.AST_RELATIONAL_GT,
                     libsbml.AST_RELATIONAL_LEQ, libsbml.AST_RELATIONAL_GEQ):
            self._check_additive_mismatch(node, corrections)

        # Check piecewise value branches
        if ntype == libsbml.AST_FUNCTION_PIECEWISE:
            self._check_piecewise_mismatch(node, corrections)

    def _check_additive_mismatch(self, node, corrections: dict):
        """At an additive/comparison node, check for scale mismatches."""
        n = node.getNumChildren()
        if n < 2:
            return

        scales = [self._compute_scale(node.getChild(i)) for i in range(n)]
        ref_scale = scales[0]

        # Find a non-zero reference
        if ref_scale == 0.0:
            for s in scales:
                if s != 0.0:
                    ref_scale = s
                    break

        if ref_scale == 0.0:
            return

        for i in range(n):
            if scales[i] == 0.0 or _scales_match(scales[i], ref_scale):
                continue

            # Mismatch! Find a parameter in child i's subtree
            child = node.getChild(i)
            params = _find_parameters_in(child, self._param_ids)

            if params:
                # Convert a parameter in the mismatched child to match ref
                subtree_correction = scales[i] / ref_scale
                param_id, param_exp = params[0]
                if param_id not in corrections and param_exp != 0.0:
                    corrections[param_id] = subtree_correction ** (1.0 / param_exp)
            elif child.getType() == libsbml.AST_NAME:
                # Mismatched child is a bare species/variable (no parameters).
                # Try to fix a parameter in the REFERENCE child (child 0)
                # so that the reference matches the species' scale.
                ref_child = node.getChild(0)
                ref_params = _find_parameters_in(ref_child, self._param_ids)
                if ref_params:
                    # ref must change to match child's scale
                    ref_correction = ref_scale / scales[i]
                    param_id, param_exp = ref_params[0]
                    if param_id not in corrections and param_exp != 0.0:
                        corrections[param_id] = ref_correction ** (1.0 / param_exp)
            # else: complex expression with no convertible params — skip.
            # The stoich factor will handle the overall flux correction.

    def _trace_rule_variable(self, node, ref_scale: float,
                             child_scale: float, corrections: dict) -> bool:
        """When a mismatched child has no convertible parameters, check if it
        references an assignment rule variable. If so, trace into that rule's
        expression to find a parameter to convert.

        Returns True if a correction was found.
        """
        if node is None:
            return False

        # Collect all assignment rule variables referenced in this subtree
        rule_var_ids = set()
        self._collect_rule_vars(node, rule_var_ids)

        if not rule_var_ids:
            return False

        # For each rule variable, check if converting a parameter inside its
        # rule expression could fix the scale mismatch
        subtree_correction = child_scale / ref_scale
        for var_id in rule_var_ids:
            # Find the assignment rule for this variable
            rule_ast = None
            for ar in self.model.assignment_rules:
                if ar.variable_id == var_id:
                    rule_ast = ar.math_ast
                    break
            if rule_ast is None:
                continue

            # Find parameters inside the rule expression.
            # Exclude dimensionless parameters (scale ~1.0) since they
            # shouldn't need unit conversion.
            params = _find_parameters_in(rule_ast, self._param_ids)
            params = [(pid, exp) for pid, exp in params
                      if not _scales_match(self.element_scales.get(pid, 1.0), 1.0)
                      and pid not in corrections]
            if not params:
                continue

            param_id, param_exp = params[0]
            if param_id in corrections or param_exp == 0.0:
                continue
            corrections[param_id] = subtree_correction ** (1.0 / param_exp)
            return True

        return False

    def _collect_rule_vars(self, node, result: set):
        """Collect assignment rule variable IDs referenced in a subtree."""
        if node is None:
            return
        if node.getType() == libsbml.AST_NAME:
            name = node.getName()
            if name in self._rule_targets:
                result.add(name)
        for i in range(node.getNumChildren()):
            self._collect_rule_vars(node.getChild(i), result)

    def _check_piecewise_mismatch(self, node, corrections: dict):
        """Check piecewise value branches for scale mismatches."""
        n = node.getNumChildren()
        if n < 3:
            return

        # Value children at indices 0, 2, 4, ...
        value_indices = list(range(0, n - 1, 2))
        if n % 2 == 1:
            value_indices.append(n - 1)

        if len(value_indices) < 2:
            return

        scales = [self._compute_scale(node.getChild(i)) for i in value_indices]
        ref_scale = scales[0]
        if ref_scale == 0.0:
            return

        for idx, (vi, s) in enumerate(zip(value_indices, scales)):
            if idx == 0 or s == 0.0 or _scales_match(s, ref_scale):
                continue
            subtree_correction = s / ref_scale
            params = _find_parameters_in(node.getChild(vi), self._param_ids)
            if params:
                param_id, param_exp = params[0]
                if param_id not in corrections and param_exp != 0.0:
                    corrections[param_id] = subtree_correction ** (1.0 / param_exp)

    # ---- Stoich factor computation ----

    def _compute_substance_scales(self):
        for sp in self.model.sp_var:
            if sp.has_only_substance_units:
                self.substance_scales[sp.id] = self.element_scales[sp.id]
            else:
                comp = self.model.id_to_compartment.get(sp.compartment_id)
                comp_scale = self.element_scales.get(comp.id, 1.0) if comp else 1.0
                self.substance_scales[sp.id] = self.element_scales[sp.id] * comp_scale

    def _compute_stoich_factors(self):
        for sp in self.model.sp_var:
            if sp.id not in self.model.stoichiometry:
                continue
            sub_scale = self.substance_scales[sp.id]
            if sub_scale == 0.0:
                continue
            for rxn_idx, _coeff in self.model.stoichiometry[sp.id]:
                kl_scale = self.kl_scales.get(rxn_idx, 1.0)
                factor = kl_scale * self.time_scale / sub_scale
                self.stoich_factors[(rxn_idx, sp.id)] = factor

    def get_stoich_factor(self, rxn_idx: int, species_id: str) -> float:
        return self.stoich_factors.get((rxn_idx, species_id), 1.0)
