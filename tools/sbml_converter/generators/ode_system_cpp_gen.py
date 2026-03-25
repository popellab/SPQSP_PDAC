"""Generate ODE_system.cpp — the core ODE implementation."""

from sbml_parser import SBMLModel, Species
from name_mapper import NameMapper
from ast_translator import ASTTranslator


def generate_ode_system_cpp(
    model: SBMLModel, mapper: NameMapper, namespace: str, class_name: str
) -> str:
    sections = []
    sections.append(_gen_preamble(model, mapper, namespace, class_name))
    sections.append(_gen_constructor(model, class_name))
    sections.append(_gen_init_solver(class_name))
    sections.append(_gen_setup_class_parameters(model, mapper, class_name))
    sections.append(_gen_setup_variables(model, mapper, class_name))
    sections.append(_gen_setup_instance_variables(model, mapper, class_name))
    sections.append(_gen_setup_instance_tolerance(class_name))
    sections.append(_gen_eval_init_assignment(model, mapper, class_name))
    sections.append(_gen_setup_events(model, mapper, class_name))
    sections.append(_gen_f_function(model, mapper, class_name))
    sections.append(_gen_g_function(model, mapper, class_name))
    sections.append(_gen_trigger_component_evaluate(model, mapper, class_name))
    sections.append(_gen_event_evaluate(model, mapper, class_name))
    sections.append(_gen_event_execution(model, mapper, class_name))
    sections.append(_gen_update_y_other(model, mapper, class_name))
    sections.append(_gen_get_header(model, mapper, class_name))
    sections.append(_gen_unit_conversion(model, mapper, class_name))
    sections.append(f"}};")  # close namespace

    return "\n".join(sections)


def _fmt(value: float) -> str:
    """Format a float for C++ with 17-digit precision."""
    if value == 0.0:
        return "0.0"
    if value == int(value) and abs(value) < 1e15:
        return f"{value:.1f}"
    return f"{value:.17g}"


# ============================================================================
# Preamble: includes, macros, static members
# ============================================================================

def _gen_preamble(model, mapper, namespace, class_name):
    n_params = len(model.compartments) + len(model.p_const)
    return f"""#include "{class_name}.h"

#define SPVAR(x) NV_DATA_S(y)[x]
#define NSPVAR(x) ptrOde->_nonspecies_var[x]
#define PARAM(x) _class_parameter[x]
#define PFILE(x) param.getVal(x)
#define QSP_W {class_name}::_QSP_weight

namespace {namespace}{{

bool {class_name}::use_steady_state = false;
bool {class_name}::use_resection = false;
double {class_name}::_QSP_weight = 1.0;
state_type {class_name}::_class_parameter = state_type({n_params}, 0);
"""


# ============================================================================
# Constructor / destructor
# ============================================================================

def _gen_constructor(model, class_name):
    return f"""
{class_name}::{class_name}()
:CVODEBase()
{{
    setupVariables();
    setupEvents();
    setupCVODE();
    update_y_other();
}}

{class_name}::{class_name}(const {class_name}& c)
{{
    setupCVODE();
}}

{class_name}::~{class_name}()
{{
}}
"""


# ============================================================================
# initSolver
# ============================================================================

def _gen_init_solver(class_name):
    return f"""
void {class_name}::initSolver(realtype t){{
    restore_y();
    int flag;

    flag = CVodeInit(_cvode_mem, f, t, _y);
    check_flag(&flag, "CVodeInit", 1);

    flag = CVodeRootInit(_cvode_mem, _nroot, g);
    check_flag(&flag, "CVodeRootInit", 1);

    return;
}}
"""


# ============================================================================
# setup_class_parameters
# ============================================================================

def _gen_setup_class_parameters(model, mapper, class_name):
    lines = []
    lines.append(f"void {class_name}::setup_class_parameters(QSPParam& param){{")

    # Compartments as parameters
    for comp in model.compartments:
        p_enum = mapper.param_enum(comp.id)
        qsp_enum = mapper.qsp_enum(comp.id)
        scaling = _fmt(comp.unit_si_scaling)
        lines.append(f"    //{comp.name}, {comp.id}")
        lines.append(f"    _class_parameter[{p_enum}] = PFILE({qsp_enum}) * {scaling};")

    # Constant parameters
    for param in model.p_const:
        p_enum = mapper.param_enum(param.id)
        qsp_enum = mapper.qsp_enum(param.id)
        scaling = _fmt(param.unit_si_scaling)
        lines.append(f"    //{param.name}, {param.id}")
        lines.append(f"    _class_parameter[{p_enum}] = PFILE({qsp_enum}) * {scaling};")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# setupVariables
# ============================================================================

def _gen_setup_variables(model, mapper, class_name):
    n_sp = len(model.sp_var)
    n_nsp = len(model.nsp_var)
    n_other = len(model.sp_other)
    return f"""
void {class_name}::setupVariables(void){{
    _species_var = std::vector<realtype>({n_sp}, 0);
    _nonspecies_var = std::vector<realtype>({n_nsp}, 0);
    _species_other = std::vector<realtype>({n_other}, 0);
    return;
}}
"""


# ============================================================================
# setup_instance_variables
# ============================================================================

def _gen_setup_instance_variables(model, mapper, class_name):
    lines = []
    lines.append(f"void {class_name}::setup_instance_variables(QSPParam& param){{")

    for sp in model.sp_var:
        enum = mapper.species_enum(sp.id)
        qsp_enum = mapper.qsp_enum(sp.id)
        scaling = _fmt(sp.unit_si_scaling)
        lines.append(f"    //{mapper.display_name(sp.id)}, {sp.id}")
        lines.append(f"    _species_var[{enum}] = PFILE({qsp_enum}) * {scaling};")

    # Non-species variables
    for i, nsp in enumerate(model.nsp_var):
        qsp_enum = mapper.qsp_enum(nsp.id)
        scaling = _fmt(nsp.unit_si_scaling if hasattr(nsp, 'unit_si_scaling') else 1.0)
        lines.append(f"    //{nsp.name}, {nsp.id}")
        lines.append(f"    _nonspecies_var[{i}] = PFILE({qsp_enum}) * {scaling};")

    lines.append("    return;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# setup_instance_tolerance
# ============================================================================

def _gen_setup_instance_tolerance(class_name):
    return f"""
void {class_name}::setup_instance_tolerance(QSPParam& param){{
    realtype reltol = PFILE(QSP_SIM_TOL_REL);
    realtype abstol_base = PFILE(QSP_SIM_TOL_ABS);
    N_Vector abstol = N_VNew_Serial(_neq, _sunctx);

    for (size_t i = 0; i < _species_var.size(); i++)
    {{
        NV_DATA_S(abstol)[i] = abstol_base * get_unit_conversion_species(i);
    }}

    int flag = CVodeSVtolerances(_cvode_mem, reltol, abstol);
    check_flag(&flag, "CVodeSVtolerances", 1);

    N_VDestroy(abstol);
    return;
}}
"""


# ============================================================================
# eval_init_assignment
# ============================================================================

def _gen_eval_init_assignment(model, mapper, class_name):
    lines = []
    lines.append(f"void {class_name}::eval_init_assignment(void){{")

    if model.initial_assignment_order:
        trans = ASTTranslator(mapper, context="event")
        # Assignment rules needed for initial assignments
        rule_targets = {ar.variable_id for ar in model.assignment_rules}
        ia_refs = set()
        for ia in model.initial_assignment_order:
            from dependency_sorter import _get_referenced_names
            ia_refs |= _get_referenced_names(ia.math_ast)
        needed_rules = [ar for ar in model.assignment_rule_order
                        if ar.variable_id in ia_refs]
        if needed_rules:
            lines.append("    //Assignment Rules required before IA")
            for ar in needed_rules:
                expr = trans.translate(ar.math_ast)
                lines.append(f"    realtype {mapper.aux_var(ar.variable_id)} = {expr};")

        lines.append("    //InitialAssignment")
        for ia in model.initial_assignment_order:
            expr = trans.translate(ia.math_ast)
            # Write to _species_var (not _y) so updateVar()->restore_y() propagates correctly
            if mapper.is_ode_species(ia.symbol_id):
                target = f"_species_var[{mapper.species_enum(ia.symbol_id)}]"
            elif mapper.is_other_species(ia.symbol_id):
                idx = next(i for i, sp in enumerate(model.sp_other) if sp.id == ia.symbol_id)
                target = f"_species_other[{idx}]"
            else:
                # Non-species (parameter/compartment) — shouldn't normally happen
                target = mapper.resolve_name(ia.symbol_id, "event")
            lines.append(f"    {target} = {expr};")

    lines.append("    updateVar();")
    lines.append("    return;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# setupEvents
# ============================================================================

def _gen_setup_events(model, mapper, class_name):
    lines = []
    lines.append(f"void {class_name}::setupEvents(void){{")
    lines.append("")

    n_events = len(model.events)
    # Each event has one trigger → one root for now (simple case)
    # More complex: parse trigger into components
    n_root = n_events  # simplified: one root per event

    lines.append(f"    _nevent = {n_events};")
    lines.append(f"    _nroot = {n_root};")
    lines.append("")

    if n_events > 0:
        lines.append(f"    _trigger_element_type = std::vector<EVENT_TRIGGER_ELEM_TYPE>(_nroot, TRIGGER_NON_INSTANT);")
        lines.append(f"    _trigger_element_satisfied = std::vector<bool>(_nroot, false);")
        lines.append(f"    _event_triggered = std::vector<bool>(_nevent, false);")
        lines.append("")

        for i, event in enumerate(model.events):
            lines.append(f"    _trigger_element_type[{i}] = TRIGGER_NON_INSTANT;")

        lines.append("")
        for i, event in enumerate(model.events):
            init_val = "true" if event.trigger_initial_value else "false"
            lines.append(f"    _event_triggered[{i}] = {init_val};")

    lines.append("")
    lines.append("    return;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# f() — the ODE right-hand side
# ============================================================================

def _gen_f_function(model, mapper, class_name):
    trans = ASTTranslator(mapper, context="f")
    lines = []
    lines.append(f"int {class_name}::f(realtype t, N_Vector y, N_Vector ydot, void *user_data){{")
    lines.append(f"    {class_name}* ptrOde = static_cast<{class_name}*>(user_data);")
    lines.append("")

    # Assignment rules (topologically sorted)
    lines.append("    //Assignment rules:")
    for ar in model.assignment_rule_order:
        expr = trans.translate(ar.math_ast)
        lines.append(f"    realtype {mapper.aux_var(ar.variable_id)} = {expr};")
    lines.append("")

    # Reaction fluxes
    lines.append("    //Reaction fluxes:")
    for i, rxn in enumerate(model.reactions):
        if rxn.kinetic_law_ast:
            expr = trans.translate(rxn.kinetic_law_ast)
        else:
            expr = "0.0"
        lines.append(f"    realtype ReactionFlux{i + 1} = {expr};")
    lines.append("")

    # ODE equations (ydot)
    lines.append("    //dydt:")
    for sp in model.sp_var:
        if sp.id not in model.stoichiometry:
            # No reactions for this species
            enum = mapper.species_enum(sp.id)
            lines.append(f"    NV_DATA_S(ydot)[{enum}] = 0.0;")
            continue

        stoich_entries = model.stoichiometry[sp.id]
        terms = []
        for rxn_idx, coeff in stoich_entries:
            flux = f"ReactionFlux{rxn_idx + 1}"
            if coeff == 1.0:
                terms.append(f"{flux}")
            elif coeff == -1.0:
                terms.append(f"-{flux}")
            elif coeff > 0:
                terms.append(f"{_fmt(coeff)}*{flux}")
            else:
                terms.append(f"{_fmt(coeff)}*{flux}")

        if not terms:
            rhs = "0.0"
        else:
            rhs = " + ".join(terms)
            # Fix double negatives: "+ -" -> "- "
            rhs = rhs.replace("+ -", "- ")

        enum = mapper.species_enum(sp.id)

        # Check if species needs compartment division
        # (if species is in concentration, not amount)
        if not sp.has_only_substance_units:
            comp = model.id_to_compartment.get(sp.compartment_id)
            if comp and not comp.constant:
                # Dynamic compartment — use AUX_VAR if it's a rule target
                if mapper.is_assignment_rule_target(comp.id):
                    comp_expr = mapper.aux_var(comp.id)
                else:
                    comp_expr = f"PARAM({mapper.param_enum(comp.id)})"
                lines.append(f"    //d({mapper.display_name(sp.id)})/dt")
                lines.append(f"    NV_DATA_S(ydot)[{enum}] = 1/{comp_expr}*({rhs});")
            else:
                # Constant compartment
                comp_enum = mapper.param_enum(sp.compartment_id)
                lines.append(f"    //d({mapper.display_name(sp.id)})/dt")
                lines.append(f"    NV_DATA_S(ydot)[{enum}] = 1/PARAM({comp_enum})*({rhs});")
        else:
            lines.append(f"    //d({mapper.display_name(sp.id)})/dt")
            lines.append(f"    NV_DATA_S(ydot)[{enum}] = {rhs};")

    lines.append("")
    lines.append("    return(0);")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# g() — root finding for events
# ============================================================================

def _gen_g_function(model, mapper, class_name):
    lines = []
    lines.append(f"int {class_name}::g(realtype t, N_Vector y, realtype *gout, void *user_data){{")
    lines.append("")
    lines.append(f"    {class_name}* ptrOde = static_cast<{class_name}*>(user_data);")
    lines.append("")

    if model.events:
        trans = ASTTranslator(mapper, context="f")

        # Compute assignment rules needed by event triggers
        needed_rules = _get_rules_needed_by_events(model, trans)
        if needed_rules:
            lines.append("    //Assignment rules:")
            for ar in needed_rules:
                expr = trans.translate(ar.math_ast)
                lines.append(f"    realtype {mapper.aux_var(ar.variable_id)} = {expr};")
            lines.append("")

        for i, event in enumerate(model.events):
            gout_expr = _trigger_to_root_expression(event.trigger_ast, trans)
            lines.append(f"    //{event.name}")
            lines.append(f"    gout[{i}] = {gout_expr};")
            lines.append("")

    lines.append("    return(0);")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _get_rules_needed_by_events(model, trans):
    """Find assignment rules needed by event triggers and assignments."""
    from dependency_sorter import _get_referenced_names
    rule_targets = {ar.variable_id for ar in model.assignment_rules}

    # Collect all names referenced by event triggers and assignments
    needed = set()
    for event in model.events:
        if event.trigger_ast:
            refs = _get_referenced_names(event.trigger_ast)
            needed |= (refs & rule_targets)
        for ea in event.event_assignments:
            if ea.math_ast:
                refs = _get_referenced_names(ea.math_ast)
                needed |= (refs & rule_targets)

    # Expand transitively
    changed = True
    while changed:
        changed = False
        for ar in model.assignment_rules:
            if ar.variable_id in needed:
                refs = _get_referenced_names(ar.math_ast) & rule_targets
                for ref in refs:
                    if ref not in needed:
                        needed.add(ref)
                        changed = True

    # Return in dependency order
    return [ar for ar in model.assignment_rule_order if ar.variable_id in needed]


def _trigger_to_root_expression(trigger_ast, trans):
    """Convert a trigger condition to a root-finding expression.

    For 'A < B', returns 'B - (A)' (positive when condition is satisfied).
    """
    import libsbml
    if trigger_ast is None:
        return "1"

    ntype = trigger_ast.getType()

    if ntype in (libsbml.AST_RELATIONAL_LT, libsbml.AST_RELATIONAL_LEQ):
        # A < B  →  B - (A)
        left = trans.translate(trigger_ast.getChild(0))
        right = trans.translate(trigger_ast.getChild(1))
        return f"{right} - ({left})"

    if ntype in (libsbml.AST_RELATIONAL_GT, libsbml.AST_RELATIONAL_GEQ):
        # A > B  →  A - (B)
        left = trans.translate(trigger_ast.getChild(0))
        right = trans.translate(trigger_ast.getChild(1))
        return f"{left} - ({right})"

    if ntype in (libsbml.AST_RELATIONAL_EQ, libsbml.AST_RELATIONAL_NEQ):
        left = trans.translate(trigger_ast.getChild(0))
        right = trans.translate(trigger_ast.getChild(1))
        return f"{left} - ({right})"

    if ntype == libsbml.AST_LOGICAL_AND:
        # For AND, use the minimum of both root functions
        parts = []
        for i in range(trigger_ast.getNumChildren()):
            parts.append(_trigger_to_root_expression(trigger_ast.getChild(i), trans))
        return f"std::min({', '.join(parts)})"

    if ntype == libsbml.AST_LOGICAL_OR:
        parts = []
        for i in range(trigger_ast.getNumChildren()):
            parts.append(_trigger_to_root_expression(trigger_ast.getChild(i), trans))
        return f"std::max({', '.join(parts)})"

    # Fallback: just translate directly
    return trans.translate(trigger_ast)


# ============================================================================
# triggerComponentEvaluate
# ============================================================================

def _gen_trigger_component_evaluate(model, mapper, class_name):
    lines = []
    lines.append(f"bool {class_name}::triggerComponentEvaluate(int i, realtype t, bool curr) {{")
    lines.append("")
    lines.append("    bool discrete = false;")
    lines.append("    realtype diff = 0;")
    lines.append("    bool eval = false;")
    lines.append("")

    if model.events:
        trans_event = ASTTranslator(mapper, context="event")

        # Compute needed assignment rules
        trans_f = ASTTranslator(mapper, context="f")
        needed_rules = _get_rules_needed_by_events(model, trans_f)
        if needed_rules:
            lines.append("    //Assignment rules:")
            for ar in needed_rules:
                expr = trans_event.translate(ar.math_ast)
                lines.append(f"    realtype {mapper.aux_var(ar.variable_id)} = {expr};")
            lines.append("")

        lines.append("    switch(i)")
        lines.append("    {")
        for i, event in enumerate(model.events):
            gout_expr = _trigger_to_root_expression(event.trigger_ast, trans_event)
            lines.append(f"    case {i}:")
            lines.append(f"        //{event.name}")
            lines.append(f"        diff = {gout_expr};")
            lines.append(f"        break;")
        lines.append("    default:")
        lines.append("        break;")
        lines.append("    }")

    lines.append("    if (!discrete){")
    lines.append("        eval = diff == 0 ? curr : (diff > 0);")
    lines.append("    }")
    lines.append("")
    lines.append("    return eval;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# eventEvaluate
# ============================================================================

def _gen_event_evaluate(model, mapper, class_name):
    lines = []
    lines.append(f"bool {class_name}::eventEvaluate(int i) {{")
    lines.append("    bool eval = false;")
    lines.append("    switch(i)")
    lines.append("    {")

    for i, event in enumerate(model.events):
        # Simple case: one trigger per event → direct getSatisfied
        lines.append(f"    case {i}:")
        lines.append(f"        eval = getSatisfied({i});")
        lines.append(f"        break;")

    lines.append("    default:")
    lines.append("        break;")
    lines.append("    }")
    lines.append("    return eval;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# eventExecution
# ============================================================================

def _gen_event_execution(model, mapper, class_name):
    lines = []
    lines.append(f"bool {class_name}::eventExecution(int i, bool delayed, realtype& dt){{")
    lines.append("")
    lines.append("    bool setDelay = false;")
    lines.append("")

    if model.events:
        trans = ASTTranslator(mapper, context="event")
        lines.append("    switch(i)")
        lines.append("    {")
        for i, event in enumerate(model.events):
            lines.append(f"    case {i}:")
            for ea in event.event_assignments:
                target = mapper.resolve_name(ea.variable_id, "event")
                expr = trans.translate(ea.math_ast)
                lines.append(f"        {target} = {expr};")
            lines.append(f"        break;")

        lines.append("    default:")
        lines.append("        break;")
        lines.append("    }")

    lines.append("")
    lines.append("    return setDelay;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# update_y_other
# ============================================================================

def _gen_update_y_other(model, mapper, class_name):
    lines = []
    lines.append(f"void {class_name}::update_y_other(void){{")
    lines.append("")

    if model.sp_other:
        trans = ASTTranslator(mapper, context="other")
        sp_other_ids = {sp.id for sp in model.sp_other}

        # Find ALL assignment rules needed (sp_other rules + their dependencies)
        from dependency_sorter import _get_referenced_names
        needed = set(sp_other_ids)
        rule_targets = {ar.variable_id for ar in model.assignment_rules}
        # Iteratively find transitive dependencies
        changed = True
        while changed:
            changed = False
            for ar in model.assignment_rules:
                if ar.variable_id in needed:
                    refs = _get_referenced_names(ar.math_ast) & rule_targets
                    for ref in refs:
                        if ref not in needed:
                            needed.add(ref)
                            changed = True

        # Emit needed rules in dependency order
        for ar in model.assignment_rule_order:
            if ar.variable_id in needed:
                expr = trans.translate(ar.math_ast)
                aux = mapper.aux_var(ar.variable_id)
                lines.append(f"    realtype {aux} = {expr};")

        lines.append("")
        for i, sp in enumerate(model.sp_other):
            aux = mapper.aux_var(sp.id)
            lines.append(f"    //{mapper.display_name(sp.id)}")
            lines.append(f"    _species_other[{i}] = {aux};")

    lines.append("")
    lines.append("    return;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# getHeader
# ============================================================================

def _gen_get_header(model, mapper, class_name):
    lines = []
    lines.append(f"std::string {class_name}::getHeader(){{")
    lines.append("")
    lines.append('    std::string s = "";')

    for sp in model.sp_var:
        name = mapper.display_name(sp.id)
        lines.append(f'    s += ",{name}";')

    for sp in model.sp_other:
        name = mapper.display_name(sp.id)
        lines.append(f'    s += ",{name}";')

    lines.append("")
    lines.append("    return s;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# Unit conversion functions
# ============================================================================

def _gen_unit_conversion(model, mapper, class_name):
    lines = []

    # get_unit_conversion_species
    lines.append(f"realtype {class_name}::get_unit_conversion_species(int i) const{{")
    lines.append("")
    lines.append("    static std::vector<realtype> scalor = {")
    lines.append("        //sp_var")
    for sp in model.sp_var:
        lines.append(f"        {_fmt(sp.unit_si_scaling)},")
    lines.append("        //sp_other")
    for sp in model.sp_other:
        lines.append(f"        {_fmt(sp.unit_si_scaling)},")
    lines.append("    };")
    lines.append("    return scalor[i];")
    lines.append("}")
    lines.append("")

    # get_unit_conversion_nspvar
    lines.append(f"realtype {class_name}::get_unit_conversion_nspvar(int i) const{{")
    lines.append("")
    lines.append("    static std::vector<realtype> scalor = {")
    for nsp in model.nsp_var:
        scaling = nsp.unit_si_scaling if hasattr(nsp, 'unit_si_scaling') else 1.0
        lines.append(f"        {_fmt(scaling)},")
    lines.append("    };")
    lines.append("    return scalor[i];")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)
