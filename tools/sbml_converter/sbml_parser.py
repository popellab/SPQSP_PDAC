"""Parse SBML model into structured data for code generation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import libsbml


@dataclass
class Compartment:
    id: str
    name: str
    size: float
    constant: bool
    unit_def_id: str  # SBML unit definition ID
    unit_si_scaling: float = 1.0  # computed later by unit_converter


@dataclass
class Species:
    id: str
    name: str
    compartment_id: str
    compartment_name: str
    initial_amount: float
    has_only_substance_units: bool
    boundary_condition: bool
    constant: bool
    unit_def_id: str
    unit_si_scaling: float = 1.0
    # Classification (set during analysis)
    is_ode_var: bool = False  # in stoichiometry → part of ODE LHS
    is_other: bool = False  # species_other (not in ODE, computed via rules)


@dataclass
class Parameter:
    id: str
    name: str
    value: float
    constant: bool
    unit_def_id: str
    unit_si_scaling: float = 1.0


@dataclass
class SpeciesReference:
    species_id: str
    stoichiometry: float


@dataclass
class Reaction:
    id: str
    name: str
    reactants: list  # list of SpeciesReference
    products: list  # list of SpeciesReference
    kinetic_law_ast: object  # libsbml ASTNode
    kinetic_law_unit_def: object  # for unit checking
    reversible: bool


@dataclass
class AssignmentRule:
    variable_id: str  # SBML ID of variable being assigned
    variable_name: str  # human-readable name
    math_ast: object  # libsbml ASTNode


@dataclass
class InitialAssignment:
    symbol_id: str
    symbol_name: str
    math_ast: object


@dataclass
class EventAssignment:
    variable_id: str
    variable_name: str
    math_ast: object


@dataclass
class Event:
    id: str
    name: str
    trigger_ast: object  # libsbml ASTNode
    trigger_initial_value: bool
    trigger_persistent: bool
    event_assignments: list  # list of EventAssignment
    has_delay: bool = False
    delay_ast: object = None


@dataclass
class SBMLModel:
    """Parsed SBML model with all elements extracted."""

    name: str
    compartments: list  # list of Compartment
    species: list  # list of Species
    parameters: list  # list of Parameter
    reactions: list  # list of Reaction
    assignment_rules: list  # list of AssignmentRule
    initial_assignments: list  # list of InitialAssignment
    events: list  # list of Event

    # Lookups (built during parsing)
    id_to_compartment: dict = field(default_factory=dict)
    id_to_species: dict = field(default_factory=dict)
    id_to_parameter: dict = field(default_factory=dict)
    id_to_element: dict = field(default_factory=dict)  # any element by ID

    # Stoichiometry map: species_id -> [(reaction_index, coefficient)]
    stoichiometry: dict = field(default_factory=dict)

    # Classification results (populated during analysis)
    sp_var: list = field(default_factory=list)  # ODE species
    sp_other: list = field(default_factory=list)  # derived species (not in ODE)
    nsp_var: list = field(default_factory=list)  # non-species variables
    p_const: list = field(default_factory=list)  # constant parameters (class params)

    # Sorted orders (populated by dependency_sorter)
    assignment_rule_order: list = field(default_factory=list)
    initial_assignment_order: list = field(default_factory=list)

    # Raw libsbml model (for unit queries)
    _sbml_model: object = None


def parse_sbml(path: Path) -> SBMLModel:
    """Parse an SBML file and return structured model data."""
    doc = libsbml.readSBMLFromFile(str(path))

    # Check for errors
    if doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR) > 0:
        errors = []
        for i in range(doc.getNumErrors()):
            err = doc.getError(i)
            if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
                errors.append(f"  Line {err.getLine()}: {err.getMessage()}")
        if errors:
            raise ValueError(f"SBML parsing errors:\n" + "\n".join(errors))

    sbml_model = doc.getModel()
    if sbml_model is None:
        raise ValueError("No model found in SBML file")

    # Extract compartments
    compartments = []
    id_to_compartment = {}
    for i in range(sbml_model.getNumCompartments()):
        c = sbml_model.getCompartment(i)
        comp = Compartment(
            id=c.getId(),
            name=c.getName() or c.getId(),
            size=c.getSize(),
            constant=c.getConstant(),
            unit_def_id=c.getUnits() if c.isSetUnits() else "",
        )
        compartments.append(comp)
        id_to_compartment[comp.id] = comp

    # Extract species
    species_list = []
    id_to_species = {}
    for i in range(sbml_model.getNumSpecies()):
        s = sbml_model.getSpecies(i)
        comp_id = s.getCompartment()
        comp_name = id_to_compartment[comp_id].name if comp_id in id_to_compartment else comp_id

        if s.isSetInitialAmount():
            init_val = s.getInitialAmount()
        elif s.isSetInitialConcentration():
            init_val = s.getInitialConcentration()
        else:
            init_val = 0.0

        sp = Species(
            id=s.getId(),
            name=s.getName() or s.getId(),
            compartment_id=comp_id,
            compartment_name=comp_name,
            initial_amount=init_val,
            has_only_substance_units=s.getHasOnlySubstanceUnits(),
            boundary_condition=s.getBoundaryCondition(),
            constant=s.getConstant(),
            unit_def_id=s.getSubstanceUnits() if s.isSetSubstanceUnits() else "",
        )
        species_list.append(sp)
        id_to_species[sp.id] = sp

    # Extract parameters
    parameters = []
    id_to_parameter = {}
    for i in range(sbml_model.getNumParameters()):
        p = sbml_model.getParameter(i)
        param = Parameter(
            id=p.getId(),
            name=p.getName() or p.getId(),
            value=p.getValue() if p.isSetValue() else 0.0,
            constant=p.getConstant(),
            unit_def_id=p.getUnits() if p.isSetUnits() else "",
        )
        parameters.append(param)
        id_to_parameter[param.id] = param

    # Extract reactions
    reactions = []
    for i in range(sbml_model.getNumReactions()):
        r = sbml_model.getReaction(i)

        reactants = []
        for j in range(r.getNumReactants()):
            sr = r.getReactant(j)
            reactants.append(SpeciesReference(
                species_id=sr.getSpecies(),
                stoichiometry=sr.getStoichiometry() if sr.isSetStoichiometry() else 1.0,
            ))

        products = []
        for j in range(r.getNumProducts()):
            sr = r.getProduct(j)
            products.append(SpeciesReference(
                species_id=sr.getSpecies(),
                stoichiometry=sr.getStoichiometry() if sr.isSetStoichiometry() else 1.0,
            ))

        kl = r.getKineticLaw()
        reactions.append(Reaction(
            id=r.getId(),
            name=r.getName() or r.getId(),
            reactants=reactants,
            products=products,
            kinetic_law_ast=kl.getMath().deepCopy() if kl and kl.getMath() else None,
            kinetic_law_unit_def=None,
            reversible=r.getReversible(),
        ))

    # Extract assignment rules
    assignment_rules = []
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        if rule.isAssignment():
            var_id = rule.getVariable()
            # Look up name
            var_name = var_id
            if var_id in id_to_species:
                var_name = id_to_species[var_id].name
            elif var_id in id_to_parameter:
                var_name = id_to_parameter[var_id].name
            elif var_id in id_to_compartment:
                var_name = id_to_compartment[var_id].name

            assignment_rules.append(AssignmentRule(
                variable_id=var_id,
                variable_name=var_name,
                math_ast=rule.getMath().deepCopy() if rule.getMath() else None,
            ))

    # Extract initial assignments
    initial_assignments = []
    for i in range(sbml_model.getNumInitialAssignments()):
        ia = sbml_model.getInitialAssignment(i)
        sym_id = ia.getSymbol()
        sym_name = sym_id
        if sym_id in id_to_species:
            sym_name = id_to_species[sym_id].name
        elif sym_id in id_to_parameter:
            sym_name = id_to_parameter[sym_id].name
        elif sym_id in id_to_compartment:
            sym_name = id_to_compartment[sym_id].name

        initial_assignments.append(InitialAssignment(
            symbol_id=sym_id,
            symbol_name=sym_name,
            math_ast=ia.getMath().deepCopy() if ia.getMath() else None,
        ))

    # Extract events
    events = []
    for i in range(sbml_model.getNumEvents()):
        e = sbml_model.getEvent(i)
        trigger = e.getTrigger()

        eas = []
        for j in range(e.getNumEventAssignments()):
            ea = e.getEventAssignment(j)
            var_id = ea.getVariable()
            var_name = var_id
            if var_id in id_to_species:
                var_name = id_to_species[var_id].name
            elif var_id in id_to_parameter:
                var_name = id_to_parameter[var_id].name

            eas.append(EventAssignment(
                variable_id=var_id,
                variable_name=var_name,
                math_ast=ea.getMath().deepCopy() if ea.getMath() else None,
            ))

        events.append(Event(
            id=e.getId(),
            name=e.getName() or e.getId(),
            trigger_ast=trigger.getMath().deepCopy() if trigger and trigger.getMath() else None,
            trigger_initial_value=trigger.getInitialValue() if trigger else True,
            trigger_persistent=trigger.getPersistent() if trigger else True,
            event_assignments=eas,
            has_delay=e.isSetDelay(),
            delay_ast=e.getDelay().getMath().deepCopy() if e.isSetDelay() and e.getDelay().getMath() else None,
        ))

    # Build ID-to-element lookup
    id_to_element = {}
    for c in compartments:
        id_to_element[c.id] = c
    for s in species_list:
        id_to_element[s.id] = s
    for p in parameters:
        id_to_element[p.id] = p

    # Build model
    model = SBMLModel(
        name=sbml_model.getName() or sbml_model.getId(),
        compartments=compartments,
        species=species_list,
        parameters=parameters,
        reactions=reactions,
        assignment_rules=assignment_rules,
        initial_assignments=initial_assignments,
        events=events,
        id_to_compartment=id_to_compartment,
        id_to_species=id_to_species,
        id_to_parameter=id_to_parameter,
        id_to_element=id_to_element,
        _sbml_model=sbml_model,
    )

    # Build stoichiometry map and classify species
    _build_stoichiometry(model)
    _classify_variables(model)

    return model


def _build_stoichiometry(model: SBMLModel):
    """Build stoichiometry map: species_id -> [(reaction_index, coefficient)]."""
    stoich = {}
    for rxn_idx, rxn in enumerate(model.reactions):
        for sr in rxn.reactants:
            if sr.species_id not in stoich:
                stoich[sr.species_id] = []
            stoich[sr.species_id].append((rxn_idx, -sr.stoichiometry))

        for sr in rxn.products:
            if sr.species_id not in stoich:
                stoich[sr.species_id] = []
            stoich[sr.species_id].append((rxn_idx, sr.stoichiometry))

    model.stoichiometry = stoich


def _classify_variables(model: SBMLModel):
    """Classify species and parameters into categories for code generation."""
    # Set of variable IDs assigned by assignment rules
    rule_targets = {ar.variable_id for ar in model.assignment_rules}

    # Set of variable IDs assigned by events
    event_targets = set()
    for ev in model.events:
        for ea in ev.event_assignments:
            event_targets.add(ea.variable_id)

    # Species classification
    for sp in model.species:
        if sp.id in model.stoichiometry and sp.id not in rule_targets:
            # Has reactions and is not overridden by a rule → ODE variable
            sp.is_ode_var = True
            model.sp_var.append(sp)
        elif sp.id in rule_targets:
            # Computed by assignment rule → species_other
            sp.is_other = True
            model.sp_other.append(sp)
        elif sp.boundary_condition or sp.constant:
            # Boundary or constant species that aren't in reactions
            pass
        else:
            # Species with no reactions and no rules — still include as ODE var
            # (will have ydot = 0, but needs to be tracked)
            sp.is_ode_var = True
            model.sp_var.append(sp)

    # Parameter classification: constant parameters become class params
    # Non-constant parameters that are rule targets are assignment rule variables
    # Non-constant parameters with event assignments become nsp_var
    for param in model.parameters:
        if param.id in rule_targets:
            continue  # handled as assignment rule variable
        if not param.constant and param.id in event_targets:
            model.nsp_var.append(param)
        else:
            model.p_const.append(param)

    # Compartments that are non-constant and rule targets are also skipped
    # Compartments that have event assignments become nsp_var
    for comp in model.compartments:
        if comp.id in rule_targets:
            continue
        if not comp.constant and comp.id in event_targets:
            model.nsp_var.append(comp)
