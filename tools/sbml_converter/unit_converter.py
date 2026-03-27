"""Compute SI unit conversion factors from SBML unit definitions."""

import libsbml
from sbml_parser import SBMLModel


def _unit_definition_scaling(ud: libsbml.UnitDefinition) -> float:
    """Compute the SI scaling factor for a unit definition.

    The scaling factor converts from the unit's base representation to SI.
    For each unit component: factor = (multiplier * 10^scale)^exponent
    Total factor = product of all component factors.
    """
    if ud is None or ud.getNumUnits() == 0:
        return 1.0

    total_scale = 1.0
    for i in range(ud.getNumUnits()):
        u = ud.getUnit(i)
        multiplier = u.getMultiplier()
        scale = u.getScale()
        exponent = u.getExponentAsDouble()
        total_scale *= (multiplier * (10.0 ** scale)) ** exponent

    return total_scale


def compute_unit_conversions(model: SBMLModel):
    """Compute SI scaling factors for all model elements.

    Populates the unit_si_scaling field on each compartment, species, and parameter.
    """
    sbml_model = model._sbml_model

    # Compartments
    for comp in model.compartments:
        sbml_comp = sbml_model.getCompartment(comp.id)
        if sbml_comp is not None:
            ud = sbml_comp.getDerivedUnitDefinition()
            if ud is not None:
                # Convert to SI
                ud_si = libsbml.UnitDefinition.convertToSI(ud)
                comp.unit_si_scaling = _unit_definition_scaling(ud_si)

    # Species
    for sp in model.species:
        sbml_sp = sbml_model.getSpecies(sp.id)
        if sbml_sp is not None:
            ud = sbml_sp.getDerivedUnitDefinition()
            if ud is not None:
                ud_si = libsbml.UnitDefinition.convertToSI(ud)
                sp.unit_si_scaling = _unit_definition_scaling(ud_si)

    # Parameters
    for param in model.parameters:
        sbml_param = sbml_model.getParameter(param.id)
        if sbml_param is not None:
            ud = sbml_param.getDerivedUnitDefinition()
            if ud is not None:
                ud_si = libsbml.UnitDefinition.convertToSI(ud)
                param.unit_si_scaling = _unit_definition_scaling(ud_si)

    # Also compute for p_const (which are a subset of parameters)
    # Already done above since p_const elements reference the same Parameter objects


def get_model_time_scale(sbml_model) -> float:
    """Extract the SI scale of the model's time unit.

    Returns the factor to convert from model time to SI seconds.
    E.g., for day -> second, returns 86400.
    """
    if sbml_model.isSetTimeUnits():
        time_units_id = sbml_model.getTimeUnits()
        ud = sbml_model.getUnitDefinition(time_units_id)
        if ud is not None:
            ud_si = libsbml.UnitDefinition.convertToSI(ud)
            return _unit_definition_scaling(ud_si)

    # For SBML L2 models (e.g. SimBiology exports), timeUnits is not set.
    # Infer from rate constant parameters: look for units containing "day".
    for i in range(sbml_model.getNumParameters()):
        p = sbml_model.getParameter(i)
        units_id = p.getUnits() if p.isSetUnits() else ""
        if "day" in units_id.lower():
            # Found a day-based parameter. Extract the day unit definition.
            # Look for MWBUILTINUNIT_day or similar
            day_ud = sbml_model.getUnitDefinition("MWBUILTINUNIT_day")
            if day_ud is not None:
                ud_si = libsbml.UnitDefinition.convertToSI(day_ud)
                return _unit_definition_scaling(ud_si)
            # Fallback: day = 86400 seconds
            return 86400.0

    # Default: second (SI), scale = 1.0
    return 1.0
