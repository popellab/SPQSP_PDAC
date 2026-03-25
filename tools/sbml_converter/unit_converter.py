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
