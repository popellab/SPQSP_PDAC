"""Generate default parameter XML file from SBML model values."""

from sbml_parser import SBMLModel
from name_mapper import NameMapper


def generate_param_xml(model: SBMLModel, mapper: NameMapper) -> str:
    """Generate the QSP section of the parameter XML file."""
    lines = []
    lines.append('<?xml version="1.0" encoding="utf-8"?>')
    lines.append("<Param>")
    lines.append("  <QSP>")

    # Simulation settings
    lines.append("    <simulation>")
    lines.append("      <start>0</start>")
    lines.append("      <step>1</step>")
    lines.append("      <n_step>360</n_step>")
    lines.append("      <tol_rel>1e-06</tol_rel>")
    lines.append("      <tol_abs>1e-09</tol_abs>")
    lines.append("    </simulation>")

    # Initial values
    lines.append("    <init_value>")

    # Compartments
    lines.append("      <Compartment>")
    for comp in model.compartments:
        tag = mapper._sanitize(comp.name)
        lines.append(f"        <{tag}>{comp.size}</{tag}>")
    lines.append("      </Compartment>")

    # Species
    lines.append("      <Species>")
    for sp in model.sp_var + model.sp_other:
        tag = mapper._sanitize(mapper._species_display_name(sp))
        lines.append(f"        <{tag}>{sp.initial_amount}</{tag}>")
    lines.append("      </Species>")

    # Parameters
    lines.append("      <Parameter>")
    for param in model.p_const:
        tag = mapper._sanitize(param.name)
        lines.append(f"        <{tag}>{param.value}</{tag}>")
    lines.append("      </Parameter>")

    lines.append("    </init_value>")
    lines.append("  </QSP>")
    lines.append("</Param>")
    lines.append("")

    return "\n".join(lines)
