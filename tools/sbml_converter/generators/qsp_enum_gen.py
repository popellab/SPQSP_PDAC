"""Generate QSP_enum.h — species and parameter enumerations."""

from sbml_parser import SBMLModel
from name_mapper import NameMapper


def generate_qsp_enum_h(model: SBMLModel, mapper: NameMapper, namespace: str) -> str:
    lines = []
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <utility>")
    lines.append("")
    lines.append(f"namespace {namespace}{{")
    lines.append("")

    # Species enum
    lines.append("// QSP Species Enum")
    lines.append("enum QSPSpeciesEnum")
    lines.append("{")
    for sp in model.sp_var:
        lines.append(f"{mapper.species_enum(sp.id)},")
    for sp in model.sp_other:
        lines.append(f"{mapper.species_enum(sp.id)},")
    lines.append("SP_species_count")
    lines.append("};")
    lines.append("")

    # Parameter enum (class parameters: compartments + constant parameters)
    lines.append("// QSP Parameter Enum (class parameters)")
    lines.append("enum QSPParamEnum")
    lines.append("{")
    for comp in model.compartments:
        lines.append(f"{mapper.param_enum(comp.id)},")
    for param in model.p_const:
        lines.append(f"{mapper.param_enum(param.id)},")
    lines.append("P_param_count")
    lines.append("};")
    lines.append("")

    lines.append(f"}};")
    lines.append("")

    return "\n".join(lines)
