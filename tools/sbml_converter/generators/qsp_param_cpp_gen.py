"""Generate QSPParam.cpp — XML path mappings and parameter loading."""

from sbml_parser import SBMLModel
from name_mapper import NameMapper


def generate_qsp_param_cpp(model: SBMLModel, mapper: NameMapper, namespace: str) -> str:
    lines = []
    lines.append('#include "QSPParam.h"')
    lines.append("#include <iostream>")
    lines.append("#include <boost/property_tree/ptree.hpp>")
    lines.append("#include <boost/property_tree/xml_parser.hpp>")
    lines.append("")
    lines.append(f"namespace {namespace}{{")
    lines.append("")

    # XML path array
    lines.append("const char* QSP_PARAM_FLOAT_XML_PATHS[] = {")

    idx = 0

    # Simulation settings (hardcoded)
    lines.append("    //simulation settings")
    sim_paths = [
        "Param.QSP.simulation.start",
        "Param.QSP.simulation.step",
        "Param.QSP.simulation.n_step",
        "Param.QSP.simulation.tol_rel",
        "Param.QSP.simulation.tol_abs",
    ]
    for path in sim_paths:
        lines.append(f'    "{path}",//{idx}')
        idx += 1

    # Compartments
    lines.append("    //compartments")
    for comp in model.compartments:
        lines.append(f'    "{mapper.xml_path(comp.id)}",//{idx}')
        idx += 1

    # Species
    lines.append("    //species")
    for sp in model.sp_var + model.sp_other:
        lines.append(f'    "{mapper.xml_path(sp.id)}",//{idx}')
        idx += 1

    # Parameters
    lines.append("    //parameters")
    for param in model.p_const:
        lines.append(f'    "{mapper.xml_path(param.id)}",//{idx}')
        idx += 1

    lines.append("};")
    lines.append("")

    # Static assert
    lines.append(f"static_assert(")
    lines.append(f"    sizeof(QSP_PARAM_FLOAT_XML_PATHS) / sizeof(QSP_PARAM_FLOAT_XML_PATHS[0])")
    lines.append(f"    == QSP_PARAM_FLOAT_COUNT,")
    lines.append(f'    "XML path array size must match QSP_PARAM_FLOAT_COUNT");')
    lines.append("")

    # Constructor
    lines.append("QSPParam::QSPParam()")
    lines.append("    :ParamBase()")
    lines.append("{")
    lines.append("    setupParam();")
    lines.append("}")
    lines.append("")

    # setupParam
    lines.append("void QSPParam::setupParam(){")
    lines.append(f"    for (size_t i = 0; i < QSP_PARAM_FLOAT_COUNT; i++)")
    lines.append("    {")
    lines.append("        _paramDesc.push_back(std::vector<std::string>({")
    lines.append("            QSP_PARAM_FLOAT_XML_PATHS[i], \"\", \"\"}));")
    lines.append("    }")
    lines.append(f"    _paramFloat = std::vector<double>(QSP_PARAM_FLOAT_COUNT, 0);")
    lines.append("}")
    lines.append("")

    lines.append("};")
    lines.append("")

    return "\n".join(lines)
