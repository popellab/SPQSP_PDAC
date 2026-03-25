"""Generate QSPParam.h — enum-driven parameter system."""

from sbml_parser import SBMLModel
from name_mapper import NameMapper


def generate_qsp_param_h(model: SBMLModel, mapper: NameMapper, namespace: str) -> str:
    lines = []
    lines.append("#ifndef __CancerVCT_QSPPARAM__")
    lines.append("#define __CancerVCT_QSPPARAM__")
    lines.append("")
    lines.append('#include "../../core/ParamBase.h"')
    lines.append("#include <string>")
    lines.append("#include <map>")
    lines.append("")
    lines.append(f"namespace {namespace} {{")
    lines.append("")

    # QSPParamFloat enum
    lines.append("enum QSPParamFloat {")
    lines.append("    //simulation settings")
    lines.append("    QSP_SIM_START,")
    lines.append("    QSP_SIM_STEP,")
    lines.append("    QSP_SIM_N_STEP,")
    lines.append("    QSP_SIM_TOL_REL,")
    lines.append("    QSP_SIM_TOL_ABS,")

    # Compartments
    lines.append("    //compartments")
    for comp in model.compartments:
        lines.append(f"    {mapper.qsp_enum(comp.id)},")

    # Species initial values
    lines.append("    //species")
    for sp in model.sp_var + model.sp_other:
        lines.append(f"    {mapper.qsp_enum(sp.id)},")

    # Parameters
    lines.append("    //parameters")
    for param in model.p_const:
        lines.append(f"    {mapper.qsp_enum(param.id)},")

    lines.append("    QSP_PARAM_FLOAT_COUNT")
    lines.append("};")
    lines.append("")

    # QSPParamInt enum (minimal)
    lines.append("enum QSPParamInt {")
    lines.append("    QSP_PARAM_INT_COUNT")
    lines.append("};")
    lines.append("")

    # Class definition
    lines.append("class QSPParam : public SP_QSP_IO::ParamBase")
    lines.append("{")
    lines.append("public:")
    lines.append("    QSPParam();")
    lines.append("    ~QSPParam(){};")
    lines.append("")
    lines.append("    //! get float parameter value by enum")
    lines.append("    inline double getFloat(QSPParamFloat idx) const {")
    lines.append("        return _paramFloat[idx];")
    lines.append("    };")
    lines.append("")
    lines.append("    //! get int parameter value by enum")
    lines.append("    inline int getInt(QSPParamInt idx) const {")
    lines.append("        return static_cast<int>(_paramFloat[idx]);")
    lines.append("    };")
    lines.append("")
    lines.append("    //! get parameter value by index (for compatibility)")
    lines.append("    inline double getVal(unsigned int n) const { return _paramFloat[n]; };")
    lines.append("")
    lines.append("private:")
    lines.append("    //! setup content of _paramDesc")
    lines.append("    virtual void setupParam();")
    lines.append("    //! process all internal parameters")
    lines.append("    virtual void processInternalParams(){};")
    lines.append("};")
    lines.append("")
    lines.append("// XML path array declaration")
    lines.append("extern const char* QSP_PARAM_FLOAT_XML_PATHS[];")
    lines.append("")
    lines.append("};")
    lines.append("#endif")
    lines.append("")

    return "\n".join(lines)
