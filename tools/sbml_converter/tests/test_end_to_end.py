"""End-to-end generation tests — structural checks on output."""

import re
import pytest


@pytest.fixture(scope="session")
def generated_files(model_with_sorting, mapper):
    """Generate all C++ files and return as dict of name -> content."""
    from generators.qsp_enum_gen import generate_qsp_enum_h
    from generators.ode_system_h_gen import generate_ode_system_h
    from generators.ode_system_cpp_gen import generate_ode_system_cpp
    from generators.qsp_param_h_gen import generate_qsp_param_h
    from generators.qsp_param_cpp_gen import generate_qsp_param_cpp
    from generators.param_xml_gen import generate_param_xml

    ns = "CancerVCT"
    cn = "ODE_system"
    return {
        "QSP_enum.h": generate_qsp_enum_h(model_with_sorting, mapper, ns),
        "ODE_system.h": generate_ode_system_h(ns, cn),
        "ODE_system.cpp": generate_ode_system_cpp(model_with_sorting, mapper, ns, cn),
        "QSPParam.h": generate_qsp_param_h(model_with_sorting, mapper, ns),
        "QSPParam.cpp": generate_qsp_param_cpp(model_with_sorting, mapper, ns),
        "param_all.xml": generate_param_xml(model_with_sorting, mapper),
    }


# ---- QSP_enum.h ----

def test_enum_h_has_species_count(generated_files):
    content = generated_files["QSP_enum.h"]
    assert "SP_species_count" in content


def test_enum_h_has_param_count(generated_files):
    content = generated_files["QSP_enum.h"]
    assert "P_param_count" in content


def test_enum_h_species_count_value(generated_files, model_with_sorting):
    """SP_species_count should equal sp_var + sp_other."""
    content = generated_files["QSP_enum.h"]
    # Count enum entries before SP_species_count
    enum_section = content.split("enum QSPSpeciesEnum")[1].split("};")[0]
    entries = [line.strip().rstrip(",") for line in enum_section.split("\n")
               if line.strip() and not line.strip().startswith("//")
               and line.strip() != "{" and "SP_species_count" not in line]
    expected = len(model_with_sorting.sp_var) + len(model_with_sorting.sp_other)
    assert len(entries) == expected


# ---- ODE_system.h ----

def test_ode_h_has_class(generated_files):
    content = generated_files["ODE_system.h"]
    assert "class ODE_system" in content
    assert "CVODEBase" in content


def test_ode_h_has_required_methods(generated_files):
    content = generated_files["ODE_system.h"]
    for method in ["static int f(", "static int g(", "setup_class_parameters",
                    "setup_instance_variables", "setup_instance_tolerance",
                    "setupVariables", "setupEvents", "update_y_other"]:
        assert method in content, f"Missing method: {method}"


# ---- ODE_system.cpp ----

def test_ode_cpp_has_macros(generated_files):
    content = generated_files["ODE_system.cpp"]
    for macro in ["#define SPVAR", "#define PARAM", "#define PFILE", "#define QSP_W"]:
        assert macro in content


def test_ode_cpp_has_f_function(generated_files):
    content = generated_files["ODE_system.cpp"]
    assert "int ODE_system::f(realtype t, N_Vector y, N_Vector ydot" in content


def test_ode_cpp_has_reaction_fluxes(generated_files, model_with_sorting):
    content = generated_files["ODE_system.cpp"]
    n_reactions = len(model_with_sorting.reactions)
    assert f"ReactionFlux{n_reactions}" in content


def test_ode_cpp_has_ydot_equations(generated_files, model_with_sorting):
    content = generated_files["ODE_system.cpp"]
    ydot_count = content.count("NV_DATA_S(ydot)")
    assert ydot_count == len(model_with_sorting.sp_var)


def test_ode_cpp_has_assignment_rules(generated_files, model_with_sorting):
    content = generated_files["ODE_system.cpp"]
    for ar in model_with_sorting.assignment_rule_order[:5]:
        assert f"AUX_VAR_{ar.variable_name.replace('.', '_')}" in content


def test_ode_cpp_sundials_7_api(generated_files):
    """Verify SUNDIALS 7.6.0 API is used (N_VNew_Serial with _sunctx)."""
    content = generated_files["ODE_system.cpp"]
    assert "N_VNew_Serial(_neq, _sunctx)" in content


def test_ode_cpp_has_events(generated_files, model_with_sorting):
    content = generated_files["ODE_system.cpp"]
    n_events = len(model_with_sorting.events)
    assert f"_nevent = {n_events}" in content


def test_ode_cpp_has_header_function(generated_files):
    content = generated_files["ODE_system.cpp"]
    assert "ODE_system::getHeader()" in content
    assert 'std::string s = ""' in content


def test_ode_cpp_has_unit_conversion(generated_files):
    content = generated_files["ODE_system.cpp"]
    assert "get_unit_conversion_species" in content
    assert "scalor" in content


# ---- QSPParam.h ----

def test_param_h_has_enum(generated_files):
    content = generated_files["QSPParam.h"]
    assert "enum QSPParamFloat" in content
    assert "QSP_SIM_START" in content
    assert "QSP_PARAM_FLOAT_COUNT" in content


def test_param_h_has_class(generated_files):
    content = generated_files["QSPParam.h"]
    assert "class QSPParam" in content
    assert "getVal" in content
    assert "getFloat" in content


# ---- QSPParam.cpp ----

def test_param_cpp_has_xml_paths(generated_files):
    content = generated_files["QSPParam.cpp"]
    assert "QSP_PARAM_FLOAT_XML_PATHS" in content
    assert "Param.QSP.simulation.start" in content
    assert "Param.QSP.init_value.Compartment" in content
    assert "Param.QSP.init_value.Species" in content
    assert "Param.QSP.init_value.Parameter" in content


def test_param_cpp_has_static_assert(generated_files):
    content = generated_files["QSPParam.cpp"]
    assert "static_assert" in content
    assert "QSP_PARAM_FLOAT_COUNT" in content


# ---- param_all.xml ----

def test_xml_has_structure(generated_files):
    content = generated_files["param_all.xml"]
    assert "<Param>" in content
    assert "<QSP>" in content
    assert "<simulation>" in content
    assert "<Compartment>" in content
    assert "<Species>" in content
    assert "<Parameter>" in content


def test_xml_has_compartment_values(generated_files):
    content = generated_files["param_all.xml"]
    assert "<V_C>5.0</V_C>" in content
    assert "<V_P>60.0</V_P>" in content


# ---- Cross-file consistency ----

def test_enum_count_matches_xml_paths(generated_files):
    """Number of QSP enum entries should match number of XML paths."""
    param_h = generated_files["QSPParam.h"]
    param_cpp = generated_files["QSPParam.cpp"]

    # Count enum entries (lines with QSP_ prefix)
    enum_section = param_h.split("enum QSPParamFloat")[1].split("};")[0]
    enum_lines = [l.strip() for l in enum_section.split("\n")
                  if l.strip().startswith("QSP_") and l.strip() != "QSP_PARAM_FLOAT_COUNT"]
    n_enum = len(enum_lines)

    # Count XML path entries
    path_section = param_cpp.split("QSP_PARAM_FLOAT_XML_PATHS")[1].split("};")[0]
    path_lines = [l for l in path_section.split("\n") if '"Param.' in l]
    n_paths = len(path_lines)

    assert n_enum == n_paths, f"Enum has {n_enum} entries but XML paths has {n_paths}"


def test_species_enum_count_matches_ydot(generated_files, model_with_sorting):
    """Number of species in enum should match ydot assignments in f()."""
    enum_h = generated_files["QSP_enum.h"]
    ode_cpp = generated_files["ODE_system.cpp"]

    enum_section = enum_h.split("enum QSPSpeciesEnum")[1].split("};")[0]
    sp_entries = [l.strip() for l in enum_section.split("\n")
                  if l.strip().startswith("SP_") and "SP_species_count" not in l]

    ydot_count = ode_cpp.count("NV_DATA_S(ydot)")

    # ydot should equal sp_var count (not sp_other)
    assert ydot_count == len(model_with_sorting.sp_var)
    # enum should include both sp_var and sp_other
    assert len(sp_entries) == len(model_with_sorting.sp_var) + len(model_with_sorting.sp_other)
