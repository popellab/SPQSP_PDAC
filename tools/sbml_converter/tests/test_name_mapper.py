"""Tests for name mapper."""

import pytest


def test_species_enum_format(parsed_model, mapper):
    """Species enums should be SP_ + compartment_name + _ + species_name."""
    for sp in parsed_model.sp_var[:5]:
        enum = mapper.species_enum(sp.id)
        assert enum.startswith("SP_")
        assert "." not in enum  # dots replaced with underscores


def test_known_species_enums(parsed_model, mapper):
    """Check specific expected enum names."""
    by_name = {}
    for sp in parsed_model.sp_var:
        display = mapper.display_name(sp.id)
        by_name[display] = mapper.species_enum(sp.id)

    assert by_name.get("V_T.C1") == "SP_V_T_C1"
    assert by_name.get("V_T.CD8") == "SP_V_T_CD8"
    assert by_name.get("V_C.aPD1") == "SP_V_C_aPD1"


def test_param_enum_format(parsed_model, mapper):
    for comp in parsed_model.compartments:
        enum = mapper.param_enum(comp.id)
        assert enum.startswith("P_")


def test_compartment_param_enums(parsed_model, mapper):
    by_name = {c.name: mapper.param_enum(c.id) for c in parsed_model.compartments}
    assert by_name["V_C"] == "P_V_C"
    assert by_name["V_T"] == "P_V_T"


def test_qsp_enum_format(parsed_model, mapper):
    for comp in parsed_model.compartments:
        enum = mapper.qsp_enum(comp.id)
        assert enum.startswith("QSP_")


def test_xml_paths(parsed_model, mapper):
    for comp in parsed_model.compartments:
        path = mapper.xml_path(comp.id)
        assert path.startswith("Param.QSP.init_value.Compartment.")

    for sp in parsed_model.sp_var[:3]:
        path = mapper.xml_path(sp.id)
        assert path.startswith("Param.QSP.init_value.Species.")


def test_aux_var_names(parsed_model, mapper):
    for ar in parsed_model.assignment_rules:
        aux = mapper.aux_var(ar.variable_id)
        assert aux.startswith("AUX_VAR_")


def test_resolve_name_species(parsed_model, mapper):
    """Species should resolve to SPVAR() in f context."""
    sp = parsed_model.sp_var[0]
    resolved = mapper.resolve_name(sp.id, "f")
    assert resolved.startswith("SPVAR(SP_")


def test_resolve_name_event_context(parsed_model, mapper):
    """Species should resolve to NV_DATA_S(_y)[] in event context."""
    sp = parsed_model.sp_var[0]
    resolved = mapper.resolve_name(sp.id, "event")
    assert "NV_DATA_S(_y)" in resolved


def test_no_duplicate_enums(parsed_model, mapper):
    """All species enum names should be unique."""
    enums = set()
    for sp in parsed_model.sp_var + parsed_model.sp_other:
        enum = mapper.species_enum(sp.id)
        assert enum not in enums, f"Duplicate species enum: {enum}"
        enums.add(enum)

    # All param enums should be unique
    enums = set()
    for comp in parsed_model.compartments:
        enum = mapper.param_enum(comp.id)
        assert enum not in enums, f"Duplicate param enum: {enum}"
        enums.add(enum)
    for p in parsed_model.p_const:
        enum = mapper.param_enum(p.id)
        assert enum not in enums, f"Duplicate param enum: {enum}"
        enums.add(enum)
