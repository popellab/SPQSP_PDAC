"""Tests for SBML parser."""


def test_compartment_count(parsed_model):
    assert len(parsed_model.compartments) == 11


def test_species_count(parsed_model):
    assert len(parsed_model.species) == 164


def test_parameter_count(parsed_model):
    assert len(parsed_model.parameters) == 398


def test_reaction_count(parsed_model):
    assert len(parsed_model.reactions) == 283


def test_assignment_rule_count(parsed_model):
    assert len(parsed_model.assignment_rules) == 65


def test_event_count(parsed_model):
    assert len(parsed_model.events) == 2


def test_species_classification(parsed_model):
    assert len(parsed_model.sp_var) == 162
    assert len(parsed_model.sp_other) == 2
    assert len(parsed_model.nsp_var) == 0


def test_param_classification(parsed_model):
    assert len(parsed_model.p_const) == 336


def test_compartment_names(parsed_model):
    names = [c.name for c in parsed_model.compartments]
    assert "V_C" in names
    assert "V_P" in names
    assert "V_T" in names
    assert "V_LN" in names


def test_compartment_values(parsed_model):
    by_name = {c.name: c for c in parsed_model.compartments}
    assert by_name["V_C"].size == 5.0
    assert by_name["V_P"].size == 60.0


def test_species_have_compartments(parsed_model):
    for sp in parsed_model.species:
        assert sp.compartment_id, f"Species {sp.name} has no compartment"
        assert sp.compartment_id in parsed_model.id_to_compartment


def test_stoichiometry_built(parsed_model):
    """At least some species should have stoichiometry entries."""
    assert len(parsed_model.stoichiometry) > 0
    # C_x should be in reactions (dead cell clearance)
    c_x = [sp for sp in parsed_model.sp_var if sp.name == "C_x"]
    assert len(c_x) == 1
    assert c_x[0].id in parsed_model.stoichiometry


def test_reactions_have_kinetic_laws(parsed_model):
    for rxn in parsed_model.reactions:
        assert rxn.kinetic_law_ast is not None, f"Reaction {rxn.name} has no kinetic law"


def test_events_have_triggers(parsed_model):
    for event in parsed_model.events:
        assert event.trigger_ast is not None
        assert len(event.event_assignments) > 0
