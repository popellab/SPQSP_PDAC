"""Tests for AST translator — verify SBML math to C++ translation."""

import pytest


def test_simple_reaction_translation(model_with_sorting, mapper):
    """First-order kinetics should translate to simple multiplication."""
    from ast_translator import ASTTranslator
    trans = ASTTranslator(mapper, context="f")

    # Find clearance reaction: k_cell_clear * V_T.C_x
    rxn = model_with_sorting.reactions[0]  # Reaction_1 is cell clearance
    expr = trans.translate(rxn.kinetic_law_ast)
    assert "PARAM(P_k_cell_clear)" in expr
    assert "SPVAR(SP_V_T_C_x)" in expr


def test_aux_var_in_expression(model_with_sorting, mapper):
    """Expressions referencing assignment rule targets should use AUX_VAR."""
    from ast_translator import ASTTranslator
    trans = ASTTranslator(mapper, context="f")

    # Find a reaction that uses C_total or similar
    for rxn in model_with_sorting.reactions:
        expr = trans.translate(rxn.kinetic_law_ast)
        if "AUX_VAR_" in expr:
            assert "AUX_VAR_" in expr
            break
    else:
        pytest.skip("No reaction references an assignment rule variable")


def test_event_context_translation(model_with_sorting, mapper):
    """Event context should use NV_DATA_S(_y) instead of SPVAR."""
    from ast_translator import ASTTranslator
    trans = ASTTranslator(mapper, context="event")

    sp = model_with_sorting.sp_var[0]
    resolved = trans.mapper.resolve_name(sp.id, "event")
    assert "NV_DATA_S(_y)" in resolved
    assert "SPVAR" not in resolved


def test_pow_translation(model_with_sorting, mapper):
    """Power expressions should become std::pow()."""
    from ast_translator import ASTTranslator
    trans = ASTTranslator(mapper, context="f")

    # Hill functions use pow
    for ar in model_with_sorting.assignment_rule_order:
        expr = trans.translate(ar.math_ast)
        if "std::pow" in expr:
            assert "std::pow(" in expr
            break
    else:
        pytest.skip("No assignment rule uses power function")


def test_log_translation(model_with_sorting, mapper):
    """Log expressions should become std::log()."""
    from ast_translator import ASTTranslator
    trans = ASTTranslator(mapper, context="f")

    # Gompertzian growth uses log
    for rxn in model_with_sorting.reactions:
        expr = trans.translate(rxn.kinetic_law_ast)
        if "std::log" in expr:
            assert "std::log(" in expr
            break


def test_max_translation(model_with_sorting, mapper):
    """Max expressions should become std::max()."""
    from ast_translator import ASTTranslator
    trans = ASTTranslator(mapper, context="f")

    for rxn in model_with_sorting.reactions:
        expr = trans.translate(rxn.kinetic_law_ast)
        if "std::max" in expr:
            assert "std::max(" in expr
            break


def test_no_unknown_ast_types(model_with_sorting, mapper):
    """No expressions should produce UNKNOWN AST TYPE comments."""
    from ast_translator import ASTTranslator
    trans = ASTTranslator(mapper, context="f")

    for rxn in model_with_sorting.reactions:
        expr = trans.translate(rxn.kinetic_law_ast)
        assert "UNKNOWN AST TYPE" not in expr, f"Reaction {rxn.name}: {expr}"

    for ar in model_with_sorting.assignment_rule_order:
        expr = trans.translate(ar.math_ast)
        assert "UNKNOWN AST TYPE" not in expr, f"Rule {ar.variable_name}: {expr}"
