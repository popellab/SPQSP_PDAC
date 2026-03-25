"""Tests for dependency sorter."""


def test_assignment_rules_sorted(model_with_sorting):
    """Assignment rules should be sorted without errors."""
    assert len(model_with_sorting.assignment_rule_order) == len(model_with_sorting.assignment_rules)


def test_dependencies_come_first(model_with_sorting, mapper):
    """C_total should come before V_T (V_T depends on C_total)."""
    order = [ar.variable_name for ar in model_with_sorting.assignment_rule_order]
    if "C_total" in order and "V_T" in order:
        assert order.index("C_total") < order.index("V_T"), \
            "C_total must be computed before V_T"


def test_t_total_before_v_t(model_with_sorting):
    """T_total should come before V_T."""
    order = [ar.variable_name for ar in model_with_sorting.assignment_rule_order]
    if "T_total" in order and "V_T" in order:
        assert order.index("T_total") < order.index("V_T")


def test_initial_assignments_sorted(model_with_sorting):
    """Initial assignments should be sorted."""
    assert len(model_with_sorting.initial_assignment_order) == len(model_with_sorting.initial_assignments)


def test_no_duplicate_rules(model_with_sorting):
    """Each rule should appear exactly once."""
    ids = [ar.variable_id for ar in model_with_sorting.assignment_rule_order]
    assert len(ids) == len(set(ids))
