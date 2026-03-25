"""Topological sort for assignment rules and initial assignments."""

from collections import defaultdict, deque

import libsbml
from sbml_parser import SBMLModel


def _get_referenced_names(ast_node: libsbml.ASTNode) -> set:
    """Recursively collect all named references in an AST."""
    names = set()
    if ast_node is None:
        return names

    if ast_node.isName():
        name = ast_node.getName()
        if name:
            names.add(name)

    for i in range(ast_node.getNumChildren()):
        names |= _get_referenced_names(ast_node.getChild(i))

    return names


def _topological_sort(nodes: list, node_id_func, deps_func) -> list:
    """Generic topological sort.

    Args:
        nodes: list of items to sort
        node_id_func: function(item) -> unique ID string
        deps_func: function(item) -> set of dependency IDs

    Returns:
        Sorted list of items (dependencies first).
    """
    id_to_node = {node_id_func(n): n for n in nodes}
    node_ids = set(id_to_node.keys())

    # Build adjacency: if A depends on B, edge B -> A
    in_degree = defaultdict(int)
    adjacency = defaultdict(list)
    for n in nodes:
        nid = node_id_func(n)
        deps = deps_func(n) & node_ids  # only care about internal deps
        in_degree[nid] += 0  # ensure entry exists
        for dep in deps:
            adjacency[dep].append(nid)
            in_degree[nid] += 1

    # Kahn's algorithm
    queue = deque([nid for nid in node_ids if in_degree[nid] == 0])
    result = []

    while queue:
        nid = queue.popleft()
        result.append(id_to_node[nid])
        for neighbor in adjacency[nid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(nodes):
        sorted_ids = {node_id_func(n) for n in result}
        remaining = [node_id_func(n) for n in nodes if node_id_func(n) not in sorted_ids]
        raise ValueError(f"Cycle detected in dependency graph. Remaining: {remaining}")

    return result


def sort_assignment_rules(model: SBMLModel) -> list:
    """Sort assignment rules by dependency order.

    Returns list of AssignmentRule in evaluation order (dependencies first).
    """
    rule_target_ids = {ar.variable_id for ar in model.assignment_rules}

    def deps(ar):
        refs = _get_referenced_names(ar.math_ast)
        # Only keep refs that are other assignment rule targets
        return refs & rule_target_ids - {ar.variable_id}

    return _topological_sort(
        model.assignment_rules,
        node_id_func=lambda ar: ar.variable_id,
        deps_func=deps,
    )


def sort_initial_assignments(model: SBMLModel) -> list:
    """Sort initial assignments by dependency order."""
    ia_target_ids = {ia.symbol_id for ia in model.initial_assignments}

    def deps(ia):
        refs = _get_referenced_names(ia.math_ast)
        return refs & ia_target_ids - {ia.symbol_id}

    return _topological_sort(
        model.initial_assignments,
        node_id_func=lambda ia: ia.symbol_id,
        deps_func=deps,
    )
