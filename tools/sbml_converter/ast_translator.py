"""Translate SBML MathML AST nodes to C++ expression strings."""

import libsbml
from name_mapper import NameMapper


class ASTTranslator:
    """Translates libsbml ASTNode trees to C++ code strings."""

    def __init__(self, mapper: NameMapper, context: str = "f"):
        """
        Args:
            mapper: NameMapper for resolving SBML IDs to C++ names
            context: 'f' (inside f()), 'event' (event functions), 'other' (update_y_other)
        """
        self.mapper = mapper
        self.context = context

    def translate(self, node: libsbml.ASTNode) -> str:
        """Translate an AST node to a C++ expression string."""
        if node is None:
            return "0"
        return self._visit(node)

    def _visit(self, node: libsbml.ASTNode) -> str:
        ntype = node.getType()

        # ---- Literals ----
        if ntype == libsbml.AST_INTEGER:
            v = node.getInteger()
            return f"{v}.0"

        if ntype == libsbml.AST_REAL:
            return self._format_real(node.getReal())

        if ntype == libsbml.AST_REAL_E:
            return self._format_real(node.getReal())

        if ntype == libsbml.AST_RATIONAL:
            return self._format_real(node.getReal())

        # ---- Names ----
        if ntype == libsbml.AST_NAME:
            name = node.getName()
            return self.mapper.resolve_name(name, self.context)

        if ntype == libsbml.AST_NAME_TIME:
            return "t"

        # ---- Constants ----
        if ntype == libsbml.AST_CONSTANT_PI:
            return "M_PI"
        if ntype == libsbml.AST_CONSTANT_E:
            return "M_E"
        if ntype == libsbml.AST_CONSTANT_TRUE:
            return "true"
        if ntype == libsbml.AST_CONSTANT_FALSE:
            return "false"

        # ---- Arithmetic operators ----
        if ntype == libsbml.AST_PLUS:
            return self._binary_op(node, " + ")

        if ntype == libsbml.AST_MINUS:
            if node.getNumChildren() == 1:
                return f"-({self._visit(node.getChild(0))})"
            return self._binary_op(node, " - ")

        if ntype == libsbml.AST_TIMES:
            return self._nary_op(node, " * ")

        if ntype == libsbml.AST_DIVIDE:
            return self._binary_op(node, " / ")

        if ntype == libsbml.AST_POWER or ntype == libsbml.AST_FUNCTION_POWER:
            base = self._visit(node.getChild(0))
            exp = self._visit(node.getChild(1))
            return f"std::pow({base}, {exp})"

        # ---- Relational operators ----
        if ntype == libsbml.AST_RELATIONAL_EQ:
            return self._binary_op(node, " == ")
        if ntype == libsbml.AST_RELATIONAL_NEQ:
            return self._binary_op(node, " != ")
        if ntype == libsbml.AST_RELATIONAL_LT:
            return self._binary_op(node, " < ")
        if ntype == libsbml.AST_RELATIONAL_GT:
            return self._binary_op(node, " > ")
        if ntype == libsbml.AST_RELATIONAL_LEQ:
            return self._binary_op(node, " <= ")
        if ntype == libsbml.AST_RELATIONAL_GEQ:
            return self._binary_op(node, " >= ")

        # ---- Logical operators ----
        if ntype == libsbml.AST_LOGICAL_AND:
            return self._nary_op(node, " && ")
        if ntype == libsbml.AST_LOGICAL_OR:
            return self._nary_op(node, " || ")
        if ntype == libsbml.AST_LOGICAL_NOT:
            return f"!({self._visit(node.getChild(0))})"
        if ntype == libsbml.AST_LOGICAL_XOR:
            # XOR as (a && !b) || (!a && b)
            a = self._visit(node.getChild(0))
            b = self._visit(node.getChild(1))
            return f"(({a} && !({b})) || (!({a}) && {b}))"

        # ---- Functions ----
        if ntype == libsbml.AST_FUNCTION_ROOT:
            if node.getNumChildren() == 2:
                degree = self._visit(node.getChild(0))
                arg = self._visit(node.getChild(1))
                if degree == "2":
                    return f"std::sqrt({arg})"
                return f"std::pow({arg}, 1.0 / {degree})"
            else:
                arg = self._visit(node.getChild(0))
                return f"std::sqrt({arg})"

        if ntype == libsbml.AST_FUNCTION_LN:
            return f"std::log({self._visit(node.getChild(0))})"

        if ntype == libsbml.AST_FUNCTION_LOG:
            if node.getNumChildren() == 2:
                base = self._visit(node.getChild(0))
                arg = self._visit(node.getChild(1))
                return f"(std::log({arg}) / std::log({base}))"
            return f"std::log10({self._visit(node.getChild(0))})"

        if ntype == libsbml.AST_FUNCTION_EXP:
            return f"std::exp({self._visit(node.getChild(0))})"

        if ntype == libsbml.AST_FUNCTION_ABS:
            return f"std::abs({self._visit(node.getChild(0))})"

        if ntype == libsbml.AST_FUNCTION_FLOOR:
            return f"std::floor({self._visit(node.getChild(0))})"

        if ntype == libsbml.AST_FUNCTION_CEILING:
            return f"std::ceil({self._visit(node.getChild(0))})"

        if ntype == libsbml.AST_FUNCTION_SIN:
            return f"std::sin({self._visit(node.getChild(0))})"

        if ntype == libsbml.AST_FUNCTION_COS:
            return f"std::cos({self._visit(node.getChild(0))})"

        if ntype == libsbml.AST_FUNCTION_TAN:
            return f"std::tan({self._visit(node.getChild(0))})"

        # ---- Piecewise (if-then-else) ----
        if ntype == libsbml.AST_FUNCTION_PIECEWISE:
            return self._piecewise(node)

        # ---- Min/Max ----
        if ntype == libsbml.AST_FUNCTION_MIN:
            args = [self._visit(node.getChild(i)) for i in range(node.getNumChildren())]
            result = args[0]
            for a in args[1:]:
                result = f"std::min({result}, {a})"
            return result

        if ntype == libsbml.AST_FUNCTION_MAX:
            args = [self._visit(node.getChild(i)) for i in range(node.getNumChildren())]
            result = args[0]
            for a in args[1:]:
                result = f"std::max({result}, {a})"
            return result

        # ---- Generic function call ----
        if ntype == libsbml.AST_FUNCTION:
            fname = node.getName()
            args = ", ".join(self._visit(node.getChild(i))
                            for i in range(node.getNumChildren()))
            # Map known MATLAB/SimBiology functions
            func_map = {
                "pow": "std::pow",
                "sqrt": "std::sqrt",
                "log": "std::log",
                "log10": "std::log10",
                "exp": "std::exp",
                "abs": "std::abs",
                "min": "std::min",
                "max": "std::max",
                "nthroot": None,  # special handling
            }
            if fname == "nthroot":
                # nthroot(x, n) -> pow(x, 1.0/n)
                x = self._visit(node.getChild(0))
                n = self._visit(node.getChild(1))
                return f"std::pow({x}, 1.0 / {n})"
            cpp_func = func_map.get(fname, fname)
            return f"{cpp_func}({args})"

        # ---- Lambda (shouldn't appear in rate laws, but handle gracefully) ----
        if ntype == libsbml.AST_LAMBDA:
            # Just translate the body (last child)
            return self._visit(node.getChild(node.getNumChildren() - 1))

        # Fallback
        return f"/* UNKNOWN AST TYPE {ntype} */"

    def _binary_op(self, node: libsbml.ASTNode, op: str) -> str:
        left = self._visit(node.getChild(0))
        right = self._visit(node.getChild(1))
        return f"({left}{op}{right})"

    def _nary_op(self, node: libsbml.ASTNode, op: str) -> str:
        parts = [self._visit(node.getChild(i)) for i in range(node.getNumChildren())]
        return "(" + op.join(parts) + ")"

    def _piecewise(self, node: libsbml.ASTNode) -> str:
        """Handle SBML piecewise (if/else) expressions.

        Structure: piece(value, condition), piece(value, condition), ..., otherwise(value)
        Children: [value1, cond1, value2, cond2, ..., otherwise_value]
        """
        n = node.getNumChildren()
        if n == 0:
            return "0"

        # Build nested ternary: cond1 ? val1 : (cond2 ? val2 : otherwise)
        parts = []
        for i in range(0, n - 1, 2):
            val = self._visit(node.getChild(i))
            cond = self._visit(node.getChild(i + 1))
            parts.append((cond, val))

        # Last child is the otherwise value (if odd number of children)
        if n % 2 == 1:
            otherwise = self._visit(node.getChild(n - 1))
        else:
            otherwise = "0"

        result = otherwise
        for cond, val in reversed(parts):
            result = f"({cond} ? {val} : {result})"

        return result

    def _format_real(self, value: float) -> str:
        """Format a float with sufficient precision for C++."""
        if value == 0.0:
            return "0.0"
        if value == int(value) and abs(value) < 1e15:
            return f"{value:.1f}"
        return f"{value:.17g}"
