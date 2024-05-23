from pylint.checkers import BaseChecker
from pylint.interfaces import HIGH
from astroid import nodes


class HydraMainChecker(BaseChecker):
    __implements__ = (BaseChecker,)

    name = "hydra-main-checker"
    priority = HIGH
    msgs = {
        "W0001": (
            "Suppressed no-value-for-parameter for Hydra main",
            "hydra-main-no-value-for-parameter",
            "Hydra main decorated functions do not need explicit cfg parameter in function signature",
        ),
    }
    options = ()

    def visit_functiondef(self, node: nodes.FunctionDef) -> None:
        if node.decorators:
            for decorator in node.decorators.nodes:
                if (
                    isinstance(decorator, nodes.Call)
                    and isinstance(decorator.func, nodes.Attribute)
                    and decorator.func.attrname == "main"
                    and isinstance(decorator.func.expr, nodes.Name)
                    and decorator.func.expr.name == "hydra"
                ):
                    # Suppress the warning E1120
                    self.add_message("W0001", node=node)
                    # Suppress the specific Pylint warning
                    self.linter.disable("no-value-for-parameter", node)


def register(linter):
    """Required method to auto register this checker."""
    linter.register_checker(HydraMainChecker(linter))
