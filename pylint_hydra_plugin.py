"""Custom pylint plugin to suppress no-value-for-parameter for Hydra main decorated functions."""

from pylint.checkers import BaseChecker
from pylint.interfaces import HIGH
from astroid import nodes


class HydraMainChecker(BaseChecker):
    """HydraMainChecker class to suppress no-value-for-parameter for Hydra main decorated functions."""

    __implements__ = (BaseChecker,)

    name = "hydra-main-checker"
    priority = HIGH
    msgs = {
        "R1001": (
            "Suppressed no-value-for-parameter for Hydra main",
            "hydra-main-no-value-for-parameter",
            "Hydra main decorated functions do not need explicit cfg parameter in function signature",
        ),
    }
    options = ()

    def visit_functiondef(self, node: nodes.FunctionDef) -> None:
        """visit_functiondef method to check for Hydra main decorated functions."""
        if node.decorators:
            for decorator in node.decorators.nodes:
                if (
                    isinstance(decorator, nodes.Call)
                    and isinstance(decorator.func, nodes.Attribute)
                    and decorator.func.attrname == "main"
                    and isinstance(decorator.func.expr, nodes.Name)
                    and decorator.func.expr.name == "hydra"
                ):
                    # suppress no-value-for-parameter for hydra.main decorated functions
                    # the add_message will still show a message tho
                    self.add_message("R1001", node=node)
                    self.linter.disable("no-value-for-parameter", scope="package")


def register(linter):
    """Required method to auto register this checker."""
    linter.register_checker(HydraMainChecker(linter))
