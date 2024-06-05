"""Custom pylint plugin to suppress attribute-defined-outside-init for pytest classes."""

from pylint.checkers import BaseChecker
from pylint.interfaces import HIGH
from astroid import nodes


class PytestAttributeChecker(BaseChecker):
    """PytestAttributeChecker class to suppress attribute-defined-outside-init for pytest classes."""

    __implements__ = (BaseChecker,)

    name = "pytest-attribute-checker"
    priority = HIGH
    msgs = {
        "R9001": (
            "Suppressed attribute-defined-outside-init for pytest classes",
            "pytest-attribute-defined-outside-init",
            "Attributes defined outside __init__ are allowed for pytest classes.",
        ),
    }
    options = ()

    def visit_assignattr(self, node: nodes.AssignAttr) -> None:
        """Visit attribute assignment to check for pytest classes."""
        if self.is_in_pytest_class(node):
            print(f"Suppressing attribute-defined-outside-init for node: {node.attrname}")
            # suppress attribute-defined-outside-init for pytest classes
            self.add_message("R9001", node=node)
            self.linter.disable("attribute-defined-outside-init", scope="package")

    def is_in_pytest_class(self, node: nodes.NodeNG) -> bool:
        """Check if the node is in a class that follows the pytest convention."""
        klass = node.scope()
        while klass and not isinstance(klass, nodes.ClassDef):
            klass = klass.parent
        if not isinstance(klass, nodes.ClassDef):
            return False
        for method in klass.body:
            if isinstance(method, nodes.FunctionDef) and (method.name.startswith("test_") or method.name == "setup_method"):
                return True
        return False


def register(linter):
    """Required method to auto register this checker."""
    print("Registering PytestAttributeChecker")
    linter.register_checker(PytestAttributeChecker(linter))
