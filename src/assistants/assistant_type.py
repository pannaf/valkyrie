"""AssistantType Enum class."""

from enum import Enum


class AssistantType(Enum):
    """
    Enum class for the different types of assistants.
    """

    GANDALF = "gandalf"
    ONBOARDING_WIZARD = "onboarding_wizard"
    GOAL_WIZARD = "goal_wizard"
    PROGRAMMING_WIZARD = "programming_wizard"
    V_WIZARD = "v_wizard"

    @classmethod
    def all_values(cls):
        return [c.value for c in cls]


def main():
    """Main entry point to demonstrate the all_values method."""
    print("All AssistantType values:", AssistantType.all_values())


if __name__ == "__main__":
    main()
