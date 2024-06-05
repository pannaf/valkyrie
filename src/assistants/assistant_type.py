"""AssistantType Enum class."""

from enum import Enum


class AssistantType(Enum):
    """
    Enum class for the different types of assistants.
    """

    GANDALF = "gandalf"
    ONBOARDING_WIZARD = "onboarding_wizard"
    GOAL_WIZARD = "goal_wizard"
