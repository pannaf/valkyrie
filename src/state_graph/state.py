"""State definition for the state graph"""

from typing import Annotated, Literal, Optional, TypedDict
from langgraph.graph.message import AnyMessage, add_messages

from src.assistants.assistant_type import AssistantType


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the dialog state."""
    if right is None:
        return left

    if right == "pop":
        return left[:-1]

    if right in AssistantType:  # only append valid AssistantType values
        return left + [right]

    return left  # otherwise ignore unexpected values


# valid dialog states stored in AssistantType Enum
AssistantTypeLiteral = Literal[tuple(AssistantType.all_values())]  # type: ignore for Pylance


class State(TypedDict):
    """State definition for the state graph"""

    messages: Annotated[list[AnyMessage], add_messages]
    user_info: dict[str, str]
    dialog_state: Annotated[list[AssistantTypeLiteral], update_dialog_stack]
