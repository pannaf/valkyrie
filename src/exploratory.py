from typing import Annotated, Literal, Optional, Callable, Union
from datetime import datetime, date
import json

import io
from pathlib import Path
from PIL import Image

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_anthropic import ChatAnthropic

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config, RunnableLambda
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from src.sandbox.db_utils import fetch_user, fetch_user_profile, update_user_profile

llm = ChatAnthropic(model="claude-3-haiku-20240307")


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state"""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: dict[str, str]
    dialog_stack: Annotated[list[Literal["assistant", "rapport_wizard", "goal_wizard"]], update_dialog_stack]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        # TODO: do i need this?
        user_id = config.get("user_id", None)
        state = {**state, "user_id": user_id}
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (not result.content or isinstance(result.content, list) and not result.content[0].get("text")):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }


## PROMPTS

rapport_wizard_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for building rapport with the user. "
            "The primary assistant delegates work to you whenever it needs to learn about the user's preferences, "
            "emotions, habits, or other personal information. "
            "When asking about the user's profile information if there's any missing information, only ask for ONE piece of information at a time. "
            "You can also help the user feel more comfortable and engaged with the primary assistant. "
            "If you need more information or the user changes their mind, escalate the task back to the main assistant. "
            "If you have completed your task, mark it as complete. "
            "\n\nCurrent user info:\n<User>{user_info}</User>"
            "\nCurrent time: {time}"
            "\n\nIf the user needs help, and none of your tools are appropriate for it, then"
            ' "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. '
            "Do not make up invalid tools or functions.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

goal_wizard_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for helping the user set and achieve their goals. "
            "The primary assistant delegates work to you whenever it needs to help the user set goals, "
            "track progress, or achieve milestones. "
            "You can also help the user stay motivated and on track. "
            "If you need more information or the user changes their mind, escalate the task back to the main assistant. "
            "If you have completed your task, mark it as complete. "
            "\n\nCurrent user info:\n<User>{user_info}</User>"
            "\nCurrent time: {time}"
            "\n\nIf the user needs help, and none of your tools are appropriate for it, then"
            ' "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. '
            "Do not make up invalid tools or functions.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful personal trainer. "
            "Your primary role is to help the user achieve their fitness goals by planning workouts for them and checking in with them. "
            "If a user requests help with a task that you are not specialized in, "
            "you can delegate the task to a specialized assistant. "
            "First check the user's profile to see if the information is already available. If anything is missing of not specified, "
            "task the Rapport Wizard to ask the user for the missing information. "
            "Only the specialized assistants are given permission to access the user's personal information. "
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            "Provide detailed information to the user, and always double-check the database before concluding that information is unavailable. "
            "\n\nCurrent user information:\n<User>{user_info}</User>"
            "\nCurrent time: {time}",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


## TOOLS


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error")


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print(f"Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


@tool
def fetch_user_info():
    """
    Fetch all known immutable information about the user: id, name, email, height, date of birth

    Returns:
        The user's information, as described above.
    """
    config = ensure_config()
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    user_profile = fetch_user(user_id)
    return user_profile


@tool
def fetch_user_profile_info():
    """
    Fetch all known mutable information about the user: weight, fitness level, activity preferences, workout constraints, goal weight,
    workout frequency, workout location, workout duration.

    Returns:
        The user's profile information, as described above.
    """
    config = ensure_config()
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    user_profile = fetch_user_profile(user_id)
    return user_profile


@tool
def set_user_profile_info(user_profile_field: str, user_profile_value: Union[str, int, float]):
    """
    Given the provided field and value, update the user's profile information.
    If the field is already set to the provided value, don't use this tool.

    The schema for the user_profiles table is as follows:
     weight REAL,
     fitness_level TEXT,
     activity_preferences TEXT,
     workout_constraints TEXT,
     goal_weight REAL,
     workout_frequency INTEGER,
     workout_location TEXT,
     workout_duration INTEGER,
    """

    config = ensure_config()
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    update_user_profile(user_id, user_profile_field, user_profile_value)

    return f"Successfully updated {user_profile_field} to {user_profile_value} for user {user_id}"


## PRIMARY ASSISTANT
class ToRapportWizard(BaseModel):
    """Transfers work to a specialized assistant to handle tasks associated with setting user profile info, including weight,
    fitness level, activity preferences, workout constraints, goal weight, workout frequency, workout location, and workout duration.
    """

    request: str = Field(description="Any necessary follup questions the rapport wizard should clarify before proceeding.")


class ToGoalWizard(BaseModel):
    """Transfers work to a specialized assistant to handle goal-setting tasks, except the rapport wizard handles marking the user's goal weight."""

    request: str = Field(description="Any necessary follup questions the goal wizard should clarify before proceeding.")


## RUNNABLES

# TODO: provide tools

rapport_wizard_safe_tools = [fetch_user_profile_info]
rapport_wizard_sensitive_tools = [set_user_profile_info]
rapport_wizard_tools = rapport_wizard_safe_tools + rapport_wizard_sensitive_tools
rapport_wizard_runnable = rapport_wizard_prompt | llm.bind_tools(rapport_wizard_tools + [CompleteOrEscalate])

goal_wizard_safe_tools = []
goal_wizard_sensitive_tools = []
goal_wizard_tools = goal_wizard_safe_tools + goal_wizard_sensitive_tools
goal_wizard_runnable = goal_wizard_prompt | llm.bind_tools(goal_wizard_tools + [CompleteOrEscalate])

primary_assistant_tools = [fetch_user_profile_info]
assistant_runnable = primary_assistant_prompt | llm.bind_tools(primary_assistant_tools + [ToRapportWizard, ToGoalWizard])


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name}."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node


builder = StateGraph(State)


def serialize_realdictrow(row):
    def convert_value(value):
        if isinstance(value, date):
            return value.isoformat()  # Convert date to ISO format string
        return value

    return {key: convert_value(value) for key, value in row.items()}


def user_info(state: State):
    _user_info = serialize_realdictrow(fetch_user_info.invoke({}))
    _user_info = json.dumps(_user_info, indent=4)

    return {"user_info": _user_info}


builder.add_node("fetch_user_info", user_info)
builder.set_entry_point("fetch_user_info")

# rapport wizard assistant
builder.add_node(
    "enter_rapport_wizard",
    create_entry_node("Rapport Building Wizard", "rapport_wizard"),
)
builder.add_node("rapport_wizard", Assistant(rapport_wizard_runnable))
builder.add_edge("enter_rapport_wizard", "rapport_wizard")
builder.add_node(
    "rapport_wizard_sensitive_tools",
    create_tool_node_with_fallback(rapport_wizard_sensitive_tools),
)
builder.add_node(
    "rapport_wizard_safe_tools",
    create_tool_node_with_fallback(rapport_wizard_safe_tools),
)


def route_rapport_wizard(
    state: State,
) -> Literal[
    "rapport_wizard_sensitive_tools",
    "rapport_wizard_safe_tools",
    "leave_skill",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_toolnames = [t.name for t in rapport_wizard_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "rapport_wizard_safe_tools"
    return "rapport_wizard_sensitive_tools"


builder.add_edge("rapport_wizard_sensitive_tools", "rapport_wizard")
builder.add_edge("rapport_wizard_safe_tools", "rapport_wizard")
builder.add_conditional_edges("rapport_wizard", route_rapport_wizard)

# goal wizard assistant
builder.add_node(
    "enter_goal_wizard",
    create_entry_node("Goal Setting Wizard", "goal_wizard"),
)
builder.add_node("goal_wizard", Assistant(goal_wizard_runnable))
builder.add_edge("enter_goal_wizard", "goal_wizard")
builder.add_node(
    "goal_wizard_safe_tools",
    create_tool_node_with_fallback(goal_wizard_safe_tools),
)
builder.add_node(
    "goal_wizard_sensitive_tools",
    create_tool_node_with_fallback(goal_wizard_sensitive_tools),
)


def route_goal_wizard(
    state: State,
) -> Literal[
    "goal_wizard_safe_tools",
    "goal_wizard_sensitive_tools",
    "leave_skill",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_toolnames = [t.name for t in goal_wizard_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "goal_wizard_safe_tools"
    return "goal_wizard_sensitive_tools"


builder.add_edge("goal_wizard_sensitive_tools", "goal_wizard")
builder.add_edge("goal_wizard_safe_tools", "goal_wizard")
builder.add_conditional_edges("goal_wizard", route_goal_wizard)


# This node will be shared for exiting all specialized assistants
def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "primary_assistant")

builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.add_node("primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools))


def route_primary_assistant(
    state: State,
) -> Literal[
    "primary_assistant_tools",
    "enter_rapport_wizard",
    "enter_goal_wizard",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToRapportWizard.__name__:
            return "enter_rapport_wizard"
        if tool_calls[0]["name"] == ToGoalWizard.__name__:
            return "enter_goal_wizard"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")


# The assistant can route to one of the delegated assistants,
# directly use a tool, or directly respond to the user
builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    {
        "enter_rapport_wizard": "enter_rapport_wizard",
        "enter_goal_wizard": "enter_goal_wizard",
        "primary_assistant_tools": "primary_assistant_tools",
        END: END,
    },
)
builder.add_edge("primary_assistant_tools", "primary_assistant")


# Each delegated workflow can directly respond to the user
# When the user responds, we want to return to the currently active workflow
def route_to_workflow(
    state: State,
) -> Literal[
    "primary_assistant",
    "rapport_wizard",
    "goal_wizard",
]:
    """If we are in a delegated state, route directly to the appropriate assistant."""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]


builder.add_conditional_edges("fetch_user_info", route_to_workflow)

# Compile graph
memory = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(
    checkpointer=memory,
    # Let the user approve or deny the use of sensitive tools
    interrupt_before=[
        "rapport_wizard_sensitive_tools",
        "goal_wizard_sensitive_tools",
    ],
)

visualize_graph = False

if visualize_graph:
    graph_path = Path("graph.png")
    image_data = io.BytesIO(graph.get_graph().draw_mermaid_png())
    image = Image.open(image_data)
    image.save(graph_path)

thread_id = "1"
config = {"configurable": {"user_id": "bf9d8cd5-3c89-40ef-965b-ad2ff148e52a", "thread_id": thread_id}}

_printed = set()

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    events = graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values")
    for event in events:
        _print_event(event, _printed)
    snapshot = graph.get_state(config)
    while snapshot.next:
        # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
        # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
        # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
        user_input = input(
            "Do you approve of the above actions? Type 'y' to continue;" " otherwise, explain your requested changed.\n\n"
        )
        if user_input.strip() == "y":
            # Just continue
            result = graph.invoke(
                None,
                config,
            )
        else:
            # Satisfy the tool invocation by
            # providing instructions on the requested changes / change of mind
            result = graph.invoke(
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                            content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                        )
                    ]
                },
                config,
            )
        snapshot = graph.get_state(config)
