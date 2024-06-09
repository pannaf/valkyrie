from typing import Literal, Callable
from datetime import date
import json

import io
from pathlib import Path
from PIL import Image

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition

from langchain_anthropic import ChatAnthropic

from langchain_core.messages import ToolMessage

from dotenv import load_dotenv

from src.prompts.yaml_prompt_loader import YamlPromptLoader
from src.state_graph.state import State
from src.tools import (
    fetch_user_info,
    fetch_user_profile_info,
    set_user_profile_info,
    fetch_goals,
    handle_create_goal,
    update_goal,
    create_tool_node_with_fallback,
    ToOnboardingWizard,
    ToGoalWizard,
    CompleteOrEscalate,
)
from src.assistants.assistant import Assistant
from src.assistants.onboarding_wizard import OnboardingWizard
from src.assistants.goal_wizard import GoalWizard

load_dotenv()


# llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)


## PROMPT LOADER

prompt_loader = YamlPromptLoader("src/prompts/prompts.yaml")

## RUNNABLES

onboarding_wizard = OnboardingWizard(llm, prompt_loader, "onboarding_wizard")
onboarding_wizard_runnable = onboarding_wizard.get_runnable()

goal_wizard = GoalWizard(llm, prompt_loader, "goal_wizard")
goal_wizard_runnable = goal_wizard.get_runnable()

primary_assistant_tools = []
primary_assistant_prompt = prompt_loader.get_prompt("gandalf")
assistant_runnable = primary_assistant_prompt | llm.bind_tools(primary_assistant_tools + [ToOnboardingWizard, ToGoalWizard])


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


def user_info(state: State):
    def serialize_realdictrow(row):
        def convert_value(value):
            if isinstance(value, date):
                return value.isoformat()  # Convert date to ISO format string
            return value

        return {key: convert_value(value) for key, value in row.items()}

    _user_info = serialize_realdictrow(fetch_user_info.invoke({}))
    _user_info = json.dumps(_user_info, indent=4)

    return {"user_info": _user_info}


builder.add_node("fetch_user_info", user_info)
builder.set_entry_point("fetch_user_info")

# onboarding wizard assistant
builder.add_node(
    "enter_onboarding_wizard",
    create_entry_node("Onboarding Wizard", "onboarding_wizard"),
)
builder.add_node("onboarding_wizard", Assistant(onboarding_wizard_runnable))
builder.add_edge("enter_onboarding_wizard", "onboarding_wizard")
builder.add_node(
    "onboarding_wizard_sensitive_tools",
    create_tool_node_with_fallback(onboarding_wizard.sensitive_tools),
)
builder.add_node(
    "onboarding_wizard_safe_tools",
    create_tool_node_with_fallback(onboarding_wizard.safe_tools),
)


def route_onboarding_wizard(
    state: State,
) -> Literal[
    "onboarding_wizard_sensitive_tools",
    "onboarding_wizard_safe_tools",
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
    safe_toolnames = [t.name for t in onboarding_wizard.safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "onboarding_wizard_safe_tools"
    return "onboarding_wizard_sensitive_tools"


builder.add_edge("onboarding_wizard_sensitive_tools", "onboarding_wizard")
builder.add_edge("onboarding_wizard_safe_tools", "onboarding_wizard")
builder.add_conditional_edges("onboarding_wizard", route_onboarding_wizard)

# goal wizard assistant
builder.add_node(
    "enter_goal_wizard",
    create_entry_node("Goal Setting Wizard", "goal_wizard"),
)
builder.add_node("goal_wizard", Assistant(goal_wizard_runnable))
builder.add_edge("enter_goal_wizard", "goal_wizard")
builder.add_node(
    "goal_wizard_safe_tools",
    create_tool_node_with_fallback(goal_wizard.safe_tools),
)
builder.add_node(
    "goal_wizard_sensitive_tools",
    create_tool_node_with_fallback(goal_wizard.sensitive_tools),
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
    safe_toolnames = [t.name for t in goal_wizard.safe_tools]
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
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and continue the conversation.",
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
    "enter_onboarding_wizard",
    "enter_goal_wizard",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToOnboardingWizard.__name__:
            return "enter_onboarding_wizard"
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
        "enter_onboarding_wizard": "enter_onboarding_wizard",
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
    "onboarding_wizard",
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
)


VISUALIZE_GRAPH = False

if VISUALIZE_GRAPH:
    graph_path = Path("graph.png")
    image_data = io.BytesIO(graph.get_graph().draw_mermaid_png())
    image = Image.open(image_data)
    image.save(graph_path)


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


THREAD_ID = "1"
config_c = {"configurable": {"user_id": "bf9d8cd5-3c89-40ef-965b-ad2ff148e52a", "thread_id": THREAD_ID}}

_printed = set()

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    events = graph.stream({"messages": [("user", user_input)]}, config_c, stream_mode="values")
    for event in events:
        _print_event(event, _printed)
