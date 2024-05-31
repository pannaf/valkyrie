from typing import Annotated, Literal, Optional, Callable, Union
from datetime import datetime, date
import json
import textwrap

import io
import uuid
from pathlib import Path
from PIL import Image

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config, RunnableLambda
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from dotenv import load_dotenv

from src.sandbox.db_utils import (
    fetch_user,
    fetch_user_profile,
    update_user_profile,
    fetch_goals_db,
    update_goal_db,
    create_empty_goal_db,
)

load_dotenv()


if True:
    # llm = ChatAnthropic(model="claude-3-haiku-20240307")
    llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
else:
    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state"""
    print(f"update_dialog_stack: {left=} {right=}")
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: dict[str, str]
    dialog_state: Annotated[list[Literal["assistant", "onboarding_wizard", "goal_wizard"]], update_dialog_stack]
    current_goal_id: Optional[str]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
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

onboarding_wizard_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            textwrap.dedent(
                """As their personal trainer named V, you are getting to know a new client. 
            Initially, you should ask them some basic getting to know you questions, such as "how are you?" "how's your day going?" or other such questions.
            <response guidelines>
            - Sound friendly and approachable, as if you were texting with a friend. For example, phrases like "Hey there!" or "Cool beans!" or "Gotcha!" or "Roger that!" may be appropriate.
            - Always reply to the user.
            - It should feel like a conversation and sound natural.
            - Use the user's name when addressing them.
            - Use emojis where appropriate.
            - Send messages that are 1-3 sentences long.
            - When asking a question, ask one question at a time.
            - When replying to the user, it may sometimes make sense to draw from the text in the ai message when calling the previous tool. Otherwise the user doesn't see that text. 
            </response guidelines>
            <task instructions>
            You have two objectives: 1. get to know the user, and 2. update their profile with any relevant information you learn about them.
            You are responsible for filling out the fields in the user's profile.
            You can only update one field at a time in the user's profile.
            Only the user knows their personal information, so you should ask them for it.
            </task instructions>
            <tools available>
            The user doesn't know you have tools available. It would be confusing to mention them.
            Even though you have these tools available, you can also choose to chat with the user without using them. For example, when
            introducing yourself, you don't need to use a tool. 
            - fetch_user_profile_info : Use this tool to fetch the user's profile information. Only use AFTER you've introduced yourself and had a brief conversation with the user that included 1-2 icebreaker questions AND asked if you can ask them some questions.  
            - set_user_profile_info : Use this tool to update the user's profile with the information you learn about them. ONLY use this tool if you've learned something about their activity preferences, workout location, workout frequency, workout duration, workout constraints, fitness level, weight, or goal weight.
            </tools available>
            <conversation structure>
            Follow these steps:
            Step 0 - If it's your first time meeting the user, introduce yourself (you are V, their new virtual personal trainer).
            Step 1 - Ask 1-2 basic getting-to-know-you icebreaker questions, one at a time.
            Step 2 - engage with the user in a brief conversation, double-clicking with them on their responses to the icebreaker questions.
            Step 3 - ask the user if it's ok to ask them some personal questions.
            Step 4 - if they respond positively, fetch the user's profile information and ask them a question. if they respond negatively, escalate to the host assistant.
            Step 5 - update the user's profile with the information you learn, if it's relevant to their profile data table entry
            </conversation structure>
            <when to return to the host assistant>
            If the user doesn't want to answer your questions, escalate to the host assistant.
            Check if the user's profile is completely filled out without any missing fields. If it is, return to the host assistant. If it isn't, continue asking questions as the Onboarding Wizard.
            </when to return to the host assistant>"""
            ),
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

goal_wizard_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            textwrap.dedent(
                """As their personal trainer named V, you are helping the user set and achieve their fitness goals.
            <response guidelines>
            - Sound friendly and approachable, as if you were texting with a friend. For example, phrases like "Hey there!" or "Cool beans!" or "Gotcha!" or "Roger that!" may be appropriate.
            - Always reply to the user.
            - It should feel like a conversation and sound natural.
            - Use the user's name when addressing them.
            - Use emojis where appropriate.
            - Send messages that are 1-3 sentences long.
            - When asking a question, ask one question at a time.
            - When replying to the user, it may sometimes make sense to draw from the text in the ai message when calling the previous tool. Otherwise the user doesn't see that text. 
            </response guidelines>
            <task instructions>
            You have one objective: help the user set their fitness goals. Each goal needs to go into the goals table in the database.
            You should guide the user to set 1-3 specific, measurable, achievable, relevant, and time-bound goals.
            There are two main types of goals: outcome goals and process goals. Outcome goals are the end result, like losing 10 pounds. Process goals are the steps you take to achieve the outcome goal, like exercising 3 times a week.
            The user might not know the difference between outcome and process goals, so you should explain it to them if necessary. 
            For every outcome goal, there should be at least one process goal supporting it.
            Every field in the goals table should be filled out for each goal. Ask the user for the information you need to fill out the fields.
            </task instructions>
            <tools available>
            The user doesn't know you have tools available. It would be confusing to mention them.
            Even though you have these tools available, you can also choose to chat with the user without using them.
            - fetch_goals : Use this tool to fetch the user's current goals.
            - handle_create_goal : Use this tool to create a new, empty goal for the user. You need to call this tool before updating the goal.
            - update_goal_state : Use this tool to update the current goal id in the state, based on the goal being discussed.
            - clear_goal_state : Use this tool to clear the current goal id in the state. Before returning to the host assistant, you should clear the current goal id.
            - update_goal : Use this tool to update the user's goal with the information you learn about them.
            </tools available>
            <conversation structure>
            Follow these steps:
            Step 0 - In 2 sentences or less, introduce that you're now going to help the user set their fitness goals. Tell them you'll be asking them some questions to help them set their goals. Tell them we'll be setting 1-3 goals.
            Step 1 - Ask the user if they already has any fitness goals. If they do, ask them to share them with you. If they don't, help them discover some goals by asking them about their fitness journey. 
            Step 2 - Update the database with as much information as you can about the user's goals. Before adding a new goal, make sure to call the tool handle_create_goal to create a new goal entry for it.
            Step 3 - Ask the user for the remaining information you need to fill out the fields in the goals table, this may be things like the start_date, end_date, current_value, etc. Only ask one question at a time when clarifying.
            </conversation structure>
            <when to return to the host assistant>
            If the user doesn't want to answer your questions, escalate to the host assistant.
            When the user has set 1-3 goals and seems done with goal setting, return to the host assistant.
            Before returning to the host assistant, you must clear the current goal id using the tool clear_goal_state.
            </when to return to the host assistant>
            <current time>{time}</current time>
            """
            ),
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# "Transfer to the Onboarding Wizard first and immediately! "
# "The Onboarding Wizard can look up for itself the user's profile and update it. "
# "Only the specialized assistants are given permission to access the user's personal information. "
# "After the Onboarding Wizard has completed its task, transfer to the Goal Wizard. "
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful personal trainer. "
            "Transfer to the Goal Wizard first and immediately! "
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            "Provide detailed information to the user, and always double-check the database before concluding that information is unavailable. "
            "\n\nInfo about the user you're currently chatting with:\n<User>{user_info}</User>"
            "\nCurrent time: {time}",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


## TOOLS


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    import ipdb

    ipdb.set_trace()
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
    cfg = ensure_config()
    configuration = cfg.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    user_profile = fetch_user(user_id)
    return user_profile


@tool
def fetch_user_profile_info():
    """
    Fetch all known mutable information about the user: activity preferences, workout location, workout frequency, workout duration, workout constraints,
         fitness level, weight, goal weight

    Returns:
        The user's profile information, as described above.
    """
    print("Fetching user profile info.......")
    cfg = ensure_config()
    configuration = cfg.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    user_profile = fetch_user_profile(user_id)
    print(f"Fetched user profile for user_id {user_id}: {user_profile}")
    return user_profile


@tool
def set_user_profile_info(user_profile_field: str, user_profile_value: Union[str, int, float, list, dict]):
    """
    Updates a user's profile information in the user_profiles table based on the provided field and value.
    If the field is already set to the provided value, don't use this tool.

    Parameters:
    - user_profile_field (str): The field in the user profile to update.
        Must be one of: 'activity_preferences', 'workout_location', 'workout_frequency', 'workout_duration', 'workout_constraints',
         'fitness_level', 'weight', 'goal_weight'
    - user_profile_value (Union[str, int, float, list, dict]): The new value to set for the specified field.
        The type of this value should match the type of the field in the database schema.

    The schema for the user_profiles table is as follows:
     - activity_preferences TEXT
     - workout_location TEXT
     - workout_frequency TEXT
     - workout_duration TEXT
     - workout_constraints TEXT
     - fitness_level TEXT
     - weight REAL
     - goal_weight REAL
    """

    cfg = ensure_config()
    configuration = cfg.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    if isinstance(user_profile_value, dict):
        user_profile_value = json.dumps(user_profile_value)

    update_user_profile(user_id, user_profile_field, user_profile_value)

    return f"Successfully updated {user_profile_field} to {user_profile_value} for user {user_id}"


@tool
def fetch_goals():
    """
    Fetch all known goals for the user.
    """

    cfg = ensure_config()
    configuration = cfg.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    goals = fetch_goals_db(user_id)
    return goals


@tool
def handle_create_goal(state: State):
    """
    Create a new goal for the user.
    Parameters:
        state: The current state of the dialog.
    """
    cfg = ensure_config()
    configuration = cfg.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    goal_id = str(uuid.uuid4())
    state["current_goal_id"] = goal_id
    create_empty_goal_db(user_id, goal_id)
    return f"Created a new goal with id {goal_id} for user {user_id}"


@tool
def update_goal_state(state: State, goal_id: str):
    """
    Update the current goal id in the state, based on the goal being discussed. To know which goal is being discussed,
    you need to look up the goals for the user which can be done with the fetch_goals tool. The goal_id should be one of the goal_ids
    from the goals table.
    Parameters:
        state: The current state of the dialog.
        goal_id: The id of the goal being discussed.
    """
    state["current_goal_id"] = goal_id
    return f"Updated current goal id to {goal_id}"


@tool
def clear_goal_state(state: State):
    """
    Clear the current goal id in the state.
    Parameters:
        state: The current state of the dialog.
    """
    state["current_goal_id"] = None
    return "Cleared current goal id"


@tool
def update_goal(state: State, goal_field: str, goal_value: Union[str, int, float, list, dict]):
    """
    Updates a user's goal information in the goals table based on the provided field and value.
    If the field is already set to the provided value, don't use this tool.

    Parameters:
    - state (State): The current state of the dialog... this is needed to know which goal to update because it has current_goal_id.
    - goal_field (str): The field in the goal to update.
        Must be one of: 'goal_type', 'description', 'target_value', 'current_value', 'unit', 'start_date', 'end_date', 'goal_status', 'notes', 'last_updated'
    - goal_value (Union[str, int, float, list, dict]): The new value to set for the specified field.
        The type of this value should match the type of the field in the database schema.

    The schema for the goals table is as follows:
     - goal_type TEXT
     - description TEXT
     - target_value TEXT
     - current_value TEXT
     - unit TEXT
     - start_date DATE
     - end_date DATE
     - goal_status TEXT
     - notes TEXT
    """

    cfg = ensure_config()
    configuration = cfg.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    goal_id = state.get("current_goal_id")

    if isinstance(goal_value, dict):
        goal_value = json.dumps(goal_value)

    update_goal_db(user_id, goal_id, goal_field, goal_value)

    return f"Successfully updated {goal_field} to {goal_value} for user {user_id}"


## PRIMARY ASSISTANT
class ToOnboardingWizard(BaseModel):
    """Transfers work to a specialized assistant to handle tasks associated with setting user profile info, including what activities
    the user prefers, where they workout, how often they workout, how long they typically workout for, any constraints they have, what
    their fitness level is, how much they weight and what their goal weight is.
    """

    request: str = Field(description="Any necessary follup questions the onboarding wizard should clarify before proceeding.")


class ToGoalWizard(BaseModel):
    """Transfers work to a specialized assistant to handle goal-setting tasks, except the onboarding wizard handles marking the user's goal weight."""

    request: str = Field(description="Any necessary follup questions the goal wizard should clarify before proceeding.")


## RUNNABLES

# TODO: provide tools

onboarding_wizard_safe_tools = [
    fetch_user_profile_info,
]
onboarding_wizard_sensitive_tools = [set_user_profile_info]
onboarding_wizard_tools = onboarding_wizard_safe_tools + onboarding_wizard_sensitive_tools
onboarding_wizard_runnable = onboarding_wizard_prompt | llm.bind_tools(onboarding_wizard_tools + [CompleteOrEscalate])

goal_wizard_safe_tools = [fetch_goals]
goal_wizard_sensitive_tools = [handle_create_goal, update_goal]
goal_wizard_tools = goal_wizard_safe_tools + goal_wizard_sensitive_tools
goal_wizard_runnable = goal_wizard_prompt | llm.bind_tools(goal_wizard_tools + [CompleteOrEscalate])

primary_assistant_tools = []
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

# onboarding wizard assistant
builder.add_node(
    "enter_onboarding_wizard",
    create_entry_node("Onboarding Wizard", "onboarding_wizard"),
)
builder.add_node("onboarding_wizard", Assistant(onboarding_wizard_runnable))
builder.add_edge("enter_onboarding_wizard", "onboarding_wizard")
builder.add_node(
    "onboarding_wizard_sensitive_tools",
    create_tool_node_with_fallback(onboarding_wizard_sensitive_tools),
)
builder.add_node(
    "onboarding_wizard_safe_tools",
    create_tool_node_with_fallback(onboarding_wizard_safe_tools),
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
    safe_toolnames = [t.name for t in onboarding_wizard_safe_tools]
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
# graph = builder.compile(
#     checkpointer=memory,
#     # Let the user approve or deny the use of sensitive tools
#     interrupt_before=[
#         "onboarding_wizard_sensitive_tools",
#         "goal_wizard_sensitive_tools",
#     ],
# )

visualize_graph = False

if visualize_graph:
    graph_path = Path("graph.png")
    image_data = io.BytesIO(graph.get_graph().draw_mermaid_png())
    image = Image.open(image_data)
    image.save(graph_path)

thread_id = "1"
config_c = {"configurable": {"user_id": "bf9d8cd5-3c89-40ef-965b-ad2ff148e52a", "thread_id": thread_id}}

_printed = set()

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    events = graph.stream({"messages": [("user", user_input)]}, config_c, stream_mode="values")
    for event in events:
        _print_event(event, _printed)

    if 0:
        snapshot = graph.get_state(config_c)
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
                    config_c,
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
                    config_c,
                )
            snapshot = graph.get_state(config_c)
