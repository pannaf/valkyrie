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
from src.assistants.base_wizard import BaseWizard

from src.tools import CompleteOrEscalate, create_tool_node_with_fallback
from src.assistants.assistant import Assistant

load_dotenv()

THREAD_ID = "1"
config_c = {"configurable": {"user_id": "bf9d8cd5-3c89-40ef-965b-ad2ff148e52a", "thread_id": THREAD_ID}}
_printed = set()


class AssistantSystem:
    def __init__(self):
        self.llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
        self.prompt_loader = YamlPromptLoader("src/prompts/prompts.yaml")

        self.wizards = {
            "onboarding_wizard": OnboardingWizard(self.llm, self.prompt_loader, "onboarding_wizard"),
            "goal_wizard": GoalWizard(self.llm, self.prompt_loader, "goal_wizard"),
        }

        self.primary_assistant_tools = []
        self.primary_assistant_prompt = self.prompt_loader.get_prompt("gandalf")
        self.assistant_runnable = self.primary_assistant_prompt | self.llm.bind_tools(
            self.primary_assistant_tools + [ToOnboardingWizard, ToGoalWizard]
        )

        self.builder = StateGraph(State)
        self._setup_graph()
        self.memory = SqliteSaver.from_conn_string(":memory:")
        self.graph = self.builder.compile(checkpointer=self.memory)

    def _create_entry_node(self, wizard_name: str) -> Callable:
        def entry_node(state: State) -> dict:
            tool_call_id = state["messages"][-1].tool_calls[0]["id"]
            return {
                "messages": [
                    ToolMessage(
                        content=f"The assistant is now the {wizard_name.replace('_', ' ').title()}. Reflect on the above conversation between the host assistant and the user."
                        f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {wizard_name.replace('_', ' ').title()}."
                        " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                        " Do not mention who you are - just act as the proxy for the assistant.",
                        tool_call_id=tool_call_id,
                    )
                ],
                "dialog_state": wizard_name,
            }

        return entry_node

    def _fetch_user_info(self, state: State):
        def serialize_realdictrow(row):
            def convert_value(value):
                if isinstance(value, date):
                    return value.isoformat()
                return value

            return {key: convert_value(value) for key, value in row.items()}

        _user_info = serialize_realdictrow(fetch_user_info.invoke({}))
        _user_info = json.dumps(_user_info, indent=4)
        return {"user_info": _user_info}

    def _pop_dialog_state(self, state: State) -> dict:
        messages = []
        if state["messages"][-1].tool_calls:
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

    def _route_wizard(self, wizard: BaseWizard, state: State):
        route = tools_condition(state)
        if route == END:
            return END
        tool_calls = state["messages"][-1].tool_calls
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
        if did_cancel:
            return "leave_skill"
        safe_toolnames = [t.name for t in wizard.safe_tools]
        if all(tc["name"] in safe_toolnames for tc in tool_calls):
            return f"{wizard.name}_safe_tools"
        return f"{wizard.name}_sensitive_tools"

    def _route_primary_assistant(
        self, state: State
    ) -> Literal["primary_assistant_tools", "enter_onboarding_wizard", "enter_goal_wizard", "__end__"]:
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

    def _route_to_workflow(self, state: State) -> Literal[
        "primary_assistant",
        "onboarding_wizard",
        "goal_wizard",
    ]:
        dialog_state = state.get("dialog_state")
        if not dialog_state:
            return "primary_assistant"
        return dialog_state[-1]

    def _setup_graph(self):
        self.builder.add_node("fetch_user_info", self._fetch_user_info)
        self.builder.set_entry_point("fetch_user_info")

        # Adding wizards to the graph
        for wizard_name, wizard in self.wizards.items():
            self.builder.add_node(f"enter_{wizard_name}", self._create_entry_node(wizard_name))
            self.builder.add_node(wizard_name, Assistant(wizard.runnable))
            self.builder.add_edge(f"enter_{wizard_name}", wizard_name)
            self.builder.add_node(f"{wizard_name}_safe_tools", wizard.create_tool_node(wizard.safe_tools))
            self.builder.add_node(f"{wizard_name}_sensitive_tools", wizard.create_tool_node(wizard.sensitive_tools))
            self.builder.add_edge(f"{wizard_name}_sensitive_tools", wizard_name)
            self.builder.add_edge(f"{wizard_name}_safe_tools", wizard_name)
            self.builder.add_conditional_edges(wizard_name, lambda state, wn=wizard_name: self._route_wizard(self.wizards[wn], state))

        # Primary Assistant
        self.builder.add_node("leave_skill", self._pop_dialog_state)
        self.builder.add_edge("leave_skill", "primary_assistant")
        self.builder.add_node("primary_assistant", Assistant(self.assistant_runnable))
        self.builder.add_node("primary_assistant_tools", create_tool_node_with_fallback(self.primary_assistant_tools))
        self.builder.add_conditional_edges(
            "primary_assistant",
            self._route_primary_assistant,
            {
                "enter_onboarding_wizard": "enter_onboarding_wizard",
                "enter_goal_wizard": "enter_goal_wizard",
                "primary_assistant_tools": "primary_assistant_tools",
                END: END,
            },
        )
        self.builder.add_edge("primary_assistant_tools", "primary_assistant")

        # Workflow routing
        self.builder.add_conditional_edges("fetch_user_info", self._route_to_workflow)

    def visualize_graph(self, path: str):
        graph_path = Path(path)
        image_data = io.BytesIO(self.graph.get_graph().draw_mermaid_png())
        image = Image.open(image_data)
        image.save(graph_path)

    def handle_event(self, user_input: str):

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

        events = self.graph.stream({"messages": [("user", user_input)]}, config_c, stream_mode="values")
        for event in events:
            _print_event(event, _printed)


if __name__ == "__main__":
    assistant_system = AssistantSystem()

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        assistant_system.handle_event(user_input)
