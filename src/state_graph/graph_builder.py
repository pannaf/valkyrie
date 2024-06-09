import json
from typing import Literal, Callable
import io
from pathlib import Path
from PIL import Image

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from src.state_graph.state import State
from src.tools import fetch_user_info, create_tool_node_with_fallback
from src.assistants.assistant import Assistant
from langchain_core.messages import ToolMessage
from src.tools import ToOnboardingWizard, ToGoalWizard, CompleteOrEscalate
from src.assistants.base_wizard import BaseWizard
from langgraph.prebuilt import tools_condition

from datetime import date


class GraphBuilder:
    def __init__(self, wizards, primary_assistant_runnable):
        self.wizards = wizards
        self.primary_assistant_runnable = primary_assistant_runnable
        self.builder = StateGraph(State)

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

    def _route_wizard(self, wizard: BaseWizard):
        def _route_wizard_fn(state: State) -> Literal["leave_skill", f"{wizard.name}_safe_tools", f"{wizard.name}_sensitive_tools"]:
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

        return _route_wizard_fn

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

    def build(self):
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
            self.builder.add_conditional_edges(
                wizard_name,
                self._route_wizard(wizard),
                {
                    f"{wizard_name}_safe_tools": f"{wizard_name}_safe_tools",
                    f"{wizard_name}_sensitive_tools": f"{wizard_name}_sensitive_tools",
                    "leave_skill": "leave_skill",
                    END: END,
                },
            )

        # Primary Assistant
        self.builder.add_node("leave_skill", self._pop_dialog_state)
        self.builder.add_edge("leave_skill", "primary_assistant")
        self.builder.add_node("primary_assistant", Assistant(self.primary_assistant_runnable))
        self.builder.add_node("primary_assistant_tools", create_tool_node_with_fallback([]))
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

        return self.builder

    def compile(self, builder):
        memory = SqliteSaver.from_conn_string(":memory:")
        return builder.compile(checkpointer=memory)

    def get_graph(self):
        builder = self.build()
        return self.compile(builder)

    def visualize_graph(self, graph, path: str):
        graph_path = Path(path)
        image_data = io.BytesIO(graph.get_graph().draw_mermaid_png())
        image = Image.open(image_data)
        image.save(graph_path)
