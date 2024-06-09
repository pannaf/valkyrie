from datetime import date
import json

from langchain_core.runnables import Runnable
from langchain_core.messages import ToolMessage
from langgraph.graph import END

from langgraph.prebuilt import tools_condition

from src.tools import CompleteOrEscalate, create_tool_node_with_fallback, fetch_user_info
from src.state_graph.state_graph_builder import StateGraphBuilder
from src.assistants.assistant import Assistant


class WizardGraphBuilder:
    def __init__(self, llm, primary_assistant_prompt):
        self.llm = llm
        self.primary_assistant_prompt = primary_assistant_prompt
        self.graph_builder = StateGraphBuilder()
        self.graph_builder.add_node("fetch_user_info", self.user_info)
        self.graph_builder.set_entry_point("fetch_user_info")
        self.wizard_classes = {}

    def register_wizard(self, wizard_class, wizard_prompt, wizard_name):
        wizard = wizard_class(self.llm, wizard_prompt)
        self.wizard_classes[wizard_name] = wizard
        self.graph_builder.add_node(f"enter_{wizard_name}", self.create_entry_node(wizard_name, wizard_name))
        self.graph_builder.add_node(wizard_name, wizard.assistant)
        self.graph_builder.add_edge(f"enter_{wizard_name}", wizard_name)
        self.graph_builder.add_node(f"{wizard_name}_sensitive_tools", wizard.create_tool_node(wizard.sensitive_tools))
        self.graph_builder.add_node(f"{wizard_name}_safe_tools", wizard.create_tool_node(wizard.safe_tools))
        self.graph_builder.add_conditional_edges(wizard_name, lambda state: self.route_wizard(state, wizard_name))
        self.graph_builder.add_edge(f"{wizard_name}_sensitive_tools", wizard_name)
        self.graph_builder.add_edge(f"{wizard_name}_safe_tools", wizard_name)

    def build(self):
        self.graph_builder.add_node("leave_skill", self.pop_dialog_state)
        self.graph_builder.add_edge("leave_skill", "primary_assistant")
        self.graph_builder.add_node("primary_assistant", Assistant(self.primary_assistant_prompt | self.llm.bind_tools([])))
        self.graph_builder.add_node("primary_assistant_tools", create_tool_node_with_fallback([]))
        self.graph_builder.add_conditional_edges(
            "primary_assistant",
            self.route_primary_assistant,
            {
                "enter_onboarding_wizard": "enter_onboarding_wizard",
                "enter_goal_wizard": "enter_goal_wizard",
                "primary_assistant_tools": "primary_assistant_tools",
                END: END,
            },
        )
        self.graph_builder.add_edge("primary_assistant_tools", "primary_assistant")
        self.graph_builder.add_conditional_edges("fetch_user_info", self.route_to_workflow)
        return self.graph_builder.compile_graph()

    @staticmethod
    def create_entry_node(assistant_name: str, new_dialog_state: str):
        def entry_node(state):
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

    @staticmethod
    def user_info(state):
        def serialize_realdictrow(row):
            def convert_value(value):
                if isinstance(value, date):
                    return value.isoformat()
                return value

            return {key: convert_value(value) for key, value in row.items()}

        _user_info = serialize_realdictrow(fetch_user_info.invoke({}))
        _user_info = json.dumps(_user_info, indent=4)
        return {"user_info": _user_info}

    def route_wizard(self, state, wizard_name):
        route = tools_condition(state)
        if route == END:
            return END
        tool_calls = state["messages"][-1].tool_calls
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
        if did_cancel:
            return "leave_skill"
        safe_toolnames = [t.name for t in self.wizard_classes[wizard_name].safe_tools]
        if all(tc["name"] in safe_toolnames for tc in tool_calls):
            return f"{wizard_name}_safe_tools"
        return f"{wizard_name}_sensitive_tools"

    @staticmethod
    def pop_dialog_state(state):
        messages = []
        if state["messages"][-1].tool_calls:
            messages.append(
                ToolMessage(
                    content="Resuming dialog with the host assistant. Please reflect on the past conversation and continue the conversation.",
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                )
            )
        return {"dialog_state": "pop", "messages": messages}

    def route_primary_assistant(self, state):
        route = tools_condition(state)
        if route == END:
            return END
        tool_calls = state["messages"][-1].tool_calls
        if tool_calls:
            for wizard_name in self.wizard_classes:
                if tool_calls[0]["name"] == f"To{wizard_name.capitalize()}Wizard":
                    return f"enter_{wizard_name}"
            return "primary_assistant_tools"
        raise ValueError("Invalid route")

    @staticmethod
    def route_to_workflow(state):
        dialog_state = state.get("dialog_state")
        if not dialog_state:
            return "primary_assistant"
        return dialog_state[-1]
