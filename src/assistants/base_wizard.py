from langchain_core.messages import ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field


class BaseWizard:
    def __init__(self, name, prompt_template, tools):
        self.name = name
        self.prompt_template = prompt_template
        self.tools = tools
        self.runnable = self.prompt_template | llm.bind_tools(self.tools + [CompleteOrEscalate])

    def create_entry_node(self, new_dialog_state: str):
        def entry_node(state: State):
            tool_call_id = state["messages"][-1].tool_calls[0]["id"]
            return {
                "messages": [
                    ToolMessage(
                        content=f"The assistant is now the {self.name}. Reflect on the above conversation between the host assistant and the user."
                        f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {self.name}."
                        " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                        " Do not mention who you are - just act as the proxy for the assistant.",
                        tool_call_id=tool_call_id,
                    )
                ],
                "dialog_state": new_dialog_state,
            }

        return entry_node

    def add_to_builder(
        self, builder, entry_node_name: str, main_node_name: str, safe_tools_node_name: str, sensitive_tools_node_name: str
    ):
        builder.add_node(entry_node_name, self.create_entry_node(main_node_name))
        builder.add_node(main_node_name, Assistant(self.runnable))
        builder.add_edge(entry_node_name, main_node_name)
        builder.add_node(safe_tools_node_name, create_tool_node_with_fallback([t for t in self.tools if t in safe_tools_node_name]))
        builder.add_node(
            sensitive_tools_node_name, create_tool_node_with_fallback([t for t in self.tools if t in sensitive_tools_node_name])
        )

        def route_wizard(state: State):
            route = tools_condition(state)
            if route == END:
                return END
            tool_calls = state["messages"][-1].tool_calls
            did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
            if did_cancel:
                return "leave_skill"
            safe_toolnames = [t.name for t in [t for t in self.tools if t in safe_tools_node_name]]
            if all(tc["name"] in safe_toolnames for tc in tool_calls):
                return safe_tools_node_name
            return sensitive_tools_node_name

        builder.add_edge(sensitive_tools_node_name, main_node_name)
        builder.add_edge(safe_tools_node_name, main_node_name)
        builder.add_conditional_edges(main_node_name, route_wizard)
