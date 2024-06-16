from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from src.prompts.yaml_prompt_loader import YamlPromptLoader
from src.state_graph.graph_builder import GraphBuilder
from src.tools import ToOnboardingWizard, ToGoalWizard, ToProgrammingWizard


class AssistantSystem:
    def __init__(self):
        load_dotenv()
        self.llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
        self.prompt_loader = YamlPromptLoader("src/prompts/prompts.yaml")

        from src.assistants.goal_wizard import GoalWizard
        from src.assistants.onboarding_wizard import OnboardingWizard
        from src.assistants.programming_wizard import ProgrammingWizard

        self.wizards = {
            "onboarding_wizard": OnboardingWizard(self.llm, self.prompt_loader, "onboarding_wizard"),
            "goal_wizard": GoalWizard(self.llm, self.prompt_loader, "goal_wizard"),
            "programming_wizard": ProgrammingWizard(self.llm, self.prompt_loader, "programming_wizard"),
        }

        self.primary_assistant_prompt = self.prompt_loader.get_prompt("gandalf")
        self.primary_assistant_runnable = self.primary_assistant_prompt | self.llm.bind_tools(
            [ToOnboardingWizard, ToGoalWizard, ToProgrammingWizard]
        )

        self.graph_builder = GraphBuilder(self.wizards, self.primary_assistant_runnable)
        self.graph = self.graph_builder.get_graph()

    def visualize_graph(self, path: str):
        self.graph_builder.visualize_graph(self.graph, path)

    def handle_event(self, user_input: str):
        THREAD_ID = "1"
        config_c = {"configurable": {"user_id": "bf9d8cd5-3c89-40ef-965b-ad2ff148e52a", "thread_id": THREAD_ID}}
        _printed = set()

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
    assistant_system.visualize_graph("graph_v3.png")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        assistant_system.handle_event(user_input)
