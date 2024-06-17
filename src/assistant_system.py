from dotenv import load_dotenv
from omegaconf import DictConfig
import hydra

from langchain_anthropic import ChatAnthropic

from src.prompts.yaml_prompt_loader import YamlPromptLoader
from src.state_graph.graph_builder import GraphBuilder
from src.tools import ToOnboardingWizard, ToGoalWizard, ToProgrammingWizard, ToVWizard
from src.utils.logtils import configure_logging, LoggingContextManager, get_bound_logger


class AssistantSystem:
    def __init__(self, cfg):
        load_dotenv()
        self.llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
        self.prompt_loader = YamlPromptLoader("src/prompts/prompts.yaml")
        self.cfg = cfg
        self.logger = get_bound_logger()

        from src.assistants.goal_wizard import GoalWizard
        from src.assistants.onboarding_wizard import OnboardingWizard
        from src.assistants.programming_wizard import ProgrammingWizard
        from src.assistants.v_wizard import VWizard

        self.wizards = {
            "onboarding_wizard": OnboardingWizard(self.llm, self.prompt_loader, "onboarding_wizard"),
            "goal_wizard": GoalWizard(self.llm, self.prompt_loader, "goal_wizard"),
            "programming_wizard": ProgrammingWizard(self.llm, self.prompt_loader, "programming_wizard"),
            "v_wizard": VWizard(self.llm, self.prompt_loader, "v_wizard"),
        }

        self.primary_assistant_prompt = self.prompt_loader.get_prompt("gandalf")
        self.primary_assistant_runnable = self.primary_assistant_prompt | self.llm.bind_tools(
            [ToOnboardingWizard, ToGoalWizard, ToProgrammingWizard, ToVWizard]
        )

        self.logger.info("Building graph")

        self.graph_builder = GraphBuilder(self.wizards, self.primary_assistant_runnable)
        self.graph = self.graph_builder.get_graph()

    def visualize_graph(self, path: str):
        self.graph_builder.visualize_graph(self.graph, path)

    def handle_event(self, user_input: str):
        _logged = set()

        def _log_event(event: dict, _logged: set, max_length=1500):
            current_state = event.get("dialog_state")
            if current_state:
                self.logger.debug(f"Current state: {current_state[-1]}")
            message = event.get("messages")
            if message:
                if isinstance(message, list):
                    message = message[-1]
                if message.id not in _logged:
                    msg_repr = message.pretty_repr(html=True)
                    if len(msg_repr) > max_length:
                        msg_repr = msg_repr[:max_length] + " ... (truncated)"
                    self.logger.info(msg_repr)
                    _logged.add(message.id)

        events = self.graph.stream({"messages": [("user", user_input)]}, self.cfg, stream_mode="values")
        for event in events:
            _log_event(event, _logged)

        return event.get("messages", [])[-1].content


@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig):
    user_id = "bf9d8cd5-3c89-40ef-965b-ad2ff148e52a"
    thread_id = "10"
    v_cfg = {"configurable": {"user_id": user_id, "thread_id": thread_id}}

    configure_logging(cfg)
    logger_context_manager = LoggingContextManager(user_id)

    with logger_context_manager as log:
        try:
            with log.catch(reraise=False):
                log.info("Starting V")

                assistant_system = AssistantSystem(v_cfg)

                v_graph_savefile = "v_graph.png"
                log.info(f"Saving graph visualization to {v_graph_savefile}")
                assistant_system.visualize_graph(v_graph_savefile)

                while True:
                    user_input = input("\n---------------- User Message ----------------\nUser: ")
                    if user_input.lower() in ["quit", "exit", "q"]:
                        log.info("User ended the conversation. Goodbye!")
                        break
                    response = assistant_system.handle_event(user_input)
                    print(f"\n----------------- V Message -----------------\nV: {response}")

        finally:
            log.info(f"Completed execution for user {logger_context_manager.user_id}")


if __name__ == "__main__":
    main()
