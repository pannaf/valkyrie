from src.tools import CompleteOrEscalate, create_tool_node_with_fallback
from src.assistants.assistant import Assistant

# from nemoguardrails import RailsConfig
# from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails


class BaseWizard:
    def __init__(self, llm, prompt_loader, prompt_name):
        self.llm = llm
        self.prompt_loader = prompt_loader
        self.prompt_name = prompt_name
        self.tools = self.safe_tools + self.sensitive_tools
        self.prompt = self.prompt_loader.get_prompt(self.prompt_name)
        self.runnable = self.prompt | self.llm.bind_tools(self.tools + [CompleteOrEscalate])
        self.assistant = Assistant(self.runnable)

    @property
    def safe_tools(self):
        raise NotImplementedError("Subclasses must define safe_tools")

    @property
    def sensitive_tools(self):
        raise NotImplementedError("Subclasses must define sensitive_tools")

    @property
    def name(self):
        raise NotImplementedError("Subclasses must define name")

    def get_runnable(self):
        return self.runnable

    def create_tool_node(self, tools):
        return create_tool_node_with_fallback(tools)
