import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from src.prompts.yaml_prompt_loader import YamlPromptLoader
from src.state_graph.graph_builder import GraphBuilder
from src.tools import ToOnboardingWizard, ToGoalWizard

THREAD_ID = "1"
config_c = {"configurable": {"user_id": "bf9d8cd5-3c89-40ef-965b-ad2ff148e52a", "thread_id": THREAD_ID}}


class AssistantSystem:
    def __init__(self):
        load_dotenv()
        self.llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
        self.prompt_loader = YamlPromptLoader("src/prompts/prompts.yaml")

        from src.assistants.goal_wizard import GoalWizard
        from src.assistants.onboarding_wizard import OnboardingWizard

        self.wizards = {
            "onboarding_wizard": OnboardingWizard(self.llm, self.prompt_loader, "onboarding_wizard"),
            "goal_wizard": GoalWizard(self.llm, self.prompt_loader, "goal_wizard"),
        }

        self.primary_assistant_prompt = self.prompt_loader.get_prompt("gandalf")
        self.primary_assistant_runnable = self.primary_assistant_prompt | self.llm.bind_tools([ToOnboardingWizard, ToGoalWizard])

        self.graph_builder = GraphBuilder(self.wizards, self.primary_assistant_runnable)
        self.graph = self.graph_builder.get_graph()

    def handle_event(self, user_input: str):
        _printed = set()

        def _print_event(event: dict, _printed: set, max_length=1500):
            current_state = event.get("dialog_state")
            if current_state:
                st.sidebar.write(f"Currently in: {current_state[-1]}")
            message = event.get("messages")
            if message:
                if isinstance(message, list):
                    message = message[-1]
                if message.id not in _printed:
                    msg_repr = message.pretty_repr(html=True)
                    if len(msg_repr) > max_length:
                        msg_repr = msg_repr[:max_length] + " ... (truncated)"
                    st.sidebar.markdown(msg_repr, unsafe_allow_html=True)
                    _printed.add(message.id)

        events = self.graph.stream({"messages": [("user", user_input)]}, config_c, stream_mode="values")
        for event in events:
            _print_event(event, _printed)

        return event.get("messages", [])[-1].content


##-------- Streamlit app
st.set_page_config(
    page_title="V ai personal trainer",
    page_icon="🏋️‍♂️",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get help": "https://github.com/pannaf/artemis",
        "Report a bug": "mailto:panna@berkeley.edu?subject=Bug%20Report&body=Please%20describe%20the%20bug%20in%20detail.",
        "About": """
        ### V : AI Personal Trainer 
        Entry for the [Generative AI Agents Developer Contest](https://www.nvidia.com/en-us/ai-data-science/generative-ai/developer-contest-with-langchain/) by NVIDIA and LangChain.

        Developed by [Panna Felsen](https://www.linkedin.com/in/panna-felsen-030a3964/).

        -------
        """,
    },
)
st.title("V : AI Personal Trainer :muscle: :woman-lifting-weights: :superhero:")

# Initialize the assistant system
if "assistant_system" not in st.session_state:
    st.session_state.assistant_system = AssistantSystem()

# Handle user input and display the assistant's responses
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Message V..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the input with the AssistantSystem
    response = st.session_state["assistant_system"].handle_event(prompt)

    # Placeholder for assistant's response
    with st.chat_message("V"):
        st.markdown(response)
    st.session_state.messages.append({"role": "V", "content": response})
