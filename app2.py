import streamlit as st
import hmac
import re
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

# Initialize session state variables if they don't exist
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False
if "password_correct" not in st.session_state:
    st.session_state["password_correct"] = None  # Use None to signify no attempt yet
if "valid_user_info" not in st.session_state:
    st.session_state["valid_user_info"] = False


def check_password(password):
    """Checks whether a password entered by the user is correct."""
    return hmac.compare_digest(password, st.secrets["password"])


def is_valid_email(email):
    """Validates the email format."""
    email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(email_regex, email) is not None


def is_valid_user_info(first_name, last_name, email):
    """Validates the user information."""
    return first_name and last_name and is_valid_email(email)


if not st.session_state["password_correct"]:

    tab1, tab2 = st.tabs(["Login", "Join Waitlist"])

    with tab1:
        st.markdown("### Login to access V")
        if not st.session_state["password_correct"]:
            with st.form("login_form"):
                first_name = st.text_input("First Name")
                last_name = st.text_input("Last Name")
                email = st.text_input("Email")
                dob = st.date_input("Date of Birth")
                password = st.text_input("Access Code", type="password")
                submit_login = st.form_submit_button("Login")

                if submit_login:
                    if is_valid_user_info(first_name, last_name, email):
                        st.session_state["valid_user_info"] = True
                        if check_password(password):
                            st.session_state["password_correct"] = True
                            # Store user information if needed
                            st.session_state["user_info"] = {
                                "first_name": first_name,
                                "last_name": last_name,
                                "email": email,
                                "dob": dob,
                            }
                        else:
                            st.session_state["password_correct"] = False
                            st.error(
                                "😕 Oops! Wrong access code.\nIf you don't have an access code, head on over to the 'Join Waitlist' tab to sign up for updates."
                            )
                    else:
                        st.session_state["valid_user_info"] = False
                        st.error("😕 Please fill in all the fields correctly.")

        if st.session_state["password_correct"]:
            st.success("Access granted. Welcome to the app!")
            st.rerun()  # Rerun to refresh the state
            # Main app code goes here

    with tab2:
        st.markdown("### Join the waitlist to be notified when V is available")
        if not st.session_state["submitted"]:
            with st.form("waitlist_form", clear_on_submit=True):
                first_name = st.text_input("First Name")
                last_name = st.text_input("Last Name")
                email = st.text_input("Email")
                submit_waitlist = st.form_submit_button("Join")
                if submit_waitlist:
                    st.session_state["submitted"] = True
                    # Add to waitlist logic here
                    st.success("Thank you for joining the waitlist! 🎉")
                    st.rerun()  # Rerun to refresh the state

        if st.session_state["submitted"]:
            st.success("Thank you for joining the waitlist! 🎉")

    st.stop()  # Do not continue if check_password is not True.

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
