import streamlit as st
import hmac
import re
import boto3
import datetime
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from src.db.db_utils import insert_user

from src.assistant_system import AssistantSystem

import hydra
from hydra.core.global_hydra import GlobalHydra


GlobalHydra.instance().clear()


def load_config():
    with hydra.initialize(config_path="configs", version_base="1.3"):
        cfg = hydra.compose(config_name="base")
    return cfg


config = load_config()
config["user"] = {"user_id": None, "thread_id": None}


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
    st.session_state["password_correct"] = None  # Using None to signify no attempt yet
if "valid_user_info" not in st.session_state:
    st.session_state["valid_user_info"] = False
if "config" not in st.session_state:
    st.session_state["config"] = config


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


dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
table = dynamodb.Table("v-waitlist")


def add_to_waitlist(first_name, last_name, email):
    try:
        table.put_item(
            Item={"email": email, "first_name": first_name, "last_name": last_name, "timestamp": datetime.datetime.now().isoformat()}
        )
    except (NoCredentialsError, PartialCredentialsError):
        st.error("😕 Oops! There was an error adding you to the waitlist. Please try again later.")
        return False


if not st.session_state["password_correct"]:

    tab1, tab2 = st.tabs(["Login", "Join Waitlist"])

    with tab1:
        st.markdown("### Login to access V")
        if not st.session_state["password_correct"]:
            with st.form("login_form"):
                first_name = st.text_input("First Name")
                last_name = st.text_input("Last Name")
                email = st.text_input("Email")

                # Capture height in feet and inches
                height_feet = st.number_input("Height (feet)", min_value=0, max_value=8, step=1)
                height_inches = st.number_input("Height (inches)", min_value=0.0, max_value=11.75, step=0.25)

                # Convert height to inches and then to centimeters
                total_height_inches = height_feet * 12 + height_inches
                height_cm = total_height_inches * 2.54

                dob = st.date_input(
                    "Date of Birth", value=None, max_value=datetime.date(2011, 12, 31), min_value=datetime.date(1900, 1, 1)
                )
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
                                "height": height_cm,
                            }
                            # add user to users table
                            user_id = insert_user(first_name, last_name, email, dob, height_cm)
                            st.session_state["config"]["user"]["user_id"] = user_id
                            st.session_state["config"]["user"]["thread_id"] = user_id
                        else:
                            st.session_state["password_correct"] = False
                            st.error(
                                "😕 Oops! Wrong access code. If you don't have an access code, head on over to the 'Join Waitlist' tab to sign up for updates."
                            )
                    else:
                        st.session_state["valid_user_info"] = False
                        st.error("😕 Please fill in all the fields correctly.")

        if st.session_state["password_correct"]:
            st.success("Access granted. Welcome to the app!")
            st.rerun()  # Rerun is needed to refresh the state
            # Main app code could go here

    with tab2:
        st.markdown("### Join the waitlist to be notified when V is available")
        if not st.session_state["submitted"]:
            with st.form("waitlist_form", clear_on_submit=True):
                first_name = st.text_input("First Name")
                last_name = st.text_input("Last Name")
                email = st.text_input("Email")
                submit_waitlist = st.form_submit_button("Join")
                if submit_waitlist:
                    if is_valid_user_info(first_name, last_name, email):
                        add_to_waitlist(first_name, last_name, email)
                        st.session_state["submitted"] = True
                        # Add to waitlist logic here
                        st.success("Successfully joined the waitlist! 🎉")
                        st.rerun()  # Rerun to refresh the state
                    else:
                        st.error("😕 Please fill in all the fields correctly.")

        if st.session_state["submitted"]:
            st.success("Successfully joined the waitlist! 🎉")

    st.stop()  # Do not continue if check_password is not True.

st.title("V : AI Personal Trainer :muscle: :woman-lifting-weights: :superhero:")

# Initialize the assistant system
if "assistant_system" not in st.session_state:
    st.session_state.assistant_system = AssistantSystem(st.session_state["config"])

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
