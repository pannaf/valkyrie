from src.assistants.base_wizard import BaseWizard
from src.tools import fetch_goals, fetch_user_activities, fetch_exercises


class ProgrammingWizard(BaseWizard):
    @property
    def safe_tools(self):

        return [fetch_goals, fetch_user_activities, fetch_exercises]

    @property
    def sensitive_tools(self):

        return []

    @property
    def name(self):
        return "programming_wizard"
