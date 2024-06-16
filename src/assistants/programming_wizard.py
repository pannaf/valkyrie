from src.assistants.base_wizard import BaseWizard


class ProgrammingWizard(BaseWizard):
    @property
    def safe_tools(self):
        from src.tools import fetch_goals, fetch_user_profile_info, fetch_exercises

        return [fetch_goals, fetch_user_profile_info, fetch_exercises]

    @property
    def sensitive_tools(self):
        # from src.tools import handle_create_goal, update_goal

        return []

    @property
    def name(self):
        return "programming_wizard"
