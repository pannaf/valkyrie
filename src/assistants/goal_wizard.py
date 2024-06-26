from src.assistants.base_wizard import BaseWizard


class GoalWizard(BaseWizard):
    @property
    def safe_tools(self):
        from src.tools import fetch_goals

        return [fetch_goals]

    @property
    def sensitive_tools(self):
        from src.tools import create_goal, update_goal

        return [create_goal, update_goal]

    @property
    def name(self):
        return "goal_wizard"
