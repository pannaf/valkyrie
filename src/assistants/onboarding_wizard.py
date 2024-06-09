from src.assistants.base_wizard import BaseWizard


class OnboardingWizard(BaseWizard):
    @property
    def safe_tools(self):
        from src.tools import fetch_user_profile_info

        return [fetch_user_profile_info]

    @property
    def sensitive_tools(self):
        from src.tools import set_user_profile_info

        return [set_user_profile_info]
