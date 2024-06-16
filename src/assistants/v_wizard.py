from src.assistants.base_wizard import BaseWizard


class VWizard(BaseWizard):
    @property
    def safe_tools(self):

        return []

    @property
    def sensitive_tools(self):

        return []

    @property
    def name(self):
        return "v_wizard"
