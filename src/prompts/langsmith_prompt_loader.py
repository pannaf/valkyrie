from typing import Dict, Any
from langchain import hub

from src.prompts.base_prompt_loader import BasePromptLoader


class LangsmithPromptLoader(BasePromptLoader):

    def _load_from_langsmith(self, prompt_name: str) -> Dict[str, Any]:
        prompt = hub.pull(prompt_name)
        return prompt

    def get_prompt(self, prompt_name: str) -> Dict[str, Any]:
        return self._load_from_langsmith(prompt_name)


##--- Usage


def main():
    langsmith_loader = LangsmithPromptLoader()
    prompt = langsmith_loader.get_prompt("onboarding_wizard_prompt:260d3cd7")
    print(prompt)


if __name__ == "__main__":
    main()


# Usage
# langsmith_loader = LangsmithPromptLoader(client=my_langsmith_client)
# prompt_from_langsmith = langsmith_loader.get_prompt('my-private-prompt')
