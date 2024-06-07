import yaml
from typing import Dict, Any
from datetime import datetime

from src.prompts.base_prompt_loader import BasePromptLoader
from langchain_core.prompts import ChatPromptTemplate


class YamlPromptLoader(BasePromptLoader):
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.prompts = self._load_from_yaml()

    def _load_from_yaml(self) -> Dict[str, Any]:
        with open(self.yaml_path, "r") as file:
            data = yaml.safe_load(file)
            return data.get("prompts", {})

    def get_prompt(self, prompt_name: str) -> ChatPromptTemplate:
        prompt_dict = self.prompts.get(prompt_name, {})
        prompt_tuple = list(prompt_dict.items())
        prompt = ChatPromptTemplate.from_messages(prompt_tuple).partial(time=datetime.now())
        return prompt


##--- Usage


def main():
    yaml_loader = YamlPromptLoader("src/prompts/prompts.yaml")
    prompt = yaml_loader.get_prompt("gandalf")
    print(prompt)


if __name__ == "__main__":
    main()
