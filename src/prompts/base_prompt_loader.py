from abc import ABC, abstractmethod
from typing import Dict, Any


class BasePromptLoader(ABC):
    @abstractmethod
    def get_prompt(self, prompt_name: str) -> Dict[str, Any]:
        pass
