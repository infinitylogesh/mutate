from abc import ABC, abstractmethod
from jinja2 import Template
from typing import Dict, List, Optional



class GenerationParser(ABC):
    def _get_regex(
        self, regex_template: str, template_params: Optional[Dict[str, str]] = {}
    ):

        if not template_params:
            return regex_template

        template = Template(regex_template)
        return template.render(**template_params)

    @abstractmethod
    def parse(self, generated_text: str) -> List[str]:
        raise NotImplementedError("Method to be implemented")


