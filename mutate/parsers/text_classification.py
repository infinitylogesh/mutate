import re
from typing import Dict, List, Optional

from mutate.parsers.parser import GenerationParser


class TextClassificationSynthesizeParser(GenerationParser):
    regex_template = """.*{{label_title}}.*:.*{{label}}.*,.*{{example_title}}.*:(.*)"""

    def __init__(
        self,
        label_title: str,
        example_title: str,
        regex_template: Optional[str] = regex_template,
    ):
        """
        Parser to parse generated text to extract synthesized examples for a given class

        Args:
            label_title (str): Name / title of the label that describes the classes.
                               Example - For sentiment classfication, the label title can be `Sentiment`
            example_title (str): Name / title of the label that describes the classes
                                 Example - For sentiment classfication, the example title can be `Review`
            regex_template (Optional[str]): Regex and jinja template to parse example generated. Defaults to regex_template.
        """

        self.template_params = {
            "label_title": label_title,
            "example_title": example_title,
        }

    def parse(self, generated_text: str, label: str) -> List[str]:
        """
        Args:
            generated_text (str): Generated text from the LLM
            label (str): Class label

        Returns:
            List[str]: List of parsed examples from the generated texts
        """

        self.template_params["label"] = label
        regex_pattern = self._get_regex(self.regex_template, self.template_params)
        return re.findall(regex_pattern, generated_text)


class TextClassificationPseudolabelParser(GenerationParser):
    regex_template = (
        """.*{{label_title}}.*:.*(.*).*,.*{{example_title}}.*:.*{{example}}"""
    )

    def __init__(
        self,
        label_title: str,
        example_title: str,
        regex_template: Optional[str] = regex_template,
    ):
        """
        Parser to parse generated text to extract lables generated for a given example
        by the text generation model

        Args:
            label_title (str): Name / title of the label that describes the classes.
                               Example - For sentiment classfication, the label title can be `Sentiment`
            example_title (str): Name / title of the label that describes the classes
                                 Example - For sentiment classfication, the example title can be `Review`
            regex_template (Optional[str]): Regex and jinja template to parse example generated. Defaults to regex_template.
        """

        self.template_params = {
            "label_title": label_title,
            "example_title": example_title,
        }

    def parse(self, generated_text: str, example: str) -> List[str]:
        self.template_params["example"] = example
        regex_pattern = self._get_regex(self.regex_template, self.template_params)
        return re.findall(regex_pattern, generated_text)
