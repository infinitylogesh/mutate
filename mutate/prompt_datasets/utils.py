from typing import List, Dict, Union, Optional
from jinja2 import Template

def _create_prompt_for_text_class_synthesize(
        template: str,
        task_desc: str,
        examples: List[str],
        labels: List[str],
        label_title: str,
        example_title: str,
    ) -> str:

        template = Template(template)
        prompt = template.render(
            task_prompt=task_desc,
            labelled_examples=list(zip(labels, examples)),
            label_title=label_title,
            example_title=example_title,
        )
        return prompt

def _create_prompt_for_text_class_pseudo_label(
    template: str,
    task_desc: str,
    labelled_examples: List[str],
    unlabelled_example: str,
    labels: str,
    label_title: str,
    example_title: str,
):

    template = Template(template)
    prompt = template.render(
        task_prompt=task_desc,
        labelled_examples=zip(labels, labelled_examples),
        example_title=example_title,
        label_title=label_title,
        unlabelled_example=unlabelled_example,
    )
    return prompt