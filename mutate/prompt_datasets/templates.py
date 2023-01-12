text_classification_synthesize_template = """
{{task_prompt}}
{% for label,example in labelled_examples %}
{{label_title}} : {{label}} , {{example_title}} : {{example}}
{%- endfor %}
{{label_title}} : {{labelled_examples[0][0]}} , {{example_title}} :"""

text_classification_pseudo_label_template = """
{{task_prompt}}
{% for label,example in labelled_examples %}
{{example_title}} : {{example}} , {{label_tile}} : {{label}}
{%- endfor %}
{{example_title}} : {{unlabelled_example}} , {{label_title}} :"""