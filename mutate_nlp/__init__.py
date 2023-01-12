from typing import Dict, List, Optional, Union

from transformers import PreTrainedModel

from mutate_nlp.pipelines.text_classification import TextClassificationSynthesize

SUPPORTED_TASKS = {"text-classification-synthesis": TextClassificationSynthesize}


def pipeline(
    task: str,
    model: Union[str, PreTrainedModel],
    device: Optional[int] = -1,
    generation_kwargs: Optional[Dict] = {},
    **kwargs,
):
    """

    A factory method to return the pipeline corresponding to a given task.

    Pipelines are made of:
            - A `Prompt Dataset` - that converts a given dataset to suitable prompts
            - Model inference - to generate text from the prompt
            - A `Parser` to parse the generated texts for examples

    Args:
        task (str): Name of the task to synthesize or psuedo label examples. Currently supported tasks are:

            - `text-classification-synthesize` - will return `TextClassificationSynthesize` pipeline for synthesizing examples for text classification.

        model (Union[str, PreTrainedModel]): Path or 🤗 model hub identifier. Currently supports Causal LM models only.
        device (Optional[int]): GPU Device to run the inference. Defaults to -1, runs on CPU.
        generate_kwargs : Keyword arguments to override default generation params. This will be passed as generation params to model.generate -
            https://github.com/huggingface/transformers/blob/fcb4f11c9232cee2adce8140a3a7689578ea97de/src/transformers/generation_utils.py#L803

    Examples:
    --------

        Using the default generation params:

        >> pipe = pipeline(
                "text-classification-synthesis",
                model="EleutherAI/gpt-neo-2.7B",
                device=1,
            )

        Overriding the generation params:

        >> pipe = pipeline(
                "text-classification-synthesis",
                model="EleutherAI/gpt-neo-2.7B",
                device=1,
                generation_kwargs = {
                                    "max_length":300,
                                    "do_sample":True,
                                    "num_return_sequences":3,
                                    "top_k":40,
                                    "top_p":0.80,
                                    "early_stopping":True,
                                    "no_repeat_ngram_size":2
                })

        Using 🤗 Datasets:

        >> task_desc = "Each item in the following contains customer service queries expressing the mentioned intent"
        >> synthesizerGen = pipe(
                        "banking77",
                        task_desc=task_desc,
                        text_column="text",
                        label_column="label",
                        text_column_alias="Queries", # as the `text_column` doesn't have a meaningful value
                        label_column_alias="Intent", # as the `label_column` doesn't have a meaningful value
                        dataset_args=["en"],
                        )

        >> for exp in synthesizerGen:
                print(exp)


        Using local csv files

        >> task_desc = "Each item in the following contains customer comments expressing the mentioned sentiment"
        >> pipe(
                "csv",
                data_files=["local/path/sentiment_classfication.csv"],
                task_desc=task_desc,
                text_column="text",
                label_column="label",
                text_column_alias="Comment",
                label_column_alias="sentiment",
                class_names=["positive","negative","neutral"],
                dataset_args=["en"],
            )

    Returns:
        the corresponding pipeline class for the task
    """

    if task in SUPPORTED_TASKS:
        pipeline_class = SUPPORTED_TASKS[task]
    else:
        raise ValueError(
            f"Task - {task} is not supported. Supported tasks -"
            f" {SUPPORTED_TASKS.keys()}"
        )

    return pipeline_class(model, device, **generation_kwargs)
