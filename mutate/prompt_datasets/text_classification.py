from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Union

from datasets import load_dataset
from jinja2 import Template
from torch.utils.data import DataLoader, IterableDataset

from mutate.prompt_datasets.templates import text_classification_synthesize_template
from mutate.prompt_datasets.utils import _create_prompt_for_text_class_synthesize


class TextClassSynthesizePromptDataset(IterableDataset):
    def __init__(
        self,
        path: str,
        task_desc: str,
        text_column: str,
        label_column: str,
        text_column_alias: Optional[str] = None,
        label_column_alias: Optional[str] = None,
        split: Optional[str] = None,
        data_files: Union[Dict, List] = None,
        data_dir: Optional[str] = None,
        dataset_args: Optional[List[str]] = [],
        dataset_kwargs: Optional[Dict[str, str]] = {},
        class_names: Optional[List[str]] = None,
        shot_count: Optional[int] = 5,
        prompt_template: Optional[str] = text_classification_synthesize_template,
    ):
        """
        Iterable dataset to Loop through the dataset by class and generate prompts.
        Uses 🤗 Datasets for loading and processing the dataset.
        Loading process is similar to 🤗 Datasets -
        https://huggingface.co/docs/datasets/loading.html

        Args:
            path (str): Path or name of the dataset.
                        Can be Huggingface dataset identifier like - `allenai/c4` or
                        For local datasets in CSV,JSON - this param can take `csv`,`json` as values
                        and corresponding path can be sepecified in the `data_files` param.
            task_desc (str): Description about the text classification task. A task description will improve
                             model generation quality. This is a good source about helpful task descriptions -
                             https://docs.cohere.ai/prompt-engineering-wiki
            text_column (str): Name of the column in the dataset which has examples to be used in the prompt and similar
                               examples are to be generated by the model
            label_column (str): Name of the column that contains the label / classes
            split (Optional[str]): Which Split of the dataset to load. Usage is same as in 🤗 Datasets. Defaults to `train` split.
            data_files (Union[Dict, List], optional): Paths to source data files .  Usage is same as in 🤗 Datasets.  Defaults to None.
            data_dir (Optional[str], optional): Directory of the dataset configuration.  Usage is same as in 🤗 Datasets.  Defaults to None.
            text_column_alias (Optional[str]):alias of `text_column`.If the `text_column` does not have a meaningful title. A meaningful title of the example can
                                          be specified here. Defaults to None.
            label_column_alias (Optional[str], optional): alias of `label_column`.If the `label_column` does not have a meaningful title. A meaningful title of the class can
                                          be specified here. Defaults to None.
            dataset_args (Optional[List[str]]): additional args to load the  🤗 Dataset. Defaults to [].
            dataset_kwargs (Optional[Dict[str,str]]): additional keyword arguments to load the 🤗 Dataset. Defaults to {}.
            class_names (Optional[List[str]], optional): Class names to be specified.
                If class names are not part of huggingface datasets features.if values in the label columns are label encoded.
                class name order is expected to follow the same order as the label encoding.Defaults to None.
            shot_count (Optional[int], optional): Number of examples to be used in the few shot prompt. Defaults to 5.

        Examples:
        ---------

        Using 🤗 Datasets:

        >> task_dec = "Each item in the following contains customer service queries expressing the mentioned intent"
        >> TextClassSynthesizePromptDataset(
                "banking77",
                task_desc=task_dec,
                text_column="text",
                label_column="label",
                text_column_alias="Queries", # since the `text_column` doesn't have a meaningful value
                label_column_alias="Intent", # since the `label_column` doesn't have a meaningful value
                dataset_args=["en"],
            )


        Using local csv files

        >> task_dec = "Each item in the following contains customer comments expressing the mentioned sentiment"
        >> TextClassSynthesizePromptDataset(
                "csv",
                data_files=["local/path/sentiment_classfication.csv"],
                task_desc=task_dec,
                text_column="text",
                label_column="label",
                example_title="Comment",
                label_title="sentiment",
                class_names=["positive","negative","neutral"],
                dataset_args=["en"],
            )
        """

        super().__init__()

        self.dataset = load_dataset(
            path,
            *dataset_args,
            split=split,
            data_files=data_files,
            data_dir=data_dir,
            **dataset_kwargs,
        )
        self.text_column = text_column
        self.label_column = label_column
        self.label_title = label_column_alias if label_column_alias else label_column
        self.example_title = text_column_alias if text_column_alias else text_column
        self.task_desc = task_desc
        self.shot_count = shot_count
        self.prompt_template = prompt_template

        if not split:
            splits = list(self.dataset.keys())
            split = "train" if "train" in splits else splits[0]
            self.dataset = self.dataset[split]

        if text_column not in self.dataset.features:
            raise ValueError(
                f"""Supplied text_column - `{text_column}` not in the dataset. Columns in the dataset are {list(self.dataset.features)}"""
            )

        if label_column not in self.dataset.features:
            raise ValueError(
                f"""Supplied label_column - `{label_column}` not in the dataset.
                                 Columns in the dataset are {list(self.dataset.features)}"""
            )

        if class_names:
            self.class_names = class_names
        elif hasattr(self.dataset.features[label_column], "names"):
            self.class_names = self.dataset.features[label_column].names
        else:
            # TODO: Support Identifying unique class from the label column
            raise ValueError(
                """Expecting `class_names` to be passed or 
                    the 🤗 datasets to have class names defined in the label column"""
            )

    @staticmethod
    def _collate_fn(tokenizer, device, batch):
        prompts = [batch_item["prompt"] for batch_item in batch]
        class_names = [batch_item["class_name"] for batch_item in batch]
        model_inputs = tokenizer.batch_encode_plus(
            prompts, padding=True, add_special_tokens=False, return_tensors="pt"
        )
        model_inputs = {key: model_inputs[key].to(device) for key in model_inputs}
        model_inputs["prompts"] = prompts
        model_inputs["class_names"] = class_names
        return model_inputs

    @property
    def _is_label_encoded(self):
        """Naive checking if the values in label column is label encoded"""
        if type(self.class_names[0]) == type(self.dataset[self.label_column][0]):
            return False
        return True

    def __iter__(self) -> Iterator[Dict[str, str]]:
        for idx, class_name in enumerate(self.class_names):
            filter_value = idx if self._is_label_encoded else class_name
            class_dataset = self.dataset.filter(
                lambda ex: ex[self.label_column] == filter_value
            )

            for i in range(0, class_dataset.num_rows, self.shot_count):
                _to = (
                    class_dataset.num_rows
                    if (i + self.shot_count) > class_dataset.num_rows
                    else (i + self.shot_count)
                )
                texts = class_dataset.select(range(i, _to))[self.text_column]
                prompt = _create_prompt_for_text_class_synthesize(
                    template=self.prompt_template,
                    task_desc=self.task_desc,
                    examples=texts,
                    labels=[class_name] * len(texts),
                    label_title=self.label_title,
                    example_title=self.example_title,
                )

                yield {"prompt": prompt, "class_name": class_name}
