# 🦠 Mutate   <br>

A library to synthesize text datasets using Large Language Models (LLM). Mutate reads through the examples in the dataset and 
generates similar examples using auto generated few shot prompts.

## 1. Installation

```
pip install git+https://github.com/infinitylogesh/mutate
```

## 2. Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dPDVl3EvmsnJc7lxWYdAnTtlJgJjR2O2?usp=sharing)


### 2.1 Synthesize text data from local csv files

```python
from mutate import pipeline

pipe = pipeline("text-classification-synthesis",
                model="EleutherAI/gpt-neo-2.7B",
                device=1)

task_desc = "Each item in the following contains movie reviews and corresponding sentiments. Possible sentimets are neg and pos"


# returns a python generator  
text_synth_gen = pipe("csv",
                    data_files=["local/path/sentiment_classfication.csv"],
                    task_desc=task_desc,
                    text_column="text",
                    label_column="label",
                    text_column_alias="Comment",
                    label_column_alias="sentiment",
                    shot_count=5,
                    class_names=["pos","neg"])

#Loop through the generator to synthesize examples by class
for synthesized_examples  in text_synth_gen:
    print(synthesized_examples)
```

<details>
<summary>Show Output</summary>

```python
{
    "text": ["The story was very dull and was a waste of my time. This was not a film I would ever watch. The acting was bad. I was bored. There were no surprises. They showed one dinosaur,",
    "I did not like this film. It was a slow and boring film, it didn't seem to have any plot, there was nothing to it. The only good part was the ending, I just felt that the film should have ended more abruptly."]
    "label":["neg","neg"]
}

{
    "text":["The Bell witch is one of the most interesting, yet disturbing films of recent years. It’s an odd and unique look at a very real, but very dark issue. With its mixture of horror, fantasy and fantasy adventure, this film is as much a horror film as a fantasy film. And it‘s worth your time. While the movie has its flaws, it is worth watching and if you are a fan of a good fantasy or horror story, you will not be disappointed."],
    "label":["pos"]
}

# and so on .....

```
</details>


### 2.2 Synthesize text data from 🤗 datasets

Under the hood Mutate uses the wonderful 🤗 datasets library for dataset processing, So it supports 🤗 datasets out of the box.

```python

from mutate import pipeline

pipe = pipeline("text-classification-synthesis",
                model="EleutherAI/gpt-neo-2.7B",
                device=1)

task_desc = "Each item in the following contains customer service queries expressing the mentioned intent"

synthesizerGen = pipe("banking77",
                    task_desc=task_desc,
                    text_column="text",
                    label_column="label",
                    # if the `text_column` doesn't have a meaningful value
                    text_column_alias="Queries", 
                    label_column_alias="Intent", # if the `label_column` doesn't have a meaningful value
                    shot_count=5,
                    dataset_args=["en"])
                       
                       
for exp in synthesizerGen:
    print(exp)

```

<details>
<summary>Show Output</summary>

```python
{"text":["How can i know if my account has been activated? (This is the one that I am confused about)",
         "Thanks! My card activated"],
"label":["activate_my_card",
         "activate_my_card"]
}

{
"text": ["How do i activate this new one? Is it possible?",
         "what is the activation process for this card?"],
"label":["activate_my_card",
         "activate_my_card"]
}

# and so on .....

```
</details>


### 2.3 I am feeling lucky : Infinetly loop through the dataset to generate examples indefinetly

**Caution**: Infinetly looping through the dataset has a higher chance of duplicate examples to be generated.

```python

from mutate import pipeline

pipe = pipeline("text-classification-synthesis",
                model="EleutherAI/gpt-neo-2.7B",
                device=1)

task_desc = "Each item in the following contains movie reviews and corresponding sentiments. Possible sentimets are neg and pos"


# returns a python generator  
text_synth_gen = pipe("csv",
                    data_files=["local/path/sentiment_classfication.csv"],
                    task_desc=task_desc,
                    text_column="text",
                    label_column="label",
                    text_column_alias="Comment",
                    label_column_alias="sentiment",
                    class_names=["pos","neg"],
                    # Flag to generate indefinite examples
                    infinite_loop=True)

#Infinite loop
for exp in synthesizerGen:
    print(exp)
```


## 3. Support
### 3.1 Currently supports
-  **Text classification dataset synthesis** : Few Shot text data synsthesize for text classification datasets using Causal LLMs ( GPT like )

### 3.2 Roadmap:
- **Other types of text Dataset synthesis** - NER , sentence pairs etc 
- Finetuning support for better quality generation
- Pseudo labelling 


## 4. Credit
- [EleutherAI](https://eluether.ai) for democratizing Large LMs.
- This library uses 🤗 [Datasets](https://huggingface.co/docs/datasets) and 🤗 [Transformers](https://huggingface.co/docs/transformers) for processing datasets and models.


## 5. References

The Idea of generating examples from Large Language Model is inspired by the works below,
- [A Few More Examples May Be Worth Billions of Parameters](https://arxiv.org/abs/2110.04374) by Yuval Kirstain, Patrick Lewis, Sebastian Riedel, Omer Levy
- [GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation](https://arxiv.org/abs/2104.08826) by Kang Min Yoo, Dongju Park, Jaewook Kang, Sang-Woo Lee, Woomyeong Park
- [Data Augmentation using Pre-trained Transformer Models](https://arxiv.org/abs/2003.02245) by Varun Kumar, Ashutosh Choudhary, Eunah Cho

