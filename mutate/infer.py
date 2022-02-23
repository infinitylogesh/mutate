from transformers import AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
from typing import Optional, List, Dict, Union
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TextGeneration:
    def __init__(self, model_name: str, device: Optional[int] = -1):
        """
        Class for text generation inference. 
        Currently supports only CausalLM models

        Args:
            model_name (str): model name or path to load model from model hub / local path
            device (Optional[int]): GPU Device to run the inference. Positive integer values runs on corresponding device ID.
                                    Defaults to -1 which runs on "cpu".
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,pad_token_id=self.tokenizer.eos_token_id)
        # Done to avoid error from tokenizer, when padding is set True for batch encoding.
        # as per this - https://github.com/huggingface/transformers/issues/4122
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device(f"cuda:{device}" if device>=0 else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def run_single_batch(
        self,
        batch,
        is_include_prompt_in_genaration: Optional[bool] = False,
        ignore_prompt_last_line: Optional[bool] = True,
        generate_args: Optional[Dict[str, str]] = None,
    ):
        batch_size = len(batch["input_ids"])
        generated_sequences = self.model.generate(
            batch["input_ids"], attention_mask=batch["attention_mask"], **generate_args
        )
        generated_sequences = generated_sequences.cpu()
        num_return_sequences = len(generated_sequences) // batch_size

        batch_generated_texts = []

        for idx, generated_sequence in enumerate(generated_sequences):
            text = self.tokenizer.decode(
                generated_sequence,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            prompt_idx = idx // num_return_sequences
            decoded_prompt = self.tokenizer.decode(
                batch["input_ids"][prompt_idx],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            prompt_length = len(decoded_prompt)
            if ignore_prompt_last_line and "\n" in decoded_prompt:
                prompt_length -= len(decoded_prompt.split("\n")[-1])

            if is_include_prompt_in_genaration:
                all_text = batch["prompts"][idx] + text[prompt_length:]
            else:
                all_text = text[prompt_length:]

            batch_generated_texts.append(all_text)

        return batch_generated_texts

