import torch
from typing import List, Dict, Any, Union
from transformers import PreTrainedTokenizerBase

class SFTDataCollator:
    """
    Custom Data Collator for SFT that masks labels for system and user roles.
    Labels for non-assistant tokens are set to -100 (ignored by CrossEntropyLoss).
    """
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        max_length: int = 1024,
        assistant_header: str = "<|assistant|>",
        user_header: str = "<|user|>",
        system_header: str = "<|system|>"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.assistant_header = assistant_header
        self.user_header = user_header
        self.system_header = system_header

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Processes a list of samples into a batch of tensors.
        Each sample is expected to have a 'messages' field (list of Dicts).
        """
        all_input_ids = []
        all_labels = []
        all_attention_mask = []

        for sample in samples:
            messages = sample.get("messages", [])
            # Construct full text using the chat template or manually if no template
            full_text = ""
            labels = []
            input_ids = []

            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                # Role Header
                header = self.system_header if role == "system" else (self.user_header if role == "user" else self.assistant_header)
                header_tokens = self.tokenizer.encode(header, add_special_tokens=False)
                
                # Content Tokens
                content_tokens = self.tokenizer.encode(content, add_special_tokens=False)
                
                # Combine
                segment_tokens = header_tokens + content_tokens
                input_ids.extend(segment_tokens)
                
                # Masking logic
                if role == "assistant":
                    # Keep labels for content, mask for header
                    labels.extend([-100] * len(header_tokens))
                    labels.extend(content_tokens)
                else:
                    # Mask everything for system/user
                    labels.extend([-100] * len(segment_tokens))

            # Add EOS token
            input_ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)

            # Truncation
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            
            all_input_ids.append(torch.tensor(input_ids))
            all_labels.append(torch.tensor(labels))
            all_attention_mask.append(torch.ones(len(input_ids)))

        # Dynamic Padding
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=pad_id
        )
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            all_labels, batch_first=True, padding_value=-100
        )
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            all_attention_mask, batch_first=True, padding_value=0
        )

        return {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
            "attention_mask": batch_attention_mask
        }
