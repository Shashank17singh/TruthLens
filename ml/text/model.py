"""Fine-tuned encoder (RoBERTa-base or DeBERTa-v3-small) with sliding 512-token windows.

The shared training loop moves one tensor per batch, so a window travels as
[2, max_length]: token ids stacked on the attention mask. A long text becomes
several windows that overlap by `stride` tokens; its score is the mean window
logit.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)


class TextClassifier(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        """[batch, 2, length] ids and mask to [batch] logits; label 1 means AI-written."""
        return self.encoder(input_ids=packed[:, 0], attention_mask=packed[:, 1]).logits[:, 0]

    def embed_forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward from input embeddings, for integrated gradients."""
        return self.encoder(inputs_embeds=embeddings, attention_mask=mask).logits[:, 0]

    def input_embeddings(self) -> nn.Embedding:
        return self.encoder.get_input_embeddings()


def encode_windows(tokenizer: PreTrainedTokenizerBase, text: str, max_length: int, stride: int) -> np.ndarray:
    """[windows, 2, max_length] int64. Every window has the special tokens; padding fills the last."""
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length",
    )
    return np.stack([np.array(encoded["input_ids"]), np.array(encoded["attention_mask"])], 1).astype(np.int64)


def build_model(config: dict, pretrained: bool = True) -> tuple[TextClassifier, PreTrainedTokenizerBase]:
    model_config = config["model"]
    name = model_config["backbone"]
    overrides = model_config.get("overrides")
    tokenizer = AutoTokenizer.from_pretrained(model_config.get("tokenizer") or name)
    if pretrained and not overrides:
        encoder = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1)
    else:
        # `overrides` builds a small random encoder for tests; otherwise only the shape is fetched.
        architecture = AutoConfig.for_model(**overrides, num_labels=1) if overrides else AutoConfig.from_pretrained(name, num_labels=1)
        encoder = AutoModelForSequenceClassification.from_config(architecture)
    return TextClassifier(encoder), tokenizer


@torch.inference_mode()
def score_text(model: TextClassifier, tokenizer: PreTrainedTokenizerBase, text: str, max_length: int, stride: int, batch: int = 8) -> float:
    """Mean logit over all windows of one text."""
    windows = torch.from_numpy(encode_windows(tokenizer, text, max_length, stride)).to(next(model.parameters()).device)
    return float(torch.cat([model(windows[index : index + batch]) for index in range(0, len(windows), batch)]).mean())


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[TextClassifier, PreTrainedTokenizerBase, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model, tokenizer = build_model(config, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), tokenizer, config
