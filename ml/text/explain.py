"""Integrated gradients on token embeddings of the fine-tuned transformer.

Attribution of token i = (x_i - x'_i) . mean over alpha of dlogit/dx at
x' + alpha (x - x'), with the baseline x' made of padding embeddings at every
real token. Special tokens keep their embeddings, so only the words are
explained. The attributions sum to logit(x) - logit(x') up to the Riemann
error, which `completeness_gap` reports. Captum is not needed for this.
"""

import torch

from ml.text.model import TextClassifier, encode_windows

STEPS = 32


def integrated_gradients(model: TextClassifier, tokenizer, text: str, max_length: int = 512, steps: int = STEPS) -> dict:
    """Attributions for the first window: tokens, scores, the logit, and the completeness gap."""
    packed = torch.from_numpy(encode_windows(tokenizer, text, max_length, 0)[:1])
    ids, mask = packed[:, 0], packed[:, 1]
    embedding = model.input_embeddings()
    with torch.no_grad():
        inputs = embedding(ids)
        special = torch.tensor(tokenizer.get_special_tokens_mask(ids[0].tolist(), already_has_special_tokens=True), dtype=torch.bool)
        baseline = inputs.clone()
        baseline[0, ~special] = embedding(torch.tensor([tokenizer.pad_token_id]))[0]

    model.eval()
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps  # midpoint rule
    total = torch.zeros_like(inputs)
    for alpha in alphas:
        point = (baseline + alpha * (inputs - baseline)).requires_grad_(True)
        logit = model.embed_forward(point, mask).sum()
        total += torch.autograd.grad(logit, point)[0]
    attributions = ((inputs - baseline) * total / steps).sum(-1)[0]

    with torch.no_grad():
        logit = float(model.embed_forward(inputs, mask)[0])
        reference = float(model.embed_forward(baseline, mask)[0])
    keep = (mask[0] == 1) & ~special
    return {
        "tokens": tokenizer.convert_ids_to_tokens(ids[0][keep].tolist()),
        "attributions": attributions[keep].tolist(),
        "logit": logit,
        "completeness_gap": abs(float(attributions.sum()) - (logit - reference)),
    }
