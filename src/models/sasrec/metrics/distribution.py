from __future__ import annotations

import torch
import torch.nn.functional as F


def entropy_from_logits(logits: torch.Tensor) -> float:
    """Шаннонова энтропия softmax распределения по всем айтемам."""
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-12)
    entropy = -(probs * log_probs).sum(dim=1).mean()
    return float(entropy.item())
