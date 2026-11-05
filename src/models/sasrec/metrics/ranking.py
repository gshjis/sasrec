from __future__ import annotations

import torch


def recall_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    """Recall@K: доля пользователей, у яких целевой айтем в топ-K."""
    _, topk = torch.topk(logits, k=min(k, logits.size(1)), dim=1)
    targets_expanded = targets.unsqueeze(1).expand_as(topk)
    hits = (topk == targets_expanded).any(dim=1)
    return float(hits.float().mean().item())


def recall_at_k_by_stratum(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_strata: torch.Tensor,
    k: int = 10,
) -> dict[str, float]:
    """Recall@K по стратам популярности (head / mid / tail).

    target_strata: тензор [batch] с индексами 0=head, 1=mid, 2=tail.
    Возвращает Recall@K для каждой страты.
    """
    _, topk = torch.topk(logits, k=min(k, logits.size(1)), dim=1)
    targets_expanded = targets.unsqueeze(1).expand_as(topk)
    hits = (topk == targets_expanded).any(dim=1)  # [batch]

    stratum_to_idx = {"head": 0, "mid": 1, "tail": 2}
    result: dict[str, float] = {}
    for stratum_name, idx in stratum_to_idx.items():
        mask = target_strata == idx
        if mask.any():
            result[f"recall@{k}_{stratum_name}"] = float(
                hits[mask].float().mean().item()
            )
        else:
            result[f"recall@{k}_{stratum_name}"] = 0.0
    return result


def ndcg_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    """NDCG@K."""
    _, topk = torch.topk(logits, k=min(k, logits.size(1)), dim=1)
    targets_expanded = targets.unsqueeze(1).expand_as(topk)
    hits = (topk == targets_expanded).float()  # [batch, k]
    positions = torch.arange(1, k + 1, device=logits.device, dtype=torch.float)
    discounts = 1.0 / torch.log2(positions + 1.0)
    dcg = (hits * discounts).sum(dim=1)
    idcg = discounts[0]  # одна релевантная позиция на первом месте
    ndcg = dcg / idcg
    return float(ndcg.mean().item())


def ndcg_at_k_by_stratum(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_strata: torch.Tensor,
    k: int = 10,
) -> dict[str, float]:
    """NDCG@K по стратам популярности."""
    _, topk = torch.topk(logits, k=min(k, logits.size(1)), dim=1)
    targets_expanded = targets.unsqueeze(1).expand_as(topk)
    hits = (topk == targets_expanded).float()  # [batch, k]
    positions = torch.arange(1, k + 1, device=logits.device, dtype=torch.float)
    discounts = 1.0 / torch.log2(positions + 1.0)
    dcg = (hits * discounts).sum(dim=1)
    idcg = discounts[0]
    ndcg = dcg / idcg  # [batch]

    stratum_to_idx = {"head": 0, "mid": 1, "tail": 2}
    result: dict[str, float] = {}
    for stratum_name, idx in stratum_to_idx.items():
        mask = target_strata == idx
        if mask.any():
            result[f"ndcg@{k}_{stratum_name}"] = float(ndcg[mask].mean().item())
        else:
            result[f"ndcg@{k}_{stratum_name}"] = 0.0
    return result


def mrr_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    """MRR@K."""
    _, topk = torch.topk(logits, k=min(k, logits.size(1)), dim=1)
    targets_expanded = targets.unsqueeze(1).expand_as(topk)
    hits = (topk == targets_expanded).float()
    positions = torch.arange(1, k + 1, device=logits.device, dtype=torch.float)
    rr = (hits / positions).sum(dim=1)
    return float(rr.mean().item())
