from __future__ import annotations

import torch


def compute_pareto_strata(
    item_freq: dict[int, float], coverage_share: float = 0.8
) -> dict[int, str]:
    """Разбить айтемы на pareto_head / pareto_tail по накопленному доле вхождений.

    Pareto head — минимальный набор самых популярных айтемов, на которые
    приходится coverage_share (по умолчанию 80%) всех вхождений.
    Pareto tail — все остальные айтемы.

    Возвращает mapping token_id -> 'pareto_head' | 'pareto_tail'.
    """
    if not item_freq:
        return {}
    sorted_items = sorted(item_freq.items(), key=lambda x: x[1], reverse=True)
    total = sum(freq for _, freq in sorted_items)
    if total <= 0:
        return {item_id: "pareto_tail" for item_id, _ in sorted_items}

    strata: dict[int, str] = {}
    cumulative = 0.0
    for item_id, freq in sorted_items:
        if cumulative < coverage_share * total:
            strata[item_id] = "pareto_head"
            cumulative += freq
        else:
            strata[item_id] = "pareto_tail"
    return strata


def targets_to_pareto_tensor(
    targets: torch.Tensor, strata: dict[int, str], device: torch.device
) -> torch.Tensor:
    """Превратить тензор target id в тензор индексов парето-групп (0=head, 1=tail)."""
    stratum_to_idx = {"pareto_head": 0, "pareto_tail": 1}
    idx_list = []
    for t in targets.detach().cpu().tolist():
        stratum = strata.get(int(t), "pareto_tail")
        idx_list.append(stratum_to_idx[stratum])
    return torch.tensor(idx_list, device=device, dtype=torch.long)


def coverage_at_k(
    logits: torch.Tensor,
    k: int = 10,
    num_items: int | None = None,
    stratum_filter: set[int] | None = None,
) -> float:
    """Покрытие@K: доля уникальных айтемов в топ-K от каталога (или от подмножества).

    Если задан stratum_filter, считается покрытие толькоアイтемами из этого множества.
    """
    _, topk = torch.topk(logits, k=min(k, logits.size(1)), dim=1)
    topk_flat = topk.detach().cpu().tolist()
    unique_items = set()
    for items in topk_flat:
        unique_items.update(int(x) for x in items)
    if stratum_filter is not None:
        unique_items &= stratum_filter
        total = max(len(stratum_filter), 1)
    else:
        total = num_items if num_items is not None else logits.size(1)
        total = max(total, 1)
    return float(len(unique_items) / total)
