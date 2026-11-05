from __future__ import annotations

import torch


def compute_strata(item_freq: dict[int, float], n_strata: int = 3) -> dict[int, str]:
    """Разбить айтемы на страты по частоте: head (популярные), mid, tail (редкие).

    item_freq: mapping token_id -> частота.
    Возвращает mapping token_id -> строка ('head' | 'mid' | 'tail').
    """
    if not item_freq:
        return {}
    sorted_items = sorted(item_freq.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_items)
    stratum_size = max(1, n // n_strata)
    stratum_names = ["head", "mid", "tail"]
    strata: dict[int, str] = {}
    for i, (item_id, _) in enumerate(sorted_items):
        stratum_idx = min(i // stratum_size, n_strata - 1)
        strata[item_id] = stratum_names[stratum_idx]
    return strata


def targets_to_strata_tensor(
    targets: torch.Tensor, strata: dict[int, str], device: torch.device
) -> torch.Tensor:
    """Превратить тензор target id в тензор индексов страт (0=head, 1=mid, 2=tail)."""
    stratum_to_idx = {"head": 0, "mid": 1, "tail": 2}
    idx_list = []
    for t in targets.detach().cpu().tolist():
        stratum = strata.get(int(t), "tail")
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
