from __future__ import annotations

import torch

from .coverage import compute_strata, targets_to_strata_tensor
from .distribution import entropy_from_logits
from .ranking import mrr_at_k, ndcg_at_k, recall_at_k

STRATA = ("head", "mid", "tail")


class MetricsAggregator:
    """Считает все метрики валидации одним аккумулятором.

    За батч: update(logits, targets); в конце: compute() -> dict с метриками:
      recall/ndcg/mrr @k, coverage@k, tail_coverage@k, head_coverage@k,
      recall@k_{head|mid|tail}, ndcg@k_{head|mid|tail}, entropy.
    """

    def __init__(
        self,
        num_items: int,
        item_freq: dict[int, float],
        ks: tuple[int, ...] = (10, 50, 100),
    ) -> None:
        self.num_items = num_items
        self.ks = ks
        self.strata = compute_strata(item_freq)
        self.tail = {i for i, s in self.strata.items() if s == "tail"}
        self.head = {i for i, s in self.strata.items() if s == "head"}

        self.n = 0
        self.entropy = 0.0
        self.sums = {f"{m}@{k}": 0.0 for m in ("recall", "ndcg", "mrr") for k in ks}
        self.stratum_hits = {k: {s: 0.0 for s in STRATA} for k in ks}
        self.stratum_ndcg = {k: {s: 0.0 for s in STRATA} for k in ks}
        self.stratum_counts = {s: 0 for s in STRATA}
        self.covered = {k: set() for k in ks}
        self.tail_covered = {k: set() for k in ks}
        self.head_covered = {k: set() for k in ks}

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        b = len(targets)
        self.n += b
        self.entropy += entropy_from_logits(logits) * b

        _, top = torch.topk(logits, max(self.ks), dim=1)
        strata_idx = targets_to_strata_tensor(targets, self.strata, logits.device)

        for k in self.ks:
            self.sums[f"recall@{k}"] += recall_at_k(logits, targets, k) * b
            self.sums[f"ndcg@{k}"] += ndcg_at_k(logits, targets, k) * b
            self.sums[f"mrr@{k}"] += mrr_at_k(logits, targets, k) * b

            for s_idx, s in enumerate(STRATA):
                mask = strata_idx == s_idx
                count = int(mask.sum().item())
                if count == 0:
                    continue
                self.stratum_counts[s] += count
                self.stratum_hits[k][s] += (
                    recall_at_k(logits[mask], targets[mask], k) * count
                )
                self.stratum_ndcg[k][s] += (
                    ndcg_at_k(logits[mask], targets[mask], k) * count
                )

            for row in top[:, :k].tolist():
                self.covered[k].update(row)
                self.tail_covered[k].update(i for i in row if i in self.tail)
                self.head_covered[k].update(i for i in row if i in self.head)

    def compute(self) -> dict[str, float]:
        n = max(self.n, 1)
        res = {name: value / n for name, value in self.sums.items()}
        res["entropy"] = self.entropy / n
        for k in self.ks:
            res[f"coverage@{k}"] = len(self.covered[k]) / max(self.num_items - 1, 1)
            res[f"tail_coverage@{k}"] = len(self.tail_covered[k]) / max(
                len(self.tail), 1
            )
            res[f"head_coverage@{k}"] = len(self.head_covered[k]) / max(
                len(self.head), 1
            )
            for s in STRATA:
                cnt = max(self.stratum_counts[s], 1)
                res[f"recall@{k}_{s}"] = self.stratum_hits[k][s] / cnt
                res[f"ndcg@{k}_{s}"] = self.stratum_ndcg[k][s] / cnt
        return res
