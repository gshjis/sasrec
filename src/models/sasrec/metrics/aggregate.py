from __future__ import annotations

import torch

from .coverage import compute_pareto_strata, targets_to_pareto_tensor
from .distribution import entropy_from_logits
from .ranking import mrr_at_k, ndcg_at_k, recall_at_k

PARETO_STRATA = ("pareto_head", "pareto_tail")


class MetricsAggregator:
    """Считает все метрики валидации одним аккумулятором.

    За батч: update(logits, targets); в конце: compute() -> dict с метриками:
      recall/ndcg/mrr @k, coverage@k,
      pareto_coverage@k, pareto_head_coverage@k, pareto_tail_coverage@k,
      entropy.
    """

    def __init__(
        self,
        num_items: int,
        item_freq: dict[int, float],
        ks: tuple[int, ...] = (10, 50, 100),
        pareto_coverage_share: float = 0.8,
    ) -> None:
        self.num_items = num_items
        self.ks = ks
        self.pareto_strata = compute_pareto_strata(
            item_freq, coverage_share=pareto_coverage_share
        )
        self.pareto_tail = {
            i for i, s in self.pareto_strata.items() if s == "pareto_tail"
        }
        self.pareto_head = {
            i for i, s in self.pareto_strata.items() if s == "pareto_head"
        }

        self.n = 0
        self.entropy = 0.0
        self.sums = {f"{m}@{k}": 0.0 for m in ("recall", "ndcg", "mrr") for k in ks}
        self.pareto_hits = {k: {s: 0.0 for s in PARETO_STRATA} for k in ks}
        self.pareto_ndcg = {k: {s: 0.0 for s in PARETO_STRATA} for k in ks}
        self.pareto_counts = {s: 0 for s in PARETO_STRATA}
        self.covered = {k: set() for k in ks}
        self.pareto_head_covered = {k: set() for k in ks}
        self.pareto_tail_covered = {k: set() for k in ks}

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        b = len(targets)
        self.n += b
        self.entropy += entropy_from_logits(logits) * b

        _, top = torch.topk(logits, max(self.ks), dim=1)
        pareto_idx = targets_to_pareto_tensor(
            targets, self.pareto_strata, logits.device
        )

        for k in self.ks:
            self.sums[f"recall@{k}"] += recall_at_k(logits, targets, k) * b
            self.sums[f"ndcg@{k}"] += ndcg_at_k(logits, targets, k) * b
            self.sums[f"mrr@{k}"] += mrr_at_k(logits, targets, k) * b

            for p_idx, s in enumerate(PARETO_STRATA):
                mask = pareto_idx == p_idx
                count = int(mask.sum().item())
                if count == 0:
                    continue
                self.pareto_counts[s] += count
                self.pareto_hits[k][s] += (
                    recall_at_k(logits[mask], targets[mask], k) * count
                )
                self.pareto_ndcg[k][s] += (
                    ndcg_at_k(logits[mask], targets[mask], k) * count
                )

            for row in top[:, :k].tolist():
                self.covered[k].update(row)
                self.pareto_head_covered[k].update(
                    i for i in row if i in self.pareto_head
                )
                self.pareto_tail_covered[k].update(
                    i for i in row if i in self.pareto_tail
                )

    def compute(self) -> dict[str, float]:
        n = max(self.n, 1)
        res = {name: value / n for name, value in self.sums.items()}
        res["entropy"] = self.entropy / n
        for k in self.ks:
            res[f"coverage@{k}"] = len(self.covered[k]) / max(self.num_items - 1, 1)
            res[f"pareto_coverage@{k}"] = len(self.covered[k]) / max(
                self.num_items - 1, 1
            )
            res[f"pareto_head_coverage@{k}"] = len(self.pareto_head_covered[k]) / max(
                len(self.pareto_head), 1
            )
            res[f"pareto_tail_coverage@{k}"] = len(self.pareto_tail_covered[k]) / max(
                len(self.pareto_tail), 1
            )
        return res
