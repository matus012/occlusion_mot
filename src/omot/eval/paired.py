"""G2a-paired (D28-final): paired comparison of two configs on identical segments.

The discriminating G2 criterion: does the trained embedder beat the ImageNet-R18 null
on the IDENTICAL segment set? Scope = intersection of both configs' association scopes
(pre-matched AND post-detectable under BOTH configs) — a fixed denominator that kills
the per-config n drift of the standalone assoc metric. Test: McNemar exact, one-sided
(alternative: config A retains more), p < 0.05 at val.

This code path is val-invariant from the D28-final freeze: identical code runs at val.
"""
from __future__ import annotations

import logging
from math import comb

from omot.eval.hidden_eval import SegmentResult

logger = logging.getLogger(__name__)


def mcnemar_exact_one_sided(wins_a: int, wins_b: int) -> float:
    """Exact one-sided McNemar p over discordant pairs.

    H0: discordant pairs split 50/50; H1: A wins more. p = P(X >= wins_a),
    X ~ Binomial(wins_a + wins_b, 0.5). No discordant pairs -> p = 1.0.
    """
    assert wins_a >= 0 and wins_b >= 0
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(wins_a, n + 1)) / 2**n


def _in_scope(r: SegmentResult) -> bool:
    return r.pre_id is not None and r.post_id is not None


def paired_g2a(
    keys: list[str],
    results_a: list[SegmentResult],
    results_b: list[SegmentResult],
) -> dict[str, object]:
    """Per-segment paired outcomes on the intersection scope + McNemar exact p.

    keys/results_a/results_b are aligned (same segment order, same length): both
    configs MUST be evaluated over the same segment index. Returns the result-JSON
    payload: outcome vectors, discordant-pair counts, retention on the fixed
    denominator, and the one-sided p-value.
    """
    assert len(keys) == len(results_a) == len(results_b), "misaligned segment lists"
    outcomes: list[dict[str, object]] = []
    for key, ra, rb in zip(keys, results_a, results_b, strict=True):
        if _in_scope(ra) and _in_scope(rb):
            outcomes.append({"key": key, "a": bool(ra.id_retained), "b": bool(rb.id_retained)})
    wins_a = sum(1 for o in outcomes if o["a"] and not o["b"])
    wins_b = sum(1 for o in outcomes if o["b"] and not o["a"])
    p = mcnemar_exact_one_sided(wins_a, wins_b)
    n = len(outcomes)
    result: dict[str, object] = {
        "n_total_segments": len(keys),
        "n_intersection": n,
        "retained_a": sum(bool(o["a"]) for o in outcomes),
        "retained_b": sum(bool(o["b"]) for o in outcomes),
        "retention_a": sum(bool(o["a"]) for o in outcomes) / n if n else 0.0,
        "retention_b": sum(bool(o["b"]) for o in outcomes) / n if n else 0.0,
        "discordant_a_only": wins_a,
        "discordant_b_only": wins_b,
        "p_value": p,
        "outcomes": outcomes,
    }
    logger.info(
        "paired G2a: n=%d, a=%d b=%d retained, discordant %d/%d, p=%.4f",
        n, result["retained_a"], result["retained_b"], wins_a, wins_b, p,
    )
    return result
