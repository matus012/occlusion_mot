"""Tests for the G2a-paired criterion (D28-final): McNemar exact + intersection scope."""
from __future__ import annotations

from math import comb

import pytest

from omot.eval.hidden_eval import SegmentResult
from omot.eval.paired import mcnemar_exact_one_sided, paired_g2a


def seg(pre: int | None, post: int | None) -> SegmentResult:
    retained = pre is not None and pre == post
    return SegmentResult(
        track_id=1, pre_id=pre, post_id=post, id_retained=retained,
        center_err=None, time_err=None,
    )


class TestMcNemarExact:
    def test_no_discordant_is_one(self) -> None:
        assert mcnemar_exact_one_sided(0, 0) == 1.0

    def test_known_binomial_tail(self) -> None:
        # 8 of 10 discordant pairs won by A: p = (C(10,8)+C(10,9)+C(10,10)) / 2^10
        expected = (comb(10, 8) + comb(10, 9) + comb(10, 10)) / 1024
        assert mcnemar_exact_one_sided(8, 2) == pytest.approx(expected)
        assert expected == pytest.approx(0.0546875)

    def test_all_wins_a(self) -> None:
        assert mcnemar_exact_one_sided(6, 0) == pytest.approx(1 / 64)

    def test_symmetry_null(self) -> None:
        # A losing every discordant pair -> p covers the whole distribution
        assert mcnemar_exact_one_sided(0, 5) == 1.0

    def test_one_sided_at_even_split(self) -> None:
        # 5/5: P(X >= 5) with X~Bin(10, .5) = 0.623..., never significant
        assert mcnemar_exact_one_sided(5, 5) == pytest.approx(0.623046875)

    def test_negative_rejected(self) -> None:
        with pytest.raises(AssertionError):
            mcnemar_exact_one_sided(-1, 2)


class TestPairedG2a:
    def test_intersection_scope_only(self) -> None:
        # seg0: in scope for both; seg1: A out of scope (no post); seg2: B out (no pre)
        keys = ["s0", "s1", "s2"]
        a = [seg(1, 1), seg(1, None), seg(3, 3)]
        b = [seg(2, 5), seg(2, 2), seg(None, 4)]
        out = paired_g2a(keys, a, b)
        assert out["n_total_segments"] == 3
        assert out["n_intersection"] == 1
        assert out["outcomes"] == [{"key": "s0", "a": True, "b": False}]
        assert out["discordant_a_only"] == 1
        assert out["discordant_b_only"] == 0
        assert out["p_value"] == pytest.approx(0.5)

    def test_concordant_pairs_do_not_move_p(self) -> None:
        keys = [f"s{i}" for i in range(4)]
        a = [seg(1, 1), seg(1, 1), seg(1, 1), seg(1, 2)]
        b = [seg(1, 1), seg(1, 1), seg(1, 2), seg(1, 1)]
        out = paired_g2a(keys, a, b)
        assert out["n_intersection"] == 4
        assert out["discordant_a_only"] == 1
        assert out["discordant_b_only"] == 1
        assert out["p_value"] == pytest.approx(0.75)  # P(X>=1), X~Bin(2,.5)

    def test_retention_on_fixed_denominator(self) -> None:
        keys = ["s0", "s1"]
        a = [seg(1, 1), seg(2, 2)]
        b = [seg(1, 3), seg(2, 2)]
        out = paired_g2a(keys, a, b)
        assert out["retention_a"] == 1.0
        assert out["retention_b"] == 0.5

    def test_misaligned_rejected(self) -> None:
        with pytest.raises(AssertionError):
            paired_g2a(["s0"], [seg(1, 1)], [])

    def test_empty_intersection(self) -> None:
        out = paired_g2a(["s0"], [seg(None, None)], [seg(1, 1)])
        assert out["n_intersection"] == 0
        assert out["p_value"] == 1.0
