"""Tests for parse.py: parse_ab and parse_row correctness, including A/B → chose_immediate mapping."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from parse import parse_ab, parse_row


class TestParseAB:
    def test_a_response(self):
        assert parse_ab("A") == ("A", False)

    def test_b_response(self):
        assert parse_ab("B") == ("B", False)

    def test_lowercase(self):
        assert parse_ab("a") == ("A", False)
        assert parse_ab("b") == ("B", False)

    def test_leading_whitespace(self):
        assert parse_ab("  A") == ("A", False)

    def test_leading_punct(self):
        assert parse_ab('"A"') == ("A", False)

    def test_empty(self):
        _, fail = parse_ab("")
        assert fail is True

    def test_none(self):
        _, fail = parse_ab(None)
        assert fail is True

    def test_other_letter(self):
        _, fail = parse_ab("C")
        assert fail is True

    def test_digit(self):
        _, fail = parse_ab("1")
        assert fail is True

    def test_prose_response(self):
        # LLM gives a reason before the letter — parse fails per design
        _, fail = parse_ab("I would choose A")
        assert fail is True


class TestChosenImmediateMapping:
    """
    Exhaustively verify: chose_immediate == True iff the model chose the immediate option.

    When immediate_first=True:  option A = immediate, option B = delayed
    When immediate_first=False: option A = delayed,   option B = immediate
    """

    def _make_row(self, raw_text: str, immediate_first: bool) -> dict:
        return {
            "raw_text": raw_text,
            "immediate_first": immediate_first,
            "trial_id": "test",
            "beneficiary_key": "self",
            "delay": "1 month",
            "immediate_amount": 50,
            "delayed_amount": 100,
        }

    def test_A_immediate_first_true_is_immediate(self):
        row = parse_row(self._make_row("A", immediate_first=True))
        assert row["chose_immediate"] is True

    def test_B_immediate_first_true_is_delayed(self):
        row = parse_row(self._make_row("B", immediate_first=True))
        assert row["chose_immediate"] is False

    def test_A_immediate_first_false_is_delayed(self):
        row = parse_row(self._make_row("A", immediate_first=False))
        assert row["chose_immediate"] is False

    def test_B_immediate_first_false_is_immediate(self):
        row = parse_row(self._make_row("B", immediate_first=False))
        assert row["chose_immediate"] is True

    def test_parse_fail_gives_none_chose_immediate(self):
        row = parse_row(self._make_row("", immediate_first=True))
        assert row["parse_fail"] is True
        assert row["chose_immediate"] is None
