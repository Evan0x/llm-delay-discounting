"""Tests for prompt generation: trial count, ID uniqueness, prompt structure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import pytest
from prompts import generate_trials


@pytest.fixture(scope="module")
def config():
    cfg_path = Path(__file__).parent.parent / "config" / "experiment.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def trials(config):
    return generate_trials(config)


def test_trial_count(trials, config):
    n_beneficiaries = len(config["beneficiaries"]) + (1 if config["control_condition"]["enabled"] else 0)
    n_delays = len(config["delays"])
    n_amounts = len(config["immediate_amounts"])
    n_reps = config["repetitions"]
    n_variants = len(config["prompt_variants"])
    n_positions = 2  # immediate_first True/False
    expected = n_beneficiaries * n_delays * n_amounts * n_reps * n_variants * n_positions
    assert len(trials) == expected


def test_trial_ids_unique(trials):
    ids = [t["trial_id"] for t in trials]
    assert len(ids) == len(set(ids)), "Duplicate trial IDs found"


def test_all_trials_have_prompt(trials):
    for t in trials:
        assert t["prompt"], f"Empty prompt for trial {t['trial_id']}"


def test_prompt_ends_with_ab_instruction(trials):
    for t in trials:
        assert "Answer with only the letter A or B." in t["prompt"]


def test_option_a_b_match_immediate_first(trials):
    for t in trials:
        if t["immediate_first"]:
            assert "today" in t["option_a"] or "points" in t["option_a"] or "$" in t["option_a"]
            assert str(t["immediate_amount"]) in t["option_a"]
        else:
            assert str(t["delayed_amount"]) in t["option_a"]


def test_delayed_amount_always_100(trials, config):
    for t in trials:
        assert t["delayed_amount"] == config["delayed_amount"]


def test_control_uses_no_beneficiary_framing(trials):
    control_trials = [t for t in trials if t["beneficiary_key"] == "control"]
    assert len(control_trials) > 0
    for t in control_trials:
        assert "Choose the better option." in t["prompt"]
        assert "{beneficiary_long}" not in t["prompt"]


def test_no_api_keys_in_prompts(trials):
    for t in trials:
        assert "AIza" not in t["prompt"]
        assert "sk-" not in t["prompt"]
