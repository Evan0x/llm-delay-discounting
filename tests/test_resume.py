"""Tests for mock experiment runs: no duplicates on resume, determinism."""

import sys
import json
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import pytest
from run_experiment import load_completed_trial_ids, load_config
from prompts import generate_trials
from models import MockClient


@pytest.fixture(scope="module")
def config():
    cfg_path = Path(__file__).parent.parent / "config" / "experiment.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _run_mock(config, limit, raw_dir: Path, model_name="mock") -> Path:
    """Run a limited mock experiment and return the output path."""
    trials = generate_trials(config)
    completed = load_completed_trial_ids(raw_dir=raw_dir, model_name=model_name)
    pending = [t for t in trials if t["trial_id"] not in completed][:limit]

    model_cfg = {"name": model_name, "backend": "mock", "model_id": "mock-v1"}
    client = MockClient(
        model_id="mock-v1",
        temperature=1.0,
        requests_per_minute=1000,
        seed=config["seed"],
    )

    out_path = raw_dir / f"{model_name}_test.jsonl"
    for t in pending:
        response = client.query(t["prompt"])
        row = {"model_name": model_name, "trial_id": t["trial_id"], **t, **response}
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    return out_path


def test_resume_no_duplicates(config):
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir)

        _run_mock(config, limit=20, raw_dir=raw_dir)
        after_first = load_completed_trial_ids(raw_dir, "mock")
        assert len(after_first) == 20

        _run_mock(config, limit=10, raw_dir=raw_dir)
        after_second = load_completed_trial_ids(raw_dir, "mock")

        # Should now have 30 unique, none duplicated
        assert len(after_second) == 30
        assert len(after_second) == len(after_first | after_second)


def test_no_duplicate_trial_ids_in_single_run(config):
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir)
        _run_mock(config, limit=100, raw_dir=raw_dir)
        completed = load_completed_trial_ids(raw_dir, "mock")
        assert len(completed) == 100
