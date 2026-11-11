"""CLI to run delay-discounting experiments and save raw JSONL logs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
import logging
from pathlib import Path
import signal
from typing import Any

from dotenv import load_dotenv
import yaml
from tqdm import tqdm

from models import build_model_client
from prompts import generate_trials

LOGGER = logging.getLogger(__name__)
_STOP = False


def _handle_sigint(signum: int, frame: Any) -> None:  # noqa: ARG001
    global _STOP
    _STOP = True


def load_config(path: str) -> dict[str, Any]:
    """Load YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_config(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Resolve model config by name."""
    for model_cfg in config["models"]:
        if model_cfg["name"] == model_name:
            return model_cfg
    names = ", ".join(m["name"] for m in config["models"])
    raise ValueError(f"Model '{model_name}' not found. Available: {names}")


def load_completed_trial_ids(raw_dir: Path, model_name: str) -> set[str]:
    """Read completed trial IDs for a model from existing raw JSONL files."""
    completed: set[str] = set()
    for path in glob.glob(str(raw_dir / "*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("model_name") == model_name and row.get("trial_id"):
                    completed.add(str(row["trial_id"]))
    return completed


def is_daily_quota_error(error_message: str | None) -> bool:
    """Best-effort detection of daily quota/rate cap failures."""
    if not error_message:
        return False
    msg = error_message.lower()
    terms = ["daily", "quota", "insufficient_quota", "resource exhausted"]
    return any(term in msg for term in terms)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one line to JSONL."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """Entry point for CLI."""
    parser = argparse.ArgumentParser(description="Run LLM delay-discounting experiment")
    parser.add_argument("--model", required=True, help="Model name from config.models[].name")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of trials")
    parser.add_argument("--config", default="config/experiment.yaml", help="Path to config YAML")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    load_dotenv()
    signal.signal(signal.SIGINT, _handle_sigint)

    config = load_config(args.config)
    model_cfg = get_model_config(config, args.model)

    trials = generate_trials(config)
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"{args.model}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"

    completed = load_completed_trial_ids(raw_dir=raw_dir, model_name=args.model)
    pending_trials = [t for t in trials if t["trial_id"] not in completed]
    if args.limit is not None:
        pending_trials = pending_trials[: args.limit]

    LOGGER.info(
        "Model=%s total_trials=%d completed=%d pending=%d output=%s",
        args.model,
        len(trials),
        len(completed),
        len(pending_trials),
        output_path,
    )

    if not pending_trials:
        LOGGER.info("Nothing to run; all selected trials are already completed.")
        return

    client = build_model_client(
        model_cfg=model_cfg,
        temperature=float(config["temperature"]),
        requests_per_minute=int(config["rate_limit"]["requests_per_minute"]),
        seed=int(config["seed"]),
    )

    for trial in tqdm(pending_trials, desc=f"Running {args.model}"):
        if _STOP:
            LOGGER.info("Received Ctrl+C; stopping cleanly.")
            break

        response = client.query(trial["prompt"])
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": args.model,
            "backend": model_cfg["backend"],
            "temperature": float(config["temperature"]),
            **trial,
            **response,
        }
        append_jsonl(output_path, row)

        if is_daily_quota_error(response.get("error")):
            LOGGER.warning("Daily quota appears to be hit; stopping cleanly.")
            break

    LOGGER.info("Run finished. Raw data appended to %s", output_path)


if __name__ == "__main__":
    main()
